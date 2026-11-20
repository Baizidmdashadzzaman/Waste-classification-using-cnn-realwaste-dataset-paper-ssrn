# %% [markdown]
# # Uncertainty-Aware Attention for Waste Classification (ICECE 2026)
#
# **Proposed contribution**: DenseNet-121 + channel attention with an
# **attention-entropy-modulated MC-Dropout**. At inference, the model runs
# `MC_PASSES` stochastic forward passes. The per-sample dropout rate is coupled to
# the entropy of the attention-focus distribution, so the model is deliberately
# *more uncertain when its attention is indecisive*.
#
# It reports:
#   * accuracy / macro-F1 / macro-ROC-AUC
#   * calibration (ECE, Brier, NLL)
#   * predictive / aleatoric / epistemic uncertainty (entropy split)
#   * correlation between attention entropy and predictive uncertainty
#   * cross-dataset validation: RealWaste (train) -> TrashNet + UGV-Bangladesh
#   * Grad-CAM + attention visualizations
#
# **Three models (ablation)**:
#   * `A-none`   : DenseNet-121 + channel attention (prior paper baseline)
#   * `B-fixed`  : same + standard MC-Dropout (fixed p = 0.5)
#   * `C-attn`   : same + attention-entropy-modulated MC-Dropout  **<-- proposed**
#
# **Kaggle setup**: attach the following datasets to Input (Add Data):
#   1. `joebeachcapital/realwaste`  (or any Kaggle copy of the RealWaste repo)
#   2. TrashNet (the original GitHub-ish layout `dataset-resized/{glass,paper,...}`)
#   3. Your private upload of the UGV Bangladesh dataset (YOLO format with
#      `train/{images,labels}` etc.)  -- optional but recommended
#
# The script auto-discovers all datasets under `/kaggle/input`, so exact dataset
# ids / folder names do not matter. If a dataset is missing, that experiment is
# skipped and a note is printed.
#
# Estimated runtime on a free T4 GPU: ~1.3 min/epoch (RealWaste, batch 32).
# 20 epochs x 3 models -> about 1.5-2.5 hours total for all three ablations.
# You can also run only the proposed model (cells 5-7) to save time.

# %%
# ----------------------------------------------------------------------
# CELL 1 - Imports, configuration, device, seeds
# ----------------------------------------------------------------------
import os
import math
import random
import shutil
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms, models

from collections import Counter
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# ---------------- config ----------------
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
PATIENCE = 4

P_DROP_BASE = 0.20      # base dropout rate (Model C floor)
P_DROP_ALPHA = 0.40     # alpha: how strongly attention entropy raises the rate
P_DROP_MAX = 0.85       # hard cap
P_DROP_FIXED = 0.50     # constant rate for the standard MC-Dropout ablation (Model B)

MC_PASSES = 20          # stochastic forward passes at MC inference time
REDO_TRAINING = True    # set False to reuse saved checkpoints and skip training
SAVE_DIR = "/kaggle/working"
CKPT_A = os.path.join(SAVE_DIR, "model_A_none.pth")
CKPT_B = os.path.join(SAVE_DIR, "model_B_fixed.pth")
CKPT_C = os.path.join(SAVE_DIR, "model_C_attn.pth")

REALWASTE_CLASSES = ["cardboard", "foodOrganics", "glass", "metal",
                     "miscellaneousTrash", "paper", "plastic",
                     "textileTrash", "vegetation"]

# TrashNet class -> RealWaste class (shared, clean mapping)
TRASHNET_MAP = {"glass": "glass", "paper": "paper", "cardboard": "cardboard",
                "plastic": "plastic", "metal": "metal", "trash": "miscellaneousTrash"}

# UGV-Bangladesh YOLO class -> coarse RealWaste class (coarse mapping)
UGV_MAP = {"bottle": "plastic", "hardplastic": "plastic", "packet": "plastic",
           "polythene": "plastic", "cocksheet": "metal",
           "mask": "textileTrash", "sandal": "textileTrash"}

# BDWaste species-name keywords -> RealWaste coarse class (checked in order!)
BDWASTE_KEYWORDS = [
    ("foodOrganics", ["egg", "peel", "rice", "sugarcane", "sugar cane", "husk",
                      "fish", "scale", "ash", "shell", "mango", "banana",
                      "potato", "lemon", "malta"]),
    ("paper", ["paper", "coffee"]),
    ("glass", ["glass"]),
    ("metal", ["wire", "can", "cane", "tin"]),
    ("plastic", ["polythene", "plastic", "bottle", "packet", "medicine", "chip"]),
    ("textileTrash", ["glove", "mask", "cloth", "textile"]),
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA available:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA NOT available - using CPU (training will be very slow).")
print("Using device:", device)
print("PyTorch:", torch.__version__, "| Torchvision:", torchvision.__version__)

# %%
# ----------------------------------------------------------------------
# CELL 2 - Dataset discovery + loaders (RealWaste, TrashNet, UGV-BD)
# ----------------------------------------------------------------------
INPUT_ROOT = "/kaggle/input"

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def listdir(path):
    return sorted(os.listdir(path)) if os.path.isdir(path) else []


def find_realwaste_root():
    """Locate the RealWaste folder: any dir whose subfolders are the 9 classes."""
    for root, dirs, files in os.walk(INPUT_ROOT):
        names = set(dirs)
        if set(REALWASTE_CLASSES).intersection(names) == set(REALWASTE_CLASSES):
            return root
    return None


def find_trashnet_root():
    """Locate TrashNet: any dir containing the 6 category folders (or `dataset-resized`)."""
    for root, dirs, files in os.walk(INPUT_ROOT):
        names = set(dirs)
        if {"glass", "paper", "cardboard", "plastic", "metal", "trash"}.issubset(names):
            return root
        # robocopy-style: nested one level deeper, e.g. <input>/<id>/dataset-resized/<class>
        if "dataset-resized" in names:
            inner = os.path.join(root, "dataset-resized")
            if {"glass", "paper", "cardboard", "plastic", "metal", "trash"} \
                    .issubset(set(os.listdir(inner))):
                return inner
    return None


def find_ugv_root():
    """Locate UGV-Bangladesh: any dir with train/valid/test each containing images+labels."""
    for root, dirs, files in os.walk(INPUT_ROOT):
        if {"train", "valid"}.issubset(set(dirs)) or {"train", "test"}.issubset(set(dirs)):
            splits = [d for d in ["train", "valid", "test"] if d in dirs]
            ok = all(os.path.isdir(os.path.join(root, s, "images")) and
                     os.path.isdir(os.path.join(root, s, "labels")) for s in splits)
            if ok and splits:
                return root, splits
    return None, []


def bdwaste_map_folder(folder):
    """BDWaste species folder name -> coarse RealWaste class (keyword based, in order)."""
    s = folder.lower().replace("_", " ").replace("-", " ").strip()
    for real_cls, kws in BDWASTE_KEYWORDS:
        for kw in kws:
            if kw in s:
                return real_cls
    return None


def find_bdwaste_roots():
    """Locate BDWaste: every directory that directly contains images (species folders)."""
    species_dirs = []
    for root, dirs, files in os.walk(INPUT_ROOT):
        imgs = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if imgs and bdwaste_map_folder(os.path.basename(root)) is not None:
            species_dirs.append((root, len(imgs)))
    if not species_dirs:
        return []
    # keep only species dirs that live under the same BDWaste root, excluding train/valid/test mirror trees
    by_parent = {}
    for d, n in species_dirs:
        parent = os.path.dirname(d)
        by_parent.setdefault(parent, []).append((d, n))
    # prefer the parent with the most species dirs (the real BDWaste root)
    best_parent = max(by_parent.items(), key=lambda kv: len(kv[1]))[0]
    return sorted(by_parent[best_parent])


REALWASTE_ROOT = find_realwaste_root()
TRASHNET_ROOT = find_trashnet_root()
UGV_ROOT, UGV_SPLITS = find_ugv_root()
BDWASTE_DIRS = find_bdwaste_roots()

print("RealWaste root :", REALWASTE_ROOT)
print("TrashNet root  :", TRASHNET_ROOT)
print("UGV-BD root    :", UGV_ROOT, UGV_SPLITS)
print("BDWaste dirs   :", len(BDWASTE_DIRS))


def split_realwaste(root):
    full = datasets.ImageFolder(root)
    class_names = full.classes
    num_classes = len(class_names)
    labels = np.array(full.targets)
    indices = np.arange(len(full))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=labels, random_state=SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=SEED)

    train_ds = deepcopy(full); train_ds.transform = train_tf
    val_ds = deepcopy(full);   val_ds.transform = val_tf
    test_ds = deepcopy(full);  test_ds.transform = val_tf
    ds = {
        "train": Subset(train_ds, sorted(train_idx)),
        "val":   Subset(val_ds, sorted(val_idx)),
        "test":  Subset(test_ds, sorted(test_idx)),
    }
    dl = {p: DataLoader(ds[p], batch_size=BATCH_SIZE, shuffle=(p == "train"),
                        num_workers=2) for p in ds}
    return class_names, ds, dl


real_classes = None
real_loaders = None
if REALWASTE_ROOT:
    real_classes, real_ds, real_loaders = split_realwaste(REALWASTE_ROOT)
    print(f"  RealWaste: {len(real_ds['train'])} train / {len(real_ds['val'])} val / "
          f"{len(real_ds['test'])} test - classes = {real_classes}")
else:
    raise RuntimeError("RealWaste dataset not found under /kaggle/input. "
                       "Attach the RealWaste dataset and re-run.")

num_classes = len(real_classes)

# %%
# ----------------------------------------------------------------------
# CELL 3 - Proposed architecture: ChannelAttention + entropy-modulated MC-Dropout
# ----------------------------------------------------------------------

def attention_entropy(gate):
    """Normalized Shannon entropy of the attention-focus distribution.

    gate: (B, C) in (0,1)  (sigmoid channel gates)
    Uses saturation-weighted focus weights w_c = g_c * |2*g_c - 1|, so
    channels that are indecisive (~0.5) contribute little and produce a
    near-uniform (high-entropy) focus distribution, while decisive channels
    (near 0 or 1) concentrate the distribution (low entropy).
    Returns e in [0, 1] (1 = maximally indecisive / confused attention).
    """
    C = gate.size(1)
    q = (2.0 * gate - 1.0).abs()                        # decisiveness, [0,1]
    w = (gate * q).clamp(min=0.0) + 1e-8                # focus weights
    w_n = w / w.sum(dim=1, keepdim=True)
    H = -(w_n * torch.log(w_n)).sum(dim=1)              # Shannon (nats)
    return H / math.log(C)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
        self._last_gate = None

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        out = torch.cat([avg_out, max_out], dim=1)                 # (B, 2C)
        gate = self.fc(out.view(x.size(0), -1)).view(x.size(0), x.size(1), 1, 1)
        self._last_gate = gate.detach().view(x.size(0), -1)        # (B, C)
        return x * gate


class AttentionDenseNet121(nn.Module):
    """DenseNet-121 with channel attention and configurable dropout strategy.

    dropout_mode:
      'none'  - no dropout after attention (deterministic single pass)
      'fixed' - standard MC-Dropout with constant p = P_DROP_FIXED
      'attn'  - PROPOSED: dropout rate per sample = f(H_attention)
    """
    def __init__(self, num_classes, dropout_mode="attn"):
        super().__init__()
        assert dropout_mode in ("none", "fixed", "attn")
        self.dropout_mode = dropout_mode
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.features = base.features
        for p in self.features.parameters():
            p.requires_grad = False
        self.attention = ChannelAttention(1024)
        self.classifier = nn.Linear(1024 * 7 * 7, num_classes)
        self._rates = None

    # ---- per-sample dropout rate ------------------------------------
    def _drop_rate(self, gate):
        if self.dropout_mode == "fixed":
            return torch.full((gate.size(0),), P_DROP_FIXED, device=gate.device)
        if self.dropout_mode == "attn":
            e = attention_entropy(gate)
            return (P_DROP_BASE + P_DROP_ALPHA * e).clamp(max=P_DROP_MAX)
        return None

    # ---- forward ------------------------------------------------------
    def forward(self, x, mode="auto", mc_passes=MC_PASSES):
        if mode == "auto":
            mode = "train" if self.training else "det"

        feats = self.features(x)                       # (B, 1024, 7, 7)
        gated = self.attention(feats)                  # store gate
        flat = gated.view(gated.size(0), -1)           # (B, 50176)

        if mode == "det" or self.dropout_mode == "none":
            return self.classifier(flat)

        rate = self._drop_rate(self.attention._last_gate)          # (B,)
        self._rates = rate
        keep = (1.0 - rate).clamp(min=0.02).unsqueeze(1)           # (B,1)
        n_passes = mc_passes if mode == "mc" else 1

        probs = []
        for _ in range(n_passes):
            mask = torch.bernoulli(keep.expand_as(flat))
            out = self.classifier(flat * mask / keep)
            probs.append(F.softmax(out, dim=1))
        probs = torch.stack(probs, dim=0)                          # (T,B,C)
        if mode == "mc":
            return probs                                           # (T,B,C)
        # train: return logits of the mean softmax for a stable CE loss
        probs_mean = probs.mean(dim=0)
        return torch.log(probs_mean.clamp_min(1e-7))


def build_model(dropout_mode, num_classes):
    m = AttentionDenseNet121(num_classes, dropout_mode=dropout_mode).to(device)
    n_tr = sum(p.numel() for p in m.attention.parameters()) + \
           sum(p.numel() for p in m.classifier.parameters())
    n_tot = sum(p.numel() for p in m.parameters())
    print(f"[{dropout_mode:>5}] params: trainable={n_tr:,} total={n_tot:,}")
    return m

# quick sanity forward pass (uses download of ImageNet weights once)
print("Sanity-checking model C forward (deterministic + MC...)")
_sanity = build_model("attn", num_classes)
_sanity.eval()
_x = torch.randn(2, 3, 224, 224).to(device)
with torch.no_grad():
    _o = _sanity(_x, mode="det")
    print("  det output:", tuple(_o.shape))
    _p = _sanity(_x, mode="mc", mc_passes=5)
    print("  mc  output:", tuple(_p.shape), "prob range [%.3f, %.3f]" % (_p.min().item(), _p.max().item()))
del _sanity

# %%
# ----------------------------------------------------------------------
# CELL 4 - Training utilities
# ----------------------------------------------------------------------

def make_optim(model):
    params = list(model.attention.parameters()) + list(model.classifier.parameters())
    opt = optim.Adam(params, lr=LEARNING_RATE)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=2)
    return opt, sched


def train_one_model(dropout_mode, ckpt_path, criterion):
    if os.path.exists(ckpt_path) and not REDO_TRAINING:
        print(f"\n=== Loading existing checkpoint for mode='{dropout_mode}' ===")
        model = build_model(dropout_mode, num_classes)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return model

    print(f"\n=== Training mode='{dropout_mode}' ===")
    model = build_model(dropout_mode, num_classes)
    opt, sched = make_optim(model)

    best_acc = 0.0
    best_loss = float("inf")
    stall = 0
    hist = {"tl": [], "ta": [], "vl": [], "va": []}

    for epoch in range(NUM_EPOCHS):
        for phase in ("train", "val"):
            model.train() if phase == "train" else model.eval()
            run_loss, run_corr, n = 0.0, 0, 0
            for inputs, labels in real_loaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                opt.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)                       # logits
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(1)
                    if phase == "train":
                        loss.backward()
                        opt.step()
                run_loss += loss.item() * inputs.size(0)
                run_corr += (preds == labels).sum().item()
                n += inputs.size(0)
            epoch_loss = run_loss / n
            epoch_acc = run_corr / n
            if phase == "train":
                hist["tl"].append(run_loss / n)
                hist["ta"].append(run_corr / n)
            else:
                hist["vl"].append(run_loss / n)
                hist["va"].append(run_corr / n)
            print(f"  ep{epoch+1:02d} [{phase:>5}] loss={epoch_loss:.4f} acc={epoch_acc:.4f}")

        sched.step(hist["vl"][-1])
        if hist["va"][-1] > best_acc:
            best_acc = hist["va"][-1]
            torch.save(model.state_dict(), ckpt_path)
            stall = 0
            print(f"  -> new best val acc {best_acc:.4f}, saved {ckpt_path}")
        if hist["vl"][-1] > best_loss:
            stall += 1
            if stall >= PATIENCE:
                print(f"  early stop @ ep{epoch+1}")
                break
        else:
            best_loss = hist["vl"][-1]
            stall = 0

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model


# class weights for imbalance (as in the original paper)
train_targets = [real_ds["train"].dataset.targets[i] for i in real_ds["train"].indices]
class_counts = Counter(train_targets)
total_train = len(train_targets)
cw = [total_train / (num_classes * class_counts[i]) for i in range(num_classes)]
cw = torch.tensor(cw, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=cw)
print("Class weights:", np.round(cw.cpu().numpy(), 3))

# %%
# ----------------------------------------------------------------------
# CELL 5 - Train Model A (baseline: attention, no dropout)
# ----------------------------------------------------------------------
model_A = train_one_model("none", CKPT_A, criterion) if REALWASTE_ROOT else None
print("Model A (no dropout) ready.")

# %%
# ----------------------------------------------------------------------
# CELL 6 - Train Model B (standard MC-Dropout, fixed p)
# ----------------------------------------------------------------------
model_B = train_one_model("fixed", CKPT_B, criterion) if REALWASTE_ROOT else None
print("Model B (fixed MC-Dropout) ready.")

# %%
# ----------------------------------------------------------------------
# CELL 7 - Train Model C (PROPOSED: attention-entropy-modulated dropout)
# ----------------------------------------------------------------------
model_C = train_one_model("attn", CKPT_C, criterion) if REALWASTE_ROOT else None
print("Model C (proposed) ready.")

# %%
# ----------------------------------------------------------------------
# CELL 8 - MC inference + calibration + uncertainty utilities
# ----------------------------------------------------------------------

def mc_forward(model, loader, mc_passes=MC_PASSES, batch_cap=None):
    """Return dict with everything needed for metrics.

    preds_*, probs_mean (deterministic-style softmax),
    entropy_total, entropy_epi (MI), entropy_alea, attn_ent, labels.
    """
    model.eval()
    P_all, L_all, AE_all = [], [], []
    n_seen = 0
    with torch.no_grad():
        for inputs, labels in loader:
            if batch_cap is not None and n_seen >= batch_cap:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            probs = model(inputs, mode="mc", mc_passes=mc_passes)          # (T,B,C)
            gate = model.attention._last_gate
            ae = attention_entropy(gate).cpu().numpy()

            p_mean = probs.mean(dim=0)                                     # (B,C)
            H_t = (-(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=2))  # (T,B)
            alea = H_t.mean(dim=0).cpu().numpy()
            total = -(p_mean * torch.log(p_mean.clamp_min(1e-9))).sum(dim=1).cpu().numpy()
            epi = torch.clamp(torch.from_numpy(total) - torch.from_numpy(alea), min=0).numpy()

            P_all.append(p_mean.cpu().numpy())
            L_all.append(labels.cpu().numpy())
            AE_all.append(ae)
            n_seen += inputs.size(0)
    P = np.concatenate(P_all)
    L = np.concatenate(L_all)
    AE = np.concatenate(AE_all)
    return {
        "probs": P, "labels": L, "attn_ent": AE,
    }


def evaluate_all(model, loader, name, mc=True, mc_passes=MC_PASSES):
    """Full metric pack for one model (MC or deterministic)."""
    model.eval()
    all_T, all_E, all_L, all_AE = [], [], [], []
    can_mc = (mc and model.dropout_mode != "none")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if can_mc:
                probs = model(inputs, mode="mc", mc_passes=mc_passes)          # (T,B,C)
                p_mean = probs.mean(dim=0)
                H_t = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=2)
                alea = H_t.mean(dim=0).cpu().numpy()
                total = -(p_mean * torch.log(p_mean.clamp_min(1e-9))).sum(dim=1).cpu().numpy()
                epi = np.clip(total - alea, 0, None)
                P = total, alea, epi
                p_mean_for_cm = p_mean
            else:
                logits = model(inputs, mode="det")
                p = F.softmax(logits, dim=1)
                total_d = (-(p * torch.log(p.clamp_min(1e-9))).sum(dim=1).cpu().numpy())
                P = (total_d, np.zeros_like(total_d), np.zeros_like(total_d))
                p_mean_for_cm = p
            AE = attention_entropy(model.attention._last_gate).cpu().numpy()
            all_T.append(p_mean_for_cm.cpu().numpy())
            all_L.append(labels.cpu().numpy())
            all_AE.append(AE)
            all_E.append((P[0], P[1], P[2]))
    probs = np.concatenate(all_T)
    labels = np.concatenate(all_L)
    AE = np.concatenate(all_AE)
    total = np.concatenate([e[0] for e in all_E])
    alea = np.concatenate([e[1] for e in all_E])
    epi = np.concatenate([e[2] for e in all_E])
    return {"probs": probs, "labels": labels, "attn_ent": AE,
            "total": total, "alea": alea, "epi": epi}


def calibration_metrics(probs, labels, n_bins=10):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if m.sum() > 0:
            ece += (m.sum() / len(labels)) * abs(acc[m].mean() - conf[m].mean())
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    brier = ((onehot - probs) ** 2).sum(axis=1).mean()
    nll = -np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean()
    return ece, brier, nll


def standard_metrics(probs, labels, target_names):
    pred = probs.argmax(axis=1)
    acc = float((pred == labels).mean())
    rep = classification_report(labels, pred, labels=np.arange(num_classes),
                                target_names=target_names, output_dict=True, zero_division=0)
    macro_f1 = rep["macro avg"]["f1-score"]
    macro_rec = rep["macro avg"]["recall"]
    macro_pre = rep["macro avg"]["precision"]
    lb = label_binarize(labels, classes=np.arange(num_classes))
    try:
        macro_auc = roc_auc_score(lb, probs, average="macro")
    except Exception:
        macro_auc = float("nan")
    return acc, macro_f1, macro_rec, macro_pre, macro_auc


def print_results_table(results):
    print("\n" + "=" * 100)
    print("FINAL RESULTS (RealWaste test set)")
    print("=" * 100)
    hdr = f"{'model':<12}{'acc':>8}{'F1':>7}{'prec':>7}{'rec':>7}{'AUC':>7}{'ECE':>7}{'Brier':>7}{'NLL':>7}"
    print(hdr)
    for name, r in results.items():
        b_ent, b_epi, b_alea = r["unc"]["total"].mean(), r["unc"]["epi"].mean(), r["unc"]["alea"].mean()
        print(f"{name:<12}{r['acc']*100:>7.2f}%{r['f1']*100:>6.2f}%{r['prec']*100:>6.2f}%"
              f"{r['rec']*100:>6.2f}%{r['auc']:>7.3f}{r['ece']:>7.3f}{r['brier']:>7.3f}{r['nll']:>7.3f}"
              f"   | H={b_ent:.3f} MI={b_epi:.3f} A={b_alea:.3f}")


def run_eval(model, loader, name, mc=True):
    d = evaluate_all(model, loader, name, mc=mc)
    ece, brier, nll = calibration_metrics(d["probs"], d["labels"])
    acc, f1, prec, rec, auc_ = standard_metrics(d["probs"], d["labels"], real_classes)
    m = {"acc": acc, "f1": f1, "prec": prec, "rec": rec, "auc": auc_,
         "ece": ece, "brier": brier, "nll": nll,
         "probs": d["probs"], "labels": d["labels"], "attn_ent": d["attn_ent"],
         "unc": {"total": d["total"], "alea": d["alea"], "epi": d["epi"]},
         "cm": confusion_matrix(d["labels"], d["probs"].argmax(1))}
    return m

# %%
# ----------------------------------------------------------------------
# CELL 9 - Evaluate all three models on RealWaste test
# ----------------------------------------------------------------------
models = {}
if model_A is not None:
    models["A-none"] = model_A
if model_B is not None:
    models["B-fixed"] = model_B
if model_C is not None:
    models["C-attn"] = model_C

results = {}
for name, mdl in models.items():
    results[name] = run_eval(mdl, real_loaders["test"], name, mc=True)

print_results_table(results)

# %%
# ----------------------------------------------------------------------
# CELL 10 - Calibration, reliability diagram, uncertainty-accuracy analysis
# ----------------------------------------------------------------------

def plot_uncertainty_analysis(results, out="uncertainty_analysis.png"):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    colors = {"A-none": "#999", "B-fixed": "#4b8bbe", "C-attn": "#3cb371"}

    # (1) Reliability diagrams
    ax = axes[0, 0]
    for name, r in results.items():
        conf = r["probs"].max(1)
        acc = (r["probs"].argmax(1) == r["labels"]).astype(float)
        edges = np.linspace(0, 1, 11)
        cc, aa = [], []
        for i in range(10):
            m = (conf >= edges[i]) & (conf < edges[i + 1])
            if m.sum() > 0:
                cc.append(conf[m].mean()); aa.append(acc[m].mean())
        ax.plot(cc, aa, "o-", label=name, color=colors[name], lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Confidence (max softmax)"); ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram"); ax.legend(); ax.grid(alpha=.3)

    # (2) Uncertainty vs accuracy (predictive entropy bins)
    ax = axes[0, 1]
    for name, r in results.items():
        u = r["unc"]["total"]
        bins = np.quantile(u, [0, .2, .4, .6, .8, 1.0])
        accs = []
        for i in range(5):
            m = (u >= bins[i]) & (u <= bins[i + 1])
            if m.sum() > 0:
                accs.append((r["probs"].argmax(1)[m] == r["labels"][m]).mean())
            else:
                accs.append(0)
        ax.plot(accs, "o-", label=name, color=colors[name], lw=2)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax.set_xlabel("Predictive-entropy quintile (Q1 = most certain)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. uncertainty"); ax.legend(); ax.grid(alpha=.3)

    # (3) Attention entropy: correct vs misclassified (proposed model only)
    ax = axes[1, 0]
    if "C-attn" in results:
        r = results["C-attn"]
        ok = r["probs"].argmax(1) == r["labels"]
        ax.hist(r["attn_ent"][ok], bins=25, alpha=.6, label="correct", color=colors["C-attn"])
        ax.hist(r["attn_ent"][~ok], bins=25, alpha=.6, label="misclassified", color="#e05a47")
        ax.set_xlabel("Attention entropy H_att"); ax.set_ylabel("count")
        ax.set_title("Attention entropy: correct vs misclassified"); ax.legend()
        ax.grid(alpha=.3)
        cc_ = np.corrcoef(r["attn_ent"], r["unc"]["total"])[0, 1]
        print(f"[C-attn] corr(attention entropy, predictive entropy) = {cc_:.3f}")

    # (4) Predictive entropy histogram (in-domain)
    ax = axes[1, 1]
    for name, r in results.items():
        ax.hist(r["unc"]["total"], bins=30, alpha=.5, label=name, color=colors[name])
    ax.set_xlabel("Predictive entropy"); ax.set_ylabel("count")
    ax.set_title("Predictive entropy distribution (RealWaste test)")
    ax.legend(); ax.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, out), dpi=200, bbox_inches="tight")
    print("Saved", os.path.join(SAVE_DIR, out))
    plt.show()

plot_uncertainty_analysis(results)

# Plain-text copy-paste block
if "C-attn" in results and "B-fixed" in results:
    rC = results["C-attn"]; rB = results["B-fixed"]
    okC = rC["probs"].argmax(1) == rC["labels"]
    okB = rB["probs"].argmax(1) == rB["labels"]
    print(f"\n[ANALYSIS] accuracy by uncertainty quintile (C-attn): ", end="")
    u = rC["unc"]["total"]; bins = np.quantile(u, [0, .2, .4, .6, .8, 1.0])
    for i in range(5):
        m = (u >= bins[i]) & (u <= bins[i+1])
        print(f"Q{i+1}={100*(rC['probs'].argmax(1)[m]==rC['labels'][m]).mean():.1f}%", end="  ")
    print()
    print(f"[ANALYSIS] mean attention entropy: correct={rC['attn_ent'][okC].mean():.4f}, "
          f"misclassified={rC['attn_ent'][~okC].mean():.4f}, "
          f"delta={rC['attn_ent'][~okC].mean()-rC['attn_ent'][okC].mean():+.4f}")
    if okB.sum() > 0:
        print(f"[ANALYSIS] B-fixed mean attention entropy: correct={rB['attn_ent'][okB].mean():.4f}, "
              f"misclassified={rB['attn_ent'][~okB].mean():.4f}")

# %%
# ----------------------------------------------------------------------
# CELL 11 - Reliability diagram + confusion matrix (proposed model)
# ----------------------------------------------------------------------

if "C-attn" in results:
    r = results["C-attn"]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(r["cm"], cmap="Blues")
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels(real_classes, rotation=45, ha="right")
    ax.set_yticklabels(real_classes)
    thresh = r["cm"].max() / 2
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, f'{r["cm"][i, j]}', ha="center", va="center",
                    color="white" if r["cm"][i, j] > thresh else "black")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    ax.set_title("Confusion matrix - proposed model (C-attn)")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "confusion_C_attn.png"), dpi=200, bbox_inches="tight")
    print("Saved", os.path.join(SAVE_DIR, "confusion_C_attn.png"))
    plt.show()

# %%
# ----------------------------------------------------------------------
# CELL 12 - Grad-CAM + attention visualizations (confident vs uncertain)
# ----------------------------------------------------------------------

def _to_uint8(hm):
    hm = np.nan_to_num(hm, nan=0.0)
    if hm.max() - hm.min() < 1e-9:
        hm = np.zeros_like(hm)
    else:
        hm = (hm - hm.min()) / (hm.max() - hm.min())
    return (255 * np.clip(hm, 0, 1)).astype(np.uint8)


def generate_gradcam(model, input_tensor, device):
    model.eval()
    img = input_tensor.unsqueeze(0).to(device)
    for p in model.features.parameters():
        p.requires_grad = True
    acts, grads = [], []

    def fh(module, inp, out):
        out.retain_grad(); acts.append(out)
    def bh(module, gin, gout):
        grads.append(gout[0].detach())

    h1 = model.attention.register_forward_hook(fh)
    h2 = model.attention.register_full_backward_hook(bh)
    model.zero_grad()
    out = model(img, mode="det")
    _, pred = out.max(1)
    out[0, pred.item()].backward(retain_graph=True)
    h1.remove(); h2.remove()
    for p in model.features.parameters():
        p.requires_grad = False

    A = acts[0][0]                                  # (1024,7,7)
    G = grads[0][0]                                 # (1024,7,7)
    w = G.mean(dim=(1, 2), keepdim=True)
    hm = torch.relu((w * A).sum(dim=0)).detach().cpu().numpy()
    hm = cv2.resize(hm, (224, 224))
    hm = _to_uint8(hm)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    in_np = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    in_np = (in_np - in_np.min()) / max(in_np.max() - in_np.min(), 1e-9)
    ov = cv2.addWeighted(np.uint8(in_np * 255), 0.55, hm, 0.45, 0)
    return cv2.cvtColor(ov, cv2.COLOR_BGR2RGB), pred.item()


def attention_map(model, input_tensor, device):
    model.eval()
    with torch.no_grad():
        img = input_tensor.unsqueeze(0).to(device)
        feats = model.features(img)
        gated = model.attention(feats)
        gate = model.attention._last_gate[0]                     # (1024,)
        w = gate.view(1, -1, 1, 1)                               # (1,1024,1,1)
        emb = (gated * w).sum(dim=1, keepdim=True)               # coarse spatial emphasis
        hm = emb[0, 0].detach().cpu().numpy()
        hm = cv2.resize(hm, (224, 224))
        hm = _to_uint8(hm)
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_VIRIDIS)
        in_np = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        in_np = (in_np - in_np.min()) / max(in_np.max() - in_np.min(), 1e-9)
        ov = cv2.addWeighted(np.uint8(in_np * 255), 0.6, hm, 0.4, 0)
        return cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)


if "C-attn" in results and _HAS_CV2:
    test_ds_orig = datasets.ImageFolder(REALWASTE_ROOT, transforms.ToTensor())
    rC = results["C-attn"]
    unc = rC["unc"]["total"]
    idx_sorted = np.argsort(unc)
    confident_idx = idx_sorted[:3]     # lowest predictive entropy
    uncertain_idx = idx_sorted[-3:]    # highest predictive entropy
    test_indices = np.array(real_ds["test"].indices)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for row, (group, idxs) in enumerate([("Confident", confident_idx), ("Uncertain", uncertain_idx)]):
        for col, i in enumerate(idxs):
            display_img, true_cls = test_ds_orig[test_indices[i]]      # unnormalized
            _, _ = display_img, true_cls
            cam_input, true_cls = real_ds["test"][i]                   # normalized tensor
            try:
                grad_img, pred_cls = generate_gradcam(models["C-attn"], cam_input, device)
            except Exception as ex:
                print("gradcam failed:", ex)
                grad_img, pred_cls = np.zeros((224, 224, 3), np.uint8), -1
            att_img = attention_map(models["C-attn"], cam_input, device)
            axes[row, 0].imshow(np.transpose(display_img.numpy(), (1, 2, 0)))
            axes[row, 0].set_title(f"True: {real_classes[true_cls]} | H={float(unc[i]):.3f}")
            axes[row, 0].axis("off")
            axes[row, 1].imshow(grad_img)
            axes[row, 1].set_title(f"Grad-CAM (pred={real_classes[pred_cls] if pred_cls!=-1 else '?'})")
            axes[row, 1].axis("off")
            axes[row, 2].imshow(att_img)
            axes[row, 2].set_title(f"Attention emphasis | H_att={float(rC['attn_ent'][i]):.3f}")
            axes[row, 2].axis("off")
    for c, lab in enumerate(["Original", "Grad-CAM", "Attention"]):
        axes[0, c].set_ylabel(lab, fontsize=12)
    plt.suptitle("Interpretability: confident (top) vs uncertain (bottom)", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "gradcam_attention_C.png"), dpi=200, bbox_inches="tight")
    print("Saved", os.path.join(SAVE_DIR, "gradcam_attention_C.png"))
    plt.show()
else:
    print("Skipping Grad-CAM visualizations (model C or cv2 missing).")

# %%
# ----------------------------------------------------------------------
# CELL 13 - Cross-dataset validation (RealWaste-trained -> TrashNet + UGV-BD)
# ----------------------------------------------------------------------

def map_to_realwaste_idx(class_name):
    d = {c: i for i, c in enumerate(real_classes)}
    return d.get(class_name, None)


def subsample_loader(ds, n, shuffle=False):
    if n is None or len(ds) <= n:
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2)
    idx = np.random.RandomState(SEED).choice(len(ds), n, replace=False)
    return DataLoader(Subset(ds, idx), batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2)


def build_trashnet_loader():
    """TrashNet ImageFolder filtered to the 6 shared classes, labels remapped to RealWaste idx."""
    class_dir = {d: os.path.join(TRASHNET_ROOT, d) for d in TRASHNET_MAP}
    samples, targets, paths = [], [], []
    for trash_class, real_class in TRASHNET_MAP.items():
        ridx = map_to_realwaste_idx(real_class)
        if ridx is None:
            continue
        d = class_dir[trash_class]
        if not os.path.isdir(d):
            print(f"  [TrashNet] missing folder {d}")
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                p = os.path.join(d, f)
                samples.append((p, ridx))
                paths.append(p)
                targets.append(ridx)
    if not samples:
        return None, 0
    # deduplicate identical images across classes by path basename (TrashNet quirk)
    seen = set()
    samples2, targets2 = [], []
    for s, t in zip(samples, targets):
        key = os.path.basename(s[0])
        if key in seen:
            continue
        seen.add(key)
        samples2.append((s[0], t)); targets2.append(t)
    ds = datasets.ImageFolder(TRASHNET_ROOT, transform=val_tf)
    ds.samples = samples2
    ds.targets = targets2
    ds.imgs = samples2
    return ds, len(samples2)


def build_ugv_loader():
    """UGV-BD YOLO format -> cropped classification images mapped to coarse RealWaste idx."""
    if not _HAS_CV2:
        print("  [UGV] cv2 missing, skipping")
        return None
    tmp = os.path.join(SAVE_DIR, "ugv_crops")
    os.makedirs(tmp, exist_ok=True)
    samples, targets = [], []
    split = "test" if "test" in UGV_SPLITS else UGV_SPLITS[0]
    img_dir = os.path.join(UGV_ROOT, split, "images")
    lbl_dir = os.path.join(UGV_ROOT, split, "labels")
    used_cls = {}
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        base = os.path.splitext(f)[0]
        lbl = os.path.join(lbl_dir, base + ".txt")
        if not os.path.exists(lbl):
            continue
        img = cv2.imread(os.path.join(img_dir, f))
        if img is None:
            continue
        h, w = img.shape[:2]
        best = None
        with open(lbl) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    xs = (cx - bw / 2, cx + bw / 2)
                    ys = (cy - bh / 2, cy + bh / 2)
                elif len(parts) == 9:
                    cid = int(parts[0])
                    coords = list(map(float, parts[1:9]))
                    xs = coords[0::2]; ys = coords[1::2]
                else:
                    continue
                x1 = int(min(xs) * w); y1 = int(min(ys) * h)
                x2 = int(max(xs) * w); y2 = int(max(ys) * h)
                area = (x2 - x1) * (y2 - y1)
                if best is None or area > best[0]:
                    best = (area, cid, x1, y1, x2, y2)
        if best is None:
            continue
        _, cid, x1, y1, x2, y2 = best
        ugv_classes = _ugv_class_names()
        name = ugv_classes.get(cid)
        real_name = UGV_MAP.get(name) if name else None
        ridx = map_to_realwaste_idx(real_name) if real_name else None
        if ridx is None:
            continue
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
        if x2 - x1 < 16 or y2 - y1 < 16:
            continue
        m = 0.10  # margin
        dx, dy = int((x2 - x1) * m), int((y2 - y1) * m)
        crop = img[max(0, y1 - dy):min(h, y2 + dy), max(0, x1 - dx):min(w, x2 + dx)]
        out_path = os.path.join(tmp, f"ugv_{f.replace('.', '_')}.jpg")
        cv2.imwrite(out_path, crop)
        used_cls[name] = used_cls.get(name, 0) + 1
        samples.append((out_path, ridx)); targets.append(ridx)
    if not samples:
        return None, 0, 0
    from PIL import Image as _PIL_Image
    class ds(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples; self.transform = transform
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, i):
            p, t = self.samples[i]
            im = cv2.imread(p)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = _PIL_Image.fromarray(im)
            return self.transform(im), t
    return ds(samples, val_tf), len(samples), dict(used_cls)


def build_bdwaste_loader():
    """BDWaste species folders -> coarse RealWaste idx (ImageFolder-style, no crops needed)."""
    if not BDWASTE_DIRS:
        return None, 0, 0
    from PIL import Image as _PIL_Image
    samples, targets, used_cls = [], [], {}
    # optionally cap per-species count to keep inference fast (full set ~2600 imgs is fine)
    for d, n in BDWASTE_DIRS:
        folder = os.path.basename(d)
        real_cls = bdwaste_map_folder(folder)
        ridx = map_to_realwaste_idx(real_cls) if real_cls else None
        if ridx is None:
            print(f"  [BDWaste] skipped folder {folder} (unmapped)")
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(d, f), ridx))
                targets.append(ridx)
                used_cls[real_cls] = used_cls.get(real_cls, 0) + 1
    if not samples:
        return None, 0, 0
    class ds(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples; self.transform = transform
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, i):
            p, t = self.samples[i]
            im = _PIL_Image.open(p).convert("RGB")
            return self.transform(im), t
    return ds(samples, val_tf), len(samples), dict(used_cls)


_ugv_names_cache = None

def _ugv_class_names():
    global _ugv_names_cache
    if _ugv_names_cache is not None:
        return _ugv_names_cache
    names = {}
    candidates = []
    for root, dirs, files in os.walk(UGV_ROOT):
        for f in files:
            if f in ("data.yaml", "data.yml") or f.endswith("_names.txt") or f == "classes.txt":
                candidates.append(os.path.join(root, f))
    for cpath in candidates:
        try:
            text = open(cpath, encoding="utf-8").read()
            tops = [l.strip() for l in text.splitlines() if l.strip()]
            if cpath.endswith((".yaml", ".yml")):
                for k, l in enumerate(tops):
                    if l.startswith("names:") and ":" in l:
                        inline = l.split(":", 1)[1].strip()
                        if inline:
                            names[0] = inline
                        for l2 in tops[k + 1:]:
                            s2 = l2.lstrip("-").strip()
                            if ":" in s2 and not s2.startswith("#"):
                                try:
                                    idx = int(s2.split(":", 1)[0].strip())
                                    val = s2.split(":", 1)[1].strip().strip("\"' ")
                                    if val:
                                        names[idx] = val
                                except ValueError:
                                    pass
                            elif s2.startswith("-") or (l2.startswith(" ") and "-" in l2):
                                val = s2.lstrip("-").strip().strip("\"' ")
                                if val:
                                    names[len(names)] = val
                        break
            else:
                for i, l in enumerate(tops):
                    if l and not l.startswith("#"):
                        names[i] = l
        except Exception:
            continue
    if not names:
        names = {0: "bottle", 1: "cocksheet", 2: "hardplastic", 3: "mask",
                 4: "medicine", 5: "packet", 6: "polythene", 7: "sandal"}
    _ugv_names_cache = names
    return names


print("\n===== CROSS-DATASET VALIDATION (trained on RealWaste) =====")

if not TRASHNET_ROOT:
    print("[TrashNet] dataset not found - skipping.")
else:
    tn_ds, tn_n = build_trashnet_loader()
    if tn_ds is None:
        print("[TrashNet] no usable samples - skipping.")
    else:
        tn_loader = subsample_loader(tn_ds, None)
        print(f"[TrashNet] {tn_n} images across 6 shared classes")
        for name, mdl in models.items():
            d = evaluate_all(mdl, tn_loader, name, mc=True)
            ece, brier, nll = calibration_metrics(d["probs"], d["labels"])
            acc = float((d["probs"].argmax(1) == d["labels"]).mean())
            print(f"  [TrashNet] {name:<12} acc={100*acc:6.2f}%  ECE={ece:.3f}  "
                  f"Brier={brier:.3f}  H_total={d['total'].mean():.3f}  H_epi={d['epi'].mean():.3f}")

if BDWASTE_DIRS:
    bd_ds, bd_n, bd_cls = build_bdwaste_loader()
    if bd_ds is None:
        print("[BDWaste] no usable samples - skipping.")
    else:
        bd_loader = subsample_loader(bd_ds, None)
        print(f"[BDWaste] {bd_n} images (coarse classes: {bd_cls})")
        for name, mdl in models.items():
            d = evaluate_all(mdl, bd_loader, name, mc=True)
            ece, brier, nll = calibration_metrics(d["probs"], d["labels"])
            acc = float((d["probs"].argmax(1) == d["labels"]).mean())
            print(f"  [BDWaste] {name:<12} acc={100*acc:6.2f}%  ECE={ece:.3f}  "
                  f"Brier={brier:.3f}  H_total={d['total'].mean():.3f}  H_epi={d['epi'].mean():.3f}")
else:
    print("[BDWaste] dataset not found - skipping.")

if UGV_ROOT:
    ufv_ds, ufv_n, ufv_cls = build_ugv_loader()
    if ufv_ds is None:
        print("[UGV-BD] no usable crops - skipping.")
    else:
        ufv_loader = subsample_loader(ufv_ds, None)
        print(f"[UGV-BD] {ufv_n} object crops (coarse classes: {ufv_cls})")
        for name, mdl in models.items():
            d = evaluate_all(mdl, ufv_loader, name, mc=True)
            ece, brier, nll = calibration_metrics(d["probs"], d["labels"])
            acc = float((d["probs"].argmax(1) == d["labels"]).mean())
            print(f"  [UGV-BD] {name:<12} acc={100*acc:6.2f}%  ECE={ece:.3f}  "
                  f"Brier={brier:.3f}  H_total={d['total'].mean():.3f}  H_epi={d['epi'].mean():.3f}")
        # domain-shift entropy analysis: in-domain vs out-of-domain
        for name, mdl in models.items():
            rd = results[name]
            od = evaluate_all(mdl, ufv_loader, name, mc=True)
            ratio = od["total"].mean() / max(rd["unc"]["total"].mean(), 1e-9)
            print(f"  [UGV-BD] {name:<12} entropy ratio (OOD/ID) = {ratio:.2f} (in-domain H={rd['unc']['total'].mean():.3f})")
else:
    print("[UGV-BD] dataset not found - skipping. (still OK; TrashNet is the primary cross-dataset benchmark)")

print("\nAll done. Results are saved under", SAVE_DIR)