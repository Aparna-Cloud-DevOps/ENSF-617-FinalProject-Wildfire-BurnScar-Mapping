"""
unet_training_v2.py
===================
Training pipeline for wildfire burn-scar severity segmentation.
7 input channels, 4 severity classes, 4 model architectures.

Models:  UNet  |  ResUNet  |  AttentionUNet  |  UNet++
Loss:    FocalDiceLoss (class-weighted)
Metrics: IoU, Dice, PR-AUC per class
Splits:  data/sentinel2/splits_v2.json (forward-chaining by fire year)

Run:
    python unet_training_v2.py --model all --epochs 60
"""

import os, json, time, warnings, argparse, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (average_precision_score,
                             confusion_matrix,
                             precision_recall_curve)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE     = os.environ.get("WILDFIRE_BASE_DIR",
           os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Fixed constants
# ---------------------------------------------------------------------------
N_INPUT   = 7         # 7 channels: Red,Green,Blue,NIR,SWIR1,NBR,dNBR
N_CLASSES = 4         # 0=Unburned/Low  1=Moderate  2=High  3=Very High
PATCH_SZ  = 256
SEED      = 42
F_BASE    = 32        # base feature channels (32→64→128→256→512 ~8M params)
DROPOUT   = 0.3       # applied in decoder blocks only

SEVERITY_NAMES  = ["Unburned/Low", "Moderate", "High", "Very High"]
SEVERITY_COLORS = ["#1a9850", "#fee08b", "#f46d43", "#a50026"]
DARK_BG         = "#1a1a2e"


# =============================================================================
# SECTION 1  DATASET
# =============================================================================

class WildfireDatasetV2(Dataset):
    """
    Load 256x256 .npz patches produced by sentinel2_dataset_v2.py.

    All patches are pre-loaded into RAM on first instantiation to eliminate
    per-epoch disk I/O on shared filesystems. With 48G RAM and ~7GB of patch
    data this reduces epoch time from ~12 min (I/O-bound) to ~2-3 min (GPU-bound).

    .npz fields:
        X        float32 (7, 256, 256)
        y        uint8   (256, 256)
    """

    def __init__(self, patch_files, augment=False, mean=None, std=None):
        self.files   = list(patch_files)
        self.augment = augment
        self.mean = torch.from_numpy(mean) if mean is not None \
                    else torch.zeros(N_INPUT, dtype=torch.float32)
        self.std  = torch.from_numpy(std)  if std  is not None \
                    else torch.ones(N_INPUT,  dtype=torch.float32)

        # Pre-load all patches into RAM (eliminates repeated disk reads)
        print(f"    Pre-loading {len(self.files)} patches into RAM ...", flush=True)
        self._cache_X = []
        self._cache_y = []
        for fp in self.files:
            data = np.load(fp, allow_pickle=True)
            X = torch.from_numpy(data["X"].astype(np.float32))
            y = torch.from_numpy(data["y"].astype(np.int64)).clamp(0, N_CLASSES - 1)
            # Normalise once at load time — no repeated division per epoch
            X = (X - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-8)
            self._cache_X.append(X)
            self._cache_y.append(y)
        print(f"    Cache ready ({len(self._cache_X)} patches).", flush=True)

    def __len__(self):
        return len(self._cache_X)

    def __getitem__(self, idx):
        X = self._cache_X[idx]
        y = self._cache_y[idx]
        if self.augment:
            X, y = _augment(X, y)
        return X, y


def _augment(X, y):
    """Apply random geometric and photometric augmentations (train only)."""
    # 1. Random horizontal flip
    if torch.rand(1) > 0.5:
        X = torch.flip(X, dims=[2])
        y = torch.flip(y, dims=[1])

    # 2. Random vertical flip
    if torch.rand(1) > 0.5:
        X = torch.flip(X, dims=[1])
        y = torch.flip(y, dims=[0])

    # 3. Random 90/180/270 rotation (lossless, exact)
    if torch.rand(1) > 0.5:
        k = torch.randint(1, 4, (1,)).item()
        X = torch.rot90(X, k, dims=[1, 2])
        y = torch.rot90(y, k, dims=[0, 1])

    # 4. Random free rotation 0–360 degrees via affine grid
    if torch.rand(1) > 0.5:
        import torch.nn.functional as F2
        angle = float(torch.empty(1).uniform_(0, 360))
        angle_rad = angle * 3.14159265 / 180.0
        cos_a, sin_a = float(np.cos(angle_rad)), float(np.sin(angle_rad))
        theta = torch.tensor([[cos_a, -sin_a, 0.0],
                               [sin_a,  cos_a, 0.0]], dtype=torch.float32)
        grid = F2.affine_grid(theta.unsqueeze(0),
                              (1, 1, PATCH_SZ, PATCH_SZ), align_corners=False)
        X = F2.grid_sample(X.unsqueeze(0), grid,
                           mode="bilinear", padding_mode="reflection",
                           align_corners=False).squeeze(0)
        y_f = F2.grid_sample(y.float().unsqueeze(0).unsqueeze(0), grid,
                             mode="nearest", padding_mode="reflection",
                             align_corners=False).squeeze().long()
        y = y_f

    # 5. Random crop + resize (scale jitter 80–100% of original)
    if torch.rand(1) > 0.5:
        import torch.nn.functional as F2
        scale = float(torch.empty(1).uniform_(0.80, 1.0))
        crop_size = int(PATCH_SZ * scale)
        max_off = PATCH_SZ - crop_size
        r0 = torch.randint(0, max_off + 1, (1,)).item()
        c0 = torch.randint(0, max_off + 1, (1,)).item()
        X = X[:, r0:r0+crop_size, c0:c0+crop_size]
        y = y[r0:r0+crop_size, c0:c0+crop_size]
        X = F2.interpolate(X.unsqueeze(0), size=(PATCH_SZ, PATCH_SZ),
                           mode="bilinear", align_corners=False).squeeze(0)
        y = F2.interpolate(y.float().unsqueeze(0).unsqueeze(0),
                           size=(PATCH_SZ, PATCH_SZ),
                           mode="nearest").squeeze().long()

    # 6. Brightness jitter on spectral channels 0–4
    if torch.rand(1) > 0.5:
        factor = float(torch.empty(1).uniform_(0.75, 1.25))
        X[:5] = X[:5] * factor

    # 7. Per-channel contrast jitter on spectral channels 0–4
    if torch.rand(1) > 0.5:
        factor = float(torch.empty(1).uniform_(0.75, 1.25))
        for c in range(5):
            ch_mean = X[c].mean()
            X[c] = (X[c] - ch_mean) * factor + ch_mean

    # 8. Gaussian noise on spectral channels 0–4
    if torch.rand(1) > 0.5:
        noise = torch.randn_like(X[:5]) * 0.03
        X[:5] = X[:5] + noise

    return X, y


def load_splits(patch_dir, splits_json):
    """
    Return (train_files, val_files, test_files) lists of .npz paths.
    Falls back to random 70/15/15 if splits_json is missing or splits are empty.
    """

    def _random_split(all_files):
        np.random.seed(SEED)
        idx = np.random.permutation(len(all_files))
        n   = len(all_files)
        tr  = [str(all_files[i]) for i in idx[:int(0.70*n)]]
        va  = [str(all_files[i]) for i in idx[int(0.70*n):int(0.85*n)]]
        te  = [str(all_files[i]) for i in idx[int(0.85*n):]]
        return tr, va, te

    all_files = sorted(Path(patch_dir).rglob("*.npz"))
    if not all_files:
        raise FileNotFoundError(
            f"No .npz patch files found under {patch_dir}. "
            "Run sentinel2_dataset_v2.py first.")

    if not os.path.exists(splits_json):
        print(f"  splits_v2.json not found -- using random 70/15/15 split")
        return _random_split(all_files)

    with open(splits_json) as f:
        splits = json.load(f)

    def ids_to_files(ids):
        out = []
        for fid in ids:
            out.extend(sorted(Path(patch_dir).rglob(f"{fid}_*.npz")))
        return [str(p) for p in out]

    tr = ids_to_files(splits["train"])
    va = ids_to_files(splits["val"])
    te = ids_to_files(splits["test"])

    if not va or not te:
        print(f"  WARNING: temporal split has val={len(va)}, test={len(te)}. "
              "Falling back to random 70/15/15 patch-level split.")
        return _random_split(all_files)

    return tr, va, te


def compute_channel_stats(files, n_samples=500):
    """Compute per-channel mean and std using Welford's online algorithm.
    Only uses TRAINING patches (avoids data leakage from val/test).
    Welford's method is numerically stable — no catastrophic cancellation
    as in the naive (sum_x2 - n*mean^2) formula.
    """
    rng   = np.random.default_rng(SEED)
    # Subsample to keep runtime manageable (500 patches ≈ 6 min on shared NFS)
    sel   = rng.choice(files, min(n_samples, len(files)), replace=False)
    M     = np.zeros(N_INPUT, dtype=np.float64)   # running mean
    S     = np.zeros(N_INPUT, dtype=np.float64)   # running sum of squared deviations
    count = 0
    for fp in sel:
        X = np.load(fp, allow_pickle=True)["X"].astype(np.float64)
        for c in range(N_INPUT):
            for v in X[c].ravel():
                count += 1
                delta  = v - M[c]           # deviation from current mean
                M[c]  += delta / count      # update mean
                S[c]  += delta * (v - M[c]) # update sum of squared deviations (Bessel's)
    std  = np.sqrt(S / max(count - 1, 1)).astype(np.float32)  # sample std
    mean = M.astype(np.float32)
    return mean, std


# =============================================================================
# SECTION 2  MODEL ARCHITECTURES
# =============================================================================

def _double_conv(in_ch, out_ch, dropout=0.0):
    layers = [
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.insert(4, nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


def _pad_to(x, ref):
    """Symmetrically pad tensor x to match the spatial size of ref.
    Required when ConvTranspose2d output is 1px smaller than the skip connection
    due to odd input dimensions (e.g. 255→127 after pooling, 127→254 after upsample).
    """
    dh = ref.size(2) - x.size(2)   # height difference in pixels
    dw = ref.size(3) - x.size(3)   # width difference in pixels
    # Pad format: [left, right, top, bottom] — asymmetric if diff is odd
    return F.pad(x, [dw//2, dw - dw//2, dh//2, dh - dh//2])


# ---- 1. Vanilla UNet -------------------------------------------------------

class _Up(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = _double_conv(out_ch * 2, out_ch, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = _pad_to(x1, x2)
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    name = "unet"

    def __init__(self, input_channels=N_INPUT, num_classes=N_CLASSES,
                 f=F_BASE, drop=DROPOUT, bilinear=False):
        super().__init__()
        self.enc1 = _double_conv(input_channels, f)
        self.enc2 = _double_conv(f,     f*2)
        self.enc3 = _double_conv(f*2,   f*4)
        self.enc4 = _double_conv(f*4,   f*8)
        self.bot  = _double_conv(f*8,   f*16, drop)
        self.pool = nn.MaxPool2d(2)
        self.up4  = _Up(f*16, f*8,  drop)
        self.up3  = _Up(f*8,  f*4,  drop)
        self.up2  = _Up(f*4,  f*2)
        self.up1  = _Up(f*2,  f)
        self.head = nn.Conv2d(f, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bot(self.pool(e4))
        d4 = self.up4(b,  e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)


# ---- 2. ResUNet (pre-activation residual blocks) ---------------------------

class _ResBlock(nn.Module):
    """Pre-activation residual block: BN+ReLU+Conv+BN+ReLU+Conv + skip."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip  = nn.Conv2d(in_ch, out_ch, 1, bias=False) \
                     if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.relu(self.bn1(x)))
        h = self.drop(h)
        h = self.conv2(F.relu(self.bn2(h)))
        return h + self.skip(x)


class _ResUp(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up    = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.block = _ResBlock(out_ch * 2, out_ch, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = _pad_to(x1, x2)
        return self.block(torch.cat([x2, x1], dim=1))


class ResUNet(nn.Module):
    name = "resunet"

    def __init__(self, input_channels=N_INPUT, num_classes=N_CLASSES,
                 f=F_BASE, drop=DROPOUT, bilinear=False):
        super().__init__()
        self.enc1 = _ResBlock(input_channels, f)
        self.enc2 = _ResBlock(f,     f*2)
        self.enc3 = _ResBlock(f*2,   f*4)
        self.enc4 = _ResBlock(f*4,   f*8)
        self.bot  = _ResBlock(f*8,   f*16, drop)
        self.pool = nn.MaxPool2d(2)
        self.up4  = _ResUp(f*16, f*8,  drop)
        self.up3  = _ResUp(f*8,  f*4,  drop)
        self.up2  = _ResUp(f*4,  f*2)
        self.up1  = _ResUp(f*2,  f)
        self.head = nn.Conv2d(f, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bot(self.pool(e4))
        d4 = self.up4(b,  e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)


# ---- 3. AttentionUNet -------------------------------------------------------

class _AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.Wg  = nn.Sequential(nn.Conv2d(F_g,   F_int, 1, bias=False),
                                 nn.BatchNorm2d(F_int))
        self.Wx  = nn.Sequential(nn.Conv2d(F_l,   F_int, 1, bias=False),
                                 nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1,     1, bias=False),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())

    def forward(self, g, x):
        g1  = self.Wg(F.interpolate(g, size=x.shape[2:],
                                    mode="bilinear", align_corners=False))
        x1  = self.Wx(x)
        psi = self.psi(F.relu(g1 + x1))
        return x * psi


class _AttUp(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.attn = _AttentionGate(out_ch, skip_ch, out_ch // 2)
        self.conv = _double_conv(out_ch + skip_ch, out_ch, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.attn(x1, x2)
        x1 = _pad_to(x1, x2)
        return self.conv(torch.cat([x2, x1], dim=1))


class AttentionUNet(nn.Module):
    name = "attention"

    def __init__(self, input_channels=N_INPUT, num_classes=N_CLASSES,
                 f=F_BASE, drop=DROPOUT, bilinear=False):
        super().__init__()
        self.enc1 = _double_conv(input_channels, f)
        self.enc2 = _double_conv(f,     f*2)
        self.enc3 = _double_conv(f*2,   f*4)
        self.enc4 = _double_conv(f*4,   f*8)
        self.bot  = _double_conv(f*8,   f*16, drop)
        self.pool = nn.MaxPool2d(2)
        self.up4  = _AttUp(f*16, f*8,  f*8,  drop)
        self.up3  = _AttUp(f*8,  f*4,  f*4,  drop)
        self.up2  = _AttUp(f*4,  f*2,  f*2)
        self.up1  = _AttUp(f*2,  f,    f)
        self.head = nn.Conv2d(f, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bot(self.pool(e4))
        d4 = self.up4(b,  e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)


# ---- 4. UNet++ (dense nested skip connections) -----------------------------

class UNetPlusPlus(nn.Module):
    """
    U-Net++ with 4 encoder depths. Dense nested skip connections.
    Channel counts verified for f=64:
      Depth 0: 64  |  Depth 1: 128  |  Depth 2: 256  |  Depth 3: 512
      Bottleneck: 1024
    """
    name = "unetpp"

    def __init__(self, input_channels=N_INPUT, num_classes=N_CLASSES,
                 f=F_BASE, drop=DROPOUT, bilinear=False):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.X00 = _double_conv(input_channels, f)
        self.X10 = _double_conv(f,    f*2)
        self.X20 = _double_conv(f*2,  f*4)
        self.X30 = _double_conv(f*4,  f*8)
        self.X40 = _double_conv(f*8,  f*16, drop)   # bottleneck

        # Up-samplers
        self.up40 = nn.ConvTranspose2d(f*16, f*8,  2, stride=2)
        self.up30 = nn.ConvTranspose2d(f*8,  f*4,  2, stride=2)
        self.up31 = nn.ConvTranspose2d(f*8,  f*4,  2, stride=2)
        self.up20 = nn.ConvTranspose2d(f*4,  f*2,  2, stride=2)
        self.up21 = nn.ConvTranspose2d(f*4,  f*2,  2, stride=2)
        self.up22 = nn.ConvTranspose2d(f*4,  f*2,  2, stride=2)
        self.up10 = nn.ConvTranspose2d(f*2,  f,    2, stride=2)
        self.up11 = nn.ConvTranspose2d(f*2,  f,    2, stride=2)
        self.up12 = nn.ConvTranspose2d(f*2,  f,    2, stride=2)
        self.up13 = nn.ConvTranspose2d(f*2,  f,    2, stride=2)

        # Dense nodes (input = all same-depth predecessors concatenated)
        # Depth 3: X31 = cat(X30, up40(X40)) = f*8+f*8 -> f*8
        self.X31 = _double_conv(f*8  + f*8,       f*8,  drop)
        # Depth 2: X21 = cat(X20, up30(X30)) = f*4+f*4 -> f*4
        self.X21 = _double_conv(f*4  + f*4,       f*4,  drop)
        # Depth 2: X22 = cat(X20, X21, up31(X31)) = f*4*3 -> f*4
        self.X22 = _double_conv(f*4  + f*4 + f*4, f*4,  drop)
        # Depth 1: X11 = cat(X10, up20(X20)) = f*2+f*2 -> f*2
        self.X11 = _double_conv(f*2  + f*2,       f*2)
        # Depth 1: X12 = cat(X10, X11, up21(X21)) = f*2*3 -> f*2
        self.X12 = _double_conv(f*2  + f*2 + f*2, f*2)
        # Depth 1: X13 = cat(X10, X11, X12, up22(X22)) = f*2*4 -> f*2
        self.X13 = _double_conv(f*2  * 4,         f*2)
        # Depth 0: X01 = cat(X00, up10(X10)) = f+f -> f
        self.X01 = _double_conv(f    + f,          f)
        # Depth 0: X02 = cat(X00, X01, up11(X11)) = f*3 -> f
        self.X02 = _double_conv(f    + f + f,      f)
        # Depth 0: X03 = cat(X00, X01, X02, up12(X12)) = f*4 -> f
        self.X03 = _double_conv(f    * 4,          f)
        # Depth 0: X04 = cat(X00..X03, up13(X13)) = f*5 -> f
        self.X04 = _double_conv(f    * 5,          f)

        self.head = nn.Conv2d(f, num_classes, 1)

    def forward(self, x):
        p = self.pool
        # Encoder
        x00 = self.X00(x)
        x10 = self.X10(p(x00))
        x20 = self.X20(p(x10))
        x30 = self.X30(p(x20))
        x40 = self.X40(p(x30))

        # Depth 3
        x31 = self.X31(torch.cat([x30, _pad_to(self.up40(x40), x30)], 1))

        # Depth 2
        x21 = self.X21(torch.cat([x20, _pad_to(self.up30(x30), x20)], 1))
        x22 = self.X22(torch.cat([x20, x21, _pad_to(self.up31(x31), x20)], 1))

        # Depth 1
        x11 = self.X11(torch.cat([x10, _pad_to(self.up20(x20), x10)], 1))
        x12 = self.X12(torch.cat([x10, x11, _pad_to(self.up21(x21), x10)], 1))
        x13 = self.X13(torch.cat([x10, x11, x12, _pad_to(self.up22(x22), x10)], 1))

        # Depth 0
        x01 = self.X01(torch.cat([x00, _pad_to(self.up10(x10), x00)], 1))
        x02 = self.X02(torch.cat([x00, x01, _pad_to(self.up11(x11), x00)], 1))
        x03 = self.X03(torch.cat([x00, x01, x02, _pad_to(self.up12(x12), x00)], 1))
        x04 = self.X04(torch.cat([x00, x01, x02, x03, _pad_to(self.up13(x13), x00)], 1))

        return self.head(x04)


MODEL_REGISTRY = {
    "unet":      UNet,
    "resunet":   ResUNet,
    "attention": AttentionUNet,
    "unetpp":    UNetPlusPlus,
}


# =============================================================================
# SECTION 3  LOSS FUNCTIONS
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=-1):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight   # stored as CPU tensor; moved to device in forward
        self.ignore = ignore_index

    def forward(self, logits, targets):
        # Always move weight to the same device as logits
        w  = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(logits, targets, weight=w,
                             ignore_index=self.ignore, reduction="none",
                             label_smoothing=0.1)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class DiceLoss(nn.Module):
    def __init__(self, n_cls=N_CLASSES, smooth=1.0):
        super().__init__()
        self.n_cls  = n_cls
        self.smooth = smooth

    def forward(self, logits, targets):
        prob = F.softmax(logits, dim=1)
        loss = 0.0
        for c in range(self.n_cls):
            pred = prob[:, c]
            gt   = (targets == c).float()
            num  = 2.0 * (pred * gt).sum() + self.smooth
            den  = pred.sum() + gt.sum() + self.smooth
            loss += 1.0 - num / den
        return loss / self.n_cls


class FocalDiceLoss(nn.Module):
    """0.5 * FocalLoss(gamma=2) + 0.5 * DiceLoss with class-weights."""

    def __init__(self, n_cls=N_CLASSES, alpha=0.5, gamma=2.0,
                 class_weights=None):
        super().__init__()
        w = torch.tensor(class_weights, dtype=torch.float32) \
            if class_weights is not None else None
        self.focal = FocalLoss(gamma=gamma, weight=w)
        self.dice  = DiceLoss(n_cls=n_cls)
        self.alpha = alpha

    def forward(self, logits, targets):
        return (self.alpha       * self.focal(logits, targets) +
                (1 - self.alpha) * self.dice(logits, targets))


# =============================================================================
# SECTION 4  METRICS
# =============================================================================

@torch.no_grad()
def compute_metrics(model, loader, device, n_cls=N_CLASSES):
    """Per-class IoU, Dice, PR-AUC; macro mIoU, mDice, mean PR-AUC."""
    model.eval()
    all_preds, all_trues, all_probs = [], [], []

    for X, y in loader:
        X, y   = X.to(device), y.to(device)
        logits = model(X)
        probs  = F.softmax(logits, dim=1)
        preds  = logits.argmax(1)
        all_preds.append(preds.cpu().numpy().ravel())
        all_trues.append(y.cpu().numpy().ravel())
        all_probs.append(probs.cpu().numpy().transpose(0,2,3,1)
                         .reshape(-1, n_cls))

    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)
    probs = np.concatenate(all_probs, axis=0)

    iou_per_cls, dice_per_cls, ap_per_cls = [], [], []
    for c in range(n_cls):
        tp = np.sum((preds == c) & (trues == c))
        fp = np.sum((preds == c) & (trues != c))
        fn = np.sum((preds != c) & (trues == c))
        iou  = tp / (tp + fp + fn + 1e-8)
        dice = 2*tp / (2*tp + fp + fn + 1e-8)
        iou_per_cls.append(float(iou))
        dice_per_cls.append(float(dice))
        try:
            ap = average_precision_score(
                (trues == c).astype(int), probs[:, c])
        except ValueError:
            ap = 0.0
        ap_per_cls.append(float(ap))

    return {
        "iou":         iou_per_cls,
        "dice":        dice_per_cls,
        "pr_auc":      ap_per_cls,
        "mean_iou":    float(np.mean(iou_per_cls)),
        "mean_dice":   float(np.mean(dice_per_cls)),
        "mean_pr_auc": float(np.mean(ap_per_cls)),
    }


@torch.no_grad()
def compute_per_fire_metrics(model, loader, device, n_cls=N_CLASSES):
    """IoU/Dice per fire_id parsed from patch filenames."""
    model.eval()
    fire_data = {}

    for batch_idx, (X, y) in enumerate(loader):
        X_dev  = X.to(device)
        logits = model(X_dev)
        probs  = F.softmax(logits, dim=1)
        preds  = logits.argmax(1)

        dataset = loader.dataset
        start   = batch_idx * loader.batch_size
        end     = min(start + X.size(0), len(dataset.files))

        for i, fp in enumerate(dataset.files[start:end]):
            stem  = Path(fp).stem
            parts = stem.split("_")
            # fire_id is everything except the last part (patch index NNNN)
            fire_id = "_".join(parts[:-1]) if len(parts) >= 2 else stem

            if fire_id not in fire_data:
                fire_data[fire_id] = {"preds": [], "trues": [], "probs": []}

            fire_data[fire_id]["preds"].append(preds[i].cpu().numpy().ravel())
            fire_data[fire_id]["trues"].append(y[i].numpy().ravel())
            fire_data[fire_id]["probs"].append(
                probs[i].cpu().numpy().transpose(1,2,0).reshape(-1, n_cls))

    rows = []
    for fire_id, d in fire_data.items():
        p  = np.concatenate(d["preds"])
        t  = np.concatenate(d["trues"])
        pr = np.concatenate(d["probs"], axis=0)

        row   = {"fire_id": fire_id, "n_patches": len(d["preds"])}
        ious, dices = [], []
        for c in range(n_cls):
            tp   = np.sum((p == c) & (t == c))
            fp   = np.sum((p == c) & (t != c))
            fn   = np.sum((p != c) & (t == c))
            iou  = tp / (tp + fp + fn + 1e-8)
            dice = 2*tp / (2*tp + fp + fn + 1e-8)
            row[f"iou_class{c}"]  = float(iou)
            row[f"dice_class{c}"] = float(dice)
            ious.append(float(iou))
            dices.append(float(dice))
        row["mean_iou"]  = float(np.mean(ious))
        row["mean_dice"] = float(np.mean(dices))
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# SECTION 5  TRAINING LOOP
# =============================================================================

def train_model(model_name, tr_ds, va_ds, te_ds,
                mean, std, device, args, out_dir):
    print(f"\n{'='*60}")
    print(f"  Training {model_name.upper()}")
    print(f"{'='*60}")

    # num_workers=0: data is pre-loaded in RAM, worker processes add overhead
    # pin_memory=False: no benefit when source tensors already live in RAM
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       num_workers=0, pin_memory=False, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                       num_workers=0, pin_memory=False)
    te_ld = DataLoader(te_ds, batch_size=args.batch, shuffle=False,
                       num_workers=0, pin_memory=False)

    # Inverse-frequency class weights — sample from pre-loaded cache (no disk I/O)
    # Rare classes (Very High severity) get upweighted so the loss doesn't ignore them.
    # Using only first 200 patches for speed; representative for a well-shuffled dataset.
    class_counts = np.zeros(N_CLASSES, dtype=np.int64)
    for y_tensor in tr_ds._cache_y[:200]:
        y_data = y_tensor.numpy().ravel()
        for c in range(N_CLASSES):
            class_counts[c] += np.sum(y_data == c)
    total         = class_counts.sum() + 1e-8
    class_freq    = class_counts / total
    # 1e-4 floor prevents division by zero if a class has 0 pixels in the sample.
    # Clip to [0.1, 50.0] prevents a single rare class from dominating the gradient.
    class_weights = np.clip(1.0 / (class_freq + 1e-4), 0.1, 50.0)
    class_weights /= class_weights.mean()   # normalize so weights average to 1.0
    print(f"  Class counts (sample): {class_counts}")
    print(f"  Class weights:         {class_weights.round(2)}")

    # Build model
    ModelClass = MODEL_REGISTRY[model_name.lower()]
    model  = ModelClass(input_channels=N_INPUT, num_classes=N_CLASSES,
                        f=F_BASE, drop=DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    criterion = FocalDiceLoss(n_cls=N_CLASSES, alpha=0.5,
                              class_weights=class_weights.tolist())
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=3, min_lr=1e-6)

    # Each model saves to its own checkpoint file so they don't overwrite each other
    ckpt_filename = "best_model.pth" if model_name == "unet" \
                    else f"{model_name}_best_model.pth"
    ckpt_path  = os.path.join(out_dir, ckpt_filename)
    history    = {"train_loss": [], "val_loss": [],
                  "val_miou": [], "val_mdice": [], "val_pr_auc": []}
    best_miou      = -1.0
    best_val_loss  = float("inf")
    patience       = 5        # stop training if EMA mIoU doesn't improve for 5 epochs
    no_improve     = 0        # counter of consecutive non-improving epochs
    ema_miou       = None     # exponential moving average of val mIoU (smoothed signal)
    ema_alpha      = 0.3      # EMA weight for new epoch; lower = smoother (less reactive to noise)
    # Using EMA instead of raw mIoU for checkpoint selection prevents saving a
    # lucky high-variance epoch as 'best' — the checkpoint reflects sustained improvement.

    # ── Sanity-check batch ───────────────────────────────────────────────────
    print("  Running sanity-check batch ...")
    model.train()
    X_s, y_s = next(iter(tr_ld))
    X_s, y_s = X_s.to(device), y_s.to(device)
    assert X_s.shape == (args.batch, N_INPUT, PATCH_SZ, PATCH_SZ), \
        f"Unexpected input shape: {X_s.shape}"
    assert y_s.shape == (args.batch, PATCH_SZ, PATCH_SZ), \
        f"Unexpected label shape: {y_s.shape}"
    with torch.no_grad():
        logits_s = model(X_s)
    assert logits_s.shape == (args.batch, N_CLASSES, PATCH_SZ, PATCH_SZ), \
        f"Unexpected output shape: {logits_s.shape}"
    loss_s = criterion(logits_s, y_s)
    assert torch.isfinite(loss_s), f"Non-finite sanity loss: {loss_s.item()}"
    # Quick prediction visualization (save 4 sample patches)
    sanity_fig = os.path.join(out_dir, f"{model_name}_sanity.png")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i in range(4):
            rgb = X_s[i, :3].cpu().numpy().transpose(1, 2, 0)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            axes[0, i].imshow(np.clip(rgb, 0, 1))
            axes[0, i].set_title(f"Input {i}")
            axes[0, i].axis("off")
            axes[1, i].imshow(y_s[i].cpu().numpy(), vmin=0, vmax=3)
            axes[1, i].set_title(f"Label {i}")
            axes[1, i].axis("off")
        plt.suptitle(f"Sanity check — loss={loss_s.item():.4f}")
        plt.tight_layout()
        fig.savefig(sanity_fig, dpi=80, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass
    print(f"  Sanity OK — input{list(X_s.shape)} output{list(logits_s.shape)} "
          f"loss={loss_s.item():.4f}  saved: {sanity_fig}")
    # ─────────────────────────────────────────────────────────────────────────

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        tr_loss = 0.0
        for X, y in tr_ld:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= max(len(tr_ld), 1)

        # Validate
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for X, y in va_ld:
                X, y = X.to(device), y.to(device)
                va_loss += criterion(model(X), y).item()
        va_loss /= max(len(va_ld), 1)

        metrics = compute_metrics(model, va_ld, device)
        scheduler.step(va_loss)   # ReduceLROnPlateau needs val_loss

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_miou"].append(metrics["mean_iou"])
        history["val_mdice"].append(metrics["mean_dice"])
        history["val_pr_auc"].append(metrics["mean_pr_auc"])

        # Exponential smoothing of val mIoU
        raw_miou = metrics["mean_iou"]
        if ema_miou is None:
            ema_miou = raw_miou
        else:
            ema_miou = ema_alpha * raw_miou + (1 - ema_alpha) * ema_miou

        elapsed = (time.time() - t0) / 60
        cur_lr  = optimizer.param_groups[0]["lr"]
        print(f"  Ep {epoch:3d}/{args.epochs} | "
              f"TrLoss {tr_loss:.4f} | ValLoss {va_loss:.4f} | "
              f"mIoU {raw_miou:.4f} (ema={ema_miou:.4f}) | "
              f"mDice {metrics['mean_dice']:.4f} | "
              f"LR {cur_lr:.2e} | {elapsed:.1f} min")

        # Save checkpoint ONLY when smoothed val mIoU improves → best_model.pth
        if ema_miou > best_miou:
            best_miou     = ema_miou
            best_val_loss = va_loss
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "miou": best_miou, "val_loss": va_loss,
                        "metrics": metrics}, ckpt_path)
            print(f"    ✓ Saved best_model.pth  (ema_mIoU={best_miou:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} — "
                      f"smoothed mIoU did not improve for {patience} epochs")
                break

    # ── Save training / validation curves ───────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        epochs_ran = list(range(1, len(history["train_loss"]) + 1))
        curves_path = os.path.join(out_dir, f"{model_name}_curves.png")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=DARK_BG)
        for ax in axes:
            ax.set_facecolor(DARK_BG)
            ax.tick_params(colors="white")
            for sp in ax.spines.values():
                sp.set_edgecolor("#444")

        # Loss curve
        axes[0].plot(epochs_ran, history["train_loss"], color="#4fc3f7", label="Train Loss")
        axes[0].plot(epochs_ran, history["val_loss"],   color="#ef5350", label="Val Loss")
        axes[0].set_xlabel("Epoch", color="white")
        axes[0].set_ylabel("Loss",  color="white")
        axes[0].set_title(f"{model_name.upper()} — Loss", color="white")
        axes[0].legend(facecolor="#16213e", labelcolor="white")

        # mIoU curve
        axes[1].plot(epochs_ran, history["val_miou"],  color="#66bb6a", label="Val mIoU")
        axes[1].plot(epochs_ran, history["val_mdice"], color="#ffa726", label="Val mDice", linestyle="--")
        axes[1].set_xlabel("Epoch", color="white")
        axes[1].set_ylabel("Score", color="white")
        axes[1].set_title(f"{model_name.upper()} — mIoU & mDice", color="white")
        axes[1].legend(facecolor="#16213e", labelcolor="white")

        plt.tight_layout()
        fig.savefig(curves_path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"  Training curves saved -> {curves_path}")
    except Exception as e:
        print(f"  Warning: could not save curves: {e}")
    # ─────────────────────────────────────────────────────────────────────────

    # Load best checkpoint for test evaluation
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    test_metrics = compute_metrics(model, te_ld, device)
    print(f"\n  Test mIoU  : {test_metrics['mean_iou']:.4f}")
    print(f"  Test mDice : {test_metrics['mean_dice']:.4f}")
    print(f"  Test PR-AUC: {test_metrics['mean_pr_auc']:.4f}")

    # Per-class details
    for c, name in enumerate(SEVERITY_NAMES):
        print(f"    {name:<15} IoU={test_metrics['iou'][c]:.4f}  "
              f"Dice={test_metrics['dice'][c]:.4f}  "
              f"AP={test_metrics['pr_auc'][c]:.4f}")

    # Save metrics JSON
    result = {
        "model":      model_name,
        "n_params":   n_params,
        "best_epoch": int(ckpt["epoch"]),
        "val_miou":   float(best_miou),
        "test":       test_metrics,
        "history":    history,
    }
    metrics_path = os.path.join(out_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    # Confusion matrix
    all_preds, all_trues = [], []
    model.eval()
    with torch.no_grad():
        for X, y in te_ld:
            X = X.to(device)
            all_preds.extend(model(X).argmax(1).cpu().numpy().ravel())
            all_trues.extend(y.numpy().ravel())
    cm = confusion_matrix(all_trues, all_preds, labels=list(range(N_CLASSES)))
    np.save(os.path.join(out_dir, f"{model_name}_confusion.npy"), cm)

    # PR curve data
    all_trues_pr, all_probs_pr = [], []
    with torch.no_grad():
        for X, y in te_ld:
            X = X.to(device)
            probs = F.softmax(model(X), dim=1)
            all_probs_pr.append(probs.cpu().numpy().transpose(0,2,3,1).reshape(-1, N_CLASSES))
            all_trues_pr.append(y.numpy().ravel())
    trues_pr = np.concatenate(all_trues_pr)
    probs_pr = np.concatenate(all_probs_pr, axis=0)
    pr_data  = {}
    for c in range(N_CLASSES):
        prec, rec, _ = precision_recall_curve(
            (trues_pr == c).astype(int), probs_pr[:, c])
        pr_data[f"prec_c{c}"] = prec.astype(np.float32)
        pr_data[f"rec_c{c}"]  = rec.astype(np.float32)
    np.savez(os.path.join(out_dir, f"{model_name}_pr_data.npz"), **pr_data)

    # Per-fire metrics
    print("  Computing per-fire metrics ...")
    pf_df = compute_per_fire_metrics(model, te_ld, device)
    pf_path = os.path.join(out_dir, "per_fire_metrics.json")
    pf_df.to_json(pf_path, orient="records", indent=2)
    print(f"  Per-fire metrics -> {pf_path}")

    return result, model, history


# =============================================================================
# SECTION 6  VISUALIZATIONS
# =============================================================================

def _cmap_severity():
    return mcolors.ListedColormap(SEVERITY_COLORS)


def plot_training_curves(histories, save_path):
    n = len(histories)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5*rows),
                             facecolor=DARK_BG)
    fig.suptitle("Training Curves -- Sentinel-2 Burn Scar Segmentation v2",
                 color="white", fontsize=14, y=1.01)
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    palette = ["#00b4d8", "#f77f00", "#06d6a0", "#ef476f"]
    for ax in axes:
        ax.set_facecolor("#16213e")
        for sp in ax.spines.values(): sp.set_color("#444466")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")

    for i, (mname, hist) in enumerate(histories.items()):
        if i >= len(axes): break
        ax  = axes[i]
        col = palette[i % len(palette)]
        ep  = range(1, len(hist["train_loss"]) + 1)
        ax.plot(ep, hist["train_loss"], color=col,        lw=1.5, label="Train Loss")
        ax.plot(ep, hist["val_loss"],   color=col, ls="--", lw=1.5, label="Val Loss")
        ax2 = ax.twinx()
        ax2.set_facecolor("#16213e")
        ax2.tick_params(colors="white")
        ax2.yaxis.label.set_color("#ffd166")
        ax2.plot(ep, hist["val_miou"], color="#ffd166", lw=2, label="Val mIoU")
        ax.set_title(mname.upper(), fontsize=11)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax2.set_ylabel("mIoU")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, labels1+labels2, fontsize=8,
                  facecolor="#16213e", labelcolor="white", loc="upper right")

    for ax in axes[n:]: ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_confusion_matrix_grid(all_results, out_dir, save_path):
    """2x2 grid of confusion matrices for all 4 models."""
    models = list(all_results.keys())
    n_rows, n_cols = 2, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 12),
                             facecolor=DARK_BG)
    for ax in axes.ravel():
        ax.set_facecolor("#16213e")

    for i, mname in enumerate(models[:4]):
        ax  = axes.ravel()[i]
        cm_path = os.path.join(out_dir, f"{mname}_confusion.npy")
        if not os.path.exists(cm_path):
            ax.axis("off")
            continue
        cm  = np.load(cm_path).astype(float)
        row_sum = cm.sum(axis=1, keepdims=True) + 1e-8
        cm_norm = cm / row_sum
        im  = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax).ax.tick_params(colors="white")
        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(SEVERITY_NAMES, color="white", rotation=30, ha="right")
        ax.set_yticklabels(SEVERITY_NAMES, color="white")
        ax.set_xlabel("Predicted", color="white")
        ax.set_ylabel("True", color="white")
        ax.set_title(mname.upper(), color="white", fontsize=11)
        for ri in range(N_CLASSES):
            for ci in range(N_CLASSES):
                ax.text(ci, ri, f"{cm_norm[ri,ci]:.2f}", ha="center", va="center",
                        color="white" if cm_norm[ri,ci] < 0.5 else "black",
                        fontsize=9)

    for ax in axes.ravel()[len(models):]: ax.axis("off")
    plt.suptitle("Confusion Matrices (Normalized)", color="white", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_pr_curves(all_results, out_dir, save_path):
    """PR curves for all models."""
    fig, axes = plt.subplots(1, N_CLASSES, figsize=(16, 5), facecolor=DARK_BG)
    palette   = ["#00b4d8", "#f77f00", "#06d6a0", "#ef476f"]

    for ci, (ax, cls_name) in enumerate(zip(axes, SEVERITY_NAMES)):
        ax.set_facecolor("#16213e")
        for sp in ax.spines.values(): sp.set_color("#444466")
        ax.tick_params(colors="white")
        ax.set_title(cls_name, color="white", fontsize=10)
        ax.set_xlabel("Recall",    color="white")
        ax.set_ylabel("Precision", color="white")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.grid(color="#333355", lw=0.5)

        for i, mname in enumerate(all_results.keys()):
            pr_path = os.path.join(out_dir, f"{mname}_pr_data.npz")
            if not os.path.exists(pr_path):
                continue
            pr = np.load(pr_path)
            prec = pr.get(f"prec_c{ci}", np.array([]))
            rec  = pr.get(f"rec_c{ci}",  np.array([]))
            if len(prec) == 0: continue
            ap   = all_results[mname]["test"]["pr_auc"][ci]
            ax.plot(rec, prec, color=palette[i % len(palette)], lw=2,
                    label=f"{mname.upper()} (AP={ap:.3f})")

        ax.legend(facecolor="#16213e", labelcolor="white", fontsize=8)

    plt.suptitle("Precision-Recall Curves by Severity Class",
                 color="white", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_per_class_iou(all_results, save_path):
    """Grouped bar chart of per-class IoU for all models."""
    models  = list(all_results.keys())
    x       = np.arange(N_CLASSES)
    width   = 0.20
    palette = ["#00b4d8", "#f77f00", "#06d6a0", "#ef476f"]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK_BG)
    ax.set_facecolor("#16213e")
    for sp in ax.spines.values(): sp.set_color("#444466")
    ax.tick_params(colors="white")

    for i, mname in enumerate(models):
        iou_vals = all_results[mname]["test"]["iou"]
        offset   = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, iou_vals, width,
                      label=mname.upper(),
                      color=palette[i % len(palette)],
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, iou_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom",
                    color="white", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(SEVERITY_NAMES, color="white", fontsize=10)
    ax.set_ylabel("IoU", color="white")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class IoU by Model", color="white", fontsize=13)
    ax.legend(facecolor="#16213e", labelcolor="white")
    ax.yaxis.label.set_color("white")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_metric_comparison(all_results, save_path):
    """Grouped bar chart: mIoU, mDice, PR-AUC for all models."""
    models   = list(all_results.keys())
    metrics  = ["mean_iou", "mean_dice", "mean_pr_auc"]
    labels   = ["mIoU", "mDice", "PR-AUC"]
    palette2 = ["#00b4d8", "#f77f00", "#06d6a0"]

    x     = np.arange(len(models))
    width = 0.22
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=DARK_BG)
    ax.set_facecolor("#16213e")

    for j, (metric, label, col) in enumerate(zip(metrics, labels, palette2)):
        vals = [all_results[m]["test"][metric] for m in models]
        bars = ax.bar(x + j*width, vals, width, label=label,
                      color=col, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom",
                    color="white", fontsize=7.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.upper() for m in models], color="white", fontsize=10)
    ax.set_ylabel("Score", color="white")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison -- Test Set Metrics v2",
                 color="white", fontsize=13)
    ax.legend(facecolor="#16213e", labelcolor="white")
    for sp in ax.spines.values(): sp.set_color("#444466")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.axhline(0.5, color="white", lw=0.5, ls="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_radar_comparison(all_results, save_path):
    """Polar radar chart comparing 5 metrics across all models."""
    categories = ["mIoU", "mDice", "PR-AUC", "Mod-IoU", "High-IoU"]
    N          = len(categories)
    angles     = [n / float(N) * 2 * np.pi for n in range(N)]
    angles    += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True),
                           facecolor=DARK_BG)
    ax.set_facecolor("#16213e")
    palette = ["#00b4d8", "#f77f00", "#06d6a0", "#ef476f"]

    for i, (mname, res) in enumerate(all_results.items()):
        t = res["test"]
        vals = [
            t["mean_iou"],
            t["mean_dice"],
            t["mean_pr_auc"],
            t["iou"][1],   # Moderate
            t["iou"][2],   # High
        ]
        vals += vals[:1]
        col = palette[i % len(palette)]
        ax.plot(angles, vals, "o-", linewidth=2, color=col, label=mname.upper())
        ax.fill(angles, vals, alpha=0.15, color=col)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="white", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], color="#888888", fontsize=8)
    ax.spines["polar"].set_color("#444466")
    ax.grid(color="#333355", lw=0.5)
    ax.set_title("Model Comparison -- Radar Chart", color="white",
                 fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              facecolor="#16213e", labelcolor="white", fontsize=10)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_severity_overview(test_files, model, device, mean, std, save_path,
                           n_fires=2, model_name=None):
    """
    3-row overview: Pre-RGB | dNBR | GT | Pred | Error | Confidence
    """
    from matplotlib.gridspec import GridSpec

    model.eval()
    cmap_sev = _cmap_severity()
    norm_sev = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_sev.N)
    cmap_err = mcolors.ListedColormap(["#d62728", "#2ca02c"])

    rng = np.random.default_rng(SEED + 99)
    sel = rng.choice(test_files, min(n_fires, len(test_files)), replace=False)

    n_cols = n_fires * 2
    fig    = plt.figure(figsize=(n_cols * 3.5, 12), facecolor=DARK_BG)
    gs     = GridSpec(3, n_cols, figure=fig, hspace=0.08, wspace=0.04)

    mean_t = mean[:3] if len(mean) >= 3 else np.zeros(3)
    std_t  = std[:3]  if len(std) >= 3  else np.ones(3)

    for fi, fp in enumerate(sel):
        data   = np.load(fp, allow_pickle=True)
        X_raw  = data["X"].astype(np.float32)   # (7,256,256) normalized
        y_true = data["y"].astype(np.int64)

        # Denormalize channels 0-2 (Red, Green, Blue)
        rgb = X_raw[:3].copy()
        for c in range(3):
            rgb[c] = rgb[c] * std_t[c] + mean_t[c]
        # Stack as R, G, B
        rgb  = np.stack([rgb[0], rgb[1], rgb[2]])
        rmin, rmax = rgb.min(), rgb.max()
        pre_rgb = np.clip((rgb - rmin) / (rmax - rmin + 1e-8), 0, 1).transpose(1,2,0)

        # dNBR is channel 6
        dnbr = X_raw[6]

        # Prediction
        X_t = torch.from_numpy(X_raw).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(X_t)
            probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred       = probs.argmax(axis=0)
        confidence = probs.max(axis=0)
        error      = (pred == y_true).astype(np.uint8)

        base_col = fi * 2

        # Row 0: Pre-fire RGB | dNBR heatmap
        ax = fig.add_subplot(gs[0, base_col])
        ax.imshow(pre_rgb); ax.axis("off")
        try:
            fire_id = str(data["fire_id"])
        except Exception:
            fire_id = Path(fp).stem
        ax.set_title(fire_id[:18], color="white", fontsize=8)

        ax = fig.add_subplot(gs[0, base_col+1])
        ax.imshow(dnbr, cmap="RdYlGn_r", vmin=-0.5, vmax=1.0)
        ax.axis("off"); ax.set_title("dNBR", color="white", fontsize=8)

        # Row 1: GT | Prediction
        ax = fig.add_subplot(gs[1, base_col])
        ax.imshow(y_true, cmap=cmap_sev, norm=norm_sev, interpolation="nearest")
        ax.axis("off"); ax.set_title("Ground Truth", color="white", fontsize=8)

        ax = fig.add_subplot(gs[1, base_col+1])
        ax.imshow(pred, cmap=cmap_sev, norm=norm_sev, interpolation="nearest")
        ax.axis("off"); ax.set_title("Prediction", color="white", fontsize=8)

        # Row 2: Error | Confidence
        ax = fig.add_subplot(gs[2, base_col])
        ax.imshow(error, cmap=cmap_err, vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"Error (acc={error.mean()*100:.1f}%)", color="white", fontsize=8)

        ax = fig.add_subplot(gs[2, base_col+1])
        ax.imshow(confidence, cmap="plasma", vmin=0, vmax=1)
        ax.axis("off"); ax.set_title("Confidence", color="white", fontsize=8)

    handles = [mpatches.Patch(color=SEVERITY_COLORS[i], label=SEVERITY_NAMES[i])
               for i in range(N_CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               facecolor="#16213e", labelcolor="white", fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    title = f"Severity Mapping Overview — {model_name}" if model_name \
            else "Severity Mapping Overview — Best Model"
    plt.suptitle(title,
                 color="white", fontsize=13, y=1.01)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_patch_grid(test_files, save_path, n_cols=8, n_rows=4):
    """4x8 grid of sample patches coloured by severity class."""
    cmap = _cmap_severity()
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    rng = np.random.default_rng(SEED + 1)
    sel = rng.choice(test_files, min(n_cols*n_rows, len(test_files)), replace=False)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols*2.5, n_rows*2.5),
                             facecolor=DARK_BG)
    for ax, fp in zip(axes.ravel(), sel):
        data = np.load(fp, allow_pickle=True)
        X    = data["X"].astype(np.float32)
        y    = data["y"].astype(np.int64)
        # Channels 0,1,2 = Red, Green, Blue
        rgb  = X[[0,1,2]]
        rgb  = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        rgb  = np.clip(rgb.transpose(1,2,0), 0, 1)
        ax.imshow(rgb)
        ax.imshow(y, cmap=cmap, norm=norm, alpha=0.45, interpolation="nearest")
        ax.axis("off")

    for ax in axes.ravel()[len(sel):]:
        ax.axis("off")

    handles = [mpatches.Patch(color=SEVERITY_COLORS[i], label=SEVERITY_NAMES[i])
               for i in range(N_CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               facecolor="#16213e", labelcolor="white",
               bbox_to_anchor=(0.5, -0.01))
    plt.suptitle("Sample 256x256 Sentinel-2 Patches with Severity Labels (v2)",
                 color="white", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# =============================================================================
# SECTION 7  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="U-Net burn-scar segmentation v2 (7 channels, 4 classes)")
    parser.add_argument("--model",       default="all",
                        help="unet|resunet|attention|unetpp|all")
    parser.add_argument("--epochs",      type=int,   default=60)
    parser.add_argument("--batch",       type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--patch_dir",   type=str,
                        default=os.path.join(BASE, "data", "sentinel2", "patches_v2"))
    parser.add_argument("--splits_json", type=str,
                        default=os.path.join(BASE, "data", "sentinel2", "splits_v2.json"))
    parser.add_argument("--out_dir",     type=str,
                        default=os.path.join(BASE, "outputs", "sentinel2_results_v2"))
    args = parser.parse_args()

    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(fig_dir,      exist_ok=True)

    # Full reproducibility
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load splits
    print("\nLoading data splits ...")
    train_files, val_files, test_files = load_splits(args.patch_dir, args.splits_json)
    print(f"  Train: {len(train_files)} patches")
    print(f"  Val  : {len(val_files)} patches")
    print(f"  Test : {len(test_files)} patches")

    if not train_files:
        print("ERROR: No patch files found. Run sentinel2_dataset_v2.py first.")
        return

    # Channel statistics — computed from TRAINING split ONLY
    # Applied consistently to val/test (no data leakage from val/test)
    # If already computed by a previous job, reuse to save ~6 minutes.
    stats_path = os.path.join(args.out_dir, "channel_stats.npz")
    if os.path.exists(stats_path):
        s    = np.load(stats_path)
        mean = s["mean"].astype(np.float32)
        std  = s["std"].astype(np.float32)
        print(f"\nChannel stats loaded from cache -> {stats_path}")
    else:
        print("\nComputing channel statistics from training split only ...")
        mean, std = compute_channel_stats(train_files)
        np.savez(stats_path, mean=mean, std=std)
        print(f"  Channel stats saved -> {stats_path}")
    print(f"  Mean: {mean.round(4)}")
    print(f"  Std : {std.round(4)}")

    # Select models
    if args.model.lower() == "all":
        models_to_run = list(MODEL_REGISTRY.keys())
    else:
        models_to_run = [m.strip() for m in args.model.lower().split(",")]
        # Support "unetplusplus" alias
        models_to_run = ["unetpp" if m in ("unetplusplus", "unet++") else m
                         for m in models_to_run]

    # Load datasets ONCE — all 4 models share identical files and normalisation.
    # Reason: Loading inside each train_model() call would double-allocate ~28 GB RAM
    # during the model handoff (old + new cache in RAM simultaneously) → OOM kill.
    # Val/test do NOT use augmentation — only training data is augmented.
    import gc
    print("\nPre-loading shared datasets into RAM (done once for all models) ...")
    tr_ds_shared = WildfireDatasetV2(train_files, augment=True,  mean=mean, std=std)
    va_ds_shared = WildfireDatasetV2(val_files,   augment=False, mean=mean, std=std)
    te_ds_shared = WildfireDatasetV2(test_files,  augment=False, mean=mean, std=std)
    print("Shared dataset cache ready.\n")

    # Train
    all_results    = {}
    all_histories  = {}
    best_model_obj = None
    best_model_name = None
    best_miou_global = -1.0

    for mname in models_to_run:
        if mname not in MODEL_REGISTRY:
            print(f"  WARNING: unknown model '{mname}', skipping")
            continue

        # Skip if checkpoint + metrics already exist.
        # Enables safe resume after GPU OOM crash or SLURM preemption —
        # already-trained models are loaded from disk rather than re-trained.
        ckpt_fname   = "best_model.pth" if mname == "unet" else f"{mname}_best_model.pth"
        ckpt_exists  = os.path.exists(os.path.join(args.out_dir, ckpt_fname))
        metrics_path = os.path.join(args.out_dir, f"{mname}_metrics.json")
        if ckpt_exists and os.path.exists(metrics_path):
            print(f"\n  ✓ {mname.upper()} checkpoint + metrics already exist — skipping training.")
            with open(metrics_path) as f:
                saved = json.load(f)
            all_results[mname]   = saved
            all_histories[mname] = {}
            if saved.get("test", {}).get("mean_iou", -1) > best_miou_global:
                best_miou_global = saved["test"]["mean_iou"]
                best_model_name  = mname
            continue

        result, model_obj, history = train_model(
            mname, tr_ds_shared, va_ds_shared, te_ds_shared,
            mean, std, device, args, args.out_dir)
        all_results[mname]   = result
        all_histories[mname] = history
        if result["test"]["mean_iou"] > best_miou_global:
            best_miou_global  = result["test"]["mean_iou"]
            best_model_obj    = model_obj
            best_model_name   = mname

    # Save combined results
    combined_path = os.path.join(args.out_dir, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results -> {combined_path}")

    # Summary table
    print("\n" + "="*70)
    print(f"{'Model':<20} {'Params':>10} {'mIoU':>8} {'mDice':>8} {'PR-AUC':>8}")
    print("-"*70)
    for mname, res in all_results.items():
        tm = res["test"]
        print(f"{mname:<20} {res['n_params']:>10,} "
              f"{tm['mean_iou']:>8.4f} {tm['mean_dice']:>8.4f} "
              f"{tm['mean_pr_auc']:>8.4f}")
    print("="*70)

    # Figures
    if all_histories:
        print("\nGenerating figures ...")

        plot_training_curves(
            all_histories,
            os.path.join(fig_dir, "training_curves.png"))

        if len(all_results) > 1:
            plot_confusion_matrix_grid(
                all_results, args.out_dir,
                os.path.join(fig_dir, "confusion_matrix.png"))

            plot_pr_curves(
                all_results, args.out_dir,
                os.path.join(fig_dir, "pr_curves.png"))

            plot_per_class_iou(
                all_results,
                os.path.join(fig_dir, "per_class_iou.png"))

            plot_metric_comparison(
                all_results,
                os.path.join(fig_dir, "metric_comparison.png"))

            plot_radar_comparison(
                all_results,
                os.path.join(fig_dir, "radar_comparison.png"))

        if best_model_obj is not None and test_files:
            plot_severity_overview(
                test_files, best_model_obj, device, mean, std,
                os.path.join(fig_dir, "severity_overview.png"))

            plot_patch_grid(
                test_files,
                os.path.join(fig_dir, "patch_grid.png"))

    print(f"\nBest model: {best_model_name} (mIoU = {best_miou_global:.4f})")
    print(f"Figures saved -> {fig_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
