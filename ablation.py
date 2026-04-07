"""
ablation.py
===========
Controlled ablation study for the UNet burn-severity model.

Three experiments — all other hyperparameters identical to main training:

  Ablation A | No spectral indices   | Input channels = 5 (RGB + NIR + SWIR1)
  Ablation B | No data augmentation  | augment=False on training split
  Ablation C | CrossEntropy loss     | Replace FocalDiceLoss with nn.CrossEntropyLoss

Outputs (under --out_dir):
  ablation_A_metrics.json
  ablation_B_metrics.json
  ablation_C_metrics.json
  ablation_summary.json
  figures/ablation_curves_A.png
  figures/ablation_curves_B.png
  figures/ablation_curves_C.png
  figures/ablation_comparison.png

Usage
-----
  python ablation.py                        # all three
  python ablation.py --run A                # single ablation
  python ablation.py --run A,B              # subset
  python ablation.py --epochs 30 --batch 8  # override epochs / batch
"""

import os
import sys
import json
import gc
import time
import random
import argparse
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Import shared components from unet_training_v2  (no side-effects on import)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet_training_v2 import (
    # constants
    N_CLASSES, PATCH_SZ, SEED, F_BASE, DROPOUT, DARK_BG,
    SEVERITY_NAMES,
    # model
    UNet,
    # data helpers
    load_splits, compute_channel_stats, _augment,
    # loss
    FocalDiceLoss,
    # metrics
    compute_metrics,
)

# ---------------------------------------------------------------------------
# Paths & fixed hyper-parameters
# ---------------------------------------------------------------------------
BASE        = os.environ.get("WILDFIRE_BASE_DIR",
              os.path.dirname(os.path.abspath(__file__)))
PATCH_DIR   = os.path.join(BASE, "data", "sentinel2", "patches_v2")
SPLITS_JSON = os.path.join(BASE, "data", "sentinel2", "splits_v2.json")
OUT_DIR     = os.path.join(BASE, "outputs", "ablation")

# Full 7-channel layout: [Red, Green, Blue, NIR, SWIR1, NBR, dNBR]
ALL_CHANNELS   = 7
# Ablation A drops NBR (ch5) and dNBR (ch6) — keeps only first 5
REDUCED_CHANNELS = 5   # [Red, Green, Blue, NIR, SWIR1]

LR           = 3e-4
WEIGHT_DECAY = 1e-4
BATCH        = 8
EPOCHS       = 60
PATIENCE     = 5
EMA_ALPHA    = 0.3


# =============================================================================
# SECTION 1  ABLATION DATASET
# =============================================================================

class AblationDataset(Dataset):
    """
    Thin wrapper around .npz patches that supports:
      - Selecting a subset of input channels  (ablation A)
      - Enabling / disabling augmentation     (ablation B)
      - Consistent normalization from training stats
    """

    def __init__(self, patch_files, channels, augment, mean, std):
        """
        Parameters
        ----------
        patch_files : list[str]   paths to .npz files
        channels    : list[int]   which channel indices to keep (e.g. [0..6] or [0..4])
        augment     : bool        apply training augmentations
        mean        : np.ndarray  (len(channels),) training-split mean
        std         : np.ndarray  (len(channels),) training-split std
        """
        self.channels = channels
        self.augment  = augment
        self.mean     = torch.from_numpy(mean.astype(np.float32))
        self.std      = torch.from_numpy(std.astype(np.float32))

        # Pre-load all patches into RAM to eliminate per-epoch NFS I/O
        files = list(patch_files)
        print(f"    Pre-loading {len(files)} patches into RAM ...", flush=True)
        self._cache_X = []
        self._cache_y = []
        for fp in files:
            data = np.load(fp, allow_pickle=True)
            X = torch.from_numpy(data["X"].astype(np.float32))[channels]  # (C,256,256)
            y = torch.from_numpy(data["y"].astype(np.int64)).clamp(0, N_CLASSES - 1)
            X = (X - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-8)
            self._cache_X.append(X)
            self._cache_y.append(y)
        print(f"    Cache ready ({len(self._cache_X)} patches).", flush=True)

    def __len__(self):
        return len(self._cache_X)

    def __getitem__(self, idx):
        X = self._cache_X[idx]
        y = self._cache_y[idx]

        # Augmentation — reuse _augment from unet_training_v2
        # Note: _augment operates on photometric channels 0–4; if reduced set
        # has fewer than 5 channels we apply noise/brightness only to available.
        if self.augment:
            X, y = _augment_safe(X, y, n_spectral=min(5, len(self.channels)))

        return X, y


def _augment_safe(X, y, n_spectral=5):
    """
    Geometric + photometric augmentation that works for any number of channels.
    Identical to _augment in unet_training_v2 but parameterises the spectral
    channel count so Ablation A (5 channels) doesn't index out of range.
    """
    # 1. Horizontal flip
    if torch.rand(1) > 0.5:
        X = torch.flip(X, dims=[2])
        y = torch.flip(y, dims=[1])
    # 2. Vertical flip
    if torch.rand(1) > 0.5:
        X = torch.flip(X, dims=[1])
        y = torch.flip(y, dims=[0])
    # 3. 90/180/270 rotation
    if torch.rand(1) > 0.5:
        k = torch.randint(1, 4, (1,)).item()
        X = torch.rot90(X, k, dims=[1, 2])
        y = torch.rot90(y, k, dims=[0, 1])
    # 4. Free rotation 0–360°
    if torch.rand(1) > 0.5:
        angle     = float(torch.empty(1).uniform_(0, 360))
        rad       = angle * 3.14159265 / 180.0
        cos_a, sin_a = float(np.cos(rad)), float(np.sin(rad))
        theta = torch.tensor([[cos_a, -sin_a, 0.0],
                               [sin_a,  cos_a, 0.0]], dtype=torch.float32)
        grid  = F.affine_grid(theta.unsqueeze(0),
                              (1, 1, PATCH_SZ, PATCH_SZ), align_corners=False)
        X = F.grid_sample(X.unsqueeze(0), grid, mode="bilinear",
                          padding_mode="reflection",
                          align_corners=False).squeeze(0)
        y = F.grid_sample(y.float().unsqueeze(0).unsqueeze(0), grid,
                          mode="nearest", padding_mode="reflection",
                          align_corners=False).squeeze().long()
    # 5. Random crop + resize
    if torch.rand(1) > 0.5:
        scale     = float(torch.empty(1).uniform_(0.80, 1.0))
        crop_sz   = int(PATCH_SZ * scale)
        max_off   = PATCH_SZ - crop_sz
        r0 = torch.randint(0, max_off + 1, (1,)).item()
        c0 = torch.randint(0, max_off + 1, (1,)).item()
        X  = X[:, r0:r0+crop_sz, c0:c0+crop_sz]
        y  = y[r0:r0+crop_sz, c0:c0+crop_sz]
        X  = F.interpolate(X.unsqueeze(0), size=(PATCH_SZ, PATCH_SZ),
                           mode="bilinear", align_corners=False).squeeze(0)
        y  = F.interpolate(y.float().unsqueeze(0).unsqueeze(0),
                           size=(PATCH_SZ, PATCH_SZ),
                           mode="nearest").squeeze().long()
    # 6. Brightness jitter (spectral channels only)
    if torch.rand(1) > 0.5:
        factor = float(torch.empty(1).uniform_(0.75, 1.25))
        X[:n_spectral] = X[:n_spectral] * factor
    # 7. Contrast jitter
    if torch.rand(1) > 0.5:
        factor = float(torch.empty(1).uniform_(0.75, 1.25))
        for c in range(n_spectral):
            ch_mean = X[c].mean()
            X[c]    = (X[c] - ch_mean) * factor + ch_mean
    # 8. Gaussian noise
    if torch.rand(1) > 0.5:
        noise = torch.randn_like(X[:n_spectral]) * 0.03
        X[:n_spectral] = X[:n_spectral] + noise
    return X, y


# =============================================================================
# SECTION 2  CROSS-ENTROPY LOSS WRAPPER  (ablation C)
# =============================================================================

class WeightedCrossEntropyLoss(nn.Module):
    """
    Standard nn.CrossEntropyLoss with inverse-frequency class weights.
    Drop-in replacement for FocalDiceLoss in ablation C.
    """
    def __init__(self, class_weights=None):
        super().__init__()
        w = torch.tensor(class_weights, dtype=torch.float32) \
            if class_weights is not None else None
        self.register_buffer("weight", w)

    def forward(self, logits, targets):
        w = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(logits, targets, weight=w)


# =============================================================================
# SECTION 3  SHARED TRAINING LOOP
# =============================================================================

def run_ablation_training(
    ablation_name,
    train_files,
    val_files,
    test_files,
    channels,          # list[int]  — e.g. list(range(7)) or list(range(5))
    augment_train,     # bool
    loss_fn,           # nn.Module  — FocalDiceLoss or WeightedCrossEntropyLoss
    mean,              # np.ndarray (n_channels,)
    std,               # np.ndarray (n_channels,)
    device,
    epochs,
    batch,
    out_dir,
):
    """
    Self-contained training loop for one ablation.
    Returns (result_dict, history_dict).
    """
    n_ch = len(channels)
    print(f"\n{'='*60}")
    print(f"  Ablation {ablation_name}")
    print(f"  input_channels={n_ch}  augment={augment_train}  "
          f"loss={type(loss_fn).__name__}")
    print(f"{'='*60}")

    # ── Datasets & loaders ──────────────────────────────────────────────────
    tr_ds = AblationDataset(train_files, channels,  augment_train, mean, std)
    va_ds = AblationDataset(val_files,   channels,  False,         mean, std)
    te_ds = AblationDataset(test_files,  channels,  False,         mean, std)

    tr_ld = DataLoader(tr_ds, batch_size=batch, shuffle=True,
                       num_workers=0, pin_memory=False, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=batch, shuffle=False,
                       num_workers=0, pin_memory=False)
    te_ld = DataLoader(te_ds, batch_size=batch, shuffle=False,
                       num_workers=0, pin_memory=False)

    # ── Model ───────────────────────────────────────────────────────────────
    model    = UNet(input_channels=n_ch, num_classes=N_CLASSES,
                    f=F_BASE, drop=DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    criterion = loss_fn.to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=3, min_lr=1e-6)

    ckpt_path = os.path.join(out_dir, f"ablation_{ablation_name}_best.pth")

    history = {"train_loss": [], "val_loss": [],
               "val_miou":   [], "val_mdice": [], "val_pr_auc": []}
    best_miou     = -1.0
    no_improve    = 0
    ema_miou      = None

    # ── Sanity check ────────────────────────────────────────────────────────
    model.train()
    X_s, y_s = next(iter(tr_ld))
    X_s, y_s = X_s.to(device), y_s.to(device)
    assert X_s.shape[1] == n_ch, \
        f"Sanity: expected {n_ch} input channels, got {X_s.shape[1]}"
    with torch.no_grad():
        logits_s = model(X_s)
    assert logits_s.shape[1] == N_CLASSES, \
        f"Sanity: expected {N_CLASSES} output classes, got {logits_s.shape[1]}"
    loss_s = criterion(logits_s, y_s)
    assert torch.isfinite(loss_s), f"Sanity: non-finite loss {loss_s.item()}"
    print(f"  Sanity OK — in{list(X_s.shape)}  out{list(logits_s.shape)}  "
          f"loss={loss_s.item():.4f}")

    # ── Training loop ────────────────────────────────────────────────────────
    # All ablations use identical hyperparameters (LR, batch, patience, EMA)
    # so that the ONLY variable is the component being ablated.
    t0 = time.time()
    for epoch in range(1, epochs + 1):

        # Train — forward pass + backprop with gradient clipping (norm ≤ 1.0)
        model.train()
        tr_loss = 0.0
        for X, y in tr_ld:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= max(len(tr_ld), 1)  # average loss per batch

        # Validate
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for X, y in va_ld:
                X, y = X.to(device), y.to(device)
                va_loss += criterion(model(X), y).item()
        va_loss /= max(len(va_ld), 1)

        val_metrics = compute_metrics(model, va_ld, device)
        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_miou"].append(val_metrics["mean_iou"])
        history["val_mdice"].append(val_metrics["mean_dice"])
        history["val_pr_auc"].append(val_metrics["mean_pr_auc"])

        # EMA smoothing
        raw_miou = val_metrics["mean_iou"]
        ema_miou = (EMA_ALPHA * raw_miou + (1 - EMA_ALPHA) * ema_miou
                    if ema_miou is not None else raw_miou)

        cur_lr  = optimizer.param_groups[0]["lr"]
        elapsed = (time.time() - t0) / 60
        print(f"  Ep {epoch:3d}/{epochs} | "
              f"TrLoss {tr_loss:.4f} | ValLoss {va_loss:.4f} | "
              f"mIoU {raw_miou:.4f} (ema={ema_miou:.4f}) | "
              f"mDice {val_metrics['mean_dice']:.4f} | "
              f"LR {cur_lr:.2e} | {elapsed:.1f} min")

        if ema_miou > best_miou:
            best_miou = ema_miou
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "ema_miou": best_miou, "val_loss": va_loss,
                        "metrics": val_metrics}, ckpt_path)
            print(f"    ✓ Saved {os.path.basename(ckpt_path)}  "
                  f"(ema_mIoU={best_miou:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} — "
                      f"no improvement for {PATIENCE} epochs")
                break

    # ── Training curves ──────────────────────────────────────────────────────
    _save_curves(history, ablation_name, out_dir)

    # ── Load best and evaluate on test ───────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    test_metrics = compute_metrics(model, te_ld, device)
    val_metrics  = compute_metrics(model, va_ld, device)   # re-eval best ckpt

    print(f"\n  [{ablation_name}] Best checkpoint — epoch {ckpt['epoch']}")
    print(f"  Val   mIoU={val_metrics['mean_iou']:.4f}  "
          f"mDice={val_metrics['mean_dice']:.4f}  "
          f"PR-AUC={val_metrics['mean_pr_auc']:.4f}")
    print(f"  Test  mIoU={test_metrics['mean_iou']:.4f}  "
          f"mDice={test_metrics['mean_dice']:.4f}  "
          f"PR-AUC={test_metrics['mean_pr_auc']:.4f}")
    for c, name in enumerate(SEVERITY_NAMES):
        print(f"    {name:<15}  "
              f"IoU={test_metrics['iou'][c]:.4f}  "
              f"Dice={test_metrics['dice'][c]:.4f}")

    result = {
        "ablation":       ablation_name,
        "description":    _ablation_description(ablation_name),
        "input_channels": n_ch,
        "augment_train":  augment_train,
        "loss_fn":        type(loss_fn).__name__,
        "n_params":       n_params,
        "best_epoch":     int(ckpt["epoch"]),
        "val":            val_metrics,
        "test":           test_metrics,
        "history":        history,
    }
    return result


def _ablation_description(name):
    return {
        "A": "No spectral indices — 5 channels (RGB + NIR + SWIR1)",
        "B": "No data augmentation — training without transforms",
        "C": "CrossEntropy loss — FocalDiceLoss replaced by CrossEntropyLoss",
    }.get(name, name)


# =============================================================================
# SECTION 4  CURVES + COMPARISON PLOT
# =============================================================================

def _save_curves(history, ablation_name, out_dir):
    try:
        import matplotlib.pyplot as plt
        fig_dir    = os.path.join(out_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        save_path  = os.path.join(fig_dir, f"ablation_curves_{ablation_name}.png")
        epochs_ran = list(range(1, len(history["train_loss"]) + 1))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=DARK_BG)
        for ax in axes:
            ax.set_facecolor(DARK_BG)
            ax.tick_params(colors="white")
            for sp in ax.spines.values():
                sp.set_edgecolor("#444")

        axes[0].plot(epochs_ran, history["train_loss"],
                     color="#4fc3f7", label="Train Loss")
        axes[0].plot(epochs_ran, history["val_loss"],
                     color="#ef5350", label="Val Loss")
        axes[0].set_xlabel("Epoch", color="white")
        axes[0].set_ylabel("Loss",  color="white")
        axes[0].set_title(f"Ablation {ablation_name} — Loss", color="white")
        axes[0].legend(facecolor="#16213e", labelcolor="white")

        axes[1].plot(epochs_ran, history["val_miou"],
                     color="#66bb6a", label="Val mIoU")
        axes[1].plot(epochs_ran, history["val_mdice"],
                     color="#ffa726", label="Val mDice", linestyle="--")
        axes[1].set_xlabel("Epoch", color="white")
        axes[1].set_ylabel("Score", color="white")
        axes[1].set_title(f"Ablation {ablation_name} — mIoU & mDice",
                          color="white")
        axes[1].legend(facecolor="#16213e", labelcolor="white")

        plt.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"  Curves saved -> {save_path}")
    except Exception as e:
        print(f"  Warning: could not save curves: {e}")


def plot_ablation_comparison(summary, out_dir):
    """
    Grouped bar chart: mIoU / mDice / PR-AUC for each ablation on val + test.
    """
    try:
        import matplotlib.pyplot as plt

        fig_dir   = os.path.join(out_dir, "figures")
        save_path = os.path.join(fig_dir, "ablation_comparison.png")

        ablations = list(summary.keys())       # ["A", "B", "C"]
        metrics   = ["mean_iou", "mean_dice", "mean_pr_auc"]
        labels    = ["mIoU", "mDice", "PR-AUC"]
        splits    = ["val", "test"]
        colors    = {"val": "#4fc3f7", "test": "#ef5350"}

        fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=DARK_BG)
        x = np.arange(len(ablations))
        width = 0.35

        for ax, metric, label in zip(axes, metrics, labels):
            ax.set_facecolor(DARK_BG)
            ax.tick_params(colors="white")
            for sp in ax.spines.values():
                sp.set_edgecolor("#444")

            for i, split in enumerate(splits):
                vals = [summary[a][split][metric]
                        if summary[a].get(split) else 0.0
                        for a in ablations]
                offset = (i - 0.5) * width
                bars = ax.bar(x + offset, vals, width,
                              color=colors[split], alpha=0.85, label=split)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{v:.3f}", ha="center", va="bottom",
                            color="white", fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"Abl.{a}\n{_ablation_description(a)[:25]}..."
                 for a in ablations],
                color="white", fontsize=7)
            ax.set_ylabel(label, color="white")
            ax.set_ylim(0, 1.05)
            ax.set_title(label, color="white")
            ax.legend(facecolor="#16213e", labelcolor="white")

        plt.suptitle("Ablation Study — UNet Comparison",
                     color="white", fontsize=12)
        plt.tight_layout()
        fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"  Comparison chart saved -> {save_path}")
    except Exception as e:
        print(f"  Warning: could not save comparison chart: {e}")


# =============================================================================
# SECTION 5  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ablation study for UNet burn-severity model")
    parser.add_argument("--run",         type=str, default="A,B,C",
                        help="Comma-separated ablations to run (A, B, C, or all)")
    parser.add_argument("--epochs",      type=int,   default=EPOCHS)
    parser.add_argument("--batch",       type=int,   default=BATCH)
    parser.add_argument("--patch_dir",   type=str,   default=PATCH_DIR)
    parser.add_argument("--splits_json", type=str,   default=SPLITS_JSON)
    parser.add_argument("--out_dir",     type=str,   default=OUT_DIR)
    args = parser.parse_args()

    run_set = {r.strip().upper() for r in args.run.split(",")}
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)

    # ── Reproducibility ──────────────────────────────────────────────────────
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  Ablation Study — UNet Burn Severity")
    print("=" * 60)
    print(f"  Device  : {device}"
          + (f" ({torch.cuda.get_device_name(0)})"
             if device.type == "cuda" else ""))
    print(f"  Run     : {sorted(run_set)}")
    print(f"  Epochs  : {args.epochs}  Batch: {args.batch}")

    # ── Load splits ──────────────────────────────────────────────────────────
    print("\nLoading splits ...")
    train_files, val_files, test_files = load_splits(args.patch_dir,
                                                      args.splits_json)
    print(f"  Train: {len(train_files)}  Val: {len(val_files)}  "
          f"Test: {len(test_files)}")
    if not train_files:
        print("ERROR: No patches found. Run sentinel2_dataset_v2.py first.")
        return

    # ── Channel stats (full 7-ch, from training split only) ──────────────────
    # Stats are computed on TRAINING data only to prevent leakage into val/test.
    # The same stats are used across all three ablations for fair comparison.
    print("\nComputing channel statistics (train only) ...")
    mean_7, std_7 = compute_channel_stats(train_files)
    print(f"  Mean (7-ch): {mean_7.round(4)}")

    # Ablation A uses only channels 0–4 (drops NBR ch5 and dNBR ch6).
    # Its normalisation stats are the matching first-5 entries of the full stats.
    mean_5 = mean_7[:REDUCED_CHANNELS]
    std_5  = std_7[:REDUCED_CHANNELS]

    # ── Class weights for loss functions ─────────────────────────────────────
    class_counts = np.zeros(N_CLASSES, dtype=np.int64)
    for fp in train_files[:200]:
        try:
            y_ = np.load(fp, allow_pickle=True)["y"].ravel()
            for c in range(N_CLASSES):
                class_counts[c] += np.sum(y_ == c)
        except Exception:
            pass
    class_freq    = class_counts / (class_counts.sum() + 1e-8)
    class_weights = np.clip(1.0 / (class_freq + 1e-4), 0.1, 50.0)
    class_weights /= class_weights.mean()
    print(f"  Class counts : {class_counts}")
    print(f"  Class weights: {class_weights.round(3)}")
    cw_list = class_weights.tolist()

    # =========================================================================
    # ABLATION CONFIGS
    # =========================================================================
    #
    # All configs share the same training loop, model size (F_BASE=32),
    # dropout (0.3), optimiser (AdamW lr=3e-4 wd=1e-4),
    # scheduler (ReduceLROnPlateau), early-stopping (patience=5), EMA (α=0.3).
    #
    # The ONLY thing that changes per ablation is highlighted below.
    # =========================================================================

    configs = {
        # ── Ablation A: remove spectral indices (NBR, dNBR) ──────────────────
        "A": dict(
            channels      = list(range(REDUCED_CHANNELS)),   # [0,1,2,3,4]
            augment_train = True,
            loss_fn       = FocalDiceLoss(n_cls=N_CLASSES, alpha=0.5,
                                          class_weights=cw_list),
            mean          = mean_5,
            std           = std_5,
        ),
        # ── Ablation B: disable augmentation ─────────────────────────────────
        "B": dict(
            channels      = list(range(ALL_CHANNELS)),        # [0..6]
            augment_train = False,                            # <-- changed
            loss_fn       = FocalDiceLoss(n_cls=N_CLASSES, alpha=0.5,
                                          class_weights=cw_list),
            mean          = mean_7,
            std           = std_7,
        ),
        # ── Ablation C: replace FocalDiceLoss with CrossEntropyLoss ──────────
        "C": dict(
            channels      = list(range(ALL_CHANNELS)),        # [0..6]
            augment_train = True,
            loss_fn       = WeightedCrossEntropyLoss(         # <-- changed
                                class_weights=cw_list),
            mean          = mean_7,
            std           = std_7,
        ),
    }

    # =========================================================================
    # RUN ABLATIONS
    # =========================================================================
    all_results = {}

    for name in ["A", "B", "C"]:
        if name not in run_set:
            print(f"\n  Skipping ablation {name} (not in --run)")
            continue

        # Skip if already completed (resume-safe after OOM/crash)
        json_path = os.path.join(args.out_dir, f"ablation_{name}_metrics.json")
        ckpt_path = os.path.join(args.out_dir, f"ablation_{name}_best.pth")
        if os.path.exists(json_path) and os.path.exists(ckpt_path):
            print(f"\n  ✓ Ablation {name} already complete — skipping.")
            with open(json_path) as f:
                all_results[name] = json.load(f)
            continue

        cfg = configs[name]
        result = run_ablation_training(
            ablation_name = name,
            train_files   = train_files,
            val_files     = val_files,
            test_files    = test_files,
            channels      = cfg["channels"],
            augment_train = cfg["augment_train"],
            loss_fn       = cfg["loss_fn"],
            mean          = cfg["mean"],
            std           = cfg["std"],
            device        = device,
            epochs        = args.epochs,
            batch         = args.batch,
            out_dir       = args.out_dir,
        )
        all_results[name] = result

        # Free dataset cache before loading next variant to avoid OOM
        gc.collect()
        torch.cuda.empty_cache()

        # Save per-ablation metrics JSON
        json_path = os.path.join(args.out_dir, f"ablation_{name}_metrics.json")
        # Remove history from individual file to keep it readable
        result_slim = {k: v for k, v in result.items() if k != "history"}
        with open(json_path, "w") as f:
            json.dump(result_slim, f, indent=2)
        print(f"  Saved -> {json_path}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    if all_results:
        summary = {}
        for name, res in all_results.items():
            summary[name] = {
                "description":    res["description"],
                "input_channels": res["input_channels"],
                "augment_train":  res["augment_train"],
                "loss_fn":        res["loss_fn"],
                "best_epoch":     res["best_epoch"],
                "val": {
                    "mean_iou":    res["val"]["mean_iou"],
                    "mean_dice":   res["val"]["mean_dice"],
                    "mean_pr_auc": res["val"]["mean_pr_auc"],
                    "iou":         res["val"]["iou"],
                },
                "test": {
                    "mean_iou":    res["test"]["mean_iou"],
                    "mean_dice":   res["test"]["mean_dice"],
                    "mean_pr_auc": res["test"]["mean_pr_auc"],
                    "iou":         res["test"]["iou"],
                },
            }

        summary_path = os.path.join(args.out_dir, "ablation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary saved -> {summary_path}")

        # Comparison chart
        plot_ablation_comparison(summary, args.out_dir)

        # ── Print summary table ───────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("  ABLATION SUMMARY")
        print("=" * 70)
        hdr = f"  {'Ablation':<10} {'Description':<36} {'Split':<5} " \
              f"{'mIoU':>6} {'mDice':>6} {'PR-AUC':>7}"
        print(hdr)
        print("  " + "-" * 66)
        for name, s in summary.items():
            short_desc = s["description"][:35]
            for split in ("val", "test"):
                m = s[split]
                print(f"  {name:<10} {short_desc:<36} {split:<5} "
                      f"{m['mean_iou']:>6.4f} "
                      f"{m['mean_dice']:>6.4f} "
                      f"{m['mean_pr_auc']:>7.4f}")
        print("=" * 70)

    print(f"\n  Done. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
