"""
cross_region_eval.py
====================
Cross-region generalisation evaluation for trained burn-severity models.

Loads a model trained on one geographic region (e.g. USA) and evaluates it
DIRECTLY on another region (e.g. Canada) with NO retraining or fine-tuning.
Measures how well learned spectral-spatial features transfer across continents.

Evaluation metrics (identical to main pipeline):
  - Per-class IoU, Dice, PR-AUC  (4 classes)
  - Macro mIoU, mDice, mean PR-AUC
  - Confusion matrix
  - Per-fire breakdown

Region detection is automatic: patches under patches_v2/USA/ are USA fires,
patches under patches_v2/CAN/ are Canada fires.

Outputs
-------
  metrics/cross_region_metrics.json
  metrics/cross_region_confusion_<src>_on_<tgt>.png
  metrics/cross_region_per_class_iou.png

Usage
-----
  # Evaluate USA-trained model on Canada patches:
  python cross_region_eval.py \
      --checkpoint outputs/sentinel2_results_v2/best_model.pth \
      --model_arch unet \
      --source_region USA \
      --target_region CAN

  # Evaluate every checkpoint found under a directory:
  python cross_region_eval.py \
      --checkpoint_dir outputs/sentinel2_results_v2 \
      --source_region USA \
      --target_region CAN

  # Both directions (USA→CAN and CAN→USA):
  python cross_region_eval.py \
      --checkpoint outputs/sentinel2_results_v2/best_model.pth \
      --both_directions

SLURM
-----
  sbatch cross_region_eval.slurm
"""

import os
import sys
import json
import time
import argparse
import gc
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score, confusion_matrix

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Import shared components (read-only, no side-effects)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet_training_v2 import (
    N_CLASSES, N_INPUT, PATCH_SZ, SEED, F_BASE, DROPOUT,
    DARK_BG, SEVERITY_NAMES,
    UNet, ResUNet, AttentionUNet, UNetPlusPlus,
    MODEL_REGISTRY,
    compute_channel_stats,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE        = os.environ.get("WILDFIRE_BASE_DIR",
              os.path.dirname(os.path.abspath(__file__)))
PATCH_DIR   = os.path.join(BASE, "data", "sentinel2", "patches_v2")
SPLITS_JSON = os.path.join(BASE, "data", "sentinel2", "splits_v2.json")
STATS_PATH  = os.path.join(BASE, "outputs", "sentinel2_results_v2",
                           "channel_stats.npz")
METRICS_DIR = os.path.join(BASE, "metrics")

REGION_DIRS = {
    "USA": "USA",
    "CAN": "CAN",
    "CANADA": "CAN",
}


# =============================================================================
# SECTION 1  PATCH DISCOVERY BY REGION
# =============================================================================

def get_region_files(patch_dir, region):
    """
    Return all .npz patch paths for a given region.

    Regions map to subdirectory names:
        USA  →  patches_v2/USA/
        CAN  →  patches_v2/CAN/
    """
    subdir = REGION_DIRS.get(region.upper())
    if subdir is None:
        raise ValueError(f"Unknown region '{region}'. "
                         f"Valid options: {list(REGION_DIRS.keys())}")

    region_path = Path(patch_dir) / subdir
    if not region_path.exists():
        raise FileNotFoundError(
            f"Region directory not found: {region_path}\n"
            f"Run sentinel2_dataset_v2.py first.")

    files = sorted(region_path.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No .npz patches found under {region_path}")
    return [str(f) for f in files]


def split_region_files_by_fire(files):
    """
    Group patch files by fire_id (filename stem without the last _NNNN index).
    Returns dict { fire_id: [file1, file2, ...] }
    Patch filename convention: {fire_id}_{NNNN}.npz
    The trailing numeric index is stripped; everything before it is the fire_id.
    """
    groups = {}
    for fp in files:
        stem    = Path(fp).stem
        parts   = stem.rsplit("_", 1)
        # Only strip the suffix if it looks like a zero-padded integer (e.g. 0042)
        fire_id = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
        groups.setdefault(fire_id, []).append(fp)
    return groups


# =============================================================================
# SECTION 2  DATASET  (no augmentation — evaluation only)
# =============================================================================

class RegionDataset(Dataset):
    """
    Minimal dataset for cross-region evaluation.
    Normalises with training-split stats; no augmentation.
    """

    def __init__(self, patch_files, mean, std, channels=None):
        """
        Parameters
        ----------
        patch_files : list[str]
        mean / std  : np.ndarray (N_INPUT,) — from source-region training split
        channels    : list[int] or None — if None, use all N_INPUT channels
        """
        self.files    = list(patch_files)
        self.channels = channels if channels is not None else list(range(N_INPUT))
        self.mean     = torch.from_numpy(
                            mean[self.channels].astype(np.float32))
        self.std      = torch.from_numpy(
                            std[self.channels].astype(np.float32))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        X    = torch.from_numpy(data["X"].astype(np.float32))[self.channels]
        y    = torch.from_numpy(data["y"].astype(np.int64)).clamp(0, N_CLASSES-1)
        X    = (X - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-8)
        return X, y


# =============================================================================
# SECTION 3  MODEL LOADING
# =============================================================================

def load_checkpoint(ckpt_path, model_arch, device, input_channels=N_INPUT):
    """
    Load a trained model from a .pth checkpoint.

    The checkpoint format (written by unet_training_v2.py / ablation.py):
        { "epoch": int, "model": state_dict, "miou": float, ... }

    Parameters
    ----------
    ckpt_path       : str   path to .pth file
    model_arch      : str   one of unet / resunet / attention / unetpp
    device          : torch.device
    input_channels  : int   number of input channels the model was trained with

    Returns
    -------
    model  (nn.Module, eval mode)
    meta   (dict with epoch, miou, etc.)
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # Support both formats: plain state_dict or wrapped dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        meta = {k: v for k, v in ckpt.items() if k != "model"}
    else:
        state_dict = ckpt
        meta = {}

    arch = model_arch.lower().strip()
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. "
                         f"Valid: {list(MODEL_REGISTRY.keys())}")

    ModelClass = MODEL_REGISTRY[arch]
    model = ModelClass(input_channels=input_channels,
                       num_classes=N_CLASSES,
                       f=F_BASE, drop=DROPOUT).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {arch.upper()} — {n_params:,} params "
          f"| epoch={meta.get('epoch','?')} "
          f"| val_mIoU={meta.get('miou', meta.get('ema_miou', '?'))}")
    return model, meta


def infer_input_channels(ckpt_path):
    """
    Infer the number of input channels from the checkpoint's first conv weight tensor.
    This avoids having to pass --input_channels explicitly when evaluating any checkpoint.
    Conv weight shape convention: (out_channels, in_channels, kH, kW)
    → shape[1] = in_channels = number of input bands the model was trained with.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd   = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt \
               else ckpt
        # Try known first-layer key names for each architecture:
        # UNet/ResUNet enc1 | UNet++ X00 | AttentionUNet enc1
        for key in ("enc1.0.weight", "X00.0.weight", "enc1.block.0.weight"):
            if key in sd:
                return int(sd[key].shape[1])
    except Exception:
        pass
    return None


# =============================================================================
# SECTION 4  EVALUATION METRICS
# =============================================================================

@torch.no_grad()
def evaluate_model(model, loader, device, n_cls=N_CLASSES):
    """
    Full pixel-level evaluation.
    Returns metrics dict with per-class and macro IoU/Dice/PR-AUC.
    Identical computation to compute_metrics() in unet_training_v2.
    """
    model.eval()
    all_preds, all_trues, all_probs = [], [], []

    for X, y in loader:
        X, y   = X.to(device), y.to(device)
        logits = model(X)
        probs  = F.softmax(logits, dim=1)
        preds  = logits.argmax(1)
        all_preds.append(preds.cpu().numpy().ravel())
        all_trues.append(y.cpu().numpy().ravel())
        all_probs.append(probs.cpu().numpy()
                         .transpose(0, 2, 3, 1)
                         .reshape(-1, n_cls))

    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)
    probs = np.concatenate(all_probs, axis=0)

    iou_list, dice_list, ap_list = [], [], []
    for c in range(n_cls):
        tp  = np.sum((preds == c) & (trues == c))
        fp  = np.sum((preds == c) & (trues != c))
        fn  = np.sum((preds != c) & (trues == c))
        iou  = float(tp / (tp + fp + fn + 1e-8))
        dice = float(2 * tp / (2 * tp + fp + fn + 1e-8))
        iou_list.append(iou)
        dice_list.append(dice)
        try:
            ap = float(average_precision_score(
                (trues == c).astype(int), probs[:, c]))
        except ValueError:
            ap = 0.0
        ap_list.append(ap)

    return {
        "iou":          iou_list,
        "dice":         dice_list,
        "pr_auc":       ap_list,
        "mean_iou":     float(np.mean(iou_list)),
        "mean_dice":    float(np.mean(dice_list)),
        "mean_pr_auc":  float(np.mean(ap_list)),
    }


@torch.no_grad()
def evaluate_per_fire(model, fire_groups, mean, std, channels,
                      device, batch=8, n_cls=N_CLASSES):
    """
    Evaluate model fire-by-fire and return a list of per-fire metric dicts.

    Parameters
    ----------
    fire_groups : dict { fire_id: [file_paths] }
    """
    model.eval()
    fire_results = []

    for fire_id, files in fire_groups.items():
        ds = RegionDataset(files, mean, std, channels)
        ld = DataLoader(ds, batch_size=batch, shuffle=False,
                        num_workers=0, pin_memory=False)
        metrics = evaluate_model(model, ld, device, n_cls)
        metrics["fire_id"]    = fire_id
        metrics["n_patches"]  = len(files)
        fire_results.append(metrics)
        del ds, ld
        gc.collect()

    fire_results.sort(key=lambda r: r["mean_iou"])
    return fire_results


# =============================================================================
# SECTION 5  VISUALISATION
# =============================================================================

def plot_confusion(cm, title, save_path):
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)
        cm_norm = cm.astype(float) / (cm.sum(1, keepdims=True) + 1e-8)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ticks = list(range(N_CLASSES))
        ax.set_xticks(ticks)
        ax.set_xticklabels(SEVERITY_NAMES, rotation=30, ha="right",
                           color="white", fontsize=8)
        ax.set_yticks(ticks)
        ax.set_yticklabels(SEVERITY_NAMES, color="white", fontsize=8)
        ax.set_xlabel("Predicted", color="white")
        ax.set_ylabel("True",      color="white")
        ax.tick_params(colors="white")
        ax.set_title(title, color="white", fontsize=9)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i,
                        f"{cm[i,j]:,}\n({cm_norm[i,j]:.2f})",
                        ha="center", va="center", fontsize=6,
                        color="white" if cm_norm[i, j] < 0.6 else "black")
        plt.tight_layout()
        fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"  Saved: {save_path}")
    except Exception as e:
        print(f"  Warning: could not save confusion: {e}")


def plot_per_class_iou(results_dict, save_path):
    """
    Bar chart: per-class IoU for each (source→target) evaluation.
    results_dict = { label_str: metrics_dict }
    """
    try:
        import matplotlib.pyplot as plt

        colors = ["#4fc3f7", "#ef5350", "#66bb6a",
                  "#ffa726", "#ab47bc", "#26c6da"]
        keys   = list(results_dict.keys())
        x      = np.arange(N_CLASSES)
        width  = 0.8 / max(len(keys), 1)

        fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

        for i, (label, metrics) in enumerate(results_dict.items()):
            offset = (i - len(keys) / 2 + 0.5) * width
            ax.bar(x + offset, metrics["iou"], width,
                   label=label, color=colors[i % len(colors)], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(SEVERITY_NAMES, color="white")
        ax.set_ylabel("IoU", color="white")
        ax.set_ylim(0, 1.05)
        ax.set_title("Cross-Region Evaluation — Per-Class IoU", color="white")
        ax.legend(facecolor="#16213e", labelcolor="white")
        plt.tight_layout()
        fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"  Saved: {save_path}")
    except Exception as e:
        print(f"  Warning: could not save per-class IoU: {e}")


def plot_per_fire_miou(fire_results, title, save_path):
    """Horizontal bar chart of per-fire mIoU sorted ascending."""
    try:
        import matplotlib.pyplot as plt

        fids  = [r["fire_id"] for r in fire_results]
        mious = [r["mean_iou"] for r in fire_results]

        fig_h = max(4, len(fids) * 0.35)
        fig, ax = plt.subplots(figsize=(9, fig_h), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

        colors = ["#ef5350" if v < 0.4 else
                  "#ffa726" if v < 0.65 else
                  "#66bb6a" for v in mious]
        ax.barh(range(len(fids)), mious, color=colors, alpha=0.85)
        ax.set_yticks(range(len(fids)))
        ax.set_yticklabels(fids, color="white", fontsize=7)
        ax.set_xlabel("mIoU", color="white")
        ax.set_xlim(0, 1)
        ax.set_title(title, color="white", fontsize=10)
        ax.axvline(np.mean(mious), color="white", linestyle="--",
                   linewidth=1, label=f"Mean={np.mean(mious):.3f}")
        ax.legend(facecolor="#16213e", labelcolor="white")
        plt.tight_layout()
        fig.savefig(save_path, dpi=110, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"  Saved: {save_path}")
    except Exception as e:
        print(f"  Warning: could not save per-fire chart: {e}")


# =============================================================================
# SECTION 6  SINGLE EVALUATION RUN
# =============================================================================

def run_cross_region(
    ckpt_path,
    model_arch,
    source_region,
    target_region,
    patch_dir,
    mean,
    std,
    channels,
    device,
    metrics_dir,
    batch=8,
):
    """
    Evaluate one checkpoint on the target region.

    Returns result dict.
    """
    src = source_region.upper()
    tgt = target_region.upper()
    tag = f"{src}_to_{tgt}_{Path(ckpt_path).stem}"

    print(f"\n{'─'*60}")
    print(f"  Cross-region: {src} → {tgt}")
    print(f"  Checkpoint  : {ckpt_path}")
    print(f"  Architecture: {model_arch.upper()}")
    print(f"{'─'*60}")

    # ── Load model ────────────────────────────────────────────────────────────
    n_ch = len(channels)
    model, ckpt_meta = load_checkpoint(ckpt_path, model_arch, device,
                                        input_channels=n_ch)

    # ── Load target-region patches ────────────────────────────────────────────
    tgt_files = get_region_files(patch_dir, tgt)
    print(f"  Target patches ({tgt}): {len(tgt_files)}")

    # ── Full evaluation ───────────────────────────────────────────────────────
    ds = RegionDataset(tgt_files, mean, std, channels)
    ld = DataLoader(ds, batch_size=batch, shuffle=False,
                    num_workers=0, pin_memory=False)

    t0      = time.time()
    metrics = evaluate_model(model, ld, device)
    elapsed = time.time() - t0

    print(f"\n  Results ({src} → {tgt}):")
    print(f"    mIoU    : {metrics['mean_iou']:.4f}")
    print(f"    mDice   : {metrics['mean_dice']:.4f}")
    print(f"    PR-AUC  : {metrics['mean_pr_auc']:.4f}")
    for c, name in enumerate(SEVERITY_NAMES):
        print(f"    {name:<15}  "
              f"IoU={metrics['iou'][c]:.4f}  "
              f"Dice={metrics['dice'][c]:.4f}  "
              f"AP={metrics['pr_auc'][c]:.4f}")

    # ── Per-fire evaluation ───────────────────────────────────────────────────
    fire_groups   = split_region_files_by_fire(tgt_files)
    print(f"\n  Per-fire evaluation ({len(fire_groups)} fires) ...")
    fire_results  = evaluate_per_fire(model, fire_groups, mean, std,
                                       channels, device, batch)

    # ── Confusion matrix (incremental — avoids accumulating all pixels in RAM) ──
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for X, y in ld:
            X     = X.to(device)
            preds = model(X).argmax(1).cpu().numpy().ravel()
            trues = y.numpy().ravel()
            cm   += confusion_matrix(trues, preds,
                                     labels=list(range(N_CLASSES)))

    # ── Save confusion matrix plot ────────────────────────────────────────────
    os.makedirs(metrics_dir, exist_ok=True)
    plot_confusion(
        cm,
        title=f"Cross-Region: {src} model → {tgt} data",
        save_path=os.path.join(metrics_dir,
                               f"cross_region_confusion_{src}_on_{tgt}.png"))

    # ── Save per-fire chart ───────────────────────────────────────────────────
    plot_per_fire_miou(
        fire_results,
        title=f"{src} model on {tgt} fires — per-fire mIoU",
        save_path=os.path.join(metrics_dir,
                               f"cross_region_per_fire_{src}_on_{tgt}.png"))

    result = {
        "source_region":  src,
        "target_region":  tgt,
        "checkpoint":     str(ckpt_path),
        "model_arch":     model_arch,
        "input_channels": n_ch,
        "n_target_patches": len(tgt_files),
        "n_target_fires":   len(fire_groups),
        "eval_time_s":      round(elapsed, 1),
        "ckpt_meta":        {k: (float(v) if isinstance(v, (int, float))
                                 else str(v))
                             for k, v in ckpt_meta.items()},
        "metrics":          metrics,
        "per_fire":         fire_results,
        "confusion_matrix": cm.tolist(),
    }
    return result, tag


# =============================================================================
# SECTION 7  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-region generalisation evaluation")

    # Checkpoint selection — exactly one of these is required
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument(
        "--checkpoint",     type=str,
        help="Path to a single .pth checkpoint")
    ckpt_group.add_argument(
        "--checkpoint_dir", type=str,
        help="Directory — evaluate ALL .pth files found inside")

    parser.add_argument("--model_arch",     type=str, default="unet",
                        help="unet|resunet|attention|unetpp  (default: unet)")
    parser.add_argument("--source_region",  type=str, default="USA",
                        help="Region the model was trained on  (default: USA)")
    parser.add_argument("--target_region",  type=str, default="CAN",
                        help="Region to evaluate on  (default: CAN)")
    parser.add_argument("--both_directions", action="store_true",
                        help="Also run target→source  (e.g. CAN→USA)")
    parser.add_argument("--patch_dir",  type=str, default=PATCH_DIR)
    parser.add_argument("--stats_path", type=str, default=STATS_PATH,
                        help="Path to channel_stats.npz from training split")
    parser.add_argument("--metrics_dir", type=str, default=METRICS_DIR)
    parser.add_argument("--batch",      type=int, default=8)
    parser.add_argument("--input_channels", type=int, default=None,
                        help="Override input channels (auto-detected if omitted)")
    args = parser.parse_args()

    os.makedirs(args.metrics_dir, exist_ok=True)

    # ── Reproducibility ──────────────────────────────────────────────────────
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  Cross-Region Generalisation Evaluation")
    print("=" * 60)
    print(f"  Device : {device}"
          + (f" ({torch.cuda.get_device_name(0)})"
             if device.type == "cuda" else ""))
    print(f"  Source : {args.source_region.upper()}")
    print(f"  Target : {args.target_region.upper()}")

    # ── Load channel stats ────────────────────────────────────────────────────
    if os.path.exists(args.stats_path):
        s    = np.load(args.stats_path)
        mean = s["mean"].astype(np.float32)
        std  = s["std"].astype(np.float32)
        print(f"  Stats  : loaded from {args.stats_path}")
    else:
        print(f"  Stats  : {args.stats_path} not found — "
              f"computing from source region patches ...")
        src_files = get_region_files(args.patch_dir, args.source_region)
        mean, std = compute_channel_stats(src_files)
        print(f"  Stats computed from {len(src_files)} source patches")

    # ── Determine input channels ──────────────────────────────────────────────
    if args.checkpoint:
        ckpts = [args.checkpoint]
    else:
        ckpts = sorted(Path(args.checkpoint_dir).rglob("*.pth"))
        ckpts = [str(p) for p in ckpts]
        print(f"  Found {len(ckpts)} checkpoint(s) in {args.checkpoint_dir}")

    if not ckpts:
        print("ERROR: No checkpoints found.")
        return

    # ── Collect evaluation pairs ──────────────────────────────────────────────
    eval_pairs = [(args.source_region, args.target_region)]
    if args.both_directions:
        eval_pairs.append((args.target_region, args.source_region))

    # ── Run evaluations ───────────────────────────────────────────────────────
    all_results  = {}
    iou_plot_data = {}

    for ckpt_path in ckpts:
        # Auto-detect input channels
        n_ch = args.input_channels or infer_input_channels(ckpt_path) or N_INPUT
        channels = list(range(n_ch))

        for src, tgt in eval_pairs:
            try:
                result, tag = run_cross_region(
                    ckpt_path    = ckpt_path,
                    model_arch   = args.model_arch,
                    source_region = src,
                    target_region = tgt,
                    patch_dir    = args.patch_dir,
                    mean         = mean[:n_ch],
                    std          = std[:n_ch],
                    channels     = channels,
                    device       = device,
                    metrics_dir  = args.metrics_dir,
                    batch        = args.batch,
                )
                all_results[tag] = result
                iou_plot_data[f"{src}→{tgt}"] = result["metrics"]
                gc.collect()
                torch.cuda.empty_cache()
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
            except Exception as e:
                import traceback
                print(f"  ERROR evaluating {ckpt_path} ({src}→{tgt}): {e}")
                traceback.print_exc()

    if not all_results:
        print("\nNo evaluations completed.")
        return

    # ── Per-class IoU comparison chart ────────────────────────────────────────
    if iou_plot_data:
        plot_per_class_iou(
            iou_plot_data,
            os.path.join(args.metrics_dir, "cross_region_per_class_iou.png"))

    # ── Save full results JSON ────────────────────────────────────────────────
    out_path = os.path.join(args.metrics_dir, "cross_region_metrics.json")

    # Remove per-fire list from JSON (large) — save separately
    json_results = {}
    for tag, res in all_results.items():
        slim = {k: v for k, v in res.items()
                if k not in ("per_fire", "confusion_matrix")}
        slim["per_fire_summary"] = {
            "n_fires":    len(res["per_fire"]),
            "mean_miou":  float(np.mean([f["mean_iou"]  for f in res["per_fire"]])),
            "std_miou":   float(np.std( [f["mean_iou"]  for f in res["per_fire"]])),
            "min_miou":   float(min(    f["mean_iou"]   for f in res["per_fire"])),
            "max_miou":   float(max(    f["mean_iou"]   for f in res["per_fire"])),
            "fires":      [{"fire_id":   f["fire_id"],
                            "n_patches": f["n_patches"],
                            "mean_iou":  f["mean_iou"],
                            "mean_dice": f["mean_dice"]}
                           for f in res["per_fire"]],
        }
        json_results[tag] = slim

    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved -> {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CROSS-REGION GENERALISATION SUMMARY")
    print("=" * 70)
    print(f"  {'Direction':<12} {'Checkpoint':<30} "
          f"{'mIoU':>6} {'mDice':>6} {'PR-AUC':>7} {'Fires':>6}")
    print("  " + "-" * 66)
    for tag, res in all_results.items():
        direction  = f"{res['source_region']}→{res['target_region']}"
        ckpt_short = Path(res["checkpoint"]).name[:29]
        m          = res["metrics"]
        n_fires    = res["n_target_fires"]
        print(f"  {direction:<12} {ckpt_short:<30} "
              f"{m['mean_iou']:>6.4f} "
              f"{m['mean_dice']:>6.4f} "
              f"{m['mean_pr_auc']:>7.4f} "
              f"{n_fires:>6}")
    print("=" * 70)
    print(f"\n  Outputs: {args.metrics_dir}")


if __name__ == "__main__":
    main()
