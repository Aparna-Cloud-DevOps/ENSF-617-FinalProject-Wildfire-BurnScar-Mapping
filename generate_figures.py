"""
generate_figures.py
===================
Standalone script that loads all saved .pth checkpoints and generates:

  1. all_models_comparison.png  — single report-ready grid:
       Rows    = test samples (2)
       Columns = Satellite RGB | Ground Truth | UNet | ResUNet | AttUNet | UNet++

  2. {model}_severity_overview.png (×4) — per-model detail:
       Row 0: Satellite RGB + dNBR
       Row 1: Ground Truth + Prediction
       Row 2: Error map + Confidence

  3. patch_grid.png — 4×8 GT label grid (no model)

Reads
-----
  outputs/sentinel2_results_v2/best_model.pth
  outputs/sentinel2_results_v2/resunet_best_model.pth
  outputs/sentinel2_results_v2/attention_best_model.pth
  outputs/sentinel2_results_v2/unetpp_best_model.pth
  outputs/sentinel2_results_v2/channel_stats.npz
  data/sentinel2/splits_v2.json

Writes
------
  outputs/sentinel2_results_v2/figures/all_models_comparison.png
  outputs/sentinel2_results_v2/figures/{model}_severity_overview.png  (×4)
  outputs/sentinel2_results_v2/figures/patch_grid.png

Usage
-----
  python generate_figures.py
"""

import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet_training_v2 import (
    MODEL_REGISTRY, N_INPUT, N_CLASSES, F_BASE, DROPOUT, SEED,
    plot_severity_overview, plot_patch_grid, _cmap_severity, load_splits,
)

# ─────────────────────────────────────────────────────────────────────────────
BASE        = os.environ.get("WILDFIRE_BASE_DIR",
              os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "outputs", "sentinel2_results_v2")
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
SPLITS_JSON = os.path.join(BASE, "data", "sentinel2", "splits_v2.json")
PATCH_DIR   = os.path.join(BASE, "data", "sentinel2", "patches_v2")
STATS_NPZ   = os.path.join(RESULTS_DIR, "channel_stats.npz")
DARK_BG     = "#1a1a2e"

CKPT_MAP = {
    "unet":      "best_model.pth",
    "resunet":   "resunet_best_model.pth",
    "attention": "attention_best_model.pth",
    "unetpp":    "unetpp_best_model.pth",
}
MODEL_LABELS = {
    "unet": "UNet", "resunet": "ResUNet",
    "attention": "AttentionUNet", "unetpp": "UNet++",
}
SEVERITY_NAMES  = ["Unburned/Low", "Moderate", "High", "Very High"]
SEVERITY_COLORS = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]

os.makedirs(FIG_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

stats = np.load(STATS_NPZ)
mean  = stats["mean"].astype(np.float32)
std   = stats["std"].astype(np.float32)

_, _, test_files = load_splits(PATCH_DIR, SPLITS_JSON)
print(f"Test patches: {len(test_files)}")


# ─────────────────────────────────────────────────────────────────────────────
def _load_model(mname, ckpt_path):
    ModelClass = MODEL_REGISTRY[mname]
    m = ModelClass(input_channels=N_INPUT, num_classes=N_CLASSES,
                   f=F_BASE, drop=DROPOUT).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m


def _predict(model, X_raw):
    """Run one patch through the model; return hard argmax class map (H,W)."""
    X_t = torch.from_numpy(X_raw).unsqueeze(0).to(device)  # add batch dim
    with torch.no_grad():
        probs = F.softmax(model(X_t), dim=1).squeeze(0).cpu().numpy()
    return probs.argmax(axis=0)   # (H, W) integer class map


def _to_rgb(X_raw, mean, std):
    """Denormalize channels 0-2 (Red, Green, Blue) and stretch to [0,1] for display.
    X_raw is normalised (mean=0, std=1); we reverse the normalisation then
    apply min-max stretching so the image looks visually natural.
    """
    rgb = X_raw[:3].copy()
    for c in range(3):
        rgb[c] = rgb[c] * std[c] + mean[c]   # reverse z-score normalisation
    rmin, rmax = rgb.min(), rgb.max()
    # Percentile stretch would be more accurate, but simple min-max works for display
    return np.clip((rgb - rmin) / (rmax - rmin + 1e-8), 0, 1).transpose(1, 2, 0)


# ── 1. all_models_comparison.png ─────────────────────────────────────────────
def plot_all_models_comparison(test_files, mean, std, n_samples=2):
    """
    Single report-ready grid:
      Rows    = test patches (n_samples)
      Columns = Satellite RGB | Ground Truth | UNet | ResUNet | AttUNet | UNet++
    Only includes models whose checkpoints exist.
    """
    available = {k: v for k, v in CKPT_MAP.items()
                 if os.path.exists(os.path.join(RESULTS_DIR, v))}
    if not available:
        print("  No checkpoints found — skipping comparison grid.")
        return

    # Load all available models
    models = {}
    for mname, ckpt_fname in available.items():
        print(f"  Loading {mname} ...")
        models[mname] = _load_model(mname, os.path.join(RESULTS_DIR, ckpt_fname))

    cmap_sev = _cmap_severity()
    norm_sev = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_sev.N)

    rng  = np.random.default_rng(SEED + 99)
    sels = rng.choice(test_files, min(n_samples, len(test_files)), replace=False)

    col_headers = ["Satellite RGB", "Ground Truth"] + \
                  [MODEL_LABELS[m] for m in models]
    n_cols = len(col_headers)
    n_rows = n_samples

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.2, n_rows * 3.4),
                             facecolor=DARK_BG)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for ri, fp in enumerate(sels):
        data   = np.load(fp, allow_pickle=True)
        X_raw  = data["X"].astype(np.float32)
        y_true = data["y"].astype(np.int64)
        rgb    = _to_rgb(X_raw, mean, std)

        try:
            fire_id = str(data["fire_id"])
        except Exception:
            fire_id = os.path.basename(fp)

        col = 0
        # Satellite RGB
        axes[ri, col].imshow(rgb)
        axes[ri, col].axis("off")
        if ri == 0:
            axes[ri, col].set_title(col_headers[col], color="white",
                                    fontsize=10, fontweight="bold")
        axes[ri, col].set_ylabel(fire_id[:20], color="white", fontsize=7,
                                 rotation=0, labelpad=60, va="center")
        col += 1

        # Ground truth
        axes[ri, col].imshow(y_true, cmap=cmap_sev, norm=norm_sev,
                             interpolation="nearest")
        axes[ri, col].axis("off")
        if ri == 0:
            axes[ri, col].set_title(col_headers[col], color="white",
                                    fontsize=10, fontweight="bold")
        col += 1

        # Each model prediction
        for mname in models:
            pred = _predict(models[mname], X_raw)
            axes[ri, col].imshow(pred, cmap=cmap_sev, norm=norm_sev,
                                 interpolation="nearest")
            axes[ri, col].axis("off")
            if ri == 0:
                axes[ri, col].set_title(MODEL_LABELS[mname], color="white",
                                        fontsize=10, fontweight="bold")
            col += 1

    # Severity legend
    handles = [mpatches.Patch(color=SEVERITY_COLORS[i], label=SEVERITY_NAMES[i])
               for i in range(N_CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               facecolor="#16213e", labelcolor="white", fontsize=10,
               bbox_to_anchor=(0.5, -0.03))

    plt.suptitle("Burn Severity Segmentation — All Models Comparison",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(pad=0.4)

    out = os.path.join(FIG_DIR, "all_models_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {out}")

    for m in models.values():
        del m
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
print("\n── 1. All-models comparison grid ───────────────────────────────────")
plot_all_models_comparison(test_files, mean, std, n_samples=2)

print("\n── 2. Per-model severity overview ──────────────────────────────────")
for mname, ckpt_fname in CKPT_MAP.items():
    ckpt_path = os.path.join(RESULTS_DIR, ckpt_fname)
    if not os.path.exists(ckpt_path):
        print(f"  [{mname.upper()}] checkpoint not found — skipping")
        continue
    print(f"  Generating {mname}_severity_overview.png ...")
    try:
        model = _load_model(mname, ckpt_path)
        out   = os.path.join(FIG_DIR, f"{mname}_severity_overview.png")
        plot_severity_overview(test_files, model, device, mean, std, out,
                               model_name=MODEL_LABELS[mname])
        print(f"  Saved: {out}")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR for {mname}: {e}")

print("\n── 3. Incorrect mapping (misclassification analysis) ───────────────")

def plot_incorrect_mapping(test_files, mean, std, n_samples=3):
    """
    For each available model, shows where and HOW the model is wrong.

    Layout per model (n_samples columns):
      Row 0: Satellite RGB
      Row 1: Ground Truth
      Row 2: Misclassification map
               - correct pixels  → grey
               - wrong pixels    → color of the INCORRECTLY predicted class
                 so you can see "predicted High when it was Moderate" etc.
      Row 3: Error-type overlay
               - each wrong pixel colored by true_class→pred_class pair
    """
    available = {k: v for k, v in CKPT_MAP.items()
                 if os.path.exists(os.path.join(RESULTS_DIR, v))}
    if not available:
        print("  No checkpoints found — skipping.")
        return

    cmap_sev = _cmap_severity()
    norm_sev = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_sev.N)

    rng  = np.random.default_rng(SEED + 77)
    sels = rng.choice(test_files, min(n_samples, len(test_files)), replace=False)

    for mname, ckpt_fname in available.items():
        print(f"  Generating {mname}_incorrect_mapping.png ...")
        model = _load_model(mname, os.path.join(RESULTS_DIR, ckpt_fname))

        fig, axes = plt.subplots(4, n_samples,
                                 figsize=(n_samples * 3.5, 14),
                                 facecolor=DARK_BG)
        if n_samples == 1:
            axes = axes[:, np.newaxis]

        row_titles = ["Satellite RGB", "Ground Truth",
                      "Misclassification Map", "Error Type"]

        for ci, fp in enumerate(sels):
            data   = np.load(fp, allow_pickle=True)
            X_raw  = data["X"].astype(np.float32)
            y_true = data["y"].astype(np.int64)
            rgb    = _to_rgb(X_raw, mean, std)

            X_t = torch.from_numpy(X_raw).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = F.softmax(model(X_t), dim=1).squeeze(0).cpu().numpy()
            pred = probs.argmax(axis=0)

            # Misclassification map: correct pixels = grey, wrong pixels = color
            # of the PREDICTED class — so "red" means the model predicted High.
            mis_rgb = np.full((*pred.shape, 3), 0.3)  # initialize all pixels to grey
            wrong_mask = (pred != y_true)
            for cls in range(N_CLASSES):
                color = mcolors.to_rgb(SEVERITY_COLORS[cls])
                mask  = wrong_mask & (pred == cls)   # wrong pixels where pred==cls
                mis_rgb[mask] = color

            # Error-type map: assigns a unique integer label to each (true→pred) class pair.
            # 4 classes → 4×3 = 12 possible wrong-direction pairs (excluding diagonal).
            # Each pair gets a distinct color from the tab20 colormap for visual separation.
            error_type = np.zeros(pred.shape, dtype=np.int32)
            pair_colors = {}
            pair_idx = 1
            for tc in range(N_CLASSES):
                for pc in range(N_CLASSES):
                    if tc != pc:   # skip correct (diagonal) predictions
                        pair_colors[pair_idx] = (tc, pc)
                        error_type[(y_true == tc) & (pred == pc)] = pair_idx
                        pair_idx += 1
            n_pairs   = pair_idx - 1
            cmap_err  = plt.cm.get_cmap("tab20", n_pairs + 1)
            norm_err  = mcolors.BoundaryNorm(range(n_pairs + 2), n_pairs + 1)

            # Row 0: Satellite RGB
            axes[0, ci].imshow(rgb)
            axes[0, ci].axis("off")
            try:
                axes[0, ci].set_title(str(data["fire_id"])[:18],
                                      color="white", fontsize=8)
            except Exception:
                axes[0, ci].set_title(os.path.basename(fp)[:18],
                                      color="white", fontsize=8)

            # Row 1: Ground truth
            axes[1, ci].imshow(y_true, cmap=cmap_sev, norm=norm_sev,
                               interpolation="nearest")
            axes[1, ci].axis("off")

            # Row 2: Misclassification map
            axes[2, ci].imshow(mis_rgb)
            pct_wrong = wrong_mask.mean() * 100
            axes[2, ci].set_title(f"Error: {pct_wrong:.1f}%",
                                  color="white", fontsize=8)
            axes[2, ci].axis("off")

            # Row 3: Error-type map
            axes[3, ci].imshow(error_type, cmap=cmap_err, norm=norm_err,
                               interpolation="nearest")
            axes[3, ci].axis("off")

        # Row labels
        for ri, rtitle in enumerate(row_titles):
            axes[ri, 0].set_ylabel(rtitle, color="white", fontsize=9,
                                   rotation=90, labelpad=8)

        # Severity legend + correct-pixel note
        handles = [mpatches.Patch(color=SEVERITY_COLORS[i],
                                  label=f"Predicted {SEVERITY_NAMES[i]} (wrong)")
                   for i in range(N_CLASSES)]
        handles.insert(0, mpatches.Patch(color="#4d4d4d", label="Correct"))
        fig.legend(handles=handles, loc="lower center", ncol=3,
                   facecolor="#16213e", labelcolor="white", fontsize=9,
                   bbox_to_anchor=(0.5, -0.03))

        plt.suptitle(f"Misclassification Analysis — {MODEL_LABELS[mname]}",
                     color="white", fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout(pad=0.4)

        out = os.path.join(FIG_DIR, f"{mname}_incorrect_mapping.png")
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"  Saved: {out}")

        del model
        torch.cuda.empty_cache()

plot_incorrect_mapping(test_files, mean, std, n_samples=3)

print("\n── 4. Patch grid (GT labels) ────────────────────────────────────────")
out = os.path.join(FIG_DIR, "patch_grid.png")
plot_patch_grid(test_files, out)
print(f"  Saved: {out}")

print("\nDone.")
