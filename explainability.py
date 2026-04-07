"""
explainability.py
=================
Grad-CAM explainability and attention-map visualisation for wildfire burn-severity
segmentation models (UNet, ResUNet, AttentionUNet, UNet++).

For each model this script:
  1. Loads the trained checkpoint.
  2. Selects 5–10 representative samples from the test split.
  3. Computes Grad-CAM w.r.t. the deepest encoder/bottleneck block, using the
     predicted class at every pixel as the target signal (segmentation Grad-CAM).
  4. For AttentionUNet: extracts and visualises the four attention-gate weight maps.
  5. Saves five images per sample:
       sample_{id}_input.png
       sample_{id}_gt.png
       sample_{id}_pred.png
       sample_{id}_gradcam.png
       sample_{id}_overlay.png
     and, for AttentionUNet only:
       sample_{id}_attention.png

Outputs
-------
  explainability_samples/{arch}/sample_{id}_*.png
  metrics/explainability_metrics.json

Usage
-----
  python explainability.py
  python explainability.py --n_samples 8 --archs unet,attention
  sbatch explainability.slurm
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# Import model registry and helpers from the training script (no modification).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet_training_v2 import (
    N_INPUT, N_CLASSES, PATCH_SZ, F_BASE, DROPOUT, SEED,
    DARK_BG, SEVERITY_NAMES, SEVERITY_COLORS, MODEL_REGISTRY,
    load_splits, compute_channel_stats, AttentionUNet,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths (overrideable via CLI)
# ─────────────────────────────────────────────────────────────────────────────
BASE        = os.environ.get("WILDFIRE_BASE_DIR",
              os.path.dirname(os.path.abspath(__file__)))
PATCH_DIR   = os.path.join(BASE, "data", "sentinel2", "patches_v2")
SPLITS_JSON = os.path.join(BASE, "data", "sentinel2", "splits_v2.json")
RESULTS_DIR = os.path.join(BASE, "outputs", "sentinel2_results_v2")
METRICS_DIR = os.path.join(BASE, "metrics")
EXPL_DIR    = os.path.join(BASE, "explainability_samples")

# Checkpoint filenames per architecture
ARCH_CKPT = {
    "unet":      "best_model.pth",
    "resunet":   "resunet_best_model.pth",
    "attention": "attention_best_model.pth",
    "unetpp":    "unetpp_best_model.pth",
}

# Sentinel-2 channel indices used for the RGB composite (R, G, B)
RGB_CH = [0, 1, 2]

# Colourmap for the 4 severity classes
SCLASS_CMAP = mcolors.ListedColormap(SEVERITY_COLORS)
SCLASS_NORM = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)


# =============================================================================
# DATASET  (minimal — no augmentation, single-sample batches for Grad-CAM)
# =============================================================================

class _PatchDataset(Dataset):
    """Load .npz patches, normalise with precomputed channel stats."""

    def __init__(self, files: list, mean: np.ndarray, std: np.ndarray):
        self.files = files
        self.mean  = torch.from_numpy(mean.astype(np.float32))
        self.std   = torch.from_numpy(std.astype(np.float32))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=True)
        X = torch.from_numpy(d["X"].astype(np.float32))               # (7,H,W)
        y = torch.from_numpy(d["y"].astype(np.int64)).clamp(0, N_CLASSES - 1)
        X = (X - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-8)
        return X, y, str(self.files[idx])


# =============================================================================
# SECTION 1 — load_model()
# =============================================================================

def load_model(arch: str, checkpoint_dir: str, device: torch.device):
    """
    Instantiate the requested U-Net variant and load its trained weights.

    Parameters
    ----------
    arch           : one of {'unet', 'resunet', 'attention', 'unetpp'}
    checkpoint_dir : directory that contains the .pth checkpoint files
    device         : torch.device

    Returns
    -------
    model  : nn.Module in eval() mode, weights loaded
    loaded : bool  — False when checkpoint file is missing (random weights)
    """
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown arch '{arch}'. "
                         f"Choose from {list(MODEL_REGISTRY.keys())}")

    model = MODEL_REGISTRY[arch](
        input_channels=N_INPUT,
        num_classes=N_CLASSES,
        f=F_BASE,
        drop=DROPOUT,
    ).to(device)

    ckpt_name = ARCH_CKPT.get(arch, f"{arch}_best_model.pth")
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)

    loaded = False
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt \
                else ckpt
        model.load_state_dict(state, strict=True)
        loaded = True
        print(f"    Loaded checkpoint: {ckpt_path}")
    else:
        print(f"    WARNING: checkpoint not found ({ckpt_path}). "
              "Using random weights — results are illustrative only.")

    model.eval()
    return model, loaded


# =============================================================================
# SECTION 2 — compute_gradcam()
# =============================================================================

class _GradCAMHook:
    """
    Forward + backward hooks attached to a single nn.Module.

    After one forward+backward pass the captured attributes are:
      .features   — (1, C, h, w)  feature maps from the target layer
      .gradients  — (1, C, h, w)  gradients of the loss w.r.t. those features
    """

    def __init__(self, module: nn.Module):
        self._features  = None
        self._gradients = None
        self._fwd = module.register_forward_hook(self._save_features)
        self._bwd = module.register_full_backward_hook(self._save_gradients)

    def _save_features(self, module, inp, out):
        self._features = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()


def _get_bottleneck(model: nn.Module) -> nn.Module:
    """
    Return the deepest encoder / bottleneck sub-module for any U-Net variant.

    Architecture  →  bottleneck attribute
    ─────────────────────────────────────
    UNet          →  model.bot      (_double_conv)
    ResUNet       →  model.bot      (_ResBlock)
    AttentionUNet →  model.bot      (_double_conv)
    UNetPlusPlus  →  model.X40      (_double_conv)  ← different name
    """
    # UNetPlusPlus names its bottleneck X40
    if hasattr(model, "X40"):
        return model.X40
    # UNet, ResUNet, AttentionUNet all use .bot
    if hasattr(model, "bot"):
        return model.bot
    raise AttributeError(
        f"Cannot locate bottleneck layer in {type(model).__name__}. "
        "Expected attribute 'bot' or 'X40'."
    )


def compute_gradcam(
    model: nn.Module,
    X: torch.Tensor,
    device: torch.device,
) -> tuple:
    """
    Compute a Grad-CAM heatmap for a segmentation model.

    The gradient signal is the sum of per-pixel logit values for each pixel's
    *predicted* class — this produces a single combined heatmap that highlights
    regions that most influenced the overall prediction map.

    The target layer is the deepest encoder / bottleneck block, resolved via
    _get_bottleneck() to handle all four U-Net variants correctly:
      UNet / ResUNet / AttentionUNet → model.bot
      UNetPlusPlus                   → model.X40

    Parameters
    ----------
    model  : nn.Module  (eval mode, on device)
    X      : (1, C, H, W) normalised input tensor  (on CPU)
    device : torch.device

    Returns
    -------
    cam      : (H, W) float32 ndarray in [0, 1]
    pred_map : (H, W) int64 ndarray — argmax predictions
    cam_time_ms : float — Grad-CAM wall-clock time (ms), GPU-synchronised
    """
    hook = _GradCAMHook(_get_bottleneck(model))

    x_in  = X.to(device)
    use_cuda = device.type == "cuda"

    # ── Timed Grad-CAM computation ────────────────────────────────────────────
    if use_cuda:
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    model.zero_grad()
    logits = model(x_in)                          # (1, 4, H, W)

    # predicted class per pixel
    pred_map = logits.detach().argmax(dim=1)       # (1, H, W)

    # Gather the logit of the predicted class for every pixel,
    # then sum → scalar target for backward.
    pred_idx = pred_map.unsqueeze(1)               # (1, 1, H, W)
    score    = logits.gather(1, pred_idx).sum()    # scalar

    score.backward()

    if use_cuda:
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    cam_time_ms = (t1 - t0) * 1000.0

    # ── Grad-CAM pooling (Selvaraju et al., 2017) ─────────────────────────────
    grads  = hook._gradients     # (1, C_bot, h, w) — gradient of score w.r.t. features
    feats  = hook._features      # (1, C_bot, h, w) — forward activation maps
    hook.remove()                # detach hooks so they don't persist across calls

    # Step 1: Global-average-pool gradients across spatial dims → channel importance
    # Each weight α_k^c = (1/Z) Σ_{i,j} (∂y^c / ∂A^k_{ij})
    weights = grads.mean(dim=[2, 3])               # (1, C_bot) — one weight per channel

    # Step 2: Weighted combination of feature maps + ReLU
    # L^c_Grad-CAM = ReLU( Σ_k α_k^c · A^k )
    # ReLU keeps only features that positively influence the prediction.
    cam_raw = (weights[:, :, None, None] * feats).sum(dim=1)  # (1, h, w)
    cam_raw = F.relu(cam_raw)

    # Step 3: Bilinear upsample from bottleneck resolution (h,w) to input (H,W)
    cam_up = F.interpolate(
        cam_raw.unsqueeze(1),
        size=(X.shape[2], X.shape[3]),
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()                      # (H, W)

    # Step 4: Min-max normalise to [0, 1] for display (avoids negative values)
    lo, hi = cam_up.min(), cam_up.max()
    if hi > lo:
        cam_up = (cam_up - lo) / (hi - lo)
    else:
        cam_up = np.zeros_like(cam_up)   # flat cam (no gradient signal) → black

    return (
        cam_up.astype(np.float32),
        pred_map.squeeze().cpu().numpy(),
        cam_time_ms,
    )


# =============================================================================
# SECTION 3 — compute_attention_maps()
# =============================================================================

def compute_attention_maps(
    model: AttentionUNet,
    X: torch.Tensor,
    device: torch.device,
) -> list:
    """
    Extract spatial attention weights from all four _AttentionGate modules
    inside an AttentionUNet.

    Each gate's ``psi`` sub-module outputs a (1, 1, H_g, W_g) sigmoid tensor.
    We upsample every gate map to the input resolution for consistent display.

    Parameters
    ----------
    model  : AttentionUNet in eval() mode
    X      : (1, C, H, W) normalised input tensor (on CPU)
    device : torch.device

    Returns
    -------
    maps : list of (gate_name: str, attn_map: ndarray (H, W) float32 [0,1])
           ordered from deepest gate (up4) to shallowest (up1)
    """
    saved = {}

    hooks = []
    # Register forward hooks on the psi (sigmoid output) of each attention gate.
    # _AttentionGate.psi outputs a (1,1,H_g,W_g) spatial attention weight map.
    # We traverse: model.up{4,3,2,1} → .attn (_AttentionGate) → .psi (nn.Sequential)
    for gate_name in ["up4", "up3", "up2", "up1"]:
        up_block = getattr(model, gate_name, None)
        if up_block is None:
            continue
        attn_gate = getattr(up_block, "attn", None)
        if attn_gate is None:
            continue
        psi = getattr(attn_gate, "psi", None)
        if psi is None:
            continue

        def _make_hook(name):
            # Use a closure to capture gate_name per iteration (avoids late-binding bug)
            def _h(module, inp, out):
                saved[name] = out.detach().squeeze().cpu().numpy()
            return _h

        hooks.append(psi.register_forward_hook(_make_hook(gate_name)))

    with torch.no_grad():
        _ = model(X.to(device))

    for h in hooks:
        h.remove()

    # upsample each gate map to input H×W
    H, W  = X.shape[2], X.shape[3]
    maps  = []
    for gname in ["up4", "up3", "up2", "up1"]:
        if gname not in saved:
            continue
        amap = saved[gname]
        # may be 2-D (h,w) or 3-D (1,h,w) depending on squeeze
        if amap.ndim == 3:
            amap = amap[0]
        # upsample with torch
        amap_t = torch.from_numpy(amap).unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
        amap_up = F.interpolate(amap_t, size=(H, W),
                                mode="bilinear",
                                align_corners=False).squeeze().numpy()
        maps.append((gname, amap_up.astype(np.float32)))

    return maps


# =============================================================================
# SECTION 4 — visualize_and_save()
# =============================================================================

def _make_rgb(X_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Convert a normalised (7, H, W) patch back to a display-ready (H, W, 3)
    uint8 RGB image via percentile stretch.
    """
    H, W = X_norm.shape[1], X_norm.shape[2]
    rgb  = np.zeros((H, W, 3), dtype=np.float32)
    for i, ch in enumerate(RGB_CH):
        rgb[:, :, i] = X_norm[ch] * std[ch] + mean[ch]
    lo  = np.percentile(rgb, 2)
    hi  = np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def visualize_and_save(
    sample_id: str,
    X_norm: np.ndarray,
    y_true: np.ndarray,
    pred_map: np.ndarray,
    cam: np.ndarray,
    out_dir: str,
    mean: np.ndarray,
    std: np.ndarray,
    attn_maps: list = None,
) -> dict:
    """
    Save all visualisation outputs for one sample.

    Files written
    -------------
    sample_{id}_input.png      — RGB false-colour composite
    sample_{id}_gt.png         — ground truth severity mask
    sample_{id}_pred.png       — model prediction mask
    sample_{id}_gradcam.png    — standalone Grad-CAM heatmap
    sample_{id}_overlay.png    — Grad-CAM overlaid on RGB
    sample_{id}_attention.png  — attention gate maps (AttentionUNet only)

    Returns
    -------
    dict mapping output key → saved file path
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = {}

    rgb = _make_rgb(X_norm, mean, std)

    legend_patches = [
        Patch(color=SEVERITY_COLORS[c], label=SEVERITY_NAMES[c])
        for c in range(N_CLASSES)
    ]

    # ── (1) Input RGB ─────────────────────────────────────────────────────────
    path = os.path.join(out_dir, f"sample_{sample_id}_input.png")
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=DARK_BG)
    ax.imshow(rgb)
    ax.set_title("RGB Composite", color="white", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    saved["input"] = path

    # ── (2) Ground truth ──────────────────────────────────────────────────────
    path = os.path.join(out_dir, f"sample_{sample_id}_gt.png")
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=DARK_BG)
    ax.imshow(y_true, cmap=SCLASS_CMAP, norm=SCLASS_NORM, interpolation="nearest")
    ax.set_title("Ground Truth", color="white", fontsize=11)
    ax.axis("off")
    ax.legend(handles=legend_patches, loc="lower right",
              facecolor=DARK_BG, labelcolor="white", fontsize=7,
              framealpha=0.6)
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    saved["gt"] = path

    # ── (3) Prediction ────────────────────────────────────────────────────────
    path = os.path.join(out_dir, f"sample_{sample_id}_pred.png")
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=DARK_BG)
    ax.imshow(pred_map, cmap=SCLASS_CMAP, norm=SCLASS_NORM,
              interpolation="nearest")
    ax.set_title("Prediction", color="white", fontsize=11)
    ax.axis("off")
    ax.legend(handles=legend_patches, loc="lower right",
              facecolor=DARK_BG, labelcolor="white", fontsize=7,
              framealpha=0.6)
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    saved["pred"] = path

    # ── (4) Standalone Grad-CAM heatmap ───────────────────────────────────────
    # jet colormap: blue=low activation, red=high activation region
    path = os.path.join(out_dir, f"sample_{sample_id}_gradcam.png")
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=DARK_BG)
    im = ax.imshow(cam, cmap="jet", vmin=0, vmax=1)
    ax.set_title("Grad-CAM Heatmap", color="white", fontsize=11)
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
    cb.outline.set_edgecolor("white")
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    saved["gradcam"] = path

    # ── (5) Grad-CAM overlay on RGB ───────────────────────────────────────────
    # alpha=0.50 blends the heatmap equally with the RGB image so the spatial
    # context (vegetation, roads) remains visible beneath the activation pattern.
    path = os.path.join(out_dir, f"sample_{sample_id}_overlay.png")
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=DARK_BG)
    ax.imshow(rgb)
    ov = ax.imshow(cam, cmap="jet", alpha=0.50, vmin=0, vmax=1)
    ax.set_title("Grad-CAM Overlay", color="white", fontsize=11)
    ax.axis("off")
    cb2 = fig.colorbar(ov, ax=ax, fraction=0.046, pad=0.04)
    cb2.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
    cb2.outline.set_edgecolor("white")
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    saved["overlay"] = path

    # ── (6) Attention maps (AttentionUNet only) ───────────────────────────────
    # Four gates (up4..up1) from deep to shallow; deep gates have coarse resolution
    # (sensitive to large burn regions) and shallow gates have fine resolution
    # (sensitive to boundary details). Shown side-by-side with the RGB for comparison.
    if attn_maps:
        path = os.path.join(out_dir, f"sample_{sample_id}_attention.png")
        n = len(attn_maps)
        fig, axes = plt.subplots(1, n + 1, figsize=(4.5 * (n + 1), 4.5),
                                 facecolor=DARK_BG)
        axes[0].imshow(rgb)
        axes[0].set_title("RGB Composite", color="white", fontsize=10)
        axes[0].axis("off")

        for i, (gname, amap) in enumerate(attn_maps):
            ax = axes[i + 1]
            im = ax.imshow(amap, cmap="hot", vmin=0, vmax=1,
                           interpolation="bilinear")
            ax.set_title(f"Attention\n{gname}", color="white", fontsize=10)
            ax.axis("off")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.yaxis.set_tick_params(color="white")
            plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
            cb.outline.set_edgecolor("white")

        fig.suptitle(f"Attention Gate Maps — sample {sample_id}",
                     color="white", fontsize=12, y=1.02)
        plt.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        saved["attention"] = path

    return saved


# =============================================================================
# SECTION 5 — Per-architecture driver
# =============================================================================

def _select_representative_samples(
    test_files: list, n: int, seed: int = SEED
) -> list:
    """
    Choose n patches that span all four severity classes as evenly as possible.

    Strategy: load y labels for all test files, group by dominant class, then
    sample ≈ n/4 from each group.  Falls back to random selection if n is
    larger than the available test set or class coverage is sparse.
    """
    rng = np.random.default_rng(seed)

    # Collect dominant class per patch (cheap: only loads 'y')
    dominant = []
    for fp in test_files:
        try:
            y = np.load(fp, allow_pickle=True)["y"]
            dominant.append(int(np.bincount(y.ravel(), minlength=N_CLASSES).argmax()))
        except Exception:
            dominant.append(-1)

    dominant = np.array(dominant)
    per_class = max(1, n // N_CLASSES)
    selected  = []

    for c in range(N_CLASSES):
        idx_c = np.where(dominant == c)[0]
        if len(idx_c) == 0:
            continue
        k = min(per_class, len(idx_c))
        chosen = rng.choice(idx_c, k, replace=False)
        selected.extend(chosen.tolist())

    # top-up if we are still short
    used  = set(selected)
    extra = [i for i in range(len(test_files)) if i not in used]
    if len(selected) < n and extra:
        k = min(n - len(selected), len(extra))
        selected.extend(rng.choice(extra, k, replace=False).tolist())

    # never exceed n
    selected = selected[:n]
    rng.shuffle(selected)
    return [test_files[i] for i in selected]


def _run_one_arch(
    arch: str,
    checkpoint_dir: str,
    sample_files: list,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    out_dir: str,
) -> dict:
    """
    Run the full Grad-CAM + attention pipeline for one architecture.
    Returns a summary dict for JSON logging.
    """
    print(f"\n  {'─'*52}")
    print(f"  Architecture : {arch.upper()}")

    model, ckpt_loaded = load_model(arch, checkpoint_dir, device)
    is_attention       = isinstance(model, AttentionUNet)

    cam_times   = []
    sample_logs = []

    for idx, fpath in enumerate(sample_files):
        # ── Load patch ────────────────────────────────────────────────────────
        data   = np.load(fpath, allow_pickle=True)
        X_raw  = data["X"].astype(np.float32)                   # (7, H, W)
        y_true = data["y"].astype(np.int64)
        y_true = np.clip(y_true, 0, N_CLASSES - 1)

        # Normalise
        X_norm = (X_raw - mean[:, None, None]) / (std[:, None, None] + 1e-8)
        X_t    = torch.from_numpy(X_norm).unsqueeze(0)          # (1, 7, H, W)

        sample_id = f"{arch}_{idx:02d}"

        # ── Grad-CAM ──────────────────────────────────────────────────────────
        cam, pred_map, t_ms = compute_gradcam(model, X_t, device)
        cam_times.append(t_ms)

        # ── Attention maps (AttentionUNet only) ───────────────────────────────
        attn_maps = None
        if is_attention:
            attn_maps = compute_attention_maps(model, X_t, device)

        # ── Save all images ───────────────────────────────────────────────────
        saved = visualize_and_save(
            sample_id=sample_id,
            X_norm=X_norm,
            y_true=y_true,
            pred_map=pred_map,
            cam=cam,
            out_dir=out_dir,
            mean=mean,
            std=std,
            attn_maps=attn_maps,
        )

        sample_logs.append({
            "sample_id":      sample_id,
            "source_file":    str(fpath),
            "cam_time_ms":    round(t_ms, 3),
            "saved_files":    list(saved.values()),
        })
        print(f"    [{idx+1}/{len(sample_files)}] {sample_id}  "
              f"cam={t_ms:.1f}ms  files={len(saved)}")

    summary = {
        "arch":             arch,
        "checkpoint_loaded": ckpt_loaded,
        "n_samples":        len(sample_files),
        "cam_time_ms_mean": round(float(np.mean(cam_times)), 3) if cam_times else 0,
        "cam_time_ms_std":  round(float(np.std(cam_times)),  3) if cam_times else 0,
        "samples":          sample_logs,
    }
    return summary


# =============================================================================
# SECTION 6 — main()
# =============================================================================

def main():
    """
    Entry point for the explainability pipeline.

    Parses CLI arguments, loads test patches, selects representative samples,
    and runs Grad-CAM (+ attention maps) for each requested architecture.
    Results are saved as PNG images and a JSON metrics file.
    """
    parser = argparse.ArgumentParser(
        description="Grad-CAM explainability for wildfire burn-severity models")
    parser.add_argument("--checkpoint_dir", type=str, default=RESULTS_DIR,
                        help="Directory containing best_model.pth checkpoints")
    parser.add_argument("--patch_dir",   type=str, default=PATCH_DIR,
                        help="Root directory of extracted .npz patches")
    parser.add_argument("--splits_json", type=str, default=SPLITS_JSON,
                        help="Path to splits_v2.json")
    parser.add_argument("--metrics_dir", type=str, default=METRICS_DIR,
                        help="Directory to write explainability_metrics.json")
    parser.add_argument("--out_dir",     type=str, default=EXPL_DIR,
                        help="Root output directory for PNG images")
    parser.add_argument("--n_samples",   type=int, default=8,
                        help="Number of representative test samples (5–10)")
    parser.add_argument("--archs",       type=str, default="all",
                        help="Comma-separated architecture names or 'all'")
    args = parser.parse_args()

    # Clamp n_samples to [5, 10]
    n_samples = max(5, min(10, args.n_samples))

    os.makedirs(args.out_dir,     exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)

    # Reproducibility
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  Grad-CAM Explainability  —  Wildfire Burn Severity")
    print("=" * 60)
    print(f"  Device      : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Samples     : {n_samples} per model")

    # ── Select architectures ──────────────────────────────────────────────────
    if args.archs.strip().lower() == "all":
        archs = list(MODEL_REGISTRY.keys())
    else:
        archs = [a.strip().lower() for a in args.archs.split(",")]
    print(f"  Architectures: {archs}")

    # ── Load test split ───────────────────────────────────────────────────────
    print("\nLoading test split ...")
    _, _, test_files = load_splits(args.patch_dir, args.splits_json)
    print(f"  Test patches available: {len(test_files)}")

    if len(test_files) == 0:
        raise RuntimeError(
            f"No test patches found under {args.patch_dir}. "
            "Run sentinel2_dataset_v2.py and gpu_train_v2.slurm first.")

    sample_files = _select_representative_samples(test_files, n_samples)
    print(f"  Selected {len(sample_files)} representative samples "
          "(balanced across severity classes)")

    # ── Channel statistics ────────────────────────────────────────────────────
    stats_path = os.path.join(args.checkpoint_dir, "channel_stats.npz")
    if os.path.exists(stats_path):
        s    = np.load(stats_path)
        mean = s["mean"].astype(np.float32)
        std  = s["std"].astype(np.float32)
        print(f"  Channel stats : {stats_path}")
    else:
        print("  channel_stats.npz not found — computing from test patches ...")
        train_files, _, _ = load_splits(args.patch_dir, args.splits_json)
        mean, std = compute_channel_stats(train_files)

    # ── Run per architecture ──────────────────────────────────────────────────
    all_results = []
    for arch in archs:
        arch_out = os.path.join(args.out_dir, arch)
        try:
            result = _run_one_arch(
                arch=arch,
                checkpoint_dir=args.checkpoint_dir,
                sample_files=sample_files,
                mean=mean,
                std=std,
                device=device,
                out_dir=arch_out,
            )
        except Exception as exc:
            print(f"  ERROR ({arch}): {exc}")
            result = {"arch": arch, "error": str(exc)}
        all_results.append(result)

    # ── Summary Grad-CAM timing figure ────────────────────────────────────────
    fig_dir = os.path.join(args.metrics_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    _plot_cam_timing(all_results, os.path.join(fig_dir, "gradcam_timing.png"))

    # ── Write JSON ────────────────────────────────────────────────────────────
    json_path = os.path.join(args.metrics_dir, "explainability_metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Metrics saved : {json_path}")

    print("\n" + "=" * 60)
    print("  Explainability complete.")
    print(f"  Figures       : {args.out_dir}/{{arch}}/sample_*_*.png")
    print("=" * 60)


# =============================================================================
# SECTION 7 — Timing summary plot
# =============================================================================

def _plot_cam_timing(all_results: list, save_path: str):
    """Bar chart of mean Grad-CAM time (ms) per architecture."""
    archs = [r["arch"] for r in all_results if "cam_time_ms_mean" in r]
    means = [r["cam_time_ms_mean"] for r in all_results
             if "cam_time_ms_mean" in r]
    stds  = [r["cam_time_ms_std"]  for r in all_results
             if "cam_time_ms_std" in r]
    if not archs:
        return

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    colors = [plt.cm.tab10(i / len(archs)) for i in range(len(archs))]
    bars   = ax.bar(archs, means, color=colors, width=0.5,
                    yerr=stds, capsize=5,
                    error_kw={"ecolor": "white", "alpha": 0.7})
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}ms", ha="center", va="bottom",
                color="white", fontsize=9)
    ax.set_ylabel("Grad-CAM time (ms)", color="white")
    ax.set_title("Grad-CAM Computation Time per Architecture",
                 color="white", fontsize=12)
    ax.tick_params(colors="white")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("white")
    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Timing figure : {save_path}")


if __name__ == "__main__":
    main()
