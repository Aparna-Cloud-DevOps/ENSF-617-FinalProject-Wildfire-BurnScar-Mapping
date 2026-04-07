"""
inference_analysis.py
=====================
Inference performance analysis for all trained U-Net variants.

For each model (UNet, ResUNet, AttentionUNet, UNet++):
  1. Load the trained checkpoint
  2. Randomly sample >= 100 patches from the test split
  3. Measure per-patch inference time (ms) using torch.cuda.synchronize()
  4. Measure peak GPU memory during inference (MB)
  5. Measure model size on disk (MB)

All inference runs in eval() mode under torch.no_grad().

Outputs
-------
  metrics/inference_metrics.json
  metrics/figures/inference_time_per_patch.png
  metrics/figures/inference_gpu_memory.png
  metrics/figures/inference_model_size.png

Usage
-----
  python inference_analysis.py
  python inference_analysis.py --n_patches 200 --batch 16
  sbatch inference_analysis.slurm
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet_training_v2 import (
    N_INPUT, N_CLASSES, PATCH_SZ, F_BASE, DROPOUT, SEED,
    DARK_BG, SEVERITY_NAMES, MODEL_REGISTRY,
    load_splits, compute_channel_stats,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE         = os.environ.get("WILDFIRE_BASE_DIR",
               os.path.dirname(os.path.abspath(__file__)))
PATCH_DIR    = os.path.join(BASE, "data", "sentinel2", "patches_v2")
SPLITS_JSON  = os.path.join(BASE, "data", "sentinel2", "splits_v2.json")
RESULTS_DIR  = os.path.join(BASE, "outputs", "sentinel2_results_v2")
METRICS_DIR  = os.path.join(BASE, "metrics")
FIG_DIR      = os.path.join(METRICS_DIR, "figures")
MIN_PATCHES  = 100


# =============================================================================
# SECTION 1 — DATASET
# =============================================================================

class PatchDataset(Dataset):
    """Minimal dataset: loads .npz patches, normalises, no augmentation."""

    def __init__(self, files, mean, std):
        self.files = list(files)
        self.mean  = torch.from_numpy(mean.astype(np.float32))
        self.std   = torch.from_numpy(std.astype(np.float32))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        X = torch.from_numpy(data["X"].astype(np.float32))   # (7, 256, 256)
        y = torch.from_numpy(data["y"].astype(np.int64)).clamp(0, N_CLASSES - 1)
        X = (X - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-8)
        return X, y


# =============================================================================
# SECTION 2 — MODEL LOADING
# =============================================================================

def load_model_checkpoint(ckpt_path, arch, device):
    """
    Load model from checkpoint.
    Runs in eval() mode — ready for inference.
    Returns (model, metadata_dict).
    """
    ModelClass = MODEL_REGISTRY[arch.lower()]
    model = ModelClass(
        input_channels=N_INPUT,
        num_classes=N_CLASSES,
        f=F_BASE,
        drop=DROPOUT,
    ).to(device)

    meta = {}
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=True)
            meta = {k: v for k, v in ckpt.items() if k != "model"}
        else:
            model.load_state_dict(ckpt, strict=True)
        print(f"    Loaded: {ckpt_path}")
    else:
        print(f"    WARNING: {ckpt_path} not found — using random init.")

    model.eval()
    return model, meta


def checkpoint_size_mb(ckpt_path):
    """File size of .pth checkpoint in MB."""
    if os.path.exists(ckpt_path):
        return round(os.path.getsize(ckpt_path) / (1024 ** 2), 2)
    return None


def model_param_size_mb(model):
    """In-memory size of parameters + buffers in MB."""
    total = sum(
        p.numel() * p.element_size()
        for p in list(model.parameters()) + list(model.buffers())
    )
    return round(total / (1024 ** 2), 2)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# SECTION 3 — TIMING AND MEMORY MEASUREMENT
# =============================================================================

def measure_inference(model, dataloader, device, n_patches_target):
    """
    Measure per-patch inference latency and peak GPU memory.

    Timing method: wall-clock time bracketed by torch.cuda.synchronize() calls.
    synchronize() blocks until all queued CUDA kernels finish, so the measured
    interval includes real GPU execution time (not just kernel launch latency).

    Without synchronize(), timing only measures host-side kernel submission
    which is much faster than actual GPU compute — a common measurement mistake.

    Returns
    -------
    dict with keys:
        per_patch_times_ms         : list[float]  one entry per patch
        mean_ms / std_ms           : average and spread of latency
        min_ms / max_ms / p50 / p95: percentile statistics
        peak_gpu_memory_mb         : max GPU memory during inference (None on CPU)
        n_patches_measured         : actual number of patches timed
        throughput_patches_per_sec : 1000 / mean_ms (patches per second)
    """
    use_cuda = device.type == "cuda"

    # Reset peak memory counter to measure only this model's footprint
    if use_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    per_patch_times = []
    n_measured      = 0

    model.eval()
    with torch.no_grad():
        for X, _ in dataloader:
            if n_measured >= n_patches_target:
                break

            X  = X.to(device, non_blocking=True)
            bs = X.shape[0]

            # Synchronize before timing to flush any pending GPU work from prior ops
            if use_cuda:
                torch.cuda.synchronize(device)
                t_start = time.perf_counter()
                _ = model(X)
                torch.cuda.synchronize(device)   # block until GPU finishes this batch
                t_end = time.perf_counter()
            else:
                t_start = time.perf_counter()
                _ = model(X)
                t_end = time.perf_counter()

            batch_ms     = (t_end - t_start) * 1000.0
            per_patch_ms = batch_ms / bs          # average per-patch time in the batch

            # Repeat the same per_patch_ms for each patch in the batch
            # (individual patch times within a batch aren't separable without profiling)
            per_patch_times.extend([per_patch_ms] * bs)
            n_measured += bs

    if use_cuda:
        # max_memory_allocated tracks the peak, not the current usage
        peak_gpu_mb = round(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1)
    else:
        peak_gpu_mb = None

    times = np.array(per_patch_times[:n_patches_target])
    throughput = 1000.0 / times.mean() if times.mean() > 0 else 0.0

    return {
        "per_patch_times_ms":       [round(t, 4) for t in times.tolist()],
        "mean_ms":                  round(float(times.mean()),           4),
        "std_ms":                   round(float(times.std()),            4),
        "min_ms":                   round(float(times.min()),            4),
        "max_ms":                   round(float(times.max()),            4),
        "p50_ms":                   round(float(np.percentile(times, 50)), 4),
        "p95_ms":                   round(float(np.percentile(times, 95)), 4),
        "peak_gpu_memory_mb":       peak_gpu_mb,
        "n_patches_measured":       int(len(times)),
        "throughput_patches_per_sec": round(throughput, 1),
    }


# =============================================================================
# SECTION 4 — VISUALISATION
# =============================================================================

def plot_bar(archs, values, ylabel, title, save_path, color="#4fc3f7"):
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

        bars = ax.bar(archs, values, color=color, alpha=0.85, width=0.5)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(values),
                    f"{v:.2f}", ha="center", va="bottom",
                    color="white", fontsize=10)

        ax.set_xticklabels(archs, color="white")
        ax.set_ylabel(ylabel, color="white")
        ax.set_title(title,   color="white")
        plt.tight_layout()
        fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"  Saved: {save_path}")
    except Exception as e:
        print(f"  Warning: could not save {title}: {e}")


def plot_latency_distribution(results, save_path):
    """Box plot of per-patch latency distributions for all models."""
    try:
        import matplotlib.pyplot as plt

        archs = list(results.keys())
        data  = [results[a]["timing"]["per_patch_times_ms"] for a in archs]

        fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

        bp = ax.boxplot(data, labels=archs, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2))
        colors = ["#4fc3f7", "#ef5350", "#66bb6a", "#ffa726"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_ylabel("Per-patch inference time (ms)", color="white")
        ax.set_title("Inference Latency Distribution per Model", color="white")
        ax.set_xticklabels(archs, color="white")
        plt.tight_layout()
        fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"  Saved: {save_path}")
    except Exception as e:
        print(f"  Warning: could not save latency distribution: {e}")


# =============================================================================
# SECTION 5 — MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inference performance analysis for U-Net models")
    parser.add_argument("--checkpoint_dir", type=str, default=RESULTS_DIR,
                        help="Directory containing .pth checkpoints")
    parser.add_argument("--patch_dir",   type=str, default=PATCH_DIR)
    parser.add_argument("--splits_json", type=str, default=SPLITS_JSON)
    parser.add_argument("--metrics_dir", type=str, default=METRICS_DIR)
    parser.add_argument("--n_patches",   type=int, default=MIN_PATCHES,
                        help="Number of test patches to measure (default 100)")
    parser.add_argument("--batch",       type=int, default=8,
                        help="Batch size during inference (default 8)")
    parser.add_argument("--archs",       type=str, default="all",
                        help="Comma-separated model names or 'all'")
    args = parser.parse_args()

    n_patches = max(args.n_patches, MIN_PATCHES)   # enforce minimum 100

    os.makedirs(args.metrics_dir, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # Reproducibility
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  Inference Performance Analysis")
    print("=" * 60)
    print(f"  Device     : {device}"
          + (f" ({torch.cuda.get_device_name(0)})"
             if device.type == "cuda" else ""))
    print(f"  Patches    : {n_patches} (minimum)")
    print(f"  Batch size : {args.batch}")

    # ── Load test split ───────────────────────────────────────────────────────
    print("\nLoading test split ...")
    _, _, test_files = load_splits(args.patch_dir, args.splits_json)
    print(f"  Test patches available: {len(test_files)}")

    if len(test_files) < MIN_PATCHES:
        print(f"  WARNING: only {len(test_files)} test patches available. "
              f"Sampling with replacement to reach {n_patches}.")
        rng        = np.random.default_rng(SEED)
        test_files = [test_files[i]
                      for i in rng.choice(len(test_files),
                                           n_patches, replace=True)]
    else:
        rng        = np.random.default_rng(SEED)
        test_files = [test_files[i]
                      for i in rng.choice(len(test_files),
                                           n_patches, replace=False)]

    print(f"  Sampled    : {len(test_files)} patches for benchmarking")

    # ── Channel stats (training split) ────────────────────────────────────────
    stats_path = os.path.join(args.checkpoint_dir, "channel_stats.npz")
    if os.path.exists(stats_path):
        s    = np.load(stats_path)
        mean = s["mean"].astype(np.float32)
        std  = s["std"].astype(np.float32)
    else:
        print("  channel_stats.npz not found — computing from test patches ...")
        mean, std = compute_channel_stats(test_files)

    # ── DataLoader ────────────────────────────────────────────────────────────
    dataset = PatchDataset(test_files, mean, std)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=False,
                         num_workers=0, pin_memory=False)

    # ── Determine architectures to benchmark ─────────────────────────────────
    if args.archs.lower() == "all":
        archs = list(MODEL_REGISTRY.keys())
    else:
        archs = [a.strip().lower() for a in args.archs.split(",")]

    # ── Benchmark each model ──────────────────────────────────────────────────
    all_results = {}

    for arch in archs:
        print(f"\n{'─'*50}")
        print(f"  Model: {arch.upper()}")
        print(f"{'─'*50}")

        # Checkpoint path — use exact naming convention:
        #   unet      → best_model.pth
        #   resunet   → resunet_best_model.pth
        #   attention → attention_best_model.pth
        #   unetpp    → unetpp_best_model.pth
        if arch == "unet":
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
        else:
            ckpt_path = os.path.join(args.checkpoint_dir, f"{arch}_best_model.pth")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")

        model, ckpt_meta = load_model_checkpoint(ckpt_path, arch, device)

        # Static metrics
        n_params    = count_parameters(model)
        mem_mb      = model_param_size_mb(model)
        ckpt_mb     = checkpoint_size_mb(ckpt_path)

        print(f"    Parameters    : {n_params:,}")
        print(f"    In-memory     : {mem_mb} MB")
        print(f"    On-disk (.pth): {ckpt_mb} MB")

        # Inference timing + GPU memory
        print(f"    Running inference on {n_patches} patches ...")
        timing = measure_inference(model, loader, device, n_patches)

        print(f"    Mean latency  : {timing['mean_ms']:.3f} ms / patch")
        print(f"    p95 latency   : {timing['p95_ms']:.3f} ms / patch")
        print(f"    Throughput    : {timing['throughput_patches_per_sec']:.1f} patches/sec")
        if timing["peak_gpu_memory_mb"] is not None:
            print(f"    Peak GPU mem  : {timing['peak_gpu_memory_mb']} MB")

        all_results[arch] = {
            "arch":             arch,
            "checkpoint_path":  ckpt_path,
            "n_parameters":     n_params,
            "model_size_mb": {
                "in_memory":    mem_mb,
                "on_disk":      ckpt_mb,
            },
            "n_patches_sampled": n_patches,
            "batch_size":        args.batch,
            "timing":            timing,
            "device":            str(device),
            "ckpt_meta": {
                k: (float(v) if isinstance(v, (int, float)) else str(v))
                for k, v in ckpt_meta.items()
                if k not in ("metrics",)
            },
        }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = os.path.join(args.metrics_dir, "inference_metrics.json")
    # Omit raw per-patch time lists from JSON to keep file readable
    json_results = {}
    for arch, res in all_results.items():
        r = {k: v for k, v in res.items()}
        r["timing"] = {k: v for k, v in res["timing"].items()
                       if k != "per_patch_times_ms"}
        json_results[arch] = r

    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved -> {out_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    archs_list  = list(all_results.keys())

    plot_bar(
        archs_list,
        [all_results[a]["timing"]["mean_ms"] for a in archs_list],
        ylabel="Mean inference time per patch (ms)",
        title="Inference Time per Patch",
        save_path=os.path.join(FIG_DIR, "inference_time_per_patch.png"),
        color="#4fc3f7",
    )

    gpu_vals = [all_results[a]["timing"]["peak_gpu_memory_mb"] or 0
                for a in archs_list]
    if any(v > 0 for v in gpu_vals):
        plot_bar(
            archs_list, gpu_vals,
            ylabel="Peak GPU memory (MB)",
            title="GPU Memory Usage During Inference (batch={})".format(
                args.batch),
            save_path=os.path.join(FIG_DIR, "inference_gpu_memory.png"),
            color="#ef5350",
        )

    plot_bar(
        archs_list,
        [all_results[a]["model_size_mb"]["on_disk"] or 0 for a in archs_list],
        ylabel="Model size on disk (MB)",
        title="Model Size on Disk (.pth)",
        save_path=os.path.join(FIG_DIR, "inference_model_size.png"),
        color="#66bb6a",
    )

    plot_latency_distribution(
        all_results,
        os.path.join(FIG_DIR, "inference_latency_distribution.png"),
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  INFERENCE SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<13} {'Params(M)':>10} {'Disk(MB)':>9} "
          f"{'Mean(ms)':>9} {'p95(ms)':>8} {'GPU(MB)':>8} {'Patches/s':>10}")
    print("  " + "-" * 66)
    for arch in archs_list:
        r  = all_results[arch]
        t  = r["timing"]
        print(f"  {arch:<13} "
              f"{r['n_parameters']/1e6:>10.2f} "
              f"{r['model_size_mb']['on_disk'] or 0:>9.1f} "
              f"{t['mean_ms']:>9.3f} "
              f"{t['p95_ms']:>8.3f} "
              f"{t['peak_gpu_memory_mb'] or 0:>8.1f} "
              f"{t['throughput_patches_per_sec']:>10.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
