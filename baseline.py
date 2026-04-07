"""
baseline.py
===========
Baseline burn-severity classifiers evaluated on the same val/test splits
produced by sentinel2_dataset_v2.py.

Two baselines
-------------
1. dNBR Threshold Classifier
   Applies fixed MTBS-standard dNBR thresholds pixel-wise to the dNBR
   channel stored in every .npz patch (channel index 6 of X, or the
   dedicated 'dnbr' key).

   Standard MTBS thresholds (dNBR × 1000 scale, converted to reflectance):
     Unburned/Low : dNBR < 0.100
     Moderate     : 0.100 ≤ dNBR < 0.270
     High         : 0.270 ≤ dNBR < 0.440
     Very High    : dNBR ≥ 0.440

2. Random Forest Classifier (optional, --rf flag)
   Trains on per-pixel features extracted from training patches:
     [R, G, B, NIR, SWIR1, NBR, dNBR]  → 7 features
   Uses 200 estimators, max_depth=20, class_weight='balanced'.
   Fitted on a random subsample of training pixels (default 500k) to
   keep memory manageable.

Outputs
-------
  outputs/sentinel2_results_v2/baseline_metrics.json
  outputs/sentinel2_results_v2/figures/baseline_confusion_val.png
  outputs/sentinel2_results_v2/figures/baseline_confusion_test.png
  outputs/sentinel2_results_v2/figures/baseline_per_class_iou.png

Usage
-----
  python baseline.py                          # dNBR threshold only
  python baseline.py --rf                     # dNBR + Random Forest
  python baseline.py --rf --max_train_px 1000000
"""

import os
import json
import argparse
import warnings
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    average_precision_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants — must match unet_training_v2.py
# ---------------------------------------------------------------------------
BASE       = os.environ.get("WILDFIRE_BASE_DIR",
             os.path.dirname(os.path.abspath(__file__)))
PATCH_DIR  = os.path.join(BASE, "data", "sentinel2", "patches_v2")
SPLITS_JSON = os.path.join(BASE, "data", "sentinel2", "splits_v2.json")
OUT_DIR    = os.path.join(BASE, "outputs", "sentinel2_results_v2")
FIG_DIR    = os.path.join(OUT_DIR, "figures")

N_CLASSES  = 4
SEED       = 42
CLASS_NAMES = ["Unburned/Low", "Moderate", "High", "Very High"]

# MTBS standard dNBR thresholds on REFLECTANCE scale (NOT the ×1000 scale in MTBS docs).
# Source: Key & Benson (2006) FIREMON protocol — converted: 100→0.100, 270→0.270, 440→0.440
# These define the severity class boundaries used by the MTBS program nationally.
DNBR_THRESHOLDS = [0.100, 0.270, 0.440]   # boundaries between classes 0|1, 1|2, 2|3

# Channel layout inside X array (7, 256, 256) — must match sentinel2_dataset_v2.py:
#   0=Red  1=Green  2=Blue  3=NIR  4=SWIR1  5=NBR  6=dNBR
CH_DNBR = 6   # index of dNBR channel used by the threshold classifier
CH_NBR  = 5   # index of pre-fire NBR (not used by threshold baseline, but available)


# =============================================================================
# SECTION 1  DATA LOADING  (reads .npz directly — no torch dependency)
# =============================================================================

def load_splits(patch_dir=PATCH_DIR, splits_json=SPLITS_JSON):
    """
    Returns (train_files, val_files, test_files) — same logic as
    unet_training_v2.py so baselines use identical splits.
    """
    all_files = sorted(Path(patch_dir).rglob("*.npz"))
    if not all_files:
        raise FileNotFoundError(f"No .npz files found under {patch_dir}")

    def _random_split(files):
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(files))
        n   = len(files)
        tr  = [str(files[i]) for i in idx[:int(0.70*n)]]
        va  = [str(files[i]) for i in idx[int(0.70*n):int(0.85*n)]]
        te  = [str(files[i]) for i in idx[int(0.85*n):]]
        return tr, va, te

    if not os.path.exists(splits_json):
        print("  splits_v2.json not found — using random 70/15/15 split")
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
        print("  WARNING: temporal split has empty val/test — falling back to random 70/15/15")
        return _random_split(all_files)

    return tr, va, te


def load_patch(fp):
    """
    Load one .npz patch.
    Returns (X, y, dnbr) where:
        X     float32 (7, 256, 256)
        y     int64   (256, 256)   labels 0–3
        dnbr  float32 (256, 256)  — from dedicated key if present, else X[6]
    """
    data = np.load(fp, allow_pickle=True)
    X    = data["X"].astype(np.float32)
    y    = data["y"].astype(np.int64).clip(0, N_CLASSES - 1)

    # Use dedicated 'dnbr' key if saved by dataset builder, else channel 6
    if "dnbr" in data:
        dnbr = data["dnbr"].astype(np.float32)
    else:
        dnbr = X[CH_DNBR]

    return X, y, dnbr


# =============================================================================
# SECTION 2  METRICS  (pure numpy — no torch)
# =============================================================================

def compute_flat_metrics(preds, trues, probs=None, n_cls=N_CLASSES):
    """
    Compute per-class and macro metrics from flat 1-D arrays.

    Parameters
    ----------
    preds : int array (N,)
    trues : int array (N,)
    probs : float array (N, n_cls) or None  — needed for PR-AUC

    Returns
    -------
    dict with keys: iou, dice, pr_auc, mean_iou, mean_dice, mean_pr_auc
    """
    iou_list, dice_list, ap_list = [], [], []

    for c in range(n_cls):
        tp = np.sum((preds == c) & (trues == c))
        fp = np.sum((preds == c) & (trues != c))
        fn = np.sum((preds != c) & (trues == c))

        iou  = float(tp / (tp + fp + fn + 1e-8))
        dice = float(2 * tp / (2 * tp + fp + fn + 1e-8))
        iou_list.append(iou)
        dice_list.append(dice)

        if probs is not None:
            try:
                ap = float(average_precision_score(
                    (trues == c).astype(int), probs[:, c]))
            except ValueError:
                ap = 0.0
        else:
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


def print_metrics(metrics, split_name, model_name):
    print(f"\n  [{model_name}] {split_name}")
    print(f"    mIoU    : {metrics['mean_iou']:.4f}")
    print(f"    mDice   : {metrics['mean_dice']:.4f}")
    print(f"    PR-AUC  : {metrics['mean_pr_auc']:.4f}")
    for c, name in enumerate(CLASS_NAMES):
        print(f"    {name:<15}  IoU={metrics['iou'][c]:.4f}  "
              f"Dice={metrics['dice'][c]:.4f}  AP={metrics['pr_auc'][c]:.4f}")


# =============================================================================
# SECTION 3  VISUALISATION
# =============================================================================

def plot_confusion(cm, title, save_path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        DARK_BG = "#1a1a2e"
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)

        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)

        ticks = list(range(N_CLASSES))
        ax.set_xticks(ticks); ax.set_xticklabels(CLASS_NAMES, rotation=30,
                                                   ha="right", color="white")
        ax.set_yticks(ticks); ax.set_yticklabels(CLASS_NAMES, color="white")
        ax.set_xlabel("Predicted", color="white")
        ax.set_ylabel("True",      color="white")
        ax.tick_params(colors="white")
        ax.set_title(title, color="white")

        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, f"{cm[i,j]:,}\n({cm_norm[i,j]:.2f})",
                        ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[i, j] < 0.6 else "black")

        plt.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"  Saved: {save_path}")
    except Exception as e:
        print(f"  Warning: could not save confusion matrix: {e}")


def plot_per_class_iou(results_dict, save_path):
    """
    Grouped bar chart: per-class IoU for each baseline on val and test.
    results_dict = { 'dNBR_val': metrics, 'dNBR_test': metrics, ... }
    """
    try:
        import matplotlib.pyplot as plt

        DARK_BG = "#1a1a2e"
        colors  = ["#4fc3f7", "#ef5350", "#66bb6a", "#ffa726",
                   "#ab47bc", "#26c6da", "#ff7043", "#d4e157"]

        fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)

        labels  = CLASS_NAMES
        n_groups = len(labels)
        keys    = list(results_dict.keys())
        n_bars  = len(keys)
        width   = 0.8 / n_bars
        x       = np.arange(n_groups)

        for i, key in enumerate(keys):
            iou_vals = results_dict[key]["iou"]
            offset   = (i - n_bars / 2 + 0.5) * width
            ax.bar(x + offset, iou_vals, width, label=key,
                   color=colors[i % len(colors)], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, color="white")
        ax.set_ylabel("IoU", color="white")
        ax.set_ylim(0, 1)
        ax.set_title("Baseline Per-Class IoU", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#16213e", labelcolor="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

        plt.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"  Saved: {save_path}")
    except Exception as e:
        print(f"  Warning: could not save per-class IoU plot: {e}")


# =============================================================================
# SECTION 4  BASELINE 1 — dNBR THRESHOLD CLASSIFIER
# =============================================================================

def dnbr_classify(dnbr_patch, thresholds=DNBR_THRESHOLDS):
    """
    Apply fixed dNBR thresholds pixel-wise.

    thresholds = [t1, t2, t3]  → 4 classes:
        dNBR < t1             → 0  (Unburned/Low)
        t1 ≤ dNBR < t2        → 1  (Moderate)
        t2 ≤ dNBR < t3        → 2  (High)
        dNBR ≥ t3             → 3  (Very High)
    """
    pred = np.zeros_like(dnbr_patch, dtype=np.int64)
    pred[dnbr_patch >= thresholds[0]] = 1
    pred[dnbr_patch >= thresholds[1]] = 2
    pred[dnbr_patch >= thresholds[2]] = 3
    return pred


def evaluate_dnbr_baseline(files, split_name):
    """
    Run dNBR threshold classifier over all patches in a split.
    Returns (metrics_dict, confusion_matrix, all_preds, all_trues).
    """
    print(f"  Evaluating dNBR threshold baseline on {split_name} "
          f"({len(files)} patches) ...")
    t0 = time.time()

    all_preds, all_trues = [], []

    for fp in files:
        try:
            _, y, dnbr = load_patch(fp)
            pred = dnbr_classify(dnbr)
            all_preds.append(pred.ravel())
            all_trues.append(y.ravel())
        except Exception as e:
            print(f"    Warning: skipping {fp}: {e}")

    if not all_preds:
        print(f"  ERROR: no patches loaded for {split_name}")
        return None, None, None, None

    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)

    # Build soft probabilities from dNBR for PR-AUC
    # Use sigmoid-scaled distance to each threshold as proxy probability
    all_probs = _dnbr_soft_probs(files)

    metrics = compute_flat_metrics(preds, trues, probs=all_probs)
    cm      = confusion_matrix(trues, preds, labels=list(range(N_CLASSES)))

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s — mIoU={metrics['mean_iou']:.4f}  "
          f"mDice={metrics['mean_dice']:.4f}")
    return metrics, cm, preds, trues


def _dnbr_soft_probs(files, thresholds=DNBR_THRESHOLDS):
    """
    Approximate class probabilities from dNBR using sigmoid of distance
    to each threshold boundary. Used only for PR-AUC calculation.
    """
    all_dnbr = []
    for fp in files:
        try:
            _, _, dnbr = load_patch(fp)
            all_dnbr.append(dnbr.ravel())
        except Exception:
            pass
    if not all_dnbr:
        return None

    d = np.concatenate(all_dnbr)   # (N,)
    scale = 20.0                    # sharpness of sigmoid

    def _sig(x):
        # Sigmoid with clipped input to prevent float overflow in exp()
        return 1.0 / (1.0 + np.exp(-np.clip(x * scale, -50, 50)))

    # Soft probabilities derived from sigmoid-smoothed threshold distances.
    # Each class occupies the probability "band" between two adjacent thresholds.
    # This is a differentiable approximation of the hard threshold classifier.
    # p(class=0): probability dNBR < t1
    p0 = 1.0 - _sig(d - thresholds[0])
    # p(class=1): probability t1 ≤ dNBR < t2
    p1 = _sig(d - thresholds[0]) - _sig(d - thresholds[1])
    p1 = np.clip(p1, 0, 1)  # can go slightly negative due to sigmoid asymmetry
    # p(class=2): probability t2 ≤ dNBR < t3
    p2 = _sig(d - thresholds[1]) - _sig(d - thresholds[2])
    p2 = np.clip(p2, 0, 1)
    # p(class=3): probability dNBR ≥ t3
    p3 = _sig(d - thresholds[2])

    probs = np.stack([p0, p1, p2, p3], axis=1)   # (N, 4)
    # Normalise rows to sum to 1
    probs /= (probs.sum(axis=1, keepdims=True) + 1e-8)
    return probs


# =============================================================================
# SECTION 5  BASELINE 2 — RANDOM FOREST
# =============================================================================

def collect_rf_features(files, max_pixels=500_000, seed=SEED):
    """
    Sample pixel features from patches for Random Forest training.

    Features per pixel: [R, G, B, NIR, SWIR1, NBR, dNBR]  (7 features)
    Labels: severity class 0–3

    Returns (X_feat, y_labels) as float32/int arrays.
    """
    rng     = np.random.default_rng(seed)
    feats   = []
    labels  = []
    total   = 0

    # Shuffle files so sampling is representative
    file_order = list(files)
    rng.shuffle(file_order)

    # Divide pixel budget evenly across patches to avoid over-sampling from any one fire.
    # This prevents a large fire with many patches from dominating the RF training data.
    per_patch_budget = max(1, max_pixels // max(len(files), 1))

    for fp in file_order:
        if total >= max_pixels:
            break
        try:
            X, y, _ = load_patch(fp)
            # X shape: (7, 256, 256) → flatten spatial dims → (65536, 7) pixel feature matrix
            px = X.reshape(7, -1).T          # (65536, 7)
            lb = y.ravel()                   # (65536,)  severity labels

            # Sample per_patch_budget pixels
            n_take = min(per_patch_budget, len(lb))
            idx    = rng.choice(len(lb), n_take, replace=False)
            feats.append(px[idx])
            labels.append(lb[idx])
            total += n_take
        except Exception as e:
            print(f"    Warning: skipping {fp}: {e}")

    if not feats:
        raise RuntimeError("No features collected for Random Forest training.")

    return np.vstack(feats).astype(np.float32), np.concatenate(labels).astype(np.int64)


def evaluate_rf_baseline(rf_model, files, split_name):
    """
    Run trained Random Forest over all patches in a split.
    Returns (metrics_dict, confusion_matrix).
    """
    print(f"  Evaluating Random Forest on {split_name} ({len(files)} patches) ...")
    t0 = time.time()

    all_preds, all_trues, all_probs = [], [], []

    for fp in files:
        try:
            X, y, _ = load_patch(fp)
            px = X.reshape(7, -1).T          # (65536, 7)
            pred  = rf_model.predict(px).astype(np.int64)
            proba = rf_model.predict_proba(px)    # (65536, n_cls)

            all_preds.append(pred)
            all_trues.append(y.ravel())
            all_probs.append(proba)
        except Exception as e:
            print(f"    Warning: skipping {fp}: {e}")

    if not all_preds:
        print(f"  ERROR: no patches evaluated for {split_name}")
        return None, None

    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)
    probs = np.vstack(all_probs)

    # Handle case where RF was trained on fewer than 4 classes
    if probs.shape[1] < N_CLASSES:
        full_probs = np.zeros((len(preds), N_CLASSES), dtype=np.float32)
        for i, cls in enumerate(rf_model.classes_):
            full_probs[:, int(cls)] = probs[:, i]
        probs = full_probs

    metrics = compute_flat_metrics(preds, trues, probs=probs)
    cm      = confusion_matrix(trues, preds, labels=list(range(N_CLASSES)))

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s — mIoU={metrics['mean_iou']:.4f}  "
          f"mDice={metrics['mean_dice']:.4f}")
    return metrics, cm


# =============================================================================
# SECTION 6  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Wildfire burn-severity baseline classifiers")
    parser.add_argument("--patch_dir",    type=str, default=PATCH_DIR)
    parser.add_argument("--splits_json",  type=str, default=SPLITS_JSON)
    parser.add_argument("--out_dir",      type=str, default=OUT_DIR)
    parser.add_argument("--rf",           action="store_true",
                        help="Also run Random Forest baseline")
    parser.add_argument("--max_train_px", type=int, default=500_000,
                        help="Max pixels sampled for RF training (default 500k)")
    parser.add_argument("--dnbr_thresholds", type=float, nargs=3,
                        default=DNBR_THRESHOLDS,
                        metavar=("T1", "T2", "T3"),
                        help="dNBR class boundaries (default: 0.10 0.27 0.44)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(FIG_DIR,      exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)

    fig_dir = os.path.join(args.out_dir, "figures")

    print("=" * 60)
    print("  Wildfire Burn-Severity Baselines")
    print("=" * 60)
    print(f"  patch_dir   : {args.patch_dir}")
    print(f"  splits_json : {args.splits_json}")
    print(f"  dNBR thresholds: {args.dnbr_thresholds}")
    print(f"  Random Forest  : {args.rf}")

    # ── Load splits ──────────────────────────────────────────────────────────
    print("\nLoading splits ...")
    train_files, val_files, test_files = load_splits(args.patch_dir,
                                                      args.splits_json)
    print(f"  Train: {len(train_files)}  Val: {len(val_files)}  "
          f"Test: {len(test_files)}")

    all_results   = {}   # will be written to baseline_metrics.json
    iou_plot_data = {}   # for per-class IoU bar chart

    # =========================================================================
    # BASELINE 1 — dNBR THRESHOLD
    # =========================================================================
    print("\n" + "─" * 50)
    print("  BASELINE 1: dNBR Threshold Classifier")
    print("─" * 50)

    dnbr_val_metrics, dnbr_val_cm, _, _ = evaluate_dnbr_baseline(
        val_files,  "Val")
    dnbr_test_metrics, dnbr_test_cm, _, _ = evaluate_dnbr_baseline(
        test_files, "Test")

    if dnbr_val_metrics:
        print_metrics(dnbr_val_metrics,  "Validation", "dNBR Threshold")
    if dnbr_test_metrics:
        print_metrics(dnbr_test_metrics, "Test",       "dNBR Threshold")

    # Confusion matrices
    if dnbr_val_cm is not None:
        plot_confusion(dnbr_val_cm,
                       "dNBR Threshold — Validation",
                       os.path.join(fig_dir, "baseline_confusion_val.png"))
    if dnbr_test_cm is not None:
        plot_confusion(dnbr_test_cm,
                       "dNBR Threshold — Test",
                       os.path.join(fig_dir, "baseline_confusion_test.png"))

    all_results["dnbr_threshold"] = {
        "thresholds": args.dnbr_thresholds,
        "val":  dnbr_val_metrics,
        "test": dnbr_test_metrics,
    }
    if dnbr_val_metrics:
        iou_plot_data["dNBR_val"]  = dnbr_val_metrics
    if dnbr_test_metrics:
        iou_plot_data["dNBR_test"] = dnbr_test_metrics

    # =========================================================================
    # BASELINE 2 — RANDOM FOREST  (optional)
    # =========================================================================
    if args.rf:
        print("\n" + "─" * 50)
        print("  BASELINE 2: Random Forest Classifier")
        print("─" * 50)

        try:
            from sklearn.ensemble import RandomForestClassifier

            print(f"  Collecting up to {args.max_train_px:,} training pixels ...")
            t0 = time.time()
            X_tr, y_tr = collect_rf_features(train_files,
                                              max_pixels=args.max_train_px)
            print(f"  Collected {len(y_tr):,} pixels in {time.time()-t0:.1f}s")
            print(f"  Class distribution: "
                  f"{np.bincount(y_tr, minlength=N_CLASSES)}")

            print("  Fitting Random Forest (200 trees, max_depth=20) ...")
            t0  = time.time()
            clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=SEED,
            )
            clf.fit(X_tr, y_tr)
            print(f"  Fitted in {time.time()-t0:.1f}s")

            # Feature importances
            feat_names = ["Red", "Green", "Blue", "NIR", "SWIR1", "NBR", "dNBR"]
            importances = clf.feature_importances_
            print("  Feature importances:")
            for fn, fi in sorted(zip(feat_names, importances),
                                  key=lambda x: -x[1]):
                print(f"    {fn:<8} {fi:.4f}")

            rf_val_metrics, rf_val_cm = evaluate_rf_baseline(
                clf, val_files,  "Val")
            rf_test_metrics, rf_test_cm = evaluate_rf_baseline(
                clf, test_files, "Test")

            if rf_val_metrics:
                print_metrics(rf_val_metrics,  "Validation", "Random Forest")
            if rf_test_metrics:
                print_metrics(rf_test_metrics, "Test",       "Random Forest")

            if rf_val_cm is not None:
                plot_confusion(rf_val_cm,
                               "Random Forest — Validation",
                               os.path.join(fig_dir, "rf_confusion_val.png"))
            if rf_test_cm is not None:
                plot_confusion(rf_test_cm,
                               "Random Forest — Test",
                               os.path.join(fig_dir, "rf_confusion_test.png"))

            all_results["random_forest"] = {
                "n_estimators": 200,
                "max_depth":    20,
                "n_train_px":   int(len(y_tr)),
                "feature_importances": dict(zip(feat_names,
                                                importances.tolist())),
                "val":  rf_val_metrics,
                "test": rf_test_metrics,
            }
            if rf_val_metrics:
                iou_plot_data["RF_val"]  = rf_val_metrics
            if rf_test_metrics:
                iou_plot_data["RF_test"] = rf_test_metrics

        except ImportError:
            print("  sklearn not available — skipping Random Forest")
        except Exception as e:
            import traceback
            print(f"  Random Forest failed: {e}")
            traceback.print_exc()

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    metrics_path = os.path.join(args.out_dir, "baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Baseline metrics saved -> {metrics_path}")

    # Per-class IoU comparison chart
    if iou_plot_data:
        plot_per_class_iou(
            iou_plot_data,
            os.path.join(fig_dir, "baseline_per_class_iou.png"))

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Baseline':<22} {'Split':<6} {'mIoU':>7} {'mDice':>7} {'PR-AUC':>8}")
    print("  " + "-" * 52)
    for name, res in all_results.items():
        for split in ("val", "test"):
            m = res.get(split)
            if m:
                print(f"  {name:<22} {split:<6} "
                      f"{m['mean_iou']:>7.4f} "
                      f"{m['mean_dice']:>7.4f} "
                      f"{m['mean_pr_auc']:>8.4f}")
    print("=" * 60)
    print(f"\n  Done. Results -> {metrics_path}")


if __name__ == "__main__":
    main()
