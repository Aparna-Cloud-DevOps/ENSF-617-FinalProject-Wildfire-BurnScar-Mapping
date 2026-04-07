"""
run_full_pipeline.py
====================
Master orchestration script for the wildfire burn-severity segmentation pipeline.

Runs every stage end-to-end in order and logs each stage to a timestamped log file.

Pipeline stages
---------------
  1. train        — unet_training_v2.py (UNet, ResUNet, AttentionUNet, UNet++)
  2. baseline     — baseline.py (dNBR threshold + Random Forest)
  3. ablation     — ablation.py (experiments A, B, C)
  4. cross_region — cross_region_eval.py (USA ↔ Canada zero-shot transfer)
  5. inference    — inference_analysis.py (timing, memory, model size)
  6. explain      — explainability.py (Grad-CAM + attention maps)
  7. figures      — generate_figures.py (publication-quality result figures)

Usage
-----
  # Run all stages
  python run_full_pipeline.py

  # Resume from a specific stage (skips earlier completed stages)
  python run_full_pipeline.py --start_stage baseline

  # Run only selected stages
  python run_full_pipeline.py --stages train,inference,figures

  # Dry-run: print commands without executing
  python run_full_pipeline.py --dry_run

SLURM
-----
  sbatch run_full_pipeline.slurm

Notes
-----
  - No existing module is modified.
  - Every stage is called via subprocess so each runs in its own Python process.
  - If a stage fails the script logs the error and stops unless --skip_errors is set.
  - Intermediate per-stage timing is written to pipeline_log/pipeline_summary.json.
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Project root — every path is derived from here
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.environ.get(
    "WILDFIRE_BASE_DIR",
    os.path.dirname(os.path.abspath(__file__)),
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared path constants (must mirror the individual scripts)
# ─────────────────────────────────────────────────────────────────────────────
PATCH_DIR    = os.path.join(BASE, "data", "sentinel2", "patches_v2")
SPLITS_JSON  = os.path.join(BASE, "data", "sentinel2", "splits_v2.json")
RESULTS_DIR  = os.path.join(BASE, "outputs", "sentinel2_results_v2")
ABLATION_DIR = os.path.join(BASE, "outputs", "ablation")
METRICS_DIR  = os.path.join(BASE, "metrics")
EXPL_DIR     = os.path.join(BASE, "explainability_samples")
LOG_DIR      = os.path.join(BASE, "pipeline_log")

# Best-model checkpoint produced by the training stage
CKPT_PATH    = os.path.join(RESULTS_DIR, "best_model.pth")
STATS_PATH   = os.path.join(RESULTS_DIR, "channel_stats.npz")

# ─────────────────────────────────────────────────────────────────────────────
# Stage ordering
# ─────────────────────────────────────────────────────────────────────────────
ALL_STAGES = [
    "train",
    "baseline",
    "ablation",
    "cross_region",
    "inference",
    "explain",
    "figures",
]


# =============================================================================
# SECTION 1 — Logging setup
# =============================================================================

def setup_logging(log_dir: str) -> logging.Logger:
    """
    Configure a logger that writes to both stdout and a timestamped file
    in log_dir/pipeline_<timestamp>.log.
    """
    os.makedirs(log_dir, exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(log_dir, f"pipeline_{ts}.log")

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info("Pipeline log: %s", logfile)
    return logger


# =============================================================================
# SECTION 2 — Stage runner
# =============================================================================

def run_stage(
    name: str,
    cmd: list,
    logger: logging.Logger,
    dry_run: bool = False,
    skip_errors: bool = False,
) -> dict:
    """
    Execute one pipeline stage as a subprocess.

    Parameters
    ----------
    name        : human-readable stage name
    cmd         : argv list, e.g. ['python', '-u', 'baseline.py', '--rf']
    logger      : Logger instance
    dry_run     : if True, print the command but do not execute
    skip_errors : if True, log failures and continue; otherwise re-raise

    Returns
    -------
    dict with keys: stage, status, duration_s, returncode, cmd
    """
    banner = "─" * 60
    logger.info("")
    logger.info(banner)
    logger.info("STAGE : %s", name.upper())
    logger.info("CMD   : %s", " ".join(cmd))
    logger.info(banner)

    result = {
        "stage":      name,
        "cmd":        " ".join(cmd),
        "status":     "skipped",
        "returncode": None,
        "duration_s": 0.0,
    }

    if dry_run:
        logger.info("[DRY-RUN] skipping execution")
        result["status"] = "dry_run"
        return result

    t0 = time.perf_counter()
    try:
        # Stream output line-by-line so training epoch logs appear in real time.
        # stderr=STDOUT merges both streams so error tracebacks are also captured.
        # WILDFIRE_BASE_DIR is injected into the child env so every script
        # resolves its own paths consistently regardless of working directory.
        # bufsize=1 forces line-buffering — each epoch line is written immediately.
        proc = subprocess.Popen(
            cmd,
            cwd=BASE,
            env={**os.environ, "WILDFIRE_BASE_DIR": BASE,
                 "PYTHONIOENCODING": "utf-8"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,          # line-buffered output (shows progress in real time)
        )
        for line in proc.stdout:
            line = line.rstrip()
            logger.info("  | %s", line)
        proc.wait()  # block until subprocess finishes; return code is set after this

        duration = time.perf_counter() - t0
        result["duration_s"] = round(duration, 2)
        result["returncode"] = proc.returncode

        if proc.returncode == 0:
            result["status"] = "success"
            logger.info("DONE  : %s  (%.1fs)", name.upper(), duration)
        else:
            result["status"] = "failed"
            logger.error(
                "FAILED: %s returned exit code %d  (%.1fs)",
                name.upper(), proc.returncode, duration,
            )
            if not skip_errors:
                raise RuntimeError(
                    f"Stage '{name}' failed with return code {proc.returncode}."
                )

    except RuntimeError:
        raise
    except Exception as exc:
        duration = time.perf_counter() - t0
        result["duration_s"] = round(duration, 2)
        result["status"]     = "error"
        logger.exception("ERROR in stage '%s': %s", name, exc)
        if not skip_errors:
            raise

    return result


# =============================================================================
# SECTION 3 — Stage command builders
# =============================================================================

def _py(script: str) -> list:
    """Return ['python', '-u', '<base>/<script>']."""
    return [sys.executable, "-u", os.path.join(BASE, script)]


def build_stage_commands(args) -> dict:
    """
    Return a dict mapping stage name → argv list for subprocess.run().
    All paths are absolute so stages work regardless of cwd.
    """
    cmds = {}

    # ── Stage 1: Training ─────────────────────────────────────────────────────
    cmds["train"] = _py("unet_training_v2.py") + [
        "--model",       args.train_models,
        "--epochs",      str(args.epochs),
        "--batch",       str(args.batch),
        "--lr",          str(args.lr),
        "--patch_dir",   PATCH_DIR,
        "--splits_json", SPLITS_JSON,
        "--out_dir",     RESULTS_DIR,
    ]

    # ── Stage 2: Baseline ─────────────────────────────────────────────────────
    baseline_cmd = _py("baseline.py") + [
        "--patch_dir",   PATCH_DIR,
        "--splits_json", SPLITS_JSON,
        "--out_dir",     RESULTS_DIR,
    ]
    if args.baseline_rf:
        baseline_cmd.append("--rf")
    cmds["baseline"] = baseline_cmd

    # ── Stage 3: Ablation ─────────────────────────────────────────────────────
    cmds["ablation"] = _py("ablation.py") + [
        "--run",         "A,B,C",
        "--epochs",      str(args.ablation_epochs),
        "--batch",       str(args.batch),
        "--patch_dir",   PATCH_DIR,
        "--splits_json", SPLITS_JSON,
        "--out_dir",     ABLATION_DIR,
    ]

    # ── Stage 4: Cross-region evaluation ─────────────────────────────────────
    cmds["cross_region"] = _py("cross_region_eval.py") + [
        "--checkpoint_dir", RESULTS_DIR,
        "--source_region",  "USA",
        "--target_region",  "CAN",
        "--both_directions",
        "--stats_path",     STATS_PATH,
        "--metrics_dir",    METRICS_DIR,
        "--batch",          str(args.batch),
    ]

    # ── Stage 5: Inference analysis ───────────────────────────────────────────
    cmds["inference"] = _py("inference_analysis.py") + [
        "--checkpoint_dir", RESULTS_DIR,
        "--patch_dir",      PATCH_DIR,
        "--splits_json",    SPLITS_JSON,
        "--metrics_dir",    METRICS_DIR,
        "--n_patches",      "100",
        "--batch",          str(args.batch),
        "--archs",          "all",
    ]

    # ── Stage 6: Explainability ───────────────────────────────────────────────
    cmds["explain"] = _py("explainability.py") + [
        "--checkpoint_dir", RESULTS_DIR,
        "--patch_dir",      PATCH_DIR,
        "--splits_json",    SPLITS_JSON,
        "--metrics_dir",    METRICS_DIR,
        "--out_dir",        EXPL_DIR,
        "--n_samples",      str(args.n_explain_samples),
        "--archs",          "all",
    ]

    # ── Stage 7: Figures ─────────────────────────────────────────────────────
    cmds["figures"] = _py("generate_figures.py")

    return cmds


# =============================================================================
# SECTION 4 — Pre-flight checks
# =============================================================================

def preflight_checks(logger: logging.Logger) -> bool:
    """
    Verify that data and checkpoint prerequisites exist before each stage.
    Returns True if all required files are present, False otherwise.
    Logs warnings but does NOT abort (caller decides).
    """
    ok = True

    # Data patches
    npz_count = len(list(Path(PATCH_DIR).rglob("*.npz")))
    if npz_count == 0:
        logger.warning(
            "No .npz patches found under %s. "
            "The 'train' stage will fail. "
            "Run gpu_download_only.slurm first to build the dataset.", PATCH_DIR
        )
        ok = False
    else:
        logger.info("Pre-flight: found %d patches under %s", npz_count, PATCH_DIR)

    # Splits JSON
    if not os.path.exists(SPLITS_JSON):
        logger.warning("splits_v2.json not found — training will use random split.")

    return ok


# =============================================================================
# SECTION 5 — Summary helpers
# =============================================================================

def _fmt_duration(seconds: float) -> str:
    """Format seconds as 'Xh Ym Zs' for readability."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def log_summary(results: list, logger: logging.Logger, log_dir: str):
    """
    Print a final summary table and write pipeline_log/pipeline_summary.json.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("  PIPELINE SUMMARY")
    logger.info("=" * 60)

    total = 0.0
    for r in results:
        status_sym = {"success": "✓", "failed": "✗",
                      "error": "!", "dry_run": "~", "skipped": "-"}.get(
            r["status"], "?"
        )
        dur = _fmt_duration(r["duration_s"])
        logger.info(
            "  %s  %-14s  %s  (%s)",
            status_sym, r["stage"], r["status"].upper(), dur,
        )
        total += r["duration_s"]

    logger.info("")
    logger.info("  Total wall time: %s", _fmt_duration(total))
    logger.info("=" * 60)

    # Write JSON
    summary_path = os.path.join(log_dir, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {"total_duration_s": round(total, 2),
             "total_duration_fmt": _fmt_duration(total),
             "stages": results},
            f, indent=2,
        )
    logger.info("Summary written: %s", summary_path)


# =============================================================================
# SECTION 6 — main()
# =============================================================================

def main():
    """
    Parse CLI arguments, run requested pipeline stages in order, and log results.
    """
    parser = argparse.ArgumentParser(
        description="Run the full wildfire burn-severity segmentation pipeline.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Stage selection ───────────────────────────────────────────────────────
    parser.add_argument(
        "--stages", type=str, default="all",
        help=(
            "Comma-separated list of stages to run, or 'all'.\n"
            f"Available: {', '.join(ALL_STAGES)}\n"
            "Example: --stages train,baseline,figures"
        ),
    )
    parser.add_argument(
        "--start_stage", type=str, default=None,
        help=(
            "Skip all stages before this one and resume from here.\n"
            "Example: --start_stage ablation"
        ),
    )

    # ── Training hyperparameters ──────────────────────────────────────────────
    parser.add_argument("--train_models",    type=str,   default="all",
                        help="Comma-separated models to train, or 'all' (default). "
                             "Use to resume after OOM: e.g. resunet,attention,unetpp")
    parser.add_argument("--epochs",          type=int,   default=60,
                        help="Epochs for main training (default 60)")
    parser.add_argument("--ablation_epochs", type=int,   default=60,
                        help="Epochs for each ablation experiment (default 60)")
    parser.add_argument("--batch",           type=int,   default=8,
                        help="Batch size (default 8)")
    parser.add_argument("--lr",              type=float, default=3e-4,
                        help="Learning rate (default 3e-4)")

    # ── Baseline options ──────────────────────────────────────────────────────
    parser.add_argument("--baseline_rf",     action="store_true",
                        help="Include Random Forest baseline (slower)")

    # ── Explainability options ────────────────────────────────────────────────
    parser.add_argument("--n_explain_samples", type=int, default=8,
                        help="Grad-CAM samples per model (5–10, default 8)")

    # ── Execution control ─────────────────────────────────────────────────────
    parser.add_argument("--skip_errors",  action="store_true",
                        help="Log failures and continue instead of aborting")
    parser.add_argument("--dry_run",      action="store_true",
                        help="Print commands without executing them")
    parser.add_argument("--log_dir",      type=str, default=LOG_DIR,
                        help="Directory for log files")

    args = parser.parse_args()

    # ── Resolve stage list ────────────────────────────────────────────────────
    if args.stages.strip().lower() == "all":
        stages_to_run = list(ALL_STAGES)
    else:
        stages_to_run = [s.strip().lower() for s in args.stages.split(",")]
        invalid = [s for s in stages_to_run if s not in ALL_STAGES]
        if invalid:
            print(f"ERROR: unknown stages: {invalid}. "
                  f"Choose from {ALL_STAGES}")
            sys.exit(1)

    # Apply --start_stage
    if args.start_stage:
        start = args.start_stage.strip().lower()
        if start not in ALL_STAGES:
            print(f"ERROR: --start_stage '{start}' not in {ALL_STAGES}")
            sys.exit(1)
        start_idx    = ALL_STAGES.index(start)
        stages_to_run = [s for s in stages_to_run
                         if ALL_STAGES.index(s) >= start_idx]

    # ── Setup ─────────────────────────────────────────────────────────────────
    logger = setup_logging(args.log_dir)

    logger.info("=" * 60)
    logger.info("  Wildfire Burn Severity — Full Pipeline")
    logger.info("  %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)
    logger.info("Base dir   : %s", BASE)
    logger.info("Stages     : %s", " → ".join(stages_to_run))
    logger.info("Dry run    : %s", args.dry_run)
    logger.info("Skip errors: %s", args.skip_errors)

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    if not args.dry_run:
        preflight_checks(logger)

    # ── Build commands ────────────────────────────────────────────────────────
    all_cmds = build_stage_commands(args)

    # ── Execute stages ────────────────────────────────────────────────────────
    results   = []
    pipeline_start = time.perf_counter()

    for stage in stages_to_run:
        if stage not in all_cmds:
            logger.warning("No command defined for stage '%s', skipping.", stage)
            results.append({"stage": stage, "status": "skipped",
                            "returncode": None, "duration_s": 0.0, "cmd": ""})
            continue

        result = run_stage(
            name        = stage,
            cmd         = all_cmds[stage],
            logger      = logger,
            dry_run     = args.dry_run,
            skip_errors = args.skip_errors,
        )
        results.append(result)

        # Abort on failure unless --skip_errors
        if result["status"] == "failed" and not args.skip_errors:
            logger.error("Pipeline aborted at stage '%s'.", stage)
            break

    pipeline_end = time.perf_counter()
    logger.info("")
    logger.info("Pipeline wall time: %s",
                _fmt_duration(pipeline_end - pipeline_start))

    # ── Final summary ─────────────────────────────────────────────────────────
    log_summary(results, logger, args.log_dir)

    # Exit with non-zero code if any stage failed
    n_failed = sum(1 for r in results if r["status"] in ("failed", "error"))
    if n_failed > 0:
        logger.error("%d stage(s) failed.", n_failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
