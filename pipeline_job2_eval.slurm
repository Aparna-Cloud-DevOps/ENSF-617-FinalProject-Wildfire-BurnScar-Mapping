#!/bin/bash
#SBATCH --job-name=wf_eval_figures
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=wf_eval_figures_%j.out
#SBATCH --error=wf_eval_figures_%j.err

# ──────────────────────────────────────────────────────────────────────────────
# pipeline_job2_eval.slurm
# Stages 2–7: baseline, ablation, cross-region, inference, explainability, figures.
# Runtime estimate: ~14–18h (ablation A,B,C dominates at ~4–6h each).
#
# PREREQUISITE: pipeline_job1_train.slurm must have completed successfully.
# Checkpoints must exist in outputs/sentinel2_results_v2/ before running this.
# ──────────────────────────────────────────────────────────────────────────────

set -e
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

module load cuda/12.6.2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gpuenv

PROJECT_DIR="/home/aparna.ayyalasomayaj/wildfire_burn_scar_mapping"
cd "$PROJECT_DIR"
export WILDFIRE_BASE_DIR="$PROJECT_DIR"

echo "======================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start    : $(date)"
echo "Stage    : EVAL+FIGURES (baseline/ablation/cross_region/inference/explain/figures)"
echo "======================================="

# Verify all four training checkpoints exist before proceeding
RESULTS="$PROJECT_DIR/outputs/sentinel2_results_v2"
MISSING=0
for CKPT_FILE in \
    "$RESULTS/best_model.pth" \
    "$RESULTS/resunet_best_model.pth" \
    "$RESULTS/attention_best_model.pth" \
    "$RESULTS/unetpp_best_model.pth"; do
    if [ ! -f "$CKPT_FILE" ]; then
        echo "ERROR: missing checkpoint: $CKPT_FILE"
        MISSING=$((MISSING + 1))
    else
        echo "OK: $CKPT_FILE"
    fi
done

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "ERROR: $MISSING checkpoint(s) missing."
    echo "Ensure pipeline_job1_train.slurm completed successfully."
    exit 1
fi

if [ ! -f "$RESULTS/channel_stats.npz" ]; then
    echo "ERROR: channel_stats.npz not found in $RESULTS"
    echo "This file is required for cross-region eval and inference analysis."
    exit 1
fi

echo ""
echo "All checkpoints and channel_stats.npz verified."
echo ""

python -u run_full_pipeline.py \
    --stages       baseline,ablation,cross_region,inference,explain,figures \
    --ablation_epochs 10 \
    --batch        8 \
    --baseline_rf \
    --n_explain_samples 5 \
    --log_dir      "$PROJECT_DIR/pipeline_log"

echo ""
echo "======================================="
echo "Evaluation complete: $(date)"
echo ""
echo "Key outputs:"
echo "  Baseline        : $PROJECT_DIR/outputs/sentinel2_results_v2/baseline_metrics.json"
echo "  Ablation        : $PROJECT_DIR/outputs/ablation/ablation_summary.json"
echo "  Cross-region    : $PROJECT_DIR/metrics/cross_region_metrics.json"
echo "  Inference       : $PROJECT_DIR/metrics/inference_metrics.json"
echo "  Explainability  : $PROJECT_DIR/explainability_samples/"
echo "  Figures         : $PROJECT_DIR/outputs/sentinel2_results_v2/figures/"
echo "======================================="
