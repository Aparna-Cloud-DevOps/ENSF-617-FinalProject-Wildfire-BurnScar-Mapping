#!/bin/bash
#SBATCH --job-name=wf_train_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=wf_train_all_%j.out
#SBATCH --error=wf_train_all_%j.err

# ──────────────────────────────────────────────────────────────────────────────
# pipeline_job1_train.slurm
# Stage 1 only: train all four deep models (UNet, ResUNet, AttentionUNet, UNet++).
# Runtime estimate: ~10–14h with early stopping (60 epochs max per model).
#
# After this job completes, submit pipeline_job2_eval.slurm.
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
echo "Stage    : TRAIN (ResUNet + AttentionUNet + UNet++  — UNet already done)"
echo "======================================="

python -u run_full_pipeline.py \
    --stages        train \
    --train_models  resunet,attention,unetpp \
    --epochs        20 \
    --batch         8 \
    --lr            3e-4 \
    --log_dir       "$PROJECT_DIR/pipeline_log"

echo ""
echo "======================================="
echo "Training complete: $(date)"
echo "Checkpoints: $PROJECT_DIR/outputs/sentinel2_results_v2/"
echo ""
echo "Next step: sbatch pipeline_job2_eval.slurm"
echo "======================================="
