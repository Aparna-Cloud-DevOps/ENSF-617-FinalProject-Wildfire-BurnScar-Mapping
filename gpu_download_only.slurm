#!/bin/bash
#SBATCH --job-name=wf_download
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=wf_download_%j.out
#SBATCH --error=wf_download_%j.err

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
echo "Start    : $(date)"
echo "======================================="

echo "=== STEP 1: Building v2 Sentinel-2 dataset ==="
python -u sentinel2_dataset_v2.py \
    --n_usa 80 --n_canada 50 \
    --min_area_ha 2000 \
    --year_start 2016 --year_end 2023 \
    --max_cloud_frac 0.70 \
    --skip_existing

echo ""
echo "--- Patch inventory ---"
python -c "
import pathlib, collections, json
files = list(pathlib.Path('data/sentinel2/patches_v2').rglob('*.npz'))
print(f'Total patches: {len(files)}')
usa = [f for f in files if '/USA/' in str(f)]
can = [f for f in files if '/CAN/' in str(f)]
print(f'  USA patches: {len(usa)}')
print(f'  CAN patches: {len(can)}')
sp = json.load(open('data/sentinel2/splits_v2.json'))
print(f'Train fires: {len(sp[\"train\"])}  Val: {len(sp[\"val\"])}  Test: {len(sp[\"test\"])}')
"

echo ""
echo "======================================="
echo "DOWNLOAD DONE: $(date)"
echo "Next: sbatch gpu_train_v2.slurm"
echo "======================================="
