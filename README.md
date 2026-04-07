# Wildfire Burn-Scar Severity Mapping

**ENSF-617 | Group 40**  
Aparna Ayyalasomayajula · Radadiya Mansi Mukesh Bhai· (Group 40)

Automated burn-severity segmentation of post-fire Sentinel-2 imagery using deep learning.  
The pipeline downloads satellite data, trains four U-Net variants, evaluates them against physical baselines, and produces publication-quality figures and a full evaluation notebook.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [Data Download](#4-data-download)
5. [Running the Pipeline](#5-running-the-pipeline)
6. [Running Individual Stages](#6-running-individual-stages)
7. [Evaluation Notebook](#7-evaluation-notebook)
8. [Results](#8-results)
9. [Ablation Study](#9-ablation-study)
10. [Key Design Decisions](#10-key-design-decisions)

---

## 1. Project Overview

Wildfire burn-scar mapping is critical for post-fire land management and ecosystem recovery assessment. This project builds an end-to-end pipeline that:

- Downloads pre- and post-fire **Sentinel-2** imagery for 130 fires across the USA and Canada (2016–2023) via the **Microsoft Planetary Computer STAC API**
- Constructs **256×256 px patches** at 10 m resolution with 7 input channels: `R, G, B, NIR, SWIR1, NBR_pre, dNBR`
- Trains four **U-Net variants** (UNet, ResUNet, AttentionUNet, UNet++) on MTBS/NBAC 4-class severity labels
- Evaluates against a **dNBR threshold baseline** and a **Random Forest baseline**
- Performs **ablation studies**, **cross-region transfer** evaluation, **inference benchmarking**, and **Grad-CAM explainability**

**Severity classes (MTBS scheme):**

| Class | dNBR range | Description |
|---|---|---|
| 0 — Unburned/Low | < 0.100 | No detectable burn or very low severity |
| 1 — Moderate | 0.100 – 0.270 | Partial canopy/understorey loss |
| 2 — High | 0.270 – 0.440 | Significant canopy mortality |
| 3 — Very High | > 0.440 | Near-complete stand replacement |

---

## 2. Repository Structure

```
wildfire_burn_scar_mapping/
│
├── sentinel2_dataset_v2.py     # Sentinel-2 download, cloud masking, patch creation
├── unet_training_v2.py         # Model definitions + training loop (all 4 architectures)
├── baseline.py                 # dNBR threshold and Random Forest baselines
├── ablation.py                 # Controlled ablation experiments (A, B, C)
├── cross_region_eval.py        # USA ↔ Canada zero-shot transfer evaluation
├── inference_analysis.py       # Inference timing and GPU memory benchmarking
├── explainability.py           # Grad-CAM and attention gate visualisations
├── generate_figures.py         # Publication-quality result figures
├── gen_fig1.py                 # Figure 1: representative patch visualisation
├── run_full_pipeline.py        # Master orchestration script (runs all stages)
│
├── evaluation_notebook.ipynb   # Full interactive evaluation notebook
│
├── pipeline_job1_train.slurm   # SLURM job: Stage 1 — model training
├── pipeline_job2_eval.slurm    # SLURM job: Stages 2–7 — evaluation + figures
├── gpu_download_only.slurm     # SLURM job: dataset download only
├── check_server_resources.sh   # Check GPU/CPU/memory availability on HPC
├── monitor.sh                  # Monitor running SLURM jobs
│
├── .gitignore
└── README.md
```

**Generated at runtime (not in repo):**

```
data/sentinel2/patches_v2/      # .npz patch files (~15 k patches)
data/sentinel2/splits_v2.json   # Train/val/test fire split
outputs/sentinel2_results_v2/   # Model checkpoints, metrics JSONs, figures
outputs/ablation/               # Ablation experiment checkpoints + summary
metrics/                        # Cross-region and inference metrics
explainability_samples/         # Grad-CAM and attention map images
pipeline_log/                   # Timestamped pipeline execution logs
```

---

## 3. Environment Setup

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (16 GB+ VRAM recommended for batch size 8)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Create conda environment

```bash
conda create -n gpuenv python=3.9 -y
conda activate gpuenv
```

### Install dependencies

```bash
# PyTorch (CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Remote sensing + geospatial
pip install rasterio geopandas pyproj shapely fiona

# Planetary Computer STAC API
pip install pystac-client planetary-computer

# Machine learning + analysis
pip install scikit-learn pandas numpy

# Visualisation
pip install matplotlib

# Jupyter
pip install jupyter notebook
```

### Environment variables

The pipeline derives all paths from a single root variable. Set it before running any script:

```bash
export WILDFIRE_BASE_DIR="/path/to/wildfire_burn_scar_mapping"
```

On HPC (SLURM), this is injected automatically by the job scripts.

---

## 4. Data Download

Downloads Sentinel-2 scenes for 130 MTBS/NBAC fires (2016–2023) and builds patch datasets.

### Via SLURM (recommended on HPC)

```bash
sbatch gpu_download_only.slurm
```

### Locally

```bash
python sentinel2_dataset_v2.py \
    --n_usa 80 --n_canada 50 \
    --min_area_ha 2000 \
    --year_start 2016 --year_end 2023 \
    --max_cloud_frac 0.70 \
    --skip_existing
```

**Expected output:**

```
data/sentinel2/patches_v2/
    USA/<fire_id>/<fire_id>_<patch_id>.npz
    CAN/<fire_id>/<fire_id>_<patch_id>.npz
data/sentinel2/splits_v2.json
```

**Dataset statistics:**

| Split | Years | Fires | Patches |
|---|---|---|---|
| Training | 2016–2020 | 104 | 11,723 |
| Validation | 2021 | 14 | 2,560 |
| Test | 2022–2023 | 12 | 1,047 |
| **Total** | **2016–2023** | **130** | **15,330** |

---

## 5. Running the Pipeline

The full pipeline runs all 7 stages in sequence. On HPC, submit as two dependent SLURM jobs.

### On HPC (SLURM) — recommended

```bash
# Stage 1: Train all four models (~10–14 h)
sbatch pipeline_job1_train.slurm

# Stage 2–7: Evaluation, ablation, figures (~14–18 h)
# Submit after job 1 completes
sbatch pipeline_job2_eval.slurm
```

### Locally (all stages)

```bash
conda activate gpuenv
export WILDFIRE_BASE_DIR="$(pwd)"

python run_full_pipeline.py
```

### Locally (selective stages)

```bash
# Train only
python run_full_pipeline.py --stages train

# Resume from ablation onwards
python run_full_pipeline.py --start_stage ablation

# Run specific stages
python run_full_pipeline.py --stages baseline,inference,figures

# Dry-run (print commands without executing)
python run_full_pipeline.py --dry_run
```

### Common options

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 60 | Training epochs per model |
| `--batch` | 8 | Batch size |
| `--lr` | 3e-4 | Learning rate |
| `--train_models` | all | Models to train: `unet,resunet,attention,unetpp` |
| `--baseline_rf` | off | Include Random Forest baseline (slower) |
| `--n_explain_samples` | 8 | Grad-CAM samples per model |
| `--skip_errors` | off | Continue pipeline if a stage fails |

---

## 6. Running Individual Stages

Each script can also be run independently.

### Training

```bash
python unet_training_v2.py \
    --model resunet \
    --epochs 60 \
    --batch 8 \
    --lr 3e-4
```

### Baselines

```bash
python baseline.py --rf   # includes Random Forest (omit flag for dNBR only)
```

### Ablation

```bash
python ablation.py --experiments A,B,C --epochs 60 --batch 8
```

### Cross-region evaluation

```bash
python cross_region_eval.py --archs all
```

### Inference benchmarking

```bash
python inference_analysis.py --archs all --n_patches 100
```

### Explainability (Grad-CAM)

```bash
python explainability.py --archs all --n_samples 8
```

### Figures

```bash
python generate_figures.py
python gen_fig1.py
```

---

## 7. Evaluation Notebook

An interactive notebook covering all evaluation sections is provided.

### Open locally

```bash
conda activate gpuenv
cd wildfire_burn_scar_mapping
jupyter notebook evaluation_notebook.ipynb
```

### Open on HPC (SSH tunnel)

```bash
# On the HPC server:
conda activate gpuenv
jupyter notebook --no-browser --port=8888

# On your local machine (new terminal):
ssh -L 8888:localhost:8888 <your_username>@<hpc_hostname>

# Then open in browser:
# http://localhost:8888
```

The notebook covers:

| Section | Content |
|---|---|
| 0 | Setup and data loading |
| 1 | Dataset and split summary |
| 2 | Overall model comparison table and bar chart |
| 3 | Training dynamics (loss + mIoU curves) |
| 4 | Per-class IoU analysis |
| 5 | Confusion matrix analysis |
| 6 | Precision–Recall curves |
| 7 | Per-fire performance breakdown |
| 8 | Ablation study |
| 9 | Baseline comparison |
| 10 | Qualitative predictions |
| 11 | Summary and conclusions |

---

## 8. Results

All metrics are reported on the held-out test set (2022–2023 fires, never seen during training).

### Overall model comparison

| Model | mIoU | mDice | PR-AUC | Parameters | Best Epoch |
|---|---|---|---|---|---|
| **ResUNet** | **0.6136** | **0.7512** | 0.8462 | 8.11 M | 8 |
| UNet++ | 0.5975 | 0.7317 | **0.8589** | 9.05 M | 20 |
| AttentionUNet | 0.5949 | 0.7284 | 0.8572 | 7.85 M | 18 |
| UNet | 0.5701 | 0.7100 | 0.8403 | 7.76 M | 20 |
| dNBR Threshold | 0.6625 | 0.7913 | — | — | — |
| Random Forest | 0.5978 | 0.7365 | — | — | — |

**ResUNet** achieves the highest mIoU and mDice among deep models, converging at epoch 8 — 60% faster than UNet. The dNBR threshold baseline outperforms all deep models on aggregate mIoU due to its strong physical prior (dNBR directly encodes burn severity), but cannot encode spatial context or class boundaries.

### ResUNet per-class performance (test set)

| Class | IoU | Dice |
|---|---|---|
| Unburned/Low | 0.8231 | 0.9030 |
| Moderate | 0.6503 | 0.7881 |
| High | 0.5379 | 0.6995 |
| Very High | 0.4432 | 0.6142 |

**Severity-difficulty gradient:** IoU decreases from Unburned/Low → Very High due to increasing class rarity (15:1 pixel imbalance) and spectral overlap at severity boundaries.

### Systematic failure modes (all architectures)

1. **High ↔ Very High confusion** — the dNBR = 0.44 label boundary has no unique spectral signature on either side
2. **Moderate → Unburned/Low confusion** — mixed spectral responses at low-intensity fire perimeter margins
3. **Very High under-prediction** — sparse training signal (~15% of pixels) and small burn patch extents

All three failure modes are consistent across architectures, confirming errors are **label/data-driven**, not model-capacity-limited.

---

## 9. Ablation Study

Controlled ablation on ResUNet — one design choice removed at a time, all other hyperparameters identical.

| ID | Configuration | mIoU | Change |
|---|---|---|---|
| Full | 7-ch + Augmentation + FocalDiceLoss | 0.6136 | — |
| A | No spectral indices (5-ch: R,G,B,NIR,SWIR1) | 0.1800 | −70.6% |
| B | No data augmentation | 0.6182 | −0.8% |
| C | CrossEntropyLoss instead of FocalDiceLoss | 0.5219 | −15.0% |

**Key findings:**

- **Spectral indices (NBR, dNBR) are the primary discriminative signal.** Removing them collapses mIoU from 0.6136 to 0.1800 — a 70.6% relative drop. The raw Sentinel-2 bands alone are insufficient to distinguish severity classes.
- **FocalDiceLoss is critical at 15:1 class imbalance.** CrossEntropy loses 15 percentage points because it treats all classes equally regardless of pixel frequency.
- **Data augmentation has minimal impact.** The forward-chaining temporal split provides sufficient regularisation through natural scene diversity.

---

## 10. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Temporal split | Forward-chaining (train ≤2020, val 2021, test ≥2022) | Prevents temporal data leakage between fire events |
| Loss function | FocalDiceLoss (0.5×Focal + 0.5×Dice) | Handles 15:1 pixel imbalance; Focal down-weights easy negatives |
| Input channels | 7 (R,G,B,NIR,SWIR1,NBR_pre,dNBR) | dNBR and NBR encode burn severity directly from physics |
| Class weights | Inverse frequency, clipped [0.1, 50.0] | Prevents extreme upweighting of Very High class |
| Checkpoint selection | EMA-smoothed val mIoU (α=0.3) | Avoids saving a lucky high-variance epoch as best |
| Cloud masking | SCL classes {0,1,2,3,8,9,10,11} masked; ≥60% valid pixels | Removes cloud, shadow, and saturated pixels per patch |
| Patch validity | ≥10% burned pixels per patch | Prevents training on background-only patches |
| Architecture | Pre-activation residual blocks (ResUNet) | Enables deeper gradient flow; better than attention gates for spectral-index tasks |
