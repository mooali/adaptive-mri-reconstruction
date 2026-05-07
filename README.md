# MRI Slice Interpolation with Deep Learning

> **The one-line pitch:** MRI scanners are slow. What if you acquired only half the data and used a neural network to reconstruct the missing slices — and could prove it works, understood why it works, and tested it on two different brain MRI datasets?

---

## The Problem

A full brain MRI scan acquires hundreds of image slices, one at a time. This takes time — and in clinical settings, **scan time directly affects patient comfort, scanner availability, and cost**.

One way to speed things up is to skip every other slice during acquisition. You end up with half the data in half the time. The challenge: can you recover the missing slices well enough that a doctor wouldn’t know the difference?

---

## The Idea

```
Real scan (slow)          Accelerated scan (fast)       This project
─────────────────         ───────────────────────       ────────────────────────
slice 1  ✓                slice 1  ✓                    slice 1  ✓  (acquired)
slice 2  ✓                slice 2  ✗  ← missing         slice 2  ← reconstructed by U-Net
slice 3  ✓                slice 3  ✓                    slice 3  ✓  (acquired)
slice 4  ✓                slice 4  ✗  ← missing         slice 4  ← reconstructed by U-Net
slice 5  ✓                slice 5  ✓                    slice 5  ✓  (acquired)
...                       ...                           ...
```

Given two neighboring acquired slices, the model learns to predict the missing one between them. This is called **2.5D interpolation** — 2D slices with 3D context from their neighbors.

---

## What Was Built

Three pipeline stages plus an ablation study:

```
src/preprocess.py         IXI NIfTI scans → resize → normalise → 2.5D pairs + baselines
src/preprocess_brats.py   BraTS2020 H5 slices → 2.5D reconstruction pairs
         ↓
src/train.py              U-Net + PlainCNN (ablation), --model unet|plaincnn,
                          --data ixi|brats, evaluation: PSNR / SSIM / MAE / FG-MAE
         ↓
src/explainability.py     Grad-CAM + Integrated Gradients + Slice Contribution +
                          Occlusion Sensitivity + MC Uncertainty (50 samples)
```

All configuration (paths, hyperparameters) lives in one place: `src/config.py`.

---

## Datasets

### IXI Brain Dataset (primary)
581 T1-weighted MRI volumes from 3 London hospitals (Guy’s, Hammersmith, IOP).
Publicly available, healthy subjects only.

- Resized to 256 × 256 per slice, normalised to [0, 1]
- 2.5D input–target pairs (every other slice skipped, ×2 acceleration)
- **42,816 total pairs** — split 70 / 15 / 15 (train / val / test)

### BraTS2020 (cross-dataset validation)
57,195 pre-extracted H5 slices, T1 channel, skull-stripped.

- Same 2.5D pipeline and preprocessing as IXI
- **28,413 reconstruction pairs**
- Skull-stripped backgrounds (~70–80 % zeros) inflate raw PSNR, so results are
  reported as **FG-MAE** (foreground MAE, pixels where ground truth > 0.02)

---

## The Model — U-Net

A U-Net is an encoder–decoder network with skip connections. Originally designed for
medical image segmentation, it excels at any task that produces a full-resolution image
from another full-resolution image.

```
Input: [left_slice, right_slice]   (2 × 256 × 256)
          │
    ┌─────▼─────┐
    │  Encoder  │   32 → 64 → 128 → 256 channels
    │           │   each level: 2× Conv + BN + ReLU → MaxPool
    └─────┬─────┘
          │  skip connections (fine spatial detail preserved)
    ┌─────▼─────┐
    │ Bottleneck│   512 channels, 16 × 16 spatial resolution
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │  Decoder  │   256 → 128 → 64 → 32 channels
    │           │   each level: Upsample + concat skip + 2× Conv
    └─────┬─────┘
          ▼
Output: predicted missing slice    (1 × 256 × 256)
```

**7.76 million parameters.** Trained for 20 epochs on an RTX 5070 Ti.
Loss = `0.8 × L1  +  0.2 × (1 − SSIM)` — pixel accuracy blended with structural similarity.

---

## Ablation — Skip Connections Are the Key Ingredient

A **PlainCNN** was trained as a controlled ablation: identical to the U-Net but with all
skip connections removed. This isolates their contribution.

| Method                   | PSNR (dB)        | SSIM               | MAE      |
|--------------------------|------------------|--------------------|----------|
| Cubic spline             | 33.19            | 0.9048             | —        |
| Linear interpolation     | 33.21            | 0.9066             | —        |
| **PlainCNN (no skips)**  | 33.06            | —                  | —        |
| **U-Net (ours)**         | **36.07 ± 6.5**  | **0.9461 ± 0.029** | 0.00932  |

Key findings:

- PlainCNN (33.06 dB) is **worse than linear interpolation** (33.21 dB) — a deep
  network without skip connections cannot beat a classical baseline on this task.
- The U-Net is **+2.86 dB over linear** and **+3.01 dB over PlainCNN**, proving skip
  connections are the essential architectural ingredient.

> **Reading PSNR:** above 35 dB means differences are barely visible to the human eye.
> Above 40 dB is essentially indistinguishable from the original.

---

## Cross-Dataset Results — BraTS2020

| Method            | PSNR (dB) | FG-MAE  | Note                                       |
|-------------------|-----------|---------|--------------------------------------------|
| **U-Net (BraTS)** | 58.65     | 0.00754 | Raw PSNR inflated by skull-strip zeros     |

FG-MAE of **0.00754** on tissue voxels confirms the model generalises to a structurally
different dataset (tumour-bearing, skull-stripped) without any architectural changes.

---

## Explainability — What Does the Model Look At?

Four complementary techniques were applied across **50 uniformly sampled test cases**.

### Grad-CAM (spatial attention)
Heat map showing which image regions drove the reconstruction output.

- **Hard samples** (low PSNR): diffuse, widespread activation — model is uncertain.
- **Medium samples**: attention concentrated on skull boundary and tissue edges.
- **Easy samples**: near-zero activation — neighboring slices are already near-identical.

### Integrated Gradients (input attribution)
Attributes each input pixel’s contribution along a baseline–input path.

| Metric                  | Value  | Interpretation                         |
|-------------------------|--------|----------------------------------------|
| Mean IG — Left slice   | 9.20e-07 | Both channels contribute equally     |
| Mean IG — Right slice  | 9.11e-07 |                                      |
| **Left/Right IG ratio** | **1.0093** | Near-perfect symmetry (1.0 = ideal) |

### Slice Contribution (channel ablation)
Zeros out each input channel in turn; measures the resulting output change.

| Metric                         | Value          |
|--------------------------------|----------------|
| Contribution — Left            | 0.0808 ± 0.049  |
| Contribution — Right           | 0.0763 ± 0.046  |
| Left/Right contribution ratio  | 1.059          |

High absolute values (vs. occlusion below) confirm the model integrates **global** slice
structure rather than attending to any local region.

### Occlusion Sensitivity (16 × 16 patch occlusion)
Slides a zeroed patch over the input; measures output sensitivity.

| Metric                        | Value          |
|-------------------------------|----------------|
| Sensitivity — Left            | 0.0003 ± 0.0002 |
| Sensitivity — Right           | 0.0003 ± 0.0002 |
| Left/Right occlusion ratio    | 1.028          |

No single patch dominates — information is distributed globally across both slices.

### MC Uncertainty (bottleneck dropout, 20 passes)
| Metric                   | Value    | Interpretation                           |
|--------------------------|----------|------------------------------------------|
| Mean variance            | 7.44e-16 | Effectively zero — model is very confident |

---

## Project Structure

```
Project/
├── Documentation/              # extended design notes
├── src/
│   ├── config.py               # single source of truth: all paths + hyperparameters
│   ├── preprocess.py           # IXI NIfTI → resize → normalise → 2.5D pairs + baselines
│   ├── preprocess_brats.py     # BraTS2020 H5 → 2.5D reconstruction pairs
│   ├── train.py                # U-Net + PlainCNN, --model and --data flags, evaluation
│   └── explainability.py       # Grad-CAM, IG, slice contribution, occlusion, MC uncertainty
├── data/
│   ├── raw/                    # put IXI .nii / .nii.gz files here
│   ├── brats/                  # put BraTS2020 H5 files here
│   └── processed/              # generated .npy arrays (git-ignored)
├── models/                     # unet_best.pth, unet_brats_best.pth (git-ignored)
├── outputs/
│   ├── figures/                # training curves, prediction grids
│   ├── metrics/                # per-sample PSNR / SSIM / MAE arrays
│   └── explainability/         # per-sample PNG figures + summary_stats.txt
├── notebooks/                  # exploratory Jupyter notebooks
├── logs/                       # SLURM job logs (HPC only)
├── requirements.txt
├── job_run_gpu.sh              # UBELIX HPC submission script
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Requirements: `torch`, `torchvision`, `pytorch-msssim`, `nibabel`, `numpy`, `scipy`,
`scikit-image`, `matplotlib`, `h5py`.

---

## Running

All commands run from the **project root**. Steps 1a and 1b are independent;
steps 2–3 depend on step 1.

```bash
# Step 1a — preprocess IXI scans (~581 volumes, takes several minutes)
python -m src.preprocess

# Step 1b — preprocess BraTS2020 (optional, for cross-dataset validation)
python -m src.preprocess_brats

# Step 2 — train U-Net on IXI
python -m src.train --model unet --data ixi

# Step 2b — ablation: PlainCNN without skip connections
python -m src.train --model plaincnn --data ixi

# Step 2c — train U-Net on BraTS
python -m src.train --model unet --data brats

# Step 3 — generate all explainability outputs (50 samples, ~15–30 min on GPU)
python -m src.explainability
```

### On UBELIX HPC

```bash
sbatch job_run_gpu.sh
```

Logs → `logs/<jobname>_<jobid>.out` and `.err`

---

## Configuration

Every number that matters lives in [src/config.py](src/config.py).

| Parameter              | Default                | What it controls                                   |
|------------------------|------------------------|----------------------------------------------------|
| `ACCELERATION_FACTOR`  | `2`                    | Slices skipped (2 = every other one, 50% missing)  |
| `TARGET_SIZE`          | `(256, 256)`           | Pixel resolution of each slice                     |
| `BATCH_SIZE`           | `8`                    | Samples per gradient step                          |
| `NUM_EPOCHS`           | `20`                   | Maximum training epochs                            |
| `LEARNING_RATE`        | `1e-3`                 | Adam initial learning rate                         |
| `LOSS_ALPHA`           | `0.8`                  | L1 vs SSIM blend (0.8 = 80 % L1, 20 % SSIM)       |
| `UNET_FEATURES`        | `[32, 64, 128, 256]`   | Channel counts across U-Net encoder levels         |
| `N_EXPL_SAMPLES`       | `50`                   | Samples analysed by explainability                 |
| `IG_STEPS`             | `50`                   | Integrated Gradients path steps                    |
| `MC_DROPOUT_PASSES`    | `10`                   | Forward passes per sample for MC uncertainty       |
| `MC_DROPOUT_P`         | `0.3`                  | Dropout probability during MC passes               |
