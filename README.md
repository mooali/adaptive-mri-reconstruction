# MRI Slice Interpolation with Deep Learning

> **The one-line pitch:** MRI scanners are slow. What if you acquired only half the data and used a neural network to reconstruct the rest — and then had the AI itself decide whether its reconstruction was good enough to trust?

---

## The Problem

A full brain MRI scan acquires hundreds of image slices, one at a time. This takes time — and in clinical settings, **scan time directly affects patient comfort, scanner availability, and cost**.

One way to speed things up is to skip every other slice during acquisition. You end up with half the data in half the time. The challenge: can you recover the missing slices well enough that a doctor wouldn't know the difference?

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

The project has four modules, each doing one thing:

```
src/preprocess.py        Load 581 brain MRI scans → build training pairs → compute baselines
         ↓
src/train.py             Define + train a U-Net → evaluate on test set → save best model
         ↓
src/explainability.py    Grad-CAM + Integrated Gradients → understand what the model looks at
         ↓
src/adaptive_decision.py Monte Carlo Dropout → estimate uncertainty → decide: trust it or rescan
```

All configuration (paths, hyperparameters, thresholds) lives in one place: `src/config.py`.

---

## Dataset

**IXI Brain Dataset** — 581 T1-weighted MRI volumes from 3 London hospitals  
(Guy's, Hammersmith, IOP). Publicly available, healthy subjects only.

Each volume was:
- Resized to 256 × 256 pixels per slice
- Normalised to [0, 1] intensity range
- Split into 2.5D input–target pairs (every other slice skipped)

This produced **42,816 training pairs** in total.

---

## The Model — U-Net

A U-Net is an encoder–decoder network with skip connections. It was originally designed for medical image segmentation and works extremely well for any task that needs to produce a full-resolution image from another full-resolution image.

```
Input: [left_slice, right_slice]   (2 × 256 × 256)
          │
    ┌─────▼─────┐
    │  Encoder  │   32 → 64 → 128 → 256 channels
    │           │   each level: 2× Conv + BN + ReLU → MaxPool
    └─────┬─────┘
          │  skip connections (fine detail preserved here)
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

**7.76 million parameters.** Trained for 20 epochs on an RTX 4090 GPU (UBELIX HPC).  
Loss = `0.8 × L1  +  0.2 × (1 − SSIM)` — a blend of pixel accuracy and structural similarity.

---

## Results

Compared against two classical baselines on 6,422 held-out test pairs:

| Method              | PSNR (dB)          | SSIM               | MAE      |
|---------------------|--------------------|--------------------|----------|
| Linear interpolation| 33.21              | 0.9066             | —        |
| Cubic spline        | 33.19              | 0.9048             | —        |
| **U-Net (ours)**    | **35.95 ± 6.54**   | **0.9455 ± 0.0294**| 0.00939  |

**+2.74 dB PSNR and +0.039 SSIM** over the best classical method.

> **Reading PSNR:** above 35 dB means the difference between original and reconstructed is barely visible to the human eye. Above 40 dB is essentially perfect.

> **Reading SSIM:** 1.0 is identical. 0.94 means structural features (edges, shapes) are very well preserved.

---

## Explainability — What Does the Model Actually Look At?

Two techniques were used to peer inside the model:

### Grad-CAM (where does it look?)
Produces a heat map over the image showing which spatial regions had the most influence on the reconstruction.

- **Hard cases** (low PSNR): attention is widespread — the model is uncertain and casts a wide net.
- **Medium cases**: attention focuses on skull boundary and tissue edges — the model uses structural anchors.
- **Easy cases** (boundary/background slices): almost no activation — the slices are so similar the model barely needs to look at anything.

### Integrated Gradients (which input pixels matter?)
Attributes each pixel of the two input slices to the final output. Key finding:

- **Left/right attribution ratio ≈ 0.95** — the model treats both neighboring slices almost equally, which is exactly what you'd want from an interpolation model. If this were 0.3 or 2.0, it would mean the model was nearly ignoring one of its inputs.

---

## Adaptive Acquisition — Should We Even Trust the Reconstruction?

This is the most novel part. Instead of always accepting the U-Net's output, the system first asks: *"How confident is the model in this reconstruction?"*

**Technique: Monte Carlo Dropout**

During normal inference, neural networks give one deterministic answer. By randomly switching off 10% of the bottleneck neurons and running 10 forward passes, each pass gives a slightly different result. The **variance across those 10 predictions** is used as an uncertainty score.

```
10 stochastic passes
    → pixel-wise variance map
        → global uncertainty score (mean variance)
            → compare to threshold
                → SAFE: use the reconstruction
                  UNSAFE: acquire the missing slices for real
```

**Decision rule:**
```
if uncertainty < 0.01:
    "Decision: SAFE — reconstruction applied"
else:
    "Decision: UNSAFE — full acquisition recommended"
```

This turns a passive reconstruction model into an **active safety gate** — the AI decides when to trust itself and when to step aside.

---

## Project Structure

```
Project/
├── documentation/              # conceptual docs and extension notes
├── src/
│   ├── config.py               # single source of truth: all paths + hyperparameters
│   ├── preprocess.py           # load NIfTI → resize → normalise → build pairs → baselines
│   ├── train.py                # U-Net + loss + training loop + evaluation
│   ├── explainability.py       # Grad-CAM and Integrated Gradients
│   └── adaptive_decision.py    # Monte Carlo Dropout + uncertainty + safety decision
├── data/
│   ├── raw/                    # put your .nii / .nii.gz files here
│   └── processed/              # generated arrays (dataset_inputs.npy, dataset_targets.npy)
├── models/                     # unet_best.pth saved here during training
├── outputs/
│   ├── figures/                # training curves, prediction grids, metric histograms
│   ├── metrics/                # .npy arrays of PSNR / SSIM / MAE per sample
│   └── explainability/         # Grad-CAM maps, IG maps, adaptive decision panels
├── logs/                       # SLURM job output (HPC only)
├── requirements.txt
├── job_run_gpu.sh              # submit all 4 steps to UBELIX in one go
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Requirements: `torch`, `torchvision`, `pytorch-msssim`, `nibabel`, `numpy`, `scipy`, `scikit-image`, `matplotlib`, `opencv-python`.

---

## Running

Run each step from the **project root folder**. They must run in order — each step depends on the output of the previous one.

```bash
# Step 1 — preprocess raw scans (takes a while for 581 volumes)
python -m src.preprocess

# Step 2 — train the U-Net (~20 epochs on GPU)
python -m src.train

# Step 3 — generate Grad-CAM and IG visualisations
python -m src.explainability

# Step 4 — run adaptive uncertainty analysis
python -m src.adaptive_decision
```

### On UBELIX HPC (all 4 steps in one job)

```bash
sbatch job_run_gpu.sh
```

Logs → `logs/<jobname>_<jobid>.out` and `.err`

---

## Configuration

Every number that matters lives in [src/config.py](src/config.py). To change anything, edit only that file.

| Parameter              | Default            | What it controls                                   |
|------------------------|--------------------|----------------------------------------------------|
| `ACCELERATION_FACTOR`  | `2`                | How many slices are skipped (2 = every other one)  |
| `TARGET_SIZE`          | `(256, 256)`       | Pixel resolution of each slice                     |
| `BATCH_SIZE`           | `8`                | Samples per gradient step                          |
| `NUM_EPOCHS`           | `20`               | Training epochs                                    |
| `LEARNING_RATE`        | `1e-3`             | Adam initial step size                             |
| `LOSS_ALPHA`           | `0.8`              | L1 vs SSIM weight in the loss (0 = all SSIM)       |
| `UNET_FEATURES`        | `[32, 64, 128, 256]` | Channel counts across U-Net encoder levels       |
| `N_EXPL_SAMPLES`       | `6`                | How many samples to analyse in explainability      |
| `MC_DROPOUT_PASSES`    | `10`               | Forward passes per sample for uncertainty estimate |
| `MC_DROPOUT_P`         | `0.1`              | Dropout probability during MC passes               |
| `UNCERTAINTY_THRESHOLD`| `0.01`             | Score above which full acquisition is recommended  |
