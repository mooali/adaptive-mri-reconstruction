"""
src/config.py — Central configuration for the entire MRI reconstruction project.

Every hyperparameter and every output path lives here. No other module should
hardcode paths or numeric constants. To change any setting (e.g. switch from
×2 to ×4 acceleration, increase epochs, lower the uncertainty threshold) edit
only this file — all downstream modules pick up the change automatically.

Paths are built with pathlib.Path so they work identically on Linux (HPC) and
Windows (local dev) without any string manipulation.

No third-party dependencies — only the Python standard library.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root and directory layout
# ---------------------------------------------------------------------------

# Resolve the project root from this file's location.
# __file__ is  .../Project/src/config.py  →  .parent = src/  →  .parent = Project/
ROOT_DIR = Path(__file__).resolve().parent.parent

# Raw NIfTI scans supplied by the user; read-only at runtime.
RAW_DIR       = ROOT_DIR / "data" / "raw"

# Preprocessed numpy arrays written by preprocess.py; read by train.py and
# explainability.py.
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Trained model weights; written by train.py, read by explainability.py and
# adaptive_decision.py.
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "unet_best.pth"

# Output subdirectories — each module writes to the relevant subfolder.
FIGURES_DIR        = ROOT_DIR / "outputs" / "figures"        # training curves, predictions
METRICS_DIR        = ROOT_DIR / "outputs" / "metrics"        # .npy metric arrays
EXPLAINABILITY_DIR = ROOT_DIR / "outputs" / "explainability" # Grad-CAM, IG, adaptive panels

# SLURM logs written by job_run_gpu.sh.
LOGS_DIR = ROOT_DIR / "logs"

# ---------------------------------------------------------------------------
# Preprocessing  (src/preprocess.py)
# ---------------------------------------------------------------------------

# Each axial slice is resized to this spatial resolution before any further
# processing.  256×256 balances detail and memory; all downstream arrays
# assume this size.
TARGET_SIZE = (256, 256)

# Simulated scanner acceleration: every ACCELERATION_FACTOR-th slice is
# treated as "acquired"; the others become reconstruction targets.
# 2 means 50% of slices are missing (every second one).
ACCELERATION_FACTOR = 2

# ---------------------------------------------------------------------------
# Dataset split  (src/train.py)
# ---------------------------------------------------------------------------

# Proportions must sum to ≤ 1.0; the remainder becomes the test set.
TRAIN_SPLIT = 0.70   # 70% for gradient updates
VAL_SPLIT   = 0.15   # 15% for early stopping / LR scheduling
# test = remaining 15% — held out until final evaluation

# Fixed seed ensures the same shuffled index ordering across runs so that
# train / val / test splits are reproducible.
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Training  (src/train.py)
# ---------------------------------------------------------------------------

# Number of samples per gradient step.  8 fits in ~12 GB GPU VRAM with the
# U-Net below; increase if more memory is available.
BATCH_SIZE = 8

# Maximum number of training epochs.  Early stopping is implicit: only the
# epoch with the lowest validation loss is saved.
NUM_EPOCHS = 20

# Adam initial learning rate.  ReduceLROnPlateau halves it when validation
# loss stops improving (patience=5 epochs).
LEARNING_RATE = 1e-3

# L2 regularisation strength in Adam.  Keeps weights small without strongly
# penalising large activations the way dropout would.
WEIGHT_DECAY = 1e-5

# Weight of the L1 term in the combined loss:
#   loss = LOSS_ALPHA * L1  +  (1 - LOSS_ALPHA) * (1 - SSIM)
# 0.8 / 0.2 favours per-pixel accuracy slightly over structural similarity.
# Raising LOSS_ALPHA produces sharper predictions; lowering it can improve
# perceptual quality at the cost of some pixel fidelity.
LOSS_ALPHA = 0.8

# ---------------------------------------------------------------------------
# U-Net architecture  (src/train.py, src/explainability.py,
#                      src/adaptive_decision.py)
# ---------------------------------------------------------------------------

# Channel counts for the four encoder levels.  Each level doubles the channel
# count, halves spatial dimensions (via MaxPool2d), and adds capacity for
# increasingly abstract features.
# Bottleneck uses features[-1] * 2 = 512 channels automatically.
UNET_FEATURES = [32, 64, 128, 256]

# Two neighboring acquired slices stacked along the channel axis.
# IN_CHANNELS = 2
# Three neighboring acquired slices stacked along the channel axis.  
# This is used for the BraTS dataset since it has fewer slices per volume, 
# so more context is needed to reconstruct the middle slice.
IN_CHANNELS = 3

# Single predicted intermediate slice.
OUT_CHANNELS = 1

# ---------------------------------------------------------------------------
# Explainability  (src/explainability.py)
# ---------------------------------------------------------------------------

# Number of samples analysed by the explainability and adaptive decision
# modules.  Indices are spread uniformly across the full dataset via linspace.
N_EXPL_SAMPLES = 100

# Number of interpolation steps for Integrated Gradients.  Higher values give
# more accurate attribution but scale linearly in compute time.  50 provides
# a good accuracy / speed trade-off for 256×256 inputs.
IG_STEPS = 50

# ---------------------------------------------------------------------------
# Adaptive Decision  (src/adaptive_decision.py)
# ---------------------------------------------------------------------------

# Number of stochastic forward passes used to estimate prediction variance.
# More passes → more stable uncertainty estimate, but proportionally slower.
MC_DROPOUT_PASSES = 10

# Dropout probability applied to the bottleneck feature map during each MC
# pass.  0.1 gives measurable variance without collapsing predictions;
# higher values increase variance range but degrade mean reconstruction quality.
MC_DROPOUT_P = 0.1

# Global uncertainty score (mean pixel variance) above which the module
# recommends full acquisition instead of relying on the reconstruction.
# This value is heuristic — it should be calibrated on a validation set
# with known-difficult cases if used in a real clinical pipeline.
UNCERTAINTY_THRESHOLD = 0.01

# ---------------------------------------------------------------------------
# Anomaly detection / Phase 2  (src/adaptive_decision.py)
# ---------------------------------------------------------------------------

# Path to the trained SliceAutoencoder weights produced by
# train_anomaly_detector().  Reused on subsequent runs to skip retraining.
ANOMALY_DETECTOR_PATH = MODELS_DIR / "anomaly_detector.pth"

# Path to the .npy scalar file that stores the calibrated anomaly threshold
# (mean + 2σ of training-set MAE).  Written alongside the model weights.
ANOMALY_THRESHOLD_PATH = MODELS_DIR / "anomaly_threshold.npy"

# Number of epochs to train the SliceAutoencoder on healthy MRI slices.
# 10 epochs converges on the IXI dataset; increase if training loss plateaus.
ANOMALY_EPOCHS = 10

# Fallback anomaly threshold used when ANOMALY_THRESHOLD_PATH does not yet
# exist (i.e. before train_anomaly_detector() has been run).
ANOMALY_DEFAULT_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# BraTS2020  (src/preprocess_brats.py, src/evaluate_brats.py)
# ---------------------------------------------------------------------------

# Raw BraTS2020 per-slice H5 files.
BRATS_DIR = ROOT_DIR / "data" / "brats" / "img"

# Preprocessed 2.5D pairs — identical format to IXI dataset_inputs/targets
BRATS_INPUTS_PATH  = PROCESSED_DIR / "brats_inputs.npy"    # shape (N, 2, 256, 256)
BRATS_TARGETS_PATH = PROCESSED_DIR / "brats_targets.npy"   # shape (N, 256, 256)
BRATS_META_PATH    = PROCESSED_DIR / "brats_meta.npy"      # shape (N,) bool
