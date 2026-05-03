from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Data
RAW_DIR       = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Model checkpoints
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "unet_best.pth"

# Outputs
FIGURES_DIR        = ROOT_DIR / "outputs" / "figures"
METRICS_DIR        = ROOT_DIR / "outputs" / "metrics"
EXPLAINABILITY_DIR = ROOT_DIR / "outputs" / "explainability"

# Logs
LOGS_DIR = ROOT_DIR / "logs"

# ---------- Preprocessing ----------
TARGET_SIZE         = (256, 256)
ACCELERATION_FACTOR = 2

# ---------- Dataset split ----------
TRAIN_SPLIT  = 0.70
VAL_SPLIT    = 0.15
RANDOM_SEED  = 42

# ---------- Training ----------
BATCH_SIZE    = 8
NUM_EPOCHS    = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-5
LOSS_ALPHA    = 0.8   # weight of L1 in combined L1 + (1-SSIM) loss

# ---------- U-Net architecture ----------
UNET_FEATURES = [32, 64, 128, 256]
IN_CHANNELS   = 2
OUT_CHANNELS  = 1

# ---------- Explainability ----------
N_EXPL_SAMPLES = 6
IG_STEPS       = 50

# ---------- Adaptive Decision ----------
MC_DROPOUT_PASSES     = 10    # number of stochastic forward passes
MC_DROPOUT_P          = 0.1   # dropout probability applied to bottleneck features
UNCERTAINTY_THRESHOLD = 0.01  # pixel-variance mean above which full acquisition is triggered
