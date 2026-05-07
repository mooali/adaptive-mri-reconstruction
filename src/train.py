#!/usr/bin/env python
"""
src/train.py — U-Net model definition, training loop, and evaluation.

Purpose
-------
Defines the full U-Net architecture for MRI slice interpolation, trains it
on the preprocessed dataset, and evaluates the best checkpoint on the test
set.  All model classes defined here (UNet, ConvBlock) are imported by
explainability.py and adaptive_decision.py to avoid code duplication.

Data flow inside main()
-----------------------
  1. Load dataset_inputs.npy and dataset_targets.npy via memory-mapped arrays.
  2. Shuffle and split indices → train / val / test (70 / 15 / 15).
  3. Wrap each split in MRISliceDataset, feed into DataLoader.
  4. Instantiate UNet, CombinedLoss, Adam optimiser, ReduceLROnPlateau.
  5. Train for NUM_EPOCHS; checkpoint whenever val loss improves.
  6. Reload best weights; run evaluate_test_set on the test split.
  7. Save metrics, training curves, and visual predictions.

Outputs
-------
  models/unet_best.pth                      — best model state_dict
  outputs/metrics/unet_test_metrics.npy     — per-sample PSNR / SSIM / MAE
  outputs/figures/training_curves.png
  outputs/figures/predictions.png
  outputs/figures/metric_distributions.png

Dependencies
------------
  Standard library  : (none beyond Python builtins)
  numpy             : array operations, metric aggregation, seed control
  torch             : model definition, training, GPU management
  torch.nn          : Conv2d, BatchNorm2d, MaxPool2d, ConvTranspose2d, …
  torch.optim       : Adam, ReduceLROnPlateau
  torch.utils.data  : Dataset, DataLoader
  pytorch_msssim    : differentiable SSIM used inside the combined loss
  skimage.metrics   : PSNR and SSIM for test-set evaluation (non-differentiable)
  matplotlib        : training curves and visual grids (Agg backend)
  src.config        : all hyperparameters and directory paths
"""

import platform
import numpy as np
import matplotlib
matplotlib.use("Agg")   # must be set before any pyplot import; enables headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch_msssim import ssim as ssim_metric  # differentiable SSIM for the loss
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from src.config import (
    PROCESSED_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LOSS_ALPHA,
    TRAIN_SPLIT,
    VAL_SPLIT,
    RANDOM_SEED,
    UNET_FEATURES,
    IN_CHANNELS,
    OUT_CHANNELS,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MRISliceDataset(Dataset):
    """
    PyTorch Dataset that wraps the preprocessed .npy arrays.

    Memory-mapping (mmap_mode='r') is used so the full arrays are never
    loaded into RAM all at once.  With ~42 K samples at (2, 256, 256) float32
    the uncompressed dataset is roughly 22 GB — far too large for most systems
    to hold in memory.  Memory-mapping lets the OS page individual samples on
    demand, keeping RAM usage proportional to the number of workers and batch
    size rather than the total dataset size.

    The optional `indices` argument is used to create non-overlapping train /
    val / test views of the same files without copying any data.

    Parameters
    ----------
    inputs_path  : Path  — path to dataset_inputs.npy  (N, 2, 256, 256)
    targets_path : Path  — path to dataset_targets.npy (N, 256, 256)
    indices      : np.ndarray or None  — subset of row indices to expose;
                   if None the full array is exposed
    """

    def __init__(self, inputs_path, targets_path, indices=None):
        self.inputs  = np.load(inputs_path,  mmap_mode="r")
        self.targets = np.load(targets_path, mmap_mode="r")
        self.indices = indices if indices is not None else np.arange(len(self.inputs))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        # np.array() materialises the memory-mapped slice into an in-memory
        # array before converting to a tensor; skipping this causes a slow
        # copy path inside torch.tensor on some platforms.
        x = torch.tensor(np.array(self.inputs[real_idx]),  dtype=torch.float32)          # (2, 256, 256)
        y = torch.tensor(np.array(self.targets[real_idx]), dtype=torch.float32).unsqueeze(0)  # (1, 256, 256)
        return x, y


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv → BN → ReLU → Conv → BN → ReLU.

    Used as the fundamental unit in both the encoder and decoder of the U-Net.
    The double-convolution pattern (from the original U-Net paper) lets each
    block increase receptive field and representational capacity without adding
    more pooling stages.

    Design choices:
    - bias=False: BatchNorm already has learnable shift (beta) and scale (gamma)
      parameters that subsume what a conv bias would do.  Removing the bias
      reduces parameter count and speeds up training slightly.
    - kernel_size=3, padding=1: keeps spatial dimensions identical after each
      conv (no shrinkage within a block).
    - inplace=True on ReLU: saves a memory allocation by modifying the tensor
      in place; safe here because the input is not needed for backprop.

    Parameters
    ----------
    in_ch  : int  — number of input channels
    out_ch : int  — number of output channels (same for both conv layers)
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net for MRI slice interpolation.

    Architecture overview
    ---------------------
    The network takes two acquired neighboring slices (stacked as channels)
    and predicts the missing slice between them.

      Input  : (B, 2, 256, 256)  — left and right acquired slices
      Output : (B, 1, 256, 256)  — predicted intermediate slice, values in [0, 1]

    Encoder (downsampling path)
      Four levels, each consisting of:
        - ConvBlock(ch_in → f)  — extracts features at the current resolution
        - MaxPool2d(2)          — halves spatial dimensions, doubles receptive field
      Channel progression: 2 → 32 → 64 → 128 → 256

    Bottleneck
      ConvBlock(256 → 512) at the lowest resolution (16×16 for 256×256 input).
      Captures the most global, abstract representation of the image.

    Decoder (upsampling path)
      Four levels, each consisting of:
        - ConvTranspose2d — learnable upsampling (doubles spatial dimensions)
        - Concatenate with the corresponding encoder skip connection
        - ConvBlock(2f → f) — fuses upsampled features with skip-connection detail
      The skip connections re-introduce fine spatial detail that was lost during
      pooling, which is critical for reconstructing sharp tissue boundaries.

    Output head
      1×1 Conv2d reduces channels to out_channels, then Sigmoid maps to [0, 1]
      to match the normalised input intensity range.

    Parameters
    ----------
    in_channels  : int         — typically 2 (two input slices)
    out_channels : int         — typically 1 (one output slice)
    features     : list[int]   — encoder channel counts; defaults to UNET_FEATURES
    """

    def __init__(self, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, features=None):
        super().__init__()
        if features is None:
            features = UNET_FEATURES

        # ── Encoder ──────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(ConvBlock(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        # ── Bottleneck ────────────────────────────────────────────────────
        # features[-1] = 256 → bottleneck = 512 channels at 16×16 resolution.
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # ── Decoder ──────────────────────────────────────────────────────
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = features[-1] * 2
        for f in reversed(features):
            # ConvTranspose2d doubles spatial resolution.
            self.upconvs.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            # After concatenating the skip connection the channel count doubles
            # (f from upconv + f from encoder skip → 2f), then ConvBlock
            # projects back down to f.
            self.decoders.append(ConvBlock(f * 2, f))
            ch = f

        # ── Output head ──────────────────────────────────────────────────
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        # Sigmoid ensures output is in [0, 1], consistent with normalised inputs
        # and with SSIM/PSNR metrics that assume data_range=1.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder: save feature maps for skip connections.
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)   # saved before pooling to preserve full resolution
            x = pool(x)

        x = self.bottleneck(x)

        # Decoder: upsample, concatenate skip, refine with ConvBlock.
        # reversed(skips) pairs each decoder level with its encoder counterpart.
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)  # channel-wise concat
            x = dec(x)

        return self.sigmoid(self.output_conv(x))


# ---------------------------------------------------------------------------
# Ablation model — PlainCNN (U-Net without skip connections)
# ---------------------------------------------------------------------------

class PlainCNN(nn.Module):
    """
    Encoder-decoder CNN identical to UNet but WITHOUT skip connections.

    Used as an ablation study to quantify the contribution of skip connections.
    The encoder and decoder channel counts match UNet exactly so the only
    difference is the absence of the concatenation step in the decoder.

    Without skip connections the decoder must reconstruct fine spatial detail
    (sharp edges, sulci, gyri boundaries) entirely from the compressed
    bottleneck representation.  Comparing PlainCNN vs UNet metrics shows
    exactly how much PSNR/SSIM the skip connections contribute.

    Architecture
    ------------
    Encoder     : same as UNet — 4 × (ConvBlock → MaxPool)
    Bottleneck  : same as UNet — ConvBlock(256 → 512)
    Decoder     : 4 × (ConvTranspose2d → ConvBlock)  — NO skip concat
                  channels: 512 → 256 → 128 → 64 → 32
    Output head : same as UNet — Conv2d(32 → 1) + Sigmoid

    Parameters / inputs / outputs are identical to UNet.
    """

    def __init__(self, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, features=None):
        super().__init__()
        if features is None:
            features = UNET_FEATURES

        # ── Encoder ──────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(ConvBlock(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        # ── Bottleneck ────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # ── Decoder — no skip connections, channels halved each level ────
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = features[-1] * 2
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            # Input is f (from upconv only — no skip concat), output is f.
            self.decoders.append(ConvBlock(f, f))
            ch = f

        # ── Output head ──────────────────────────────────────────────────
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid     = nn.Sigmoid()

    def forward(self, x):
        # Encoder: discard skip connections (not used in decoder).
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            x = pool(x)

        x = self.bottleneck(x)

        # Decoder: upsample + refine, no concatenation with encoder features.
        for upconv, dec in zip(self.upconvs, self.decoders):
            x = upconv(x)
            x = dec(x)

        return self.sigmoid(self.output_conv(x))

class CombinedLoss(nn.Module):
    """
    Weighted combination of L1 (pixel accuracy) and SSIM (structural fidelity).

      loss = LOSS_ALPHA * L1(pred, target)  +  (1 - LOSS_ALPHA) * (1 - SSIM)

    Why combine both?
    - L1 alone produces slightly blurry results because it optimises
      average pixel error without penalising structural distortion.
    - SSIM alone is noisy to optimise and can produce artefacts.
    - The 0.8 / 0.2 split gives L1 primary control of convergence while
      SSIM steers the network towards structurally coherent predictions.

    The pytorch_msssim implementation is differentiable and GPU-compatible,
    unlike skimage.metrics.structural_similarity which is CPU-only.

    Parameters
    ----------
    alpha : float  — weight of the L1 term (0–1); defaults to LOSS_ALPHA
    """

    def __init__(self, alpha=LOSS_ALPHA):
        super().__init__()
        self.alpha = alpha
        self.l1    = nn.L1Loss()

    def forward(self, pred, target):
        l1_val   = self.l1(pred, target)
        # ssim_metric returns a similarity in [0, 1]; subtracting from 1 gives
        # a loss that goes to 0 as predictions become structurally identical.
        ssim_val = ssim_metric(pred, target, data_range=1.0, size_average=True)
        return self.alpha * l1_val + (1 - self.alpha) * (1 - ssim_val)


# ---------------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    """
    Run one full pass over the training set, updating model weights.

    Loss is accumulated as a weighted sum and divided by dataset size at the
    end so the returned value is the mean per-sample loss (independent of
    batch size or dataset size).

    Parameters
    ----------
    model     : UNet
    loader    : DataLoader  — training split
    optimizer : torch.optim.Optimizer
    criterion : CombinedLoss
    device    : torch.device

    Returns
    -------
    float  — mean training loss for this epoch
    """
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        # Multiply by batch size so summing across batches gives the total
        # loss, which we then normalise by the dataset size below.
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def val_epoch(model, loader, criterion, device):
    """
    Evaluate model on the validation set without updating weights.

    torch.no_grad() disables gradient tracking, roughly halving memory
    usage and speeding up inference.

    Parameters
    ----------
    model     : UNet
    loader    : DataLoader  — validation split
    criterion : CombinedLoss
    device    : torch.device

    Returns
    -------
    float  — mean validation loss for this epoch
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total_loss += criterion(model(x), y).item() * x.size(0)
    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Test-set evaluation
# ---------------------------------------------------------------------------

def evaluate_test_set(model, loader, device):
    """
    Compute per-sample PSNR, SSIM, and MAE on the held-out test set.

    Predictions are moved to CPU and converted to numpy before calling
    skimage metrics, which do not support GPU tensors.

    Parameters
    ----------
    model  : UNet
    loader : DataLoader  — test split
    device : torch.device

    Returns
    -------
    dict with keys:
      'psnr_mean', 'psnr_std'        — aggregate PSNR statistics
      'ssim_mean', 'ssim_std'        — aggregate SSIM statistics
      'mae_mean',  'mae_std'         — aggregate MAE statistics
      'per_sample_psnr'              — list of per-sample PSNR values
      'per_sample_ssim'              — list of per-sample SSIM values
      'per_sample_mae'               — list of per-sample MAE values
    """
    model.eval()
    psnr_scores, ssim_scores, mae_scores = [], [], []
    fg_mae_scores = []   # foreground-only MAE (brain pixels where gt > 0.02)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).cpu().numpy()  # (B, 1, H, W)
            gt   = y.numpy()                           # (B, 1, H, W)
            for i in range(len(pred)):
                p, g = pred[i, 0], gt[i, 0]           # each (H, W)
                psnr_scores.append(psnr(g, p, data_range=1.0))
                ssim_scores.append(ssim(g, p, data_range=1.0))
                mae_scores.append(float(np.mean(np.abs(g - p))))
                fg = g > 0.02   # foreground mask: non-background pixels
                if fg.any():
                    fg_mae_scores.append(float(np.mean(np.abs(g[fg] - p[fg]))))

    result = {
        "psnr_mean"      : np.mean(psnr_scores),
        "psnr_std"       : np.std(psnr_scores),
        "ssim_mean"      : np.mean(ssim_scores),
        "ssim_std"       : np.std(ssim_scores),
        "mae_mean"       : np.mean(mae_scores),
        "mae_std"        : np.std(mae_scores),
        "per_sample_psnr": psnr_scores,
        "per_sample_ssim": ssim_scores,
        "per_sample_mae" : mae_scores,
    }
    if fg_mae_scores:
        result["fg_mae_mean"] = np.mean(fg_mae_scores)
        result["fg_mae_std"]  = np.std(fg_mae_scores)
    return result


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_training_curves(history, save_path=None):
    """
    Plot train and validation loss over epochs.

    Useful for diagnosing overfitting (val loss diverges from train loss)
    or underfitting (both plateau early).

    Parameters
    ----------
    history   : dict with 'train_loss' and 'val_loss' lists (one value per epoch)
    save_path : Path or None
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["train_loss"], label="Train loss", linewidth=1.5)
    ax.plot(history["val_loss"],   label="Val loss",   linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (L1 + SSIM)")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def visualize_predictions(model, dataset, device, n_samples=4, seed=42, save_path=None):
    """
    Display side-by-side: left input, right input, ground truth, prediction, error.

    One row per randomly selected test sample.  Per-sample metrics are printed
    as x-axis labels on the prediction column so quality is immediately visible.

    Parameters
    ----------
    model     : UNet
    dataset   : MRISliceDataset  — test split
    device    : torch.device
    n_samples : int   — number of rows in the grid
    seed      : int   — RNG seed for sample selection (reproducible output)
    save_path : Path or None
    """
    model.eval()
    rng     = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), n_samples, replace=False)

    fig = plt.figure(figsize=(20, 5 * n_samples))
    gs  = gridspec.GridSpec(n_samples, 5, figure=fig, hspace=0.4, wspace=0.3)
    col_titles = ["Left Input", "Right Input", "Ground Truth", "U-Net Prediction", "Error Map"]

    for row, idx in enumerate(indices):
        x, y = dataset[idx]
        with torch.no_grad():
            pred = model(x.unsqueeze(0).to(device)).squeeze().cpu().numpy()
        left, right, gt = x[0].numpy(), x[1].numpy(), y[0].numpy()
        error = np.abs(gt - pred)

        imgs  = [left, right, gt, pred, error]
        cmaps = ["gray", "gray", "gray", "gray", "hot"]
        vmins = [0, 0, 0, 0, 0]
        vmaxs = [1, 1, 1, 1, error.max()]

        for col, (img, cmap, vmin, vmax, title) in enumerate(
                zip(imgs, cmaps, vmins, vmaxs, col_titles)):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(title, fontsize=12, fontweight="bold")
            if col == 3:
                # Annotate the prediction panel with per-sample quality scores.
                ax.set_xlabel(
                    f"PSNR={psnr(gt, pred, data_range=1.0):.1f}dB  "
                    f"SSIM={ssim(gt, pred, data_range=1.0):.3f}  "
                    f"MAE={np.mean(error):.4f}",
                    fontsize=9,
                )
            ax.axis("off")

    fig.suptitle("U-Net Reconstruction — Visual Evaluation", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_metric_distributions(metrics, save_path=None):
    """
    Histogram of per-sample PSNR, SSIM, and MAE over the test set.

    Mean and median lines reveal whether the distribution is symmetric or
    skewed by a tail of very hard / very easy samples.

    Parameters
    ----------
    metrics   : dict  — output of evaluate_test_set
    save_path : Path or None
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("U-Net Test Set Metric Distributions", fontsize=13)

    for ax, (scores, label, unit) in zip(axes, [
        (metrics["per_sample_psnr"], "PSNR", "dB"),
        (metrics["per_sample_ssim"], "SSIM", ""),
        (metrics["per_sample_mae"],  "MAE",  ""),
    ]):
        ax.hist(scores, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(scores),   color="red",    linestyle="--", linewidth=1.5, label="Mean")
        ax.axvline(np.median(scores), color="orange", linestyle="--", linewidth=1.5, label="Median")
        ax.set_xlabel(f"{label} {unit}")
        ax.set_ylabel("Number of samples")
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Run the full training and evaluation pipeline.

    Splits are index-based (no data copying), the best checkpoint is saved
    whenever validation loss improves, and all outputs are written to the
    directories defined in config.py.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Train MRI slice interpolation model.")
    parser.add_argument(
        "--model", choices=["unet", "plaincnn"], default="unet",
        help="Architecture to train. 'unet' (default) = full U-Net with skip connections. "
             "'plaincnn' = ablation without skip connections.",
    )
    parser.add_argument(
        "--data", choices=["ixi", "brats"], default="ixi",
        help="Dataset to use: 'ixi' (default, data/processed/dataset_*.npy) or "
             "'brats' (data/processed/brats_*.npy).",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; load existing checkpoint and run test-set evaluation only.",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.data == "brats":
        from src.config import BRATS_INPUTS_PATH, BRATS_TARGETS_PATH
        inputs_path  = BRATS_INPUTS_PATH
        targets_path = BRATS_TARGETS_PATH
        print("Dataset: BraTS2020 reconstruction pairs")
    else:
        inputs_path  = PROCESSED_DIR / "dataset_inputs.npy"
        targets_path = PROCESSED_DIR / "dataset_targets.npy"
        print("Dataset: IXI")

    # Peek at the total number of samples using a memory-mapped read so we
    # do not load the full array just to get its length.
    total = len(np.load(inputs_path, mmap_mode="r"))
    print(f"Total pairs: {total:,}")

    # Create a reproducible random permutation then slice it into splits.
    np.random.seed(RANDOM_SEED)
    all_indices = np.random.permutation(total)
    n_train     = int(TRAIN_SPLIT * total)
    n_val       = int(VAL_SPLIT   * total)

    train_idx = all_indices[:n_train]
    val_idx   = all_indices[n_train : n_train + n_val]
    test_idx  = all_indices[n_train + n_val :]   # remaining ~15%

    train_set = MRISliceDataset(inputs_path, targets_path, train_idx)
    val_set   = MRISliceDataset(inputs_path, targets_path, val_idx)
    test_set  = MRISliceDataset(inputs_path, targets_path, test_idx)

    # num_workers=4 lets the data loader prefetch batches in background
    # processes while the GPU is busy with the previous batch.
    # pin_memory=True speeds up host→GPU transfers when CUDA is available.
    _num_workers = 0 if platform.system() == "Windows" else 4
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=_num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=_num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=_num_workers, pin_memory=True)

    print(f"Train: {len(train_set):,}  Val: {len(val_set):,}  Test: {len(test_set):,}")

    # ── Model selection ───────────────────────────────────────────────────
    data_suffix = "_brats" if args.data == "brats" else ""

    if args.model == "plaincnn":
        model      = PlainCNN().to(device)
        model_name = "PlainCNN (no skip connections)"
        model_path = MODELS_DIR / f"plaincnn_best{data_suffix}.pth"
        metrics_suffix = f"_plaincnn{data_suffix}"
        figures_suffix = f"_plaincnn{data_suffix}"
    else:
        model      = UNet().to(device)
        model_name = "U-Net"
        model_path = MODELS_DIR / f"unet_best{data_suffix}.pth"
        metrics_suffix = data_suffix
        figures_suffix = data_suffix

    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Halve the LR if val loss does not improve for 5 consecutive epochs.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name} parameters: {total_params:,}")

    best_val   = float("inf")
    history    = {"train_loss": [], "val_loss": []}

    if args.eval_only:
        if not model_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {model_path}. Run without --eval-only first.")
        print(f"\nEval-only mode — loading {model_path.name}")
    else:
        print(f"\nTraining for {NUM_EPOCHS} epochs...\n")
        print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'LR':>10}")
        print("-" * 50)

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss   = val_epoch(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            lr = optimizer.param_groups[0]["lr"]
            print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | {lr:>10.2e}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), model_path)
                print(f"         >> Best model saved (val_loss={best_val:.6f})")

        print(f"\nTraining complete. Best val loss: {best_val:.6f}")
        plot_training_curves(history, save_path=FIGURES_DIR / f"training_curves{figures_suffix}.png")

    # Reload the best weights (not necessarily the last epoch) for evaluation.
    model.load_state_dict(torch.load(model_path, map_location=device))
    metrics = evaluate_test_set(model, test_loader, device)

    print(f"\n========== TEST SET RESULTS — {model_name} ==========")
    print(f"{'Metric':<10} {'Mean':>10} {'Std':>10}")
    print("-" * 32)
    print(f"{'PSNR':<10} {metrics['psnr_mean']:>9.2f}dB {metrics['psnr_std']:>9.2f}")
    print(f"{'SSIM':<10} {metrics['ssim_mean']:>10.4f} {metrics['ssim_std']:>10.4f}")
    print(f"{'MAE':<10} {metrics['mae_mean']:>10.5f} {metrics['mae_std']:>10.5f}")

    if "fg_mae_mean" in metrics:
        print(f"{'FG-MAE':<10} {metrics['fg_mae_mean']:>10.5f} {metrics['fg_mae_std']:>10.5f}")
        print("  (FG-MAE = MAE over foreground pixels only, gt > 0.02)")

    # Baseline comparison — IXI values only; BraTS PSNR is inflated by
    # skull-stripped background zeros so the IXI baseline is not comparable.
    if args.data == "ixi":
        baseline_psnr = 33.21
        baseline_ssim = 0.9066
        print(
            f"\n  Improvement over linear baseline: "
            f"PSNR {metrics['psnr_mean'] - baseline_psnr:+.2f} dB  |  "
            f"SSIM {metrics['ssim_mean'] - baseline_ssim:+.4f}"
        )
    else:
        print("\n  Note: BraTS PSNR is inflated by skull-stripped background zeros.")
        print("  Use FG-MAE (foreground MAE) as the primary metric for BraTS.")

    np.save(METRICS_DIR / f"unet_test_metrics{metrics_suffix}.npy", metrics)
    print(f"Metrics saved to {METRICS_DIR / f'unet_test_metrics{metrics_suffix}.npy'}")

    visualize_predictions(model, test_set, device, n_samples=4,
                          save_path=FIGURES_DIR / f"predictions{figures_suffix}.png")
    plot_metric_distributions(metrics, save_path=FIGURES_DIR / f"metric_distributions{figures_suffix}.png")


if __name__ == "__main__":
    main()
