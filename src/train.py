#!/usr/bin/env python
"""
U-Net training pipeline for MRI slice interpolation.

Reads:   data/processed/dataset_inputs.npy   (N, 2, 256, 256)
         data/processed/dataset_targets.npy  (N, 256, 256)

Writes:  models/unet_best.pth
         outputs/metrics/unet_test_metrics.npy
         outputs/figures/training_curves.png
         outputs/figures/predictions.png
         outputs/figures/metric_distributions.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch_msssim import ssim as ssim_metric
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
    """Memory-mapped dataset of (input_pair, target_slice) samples."""

    def __init__(self, inputs_path, targets_path, indices=None):
        self.inputs  = np.load(inputs_path,  mmap_mode="r")
        self.targets = np.load(targets_path, mmap_mode="r")
        self.indices = indices if indices is not None else np.arange(len(self.inputs))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = torch.tensor(np.array(self.inputs[real_idx]),  dtype=torch.float32)          # (2, 256, 256)
        y = torch.tensor(np.array(self.targets[real_idx]), dtype=torch.float32).unsqueeze(0)  # (1, 256, 256)
        return x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two conv layers each followed by BatchNorm + ReLU."""

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

    Input  : (B, 2, 256, 256) — two neighboring acquired slices
    Output : (B, 1, 256, 256) — predicted intermediate slice in [0, 1]
    """

    def __init__(self, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, features=None):
        super().__init__()
        if features is None:
            features = UNET_FEATURES

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(ConvBlock(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Decoder
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = features[-1] * 2
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(f * 2, f))
            ch = f

        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid     = nn.Sigmoid()

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.sigmoid(self.output_conv(x))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """alpha * L1 + (1 - alpha) * (1 - SSIM)."""

    def __init__(self, alpha=LOSS_ALPHA):
        super().__init__()
        self.alpha = alpha
        self.l1    = nn.L1Loss()

    def forward(self, pred, target):
        l1_val   = self.l1(pred, target)
        ssim_val = ssim_metric(pred, target, data_range=1.0, size_average=True)
        return self.alpha * l1_val + (1 - self.alpha) * (1 - ssim_val)


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total_loss += criterion(model(x), y).item() * x.size(0)
    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_test_set(model, loader, device):
    model.eval()
    psnr_scores, ssim_scores, mae_scores = [], [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).cpu().numpy()
            gt   = y.numpy()
            for i in range(len(pred)):
                p, g = pred[i, 0], gt[i, 0]
                psnr_scores.append(psnr(g, p, data_range=1.0))
                ssim_scores.append(ssim(g, p, data_range=1.0))
                mae_scores.append(float(np.mean(np.abs(g - p))))
    return {
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


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_training_curves(history, save_path=None):
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
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    inputs_path  = PROCESSED_DIR / "dataset_inputs.npy"
    targets_path = PROCESSED_DIR / "dataset_targets.npy"

    total = len(np.load(inputs_path, mmap_mode="r"))
    print(f"Total pairs: {total:,}")

    np.random.seed(RANDOM_SEED)
    all_indices = np.random.permutation(total)
    n_train     = int(TRAIN_SPLIT * total)
    n_val       = int(VAL_SPLIT   * total)

    train_idx = all_indices[:n_train]
    val_idx   = all_indices[n_train : n_train + n_val]
    test_idx  = all_indices[n_train + n_val :]

    train_set = MRISliceDataset(inputs_path, targets_path, train_idx)
    val_set   = MRISliceDataset(inputs_path, targets_path, val_idx)
    test_set  = MRISliceDataset(inputs_path, targets_path, test_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_set):,}  Val: {len(val_set):,}  Test: {len(test_set):,}")

    model     = UNet().to(device)
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-Net parameters: {total_params:,}")

    model_path = MODELS_DIR / "unet_best.pth"
    best_val   = float("inf")
    history    = {"train_loss": [], "val_loss": []}

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
    plot_training_curves(history, save_path=FIGURES_DIR / "training_curves.png")

    # Evaluate
    model.load_state_dict(torch.load(model_path, map_location=device))
    metrics = evaluate_test_set(model, test_loader, device)

    print("\n========== TEST SET RESULTS ==========")
    print(f"{'Metric':<10} {'Mean':>10} {'Std':>10}")
    print("-" * 32)
    print(f"{'PSNR':<10} {metrics['psnr_mean']:>9.2f}dB {metrics['psnr_std']:>9.2f}")
    print(f"{'SSIM':<10} {metrics['ssim_mean']:>10.4f} {metrics['ssim_std']:>10.4f}")
    print(f"{'MAE':<10} {metrics['mae_mean']:>10.5f} {metrics['mae_std']:>10.5f}")

    baseline_psnr = 33.21
    baseline_ssim = 0.9066
    print(
        f"\n  Improvement over linear baseline: "
        f"PSNR {metrics['psnr_mean'] - baseline_psnr:+.2f} dB  |  "
        f"SSIM {metrics['ssim_mean'] - baseline_ssim:+.4f}"
    )

    np.save(METRICS_DIR / "unet_test_metrics.npy", metrics)
    print(f"Metrics saved to {METRICS_DIR / 'unet_test_metrics.npy'}")

    visualize_predictions(model, test_set, device, n_samples=4,
                          save_path=FIGURES_DIR / "predictions.png")
    plot_metric_distributions(metrics, save_path=FIGURES_DIR / "metric_distributions.png")


if __name__ == "__main__":
    main()
