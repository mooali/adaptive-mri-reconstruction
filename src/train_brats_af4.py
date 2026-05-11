#!/usr/bin/env python
"""
src/train_brats_af4.py

Trains a U-Net for AF=4 BraTS slice reconstruction.
Input: (3, H, W) — [left, right, t_map]
Target: (1, H, W) — missing slice at fractional position t

Uses the same volume-level train/val/test split as the AF=2 experiment
so results are directly comparable.
"""
import platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch_msssim import ssim as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from src.config import (
    PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LOSS_ALPHA, TRAIN_SPLIT, VAL_SPLIT, RANDOM_SEED, UNET_FEATURES
)
from src.train import UNet, CombinedLoss, train_epoch, val_epoch

IN_CHANNELS = 3
OUT_CHANNELS = 1
CKPT_NAME = "unet_brats_af4_best.pth"

INPUTS_PATH  = PROCESSED_DIR / "brats_inputs_af4.npy"
TARGETS_PATH = PROCESSED_DIR / "brats_targets_af4.npy"
VIDS_PATH    = PROCESSED_DIR / "brats_volume_ids_af4.npy"


# ── Dataset ───────────────────────────────────────────────────────────────────
class AF4Dataset(Dataset):
    def __init__(self, inputs_path, targets_path, indices):
        self.inputs  = np.load(inputs_path,  mmap_mode="r")
        self.targets = np.load(targets_path, mmap_mode="r")
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ri = self.indices[idx]
        x  = torch.tensor(np.array(self.inputs[ri]),  dtype=torch.float32)   # (3, H, W)
        y  = torch.tensor(np.array(self.targets[ri]), dtype=torch.float32).unsqueeze(0)
        return x, y


# ── Volume-level split (identical logic to AF=2) ──────────────────────────────
def volume_split(vol_ids):
    unique_vols = np.unique(vol_ids)
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(unique_vols)
    n_train = int(TRAIN_SPLIT * len(shuffled))
    n_val   = int(VAL_SPLIT   * len(shuffled))
    train_vols = set(shuffled[:n_train])
    val_vols   = set(shuffled[n_train:n_train + n_val])
    test_vols  = set(shuffled[n_train + n_val:])

    train_idx = np.where([v in train_vols for v in vol_ids])[0]
    val_idx   = np.where([v in val_vols   for v in vol_ids])[0]
    test_idx  = np.where([v in test_vols  for v in vol_ids])[0]
    return train_idx, val_idx, test_idx


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_test(model, loader, device):
    model.eval()
    psnr_s, ssim_s, mae_s = [], [], []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).cpu().numpy()
            gts   = y.numpy()
            for i in range(len(preds)):
                g, p = gts[i, 0], preds[i, 0]
                psnr_s.append(psnr(g, p, data_range=1.0))
                ssim_s.append(ssim(g, p, data_range=1.0))
                mae_s.append(float(np.abs(g - p).mean()))
    return {
        "psnr_mean": np.mean(psnr_s), "psnr_std": np.std(psnr_s),
        "ssim_mean": np.mean(ssim_s), "ssim_std": np.std(ssim_s),
        "mae_mean":  np.mean(mae_s),  "mae_std":  np.std(mae_s),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vol_ids = np.load(VIDS_PATH, mmap_mode="r")
    total   = len(np.load(INPUTS_PATH, mmap_mode="r"))
    print(f"Total AF=4 samples: {total}")

    train_idx, val_idx, test_idx = volume_split(vol_ids)
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    num_workers = 0 if platform.system() == "Windows" else 4

    train_set = AF4Dataset(INPUTS_PATH, TARGETS_PATH, train_idx)
    val_set   = AF4Dataset(INPUTS_PATH, TARGETS_PATH, val_idx)
    test_set  = AF4Dataset(INPUTS_PATH, TARGETS_PATH, test_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    model     = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, features=UNET_FEATURES).to(device)
    criterion = CombinedLoss(alpha=LOSS_ALPHA)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-Net parameters: {n_params:,}  (IN_CHANNELS={IN_CHANNELS})")

    ckpt_path = MODELS_DIR / CKPT_NAME
    best_val  = float("inf")
    history   = {"train_loss": [], "val_loss": []}

    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'LR':>10}")
    print("-" * 46)

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss = val_epoch(model, val_loader, criterion, device)
        scheduler.step(vl_loss)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6}  {tr_loss:>12.6f}  {vl_loss:>12.6f}  {lr:>10.2e}")

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"         ✓ saved  (val={best_val:.6f})")

    print(f"\nTraining complete. Best val loss: {best_val:.6f}")

    # Training curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["train_loss"], label="Train", linewidth=1.5)
    ax.plot(history["val_loss"],   label="Val",   linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("AF=4 Training Curves")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "training_curves_af4.png", dpi=150)
    plt.close()

    # Test evaluation
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    metrics = evaluate_test(model, test_loader, device)

    print("\n========== AF=4 TEST SET RESULTS ==========")
    print(f"{'Metric':<10}  {'Mean':>10}  {'Std':>10}")
    print("-" * 34)
    print(f"{'PSNR':<10}  {metrics['psnr_mean']:>9.2f}dB  {metrics['psnr_std']:>10.2f}")
    print(f"{'SSIM':<10}  {metrics['ssim_mean']:>10.4f}  {metrics['ssim_std']:>10.4f}")
    print(f"{'MAE':<10}  {metrics['mae_mean']:>10.5f}  {metrics['mae_std']:>10.5f}")

    np.save(METRICS_DIR / "unet_af4_test_metrics.npy", metrics)
    print(f"\nMetrics saved → {METRICS_DIR / 'unet_af4_test_metrics.npy'}")
    print(f"Checkpoint   → {ckpt_path}")


if __name__ == "__main__":
    main()