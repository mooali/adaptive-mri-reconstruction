#!/usr/bin/env python
"""
src/train_brats_volume_split.py

Train/evaluate the existing U-Net with BraTS arrays using patient-level splitting.
This version filters trivial near-empty target slices during test evaluation and
visualisation so the reported metrics are more meaningful.
"""

import platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
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
    TRAIN_SPLIT,
    VAL_SPLIT,
    RANDOM_SEED,
)
from src.train import (
    MRISliceDataset,
    UNet,
    CombinedLoss,
    train_epoch,
    val_epoch,
    plot_training_curves,
    plot_metric_distributions,
)


MIN_FOREGROUND_FRACTION = 0.01
MIN_TARGET_STD = 0.02


def is_informative_target(target, min_foreground_fraction=MIN_FOREGROUND_FRACTION, min_target_std=MIN_TARGET_STD):
    target = np.asarray(target, dtype=np.float32)
    foreground_fraction = np.mean(target > 0)
    return (foreground_fraction >= min_foreground_fraction) and (target.std() >= min_target_std)


def split_by_volume(volume_ids, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT, seed=RANDOM_SEED):
    unique_vols = np.unique(volume_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_vols)

    n_train = int(train_split * len(shuffled))
    n_val = int(val_split * len(shuffled))
    train_vols = set(shuffled[:n_train])
    val_vols = set(shuffled[n_train:n_train + n_val])
    test_vols = set(shuffled[n_train + n_val:])

    train_idx = np.where(np.isin(volume_ids, list(train_vols)))[0]
    val_idx = np.where(np.isin(volume_ids, list(val_vols)))[0]
    test_idx = np.where(np.isin(volume_ids, list(test_vols)))[0]
    return train_idx, val_idx, test_idx, train_vols, val_vols, test_vols


def evaluate_test_set_filtered(model, loader, device,
                               min_foreground_fraction=MIN_FOREGROUND_FRACTION,
                               min_target_std=MIN_TARGET_STD):
    model.eval()
    psnr_scores, ssim_scores, mae_scores = [], [], []
    skipped_trivial = 0
    total_seen = 0

    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).cpu().numpy()
            gt = y.numpy()

            for i in range(len(pred)):
                p = pred[i, 0]
                g = gt[i, 0]
                total_seen += 1

                if not is_informative_target(g, min_foreground_fraction, min_target_std):
                    skipped_trivial += 1
                    continue

                psnr_scores.append(psnr(g, p, data_range=1.0))
                ssim_scores.append(ssim(g, p, data_range=1.0))
                mae_scores.append(float(np.mean(np.abs(g - p))))

    if len(psnr_scores) == 0:
        raise RuntimeError(
            "No informative test slices remained after filtering. "
            "Lower MIN_FOREGROUND_FRACTION or MIN_TARGET_STD."
        )

    return {
        "psnr_mean": float(np.mean(psnr_scores)),
        "psnr_std": float(np.std(psnr_scores)),
        "ssim_mean": float(np.mean(ssim_scores)),
        "ssim_std": float(np.std(ssim_scores)),
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
        "per_sample_psnr": np.array(psnr_scores, dtype=np.float32),
        "per_sample_ssim": np.array(ssim_scores, dtype=np.float32),
        "per_sample_mae": np.array(mae_scores, dtype=np.float32),
        "n_total_seen": int(total_seen),
        "n_used": int(len(psnr_scores)),
        "n_skipped_trivial": int(skipped_trivial),
        "min_foreground_fraction": float(min_foreground_fraction),
        "min_target_std": float(min_target_std),
    }


def visualize_predictions_filtered(model, dataset, device, n_samples=4, seed=42,
                                   save_path=None,
                                   min_foreground_fraction=MIN_FOREGROUND_FRACTION,
                                   min_target_std=MIN_TARGET_STD):
    model.eval()
    rng = np.random.default_rng(seed)

    informative_indices = []
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        gt = y[0].numpy()
        if is_informative_target(gt, min_foreground_fraction, min_target_std):
            informative_indices.append(idx)

    if len(informative_indices) == 0:
        raise RuntimeError(
            "No informative slices available for visualisation after filtering."
        )

    n_samples = min(n_samples, len(informative_indices))
    chosen = rng.choice(informative_indices, n_samples, replace=False)

    fig = plt.figure(figsize=(20, 5 * n_samples))
    gs = gridspec.GridSpec(n_samples, 5, figure=fig, hspace=0.4, wspace=0.3)
    col_titles = ["Left Input", "Right Input", "Ground Truth", "U-Net Prediction", "Error Map"]

    for row, idx in enumerate(chosen):
        x, y = dataset[idx]
        with torch.no_grad():
            pred = model(x.unsqueeze(0).to(device)).squeeze().cpu().numpy()

        left = x[0].numpy()
        right = x[1].numpy()
        gt = y[0].numpy()
        error = np.abs(gt - pred)

        sample_psnr = psnr(gt, pred, data_range=1.0)
        sample_ssim = ssim(gt, pred, data_range=1.0)
        sample_mae = float(np.mean(error))

        imgs = [left, right, gt, pred, error]
        cmaps = ["gray", "gray", "gray", "gray", "hot"]
        vmins = [0, 0, 0, 0, 0]
        vmaxs = [1, 1, 1, 1, max(error.max(), 1e-8)]

        for col, (img, cmap, vmin, vmax, title) in enumerate(zip(imgs, cmaps, vmins, vmaxs, col_titles)):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(title, fontsize=12, fontweight="bold")
            if col == 3:
                ax.set_xlabel(
                    f"PSNR {sample_psnr:.1f} dB\n"
                    f"SSIM {sample_ssim:.3f}\n"
                    f"MAE {sample_mae:.4f}",
                    fontsize=9,
                )
            ax.axis("off")

    fig.suptitle("U-Net Reconstruction — Informative Test Slices Only", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def print_metrics_summary(metrics):
    print("\n========== FILTERED TEST SET RESULTS ==========")
    print(f"Informative filter: foreground >= {metrics['min_foreground_fraction']:.3f}, std >= {metrics['min_target_std']:.3f}")
    print(f"Used {metrics['n_used']} / {metrics['n_total_seen']} test slices "
          f"(skipped trivial: {metrics['n_skipped_trivial']})")
    print()
    print(f"{'Metric':<10}{'Mean':>12}{'Std':>12}")
    print("-" * 34)
    print(f"{'PSNR':<10}{metrics['psnr_mean']:>10.2f}dB{metrics['psnr_std']:>12.2f}")
    print(f"{'SSIM':<10}{metrics['ssim_mean']:>12.4f}{metrics['ssim_std']:>12.4f}")
    print(f"{'MAE':<10}{metrics['mae_mean']:>12.5f}{metrics['mae_std']:>12.5f}")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    inputs_path = PROCESSED_DIR / "brats_inputs_rb.npy"
    targets_path = PROCESSED_DIR / "brats_targets_rb.npy"
    volume_ids_path = PROCESSED_DIR / "brats_volume_ids_rb.npy"


    volume_ids = np.load(volume_ids_path)
    total = len(volume_ids)
    print(f"Total pairs: {total:,}")
    print(f"Unique volumes: {len(np.unique(volume_ids))}")

    train_idx, val_idx, test_idx, train_vols, val_vols, test_vols = split_by_volume(volume_ids)
    print(f"Train volumes: {len(train_vols)} | Val volumes: {len(val_vols)} | Test volumes: {len(test_vols)}")
    print(f"Train pairs: {len(train_idx):,} | Val pairs: {len(val_idx):,} | Test pairs: {len(test_idx):,}")

    train_set = MRISliceDataset(inputs_path, targets_path, train_idx)
    val_set = MRISliceDataset(inputs_path, targets_path, val_idx)
    test_set = MRISliceDataset(inputs_path, targets_path, test_idx)

    _num_workers = 0 if platform.system() == "Windows" else 4
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=_num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=_num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=_num_workers, pin_memory=True)

    model = UNet().to(device)
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    model_path = MODELS_DIR / "unet_brats_volume_split_best.pth"
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}

    print(f"\nTraining for {NUM_EPOCHS} epochs...\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = val_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"{epoch:>3} | train={train_loss:.6f} | val={val_loss:.6f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"    >> saved best model ({best_val:.6f})")

    plot_training_curves(history, save_path=FIGURES_DIR / "training_curves_brats_volume_split.png")

    model.load_state_dict(torch.load(model_path, map_location=device))

    metrics = evaluate_test_set_filtered(model, test_loader, device)
    np.save(METRICS_DIR / "unet_test_metrics_brats_volume_split.npy", metrics, allow_pickle=True)
    print_metrics_summary(metrics)

    visualize_predictions_filtered(
        model,
        test_set,
        device,
        n_samples=4,
        save_path=FIGURES_DIR / "predictions_brats_volume_split.png",
    )

    plot_metric_distributions(
        {
            "per_sample_psnr": metrics["per_sample_psnr"],
            "per_sample_ssim": metrics["per_sample_ssim"],
            "per_sample_mae": metrics["per_sample_mae"],
        },
        save_path=FIGURES_DIR / "metric_distributions_brats_volume_split.png",
    )


if __name__ == "__main__":
    main()