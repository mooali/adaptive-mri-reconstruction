#!/usr/bin/env python
"""
src/analyze_brats_reconstruction.py

Evaluate a trained model on BraTS robust arrays and compare tumour vs healthy
target slices using brats_meta_rb.npy, after patient-level split and the same
informative-slice filtering used in train_brats_volume_split.py.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from src.config import PROCESSED_DIR, MODELS_DIR, RANDOM_SEED, TRAIN_SPLIT, VAL_SPLIT
from src.train import UNet

MIN_FOREGROUND_FRACTION = 0.01
MIN_TARGET_STD = 0.02


def is_informative_target(target,
                          min_foreground_fraction=MIN_FOREGROUND_FRACTION,
                          min_target_std=MIN_TARGET_STD):
    target = np.asarray(target, dtype=np.float32)
    foreground_fraction = np.mean(target > 0)
    return (foreground_fraction >= min_foreground_fraction) and (target.std() >= min_target_std)


class IndexedDataset(Dataset):
    def __init__(self, inputs_path, targets_path, meta_path, volume_ids_path, indices):
        self.inputs = np.load(inputs_path, mmap_mode="r")
        self.targets = np.load(targets_path, mmap_mode="r")
        self.meta = np.load(meta_path, mmap_mode="r")
        self.volume_ids = np.load(volume_ids_path, mmap_mode="r")
        self.indices = np.array(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        x = torch.tensor(np.array(self.inputs[i]), dtype=torch.float32)
        y = torch.tensor(np.array(self.targets[i]), dtype=torch.float32).unsqueeze(0)
        m = bool(self.meta[i])
        v = str(self.volume_ids[i])
        return x, y, m, v, i


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
    return train_idx, val_idx, test_idx


def summarize(rows):
    ps = np.array([r["psnr"] for r in rows], dtype=float)
    ss = np.array([r["ssim"] for r in rows], dtype=float)
    ma = np.array([r["mae"] for r in rows], dtype=float)

    return {
        "n": int(len(rows)),
        "psnr_mean": float(ps.mean()) if len(ps) else float("nan"),
        "psnr_std": float(ps.std()) if len(ps) else float("nan"),
        "ssim_mean": float(ss.mean()) if len(ss) else float("nan"),
        "ssim_std": float(ss.std()) if len(ss) else float("nan"),
        "mae_mean": float(ma.mean()) if len(ma) else float("nan"),
        "mae_std": float(ma.std()) if len(ma) else float("nan"),
    }


def print_stats(title, stats):
    print(f"\n{title}")
    print(f"n = {stats['n']}")
    print(f"{'Metric':<10}{'Mean':>12}{'Std':>12}")
    print("-" * 34)
    print(f"{'PSNR':<10}{stats['psnr_mean']:>10.2f}dB{stats['psnr_std']:>12.2f}")
    print(f"{'SSIM':<10}{stats['ssim_mean']:>12.4f}{stats['ssim_std']:>12.4f}")
    print(f"{'MAE':<10}{stats['mae_mean']:>12.5f}{stats['mae_std']:>12.5f}")


def main():
    inputs_path = PROCESSED_DIR / "brats_inputs_rb.npy"
    targets_path = PROCESSED_DIR / "brats_targets_rb.npy"
    meta_path = PROCESSED_DIR / "brats_meta_rb.npy"
    volume_ids_path = PROCESSED_DIR / "brats_volume_ids_rb.npy"
    model_path = MODELS_DIR / "unet_brats_volume_split_best.pth"

    volume_ids = np.load(volume_ids_path)
    _, _, test_idx = split_by_volume(volume_ids)

    ds = IndexedDataset(inputs_path, targets_path, meta_path, volume_ids_path, test_idx)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    rows = []
    skipped_trivial = 0

    with torch.no_grad():
        for x, y, m, v, idx in loader:
            pred = model(x.to(device)).cpu().numpy()
            gt = y.numpy()

            for i in range(len(pred)):
                p = pred[i, 0]
                g = gt[i, 0]

                if not is_informative_target(g):
                    skipped_trivial += 1
                    continue

                rows.append({
                    "index": int(idx[i]),
                    "volume_id": str(v[i]),
                    "is_tumor": bool(m[i]),
                    "psnr": float(psnr(g, p, data_range=1.0)),
                    "ssim": float(ssim(g, p, data_range=1.0)),
                    "mae": float(np.mean(np.abs(g - p))),
                })

    all_stats = summarize(rows)
    tumor_stats = summarize([r for r in rows if r["is_tumor"]])
    healthy_stats = summarize([r for r in rows if not r["is_tumor"]])

    print("\n========== BRATS TUMOUR VS HEALTHY ANALYSIS ==========")
    print(f"Informative filter: foreground >= {MIN_FOREGROUND_FRACTION:.3f}, std >= {MIN_TARGET_STD:.3f}")
    print(f"Used {len(rows)} filtered test slices")
    print(f"Skipped trivial test slices: {skipped_trivial}")

    print_stats("All filtered test slices", all_stats)
    print_stats("Tumour target slices", tumor_stats)
    print_stats("Healthy target slices", healthy_stats)

    print("\nTumour - Healthy deltas")
    print(f"PSNR delta: {tumor_stats['psnr_mean'] - healthy_stats['psnr_mean']:.2f} dB")
    print(f"SSIM delta: {tumor_stats['ssim_mean'] - healthy_stats['ssim_mean']:.4f}")
    print(f"MAE delta:  {tumor_stats['mae_mean'] - healthy_stats['mae_mean']:.5f}")


if __name__ == "__main__":
    main()