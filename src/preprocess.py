#!/usr/bin/env python
"""
Preprocessing pipeline: load raw NIfTI MRI scans, build 2.5D slice pairs,
and compute interpolation baselines.

Outputs written to:
  data/processed/dataset_inputs.npy        (N, 2, 256, 256)
  data/processed/dataset_targets.npy       (N, 256, 256)
  outputs/metrics/baseline_metrics_linear.npy
  outputs/metrics/baseline_metrics_spline.npy
  outputs/figures/baseline_distributions.png
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import interpolate
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

from src.config import (
    ACCELERATION_FACTOR,
    FIGURES_DIR,
    METRICS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    TARGET_SIZE,
)


# ---------------------------------------------------------------------------
# Volume utilities
# ---------------------------------------------------------------------------

def resize_volume(vol):
    """Resize every axial slice of a (H, W, S) volume to TARGET_SIZE."""
    h, w, num_slices = vol.shape
    resized = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], num_slices), dtype=np.float32)
    for i in range(num_slices):
        resized[:, :, i] = resize(
            vol[:, :, i], TARGET_SIZE,
            mode="reflect", anti_aliasing=True, preserve_range=True,
        )
    return resized


def normalize_volume(vol):
    """Min-max normalize a volume to [0, 1]."""
    v_min, v_max = vol.min(), vol.max()
    if v_max - v_min == 0:
        return vol.astype(np.float32)
    return ((vol - v_min) / (v_max - v_min)).astype(np.float32)


def load_and_preprocess(filepath):
    """Load a NIfTI file and return preprocessed slices of shape (S, 256, 256)."""
    img    = nib.load(filepath)
    vol    = img.get_fdata()
    vol    = resize_volume(vol)
    vol    = normalize_volume(vol)          # (256, 256, S)
    slices = np.moveaxis(vol, -1, 0)        # (S, 256, 256)
    return slices


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def simulate_acquisition(slices, acceleration_factor=ACCELERATION_FACTOR):
    """Return lists of acquired and missing slice indices for a given AF."""
    total        = len(slices)
    acquired_idx = list(range(0, total, acceleration_factor))
    missing_idx  = [i for i in range(total) if i not in acquired_idx]
    return acquired_idx, missing_idx


def build_25D_dataset(slices, acceleration_factor=ACCELERATION_FACTOR):
    """
    Build (input, target) pairs for 2.5D interpolation.

    Each input is [left_slice, right_slice] (shape 2×256×256).
    Each target is the ground-truth missing slice between them.
    """
    acquired_idx, _ = simulate_acquisition(slices, acceleration_factor)
    inputs, targets, meta = [], [], []
    for i in range(len(acquired_idx) - 1):
        left_idx, right_idx = acquired_idx[i], acquired_idx[i + 1]
        for mid_idx in range(left_idx + 1, right_idx):
            inputs.append(np.stack([slices[left_idx], slices[right_idx]], axis=0))
            targets.append(slices[mid_idx])
            meta.append({"left": left_idx, "right": right_idx, "target": mid_idx})
    return np.array(inputs), np.array(targets), meta


# ---------------------------------------------------------------------------
# Baseline interpolation methods
# ---------------------------------------------------------------------------

def linear_interpolation(slices, acceleration_factor=ACCELERATION_FACTOR):
    acquired_idx, _ = simulate_acquisition(slices, acceleration_factor)
    recon = slices.copy()
    for i in range(len(acquired_idx) - 1):
        left_idx, right_idx = acquired_idx[i], acquired_idx[i + 1]
        between = list(range(left_idx + 1, right_idx))
        for j, mid_idx in enumerate(between):
            t = (j + 1) / (len(between) + 1)
            recon[mid_idx] = (1 - t) * slices[left_idx] + t * slices[right_idx]
    return recon


def spline_interpolation(slices, acceleration_factor=ACCELERATION_FACTOR, kind="cubic"):
    acquired_idx, missing_idx = simulate_acquisition(slices, acceleration_factor)
    H, W = slices.shape[1], slices.shape[2]
    acquired_data = slices[acquired_idx].reshape(len(acquired_idx), -1)
    f = interpolate.interp1d(
        acquired_idx, acquired_data, axis=0,
        kind=kind, fill_value="extrapolate",
    )
    recon = slices.copy()
    if missing_idx:
        predicted = np.clip(f(missing_idx), 0, 1)
        recon[missing_idx] = predicted.reshape(-1, H, W)
    return recon


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_reconstruction(original, reconstructed, missing_indices):
    psnr_scores, ssim_scores = [], []
    for idx in missing_indices:
        orig  = original[idx]
        recon = np.clip(reconstructed[idx], 0, 1)
        if np.allclose(orig, recon):
            continue
        psnr_scores.append(psnr(orig, recon, data_range=1.0))
        ssim_scores.append(ssim(orig, recon, data_range=1.0))
    return {
        "psnr_mean": np.mean(psnr_scores) if psnr_scores else float("nan"),
        "psnr_std" : np.std(psnr_scores)  if psnr_scores else float("nan"),
        "ssim_mean": np.mean(ssim_scores) if ssim_scores else float("nan"),
        "ssim_std" : np.std(ssim_scores)  if ssim_scores else float("nan"),
    }


def aggregate_metrics(metrics_list):
    psnr_means = [m["psnr_mean"] for m in metrics_list if not np.isnan(m["psnr_mean"])]
    ssim_means = [m["ssim_mean"] for m in metrics_list if not np.isnan(m["ssim_mean"])]
    return {
        "psnr_mean"    : np.mean(psnr_means),
        "psnr_std"     : np.std(psnr_means),
        "ssim_mean"    : np.mean(ssim_means),
        "ssim_std"     : np.std(ssim_means),
        "per_scan_psnr": psnr_means,
        "per_scan_ssim": ssim_means,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_baseline_distributions(agg_lin, agg_spl, n_files, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"Baseline Quality Distribution across {n_files} Scans (×{ACCELERATION_FACTOR})",
        fontsize=13,
    )
    for ax, (lin, spl, metric) in zip(axes, [
        (agg_lin["per_scan_psnr"], agg_spl["per_scan_psnr"], "Mean PSNR per scan (dB)"),
        (agg_lin["per_scan_ssim"], agg_spl["per_scan_ssim"], "Mean SSIM per scan"),
    ]):
        ax.hist(lin, bins=30, alpha=0.6, label="Linear",       color="steelblue")
        ax.hist(spl, bins=30, alpha=0.6, label="Cubic Spline", color="orange")
        ax.axvline(np.mean(lin), color="steelblue", linestyle="--", linewidth=1.5)
        ax.axvline(np.mean(spl), color="orange",    linestyle="--", linewidth=1.5)
        ax.set_xlabel(metric)
        ax.set_ylabel("Number of scans")
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
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    nii_files = sorted(
        glob.glob(str(RAW_DIR / "*.nii")) +
        glob.glob(str(RAW_DIR / "*.nii.gz"))
    )
    print(f"Found {len(nii_files)} NIfTI files in {RAW_DIR}")
    if not nii_files:
        print("No files found. Place raw .nii / .nii.gz scans in data/raw/ and re-run.")
        return

    all_inputs, all_targets = [], []
    all_metrics_linear, all_metrics_spline = [], []
    failed_files = []

    print(f"Processing {len(nii_files)} files...\n")
    for file_idx, filepath in enumerate(nii_files):
        fname = os.path.basename(filepath)
        try:
            slices = load_and_preprocess(filepath)
            inputs, targets, _ = build_25D_dataset(slices, ACCELERATION_FACTOR)
            all_inputs.append(inputs)
            all_targets.append(targets)

            _, missing_idx = simulate_acquisition(slices, ACCELERATION_FACTOR)
            recon_lin      = linear_interpolation(slices, ACCELERATION_FACTOR)
            recon_spl      = spline_interpolation(slices, ACCELERATION_FACTOR)
            m_lin          = evaluate_reconstruction(slices, recon_lin, missing_idx)
            m_spl          = evaluate_reconstruction(slices, recon_spl, missing_idx)
            all_metrics_linear.append(m_lin)
            all_metrics_spline.append(m_spl)

            print(
                f"[{file_idx+1:>3}/{len(nii_files)}] {fname:<55} "
                f"slices={len(slices):>3}  pairs={len(inputs):>4}  "
                f"PSNR_lin={m_lin['psnr_mean']:.2f}dB  "
                f"PSNR_spl={m_spl['psnr_mean']:.2f}dB"
            )
        except Exception as e:
            print(f"[{file_idx+1:>3}/{len(nii_files)}] FAILED: {fname} — {e}")
            failed_files.append(filepath)

    dataset_inputs  = np.concatenate(all_inputs,  axis=0)
    dataset_targets = np.concatenate(all_targets, axis=0)

    np.save(PROCESSED_DIR / "dataset_inputs.npy",  dataset_inputs)
    np.save(PROCESSED_DIR / "dataset_targets.npy", dataset_targets)
    np.save(METRICS_DIR   / "baseline_metrics_linear.npy", all_metrics_linear)
    np.save(METRICS_DIR   / "baseline_metrics_spline.npy", all_metrics_spline)

    agg_lin = aggregate_metrics(all_metrics_linear)
    agg_spl = aggregate_metrics(all_metrics_spline)

    print("\n========== AGGREGATE BASELINE ==========")
    print(f"{'Method':<20} {'PSNR (dB)':>14} {'SSIM':>14}")
    print("-" * 50)
    print(
        f"{'Linear Interp.':<20} {agg_lin['psnr_mean']:>8.2f} ± {agg_lin['psnr_std']:.2f}"
        f"  {agg_lin['ssim_mean']:>6.4f} ± {agg_lin['ssim_std']:.4f}"
    )
    print(
        f"{'Cubic Spline':<20} {agg_spl['psnr_mean']:>8.2f} ± {agg_spl['psnr_std']:.2f}"
        f"  {agg_spl['ssim_mean']:>6.4f} ± {agg_spl['ssim_std']:.4f}"
    )

    plot_baseline_distributions(
        agg_lin, agg_spl, len(nii_files),
        save_path=FIGURES_DIR / "baseline_distributions.png",
    )

    print(f"\nOutputs saved to {PROCESSED_DIR}")
    print(f"  dataset_inputs.npy  : {dataset_inputs.shape}")
    print(f"  dataset_targets.npy : {dataset_targets.shape}")
    if failed_files:
        print(f"  {len(failed_files)} file(s) failed")


if __name__ == "__main__":
    main()
