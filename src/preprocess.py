#!/usr/bin/env python
"""
src/preprocess.py — Raw MRI preprocessing and 2.5D dataset construction.

Purpose
-------
Converts a folder of raw NIfTI brain volumes into a pair of numpy arrays
that the training pipeline can load directly, and benchmarks two classical
interpolation methods as baselines.

Pipeline (executed by main())
-----------------------------
  1. Discover all .nii / .nii.gz files under data/raw/
  2. For each volume:
       a. Load with nibabel → float64 array (H, W, S)
       b. Resize each axial slice to 256×256 (skimage)
       c. Min-max normalise the whole volume to [0, 1]
       d. Build 2.5D input–target pairs (simulate ×2 acceleration)
       e. Compute linear and cubic-spline baseline metrics
  3. Concatenate all pairs → dataset_inputs.npy, dataset_targets.npy
  4. Save baseline metrics and a distribution plot

Outputs
-------
  data/processed/dataset_inputs.npy          shape (N, 2, 256, 256)
  data/processed/dataset_targets.npy         shape (N, 256, 256)
  outputs/metrics/baseline_metrics_linear.npy
  outputs/metrics/baseline_metrics_spline.npy
  outputs/figures/baseline_distributions.png

Dependencies
------------
  Standard library : glob, os
  nibabel          : loading NIfTI volumes (.nii / .nii.gz)
  numpy            : array operations, saving .npy files
  scipy.interpolate: cubic spline baseline (interp1d)
  skimage.transform: anti-aliased slice resizing
  skimage.metrics  : PSNR and SSIM evaluation
  matplotlib       : baseline distribution histogram (Agg backend — headless safe)
  src.config       : RAW_DIR, PROCESSED_DIR, TARGET_SIZE, ACCELERATION_FACTOR, …
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe for HPC without a display
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
    """
    Resize every axial slice of a (H, W, S) float volume to TARGET_SIZE.

    Slices are resized independently rather than the whole 3-D volume because
    we only process the axial plane.  Anti-aliasing prevents aliasing artefacts
    when downsampling and `preserve_range=True` keeps the raw intensity scale
    intact (normalisation is done separately).

    Parameters
    ----------
    vol : np.ndarray  shape (H, W, S)  — raw voxel intensities

    Returns
    -------
    np.ndarray  shape (TARGET_SIZE[0], TARGET_SIZE[1], S)  float32
    """
    _, _, num_slices = vol.shape
    resized = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], num_slices), dtype=np.float32)
    for i in range(num_slices):
        resized[:, :, i] = resize(
            vol[:, :, i], TARGET_SIZE,
            mode="reflect",        # pads edge pixels by reflection before filtering
            anti_aliasing=True,    # applies a Gaussian pre-filter when shrinking
            preserve_range=True,   # do not rescale intensity during resize
        )
    return resized


def normalize_volume(vol):
    """
    Min-max normalise an entire volume to the range [0, 1].

    Normalisation is per-scan rather than per-slice so that relative intensity
    differences between slices are preserved.  Different MRI scanners and
    protocols produce very different absolute intensity ranges; without
    normalisation the network would need to learn those offsets instead of
    the reconstruction task.

    Handles the degenerate case (constant volume) by returning the array
    unchanged as float32.

    Parameters
    ----------
    vol : np.ndarray  — any shape, any float dtype

    Returns
    -------
    np.ndarray  same shape, dtype float32, values in [0, 1]
    """
    v_min, v_max = vol.min(), vol.max()
    if v_max - v_min == 0:
        return vol.astype(np.float32)
    return ((vol - v_min) / (v_max - v_min)).astype(np.float32)


def load_and_preprocess(filepath):
    """
    Load one NIfTI file and return its preprocessed axial slices.

    Steps:
      1. nibabel reads the file and returns a float64 volume (H, W, S).
      2. resize_volume → uniform spatial resolution (256×256 per slice).
      3. normalize_volume → intensities in [0, 1].
      4. np.moveaxis converts (H, W, S) → (S, H, W) so that the slice index
         is the leading dimension, matching PyTorch's (C, H, W) convention
         when each slice becomes a channel.

    Parameters
    ----------
    filepath : str or Path  — path to a .nii or .nii.gz file

    Returns
    -------
    np.ndarray  shape (S, 256, 256)  float32
    """
    img    = nib.load(filepath)
    vol    = img.get_fdata()          # float64, shape (H, W, S)
    vol    = resize_volume(vol)       # float32, shape (256, 256, S)
    vol    = normalize_volume(vol)    # float32, values in [0, 1]
    slices = np.moveaxis(vol, -1, 0)  # float32, shape (S, 256, 256)
    return slices


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def simulate_acquisition(slices, acceleration_factor=ACCELERATION_FACTOR):
    """
    Decide which slice indices are "acquired" and which are "missing".

    A real ×2-accelerated scan acquires every second k-space line, which
    roughly corresponds to acquiring every other slice.  Here we model it
    simply: slices at even indices (0, 2, 4, …) are acquired; odd indices
    are treated as missing and must be reconstructed.

    Parameters
    ----------
    slices            : np.ndarray  shape (S, H, W)
    acceleration_factor : int  — 2 means every 2nd slice is acquired

    Returns
    -------
    acquired_idx : list[int]  — indices of slices the scanner would collect
    missing_idx  : list[int]  — indices of slices that must be inferred
    """
    total        = len(slices)
    acquired_idx = list(range(0, total, acceleration_factor))
    missing_idx  = [i for i in range(total) if i not in acquired_idx]
    return acquired_idx, missing_idx


def build_25D_dataset(slices, acceleration_factor=ACCELERATION_FACTOR):
    """
    Convert a preprocessed volume into (input, target) pairs for training.

    The "2.5D" name reflects that we use 2D slices but provide context from
    the third dimension (the neighboring acquired slices on each side).

    For every pair of consecutive acquired slices [left, right], we create
    one training example for each missing slice that lies between them:
      - input  = np.stack([left_slice, right_slice], axis=0)  shape (2, H, W)
      - target = the ground-truth missing slice                shape (H, W)

    At ×2 acceleration there is exactly one missing slice between each pair,
    giving one example per acquired-pair interval.

    Parameters
    ----------
    slices            : np.ndarray  shape (S, H, W)
    acceleration_factor : int

    Returns
    -------
    inputs  : np.ndarray  shape (M, 2, H, W)   — stacked neighbor pairs
    targets : np.ndarray  shape (M, H, W)       — ground-truth missing slices
    meta    : list[dict]  — {'left', 'right', 'target'} index triplets for
              traceability (not saved, useful for debugging)
    """
    acquired_idx, _ = simulate_acquisition(slices, acceleration_factor)
    inputs, targets, meta = [], [], []

    for i in range(len(acquired_idx) - 1):
        left_idx, right_idx = acquired_idx[i], acquired_idx[i + 1]

        # At ×2 AF this inner loop executes exactly once (mid_idx = left+1).
        # The loop is kept general so the function works for higher AFs too.
        for mid_idx in range(left_idx + 1, right_idx):
            inputs.append(np.stack([slices[left_idx], slices[right_idx]], axis=0))
            targets.append(slices[mid_idx])
            meta.append({"left": left_idx, "right": right_idx, "target": mid_idx})

    return np.array(inputs), np.array(targets), meta


# ---------------------------------------------------------------------------
# Baseline interpolation methods
# ---------------------------------------------------------------------------

def linear_interpolation(slices, acceleration_factor=ACCELERATION_FACTOR):
    """
    Fill missing slices by weighted linear blending of their two neighbors.

    For a missing slice at position mid between acquired slices left and right:
      t   = fractional position within the gap, ranging (0, 1) exclusive
      out = (1 - t) * left_slice  +  t * right_slice

    This is the simplest possible interpolation — no curvature information,
    just a straight-line blend in intensity space.

    Parameters
    ----------
    slices            : np.ndarray  shape (S, H, W)  — original volume
    acceleration_factor : int

    Returns
    -------
    np.ndarray  same shape as slices, missing positions filled in
    """
    acquired_idx, _ = simulate_acquisition(slices, acceleration_factor)
    recon = slices.copy()

    for i in range(len(acquired_idx) - 1):
        left_idx, right_idx = acquired_idx[i], acquired_idx[i + 1]
        between = list(range(left_idx + 1, right_idx))
        for j, mid_idx in enumerate(between):
            # t=0 → identical to left slice; t=1 → identical to right slice.
            t = (j + 1) / (len(between) + 1)
            recon[mid_idx] = (1 - t) * slices[left_idx] + t * slices[right_idx]

    return recon


def spline_interpolation(slices, acceleration_factor=ACCELERATION_FACTOR, kind="cubic"):
    """
    Fill missing slices using cubic spline interpolation along the slice axis.

    Each pixel position (h, w) is treated as an independent 1-D signal sampled
    at the acquired slice indices.  scipy.interpolate.interp1d fits a spline
    through those samples and evaluates it at the missing indices.

    Implementation detail: the (S, H, W) volume is first flattened to
    (S, H*W) so interp1d can interpolate all pixels simultaneously with a
    single call (axis=0).  The result is clipped to [0, 1] because splines
    can overshoot near steep intensity gradients.

    Parameters
    ----------
    slices            : np.ndarray  shape (S, H, W)
    acceleration_factor : int
    kind              : str  — interpolation order passed to interp1d
                        ('linear', 'quadratic', 'cubic', …)

    Returns
    -------
    np.ndarray  same shape as slices
    """
    acquired_idx, missing_idx = simulate_acquisition(slices, acceleration_factor)
    H, W = slices.shape[1], slices.shape[2]

    # Flatten spatial dimensions so every pixel is a separate 1-D series.
    acquired_data = slices[acquired_idx].reshape(len(acquired_idx), -1)  # (n_acq, H*W)

    f = interpolate.interp1d(
        acquired_idx, acquired_data, axis=0,
        kind=kind,
        fill_value="extrapolate",  # allows prediction beyond the last acquired slice
    )

    recon = slices.copy()
    if missing_idx:
        predicted = np.clip(f(missing_idx), 0, 1)          # (n_miss, H*W)
        recon[missing_idx] = predicted.reshape(-1, H, W)   # back to (n_miss, H, W)

    return recon


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_reconstruction(original, reconstructed, missing_indices):
    """
    Compute PSNR and SSIM between original and reconstructed missing slices.

    Only missing slices are scored (acquired slices are never changed).
    Slices where original and reconstructed are identical (which can happen at
    volume boundaries) are silently skipped to avoid inflating the PSNR.

    Parameters
    ----------
    original       : np.ndarray  shape (S, H, W)  — ground-truth volume
    reconstructed  : np.ndarray  shape (S, H, W)  — interpolated volume
    missing_indices: list[int]   — indices to score

    Returns
    -------
    dict with keys 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std'
    Values are NaN if no scorable slices exist.
    """
    psnr_scores, ssim_scores = [], []
    for idx in missing_indices:
        orig  = original[idx]
        recon = np.clip(reconstructed[idx], 0, 1)
        if np.allclose(orig, recon):
            continue   # skip trivially identical slices (e.g. from extrapolation)
        psnr_scores.append(psnr(orig, recon, data_range=1.0))
        ssim_scores.append(ssim(orig, recon, data_range=1.0))

    return {
        "psnr_mean": np.mean(psnr_scores) if psnr_scores else float("nan"),
        "psnr_std" : np.std(psnr_scores)  if psnr_scores else float("nan"),
        "ssim_mean": np.mean(ssim_scores) if ssim_scores else float("nan"),
        "ssim_std" : np.std(ssim_scores)  if ssim_scores else float("nan"),
    }


def aggregate_metrics(metrics_list):
    """
    Aggregate per-scan metric dicts into dataset-level statistics.

    NaN entries (from scans where no valid slices existed) are excluded so
    they do not pull the mean down.

    Parameters
    ----------
    metrics_list : list[dict]  — one dict per scan from evaluate_reconstruction

    Returns
    -------
    dict with 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std',
              'per_scan_psnr' (list), 'per_scan_ssim' (list)
    """
    psnr_means = [m["psnr_mean"] for m in metrics_list if not np.isnan(m["psnr_mean"])]
    ssim_means = [m["ssim_mean"] for m in metrics_list if not np.isnan(m["ssim_mean"])]
    return {
        "psnr_mean"    : np.mean(psnr_means),
        "psnr_std"     : np.std(psnr_means),
        "ssim_mean"    : np.mean(ssim_means),
        "ssim_std"     : np.std(ssim_means),
        "per_scan_psnr": psnr_means,  # kept for per-scan histogram
        "per_scan_ssim": ssim_means,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_baseline_distributions(agg_lin, agg_spl, n_files, save_path=None):
    """
    Plot overlapping histograms of per-scan PSNR and SSIM for both baselines.

    A bimodal distribution indicates that the dataset comes from scanners with
    different protocols — the mean is then a poor summary statistic.  Vertical
    dashed lines show the overall mean for each method.

    Parameters
    ----------
    agg_lin   : dict  — output of aggregate_metrics for linear baseline
    agg_spl   : dict  — output of aggregate_metrics for spline baseline
    n_files   : int   — total number of processed scans (for title)
    save_path : Path or None  — if given, figure is saved here and closed
    """
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
    """
    Run the full preprocessing pipeline.

    Uses a three-phase approach to avoid accumulating gigabytes of data in RAM:
      Phase 1 — read only NIfTI headers to count the total number of pairs.
      Phase 2 — pre-allocate numpy memmaps on disk (zero RAM cost).
      Phase 3 — fill memmaps in-place; each scan is written then discarded.
    """
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

    # ------------------------------------------------------------------
    # Phase 1 — header-only count (no voxel data loaded)
    # ------------------------------------------------------------------
    print("Phase 1: counting pairs from NIfTI headers...")
    counted_files = []
    total_pairs = 0
    for filepath in nii_files:
        try:
            n_slices = nib.load(filepath).shape[2]   # header only, no pixel data
            acquired = list(range(0, n_slices, ACCELERATION_FACTOR))
            total_pairs += max(0, len(acquired) - 1)
            counted_files.append(filepath)
        except Exception as e:
            print(f"  Header read failed: {os.path.basename(filepath)} — {e}")

    print(f"  {len(counted_files)} files OK, {total_pairs:,} total pairs expected\n")
    if total_pairs == 0:
        print("No pairs to write. Aborting.")
        return

    # ------------------------------------------------------------------
    # Phase 2 — pre-allocate memmaps on disk (zero RAM)
    # ------------------------------------------------------------------
    inputs_path  = PROCESSED_DIR / "dataset_inputs.npy"
    targets_path = PROCESSED_DIR / "dataset_targets.npy"

    print("Phase 2: pre-allocating memmaps on disk...")
    dataset_inputs  = np.lib.format.open_memmap(
        str(inputs_path),  mode="w+", dtype=np.float32,
        shape=(total_pairs, 2, TARGET_SIZE[0], TARGET_SIZE[1]),
    )
    dataset_targets = np.lib.format.open_memmap(
        str(targets_path), mode="w+", dtype=np.float32,
        shape=(total_pairs, TARGET_SIZE[0], TARGET_SIZE[1]),
    )
    print(f"  {inputs_path.name}   shape={dataset_inputs.shape}")
    print(f"  {targets_path.name}  shape={dataset_targets.shape}\n")

    # ------------------------------------------------------------------
    # Phase 3 — processing loop writes directly into the memmaps
    # ------------------------------------------------------------------
    all_metrics_linear, all_metrics_spline = [], []
    failed_files = []
    write_idx = 0

    print(f"Phase 3: processing {len(counted_files)} files...\n")
    for file_idx, filepath in enumerate(counted_files):
        fname = os.path.basename(filepath)
        try:
            slices = load_and_preprocess(filepath)
            inputs, targets, _ = build_25D_dataset(slices, ACCELERATION_FACTOR)
            n = len(inputs)
            dataset_inputs[write_idx : write_idx + n]  = inputs
            dataset_targets[write_idx : write_idx + n] = targets
            write_idx += n

            _, missing_idx = simulate_acquisition(slices, ACCELERATION_FACTOR)
            recon_lin = linear_interpolation(slices, ACCELERATION_FACTOR)
            recon_spl = spline_interpolation(slices, ACCELERATION_FACTOR)
            m_lin     = evaluate_reconstruction(slices, recon_lin, missing_idx)
            m_spl     = evaluate_reconstruction(slices, recon_spl, missing_idx)
            all_metrics_linear.append(m_lin)
            all_metrics_spline.append(m_spl)

            print(
                f"[{file_idx+1:>3}/{len(counted_files)}] {fname:<55} "
                f"slices={len(slices):>3}  pairs={n:>4}  "
                f"PSNR_lin={m_lin['psnr_mean']:.2f}dB  "
                f"PSNR_spl={m_spl['psnr_mean']:.2f}dB"
            )
        except Exception as e:
            print(f"[{file_idx+1:>3}/{len(counted_files)}] FAILED: {fname} — {e}")
            failed_files.append(filepath)

    dataset_inputs.flush()
    dataset_targets.flush()

    # If files failed after passing the header check, trim trailing zeros.
    if write_idx < total_pairs:
        print(
            f"\nWarning: {total_pairs - write_idx} pre-allocated rows unused "
            f"({len(failed_files)} file(s) failed after header check). Trimming..."
        )
        CHUNK = 256
        tmp_in_path  = PROCESSED_DIR / "dataset_inputs.tmp.npy"
        tmp_tgt_path = PROCESSED_DIR / "dataset_targets.tmp.npy"
        trimmed_in  = np.lib.format.open_memmap(
            str(tmp_in_path),  mode="w+", dtype=np.float32,
            shape=(write_idx, 2, TARGET_SIZE[0], TARGET_SIZE[1]),
        )
        trimmed_tgt = np.lib.format.open_memmap(
            str(tmp_tgt_path), mode="w+", dtype=np.float32,
            shape=(write_idx, TARGET_SIZE[0], TARGET_SIZE[1]),
        )
        for start in range(0, write_idx, CHUNK):
            end = min(start + CHUNK, write_idx)
            trimmed_in[start:end]  = dataset_inputs[start:end]
            trimmed_tgt[start:end] = dataset_targets[start:end]
        trimmed_in.flush()
        trimmed_tgt.flush()
        del dataset_inputs, dataset_targets, trimmed_in, trimmed_tgt
        os.replace(str(tmp_in_path),  str(inputs_path))
        os.replace(str(tmp_tgt_path), str(targets_path))
    else:
        del dataset_inputs, dataset_targets

    np.save(METRICS_DIR / "baseline_metrics_linear.npy", all_metrics_linear)
    np.save(METRICS_DIR / "baseline_metrics_spline.npy", all_metrics_spline)

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
        agg_lin, agg_spl, len(counted_files),
        save_path=FIGURES_DIR / "baseline_distributions.png",
    )

    final_inputs  = np.load(inputs_path,  mmap_mode="r")
    final_targets = np.load(targets_path, mmap_mode="r")
    print(f"\nOutputs saved to {PROCESSED_DIR}")
    print(f"  dataset_inputs.npy  : {final_inputs.shape}")
    print(f"  dataset_targets.npy : {final_targets.shape}")
    if failed_files:
        print(f"  {len(failed_files)} file(s) failed")


if __name__ == "__main__":
    main()
