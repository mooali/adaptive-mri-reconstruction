#!/usr/bin/env python
"""
src/evaluate_brats.py — Anomaly detection evaluation on BraTS2020.

Purpose
-------
Compares the anomaly score (autoencoder reconstruction MAE) distributions of:
  • IXI healthy slices  — the distribution the autoencoder was trained on
  • BraTS2020 T1 slices — pathological brains with gliomas / glioblastomas

A well-calibrated anomaly detector should produce systematically higher MAE
on BraTS slices, with clear separation around the calibrated threshold
(mean + 2σ of IXI training-set MAE).  This evaluation validates that the
module generalises beyond its training distribution.

Usage
-----
  python -m src.evaluate_brats
  python -m src.evaluate_brats --brats-dir D:/data/brats2020
  python -m src.evaluate_brats --brats-dir data/brats --max-files 20

BraTS H5 format
---------------
The script auto-detects two common BraTS2020 HDF5 layouts:
  • (H, W, D, 4)  channels-last  — e.g. awsaf49/brats20-dataset-training-validation
  • (4, H, W, D)  channels-first

In both cases the four channels are [FLAIR, T1, T1ce, T2] so T1 is index 1.
If your files use a different layout, pass --t1-channel accordingly.

Outputs
-------
  outputs/figures/anomaly_brats_vs_ixi.png   — overlapping MAE distributions
  (printed to stdout)                         — per-file summary and statistics

Dependencies
------------
  h5py             : reading BraTS HDF5 files  (pip install h5py)
  numpy, torch, matplotlib
  src.config            : ROOT_DIR, paths, TARGET_SIZE
  src.preprocess        : resize_volume, normalize_volume
  src.adaptive_decision : SliceAutoencoder, train_anomaly_detector
"""

import argparse
import glob
import os
import pathlib

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import (
    ROOT_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    ANOMALY_DETECTOR_PATH,
    ANOMALY_THRESHOLD_PATH,
    TARGET_SIZE,
    BRATS_DIR,
    BRATS_HEALTHY_PATH,
    BRATS_TUMOR_PATH,
    BRATS_DETECTOR_PATH,
    BRATS_THRESHOLD_PATH,
)
from src.adaptive_decision import SliceAutoencoder, train_anomaly_detector
from src.preprocess import resize_volume, normalize_volume


# ---------------------------------------------------------------------------
# BraTS loading
# ---------------------------------------------------------------------------

def load_volume_from_slices(slice_files, t1_channel=1):
    """
    Reconstruct a full 3-D T1 volume from a list of per-slice BraTS H5 files.

    Each file contains one axial 2-D slice as (H, W, 4) channels-last.
    Files are assumed to be sorted in ascending slice-index order.
    Stacking them along axis=2 gives (H, W, D) — the same layout IXI uses —
    so that normalize_volume operates across the whole volume (not per-slice).
    This matches the IXI preprocessing that the autoencoder was trained on.

    Also collects the tumour mask (key 'mask', shape (H, W, 3)) to identify
    which slices contain labelled pathology.

    Parameters
    ----------
    slice_files : list[str]  — sorted H5 paths for one patient volume
    t1_channel  : int        — modality index for T1 (default 1)

    Returns
    -------
    vol        : np.ndarray  shape (H, W, D)  float32  — raw T1 stack
    has_tumor  : np.ndarray  shape (D,)       bool     — True if slice has any mask label
    """
    slices_img  = []
    has_tumor   = []
    for fp in slice_files:
        with h5py.File(fp, "r") as f:
            img  = f["image"][()]   # (H, W, 4)
            mask = f["mask"][()] if "mask" in f else None
        slices_img.append(img[:, :, t1_channel].astype(np.float32))
        has_tumor.append(bool(mask is not None and mask.any()))
    vol = np.stack(slices_img, axis=2)   # (H, W, D)
    return vol, np.array(has_tumor, dtype=bool)


def preprocess_brats_volume(vol):
    """
    Apply the same resize + volume-level normalise pipeline as IXI preprocessing.

    Parameters
    ----------
    vol : np.ndarray  shape (H, W, D)  — raw T1 intensities

    Returns
    -------
    np.ndarray  shape (D, 256, 256)  float32, values in [0, 1]
    """
    vol    = resize_volume(vol)       # (256, 256, D)  float32
    vol    = normalize_volume(vol)    # volume-level min-max → [0, 1]
    slices = np.moveaxis(vol, -1, 0)  # (D, 256, 256)
    return slices


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_slices(autoencoder, slices, device, batch_size=64, foreground_only=False,
                 score_percentile=100):
    """
    Compute per-slice autoencoder reconstruction error scores.

    Parameters
    ----------
    autoencoder       : SliceAutoencoder  — must be in eval() mode
    slices            : np.ndarray  shape (S, H, W)
    device            : torch.device
    batch_size        : int
    foreground_only   : bool — if True, restrict scoring to non-zero pixels
                        (brain tissue mask for skull-stripped BraTS data).
    score_percentile  : int in [1, 100].
                        100 → mean foreground MAE (default, original behaviour).
                        <100 → use that percentile of per-pixel errors in the
                        foreground region instead of the mean.  E.g. 95 gives
                        the 95th-percentile pixel error, which captures
                        localised high-error regions (tumours) without dilution
                        from surrounding healthy tissue.

    Returns
    -------
    np.ndarray  shape (S,)  float32  — per-slice anomaly score
    """
    autoencoder.eval()
    all_scores = []
    use_mean = (score_percentile >= 100)
    with torch.no_grad():
        for start in range(0, len(slices), batch_size):
            batch = slices[start : start + batch_size][:, np.newaxis, :, :]  # (B,1,H,W)
            x     = torch.tensor(batch, dtype=torch.float32).to(device)
            recon = autoencoder(x)
            err   = torch.abs(recon - x)   # (B, 1, H, W)
            err_np = err.cpu().numpy()      # (B, 1, H, W)
            x_np   = x.cpu().numpy()

            if use_mean:
                if foreground_only:
                    fg_mask  = (x_np > 0).astype(np.float32)
                    fg_count = fg_mask.sum(axis=(1, 2, 3)).clip(min=1)
                    scores   = (err_np * fg_mask).sum(axis=(1, 2, 3)) / fg_count
                else:
                    scores = err_np.mean(axis=(1, 2, 3))
            else:
                # Percentile over foreground pixels only (or all pixels).
                scores = np.empty(len(err_np), dtype=np.float32)
                for b in range(len(err_np)):
                    pixels = err_np[b, 0]          # (H, W)
                    if foreground_only:
                        fg = pixels[x_np[b, 0] > 0]
                        pixels = fg if len(fg) else pixels.ravel()
                    scores[b] = np.percentile(pixels, score_percentile)

            all_scores.extend(scores.tolist())
    return np.array(all_scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# BraTS-domain autoencoder training
# ---------------------------------------------------------------------------

# (BRATS_DETECTOR_PATH and BRATS_THRESHOLD_PATH now imported from src.config)


def train_on_brats(vol_map, train_keys, device, t1_channel=1,
                   epochs=10, batch_size=64, lr=1e-3, threshold_sigma=2.0,
                   score_percentile=100, threshold_percentile=None,
                   use_preprocessed=False):
    """
    Train the SliceAutoencoder on non-tumour BraTS slices and calibrate threshold.

    When use_preprocessed=True and data/processed/brats_healthy.npy exists the
    function loads slices directly from that array (fast, avoids re-reading H5
    files).  Otherwise it loads raw H5 files on-the-fly from vol_map/train_keys.

    Only slices whose mask is entirely zero are used — these represent healthy
    brain tissue in the BraTS domain.  Training on this in-domain distribution
    gives the autoencoder a tight model of normal skull-stripped brain anatomy,
    so that tumour slices (unusual texture / signal) produce higher MAE.

    The calibrated threshold is mean + 2σ of the training-set MAE, matching
    the same calibration strategy used for the IXI autoencoder.

    Parameters
    ----------
    vol_map          : dict[str → list[str]]  — per-volume sorted H5 file lists
    train_keys       : list[str]              — patient volume IDs to train on
    device           : torch.device
    t1_channel       : int                    — modality index (default 1 = T1)
    epochs           : int
    batch_size       : int
    lr               : float
    use_preprocessed : bool — load from brats_healthy.npy instead of H5 files

    Returns
    -------
    autoencoder : SliceAutoencoder  (eval mode)
    threshold   : float
    """
    import torch.nn as nn

    # ── Collect all non-tumour slices ───────────────────────────────────────
    if use_preprocessed and BRATS_HEALTHY_PATH.exists():
        print(f"  Loading preprocessed healthy slices from {BRATS_HEALTHY_PATH.name}...")
        healthy = np.load(BRATS_HEALTHY_PATH, mmap_mode="r")
        # Draw a random 70% subset for training to match the vol-split behaviour
        np.random.seed(42)
        idx     = np.random.permutation(len(healthy))[:int(len(healthy) * 0.70)]
        healthy = np.array(healthy[idx])  # load selected rows into RAM
        print(f"  Training set: {len(healthy):,} non-tumour slices (from preprocessed array)")
    else:
        print(f"  Collecting non-tumour slices from {len(train_keys)} training volumes...")
        healthy_slices = []
        for vkey in train_keys:
            try:
                vol, has_tumor = load_volume_from_slices(vol_map[vkey], t1_channel=t1_channel)
                slices = preprocess_brats_volume(vol)    # (D, 256, 256)
                healthy_slices.append(slices[~has_tumor])
            except Exception as e:
                print(f"    Skipping vol={vkey}: {e}")

        if not healthy_slices:
            raise RuntimeError("No healthy BraTS slices collected for training.")

        healthy = np.concatenate(healthy_slices, axis=0)   # (N, 256, 256)
        print(f"  Training set: {len(healthy):,} non-tumour slices")

    # ── Train ──────────────────────────────────────────────────────────────
    autoencoder = SliceAutoencoder().to(device)
    optimizer   = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    criterion   = nn.L1Loss()

    autoencoder.train()
    for epoch in range(epochs):
        perm       = np.random.permutation(len(healthy))
        total_loss = 0.0
        n_batches  = 0
        for start in range(0, len(healthy), batch_size):
            idx   = perm[start : start + batch_size]
            batch = healthy[idx][:, np.newaxis, :, :]   # (B, 1, H, W)
            x     = torch.tensor(batch, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            loss  = criterion(autoencoder(x), x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        print(f"    Epoch {epoch+1:>2}/{epochs}  loss={total_loss/n_batches:.6f}")

    # ── Calibrate threshold on training set ───────────────────────────────
    autoencoder.eval()
    errors = score_slices(autoencoder, healthy, device, foreground_only=True,
                          score_percentile=score_percentile)
    if threshold_percentile is not None:
        threshold = float(np.percentile(errors, threshold_percentile))
        print(f"  Calibrated threshold: {threshold:.6f}  "
              f"(p{threshold_percentile} of non-tumour training scores)")
    else:
        threshold = float(errors.mean() + threshold_sigma * errors.std())
        print(f"  Calibrated threshold: {threshold:.6f}  "
              f"(μ={errors.mean():.6f}, σ={errors.std():.6f}, k={threshold_sigma})")

    # ── Save ──────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(autoencoder.state_dict(), BRATS_DETECTOR_PATH)
    np.save(BRATS_THRESHOLD_PATH, np.array([threshold]))
    print(f"  Saved → {BRATS_DETECTOR_PATH.name}")

    return autoencoder, threshold


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_comparison(ixi_scores, brats_notum_scores, brats_tumor_scores, brats_threshold, save_path):
    """
    Overlapping histograms: IXI healthy | BraTS non-tumour | BraTS tumour.

    Parameters
    ----------
    ixi_scores         : np.ndarray  — per-slice full-image MAE for IXI
    brats_notum_scores : np.ndarray  — per-slice foreground MAE, non-tumour BraTS slices
    brats_tumor_scores : np.ndarray  — per-slice foreground MAE, tumour BraTS slices
    brats_threshold    : float       — within-domain BraTS threshold (mean+2σ of non-tumour)
    save_path          : Path
    """
    notum_flag = (brats_notum_scores > brats_threshold).mean() * 100 if len(brats_notum_scores) else float("nan")
    tumor_flag = (brats_tumor_scores > brats_threshold).mean() * 100 if len(brats_tumor_scores) else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Anomaly Detection: BraTS non-tumour vs BraTS tumour slices\n"
        f"Within-domain threshold={brats_threshold:.6f}  |  "
        f"Non-tumour flagged: {notum_flag:.1f}%  |  "
        f"Tumour flagged: {tumor_flag:.1f}%",
        fontsize=11, fontweight="bold",
    )

    # ── Left: overlapping histograms ───────────────────────────────────────
    ax = axes[0]
    if len(brats_notum_scores):
        ax.hist(brats_notum_scores, bins=60, alpha=0.6, color="steelblue",
                label=f"BraTS non-tumour (n={len(brats_notum_scores):,})")
    if len(brats_tumor_scores):
        ax.hist(brats_tumor_scores, bins=60, alpha=0.6, color="tomato",
                label=f"BraTS tumour slices (n={len(brats_tumor_scores):,})")
    ax.axvline(brats_threshold, color="black", linestyle="--", linewidth=2,
               label=f"BraTS threshold={brats_threshold:.4f}")
    if len(brats_notum_scores):
        ax.axvline(brats_notum_scores.mean(), color="steelblue", linestyle=":", linewidth=1.5,
                   label=f"Non-tumour mean={brats_notum_scores.mean():.4f}")
    if len(brats_tumor_scores):
        ax.axvline(brats_tumor_scores.mean(), color="tomato", linestyle=":", linewidth=1.5,
                   label=f"Tumour mean={brats_tumor_scores.mean():.4f}")
    ax.set_xlabel("Per-slice foreground MAE (brain tissue only)")
    ax.set_ylabel("Number of slices")
    ax.set_title("MAE Distribution (within BraTS domain)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Right: cumulative distribution ─────────────────────────────────────
    ax = axes[1]
    series = []
    if len(brats_notum_scores):
        series.append((brats_notum_scores, "steelblue", "BraTS non-tumour"))
    if len(brats_tumor_scores):
        series.append((brats_tumor_scores, "tomato", "BraTS tumour slices"))
    for scores, color, label in series:
        sorted_s = np.sort(scores)
        cdf      = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
        ax.plot(sorted_s, cdf, color=color, linewidth=2, label=label)
    ax.axvline(brats_threshold, color="black", linestyle="--", linewidth=2,
               label="BraTS threshold")
    ax.set_xlabel("Per-slice foreground MAE")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Cumulative Distribution (CDF)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BraTS2020 anomaly detection against healthy IXI baseline."
    )
    parser.add_argument(
        "--brats-dir", default="data/brats",
        help="Directory containing BraTS2020 .h5 files (default: data/brats/)",
    )
    parser.add_argument(
        "--t1-channel", type=int, default=1,
        help="Channel index for T1 modality in the H5 image array (default: 1).",
    )
    parser.add_argument(
        "--max-volumes", type=int, default=None,
        help="Cap on number of patient volumes to process (omit for all).",
    )
    parser.add_argument(
        "--retrain-on-brats", action="store_true",
        help="Retrain the autoencoder on BraTS non-tumour slices instead of using the IXI model.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Training epochs when --retrain-on-brats is set (default: 10).",
    )
    parser.add_argument(
        "--use-preprocessed", action="store_true",
        help="Load slices from data/processed/brats_healthy.npy and brats_tumor.npy "
             "(created by preprocess_brats.py) instead of reading H5 files at runtime. "
             "Much faster for repeated runs — run preprocess_brats.py first.",
    )
    parser.add_argument(
        "--train-split", type=float, default=0.70,
        help="Fraction of volumes used for training when --retrain-on-brats (default: 0.70).",
    )
    parser.add_argument(
        "--threshold-sigma", type=float, default=2.0,
        help="Threshold = mean + k*sigma of non-tumour MAE (default: 2.0). "
             "Lower values flag more aggressively (try 1.0–1.5 for higher sensitivity).",
    )
    parser.add_argument(
        "--score-percentile", type=int, default=100,
        help="Percentile of per-pixel errors used as the slice score (default: 100 = mean). "
             "Use 90–99 to focus on the worst-reconstructed region (tumour hotspot) "
             "rather than the global mean, which dilutes small localised anomalies.",
    )
    parser.add_argument(
        "--threshold-percentile", type=int, default=None,
        help="Set the anomaly threshold as the Nth percentile of non-tumour training scores "
             "(e.g. 95 guarantees ~5%% FPR on healthy tissue). "
             "When set, overrides --threshold-sigma. Recommended: 95 with --score-percentile 95.",
    )
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Resolve BraTS directory ────────────────────────────────────────────
    brats_path = pathlib.Path(args.brats_dir)
    if not brats_path.is_absolute():
        brats_path = ROOT_DIR / args.brats_dir

    h5_files = sorted(glob.glob(str(brats_path / "**" / "*.h5"), recursive=True))
    if not h5_files:
        h5_files = sorted(glob.glob(str(brats_path / "*.h5")))
    if not h5_files:
        print(f"No .h5 files found under {brats_path}")
        print("Place BraTS2020 H5 files there or pass --brats-dir <path>.")
        return

    # Group per-slice files by patient volume (volume_XXX_slice_YYY.h5 → key XXX)
    import collections, re as _re
    vol_map = collections.defaultdict(list)
    for fp in h5_files:
        m = _re.search(r'volume_(\w+)_slice', os.path.basename(fp))
        key = m.group(1) if m else os.path.basename(fp)
        vol_map[key].append(fp)
    volume_keys = sorted(vol_map.keys())
    # Sort slices within each volume by slice index
    for k in volume_keys:
        vol_map[k].sort(key=lambda p: int(_re.search(r'slice_(\d+)', os.path.basename(p)).group(1))
                        if _re.search(r'slice_(\d+)', os.path.basename(p)) else 0)

    if args.max_volumes:
        volume_keys = volume_keys[: args.max_volumes]
    print(f"Found {len(h5_files):,} H5 files → {len(vol_map)} patient volumes in {brats_path}")

    # ── Train/test split by volume ─────────────────────────────────────────
    np.random.seed(42)
    shuffled   = np.random.permutation(volume_keys).tolist()
    n_train    = max(1, int(len(shuffled) * args.train_split))
    train_keys = shuffled[:n_train]
    eval_keys  = volume_keys   # always evaluate on all provided volumes
    print(f"Volume split: {n_train} train / {len(shuffled)-n_train} test  "
          f"(evaluating {len(eval_keys)} volume(s))\n")

    # ── Load or train the autoencoder ──────────────────────────────────────
    autoencoder = SliceAutoencoder().to(device)

    if args.retrain_on_brats:
        if BRATS_DETECTOR_PATH.exists() and BRATS_THRESHOLD_PATH.exists():
            print(f"Loading existing BraTS autoencoder: {BRATS_DETECTOR_PATH.name}")
            autoencoder.load_state_dict(torch.load(BRATS_DETECTOR_PATH, map_location=device))
            autoencoder.eval()
            threshold = float(np.load(BRATS_THRESHOLD_PATH)[0])
            print(f"  threshold={threshold:.6f}")
        else:
            print("Training BraTS-domain autoencoder on non-tumour slices...")
            autoencoder, threshold = train_on_brats(
                vol_map, train_keys, device,
                t1_channel=args.t1_channel,
                epochs=args.epochs,
                threshold_sigma=args.threshold_sigma,
                score_percentile=args.score_percentile,
                threshold_percentile=args.threshold_percentile,
                use_preprocessed=args.use_preprocessed,
            )
            autoencoder.eval()
    else:
        if ANOMALY_DETECTOR_PATH.exists() and ANOMALY_THRESHOLD_PATH.exists():
            autoencoder.load_state_dict(torch.load(ANOMALY_DETECTOR_PATH, map_location=device))
            autoencoder.eval()
            threshold = float(np.load(ANOMALY_THRESHOLD_PATH)[0])
            print(f"Loaded IXI autoencoder: {ANOMALY_DETECTOR_PATH.name}  (threshold={threshold:.6f})")
        else:
            print("Anomaly detector not found — training from scratch on IXI data...")
            inputs = np.load(PROCESSED_DIR / "dataset_inputs.npy", mmap_mode="r")
            autoencoder, threshold = train_anomaly_detector(inputs, device)
            autoencoder.eval()

    # ── IXI baseline (only shown when using IXI model) ─────────────────────
    if not args.retrain_on_brats:
        print("\nScoring IXI healthy slices for reference...")
        inputs     = np.load(PROCESSED_DIR / "dataset_inputs.npy", mmap_mode="r")
        ixi_idx    = np.arange(0, len(inputs), 20)
        ixi_slices = np.array(inputs[ixi_idx, 0])
        ixi_scores = score_slices(autoencoder, ixi_slices, device, foreground_only=False,
                                   score_percentile=args.score_percentile)
        print(f"  IXI: {len(ixi_scores):,} slices  mean={ixi_scores.mean():.6f}  "
              f"std={ixi_scores.std():.6f}")
    else:
        ixi_scores = np.array([])

    # ── BraTS scoring ──────────────────────────────────────────────────────
    if args.use_preprocessed and BRATS_HEALTHY_PATH.exists() and BRATS_TUMOR_PATH.exists():
        # Fast path: load from pre-built numpy arrays (skip H5 I/O entirely)
        print("Scoring from preprocessed numpy arrays...")
        healthy_arr = np.load(BRATS_HEALTHY_PATH, mmap_mode="r")
        tumor_arr   = np.load(BRATS_TUMOR_PATH,   mmap_mode="r")
        print(f"  healthy: {len(healthy_arr):,} slices   tumor: {len(tumor_arr):,} slices")

        brats_notum_scores = score_slices(autoencoder, healthy_arr, device,
                                          foreground_only=True,
                                          score_percentile=args.score_percentile)
        brats_tumor_scores = score_slices(autoencoder, tumor_arr, device,
                                          foreground_only=True,
                                          score_percentile=args.score_percentile)
        failed = []
    else:
        # Slow path: load H5 files on-the-fly, grouped by volume
        print(f"Scoring {len(eval_keys)} BraTS patient volume(s) from H5 files...")
        brats_all_scores        = []
        brats_tumor_scores_list = []
        brats_notum_scores_list = []
        failed = []

        for i, vkey in enumerate(eval_keys):
            slice_files = vol_map[vkey]
            try:
                vol, has_tumor = load_volume_from_slices(slice_files, t1_channel=args.t1_channel)
                slices = preprocess_brats_volume(vol)
                scores = score_slices(autoencoder, slices, device, foreground_only=True,
                                       score_percentile=args.score_percentile)
                brats_all_scores.append(scores)
                if has_tumor.any():
                    brats_tumor_scores_list.append(scores[has_tumor])
                if (~has_tumor).any():
                    brats_notum_scores_list.append(scores[~has_tumor])
                tum_mean   = scores[has_tumor].mean()  if has_tumor.any()  else float("nan")
                notum_mean = scores[~has_tumor].mean() if (~has_tumor).any() else float("nan")
                print(
                    f"  [{i+1:>3}/{len(eval_keys)}] vol={vkey:<6}  "
                    f"slices={len(slices):>3}  tumour={has_tumor.sum():>3}  "
                    f"tum_mae={tum_mean:.6f}  notum_mae={notum_mean:.6f}"
                )
            except Exception as e:
                print(f"  [{i+1:>3}/{len(eval_keys)}] FAILED: vol={vkey} — {e}")
                failed.append(vkey)

        if not brats_all_scores:
            print("No BraTS volumes processed successfully. Aborting.")
            return

        brats_tumor_scores = np.concatenate(brats_tumor_scores_list) if brats_tumor_scores_list else np.array([])
        brats_notum_scores = np.concatenate(brats_notum_scores_list) if brats_notum_scores_list else np.array([])

    # ── Within-domain BraTS threshold ─────────────────────────────────────
    # Because IXI is not skull-stripped the autoencoder's IXI-calibrated
    # threshold (mean+2σ on full-image MAE) is not comparable to BraTS
    # foreground MAE.  Calibrate a second threshold on BraTS non-tumor slices
    # (mean + 2σ) so the separation between healthy and pathological BraTS
    # slices is measured on a level playing field.
    if len(brats_notum_scores) >= 10:
        if args.threshold_percentile is not None:
            brats_threshold = float(np.percentile(brats_notum_scores, args.threshold_percentile))
            print(f"\n  BraTS within-domain threshold: {brats_threshold:.6f}  "
                  f"(p{args.threshold_percentile} of non-tumour scores)")
        else:
            brats_threshold = float(brats_notum_scores.mean() + args.threshold_sigma * brats_notum_scores.std())
            print(f"\n  BraTS within-domain threshold: {brats_threshold:.6f}  "
                  f"(μ={brats_notum_scores.mean():.6f}, σ={brats_notum_scores.std():.6f}, k={args.threshold_sigma})")
    else:
        brats_threshold = threshold   # fall back to IXI threshold if too few healthy slices
        print("\n  Warning: too few non-tumour slices to calibrate BraTS threshold; using IXI threshold.")

    # ── Summary ───────────────────────────────────────────────────────────
    def _pct(arr, thr):
        return 100 * (arr > thr).mean() if len(arr) else float("nan")

    print("\n" + "=" * 75)
    print("ANOMALY DETECTION SUMMARY")
    print("=" * 75)
    print(f"  {'Dataset':<30} {'Slices':>8} {'Mean MAE':>12} {'Std':>8}  IXI-thr  BraTS-thr")
    print("-" * 75)
    if len(ixi_scores):
        print(
            f"  {'IXI (healthy, full-image)':<30} {len(ixi_scores):>8,} "
            f"{ixi_scores.mean():>12.6f} {ixi_scores.std():>8.6f}"
        )
    else:
        print(f"  {'IXI (healthy, full-image)':<30} {'—':>8}  {'(BraTS-domain mode)':>21}")
    print(
        f"  {'BraTS non-tumour (fg MAE)':<30} {len(brats_notum_scores):>8,} "
        f"{brats_notum_scores.mean():>12.6f} {brats_notum_scores.std():>8.6f}  "
        f"{_pct(brats_notum_scores, threshold):>5.1f}%   {_pct(brats_notum_scores, brats_threshold):>5.1f}%"
        if len(brats_notum_scores) else f"  {'BraTS non-tumour':<30} {'N/A':>8}"
    )
    print(
        f"  {'BraTS tumour slices (fg MAE)':<30} {len(brats_tumor_scores):>8,} "
        f"{brats_tumor_scores.mean():>12.6f} {brats_tumor_scores.std():>8.6f}  "
        f"{_pct(brats_tumor_scores, threshold):>5.1f}%   {_pct(brats_tumor_scores, brats_threshold):>5.1f}%"
        if len(brats_tumor_scores) else f"  {'BraTS tumour slices':<30} {'N/A':>8}"
    )
    print(f"\n  IXI-calibrated threshold (full-image MAE)  : {threshold:.6f}")
    print(f"  BraTS within-domain threshold (fg MAE)     : {brats_threshold:.6f}")
    if len(brats_tumor_scores) and len(brats_notum_scores):
        sep = brats_tumor_scores.mean() - brats_notum_scores.mean()
        flag_diff = _pct(brats_tumor_scores, brats_threshold) - _pct(brats_notum_scores, brats_threshold)
        print(f"\n  Tumour vs non-tumour MAE separation : {sep:+.6f}  "
              f"({'tumour higher ✓' if sep > 0 else 'no separation — possible calibration issue'})")
        print(f"  Extra flagging on tumour slices     : {flag_diff:+.1f} pp  "
              f"(using BraTS within-domain threshold)")
    print("=" * 75)

    if failed:
        print(f"\n  {len(failed)} volume(s) failed to load.")

    # ── Plot ──────────────────────────────────────────────────────────────
    save_path = FIGURES_DIR / "anomaly_brats_vs_ixi.png"
    plot_comparison(ixi_scores, brats_notum_scores, brats_tumor_scores, brats_threshold, save_path)


if __name__ == "__main__":
    main()
