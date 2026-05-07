#!/usr/bin/env python
"""
src/preprocess_brats.py — Preprocess BraTS2020 H5 slices into numpy arrays.

Reads per-slice H5 files, groups them by patient volume, applies the same
resize + volume-level normalisation pipeline used for IXI data, then splits
slices by tumour label and saves two memory-mapped numpy arrays:

  data/processed/brats_healthy.npy   shape (S_h, 256, 256)  float32
  data/processed/brats_tumor.npy     shape (S_t, 256, 256)  float32

The healthy array is the training corpus for the BraTS-domain autoencoder.
The tumor array is the test set for anomaly detection evaluation.

Usage
-----
  python -m src.preprocess_brats
  python -m src.preprocess_brats --brats-dir data/brats --channel 1
  python -m src.preprocess_brats --channel 0 --max-volumes 50
  python -m src.preprocess_brats --force   # overwrite existing arrays

BraTS H5 format
---------------
  Each file covers one axial slice: image shape (H, W, 4) channels-last.
  Channels: [FLAIR=0, T1=1, T1ce=2, T2=3]
  Mask shape (H, W, 3): any non-zero value means the slice contains tumour.
"""

import argparse
import collections
import glob
import os
import pathlib
import re

import h5py
import numpy as np

from src.config import (
    ROOT_DIR,
    PROCESSED_DIR,
    TARGET_SIZE,
    BRATS_DIR,
    BRATS_HEALTHY_PATH,
    BRATS_TUMOR_PATH,
)
from src.preprocess import resize_volume, normalize_volume


# ---------------------------------------------------------------------------
# Volume loading helpers  (same logic as evaluate_brats.py)
# ---------------------------------------------------------------------------

def _load_volume(slice_files, channel):
    """
    Stack per-slice H5 files into a (H, W, D) volume and collect tumour flags.

    Parameters
    ----------
    slice_files : list[str]   sorted H5 paths for one patient
    channel     : int         modality index (0=FLAIR, 1=T1, 2=T1ce, 3=T2)

    Returns
    -------
    vol       : np.ndarray  (H, W, D)  float32  raw intensities
    has_tumor : np.ndarray  (D,)       bool     True if slice has any mask label
    """
    imgs, flags = [], []
    for fp in slice_files:
        with h5py.File(fp, "r") as f:
            img  = f["image"][()]               # (H, W, 4)
            mask = f["mask"][()] if "mask" in f else None
        imgs.append(img[:, :, channel].astype(np.float32))
        flags.append(bool(mask is not None and mask.any()))
    vol = np.stack(imgs, axis=2)               # (H, W, D)
    return vol, np.array(flags, dtype=bool)


def _preprocess_volume(vol):
    """
    Resize to 256×256 and normalise the full volume to [0, 1].

    Returns
    -------
    slices : np.ndarray  (D, 256, 256)  float32
    """
    vol    = resize_volume(vol)        # (256, 256, D)
    vol    = normalize_volume(vol)     # volume-level min-max
    return np.moveaxis(vol, -1, 0)    # (D, 256, 256)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess BraTS2020 H5 slices → numpy arrays for training."
    )
    parser.add_argument(
        "--brats-dir", default=None,
        help="Directory with BraTS2020 .h5 files (default: data/brats/).",
    )
    parser.add_argument(
        "--channel", type=int, default=1,
        help="Modality channel: 0=FLAIR, 1=T1 (default), 2=T1ce, 3=T2.",
    )
    parser.add_argument(
        "--max-volumes", type=int, default=None,
        help="Process only this many patient volumes (for quick tests).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output arrays.",
    )
    args = parser.parse_args()

    # ── Resolve paths ──────────────────────────────────────────────────────
    brats_path = pathlib.Path(args.brats_dir) if args.brats_dir else BRATS_DIR
    if not brats_path.is_absolute():
        brats_path = ROOT_DIR / brats_path

    healthy_out = BRATS_HEALTHY_PATH
    tumor_out   = BRATS_TUMOR_PATH

    if healthy_out.exists() and tumor_out.exists() and not args.force:
        print(f"Preprocessed arrays already exist:")
        print(f"  {healthy_out}  shape={np.load(healthy_out, mmap_mode='r').shape}")
        print(f"  {tumor_out}  shape={np.load(tumor_out, mmap_mode='r').shape}")
        print("Use --force to reprocess.")
        return

    # ── Discover H5 files ──────────────────────────────────────────────────
    h5_files = sorted(glob.glob(str(brats_path / "**" / "*.h5"), recursive=True))
    if not h5_files:
        h5_files = sorted(glob.glob(str(brats_path / "*.h5")))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under {brats_path}")

    # Group per-slice files by volume key (volume_XXX_slice_YYY.h5 → XXX)
    vol_map = collections.defaultdict(list)
    for fp in h5_files:
        m = re.search(r'volume_(\w+)_slice', os.path.basename(fp))
        key = m.group(1) if m else os.path.basename(fp)
        vol_map[key].append(fp)

    volume_keys = sorted(vol_map.keys())
    for k in volume_keys:
        vol_map[k].sort(
            key=lambda p: int(re.search(r'slice_(\d+)', os.path.basename(p)).group(1))
            if re.search(r'slice_(\d+)', os.path.basename(p)) else 0
        )

    if args.max_volumes:
        volume_keys = volume_keys[: args.max_volumes]

    print(f"Found {len(h5_files):,} H5 files → {len(vol_map)} volumes  "
          f"(processing {len(volume_keys)})")
    print(f"Channel: {args.channel}  (0=FLAIR, 1=T1, 2=T1ce, 3=T2)")

    # ── Phase 1: count slices ──────────────────────────────────────────────
    print("\nPhase 1 — counting slices per volume...")
    n_healthy = 0
    n_tumor   = 0
    failed    = []

    for i, vkey in enumerate(volume_keys):
        try:
            _, has_tumor = _load_volume(vol_map[vkey], channel=args.channel)
            n_healthy += int((~has_tumor).sum())
            n_tumor   += int(has_tumor.sum())
        except Exception as e:
            print(f"  [{i+1}] FAILED vol={vkey}: {e}")
            failed.append(vkey)
        if (i + 1) % 50 == 0:
            print(f"  Counted {i+1}/{len(volume_keys)} volumes  "
                  f"(healthy={n_healthy:,}  tumor={n_tumor:,})")

    for k in failed:
        volume_keys.remove(k)

    print(f"\nTotal slices — healthy: {n_healthy:,}   tumor: {n_tumor:,}")
    if n_healthy == 0:
        raise RuntimeError("No healthy slices found — check --channel and --brats-dir.")

    # ── Phase 2: pre-allocate memory-mapped arrays ────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    H, W = TARGET_SIZE

    print(f"\nPhase 2 — pre-allocating arrays on disk...")
    mm_healthy = np.lib.format.open_memmap(
        str(healthy_out), mode="w+", dtype=np.float32, shape=(n_healthy, H, W)
    )
    mm_tumor = np.lib.format.open_memmap(
        str(tumor_out), mode="w+", dtype=np.float32, shape=(n_tumor, H, W)
    )
    size_gb_h = n_healthy * H * W * 4 / 1e9
    size_gb_t = n_tumor   * H * W * 4 / 1e9
    print(f"  {healthy_out.name}  {mm_healthy.shape}  ({size_gb_h:.2f} GB)")
    print(f"  {tumor_out.name}  {mm_tumor.shape}  ({size_gb_t:.2f} GB)")

    # ── Phase 3: fill arrays ───────────────────────────────────────────────
    print(f"\nPhase 3 — preprocessing and writing slices...")
    hi = 0   # write cursor for healthy
    ti = 0   # write cursor for tumor

    for i, vkey in enumerate(volume_keys):
        try:
            vol, has_tumor = _load_volume(vol_map[vkey], channel=args.channel)
            slices = _preprocess_volume(vol)    # (D, 256, 256)

            healthy_slices = slices[~has_tumor]
            tumor_slices   = slices[has_tumor]

            if len(healthy_slices):
                mm_healthy[hi : hi + len(healthy_slices)] = healthy_slices
                hi += len(healthy_slices)
            if len(tumor_slices):
                mm_tumor[ti : ti + len(tumor_slices)] = tumor_slices
                ti += len(tumor_slices)

        except Exception as e:
            print(f"  [{i+1}] FAILED vol={vkey}: {e}")

        if (i + 1) % 50 == 0 or (i + 1) == len(volume_keys):
            print(f"  [{i+1:>3}/{len(volume_keys)}]  written  "
                  f"healthy={hi:>6,}  tumor={ti:>6,}")

    # Trim if any volumes failed during phase 3
    if hi < n_healthy:
        print(f"  Trimming healthy array: {n_healthy} → {hi}")
        final_healthy = np.lib.format.open_memmap(
            str(healthy_out.with_name("_tmp_h.npy")), mode="w+",
            dtype=np.float32, shape=(hi, H, W)
        )
        final_healthy[:] = mm_healthy[:hi]
        del mm_healthy, final_healthy
        healthy_out.with_name("_tmp_h.npy").replace(healthy_out)
    else:
        del mm_healthy

    if ti < n_tumor:
        print(f"  Trimming tumor array: {n_tumor} → {ti}")
        final_tumor = np.lib.format.open_memmap(
            str(tumor_out.with_name("_tmp_t.npy")), mode="w+",
            dtype=np.float32, shape=(ti, H, W)
        )
        final_tumor[:] = mm_tumor[:ti]
        del mm_tumor, final_tumor
        tumor_out.with_name("_tmp_t.npy").replace(tumor_out)
    else:
        del mm_tumor

    print(f"\nDone.")
    print(f"  {healthy_out}  shape=({hi}, {H}, {W})")
    print(f"  {tumor_out}  shape=({ti}, {H}, {W})")
    print(f"\nNext step — train anomaly detector on BraTS healthy slices:")
    print(f"  python -m src.evaluate_brats --use-preprocessed --retrain-on-brats")


if __name__ == "__main__":
    main()
