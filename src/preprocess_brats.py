#!/usr/bin/env python
"""
src/preprocess_brats.py — Build 2.5D reconstruction pairs from BraTS2020 H5 slices.

Reads per-slice H5 files, groups them by patient volume, applies the same
resize + volume-level normalisation pipeline used for IXI, then builds 2.5D
input-target pairs using simulate_acquisition / build_25D_dataset from
preprocess.py.

Outputs
-------
  data/processed/brats_inputs.npy    shape (N, 2, 256, 256)  float32
  data/processed/brats_targets.npy   shape (N, 256, 256)     float32

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
    ACCELERATION_FACTOR,
    BRATS_DIR,
    BRATS_INPUTS_PATH,
    BRATS_TARGETS_PATH,
)
from src.preprocess import resize_volume, normalize_volume, build_25D_dataset


# ---------------------------------------------------------------------------
# Volume loading helper
# ---------------------------------------------------------------------------

def _load_volume(slice_files, channel):
    """
    Stack sorted per-slice H5 files into a (H, W, D) volume.

    Parameters
    ----------
    slice_files : list[str]   sorted H5 paths for one patient
    channel     : int         modality index (0=FLAIR, 1=T1, 2=T1ce, 3=T2)

    Returns
    -------
    vol : np.ndarray  (H, W, D)  float32
    """
    imgs = []
    for fp in slice_files:
        with h5py.File(fp, "r") as f:
            img = f["image"][()]               # (H, W, 4)
        imgs.append(img[:, :, channel].astype(np.float32))
    return np.stack(imgs, axis=2)              # (H, W, D)


def _preprocess_volume(vol):
    """
    Resize to 256×256 and normalise the full volume to [0, 1].

    Returns (D, 256, 256) float32.
    """
    vol = resize_volume(vol)      # (256, 256, D)
    vol = normalize_volume(vol)   # volume-level min-max → [0, 1]
    return np.moveaxis(vol, -1, 0)  # (D, 256, 256)


def _count_pairs(n_slices, acceleration_factor=ACCELERATION_FACTOR):
    """Number of 2.5D pairs produced by build_25D_dataset for a volume with n_slices."""
    n_acquired = len(range(0, n_slices, acceleration_factor))
    return max(0, n_acquired - 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess BraTS2020 H5 slices → 2.5D reconstruction pairs."
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

    inputs_out  = BRATS_INPUTS_PATH
    targets_out = BRATS_TARGETS_PATH

    if inputs_out.exists() and targets_out.exists() and not args.force:
        print("Preprocessed arrays already exist:")
        print(f"  {inputs_out}  shape={np.load(inputs_out, mmap_mode='r').shape}")
        print(f"  {targets_out}  shape={np.load(targets_out, mmap_mode='r').shape}")
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
    print(f"Acceleration: ×{ACCELERATION_FACTOR}")

    # ── Phase 1: count total pairs ─────────────────────────────────────────
    print("\nPhase 1 — counting reconstruction pairs per volume...")
    n_pairs_total = 0
    failed        = []

    for i, vkey in enumerate(volume_keys):
        try:
            n_slices = len(vol_map[vkey])
            n_pairs_total += _count_pairs(n_slices)
        except Exception as e:
            print(f"  [{i+1}] FAILED vol={vkey}: {e}")
            failed.append(vkey)
        if (i + 1) % 100 == 0:
            print(f"  Counted {i+1}/{len(volume_keys)} volumes  pairs so far={n_pairs_total:,}")

    for k in failed:
        volume_keys.remove(k)

    print(f"\nTotal reconstruction pairs: {n_pairs_total:,}")
    if n_pairs_total == 0:
        raise RuntimeError("No pairs found — check --brats-dir and --channel.")

    # ── Phase 2: pre-allocate memory-mapped arrays ─────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    H, W = TARGET_SIZE

    print(f"\nPhase 2 — pre-allocating arrays on disk...")
    mm_inputs  = np.lib.format.open_memmap(
        str(inputs_out),  mode="w+", dtype=np.float32, shape=(n_pairs_total, 2, H, W)
    )
    mm_targets = np.lib.format.open_memmap(
        str(targets_out), mode="w+", dtype=np.float32, shape=(n_pairs_total, H, W)
    )
    size_in  = n_pairs_total * 2 * H * W * 4 / 1e9
    size_tgt = n_pairs_total *     H * W * 4 / 1e9
    print(f"  {inputs_out.name}   {mm_inputs.shape}   ({size_in:.2f} GB)")
    print(f"  {targets_out.name}  {mm_targets.shape}  ({size_tgt:.2f} GB)")

    # ── Phase 3: fill arrays ───────────────────────────────────────────────
    print(f"\nPhase 3 — preprocessing volumes and writing pairs...")
    cursor = 0

    for i, vkey in enumerate(volume_keys):
        try:
            vol    = _load_volume(vol_map[vkey], channel=args.channel)
            slices = _preprocess_volume(vol)              # (D, 256, 256)
            inputs, targets, _ = build_25D_dataset(slices, ACCELERATION_FACTOR)

            n = len(inputs)
            if n:
                mm_inputs[cursor : cursor + n]  = inputs
                mm_targets[cursor : cursor + n] = targets
                cursor += n

        except Exception as e:
            print(f"  [{i+1}] FAILED vol={vkey}: {e}")

        if (i + 1) % 50 == 0 or (i + 1) == len(volume_keys):
            print(f"  [{i+1:>3}/{len(volume_keys)}]  written pairs={cursor:>7,}")

    # Trim if any volumes failed during phase 3
    if cursor < n_pairs_total:
        print(f"\nTrimming arrays: {n_pairs_total} → {cursor} pairs")
        tmp_in  = inputs_out.with_name("_tmp_inputs.npy")
        tmp_tgt = targets_out.with_name("_tmp_targets.npy")
        final_in = np.lib.format.open_memmap(
            str(tmp_in),  mode="w+", dtype=np.float32, shape=(cursor, 2, H, W)
        )
        final_tgt = np.lib.format.open_memmap(
            str(tmp_tgt), mode="w+", dtype=np.float32, shape=(cursor, H, W)
        )
        final_in[:]  = mm_inputs[:cursor]
        final_tgt[:] = mm_targets[:cursor]
        del mm_inputs, mm_targets, final_in, final_tgt
        tmp_in.replace(inputs_out)
        tmp_tgt.replace(targets_out)
    else:
        del mm_inputs, mm_targets

    print(f"\nDone.")
    print(f"  {inputs_out}   shape=({cursor}, 2, {H}, {W})")
    print(f"  {targets_out}  shape=({cursor}, {H}, {W})")
    print(f"\nNext step — train U-Net on BraTS reconstruction pairs:")
    print(f"  python -m src.train --data brats")


if __name__ == "__main__":
    main()
