#!/usr/bin/env python
"""
src/preprocess_brats_reconstruction_noblack.py

BraTS reconstruction preprocessing that avoids the "everything looks black"
problem by using robust foreground normalization instead of simple volume-wide
min-max normalization.

Main idea
---------
BraTS is already standardized/skull-stripped. A second naive volume min-max can
compress the useful brain intensity range when there are very bright outliers or
large black background regions. This script:

1. loads one modality per volume (default: T1ce, channel=2)
2. resizes slices to TARGET_SIZE
3. computes a foreground mask from nonzero voxels
4. normalizes only foreground intensities using robust percentiles
   (default 1st to 99th percentile)
5. clips to [0, 1]
6. builds the same 2.5D reconstruction pairs as the IXI pipeline
7. saves tumour flags and volume IDs per pair for later analysis

Outputs
-------
data/processed/brats_inputs_rb.npy
 data/processed/brats_targets_rb.npy
 data/processed/brats_meta_rb.npy
 data/processed/brats_volume_ids_rb.npy
"""

import argparse
import collections
import glob
import os
import pathlib
import re

import h5py
import numpy as np
from skimage.transform import resize

from src.config import ROOT_DIR, PROCESSED_DIR, TARGET_SIZE, ACCELERATION_FACTOR, BRATS_DIR
from src.preprocess import build_25D_dataset


BRATS_INPUTS_PATH = PROCESSED_DIR / "brats_inputs_rb.npy"
BRATS_TARGETS_PATH = PROCESSED_DIR / "brats_targets_rb.npy"
BRATS_META_PATH = PROCESSED_DIR / "brats_meta_rb.npy"
BRATS_VOLUME_IDS_PATH = PROCESSED_DIR / "brats_volume_ids_rb.npy"


def resize_volume(vol: np.ndarray, target_size=TARGET_SIZE) -> np.ndarray:
    h, w, s = vol.shape
    out = np.zeros((target_size[0], target_size[1], s), dtype=np.float32)
    for i in range(s):
        out[:, :, i] = resize(
            vol[:, :, i],
            target_size,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)
    return out


def robust_foreground_normalize(vol: np.ndarray, low_q=1.0, high_q=99.0, eps=1e-8) -> np.ndarray:
    vol = vol.astype(np.float32, copy=False)
    fg = vol > 0
    if not np.any(fg):
        return vol.astype(np.float32)

    vals = vol[fg]
    lo = float(np.percentile(vals, low_q))
    hi = float(np.percentile(vals, high_q))

    if hi <= lo + eps:
        out = np.zeros_like(vol, dtype=np.float32)
        out[fg] = 1.0
        return out

    out = np.zeros_like(vol, dtype=np.float32)
    out[fg] = (vol[fg] - lo) / (hi - lo)
    np.clip(out, 0.0, 1.0, out=out)
    return out


def load_volume(slice_files: list[str], channel: int):
    imgs, flags = [], []
    for fp in slice_files:
        with h5py.File(fp, "r") as f:
            img = f["image"][:, :, channel].astype(np.float32)
            mask = f["mask"][:] if "mask" in f else None
        imgs.append(img)
        flags.append(bool(mask is not None and mask.any()))
    vol = np.stack(imgs, axis=2)
    return vol, np.array(flags, dtype=bool)


def preprocess_volume(vol: np.ndarray, low_q=1.0, high_q=99.0) -> np.ndarray:
    vol = resize_volume(vol)
    vol = robust_foreground_normalize(vol, low_q=low_q, high_q=high_q)
    return np.moveaxis(vol, -1, 0)


def discover_volumes(brats_path: pathlib.Path):
    h5files = sorted(glob.glob(str(brats_path / "**/*.h5"), recursive=True))
    if not h5files:
        h5files = sorted(glob.glob(str(brats_path / "*.h5")))
    if not h5files:
        raise FileNotFoundError(f"No .h5 files found under {brats_path}")

    volmap = collections.defaultdict(list)
    for fp in h5files:
        base = os.path.basename(fp)
        m = re.search(r"(volume_\d+)_slice_\d+\.h5$", base)
        key = m.group(1) if m else base
        volmap[key].append(fp)

    volume_keys = sorted(volmap.keys())
    for k in volume_keys:
        volmap[k].sort(key=lambda p: int(re.search(r"_slice_(\d+)\.h5$", os.path.basename(p)).group(1)))
    return h5files, volmap, volume_keys


def count_pairs(volume_keys, volmap):
    total_pairs = 0
    for vkey in volume_keys:
        n_slices = len(volmap[vkey])
        acquired = list(range(0, n_slices, ACCELERATION_FACTOR))
        total_pairs += max(0, len(acquired) - 1)
    return total_pairs


def main():
    parser = argparse.ArgumentParser(description="Preprocess BraTS with robust foreground normalization for reconstruction.")
    parser.add_argument("--brats-dir", default=None)
    parser.add_argument("--channel", type=int, default=2, help="0=FLAIR, 1=T1, 2=T1ce (default), 3=T2")
    parser.add_argument("--low-q", type=float, default=1.0, help="Lower foreground percentile for normalization.")
    parser.add_argument("--high-q", type=float, default=99.0, help="Upper foreground percentile for normalization.")
    parser.add_argument("--max-volumes", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    brats_path = pathlib.Path(args.brats_dir) if args.brats_dir else BRATS_DIR
    if not brats_path.is_absolute():
        brats_path = ROOT_DIR / brats_path

    outputs = [BRATS_INPUTS_PATH, BRATS_TARGETS_PATH, BRATS_META_PATH, BRATS_VOLUME_IDS_PATH]
    if all(p.exists() for p in outputs) and not args.force:
        print("Robust BraTS reconstruction arrays already exist. Use --force to rebuild.")
        return

    h5files, volmap, volume_keys = discover_volumes(brats_path)
    if args.max_volumes:
        volume_keys = volume_keys[:args.max_volumes]

    print(f"Found {len(h5files)} H5 files across {len(volmap)} volumes.")
    print(f"Processing {len(volume_keys)} volumes | channel={args.channel} | robust q=({args.low_q}, {args.high_q})")

    total_pairs = count_pairs(volume_keys, volmap)
    if total_pairs == 0:
        raise RuntimeError("No reconstruction pairs found.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    H, W = TARGET_SIZE

    mm_inputs = np.lib.format.open_memmap(str(BRATS_INPUTS_PATH), mode="w+", dtype=np.float32, shape=(total_pairs, 2, H, W))
    mm_targets = np.lib.format.open_memmap(str(BRATS_TARGETS_PATH), mode="w+", dtype=np.float32, shape=(total_pairs, H, W))
    mm_meta = np.lib.format.open_memmap(str(BRATS_META_PATH), mode="w+", dtype=bool, shape=(total_pairs,))
    mm_vids = np.lib.format.open_memmap(str(BRATS_VOLUME_IDS_PATH), mode="w+", dtype="<U32", shape=(total_pairs,))

    wi = 0
    for i, vkey in enumerate(volume_keys, 1):
        vol, has_tumor = load_volume(volmap[vkey], args.channel)
        slices = preprocess_volume(vol, low_q=args.low_q, high_q=args.high_q)
        inputs, targets, meta = build_25D_dataset(slices, ACCELERATION_FACTOR)
        n = len(inputs)
        mm_inputs[wi:wi+n] = inputs
        mm_targets[wi:wi+n] = targets
        for j, entry in enumerate(meta):
            mm_meta[wi+j] = bool(has_tumor[entry['target']])
            mm_vids[wi+j] = vkey
        wi += n
        if i % 25 == 0 or i == len(volume_keys):
            print(f"[{i}/{len(volume_keys)}] pairs={wi}")

    mm_inputs.flush(); mm_targets.flush(); mm_meta.flush(); mm_vids.flush()
    print("Done.")
    print(f"  {BRATS_INPUTS_PATH.name} shape={np.load(BRATS_INPUTS_PATH, mmap_mode='r').shape}")
    print(f"  {BRATS_TARGETS_PATH.name} shape={np.load(BRATS_TARGETS_PATH, mmap_mode='r').shape}")
    print(f"  {BRATS_META_PATH.name} shape={np.load(BRATS_META_PATH, mmap_mode='r').shape}")
    print(f"  {BRATS_VOLUME_IDS_PATH.name} shape={np.load(BRATS_VOLUME_IDS_PATH, mmap_mode='r').shape}")


if __name__ == "__main__":
    main()
