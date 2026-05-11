#!/usr/bin/env python
"""
src/preprocess_brats_af4.py

BraTS AF=4 reconstruction preprocessing.
Acceleration Factor = 4: acquires every 4th slice, reconstructs the 3 in
between at fractional positions t = 0.25, 0.50, 0.75.

Differences from AF=2 (preprocess_brats_reconstruction_noblack.py):
  - Stride = 4 instead of 2
  - 3 samples per acquired pair instead of 1
  - Input shape: (3, H, W) — [left, right, t_map] where t_map is a constant
    plane filled with the fractional position t.  The U-Net uses this to know
    which of the 3 missing slices it is predicting.
  - Output files use _af4 suffix to avoid overwriting AF=2 data.

Outputs
-------
data/processed/brats_inputs_af4.npy      (N, 3, H, W)  float32
data/processed/brats_targets_af4.npy     (N, H, W)     float32
data/processed/brats_meta_af4.npy        (N,)           bool  (tumour flag)
data/processed/brats_t_af4.npy           (N,)           float32 (position t)
data/processed/brats_volume_ids_af4.npy  (N,)           <U32
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

from src.config import ROOT_DIR, PROCESSED_DIR, TARGET_SIZE, BRATS_DIR

AF = 4   # acceleration factor

BRATS_INPUTS_PATH    = PROCESSED_DIR / "brats_inputs_af4.npy"
BRATS_TARGETS_PATH   = PROCESSED_DIR / "brats_targets_af4.npy"
BRATS_META_PATH      = PROCESSED_DIR / "brats_meta_af4.npy"
BRATS_T_PATH         = PROCESSED_DIR / "brats_t_af4.npy"
BRATS_VOLUME_IDS_PATH = PROCESSED_DIR / "brats_volume_ids_af4.npy"


# ── Helpers (identical to original) ──────────────────────────────────────────
def resize_volume(vol: np.ndarray, target_size=TARGET_SIZE) -> np.ndarray:
    h, w, s = vol.shape
    out = np.zeros((target_size[0], target_size[1], s), dtype=np.float32)
    for i in range(s):
        out[:, :, i] = resize(
            vol[:, :, i], target_size,
            mode="reflect", anti_aliasing=True, preserve_range=True,
        ).astype(np.float32)
    return out


def robust_foreground_normalize(vol: np.ndarray, low_q=1.0, high_q=99.0, eps=1e-8) -> np.ndarray:
    vol = vol.astype(np.float32, copy=False)
    fg  = vol > 0
    if not np.any(fg):
        return vol.astype(np.float32)
    vals = vol[fg]
    lo, hi = float(np.percentile(vals, low_q)), float(np.percentile(vals, high_q))
    if hi <= lo + eps:
        out = np.zeros_like(vol, dtype=np.float32)
        out[fg] = 1.0
        return out
    out = np.zeros_like(vol, dtype=np.float32)
    out[fg] = (vol[fg] - lo) / (hi - lo)
    np.clip(out, 0.0, 1.0, out=out)
    return out


def load_volume(slice_files: list, channel: int):
    imgs, flags = [], []
    for fp in slice_files:
        with h5py.File(fp, "r") as f:
            img  = f["image"][:, :, channel].astype(np.float32)
            mask = f["mask"][:] if "mask" in f else None
        imgs.append(img)
        flags.append(bool(mask is not None and mask.any()))
    vol = np.stack(imgs, axis=2)   # (H, W, D)
    return vol, np.array(flags, dtype=bool)


def preprocess_volume(vol: np.ndarray, low_q=1.0, high_q=99.0) -> np.ndarray:
    vol = resize_volume(vol)
    vol = robust_foreground_normalize(vol, low_q=low_q, high_q=high_q)
    return np.moveaxis(vol, -1, 0)   # (D, H, W)


def discover_volumes(brats_path: pathlib.Path):
    h5files = sorted(glob.glob(str(brats_path / "**/*.h5"), recursive=True))
    if not h5files:
        h5files = sorted(glob.glob(str(brats_path / "*.h5")))
    if not h5files:
        raise FileNotFoundError(f"No .h5 files found under {brats_path}")
    volmap = collections.defaultdict(list)
    for fp in h5files:
        base = os.path.basename(fp)
        m    = re.search(r"(volume_\d+)_slice_\d+\.h5$", base)
        key  = m.group(1) if m else base
        volmap[key].append(fp)
    volume_keys = sorted(volmap.keys())
    for k in volume_keys:
        volmap[k].sort(key=lambda p: int(
            re.search(r"_slice_(\d+)\.h5$", os.path.basename(p)).group(1)))
    return h5files, volmap, volume_keys


# ── AF=4 pair builder ─────────────────────────────────────────────────────────
def build_af4_dataset(slices: np.ndarray, tumour_flags: np.ndarray):
    """
    Build AF=4 samples from a preprocessed volume.

    For each pair of acquired slices (stride=4), generate one sample per
    missing slice at positions t = offset / gap.

    Parameters
    ----------
    slices       : (D, H, W) float32 in [0, 1]
    tumour_flags : (D,) bool

    Returns
    -------
    inputs   : list of (3, H, W) arrays  [left, right, t_map]
    targets  : list of (H, W) arrays
    t_values : list of float  (fractional positions)
    tumour   : list of bool   (tumour flag of target slice)
    """
    D, H, W = slices.shape
    acquired = list(range(0, D, AF))
    inputs, targets, t_values, tumour = [], [], [], []

    for i in range(len(acquired) - 1):
        li = acquired[i]
        ri = acquired[i + 1]
        gap = ri - li   # always AF unless near the end

        for offset in range(1, gap):
            mid = li + offset
            t   = offset / gap   # 0.25, 0.50, 0.75 for gap=4

            left  = slices[li]   # (H, W)
            right = slices[ri]
            t_map = np.full((H, W), t, dtype=np.float32)

            inputs.append(np.stack([left, right, t_map], axis=0))  # (3, H, W)
            targets.append(slices[mid])
            t_values.append(t)
            tumour.append(bool(tumour_flags[mid]) if mid < len(tumour_flags) else False)

    return inputs, targets, t_values, tumour


def count_samples(volume_keys, volmap):
    total = 0
    for vkey in volume_keys:
        n = len(volmap[vkey])
        acquired = list(range(0, n, AF))
        for i in range(len(acquired) - 1):
            total += acquired[i + 1] - acquired[i] - 1
    return total


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BraTS AF=4 preprocessing.")
    parser.add_argument("--brats-dir",  default=None)
    parser.add_argument("--channel",    type=int,   default=2,    help="0=FLAIR,1=T1,2=T1ce,3=T2")
    parser.add_argument("--low-q",      type=float, default=1.0)
    parser.add_argument("--high-q",     type=float, default=99.0)
    parser.add_argument("--max-volumes", type=int,  default=None)
    parser.add_argument("--force",      action="store_true")
    args = parser.parse_args()

    brats_path = pathlib.Path(args.brats_dir) if args.brats_dir else BRATS_DIR
    if not brats_path.is_absolute():
        brats_path = ROOT_DIR / brats_path

    outputs = [BRATS_INPUTS_PATH, BRATS_TARGETS_PATH,
               BRATS_META_PATH, BRATS_T_PATH, BRATS_VOLUME_IDS_PATH]
    if all(p.exists() for p in outputs) and not args.force:
        print("AF=4 arrays already exist. Use --force to rebuild.")
        return

    h5files, volmap, volume_keys = discover_volumes(brats_path)
    if args.max_volumes:
        volume_keys = volume_keys[:args.max_volumes]

    total = count_samples(volume_keys, volmap)
    print(f"Volumes: {len(volume_keys)}  |  Expected samples: {total}")
    print(f"Channel: {args.channel}  |  AF={AF}  |  Input channels: 3 (left, right, t_map)")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    H, W = TARGET_SIZE

    mm_inputs  = np.lib.format.open_memmap(str(BRATS_INPUTS_PATH),     mode="w+", dtype=np.float32, shape=(total, 3, H, W))
    mm_targets = np.lib.format.open_memmap(str(BRATS_TARGETS_PATH),    mode="w+", dtype=np.float32, shape=(total, H, W))
    mm_meta    = np.lib.format.open_memmap(str(BRATS_META_PATH),       mode="w+", dtype=bool,       shape=(total,))
    mm_t       = np.lib.format.open_memmap(str(BRATS_T_PATH),          mode="w+", dtype=np.float32, shape=(total,))
    mm_vids    = np.lib.format.open_memmap(str(BRATS_VOLUME_IDS_PATH), mode="w+", dtype="<U32",     shape=(total,))

    wi = 0
    for i, vkey in enumerate(volume_keys, 1):
        try:
            vol, has_tumor = load_volume(volmap[vkey], args.channel)
            slices = preprocess_volume(vol, low_q=args.low_q, high_q=args.high_q)
            inp, tgt, t_vals, tum = build_af4_dataset(slices, has_tumor)
        except Exception as e:
            print(f"  SKIP {vkey}: {e}")
            continue

        n = len(inp)
        if n == 0:
            continue

        mm_inputs [wi:wi+n] = np.stack(inp,  axis=0)
        mm_targets[wi:wi+n] = np.stack(tgt,  axis=0)
        mm_meta   [wi:wi+n] = np.array(tum,  dtype=bool)
        mm_t      [wi:wi+n] = np.array(t_vals, dtype=np.float32)
        mm_vids   [wi:wi+n] = vkey
        wi += n

        if i % 25 == 0 or i == len(volume_keys):
            print(f"  [{i}/{len(volume_keys)}]  samples written: {wi}")

    # Trim if any volumes were skipped
    if wi < total:
        print(f"Trimming arrays from {total} to {wi} (some volumes skipped)")
        for path, dtype, extra_dims in [
            (BRATS_INPUTS_PATH,     np.float32, (3, H, W)),
            (BRATS_TARGETS_PATH,    np.float32, (H, W)),
            (BRATS_META_PATH,       bool,       ()),
            (BRATS_T_PATH,          np.float32, ()),
            (BRATS_VOLUME_IDS_PATH, "<U32",     ()),
        ]:
            shape = (wi,) + (extra_dims if isinstance(extra_dims, tuple) and extra_dims != () else ())
            tmp = path.with_name("_tmp_" + path.name)
            mm  = np.lib.format.open_memmap(str(tmp), mode="w+", dtype=dtype, shape=shape)
            mm[:] = np.load(path, mmap_mode="r")[:wi]
            del mm
            tmp.replace(path)

    for mm in [mm_inputs, mm_targets, mm_meta, mm_t, mm_vids]:
        try:
            mm.flush()
        except Exception:
            pass

    print("\nDone.")
    print(f"  inputs   {np.load(BRATS_INPUTS_PATH,     mmap_mode='r').shape}")
    print(f"  targets  {np.load(BRATS_TARGETS_PATH,    mmap_mode='r').shape}")
    print(f"  meta     {np.load(BRATS_META_PATH,       mmap_mode='r').shape}")
    print(f"  t vals   {np.load(BRATS_T_PATH,          mmap_mode='r').shape}")
    print(f"  vol IDs  {np.load(BRATS_VOLUME_IDS_PATH, mmap_mode='r').shape}")


if __name__ == "__main__":
    main()