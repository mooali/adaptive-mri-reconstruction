#!/usr/bin/env python
"""
src/compare_brats_af4.py

Evaluation of AF=4 U-Net vs baselines on BraTS test set.
Uses brats_*_af4.npy preprocessed by preprocess_brats_af4.py.

Input shape:  (3, H, W) — [left, right, t_map]
Linear pred:  (1 - t) * left + t * right   (position-aware)
Spline pred:  same as linear (only 2 endpoints → cubic collapses to linear)
U-Net pred:   model(x) where x includes t_map as channel 2

Stratified by:
  - All slices
  - Non-tumour slices
  - Tumour-containing slices

Also breaks down by position t = 0.25 | 0.50 | 0.75
"""

import numpy as np
import torch
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Dataset

from src.config import PROCESSED_DIR, MODELS_DIR, RANDOM_SEED, TRAIN_SPLIT, VAL_SPLIT
from src.train import UNet

BATCH_SIZE  = 32
IN_CHANNELS = 3   # left, right, t_map
AF4_CKPT    = "unet_brats_af4_best.pth"   # checkpoint saved by train_brats_af4.py

# ── Load test split ───────────────────────────────────────────────────────────
def load_test_split():
    inputs   = np.load(PROCESSED_DIR / "brats_inputs_af4.npy",     mmap_mode="r")
    targets  = np.load(PROCESSED_DIR / "brats_targets_af4.npy",    mmap_mode="r")
    t_vals   = np.load(PROCESSED_DIR / "brats_t_af4.npy",          mmap_mode="r")
    tumour   = np.load(PROCESSED_DIR / "brats_meta_af4.npy",       mmap_mode="r")
    vol_ids  = np.load(PROCESSED_DIR / "brats_volume_ids_af4.npy", mmap_mode="r")

    unique_vols = np.unique(vol_ids)
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(unique_vols)
    n_train = int(TRAIN_SPLIT * len(shuffled))
    n_val   = int(VAL_SPLIT   * len(shuffled))
    test_vols = set(shuffled[n_train + n_val:])

    idx = np.where(np.array([v in test_vols for v in vol_ids]))[0]
    return inputs, targets, t_vals, tumour, idx

# ── Dataset ───────────────────────────────────────────────────────────────────
class AF4Dataset(Dataset):
    def __init__(self, inputs, targets, t_vals, idx):
        self.inputs  = inputs
        self.targets = targets
        self.t_vals  = t_vals
        self.idx     = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        ri = self.idx[i]
        x  = torch.tensor(np.array(self.inputs[ri]),  dtype=torch.float32)   # (3, H, W)
        y  = torch.tensor(np.array(self.targets[ri]), dtype=torch.float32).unsqueeze(0)
        t  = float(self.t_vals[ri])
        return x, y, t

# ── Informative filter ────────────────────────────────────────────────────────
def is_informative(g):
    return np.mean(g > 0) >= 0.01 and g.std() >= 0.02

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute(gt, pred):
    g = np.asarray(gt,   np.float32)
    p = np.clip(np.asarray(pred, np.float32), 0, 1)
    return (psnr(g, p, data_range=1.0),
            ssim(g, p, data_range=1.0),
            float(np.abs(g - p).mean()))

def summarize(rows):
    if not rows:
        return None
    ps = np.array([r[0] for r in rows])
    ss = np.array([r[1] for r in rows])
    ms = np.array([r[2] for r in rows])
    return {"n": len(rows),
            "psnr": (ps.mean(), ps.std()),
            "ssim": (ss.mean(), ss.std()),
            "mae":  (ms.mean(), ms.std())}

def print_table(title, lin, spl, unet):
    print(f"\n{'═'*68}")
    print(f"  {title}")
    print(f"{'═'*68}")
    print(f"{'Method':<18}{'n':>6}  {'PSNR':>12}  {'SSIM':>10}  {'MAE':>10}")
    print(f"{'─'*68}")
    for name, s in [("Linear interp", lin), ("Cubic spline", spl), ("U-Net (ours)", unet)]:
        if s is None:
            continue
        print(f"{name:<18}{s['n']:>6}  {s['psnr'][0]:>8.2f} dB  "
              f"{s['ssim'][0]:>10.4f}  {s['mae'][0]:>10.5f}")
    if lin and unet:
        print(f"\n  U-Net vs Linear:  "
              f"ΔPSNR = {unet['psnr'][0]-lin['psnr'][0]:+.2f} dB  "
              f"ΔSSIM = {unet['ssim'][0]-lin['ssim'][0]:+.4f}  "
              f"ΔMAE  = {unet['mae'][0]-lin['mae'][0]:+.5f}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading AF=4 test split...")
    inputs, targets, t_vals, tumour, idx = load_test_split()
    is_tumour = tumour[idx].astype(bool)
    t_arr     = t_vals[idx].astype(np.float32)
    print(f"Test samples : {len(idx)}")
    print(f"  tumour     : {is_tumour.sum()}")
    print(f"  non-tumour : {(~is_tumour).sum()}")
    print(f"  t=0.25     : {(np.abs(t_arr - 0.25) < 0.01).sum()}")
    print(f"  t=0.50     : {(np.abs(t_arr - 0.50) < 0.01).sum()}")
    print(f"  t=0.75     : {(np.abs(t_arr - 0.75) < 0.01).sum()}")

    print(f"\nLoading U-Net ({AF4_CKPT})...")
    model = UNet(in_channels=IN_CHANNELS).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / AF4_CKPT, map_location=device))
    model.eval()

    ds     = AF4Dataset(inputs, targets, t_vals, idx)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Storage: each entry is (psnr, ssim, mae) tuple + metadata
    lin_rows, spl_rows, unet_rows = [], [], []
    keep_mask   = []   # which samples pass the informative filter
    sample_i    = 0

    print("Evaluating...")
    with torch.no_grad():
        for x, y, t_batch in loader:
            B      = x.shape[0]
            left   = x[:, 0].numpy()   # (B, H, W)
            right  = x[:, 1].numpy()
            gt     = y[:, 0].numpy()
            unet_p = model(x.to(device)).cpu().numpy()[:, 0]

            for i in range(B):
                g = gt[i]
                if not is_informative(g):
                    keep_mask.append(False)
                    sample_i += 1
                    continue

                keep_mask.append(True)
                t = float(t_batch[i])

                lin_pred  = (1 - t) * left[i] + t * right[i]
                spl_pred  = lin_pred   # 2-point cubic = linear

                lin_rows.append(compute(g, lin_pred))
                spl_rows.append(compute(g, spl_pred))
                unet_rows.append(compute(g, unet_p[i]))
                sample_i += 1

    keep_mask = np.array(keep_mask, dtype=bool)
    is_tumour_f = is_tumour[keep_mask]
    t_arr_f     = t_arr[keep_mask]

    # ── Overall and stratified tables ─────────────────────────────────────────
    def subset(rows, mask):
        return [r for r, k in zip(rows, mask) if k]

    for title, mask in [
        ("AF=4 — ALL TEST SLICES",           np.ones(len(lin_rows), dtype=bool)),
        ("AF=4 — NON-TUMOUR SLICES",         ~is_tumour_f),
        ("AF=4 — TUMOUR-CONTAINING SLICES",   is_tumour_f),
    ]:
        print_table(
            title,
            summarize(subset(lin_rows,  mask)),
            summarize(subset(spl_rows,  mask)),
            summarize(subset(unet_rows, mask)),
        )

    # ── Per-position breakdown ────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("  Per-position breakdown (t = fractional position of missing slice)")
    print(f"{'─'*68}")
    print(f"{'t':>6}  {'Method':<14}{'n':>6}  {'PSNR':>12}  {'SSIM':>10}  {'MAE':>10}")
    print(f"{'─'*68}")

    for t_target in [0.25, 0.50, 0.75]:
        t_mask = np.abs(t_arr_f - t_target) < 0.01
        if not t_mask.any():
            continue
        for name, rows in [("Linear", lin_rows), ("U-Net", unet_rows)]:
            s = summarize(subset(rows, t_mask))
            if s:
                print(f"{t_target:>6.2f}  {name:<14}{s['n']:>6}  "
                      f"{s['psnr'][0]:>8.2f} dB  "
                      f"{s['ssim'][0]:>10.4f}  "
                      f"{s['mae'][0]:>10.5f}")
        print()

    np.save(PROCESSED_DIR / "af4_evaluation.npy",
            {"linear": summarize(lin_rows),
             "unet":   summarize(unet_rows),
             "is_tumour": is_tumour_f,
             "t_vals": t_arr_f},
            allow_pickle=True)
    print(f"Saved → {PROCESSED_DIR / 'af4_evaluation.npy'}")


if __name__ == "__main__":
    main()