#!/usr/bin/env python
"""
src/compare_brats_stratified.py

Stratified evaluation of U-Net vs baselines on BraTS test set.
Uses the correctly preprocessed AF=2 pairs (same as training).
Reports metrics for: All slices | Non-tumour | Tumour-containing.
"""
import numpy as np
import torch
from pathlib import Path
from scipy.ndimage import zoom
from scipy import interpolate
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Dataset

from src.config import PROCESSED_DIR, MODELS_DIR, RANDOM_SEED, TRAIN_SPLIT, VAL_SPLIT
from src.train import UNet

BATCH_SIZE = 32

# ── Load preprocessed data ────────────────────────────────────────────────────
def load_test_split():
    inputs     = np.load(PROCESSED_DIR / "brats_inputs_rb.npy",     mmap_mode="r")  # (N, 2, H, W)
    targets    = np.load(PROCESSED_DIR / "brats_targets_rb.npy",    mmap_mode="r")  # (N, H, W)
    volume_ids = np.load(PROCESSED_DIR / "brats_volume_ids_rb.npy", mmap_mode="r")  # (N,) str
    tumour     = np.load(PROCESSED_DIR / "brats_meta_rb.npy",       mmap_mode="r")  # (N,) bool

    # Reproduce exact test split
    unique_vols = np.unique(volume_ids)
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(unique_vols)
    n_train = int(TRAIN_SPLIT * len(shuffled))
    n_val   = int(VAL_SPLIT   * len(shuffled))
    test_vols = set(shuffled[n_train + n_val:])

    mask = np.array([v in test_vols for v in volume_ids])
    idx  = np.where(mask)[0]

    return inputs, targets, tumour, idx

# ── Dataset ───────────────────────────────────────────────────────────────────
class BraTSDataset(Dataset):
    def __init__(self, inputs, targets, idx):
        self.inputs  = inputs
        self.targets = targets
        self.idx     = idx
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        ri = self.idx[i]
        x = torch.tensor(np.array(self.inputs[ri]),  dtype=torch.float32)   # (2, H, W)
        y = torch.tensor(np.array(self.targets[ri]), dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        return x, y

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute(gt, pred):
    g = np.asarray(gt,   np.float32)
    p = np.clip(np.asarray(pred, np.float32), 0, 1)
    return psnr(g, p, data_range=1.0), ssim(g, p, data_range=1.0), float(np.abs(g - p).mean())

def summarize(psnrs, ssims, maes):
    if not psnrs:
        return
    p, s, m = np.array(psnrs), np.array(ssims), np.array(maes)
    return {"n": len(p),
            "psnr": (p.mean(), p.std()),
            "ssim": (s.mean(), s.std()),
            "mae":  (m.mean(), m.std())}

def print_table(title, results):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")
    print(f"{'Method':<18}{'n':>6}  {'PSNR':>12}  {'SSIM':>10}  {'MAE':>10}")
    print(f"{'─'*65}")
    for name, s in results:
        if s is None:
            continue
        print(f"{name:<18}{s['n']:>6}  {s['psnr'][0]:>8.2f} dB  "
              f"{s['ssim'][0]:>10.4f}  {s['mae'][0]:>10.5f}")
    lin  = next((s for n, s in results if "Linear" in n), None)
    unet = next((s for n, s in results if "U-Net"  in n), None)
    if lin and unet:
        print(f"\n  U-Net vs Linear:  "
              f"ΔPSNR = {unet['psnr'][0]-lin['psnr'][0]:+.2f} dB  "
              f"ΔSSIM = {unet['ssim'][0]-lin['ssim'][0]:+.4f}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading preprocessed test split...")
    inputs, targets, tumour, idx = load_test_split()
    print(f"Test samples: {len(idx)}  (tumour: {tumour[idx].sum()}, non-tumour: {(~tumour[idx]).sum()})")

    print("Loading U-Net...")
    model = UNet().to(device)
    model.load_state_dict(torch.load(
        MODELS_DIR / "unet_brats_volume_split_best.pth", map_location=device))
    model.eval()

    # ── Collect all predictions ────────────────────────────────────────────────
    ds     = BraTSDataset(inputs, targets, idx)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    lin_p, lin_s, lin_m   = [], [], []
    spl_p, spl_s, spl_m   = [], [], []
    unet_p, unet_s, unet_m = [], [], []
    is_tumour_all = tumour[idx]   # boolean array, aligned with idx order

    print("Evaluating...")
    sample_i = 0
    with torch.no_grad():
        for x, y in loader:
            B = x.shape[0]
            left  = x[:, 0].numpy()   # (B, H, W)
            right = x[:, 1].numpy()
            gt    = y[:, 0].numpy()   # (B, H, W)

            # Linear: midpoint
            lin_pred = 0.5 * left + 0.5 * right

            # Cubic spline: per-sample along 3-point axis [left, target, right]
            # We treat indices [0, 1, 2] and predict at 1 — but we only have
            # left and right (0 and 2), so spline = linear for 3 points.
            # Instead, use the actual stored triplet structure:
            # spline through (0→left, 2→right) evaluated at 1 → same as linear.
            # For a meaningful spline we need 4+ points, which we don't have here.
            # Use linear as spline proxy (identical to original baseline script).
            spl_pred = lin_pred  # 3-point spline collapses to linear

            # U-Net
            unet_pred = model(x.to(device)).cpu().numpy()[:, 0]  # (B, H, W)

            for i in range(B):
                g = gt[i]
                
                if g.std() < 0.02 or np.mean(g > 0) < 0.01:
                    continue
            
                lp, ls, lm = compute(gt[i], lin_pred[i])
                lin_p.append(lp); lin_s.append(ls); lin_m.append(lm)

                sp, ss, sm = compute(gt[i], spl_pred[i])
                spl_p.append(sp); spl_s.append(ss); spl_m.append(sm)

                up, us, um = compute(gt[i], unet_pred[i])
                unet_p.append(up); unet_s.append(us); unet_m.append(um)

            sample_i += B

    is_tumour = np.array(is_tumour_all, dtype=bool)

    # ── Print stratified tables ───────────────────────────────────────────────
    for title, mask in [
        ("ALL TEST SLICES",           np.ones(len(idx), dtype=bool)),
        ("NON-TUMOUR SLICES",         ~is_tumour),
        ("TUMOUR-CONTAINING SLICES",   is_tumour),
    ]:
        m = mask
        print_table(title, [
            ("Linear interp", summarize(
                [v for v,k in zip(lin_p,  m) if k],
                [v for v,k in zip(lin_s,  m) if k],
                [v for v,k in zip(lin_m,  m) if k])),
            ("Cubic spline",  summarize(
                [v for v,k in zip(spl_p,  m) if k],
                [v for v,k in zip(spl_s,  m) if k],
                [v for v,k in zip(spl_m,  m) if k])),
            ("U-Net (ours)",  summarize(
                [v for v,k in zip(unet_p, m) if k],
                [v for v,k in zip(unet_s, m) if k],
                [v for v,k in zip(unet_m, m) if k])),
        ])

if __name__ == "__main__":
    main()