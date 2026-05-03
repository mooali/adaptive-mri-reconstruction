#!/usr/bin/env python
"""
Adaptive acquisition decision module (proof-of-concept).

Uses Monte Carlo Dropout to estimate per-pixel prediction uncertainty, then
applies a threshold-based safety gate to decide whether U-Net reconstruction
is reliable or whether full MRI acquisition is required.

Pipeline for each sample
------------------------
  1. mc_forward_passes  — N stochastic forward passes via bottleneck dropout hook
  2. compute_uncertainty — pixel-wise variance + global scalar score
  3. make_decision       — threshold gate: SAFE | UNSAFE
  4. save_uncertainty_plots — write visualisations to outputs/explainability/

The existing UNet is never retrained or modified. Dropout is injected only
through a temporary forward hook that is removed after the MC passes.

Reads:   models/unet_best.pth
         data/processed/dataset_inputs.npy
         data/processed/dataset_targets.npy

Writes (per sample):
         outputs/explainability/adaptive_panel_{idx}.png
         outputs/explainability/uncertainty_map_{idx}.png
         outputs/explainability/reconstructed_image_{idx}.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from src.config import (
    PROCESSED_DIR,
    MODELS_DIR,
    EXPLAINABILITY_DIR,
    N_EXPL_SAMPLES,
    MC_DROPOUT_PASSES,
    MC_DROPOUT_P,
    UNCERTAINTY_THRESHOLD,
)
from src.train import UNet


# ---------------------------------------------------------------------------
# MC Dropout hook
# ---------------------------------------------------------------------------

def enable_mc_dropout(model, p=MC_DROPOUT_P):
    """
    Register a temporary forward hook on the U-Net bottleneck that applies
    F.dropout with training=True, making each forward pass stochastic.

    Returns the hook handle; caller must call handle.remove() when done.
    """
    def _dropout_hook(module, input, output):
        return F.dropout(output, p=p, training=True)

    return model.bottleneck.register_forward_hook(_dropout_hook)


# ---------------------------------------------------------------------------
# MC forward passes
# ---------------------------------------------------------------------------

def mc_forward_passes(model, x, device, n_passes=MC_DROPOUT_PASSES, dropout_p=MC_DROPOUT_P):
    """
    Run n_passes stochastic forward passes through the model.

    Parameters
    ----------
    model     : UNet (eval mode, weights frozen)
    x         : (1, 2, 256, 256) input tensor (CPU)
    device    : torch.device
    n_passes  : number of stochastic samples

    Returns
    -------
    np.ndarray of shape (n_passes, H, W), dtype float32
    """
    model.eval()
    handle = enable_mc_dropout(model, p=dropout_p)

    predictions = []
    with torch.no_grad():
        for _ in range(n_passes):
            pred = model(x.to(device)).squeeze().cpu().numpy()
            predictions.append(pred)

    handle.remove()
    return np.stack(predictions, axis=0)  # (n_passes, H, W)


# ---------------------------------------------------------------------------
# Uncertainty estimation
# ---------------------------------------------------------------------------

def compute_uncertainty(predictions):
    """
    Summarise a set of stochastic predictions into a reconstruction and an
    uncertainty estimate.

    Parameters
    ----------
    predictions : np.ndarray  (N, H, W)

    Returns
    -------
    mean_pred       : np.ndarray  (H, W)  — mean reconstruction
    uncertainty_map : np.ndarray  (H, W)  — pixel-wise variance
    global_score    : float               — mean of the variance map
    """
    mean_pred       = predictions.mean(axis=0)
    uncertainty_map = predictions.var(axis=0)
    global_score    = float(uncertainty_map.mean())
    return mean_pred, uncertainty_map, global_score


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------

def make_decision(uncertainty_score, threshold=UNCERTAINTY_THRESHOLD):
    """
    Apply the safety gate.

    Returns "SAFE"   if uncertainty_score < threshold  → reconstruction applied
    Returns "UNSAFE" otherwise                         → full acquisition recommended
    """
    return "SAFE" if uncertainty_score < threshold else "UNSAFE"


def format_decision(decision, uncertainty_score):
    """Return the human-readable decision string."""
    if decision == "SAFE":
        return (
            f"Decision: SAFE — reconstruction applied  "
            f"(score={uncertainty_score:.6f} < threshold={UNCERTAINTY_THRESHOLD})"
        )
    return (
        f"Decision: UNSAFE — full acquisition recommended  "
        f"(score={uncertainty_score:.6f} >= threshold={UNCERTAINTY_THRESHOLD})"
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_uncertainty_plots(mean_pred, uncertainty_map, gt, x_np,
                            sample_idx, uncertainty_score, decision, save_dir):
    """
    Save three files for one sample:

      adaptive_panel_{idx}.png      — 6-panel summary figure
      uncertainty_map_{idx}.png     — standalone uncertainty heatmap
      reconstructed_image_{idx}.png — standalone mean reconstruction
    """
    is_safe     = decision == "SAFE"
    label_color = "green" if is_safe else "red"
    label_text  = "SAFE — reconstruction applied" if is_safe else "UNSAFE — full acquisition recommended"

    # ── 6-panel summary ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Adaptive Acquisition — Sample {sample_idx}\n"
        f"MC passes: {MC_DROPOUT_PASSES}  |  "
        f"Uncertainty: {uncertainty_score:.6f}  |  "
        f"Threshold: {UNCERTAINTY_THRESHOLD}  |  "
        f"Decision: {label_text}",
        fontsize=12, fontweight="bold", color=label_color,
    )

    panels = [
        (axes[0, 0], x_np[0],         "gray", "Input: Left Slice",                     0,    1),
        (axes[0, 1], x_np[1],         "gray", "Input: Right Slice",                    0,    1),
        (axes[0, 2], gt,              "gray", "Ground Truth",                           0,    1),
        (axes[1, 0], mean_pred,       "gray", f"Mean Reconstruction (N={MC_DROPOUT_PASSES})", 0, 1),
        (axes[1, 1], uncertainty_map, "hot",  "Uncertainty Map (pixel variance)",       None, None),
        (axes[1, 2], uncertainty_map, "hot",  "Uncertainty Overlay on Reconstruction",  None, None),
    ]

    for ax, img, cmap, title, vmin, vmax in panels:
        if title.startswith("Uncertainty Overlay"):
            ax.imshow(mean_pred, cmap="gray", vmin=0, vmax=1)
            im = ax.imshow(img, cmap="hot", alpha=0.5, vmin=0, vmax=uncertainty_map.max())
        elif vmin is not None:
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(img, cmap=cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    panel_path = save_dir / f"adaptive_panel_{sample_idx}.png"
    plt.savefig(panel_path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {panel_path.name}")
    plt.close()

    # ── standalone uncertainty map ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(uncertainty_map, cmap="hot")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Uncertainty Map — Sample {sample_idx}\n(pixel-wise variance)", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    umap_path = save_dir / f"uncertainty_map_{sample_idx}.png"
    plt.savefig(umap_path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {umap_path.name}")
    plt.close()

    # ── standalone reconstruction ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mean_pred, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Mean Reconstruction — Sample {sample_idx}\n(N={MC_DROPOUT_PASSES} passes)", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    recon_path = save_dir / f"reconstructed_image_{sample_idx}.png"
    plt.savefig(recon_path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {recon_path.name}")
    plt.close()


# ---------------------------------------------------------------------------
# Per-sample orchestration
# ---------------------------------------------------------------------------

def analyze_sample(model, inputs, targets, sample_idx, device,
                   save_dir, threshold=UNCERTAINTY_THRESHOLD):
    """
    Run the full adaptive pipeline for one sample.

    Returns (decision, uncertainty_score).
    """
    x_np = np.array(inputs[sample_idx])
    y_np = np.array(targets[sample_idx])
    x    = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)

    predictions                       = mc_forward_passes(model, x, device)
    mean_pred, uncertainty_map, score = compute_uncertainty(predictions)
    decision                          = make_decision(score, threshold)

    sample_psnr = psnr(y_np, mean_pred, data_range=1.0)
    sample_ssim = ssim(y_np, mean_pred, data_range=1.0)

    status = "SAFE  " if decision == "SAFE" else "UNSAFE"
    print(
        f"  [{status}] sample={sample_idx:>5}  "
        f"PSNR={sample_psnr:.2f}dB  SSIM={sample_ssim:.4f}  "
        f"uncertainty={score:.6f}"
    )

    save_uncertainty_plots(
        mean_pred, uncertainty_map, y_np, x_np,
        sample_idx, score, decision, save_dir,
    )

    return decision, score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    EXPLAINABILITY_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = UNet().to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "unet_best.pth", map_location=device))
    model.eval()
    print(f"Model: {MODELS_DIR / 'unet_best.pth'}")

    inputs  = np.load(PROCESSED_DIR / "dataset_inputs.npy",  mmap_mode="r")
    targets = np.load(PROCESSED_DIR / "dataset_targets.npy", mmap_mode="r")

    sample_idxs = np.linspace(0, len(inputs) - 1, N_EXPL_SAMPLES, dtype=int)

    print(f"\nSettings: MC passes={MC_DROPOUT_PASSES}  dropout_p={MC_DROPOUT_P}  threshold={UNCERTAINTY_THRESHOLD}")
    print(f"Analysing {N_EXPL_SAMPLES} samples...\n")

    results = []
    for idx in sample_idxs:
        decision, score = analyze_sample(
            model, inputs, targets, int(idx), device, EXPLAINABILITY_DIR,
        )
        results.append((int(idx), decision, score))
        print()

    # ── summary ───────────────────────────────────────────────────────────
    n_safe   = sum(1 for _, d, _ in results if d == "SAFE")
    n_unsafe = len(results) - n_safe
    scores   = [s for _, _, s in results]

    print("=" * 55)
    print("ADAPTIVE ACQUISITION SUMMARY")
    print("=" * 55)
    print(f"  Samples analysed            : {len(results)}")
    print(f"  SAFE   (reconstruct)        : {n_safe}  ({100*n_safe/len(results):.0f}%)")
    print(f"  UNSAFE (full acquisition)   : {n_unsafe}  ({100*n_unsafe/len(results):.0f}%)")
    print(f"  Mean uncertainty score      : {np.mean(scores):.6f}")
    print(f"  Min / Max uncertainty       : {np.min(scores):.6f} / {np.max(scores):.6f}")
    print(f"  Threshold                   : {UNCERTAINTY_THRESHOLD}")
    print("=" * 55)
    print(f"\nOutputs → {EXPLAINABILITY_DIR}")


if __name__ == "__main__":
    main()
