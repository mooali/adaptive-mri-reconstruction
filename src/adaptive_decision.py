#!/usr/bin/env python
"""
src/adaptive_decision.py — Uncertainty-driven adaptive acquisition decision.

Purpose
-------
Implements the proof-of-concept adaptive acquisition strategy described in
documentation/adaptive_mri_acquisition_extension.md.

The core idea: instead of always accepting the U-Net's reconstruction, first
measure *how confident* the model is.  If uncertainty is low the reconstruction
is safe to use; if uncertainty is high the scanner should acquire the missing
slices normally.

This module is a simulation — there is no real scanner interface.  It
demonstrates the decision logic and uncertainty estimation on the existing
IXI dataset.

Decision pipeline (per sample)
-------------------------------
  1. enable_mc_dropout  — attach a temporary dropout hook to the bottleneck
  2. mc_forward_passes  — run N_PASSES stochastic forward passes
  3. compute_uncertainty — pixel-wise variance across passes → global score
  4. make_decision       — compare score to UNCERTAINTY_THRESHOLD
  5. save_uncertainty_plots — write 3 PNG files per sample
  6. handle.remove()    — clean up the hook so the model is unmodified

Why Monte Carlo Dropout?
------------------------
Standard neural networks produce a single deterministic output.  Monte Carlo
Dropout (Gal & Ghahramani, 2016) approximates Bayesian inference by keeping
dropout *active* during inference and running multiple forward passes.  Each
pass randomly zeros a fraction of neurons, producing a slightly different
prediction.  The *variance* across those predictions is used as a proxy for
the model's epistemic uncertainty (how much it "doesn't know").

Why the bottleneck?
-------------------
Dropout is injected at the bottleneck — the layer with the most global,
abstract features and the fewest spatial dimensions (16×16 for 256×256
input).  Zeroing neurons here has a larger effect on the output than dropping
pixels near the input or output because the bottleneck is the information
bottleneck of the entire network.

Hook design
-----------
Rather than modifying UNet or adding a dropout layer to the training code,
a PyTorch forward hook is registered on model.bottleneck.  The hook applies
F.dropout(training=True) — which is always active regardless of model mode —
to the bottleneck output before it propagates to the decoder.  After all MC
passes the hook is removed via handle.remove(), leaving the model exactly as
it was before.  This design requires zero changes to train.py.

Outputs
-------
  outputs/explainability/adaptive_panel_{idx}.png      — 6-panel summary
  outputs/explainability/uncertainty_map_{idx}.png     — standalone heatmap
  outputs/explainability/reconstructed_image_{idx}.png — standalone reconstruction

Dependencies
------------
  numpy              : stacking predictions, computing variance and mean
  torch              : model loading, tensor operations, forward hooks
  torch.nn.functional: F.dropout used inside the bottleneck hook
  matplotlib         : 6-panel summary and standalone figures (Agg backend)
  skimage.metrics    : PSNR and SSIM for logging alongside the decision
  src.config         : PROCESSED_DIR, MODELS_DIR, EXPLAINABILITY_DIR,
                       N_EXPL_SAMPLES, MC_DROPOUT_PASSES, MC_DROPOUT_P,
                       UNCERTAINTY_THRESHOLD
  src.train          : UNet (imported to share the trained architecture)
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
    Attach a dropout hook to the U-Net bottleneck for stochastic inference.

    The inner function _dropout_hook is called automatically by PyTorch after
    every forward pass through model.bottleneck.  It receives the layer's
    output tensor and returns a randomly zeroed version of it.

    Passing training=True to F.dropout forces it to apply dropout even when
    the model is in eval() mode — this is the key MC Dropout trick.  In normal
    eval mode F.dropout is a no-op, which is the correct behaviour for
    deterministic inference but wrong for uncertainty estimation.

    The returned handle must be stored by the caller and removed after use:
      handle = enable_mc_dropout(model)
      # ... run inference ...
      handle.remove()   ← restores model to its original state

    Parameters
    ----------
    model : UNet  — the trained model; not modified in any persistent way
    p     : float — fraction of bottleneck neurons to zero per forward pass

    Returns
    -------
    torch.utils.hooks.RemovableHook  — call .remove() when MC passes are done
    """
    def _dropout_hook(module, input, output):
        # training=True makes F.dropout always active, bypassing eval-mode no-op.
        return F.dropout(output, p=p, training=True)

    return model.bottleneck.register_forward_hook(_dropout_hook)


# ---------------------------------------------------------------------------
# MC forward passes
# ---------------------------------------------------------------------------

def mc_forward_passes(model, x, device, n_passes=MC_DROPOUT_PASSES, dropout_p=MC_DROPOUT_P):
    """
    Run multiple stochastic forward passes and collect predictions.

    The model stays in eval() mode throughout (BatchNorm uses stable running
    statistics).  Only the bottleneck hook introduces stochasticity by
    randomly zeroing activations on each pass, producing different outputs
    even for identical inputs.

    torch.no_grad() is used because we do not need gradients here — we only
    need the output values, not their derivatives.

    Parameters
    ----------
    model     : UNet  — must already be in eval() mode with weights loaded
    x         : torch.Tensor  shape (1, 2, 256, 256)  — single sample, on CPU
    device    : torch.device  — GPU or CPU
    n_passes  : int   — number of stochastic samples to collect
    dropout_p : float — passed to enable_mc_dropout

    Returns
    -------
    np.ndarray  shape (n_passes, H, W)  float32
      Stack of n_passes slightly different reconstructions of the same input.
    """
    model.eval()
    handle = enable_mc_dropout(model, p=dropout_p)

    predictions = []
    with torch.no_grad():
        for _ in range(n_passes):
            # Each call produces a different result because the hook drops
            # different neurons at random on every forward pass.
            pred = model(x.to(device)).squeeze().cpu().numpy()
            predictions.append(pred)

    # Always remove the hook when done so subsequent calls to the model
    # (e.g. from explainability.py) are not affected.
    handle.remove()

    return np.stack(predictions, axis=0)  # (n_passes, H, W)


# ---------------------------------------------------------------------------
# Uncertainty estimation
# ---------------------------------------------------------------------------

def compute_uncertainty(predictions):
    """
    Summarise stochastic predictions into a mean reconstruction and an
    uncertainty estimate.

    The mean prediction is the best single estimate of the missing slice —
    averaging over the ensemble reduces the noise introduced by dropout.

    The pixel-wise variance measures how much each pixel's predicted intensity
    varied across the n_passes stochastic samples.  High variance at a pixel
    means the model is uncertain about that location's intensity.

    The global uncertainty score collapses the spatial map to a single number
    that can be compared against the threshold in make_decision.

    Parameters
    ----------
    predictions : np.ndarray  shape (N, H, W)
      Stack of N stochastic predictions from mc_forward_passes.

    Returns
    -------
    mean_pred       : np.ndarray  shape (H, W)  — ensemble mean reconstruction
    uncertainty_map : np.ndarray  shape (H, W)  — per-pixel variance
    global_score    : float  — spatial mean of the variance map; used for
                      the threshold comparison in make_decision
    """
    mean_pred       = predictions.mean(axis=0)
    uncertainty_map = predictions.var(axis=0)   # pixel-wise variance
    global_score    = float(uncertainty_map.mean())
    return mean_pred, uncertainty_map, global_score


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------

def make_decision(uncertainty_score, threshold=UNCERTAINTY_THRESHOLD):
    """
    Apply the safety gate: compare the global uncertainty score to a threshold.

    Design rationale — asymmetric risk
    ------------------------------------
    In a medical imaging context, the cost of a missed abnormality
    (false SAFE) is much higher than the cost of an unnecessary full scan
    (false UNSAFE).  The threshold is therefore set conservatively: the model
    must be quite confident before reconstruction is accepted.  Lowering
    UNCERTAINTY_THRESHOLD in config.py makes the gate stricter (more full
    acquisitions); raising it accepts more reconstructions.

    Parameters
    ----------
    uncertainty_score : float  — output of compute_uncertainty's global_score
    threshold         : float  — UNCERTAINTY_THRESHOLD from config.py

    Returns
    -------
    str  — "SAFE"   → reconstruction is reliable; use the U-Net output
           "UNSAFE" → reconstruction is unreliable; acquire remaining slices
    """
    return "SAFE" if uncertainty_score < threshold else "UNSAFE"


def format_decision(decision, uncertainty_score):
    """
    Build the human-readable decision string printed to stdout and figure titles.

    Parameters
    ----------
    decision          : str    — "SAFE" or "UNSAFE"
    uncertainty_score : float

    Returns
    -------
    str  — e.g. "Decision: SAFE — reconstruction applied  (score=0.003 < threshold=0.01)"
    """
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
    Write three PNG files per sample to save_dir.

    File 1 — adaptive_panel_{idx}.png
      A 2×3 grid showing the full context:
        Row 0: left input | right input | ground truth
        Row 1: mean reconstruction | uncertainty map | uncertainty overlay
      The figure title colour is green for SAFE and red for UNSAFE, giving
      an immediate visual signal when browsing outputs.

    File 2 — uncertainty_map_{idx}.png
      Standalone hot-colourmap uncertainty map suitable for inclusion in
      reports or presentations without the surrounding context panels.

    File 3 — reconstructed_image_{idx}.png
      Standalone greyscale mean reconstruction.

    Parameters
    ----------
    mean_pred         : np.ndarray  (H, W)  — ensemble mean
    uncertainty_map   : np.ndarray  (H, W)  — pixel-wise variance
    gt                : np.ndarray  (H, W)  — ground truth slice
    x_np              : np.ndarray  (2, H, W) — left and right input slices
    sample_idx        : int
    uncertainty_score : float
    decision          : str   — "SAFE" or "UNSAFE"
    save_dir          : Path
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

    # Each tuple: (axis, image, cmap, title, vmin, vmax)
    # vmin/vmax=None triggers matplotlib auto-scaling (used for variance maps
    # whose range varies between samples).
    panels = [
        (axes[0, 0], x_np[0],         "gray", "Input: Left Slice",                          0,    1),
        (axes[0, 1], x_np[1],         "gray", "Input: Right Slice",                         0,    1),
        (axes[0, 2], gt,              "gray", "Ground Truth",                                0,    1),
        (axes[1, 0], mean_pred,       "gray", f"Mean Reconstruction (N={MC_DROPOUT_PASSES})", 0,  1),
        (axes[1, 1], uncertainty_map, "hot",  "Uncertainty Map (pixel variance)",            None, None),
        (axes[1, 2], uncertainty_map, "hot",  "Uncertainty Overlay on Reconstruction",       None, None),
    ]

    for ax, img, cmap, title, vmin, vmax in panels:
        if title.startswith("Uncertainty Overlay"):
            # Show the uncertainty as a semi-transparent heat map on top of
            # the mean reconstruction so spatial structure and uncertainty
            # are visible in the same image.
            ax.imshow(mean_pred, cmap="gray", vmin=0, vmax=1)
            im = ax.imshow(img, cmap="hot", alpha=0.5, vmin=0, vmax=uncertainty_map.max())
        elif vmin is not None:
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(img, cmap=cmap)   # auto-scale for variance maps
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
    ax.set_title(
        f"Mean Reconstruction — Sample {sample_idx}\n(N={MC_DROPOUT_PASSES} passes)",
        fontsize=11,
    )
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

    Connects all the pieces: load → MC passes → uncertainty → decision →
    print → visualise.  This function is the main building block; main()
    calls it in a loop over sample_idxs.

    Parameters
    ----------
    model      : UNet
    inputs     : np.memmap  shape (N, 2, H, W)
    targets    : np.memmap  shape (N, H, W)
    sample_idx : int
    device     : torch.device
    save_dir   : Path  — directory for output PNG files
    threshold  : float — passed to make_decision; defaults to config value

    Returns
    -------
    decision : str    — "SAFE" or "UNSAFE"
    score    : float  — global uncertainty score for this sample
    """
    x_np = np.array(inputs[sample_idx])   # materialise out of memory map
    y_np = np.array(targets[sample_idx])
    x    = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)  # (1, 2, H, W)

    predictions                       = mc_forward_passes(model, x, device)
    mean_pred, uncertainty_map, score = compute_uncertainty(predictions)
    decision                          = make_decision(score, threshold)

    # Log quality metrics alongside the uncertainty so the relationship
    # between PSNR and uncertainty is visible in the console output.
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
    """
    Analyse N_EXPL_SAMPLES representative samples and print a summary table.

    Sample indices are spread uniformly across the full dataset (same
    linspace strategy as explainability.py) so results are not biased
    towards easy early-dataset samples.
    """
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

    print(
        f"\nSettings: MC passes={MC_DROPOUT_PASSES}  "
        f"dropout_p={MC_DROPOUT_P}  "
        f"threshold={UNCERTAINTY_THRESHOLD}"
    )
    print(f"Analysing {N_EXPL_SAMPLES} samples...\n")

    results = []
    for idx in sample_idxs:
        decision, score = analyze_sample(
            model, inputs, targets, int(idx), device, EXPLAINABILITY_DIR,
        )
        results.append((int(idx), decision, score))
        print()   # blank line between samples for readability

    # ── summary table ─────────────────────────────────────────────────────
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
