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
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from src.config import (
    PROCESSED_DIR,
    MODELS_DIR,
    EXPLAINABILITY_DIR,
    FIGURES_DIR,
    N_EXPL_SAMPLES,
    MC_DROPOUT_PASSES,
    MC_DROPOUT_P,
    UNCERTAINTY_THRESHOLD,
    ANOMALY_DETECTOR_PATH,
    ANOMALY_THRESHOLD_PATH,
    ANOMALY_EPOCHS,
    ANOMALY_DEFAULT_THRESHOLD,
)
from src.train import UNet


# ---------------------------------------------------------------------------
# Phase 2 — Anomaly detection model
# ---------------------------------------------------------------------------

class SliceAutoencoder(nn.Module):
    """
    Convolutional autoencoder for unsupervised anomaly detection on MRI slices.

    Trained exclusively on healthy (IXI) slices, the autoencoder learns to
    reconstruct normal brain anatomy with low error.  A slice that deviates
    from the learned healthy distribution produces a higher mean absolute
    reconstruction error, which serves as the anomaly score for Phase 2.

    Architecture
    ------------
    Encoder  : 3 × (Conv2d-BN-ReLU → MaxPool2d)  channels: 1 → 16 → 32 → 64
               spatial: 256 → 128 → 64 → 32
    Decoder  : 3 × ConvTranspose2d-BN-ReLU        channels: 64 → 32 → 16 → 1
               spatial: 32 → 64 → 128 → 256
    Output   : Sigmoid (keeps values in [0, 1], matching normalised input)

    Input / output
    --------------
    Both expect a tensor of shape (B, 1, 256, 256).
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 256 → 128
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 128 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 64 → 32
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 32 → 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 64 → 128
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16,  1, kernel_size=2, stride=2),  # 128 → 256
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


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
# Phase 2 — Anomaly detection helpers
# ---------------------------------------------------------------------------

def train_anomaly_detector(inputs, device, epochs=ANOMALY_EPOCHS):
    """
    Train a SliceAutoencoder on acquired healthy MRI slices and calibrate
    the anomaly threshold from the training-set reconstruction error.

    Because the IXI dataset contains only healthy brains, the autoencoder
    learns a tight model of normal anatomy.  The calibrated threshold is
    mean + 2σ of the per-slice MAE over the training set; slices whose
    reconstruction error exceeds this value are flagged as suspicious.

    Parameters
    ----------
    inputs : np.memmap  shape (N, 2, H, W) — full preprocessed dataset
    device : torch.device
    epochs : int — training epochs (ANOMALY_EPOCHS from config)

    Returns
    -------
    autoencoder : SliceAutoencoder  — trained model in eval() mode
    threshold   : float             — calibrated anomaly threshold
    """
    from torch.utils.data import DataLoader, TensorDataset

    autoencoder = SliceAutoencoder().to(device)
    optimizer   = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion   = nn.L1Loss()

    # Use the same 70 % training split as the reconstruction model so the
    # anomaly detector never sees held-out or test-set slices.
    n_train    = int(len(inputs) * 0.70)
    batch_size = 64

    print(f"  Training anomaly detector on {n_train * 2:,} healthy slices "
          f"for {epochs} epochs...")

    autoencoder.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches  = 0
        perm       = np.random.permutation(n_train)

        for start in range(0, n_train, batch_size):
            batch_idx = perm[start : start + batch_size]
            batch     = np.array(inputs[batch_idx])          # (B, 2, H, W)
            # Treat each acquired slice independently → (2B, 1, H, W)
            slices    = batch.reshape(-1, 1, batch.shape[2], batch.shape[3])
            x         = torch.tensor(slices, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            recon = autoencoder(x)
            loss  = criterion(recon, x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        print(f"    Epoch {epoch + 1}/{epochs}  loss={total_loss / n_batches:.6f}")

    # ── Calibrate threshold on the training set ────────────────────────────
    autoencoder.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, n_train, batch_size):
            batch  = np.array(inputs[start : start + batch_size])
            slices = batch.reshape(-1, 1, batch.shape[2], batch.shape[3])
            x      = torch.tensor(slices, dtype=torch.float32).to(device)
            recon  = autoencoder(x)
            # Per-slice MAE: mean over (H, W) dimensions.
            mae    = torch.abs(recon - x).mean(dim=(1, 2, 3)).cpu().numpy()
            errors.extend(mae.tolist())

    errors    = np.array(errors)
    threshold = float(errors.mean() + 2.0 * errors.std())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(autoencoder.state_dict(), ANOMALY_DETECTOR_PATH)
    np.save(ANOMALY_THRESHOLD_PATH, np.array([threshold]))

    print(f"  Calibrated threshold: {threshold:.6f}  (μ={errors.mean():.6f}, "
          f"σ={errors.std():.6f})")
    print(f"  Saved → {ANOMALY_DETECTOR_PATH.name}")

    return autoencoder, threshold


def compute_anomaly_score(autoencoder, x_np, device):
    """
    Run the SliceAutoencoder on the two acquired slices of a sample and
    return per-slice reconstruction errors.

    Parameters
    ----------
    autoencoder : SliceAutoencoder  — trained model in eval() mode
    x_np        : np.ndarray  shape (2, H, W) — left and right acquired slices
    device      : torch.device

    Returns
    -------
    scores     : np.ndarray  shape (2,)     — per-slice MAE
    recon_np   : np.ndarray  shape (2, H, W) — autoencoder reconstructions
    """
    autoencoder.eval()
    slices = x_np[:, np.newaxis, :, :]                           # (2, 1, H, W)
    x      = torch.tensor(slices, dtype=torch.float32).to(device)

    with torch.no_grad():
        recon = autoencoder(x)

    recon_np = recon.squeeze(1).cpu().numpy()                    # (2, H, W)
    scores   = np.abs(recon_np - x_np).mean(axis=(1, 2))        # (2,)
    return scores, recon_np


def classify_anomaly(anomaly_scores, threshold):
    """
    Classify the acquired slices as NORMAL or SUSPICIOUS.

    A sample is flagged SUSPICIOUS if *either* acquired slice exceeds the
    threshold.  This conservative (AND-safe) rule ensures that a single
    abnormal slice triggers full acquisition regardless of the other slice.

    Parameters
    ----------
    anomaly_scores : np.ndarray  shape (2,) — per-slice MAE from the autoencoder
    threshold      : float — calibrated anomaly threshold

    Returns
    -------
    str  — "NORMAL" or "SUSPICIOUS"
    """
    return "SUSPICIOUS" if np.any(anomaly_scores > threshold) else "NORMAL"


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


def make_combined_decision(uncertainty_score, anomaly_class,
                            threshold=UNCERTAINTY_THRESHOLD):
    """
    Combine Phase 2 (anomaly classification) and Phase 3 (uncertainty gate)
    into a single SAFE / UNSAFE verdict.

    Both conditions must pass for reconstruction to be accepted:
      • uncertainty_score < threshold  (Phase 3a gate — model is confident)
      • anomaly_class == "NORMAL"      (Phase 2 gate — no anomaly detected)

    If either condition fails the decision is UNSAFE, meaning the missing
    slices should be acquired normally rather than reconstructed.

    Parameters
    ----------
    uncertainty_score : float — global MC-dropout variance score
    anomaly_class     : str   — "NORMAL" or "SUSPICIOUS" from classify_anomaly
    threshold         : float — UNCERTAINTY_THRESHOLD from config.py

    Returns
    -------
    str  — "SAFE" or "UNSAFE"
    """
    if uncertainty_score >= threshold:
        return "UNSAFE"
    if anomaly_class == "SUSPICIOUS":
        return "UNSAFE"
    return "SAFE"


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
                            sample_idx, uncertainty_score, decision,
                            anomaly_scores, anomaly_recons, anomaly_class,
                            anomaly_threshold, save_dir):
    """
    Write three PNG files per sample to save_dir.

    File 1 — adaptive_panel_{idx}.png
      A 3×3 grid covering all three pipeline phases:
        Row 0 : left input      | right input    | ground truth
        Row 1 : mean recon      | uncertainty map | uncertainty overlay
        Row 2 : left anomaly err | right anomaly err | Phase 2 classification
      The figure title colour is green for SAFE and red for UNSAFE.

    File 2 — uncertainty_map_{idx}.png
      Standalone hot-colourmap uncertainty map.

    File 3 — reconstructed_image_{idx}.png
      Standalone greyscale mean reconstruction.

    Parameters
    ----------
    mean_pred         : np.ndarray  (H, W)   — ensemble mean
    uncertainty_map   : np.ndarray  (H, W)   — pixel-wise variance
    gt                : np.ndarray  (H, W)   — ground truth slice
    x_np              : np.ndarray  (2, H, W) — left and right acquired slices
    sample_idx        : int
    uncertainty_score : float
    decision          : str   — "SAFE" or "UNSAFE"
    anomaly_scores    : np.ndarray  (2,)     — per-slice MAE from autoencoder
    anomaly_recons    : np.ndarray  (2, H, W) — autoencoder reconstructions
    anomaly_class     : str   — "NORMAL" or "SUSPICIOUS"
    anomaly_threshold : float
    save_dir          : Path
    """
    is_safe      = decision == "SAFE"
    label_color  = "green" if is_safe else "red"
    label_text   = "SAFE — reconstruction applied" if is_safe else "UNSAFE — full acquisition recommended"
    anom_color   = "green" if anomaly_class == "NORMAL" else "orange"

    # ── 9-panel summary (3 rows × 3 columns) ───────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(
        f"Adaptive Acquisition — Sample {sample_idx}\n"
        f"Phase 2 (anomaly): {anomaly_class}  "
        f"[L={anomaly_scores[0]:.4f}, R={anomaly_scores[1]:.4f}, thr={anomaly_threshold:.4f}]  |  "
        f"Phase 3 (uncertainty): {uncertainty_score:.6f}  thr={UNCERTAINTY_THRESHOLD}  |  "
        f"Decision: {label_text}",
        fontsize=11, fontweight="bold", color=label_color,
    )

    # ── Row 0: acquired context ─────────────────────────────────────────────
    for ax, img, title in [
        (axes[0, 0], x_np[0], "Input: Left Slice (acquired)"),
        (axes[0, 1], x_np[1], "Input: Right Slice (acquired)"),
        (axes[0, 2], gt,      "Ground Truth"),
    ]:
        im = ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    # ── Row 1: Phase 3 — uncertainty ────────────────────────────────────────
    im = axes[1, 0].imshow(mean_pred, cmap="gray", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 0].set_title(f"Mean Reconstruction (N={MC_DROPOUT_PASSES})",
                         fontsize=10, fontweight="bold")
    axes[1, 0].axis("off")

    im = axes[1, 1].imshow(uncertainty_map, cmap="hot")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    axes[1, 1].set_title("Uncertainty Map (pixel variance)", fontsize=10, fontweight="bold")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(mean_pred, cmap="gray", vmin=0, vmax=1)
    im = axes[1, 2].imshow(uncertainty_map, cmap="hot", alpha=0.5,
                            vmin=0, vmax=uncertainty_map.max())
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    axes[1, 2].set_title("Uncertainty Overlay on Reconstruction", fontsize=10, fontweight="bold")
    axes[1, 2].axis("off")

    # ── Row 2: Phase 2 — anomaly detection ─────────────────────────────────
    left_err  = np.abs(anomaly_recons[0] - x_np[0])
    right_err = np.abs(anomaly_recons[1] - x_np[1])

    im = axes[2, 0].imshow(left_err, cmap="hot", vmin=0)
    plt.colorbar(im, ax=axes[2, 0], fraction=0.046, pad=0.04)
    axes[2, 0].set_title(
        f"Phase 2: Left Slice Anomaly Error\n(MAE={anomaly_scores[0]:.4f})",
        fontsize=10, fontweight="bold",
    )
    axes[2, 0].axis("off")

    im = axes[2, 1].imshow(right_err, cmap="hot", vmin=0)
    plt.colorbar(im, ax=axes[2, 1], fraction=0.046, pad=0.04)
    axes[2, 1].set_title(
        f"Phase 2: Right Slice Anomaly Error\n(MAE={anomaly_scores[1]:.4f})",
        fontsize=10, fontweight="bold",
    )
    axes[2, 1].axis("off")

    # Text summary panel for Phase 2 classification result
    axes[2, 2].axis("off")
    summary_lines = [
        "Phase 2 — Anomaly Classification",
        "",
        f"Left  slice MAE : {anomaly_scores[0]:.6f}",
        f"Right slice MAE : {anomaly_scores[1]:.6f}",
        f"Threshold       : {anomaly_threshold:.6f}",
        "",
        f"Classification  : {anomaly_class}",
        "",
        f"Phase 3 uncertainty : {uncertainty_score:.6f}",
        f"Uncertainty thr     : {UNCERTAINTY_THRESHOLD}",
        "",
        f"Final decision  : {decision}",
    ]
    axes[2, 2].text(
        0.05, 0.95, "\n".join(summary_lines),
        transform=axes[2, 2].transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
        color=label_color,
        bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8),
    )
    axes[2, 2].set_title("Phase 2 + Phase 3 Summary", fontsize=10, fontweight="bold")

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

def analyze_sample(model, autoencoder, anomaly_threshold,
                   inputs, targets, sample_idx, device,
                   save_dir, threshold=UNCERTAINTY_THRESHOLD):
    """
    Run the full two-phase adaptive pipeline for one sample.

    Phase 2 — Detection
      The SliceAutoencoder reconstructs each acquired slice.  High
      reconstruction error signals that the slice differs from normal
      healthy anatomy → "SUSPICIOUS".

    Phase 3 — Reconstruction / full acquisition
      Monte Carlo Dropout runs N stochastic forward passes through the U-Net.
      The pixel-wise variance across passes gives the uncertainty score.
      Combined with the Phase 2 result, the final decision is:
        SAFE   → Phase 2 normal  AND  Phase 3 uncertainty < threshold
        UNSAFE → Phase 2 suspicious  OR  Phase 3 uncertainty ≥ threshold

    Parameters
    ----------
    model             : UNet
    autoencoder       : SliceAutoencoder  — trained anomaly detector
    anomaly_threshold : float             — calibrated MAE threshold
    inputs            : np.memmap  shape (N, 2, H, W)
    targets           : np.memmap  shape (N, H, W)
    sample_idx        : int
    device            : torch.device
    save_dir          : Path  — directory for output PNG files
    threshold         : float — uncertainty threshold; defaults to config value

    Returns
    -------
    decision      : str   — "SAFE" or "UNSAFE"
    uncert_score  : float — global uncertainty score (Phase 3)
    anomaly_class : str   — "NORMAL" or "SUSPICIOUS" (Phase 2)
    anomaly_score : float — mean per-slice MAE from the autoencoder
    """
    x_np = np.array(inputs[sample_idx])   # materialise out of memory map
    y_np = np.array(targets[sample_idx])
    x    = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)  # (1, 2, H, W)

    # ── Phase 2: anomaly detection on acquired slices ──────────────────────
    anomaly_scores, anomaly_recons = compute_anomaly_score(autoencoder, x_np, device)
    anomaly_class                  = classify_anomaly(anomaly_scores, anomaly_threshold)
    mean_anomaly_score             = float(anomaly_scores.mean())

    # ── Phase 3: MC-dropout uncertainty estimation ─────────────────────────
    predictions                        = mc_forward_passes(model, x, device)
    mean_pred, uncertainty_map, score  = compute_uncertainty(predictions)

    # ── Combined decision ──────────────────────────────────────────────────
    decision = make_combined_decision(score, anomaly_class, threshold)

    # Log quality metrics alongside both phase scores.
    sample_psnr = psnr(y_np, mean_pred, data_range=1.0)
    sample_ssim = ssim(y_np, mean_pred, data_range=1.0)

    status = "SAFE  " if decision == "SAFE" else "UNSAFE"
    print(
        f"  [{status}] sample={sample_idx:>5}  "
        f"PSNR={sample_psnr:.2f}dB  SSIM={sample_ssim:.4f}  "
        f"uncertainty={score:.6f}  anomaly={anomaly_class} "
        f"(L={anomaly_scores[0]:.4f}, R={anomaly_scores[1]:.4f})"
    )

    save_uncertainty_plots(
        mean_pred, uncertainty_map, y_np, x_np,
        sample_idx, score, decision,
        anomaly_scores, anomaly_recons, anomaly_class,
        anomaly_threshold, save_dir,
    )

    return decision, score, anomaly_class, mean_anomaly_score


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

    # ── Phase 2: load or train the anomaly detector ────────────────────────
    autoencoder = SliceAutoencoder().to(device)
    if ANOMALY_DETECTOR_PATH.exists() and ANOMALY_THRESHOLD_PATH.exists():
        autoencoder.load_state_dict(
            torch.load(ANOMALY_DETECTOR_PATH, map_location=device)
        )
        autoencoder.eval()
        anomaly_threshold = float(np.load(ANOMALY_THRESHOLD_PATH)[0])
        print(f"Anomaly detector: {ANOMALY_DETECTOR_PATH.name}  "
              f"(threshold={anomaly_threshold:.6f})")
    else:
        print("\nAnomaly detector not found — training from scratch...")
        autoencoder, anomaly_threshold = train_anomaly_detector(inputs, device)
        autoencoder.eval()

    sample_idxs = np.linspace(0, len(inputs) - 1, N_EXPL_SAMPLES, dtype=int)

    print(
        f"\nSettings: MC passes={MC_DROPOUT_PASSES}  "
        f"dropout_p={MC_DROPOUT_P}  "
        f"uncertainty_threshold={UNCERTAINTY_THRESHOLD}  "
        f"anomaly_threshold={anomaly_threshold:.6f}"
    )
    print(f"Analysing {N_EXPL_SAMPLES} samples...\n")

    results = []
    for idx in sample_idxs:
        decision, score, anomaly_class, anomaly_score = analyze_sample(
            model, autoencoder, anomaly_threshold,
            inputs, targets, int(idx), device, EXPLAINABILITY_DIR,
        )
        # Also record PSNR and SSIM for the scatter plot
        x_np = np.array(inputs[int(idx)])
        y_np = np.array(targets[int(idx)])
        x_t  = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_np = model(x_t.to(device)).squeeze().cpu().numpy()
        sample_psnr = psnr(y_np, pred_np, data_range=1.0)
        sample_ssim = ssim(y_np, pred_np, data_range=1.0)
        results.append((int(idx), decision, score, anomaly_class, anomaly_score,
                        sample_psnr, sample_ssim))
        print()   # blank line between samples for readability

    # ── summary table ──────────────────────────────────────────────────────
    # Recount after dynamic uncertainty threshold calibration:
    # use 75th percentile of observed scores so the top 25% are flagged UNSAFE.
    uncert_scores = [s    for _, _, s, _, _, _, _ in results]
    anom_scores   = [a    for _, _, _, _, a, _, _ in results]
    psnr_vals     = [p    for _, _, _, _, _, p, _ in results]
    ssim_vals     = [s    for _, _, _, _, _, _, s in results]

    adaptive_uncert_threshold = float(np.percentile(uncert_scores, 75))

    n_normal     = sum(1 for _, _, _, ac, *_ in results if ac == "NORMAL")
    n_suspicious = len(results) - n_normal
    # Phase 3 uses the adaptive threshold; anomaly flag unchanged
    n_unsafe  = sum(1 for _, _, s, _, a, _, _ in results
                    if s > adaptive_uncert_threshold or a > anomaly_threshold)
    n_safe    = len(results) - n_unsafe

    print("=" * 65)
    print("ADAPTIVE ACQUISITION SUMMARY")
    print("=" * 65)
    print(f"  Samples analysed              : {len(results)}")
    print(f"  --- Phase 2 (anomaly) ---")
    print(f"  NORMAL     (no anomaly)       : {n_normal}  "
          f"({100 * n_normal / len(results):.0f}%)")
    print(f"  SUSPICIOUS (anomaly detected) : {n_suspicious}  "
          f"({100 * n_suspicious / len(results):.0f}%)")
    print(f"  Mean anomaly score            : {np.mean(anom_scores):.6f}")
    print(f"  Anomaly threshold             : {anomaly_threshold:.6f}")
    print(f"  --- Phase 3 (uncertainty) ---")
    print(f"  Mean uncertainty score        : {np.mean(uncert_scores):.3e}")
    print(f"  Min / Max uncertainty         : {np.min(uncert_scores):.3e} / "
          f"{np.max(uncert_scores):.3e}")
    print(f"  Threshold (p75 adaptive)      : {adaptive_uncert_threshold:.3e}")
    print(f"  --- Combined decision ---")
    print(f"  SAFE   (reconstruct)          : {n_safe}  "
          f"({100 * n_safe / len(results):.0f}%)")
    print(f"  UNSAFE (full acquisition)     : {n_unsafe}  "
          f"({100 * n_unsafe / len(results):.0f}%)")
    print("=" * 65)
    print(f"\nOutputs → {EXPLAINABILITY_DIR}")

    # ── Uncertainty vs PSNR scatter plot ───────────────────────────────────
    # Validates the adaptive gate: if uncertainty is a meaningful proxy for
    # reconstruction quality, high-uncertainty samples should have low PSNR.
    # Spearman correlation captures monotonic relationships (not just linear).
    try:
        from scipy.stats import spearmanr
        uncert_arr = np.array(uncert_scores)
        psnr_arr   = np.array(psnr_vals)
        ssim_arr   = np.array(ssim_vals)

        rho_psnr, p_psnr = spearmanr(uncert_arr, psnr_arr)
        rho_ssim, p_ssim = spearmanr(uncert_arr, ssim_arr)

        print(f"\n  Spearman ρ (uncertainty vs PSNR) : {rho_psnr:+.3f}  (p={p_psnr:.3f})")
        print(f"  Spearman ρ (uncertainty vs SSIM) : {rho_ssim:+.3f}  (p={p_ssim:.3f})")
        if rho_psnr < -0.3:
            print("  → Negative correlation: higher uncertainty predicts lower PSNR ✓")
        else:
            print("  → Weak/no correlation: uncertainty may not track reconstruction quality here")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"Uncertainty vs Reconstruction Quality (N={len(results)} samples)\n"
            f"Spearman ρ: uncertainty↔PSNR = {rho_psnr:+.3f} (p={p_psnr:.3f}), "
            f"uncertainty↔SSIM = {rho_ssim:+.3f} (p={p_ssim:.3f})",
            fontsize=11, fontweight="bold",
        )

        for ax, yvals, ylabel, rho, p_val in [
            (axes[0], psnr_arr, "PSNR (dB)", rho_psnr, p_psnr),
            (axes[1], ssim_arr, "SSIM",      rho_ssim, p_ssim),
        ]:
            colors = ["tomato" if d == "UNSAFE" else "steelblue"
                      for _, d, *_ in results]
            ax.scatter(uncert_arr, yvals, c=colors, alpha=0.75, s=60, edgecolors="none")
            ax.axvline(UNCERTAINTY_THRESHOLD, color="black", linestyle="--",
                       linewidth=1.5, label=f"Threshold={UNCERTAINTY_THRESHOLD}")
            # Trend line
            z = np.polyfit(uncert_arr, yvals, 1)
            xfit = np.linspace(uncert_arr.min(), uncert_arr.max(), 100)
            ax.plot(xfit, np.polyval(z, xfit), color="gray", linestyle="-",
                    linewidth=1.5, alpha=0.8, label="Linear fit")
            ax.set_xlabel("MC-Dropout Uncertainty Score (pixel variance)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"ρ = {rho:+.3f}  (p={p_val:.3f})", fontsize=10)
            # Legend: coloured dots for SAFE/UNSAFE
            from matplotlib.lines import Line2D
            legend_els = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                       markersize=9, label='SAFE'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato',
                       markersize=9, label='UNSAFE'),
                Line2D([0], [0], color='black', linestyle='--', label='Threshold'),
            ]
            ax.legend(handles=legend_els, fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        scatter_path = FIGURES_DIR / "uncertainty_vs_quality.png"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved scatter plot → {scatter_path.name}")
    except ImportError:
        print("\n  (Install scipy for Spearman correlation: pip install scipy)")


if __name__ == "__main__":
    main()
