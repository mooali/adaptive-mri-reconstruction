#!/usr/bin/env python
"""
src/explainability.py ÔÇö Grad-CAM and Integrated Gradients for the trained U-Net.

Purpose
-------
After training, this module answers the question: *what does the model look at
when reconstructing a missing slice?*  Two complementary techniques are used:

  Grad-CAM (spatial attention)
    Shows *where* in the image the network focuses its attention.  A heat map
    is overlaid on the ground-truth slice; bright regions were most influential
    for the overall output.

  Integrated Gradients (input attribution)
    Shows *which pixels of the two input slices* the network relies on most.
    Separate attribution maps are produced for the left and right input slices,
    and their difference reveals whether the network treats both inputs equally.

Both methods operate on the already-trained model and require no retraining.

Pipeline inside main()
----------------------
  1. Load model weights from models/unet_best.pth.
  2. Load dataset arrays (memory-mapped, read-only).
  3. Select N_EXPL_SAMPLES indices spread uniformly across the dataset.
  4. For each sample: compute Grad-CAM + IG ÔåÆ save 10-panel figure.
  5. Score 1000 samples by PSNR ÔåÆ select hard / medium / easy cases ÔåÆ
     save comparison figure.
  6. Print aggregate attribution statistics.

Outputs
-------
  outputs/explainability/explainability_sample_{idx}.png  (one per sample)
  outputs/explainability/gradcam_easy_vs_hard.png

Dependencies
------------
  numpy              : array operations and statistics
  torch              : gradient computation, model forward pass
  torch.nn.functional: F.relu, F.interpolate used inside Grad-CAM
  matplotlib         : multi-panel figures (Agg backend ÔÇö headless safe)
  matplotlib.colors  : TwoSlopeNorm for signed IG difference map
  skimage.metrics    : PSNR and SSIM for labelling and sample scoring
  src.config         : PROCESSED_DIR, MODELS_DIR, EXPLAINABILITY_DIR,
                       N_EXPL_SAMPLES, IG_STEPS
  src.train          : UNet (imported to avoid duplicating the architecture)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from src.config import (
    PROCESSED_DIR,
    MODELS_DIR,
    EXPLAINABILITY_DIR,
    N_EXPL_SAMPLES,
    IG_STEPS,
    MC_DROPOUT_PASSES,
    MC_DROPOUT_P,
)
from src.train import UNet   # reuse the exact same architecture as was trained


# ---------------------------------------------------------------------------
# MC Dropout helpers (for uncertainty estimation)
# ---------------------------------------------------------------------------

def enable_mc_dropout(model, p=MC_DROPOUT_P):
    """Attach a forward hook to the bottleneck that applies dropout at inference."""
    def _dropout_hook(module, input, output):
        return F.dropout(output, p=p, training=True)
    return model.bottleneck.register_forward_hook(_dropout_hook)


def mc_forward_passes(model, x, device, n_passes=MC_DROPOUT_PASSES, dropout_p=MC_DROPOUT_P):
    """
    Run n_passes stochastic forward passes through the model.

    Returns
    -------
    np.ndarray  shape (n_passes, H, W)
    """
    model.eval()
    handle = enable_mc_dropout(model, p=dropout_p)
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            pred = model(x.to(device)).squeeze().cpu().numpy()
            preds.append(pred)
    handle.remove()
    return np.stack(preds, axis=0)


def summarise_uncertainty(predictions):
    """
    Summarise MC predictions into mean, variance map, and global score.

    Parameters
    ----------
    predictions : np.ndarray  shape (n_passes, H, W)

    Returns
    -------
    mean_pred       : np.ndarray  (H, W)
    uncertainty_map : np.ndarray  (H, W)  per-pixel variance
    global_score    : float  mean variance across all pixels
    """
    mean_pred       = predictions.mean(axis=0)
    uncertainty_map = predictions.var(axis=0)
    global_score    = float(uncertainty_map.mean())
    return mean_pred, uncertainty_map, global_score


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping adapted for regression.

    How it works
    ------------
    1. A forward hook records the feature map (activations) at the chosen
       layer whenever a forward pass is made.
    2. A backward hook records the gradient of the scalar output score
       with respect to those same feature maps.
    3. Gradients are global-average-pooled over spatial dimensions to obtain
       per-channel importance weights.
    4. The weighted sum of activations is ReLU'd (negative contributions
       are suppressed) and upsampled to the input resolution.
    5. The result is min-max normalised to [0, 1] for display.

    Target layer choice
    -------------------
    The *last decoder block* (model.decoders[-1]) is used rather than the
    bottleneck because:
    - It is at full spatial resolution, so the resulting map aligns pixel-for-
      pixel with the input without needing heavy upsampling.
    - Its gradients are stronger (shorter path to the output) than those at
      the bottleneck.

    BatchNorm and model.train()
    ---------------------------
    generate() temporarily switches the model to train() mode.  In eval()
    mode BatchNorm uses running statistics accumulated during training, which
    can saturate or zero-out gradients for inputs that differ from the training
    distribution.  Using per-batch statistics (train mode) keeps gradients
    alive for arbitrary inference inputs.

    Parameters
    ----------
    model        : UNet
    target_layer : nn.Module  ÔÇö the layer whose activations are visualised;
                   typically model.decoders[-1]
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None  # populated by the backward hook
        self.activations  = None  # populated by the forward hook
        self._register_hooks()

    def _register_hooks(self):
        """Attach forward and backward hooks to the target layer."""

        def forward_hook(module, input, output):
            # Capture the layer's output tensor on every forward pass.
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] is the gradient flowing back into the layer.
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x):
        """
        Compute and return the normalised Grad-CAM heat map for input x.

        Parameters
        ----------
        x : torch.Tensor  shape (1, 2, H, W) ÔÇö single-sample input tensor

        Returns
        -------
        np.ndarray  shape (H, W)  values in [0, 1]
        Returns a zero map if no gradients were captured (misconfigured layer).
        """
        self.model.train()          # see class docstring for why
        x      = x.requires_grad_(True)
        output = self.model(x)

        self.model.zero_grad()
        # Reduce output to a scalar for backprop.  mean() is natural for a
        # regression output (no class logit to select).
        output.mean().backward()
        self.model.eval()           # restore eval mode after gradient pass

        if self.gradients is None or self.activations is None:
            print("WARNING: No gradients captured ÔÇö check target layer.")
            return np.zeros((256, 256))

        # Global average pool gradients over (H, W) ÔåÆ per-channel weight.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activations; ReLU keeps only positively contributing
        # channels (those whose gradient and activation agree in sign).
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))

        if cam.max() == 0:
            return np.zeros((256, 256))

        # Upsample from the decoder's spatial resolution back to 256├ù256.
        cam = F.interpolate(cam, size=(256, 256), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        # Normalise to [0, 1] for consistent display across samples.
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ---------------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------------

def compute_integrated_gradients(model, x, device, n_steps=IG_STEPS):
    """
    Attribute each input pixel's contribution to the model's output.

    Theory
    ------
    Integrated Gradients (Sundararajan et al., 2017) defines attribution as:

      IG_i(x) = (x_i - x'_i) ├ù Ôê½ÔéÇ┬╣ ÔêéF(x' + ╬▒(x - x')) / Ôêéx_i  d╬▒

    where x' is a baseline input (here: all zeros ÔÇö a blank image), and
    the integral is approximated by summing gradients along a linear path
    from baseline to input in n_steps equal steps.

    The absolute value is taken at the end because we care about *magnitude*
    of influence, not sign (both channels are positive pixel intensities so
    a symmetric sign interpretation is not meaningful here).

    Parameters
    ----------
    model   : UNet  ÔÇö must be in eval() mode with frozen weights
    x       : torch.Tensor  shape (1, 2, H, W)  ÔÇö single sample input
    device  : torch.device
    n_steps : int  ÔÇö number of interpolation steps (more ÔåÆ more accurate,
              but scales linearly in memory and time)

    Returns
    -------
    np.ndarray  shape (2, H, W)  ÔÇö per-channel attribution magnitude
      [0] = attribution for left input slice
      [1] = attribution for right input slice
    """
    baseline = torch.zeros_like(x).to(device)  # black image ÔÇö neutral reference
    x        = x.to(device)
    attrs    = torch.zeros_like(x).to(device)  # accumulator

    for step in range(n_steps):
        # Alpha linearly interpolates from 0 (baseline) to (n_steps-1)/n_steps Ôëê 1.
        alpha  = step / n_steps
        # Detach before requires_grad so the interpolation itself is not in
        # the computation graph; only the model forward pass needs grad.
        interp = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)

        model.zero_grad()
        model(interp).mean().backward()
        attrs += interp.grad.detach()

    # Scale accumulated gradients by the input - baseline difference.
    # This implements the trapezoidal approximation of the path integral.
    attrs = (x.detach() - baseline) * (attrs / n_steps)
    return np.abs(attrs.squeeze().cpu().numpy())  # (2, H, W)


# ---------------------------------------------------------------------------
# Sample loading helper
# ---------------------------------------------------------------------------

def get_sample(inputs, targets, idx, device):
    """
    Materialise one memory-mapped sample as numpy arrays and a GPU tensor.

    np.array() is called explicitly to force a copy out of the memory map
    before tensor conversion, avoiding a slow path on some platforms.

    Parameters
    ----------
    inputs  : np.memmap  shape (N, 2, H, W)
    targets : np.memmap  shape (N, H, W)
    idx     : int
    device  : torch.device

    Returns
    -------
    x    : torch.Tensor  shape (1, 2, H, W)  on device
    x_np : np.ndarray   shape (2, H, W)
    y_np : np.ndarray   shape (H, W)
    """
    x_np = np.array(inputs[idx])
    y_np = np.array(targets[idx])
    x    = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(device)
    return x, x_np, y_np


# ---------------------------------------------------------------------------
# Slice contribution and occlusion sensitivity
# ---------------------------------------------------------------------------

def compute_slice_contribution(model, x, device):
    """
    Channel ablation: measure each input slice's contribution to the prediction.

    Zeroes out each channel in turn and computes the change in output relative
    to the full prediction.  A larger change means that channel was more important.

    Returns
    -------
    left_contrib_map  : np.ndarray  (H, W)  |pred_full - pred_no_left|
    right_contrib_map : np.ndarray  (H, W)  |pred_full - pred_no_right|
    dominance_map     : np.ndarray  (H, W)  left_contrib - right_contrib
    left_score        : float  mean left contribution
    right_score       : float  mean right contribution
    """
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        pred_full = model(x).squeeze().cpu().numpy()

        x_no_left = x.clone()
        x_no_left[:, 0] = 0.0
        pred_no_left = model(x_no_left).squeeze().cpu().numpy()

        x_no_right = x.clone()
        x_no_right[:, 1] = 0.0
        pred_no_right = model(x_no_right).squeeze().cpu().numpy()

    left_contrib_map  = np.abs(pred_full - pred_no_left)
    right_contrib_map = np.abs(pred_full - pred_no_right)
    dominance_map     = left_contrib_map - right_contrib_map
    left_score        = float(left_contrib_map.mean())
    right_score       = float(right_contrib_map.mean())
    return left_contrib_map, right_contrib_map, dominance_map, left_score, right_score


def compute_occlusion_sensitivity(model, x, device, patch_size=16, stride=16, fill_value=0.0):
    """
    Slide a zeroed patch over each input channel and measure output change.

    Parameters
    ----------
    patch_size : int  patch height and width (default 16 for speed)
    stride     : int  step between patches (default 16 = non-overlapping)
    fill_value : float  value to fill the occluded region

    Returns
    -------
    left_occ_map  : np.ndarray  (H, W)  sensitivity of output to left channel patches
    right_occ_map : np.ndarray  (H, W)  sensitivity of output to right channel patches
    left_score    : float  mean left occlusion sensitivity
    right_score   : float  mean right occlusion sensitivity
    """
    model.eval()
    x = x.to(device)
    H, W = x.shape[2], x.shape[3]

    with torch.no_grad():
        base_pred = model(x).squeeze().cpu().numpy()

    left_occ_map  = np.zeros((H, W), dtype=np.float32)
    right_occ_map = np.zeros((H, W), dtype=np.float32)
    count_map     = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, H - patch_size + 1, stride):
            for xp in range(0, W - patch_size + 1, stride):
                x_occ = x.clone()
                x_occ[:, 0, y:y + patch_size, xp:xp + patch_size] = fill_value
                diff_left = float(np.abs(base_pred - model(x_occ).squeeze().cpu().numpy()).mean())
                left_occ_map[y:y + patch_size, xp:xp + patch_size] += diff_left

                x_occ = x.clone()
                x_occ[:, 1, y:y + patch_size, xp:xp + patch_size] = fill_value
                diff_right = float(np.abs(base_pred - model(x_occ).squeeze().cpu().numpy()).mean())
                right_occ_map[y:y + patch_size, xp:xp + patch_size] += diff_right

                count_map[y:y + patch_size, xp:xp + patch_size] += 1

    count_map = np.maximum(count_map, 1)
    left_occ_map  /= count_map
    right_occ_map /= count_map
    left_score  = float(left_occ_map.mean())
    right_score = float(right_occ_map.mean())
    return left_occ_map, right_occ_map, left_score, right_score


# ---------------------------------------------------------------------------
# Per-sample explainability figure
# ---------------------------------------------------------------------------

def visualize_explainability(model, gradcam, inputs, targets, sample_idx, device, save_dir=None):
    """
    Produce a 10-panel explainability figure for one sample and save it.

    Panel layout (2 rows ├ù 5 columns)
    ----------------------------------
    Row 0: Left Input | U-Net Prediction | Ground Truth | Error Map | Right Input
    Row 1: IG Left    | IG Right          | Grad-CAM     | CAM Overlay | IG Difference

    The IG Difference panel uses a diverging colourmap (RdBu_r) centred at
    zero so that pixels where the left slice was more influential appear red
    and pixels where the right slice dominated appear blue.

    Parameters
    ----------
    model      : UNet
    gradcam    : GradCAM  ÔÇö already initialised with the correct target layer
    inputs     : np.memmap  shape (N, 2, H, W)
    targets    : np.memmap  shape (N, H, W)
    sample_idx : int
    device     : torch.device
    save_dir   : Path or None  ÔÇö directory for the output PNG
    """
    x, x_np, y_np = get_sample(inputs, targets, sample_idx, device)

    with torch.no_grad():
        pred_np = model(x).squeeze().cpu().numpy()

    cam        = gradcam.generate(x.clone())
    attrs      = compute_integrated_gradients(model, x.clone(), device)
    left_attr  = attrs[0]   # (H, W) ÔÇö attribution for the left input slice
    right_attr = attrs[1]   # (H, W) ÔÇö attribution for the right input slice

    sample_psnr = psnr(y_np, pred_np, data_range=1.0)
    sample_ssim = ssim(y_np, pred_np, data_range=1.0)
    error_map   = np.abs(y_np - pred_np)

    fig = plt.figure(figsize=(24, 10))
    gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(
        f"Explainability Analysis ÔÇö Sample {sample_idx}\n"
        f"PSNR={sample_psnr:.2f}dB   SSIM={sample_ssim:.4f}",
        fontsize=14, fontweight="bold",
    )

    # Each tuple: (row, col, image_array, colourmap, title, vmin, vmax)
    # vmin/vmax=None triggers auto-scaling for that panel.
    panels = [
        (0, 0, x_np[0],    "gray",   "Input: Left Slice",           0,    1),
        (0, 1, pred_np,    "gray",   "U-Net Prediction",            0,    1),
        (0, 2, y_np,       "gray",   "Ground Truth",                0,    1),
        (0, 3, error_map,  "hot",    "Error Map  |GT ÔêÆ Pred|",      0,    error_map.max()),
        (0, 4, x_np[1],    "gray",   "Input: Right Slice",          0,    1),
        (1, 0, left_attr,  "hot",    "IG: Left Slice Attribution",  0,    left_attr.max()),
        (1, 1, right_attr, "hot",    "IG: Right Slice Attribution", 0,    right_attr.max()),
        (1, 2, cam,        "jet",    "Grad-CAM (Last Decoder)",     0,    1),
        (1, 3, cam,        "jet",    "Grad-CAM Overlay",            0,    1),
        (1, 4, left_attr - right_attr, "RdBu_r",
               "IG Difference\n(Left ÔêÆ Right)",                     None, None),
    ]

    for row, col, img, cmap, title, vmin, vmax in panels:
        ax = fig.add_subplot(gs[row, col])

        if title == "Grad-CAM Overlay":
            # Superimpose the CAM heat map on the greyscale ground truth.
            ax.imshow(y_np, cmap="gray", vmin=0, vmax=1)
            ax.imshow(cam,  cmap="jet",  alpha=0.45, vmin=0, vmax=1)
        elif title.startswith("IG Difference"):
            # TwoSlopeNorm centres the colourmap at zero regardless of the
            # actual min/max, making left-vs-right asymmetry immediately visible.
            norm = TwoSlopeNorm(vmin=img.min(), vcenter=0, vmax=img.max())
            im   = ax.imshow(img, cmap=cmap, norm=norm)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    if save_dir:
        path = save_dir / f"explainability_sample_{sample_idx}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Dependency maps figure (slice contribution + occlusion)
# ---------------------------------------------------------------------------

def visualize_dependency_maps(model, inputs, targets, sample_idx, device, save_dir=None):
    """
    Produce a 2x3 figure showing slice contribution maps and occlusion sensitivity.

    Panel layout (2 rows x 3 columns)
    ----------------------------------
    Row 0: Left Contribution | Right Contribution | Dominance Map
    Row 1: Left Occlusion    | Right Occlusion    | Occlusion Difference

    Parameters
    ----------
    model      : UNet
    inputs     : np.memmap
    targets    : np.memmap
    sample_idx : int
    device     : torch.device
    save_dir   : Path or None
    """
    x, _, _ = get_sample(inputs, targets, sample_idx, device)

    left_contrib, right_contrib, dominance, left_cs, right_cs = \
        compute_slice_contribution(model, x.clone(), device)
    left_occ, right_occ, left_os, right_os = \
        compute_occlusion_sensitivity(model, x.clone(), device)

    occ_diff = left_occ - right_occ

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Dependency Maps \u2014 Sample {sample_idx}\n"
        f"Contribution L/R={left_cs:.4f}/{right_cs:.4f}   "
        f"Occlusion L/R={left_os:.4f}/{right_os:.4f}",
        fontsize=13, fontweight="bold",
    )

    def _show(ax, img, cmap, title, vmin=None, vmax=None, diverging=False):
        if diverging and img.min() < 0 < img.max():
            norm = TwoSlopeNorm(vmin=img.min(), vcenter=0, vmax=img.max())
            im = ax.imshow(img, cmap=cmap, norm=norm)
        else:
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    _show(axes[0, 0], left_contrib,  "hot",    f"Left Slice Contribution\n(mean={left_cs:.4f})",   0, left_contrib.max())
    _show(axes[0, 1], right_contrib, "hot",    f"Right Slice Contribution\n(mean={right_cs:.4f})", 0, right_contrib.max())
    _show(axes[0, 2], dominance,     "RdBu_r", "Dominance Map\n(+red=left, \u2212blue=right)",       diverging=True)
    _show(axes[1, 0], left_occ,      "viridis", f"Left Occlusion Sensitivity\n(mean={left_os:.4f})",  0, left_occ.max())
    _show(axes[1, 1], right_occ,     "viridis", f"Right Occlusion Sensitivity\n(mean={right_os:.4f})", 0, right_occ.max())
    _show(axes[1, 2], occ_diff,      "RdBu_r", "Occlusion Difference\n(+red=left, \u2212blue=right)",  diverging=True)

    plt.tight_layout()
    if save_dir:
        path = save_dir / f"dependency_maps_sample_{sample_idx}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Easy vs hard comparison
# ---------------------------------------------------------------------------

def compare_easy_vs_hard(model, gradcam, inputs, targets, device, n_each=3, save_path=None):
    """
    Visualise how Grad-CAM attention differs across reconstruction difficulty.

    Approach
    --------
    Score the first 1000 samples by PSNR (a proxy for reconstruction
    difficulty), then select n_each samples from the bottom (hard), middle
    (medium), and top (easy) of the ranking.  For each, show:
      - Left column: Grad-CAM heat map overlaid on the ground truth.
      - Right column: Absolute error map.

    Expected pattern
    ----------------
    Hard samples  ÔÇö widespread, diffuse Grad-CAM activation; model is uncertain.
    Medium samples ÔÇö attention focused on structural edges and tissue boundaries.
    Easy samples  ÔÇö almost no activation; slices are nearly identical so the
                    model barely needs to "look" at anything.

    Parameters
    ----------
    model     : UNet
    gradcam   : GradCAM
    inputs    : np.memmap  shape (N, 2, H, W)
    targets   : np.memmap  shape (N, H, W)
    device    : torch.device
    n_each    : int   ÔÇö samples per difficulty tier
    save_path : Path or None
    """
    print("Scoring samples to find easy / medium / hard cases...")
    scored = []
    for idx in range(min(1000, len(inputs))):
        x, _, y_np = get_sample(inputs, targets, idx, device)
        with torch.no_grad():
            pred = model(x).squeeze().cpu().numpy()
        scored.append((psnr(y_np, pred, data_range=1.0), idx))

    scored.sort(key=lambda t: t[0])   # ascending ÔåÆ hard first
    n      = len(scored)
    hard   = [scored[i][1] for i in range(n_each)]
    medium = [scored[i][1] for i in range(n // 2 - n_each // 2, n // 2 + n_each // 2 + 1)][:n_each]
    easy   = [scored[i][1] for i in range(n - n_each, n)]

    categories = [
        ("Hard   (low PSNR)",  hard,   "red"),
        ("Medium (mid PSNR)",  medium, "orange"),
        ("Easy   (high PSNR)", easy,   "green"),
    ]

    fig, axes = plt.subplots(3, n_each * 2, figsize=(n_each * 8, 14))
    fig.suptitle(
        "Grad-CAM: Easy vs Medium vs Hard Reconstructions\n"
        "(all from ├ù2 test set ÔÇö the task the model was trained on)",
        fontsize=13, fontweight="bold",
    )

    for row, (label, indices, color) in enumerate(categories):
        for col, idx in enumerate(indices):
            x, _, y_np = get_sample(inputs, targets, idx, device)
            with torch.no_grad():
                pred = model(x).squeeze().cpu().numpy()
            cam         = gradcam.generate(x.clone())
            sample_psnr = psnr(y_np, pred, data_range=1.0)
            sample_ssim = ssim(y_np, pred, data_range=1.0)

            # Left sub-column: CAM overlay.
            ax1 = axes[row, col * 2]
            ax1.imshow(y_np, cmap="gray", vmin=0, vmax=1)
            ax1.imshow(cam,  cmap="jet",  alpha=0.5, vmin=0, vmax=1)
            ax1.set_title(
                f"{label}\nPSNR={sample_psnr:.1f}dB  SSIM={sample_ssim:.3f}",
                fontsize=9, color=color, fontweight="bold",
            )
            ax1.axis("off")

            # Right sub-column: absolute error map.
            ax2   = axes[row, col * 2 + 1]
            error = np.abs(y_np - pred)
            ax2.imshow(error, cmap="hot", vmin=0, vmax=0.3)
            ax2.set_title(f"Error Map\nMAE={error.mean():.4f}", fontsize=9)
            ax2.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Attribution statistics
# ---------------------------------------------------------------------------

def print_attribution_stats(model, gradcam, inputs, targets, sample_idxs, device):
    """
    Print and save extended summary statistics across multiple samples.

    Covers: IG attribution, Grad-CAM, MC uncertainty, slice contribution,
    and occlusion sensitivity ratios.

    Parameters
    ----------
    model       : UNet
    gradcam     : GradCAM
    inputs      : np.memmap
    targets     : np.memmap
    sample_idxs : array-like
    device      : torch.device
    """
    mean_left, mean_right, mean_cam = [], [], []
    unc_scores = []
    left_cs_list, right_cs_list = [], []
    left_os_list, right_os_list = [], []

    for idx in sample_idxs:
        x, _, _ = get_sample(inputs, targets, idx, device)
        attrs = compute_integrated_gradients(model, x.clone(), device)
        cam   = gradcam.generate(x.clone())
        mean_left.append(attrs[0].mean())
        mean_right.append(attrs[1].mean())
        mean_cam.append(cam.mean())

        mc_preds = mc_forward_passes(model, x.clone(), device, n_passes=20)
        _, _, unc = summarise_uncertainty(mc_preds)
        unc_scores.append(unc)

        _, _, _, left_cs, right_cs = compute_slice_contribution(model, x.clone(), device)
        left_cs_list.append(left_cs)
        right_cs_list.append(right_cs)

        _, _, left_os, right_os = compute_occlusion_sensitivity(model, x.clone(), device)
        left_os_list.append(left_os)
        right_os_list.append(right_os)

    ratio    = np.mean(mean_left)    / (np.mean(mean_right)    + 1e-12)
    cs_ratio = np.mean(left_cs_list) / (np.mean(right_cs_list) + 1e-12)
    os_ratio = np.mean(left_os_list) / (np.mean(right_os_list) + 1e-12)

    lines = [
        "========== EXPLAINABILITY SUMMARY ==========",
        f"Mean IG — Left slice          : {np.mean(mean_left):.2e} ± {np.std(mean_left):.2e}",
        f"Mean IG — Right slice         : {np.mean(mean_right):.2e} ± {np.std(mean_right):.2e}",
        f"Mean Grad-CAM                 : {np.mean(mean_cam):.5f} ± {np.std(mean_cam):.5f}",
        "",
        f"Left/Right IG ratio           : {ratio:.4f}  (1.0 = perfectly symmetric)",
        "",
        f"MC Uncertainty (mean var)     : {np.mean(unc_scores):.2e} ± {np.std(unc_scores):.2e}",
        "",
        f"Slice contribution — Left     : {np.mean(left_cs_list):.4f} ± {np.std(left_cs_list):.4f}",
        f"Slice contribution — Right    : {np.mean(right_cs_list):.4f} ± {np.std(right_cs_list):.4f}",
        f"Left/Right contribution ratio  : {cs_ratio:.4f}",
        "",
        f"Occlusion sensitivity — Left  : {np.mean(left_os_list):.4f} ± {np.std(left_os_list):.4f}",
        f"Occlusion sensitivity — Right : {np.mean(right_os_list):.4f} ± {np.std(right_os_list):.4f}",
        f"Left/Right occlusion ratio     : {os_ratio:.4f}",
    ]

    for line in lines:
        print(line)

    stats_path = EXPLAINABILITY_DIR / "explainability_summary_stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary saved to: {stats_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Run the full explainability analysis on N_EXPL_SAMPLES representative samples.
    """
    EXPLAINABILITY_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = UNet().to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "unet_best.pth", map_location=device))
    model.eval()
    print(f"Model loaded from {MODELS_DIR / 'unet_best.pth'}")

    inputs  = np.load(PROCESSED_DIR / "dataset_inputs.npy",  mmap_mode="r")
    targets = np.load(PROCESSED_DIR / "dataset_targets.npy", mmap_mode="r")

    # Hook Grad-CAM onto the final decoder block (full spatial resolution).
    gradcam = GradCAM(model, target_layer=model.decoders[-1])

    # Spread sample indices uniformly so diverse regions of the dataset
    # are covered rather than clustering near the start or end.
    sample_idxs = np.linspace(0, len(inputs) - 1, N_EXPL_SAMPLES, dtype=int)

    print(f"\nGenerating explainability maps for {N_EXPL_SAMPLES} samples...")
    for idx in sample_idxs:
        print(f"  Processing sample {idx}...")
        visualize_explainability(
            model, gradcam, inputs, targets, int(idx), device,
            save_dir=EXPLAINABILITY_DIR,
        )
        visualize_dependency_maps(
            model, inputs, targets, int(idx), device,
            save_dir=EXPLAINABILITY_DIR,
        )

    compare_easy_vs_hard(
        model, gradcam, inputs, targets, device, n_each=3,
        save_path=EXPLAINABILITY_DIR / "gradcam_easy_vs_hard.png",
    )

    print_attribution_stats(model, gradcam, inputs, targets, sample_idxs, device)
    print(f"\nAll outputs saved to: {EXPLAINABILITY_DIR}")


if __name__ == "__main__":
    main()
