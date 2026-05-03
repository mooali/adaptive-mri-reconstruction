#!/usr/bin/env python
"""
Explainability analysis for the trained U-Net: Grad-CAM and Integrated Gradients.

Reads:   models/unet_best.pth
         data/processed/dataset_inputs.npy
         data/processed/dataset_targets.npy

Writes:  outputs/explainability/explainability_sample_<idx>.png  (one per sample)
         outputs/explainability/gradcam_easy_vs_hard.png
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
)
from src.train import UNet


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """Gradient-weighted Class Activation Mapping for a target conv layer."""

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x):
        # Use train() so BatchNorm computes per-batch statistics (stronger gradients)
        self.model.train()
        x      = x.requires_grad_(True)
        output = self.model(x)
        self.model.zero_grad()
        output.mean().backward()
        self.model.eval()

        if self.gradients is None or self.activations is None:
            print("WARNING: No gradients captured — check target layer.")
            return np.zeros((256, 256))

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1, keepdim=True))

        if cam.max() == 0:
            return np.zeros((256, 256))

        cam = F.interpolate(cam, size=(256, 256), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ---------------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------------

def compute_integrated_gradients(model, x, device, n_steps=IG_STEPS):
    """
    Return per-pixel attribution map of shape (2, 256, 256) for a single sample.
    Baseline is a zero image; path integrates from baseline to input.
    """
    baseline = torch.zeros_like(x).to(device)
    x        = x.to(device)
    attrs    = torch.zeros_like(x).to(device)

    for step in range(n_steps):
        alpha  = step / n_steps
        interp = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        model.zero_grad()
        model(interp).mean().backward()
        attrs += interp.grad.detach()

    attrs = (x.detach() - baseline) * (attrs / n_steps)
    return np.abs(attrs.squeeze().cpu().numpy())


# ---------------------------------------------------------------------------
# Sample helpers
# ---------------------------------------------------------------------------

def get_sample(inputs, targets, idx, device):
    x_np = np.array(inputs[idx])
    y_np = np.array(targets[idx])
    x    = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(device)
    return x, x_np, y_np


# ---------------------------------------------------------------------------
# Per-sample explainability plot
# ---------------------------------------------------------------------------

def visualize_explainability(model, gradcam, inputs, targets, sample_idx, device, save_dir=None):
    x, x_np, y_np = get_sample(inputs, targets, sample_idx, device)

    with torch.no_grad():
        pred_np = model(x).squeeze().cpu().numpy()

    cam        = gradcam.generate(x.clone())
    attrs      = compute_integrated_gradients(model, x.clone(), device)
    left_attr  = attrs[0]
    right_attr = attrs[1]

    sample_psnr = psnr(y_np, pred_np, data_range=1.0)
    sample_ssim = ssim(y_np, pred_np, data_range=1.0)
    error_map   = np.abs(y_np - pred_np)

    fig = plt.figure(figsize=(24, 10))
    gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(
        f"Explainability Analysis — Sample {sample_idx}\n"
        f"PSNR={sample_psnr:.2f}dB   SSIM={sample_ssim:.4f}",
        fontsize=14, fontweight="bold",
    )

    panels = [
        # (row, col, image, cmap, title, vmin, vmax)
        (0, 0, x_np[0],    "gray",   "Input: Left Slice",          0,              1),
        (0, 1, pred_np,    "gray",   "U-Net Prediction",           0,              1),
        (0, 2, y_np,       "gray",   "Ground Truth",               0,              1),
        (0, 3, error_map,  "hot",    "Error Map  |GT − Pred|",     0,              error_map.max()),
        (0, 4, x_np[1],    "gray",   "Input: Right Slice",         0,              1),
        (1, 0, left_attr,  "hot",    "IG: Left Slice Attribution", 0,              left_attr.max()),
        (1, 1, right_attr, "hot",    "IG: Right Slice Attribution",0,              right_attr.max()),
        (1, 2, cam,        "jet",    "Grad-CAM (Last Decoder)",    0,              1),
        (1, 3, cam,        "jet",    "Grad-CAM Overlay",           0,              1),
        (1, 4, left_attr - right_attr, "RdBu_r",
               "IG Difference\n(Left − Right)",                    None,           None),
    ]

    for row, col, img, cmap, title, vmin, vmax in panels:
        ax = fig.add_subplot(gs[row, col])
        if title == "Grad-CAM Overlay":
            ax.imshow(y_np, cmap="gray", vmin=0, vmax=1)
            ax.imshow(cam,  cmap="jet",  alpha=0.45, vmin=0, vmax=1)
        elif title.startswith("IG Difference"):
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
# Easy vs hard comparison
# ---------------------------------------------------------------------------

def compare_easy_vs_hard(model, gradcam, inputs, targets, device, n_each=3, save_path=None):
    """Score first 1000 samples by PSNR and visualise hard / medium / easy cases."""
    print("Scoring samples to find easy / medium / hard cases...")
    scored = []
    for idx in range(min(1000, len(inputs))):
        x, _, y_np = get_sample(inputs, targets, idx, device)
        with torch.no_grad():
            pred = model(x).squeeze().cpu().numpy()
        scored.append((psnr(y_np, pred, data_range=1.0), idx))

    scored.sort(key=lambda t: t[0])
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
        "(all from ×2 test set — the task the model was trained on)",
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

            ax1 = axes[row, col * 2]
            ax1.imshow(y_np, cmap="gray", vmin=0, vmax=1)
            ax1.imshow(cam,  cmap="jet",  alpha=0.5, vmin=0, vmax=1)
            ax1.set_title(
                f"{label}\nPSNR={sample_psnr:.1f}dB  SSIM={sample_ssim:.3f}",
                fontsize=9, color=color, fontweight="bold",
            )
            ax1.axis("off")

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
    mean_left, mean_right, mean_cam = [], [], []
    for idx in sample_idxs:
        x, _, _ = get_sample(inputs, targets, idx, device)
        attrs   = compute_integrated_gradients(model, x.clone(), device)
        cam     = gradcam.generate(x.clone())
        mean_left.append(attrs[0].mean())
        mean_right.append(attrs[1].mean())
        mean_cam.append(cam.mean())

    print("\n========== EXPLAINABILITY SUMMARY ==========")
    print(f"Mean IG — Left slice  : {np.mean(mean_left):.2e} ± {np.std(mean_left):.2e}")
    print(f"Mean IG — Right slice : {np.mean(mean_right):.2e} ± {np.std(mean_right):.2e}")
    print(f"Mean Grad-CAM         : {np.mean(mean_cam):.5f} ± {np.std(mean_cam):.5f}")
    ratio = np.mean(mean_left) / (np.mean(mean_right) + 1e-12)
    print(f"\nLeft/Right IG ratio   : {ratio:.4f}  (1.0 = perfectly symmetric)")


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
    print(f"Model loaded from {MODELS_DIR / 'unet_best.pth'}")

    inputs  = np.load(PROCESSED_DIR / "dataset_inputs.npy",  mmap_mode="r")
    targets = np.load(PROCESSED_DIR / "dataset_targets.npy", mmap_mode="r")

    gradcam     = GradCAM(model, target_layer=model.decoders[-1])
    sample_idxs = np.linspace(0, len(inputs) - 1, N_EXPL_SAMPLES, dtype=int)

    print(f"\nGenerating explainability maps for {N_EXPL_SAMPLES} samples...")
    for idx in sample_idxs:
        print(f"  Processing sample {idx}...")
        visualize_explainability(
            model, gradcam, inputs, targets, int(idx), device,
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
