#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# In[2]:


DATASET_DIR   = r'/home/ma18l096/dl_project/DL_slices'
MODEL_PATH    = r'/home/ma18l096/dl_project/DL_slices/unet_best.pth'
OUTPUT_DIR    = r'/home/ma18l096/dl_project/DL_slices/explainability'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# In[3]:


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.encoders  = nn.ModuleList()
        self.pools     = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(ConvBlock(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        self.upconvs    = nn.ModuleList()
        self.decoders   = nn.ModuleList()
        ch = features[-1] * 2
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(f * 2, f))
            ch = f
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid     = nn.Sigmoid()

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.sigmoid(self.output_conv(x))

# Load trained model
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"Model loaded from {MODEL_PATH}")


# In[4]:


inputs_path  = os.path.join(DATASET_DIR, 'dataset_inputs.npy')
targets_path = os.path.join(DATASET_DIR, 'dataset_targets.npy')

inputs  = np.load(inputs_path,  mmap_mode='r')
targets = np.load(targets_path, mmap_mode='r')

# Pick a few representative samples spread across the dataset
N_SAMPLES   = 6
sample_idxs = np.linspace(0, len(inputs) - 1, N_SAMPLES, dtype=int)

def get_sample(idx):
    """Return (input_tensor, target_tensor, numpy arrays) for one sample."""
    x_np  = np.array(inputs[idx])                          # (2, 256, 256)
    y_np  = np.array(targets[idx])                         # (256, 256)
    x     = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, 2, 256, 256)
    return x, x_np, y_np


# In[5]:


class GradCAM:
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
        # Use train() so BatchNorm doesn't kill gradients
        # but disable dropout if you had any
        self.model.train()
        x      = x.requires_grad_(True)
        output = self.model(x)
        score  = output.mean()
        self.model.zero_grad()
        score.backward()
        self.model.eval()   # restore eval after

        if self.gradients is None or self.activations is None:
            print("WARNING: No gradients captured — check target layer")
            return np.zeros((256, 256))

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)

        if cam.max() == 0:
            print("WARNING: CAM is all zeros after ReLU")
            return np.zeros((256, 256))

        cam = F.interpolate(cam, size=(256, 256), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# Hook into the LAST decoder block instead of bottleneck
# This is much closer to the output so gradients are stronger
gradcam = GradCAM(model, target_layer=model.decoders[-1])


# In[10]:


def compute_integrated_gradients(x, n_steps=50):
    baseline = torch.zeros_like(x).to(DEVICE)
    x        = x.to(DEVICE)
    attrs    = torch.zeros_like(x).to(DEVICE) 

    for step in range(n_steps):
        alpha  = step / n_steps
        interp = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)

        model.zero_grad()
        output = model(interp)
        score  = output.mean()
        score.backward()

        attrs += interp.grad.detach()

    attrs = (x.detach() - baseline) * (attrs / n_steps)
    return np.abs(attrs.squeeze().cpu().numpy())


# In[11]:


def visualize_explainability(sample_idx, save=True):
    x, x_np, y_np = get_sample(sample_idx)

    # Get prediction
    with torch.no_grad():
        pred_np = model(x).squeeze().cpu().numpy()

    # Compute explanations
    cam  = gradcam.generate(x.clone())
    attrs = compute_integrated_gradients(x.clone())   # (2, 256, 256)

    left_attr  = attrs[0]   # IG attribution for left input slice
    right_attr = attrs[1]   # IG attribution for right input slice

    # Metrics
    sample_psnr = psnr(y_np, pred_np, data_range=1.0)
    sample_ssim = ssim(y_np, pred_np, data_range=1.0)
    error_map   = np.abs(y_np - pred_np)

    # Plot
    fig = plt.figure(figsize=(24, 10))
    gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(
        f'Explainability Analysis — Sample {sample_idx}\n'
        f'PSNR={sample_psnr:.2f}dB   SSIM={sample_ssim:.4f}',
        fontsize=14, fontweight='bold'
    )

    panels = [
        # Row 0
        (0, 0, x_np[0],    'gray',  'Input: Left Slice',           0, 1),
        (0, 1, pred_np,    'gray',  'U-Net Prediction',            0, 1),
        (0, 2, y_np,       'gray',  'Ground Truth',                0, 1),
        (0, 3, error_map,  'hot',   'Error Map  |GT − Pred|',      0, error_map.max()),
        (0, 4, x_np[1],    'gray',  'Input: Right Slice',          0, 1),
        # Row 1
        (1, 0, left_attr,  'hot',   'IG: Left Slice Attribution',  0, left_attr.max()),
        (1, 1, right_attr, 'hot',   'IG: Right Slice Attribution', 0, right_attr.max()),
        (1, 2, cam,        'jet',   'Grad-CAM (Bottleneck)',       0, 1),
        (1, 3, cam,        'jet',   'Grad-CAM Overlay',            0, 1),
        (1, 4, (left_attr - right_attr), 'RdBu_r',
               'IG Difference\n(Left − Right)', None, None),
    ]

    for row, col, img, cmap, title, vmin, vmax in panels:
        ax = fig.add_subplot(gs[row, col])

        if title == 'Grad-CAM Overlay':
            # Overlay CAM on ground truth
            ax.imshow(y_np, cmap='gray', vmin=0, vmax=1)
            ax.imshow(cam,  cmap='jet',  alpha=0.45, vmin=0, vmax=1)
        elif title.startswith('IG Difference'):
            norm = TwoSlopeNorm(vmin=img.min(), vcenter=0, vmax=img.max())
            im   = ax.imshow(img, cmap=cmap, norm=norm)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

    if save:
        path = os.path.join(OUTPUT_DIR, f'explainability_sample_{sample_idx}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.show()
    plt.close()

# Run for all selected samples
print(f"\nGenerating explainability maps for {N_SAMPLES} samples...")
for idx in sample_idxs:
    print(f"  Processing sample {idx}...")
    visualize_explainability(idx)


# Image 1 — Explainability Analysis (Sample 25689, PSNR=25.70dB)
# This is a hard sample. The bright cyan/yellow arc in the Grad-CAM corresponds to the skull boundary and outer cortex — the model is focusing heavily on the brain edge to anchor its reconstruction. This makes anatomical sense because the skull/cortex boundary is the sharpest intensity transition in the image and gives the model the strongest spatial reference point.
# The error map shows errors spread across the whole brain interior, while the Grad-CAM focuses on the boundary — this is actually a slight concern, suggesting the model uses the edges as anchors but struggles with interior tissue detail.

# In[12]:


def compare_easy_vs_hard(n_each=3):
    """
    Compare EASY vs MEDIUM vs HARD samples from the ×2 test set,
    ranked by how well the model reconstructs them (PSNR).
    """
    print("Scoring all samples to find easy/medium/hard cases...")

    scored = []
    for idx in range(min(1000, len(inputs))):   # score first 1000 for speed
        x, x_np, y_np = get_sample(idx)
        with torch.no_grad():
            pred = model(x).squeeze().cpu().numpy()
        score = psnr(y_np, pred, data_range=1.0)
        scored.append((score, idx))

    scored.sort(key=lambda t: t[0])

    # Pick n_each from bottom (hard), middle, and top (easy)
    n      = len(scored)
    hard   = [scored[i][1] for i in range(n_each)]
    medium = [scored[i][1] for i in range(n//2 - n_each//2, n//2 + n_each//2 + 1)][:n_each]
    easy   = [scored[i][1] for i in range(n - n_each, n)]

    categories = [
        ('Hard   (low PSNR)',   hard,   'red'),
        ('Medium (mid PSNR)',   medium, 'orange'),
        ('Easy   (high PSNR)',  easy,   'green'),
    ]

    fig, axes = plt.subplots(3, n_each * 2, figsize=(n_each * 8, 14))
    fig.suptitle(
        'Grad-CAM: Easy vs Medium vs Hard Reconstructions\n'
        '(all from ×2 test set — the task the model was trained on)',
        fontsize=13, fontweight='bold'
    )

    for row, (label, indices, color) in enumerate(categories):
        for col, idx in enumerate(indices):
            x, x_np, y_np = get_sample(idx)

            with torch.no_grad():
                pred = model(x).squeeze().cpu().numpy()

            cam         = gradcam.generate(x.clone())
            sample_psnr = psnr(y_np, pred, data_range=1.0)
            sample_ssim = ssim(y_np, pred, data_range=1.0)

            # Left column: Grad-CAM overlay on ground truth
            ax1 = axes[row, col * 2]
            ax1.imshow(y_np, cmap='gray', vmin=0, vmax=1)
            ax1.imshow(cam,  cmap='jet',  alpha=0.5, vmin=0, vmax=1)
            ax1.set_title(
                f'{label}\nPSNR={sample_psnr:.1f}dB  SSIM={sample_ssim:.3f}',
                fontsize=9, color=color, fontweight='bold'
            )
            ax1.axis('off')

            # Right column: error map
            ax2 = axes[row, col * 2 + 1]
            error = np.abs(y_np - pred)
            ax2.imshow(error, cmap='hot', vmin=0, vmax=0.3)
            ax2.set_title(f'Error Map\nMAE={error.mean():.4f}', fontsize=9)
            ax2.axis('off')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'gradcam_easy_vs_hard.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.show()
    plt.close()

compare_easy_vs_hard(n_each=3)


# Hard cases (PSNR 23–26dB) — Grad-CAM is bright and widespread, covering large regions of the brain including ventricles and deep structures. The model is casting a wide net because it's uncertain. Error maps are red everywhere — high error throughout.
# 
# Medium cases (PSNR 31–32dB) — Grad-CAM concentrates on the skull boundary and major tissue interfaces. The model has found a reliable strategy but still relies heavily on structural edges. Error maps show moderate error mostly in high-contrast regions.
# 
# Easy cases (PSNR 51–52dB) — Grad-CAM collapses to tiny bright spots with almost no activation elsewhere. The error maps are nearly black — essentially zero error. The model barely needs to "look" at anything because the slices are so similar (these are likely background/edge-of-brain slices).
# 

# In[14]:


print("\nComputing mean attribution statistics across all samples...")

mean_left_attr  = []
mean_right_attr = []
mean_cam        = []

for idx in sample_idxs:
    x, _, _ = get_sample(idx)
    attrs   = compute_integrated_gradients(x.clone())
    cam     = gradcam.generate(x.clone())
    mean_left_attr.append(attrs[0].mean())
    mean_right_attr.append(attrs[1].mean())
    mean_cam.append(cam.mean())

print("\n========== EXPLAINABILITY SUMMARY ==========")
print(f"Mean IG attribution — Left slice  : {np.mean(mean_left_attr):.2e} ± {np.std(mean_left_attr):.2e}")
print(f"Mean IG attribution — Right slice : {np.mean(mean_right_attr):.2e} ± {np.std(mean_right_attr):.2e}")
print(f"Mean Grad-CAM activation          : {np.mean(mean_cam):.5f} ± {np.std(mean_cam):.5f}")
print(f"\nLeft/Right attribution ratio      : {np.mean(mean_left_attr)/np.mean(mean_right_attr):.4f}")
print("  (1.0 = perfectly symmetric, >1.0 = model leans on left slice more)")
print(f"\nAll outputs saved to: {OUTPUT_DIR}")


# Left/Right attribution ratio: 0.9482
# This is the most meaningful number. It means the model relies on the left and right input slices almost equally — 0.9482 is very close to 1.0 (perfect symmetry). This is exactly what you want from an interpolation model. If it were say 0.3 or 1.8, it would mean the model was nearly ignoring one of its two inputs, which would be a design flaw.
# 
# IG attributions: 6.06e-07 vs 6.39e-07
# These are extremely small values, which actually tells you something important — the model doesn't rely heavily on any single specific pixel. Instead it distributes its attention broadly across the whole image. This is typical behaviour for a U-Net doing interpolation because it's essentially doing a smooth blend guided by many pixels simultaneously rather than focusing on a few key points. The visual IG maps in Image 1 confirm this — you can see low-level attribution spread across edges and tissue boundaries rather than concentrated hotspots.
# 
# Grad-CAM activation: 0.00765 ± 0.00595
# The mean activation is low but the standard deviation is nearly as large as the mean (0.006 vs 0.008), which means activation varies a lot between samples. This directly matches what you saw visually — hard samples have much higher activation (the model is working hard, attention is spread wide) while easy samples have near-zero activation (trivial task). The high variance is a feature, not a problem.
# 
# Putting it all together — what you can say about your model:
# The model behaves like a trustworthy interpolation network. It treats both neighboring slices as roughly equally important, it distributes attention across anatomically meaningful regions rather than fixating on irrelevant areas, and its level of attention scales appropriately with how difficult the reconstruction task is. These are all positive indicators for a medical imaging model — the explainability analysis provides no red flags suggesting the model is exploiting spurious shortcuts or ignoring its inputs.
# The one limitation worth noting in your write-up is that the IG values are so small that pixel-level attribution is difficult to interpret with confidence — Grad-CAM is the more reliable explainability signal for this particular architecture.
