#!/usr/bin/env python3
"""
Generate composite comparison images.
Row 1: 6 input frames
Row 2: 6 predicted frames
Row 3: 6 ground truth frames
Row 4: symmetric difference (|pred - truth|)
"""

import os
import sys
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from models import MODEL_REGISTRY

# BOM radar colormap matching the reflectivity levels in preprocess.py
# 0.0 = no data, 1/15 through 14/15 = increasing reflectivity
# Must have 16 entries to cover [0, 1/15, 2/15, ..., 14/15, 1.0]
RADAR_COLORS = [
    (0, 0, 0),           # 0/15 = no data (black)
    (0.96, 0.96, 1.0),   # 1/15 = very faint (245,245,255)
    (0.71, 0.71, 1.0),   # 2/15 = lavender (180,180,255)
    (0.47, 0.47, 1.0),   # 3/15 = light blue (120,120,255)
    (0.08, 0.08, 1.0),   # 4/15 = blue (20,20,255)
    (0, 0.85, 0.76),     # 5/15 = cyan (0,216,195)
    (0, 0.59, 0.56),     # 6/15 = teal (0,150,144)
    (0, 0.40, 0.40),     # 7/15 = dark teal (0,102,102)
    (1.0, 1.0, 0),       # 8/15 = yellow
    (1.0, 0.78, 0),      # 9/15 = gold
    (1.0, 0.59, 0),      # 10/15 = orange
    (1.0, 0.39, 0),      # 11/15 = dark orange
    (1.0, 0, 0),         # 12/15 = red
    (0.78, 0, 0),        # 13/15 = dark red
    (0.47, 0, 0),        # 14/15 = very dark red
    (0.25, 0, 0),        # 15/15 = extreme
]
RADAR_CMAP = ListedColormap(RADAR_COLORS)

# Difference colormap: black (no diff) → yellow → red
DIFF_COLORS = [
    (0, 0, 0),
    (0.2, 0.1, 0),
    (0.5, 0.3, 0),
    (0.8, 0.6, 0),
    (1.0, 0.8, 0),
    (1.0, 0.5, 0),
    (1.0, 0.2, 0),
    (1.0, 0, 0),
]
DIFF_CMAP = ListedColormap(DIFF_COLORS)


def load_model(model_name, checkpoint_dir, device):
    model = MODEL_REGISTRY[model_name]()
    ckpt = os.path.join(checkpoint_dir, model_name, 'best.pt')
    if not os.path.exists(ckpt):
        ckpt = os.path.join(checkpoint_dir, model_name, 'final.pt')
    if not os.path.exists(ckpt):
        print(f"No checkpoint found for {model_name}")
        return None
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def make_comparison(model, model_name, data_path, output_dir, device, n_examples=4):
    """Generate composite comparison images."""
    data = np.load(data_path)['data']  # (N, 12, 128, 128)
    os.makedirs(output_dir, exist_ok=True)

    # Pick examples with interesting weather (sort by total reflectivity, take top ones)
    weather_scores = data[:, :, :, :].sum(axis=(1, 2, 3))
    interesting_idx = np.argsort(weather_scores)[-n_examples * 5:]  # pool of interesting
    np.random.seed(42)
    chosen = np.random.choice(interesting_idx, n_examples, replace=False)
    chosen.sort()

    for ex_num, idx in enumerate(chosen):
        seq = data[idx]  # (12, 128, 128)
        inputs = torch.from_numpy(seq[:6]).unsqueeze(0).to(device)  # (1, 6, 128, 128)
        truth = seq[6:]  # (6, 128, 128)

        with torch.no_grad():
            pred = model(inputs).cpu().numpy()[0]  # (6, 128, 128)

        diff = np.abs(pred - truth)

        # Build the 4-row composite
        fig, axes = plt.subplots(4, 6, figsize=(18, 12))

        row_labels = ['Input', 'Predicted', 'Ground Truth', '|Difference|']

        for t in range(6):
            # Row 0: Input
            axes[0, t].imshow(seq[t], vmin=0, vmax=1, cmap=RADAR_CMAP)
            axes[0, t].axis('off')
            axes[0, t].set_title(f't-{6-t}' if t < 5 else 't=0', fontsize=10)

            # Row 1: Predicted
            axes[1, t].imshow(pred[t], vmin=0, vmax=1, cmap=RADAR_CMAP)
            axes[1, t].axis('off')
            axes[1, t].set_title(f't+{t+1}', fontsize=10)

            # Row 2: Ground truth
            axes[2, t].imshow(truth[t], vmin=0, vmax=1, cmap=RADAR_CMAP)
            axes[2, t].axis('off')

            # Row 3: Difference
            axes[3, t].imshow(diff[t], vmin=0, vmax=0.5, cmap=DIFF_CMAP)
            axes[3, t].axis('off')

        # Row labels
        for row, label in enumerate(row_labels):
            axes[row, 0].text(-0.15, 0.5, label, transform=axes[row, 0].transAxes,
                              fontsize=12, fontweight='bold', va='center', ha='right',
                              rotation=90)

        # Per-frame MSE in difference row
        for t in range(6):
            frame_mse = np.mean(diff[t] ** 2)
            axes[3, t].set_title(f'MSE: {frame_mse:.5f}', fontsize=8)

        overall_mse = np.mean(diff ** 2)
        plt.suptitle(f'{model_name} — Example {ex_num+1} (MSE: {overall_mse:.5f})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        out_path = os.path.join(output_dir, f'{model_name}_example_{ex_num+1}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--data', type=str,
                        default=os.path.expanduser('~/projects/BOM/data/test.npz'))
    parser.add_argument('--n', type=int, default=4, help='Number of examples')
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    ckpt_dir = os.path.expanduser('~/projects/BOM/checkpoints')
    output_dir = os.path.expanduser(f'~/projects/BOM/results/{args.model}')

    model = load_model(args.model, ckpt_dir, device)
    if model is None:
        sys.exit(1)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({params:,} params)")
    print(f"Generating {args.n} comparison images...")

    make_comparison(model, args.model, args.data, output_dir, device, args.n)
    print("Done!")


if __name__ == "__main__":
    main()
