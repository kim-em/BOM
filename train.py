#!/usr/bin/env python3
"""
Training script for radar nowcasting models.

Usage:
    python train.py --model unet          # Experiment 3 (fastest)
    python train.py --model simvp         # Experiment 2
    python train.py --model convgru       # Experiment 1
    python train.py --model ds_convlstm   # Experiment 4
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import MODEL_REGISTRY


# --- Dataset ---

class RadarSequenceDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)['data']  # (N, 12, 128, 128)
        self.inputs = data[:, :6]   # first 6 frames
        self.targets = data[:, 6:]  # last 6 frames
        print(f"  Loaded {len(self)} sequences from {npz_path}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx])   # (6, 128, 128)
        y = torch.from_numpy(self.targets[idx])   # (6, 128, 128)
        return x, y


# --- Loss ---

def make_radar_mask(size=128, device='cpu'):
    """Create a circular mask for the radar coverage area.
    Pixels outside the circle are not penalized in the loss."""
    y, x = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing='ij'
    )
    center = size / 2.0
    radius = center - 1  # slightly inside the image edge
    mask = ((x - center) ** 2 + (y - center) ** 2) <= radius ** 2
    return mask.float().to(device)  # (128, 128)


# Global mask — created once per training run
_radar_mask = None

def get_radar_mask(device):
    global _radar_mask
    if _radar_mask is None or _radar_mask.device != device:
        _radar_mask = make_radar_mask(128, device)
    return _radar_mask


def weighted_mse_loss(pred, target, min_weight=0.01):
    """MSE weighted by reflectivity squared, masked to radar circle only."""
    mask = get_radar_mask(pred.device)  # (128, 128)
    weights = torch.clamp(target ** 2, min=min_weight)
    pixel_loss = weights * (pred - target) ** 2
    # Zero out loss for pixels outside radar circle
    masked_loss = pixel_loss * mask.unsqueeze(0).unsqueeze(0)
    return masked_loss.sum() / mask.sum() / pred.shape[0] / pred.shape[1]


def compute_csi(pred, target, threshold):
    """Critical Success Index at a given reflectivity threshold."""
    pred_pos = pred > threshold
    target_pos = target > threshold
    hits = (pred_pos & target_pos).float().sum()
    misses = (~pred_pos & target_pos).float().sum()
    false_alarms = (pred_pos & ~target_pos).float().sum()
    denom = hits + misses + false_alarms
    return (hits / denom).item() if denom > 0 else 0.0


# --- Training ---

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = weighted_mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_mse = 0
    csi_scores = {0.3: 0, 0.5: 0, 0.7: 0}
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += weighted_mse_loss(pred, y).item() * x.size(0)
        total_mse += nn.functional.mse_loss(pred, y).item() * x.size(0)
        for thresh in csi_scores:
            csi_scores[thresh] += compute_csi(pred, y, thresh) * x.size(0)
        n += x.size(0)

    return {
        'weighted_mse': total_loss / n,
        'mse': total_mse / n,
        'csi_0.3': csi_scores[0.3] / n,
        'csi_0.5': csi_scores[0.5] / n,
        'csi_0.7': csi_scores[0.7] / n,
    }


def save_example_predictions(model, loader, device, output_dir, epoch):
    """Save a visual comparison of predictions vs ground truth."""
    model.eval()
    x, y = next(iter(loader))
    x, y = x[:4].to(device), y[:4].to(device)
    with torch.no_grad():
        pred = model(x)

    fig, axes = plt.subplots(4, 12, figsize=(24, 8))
    for row in range(4):
        for t in range(6):
            axes[row, t].imshow(x[row, t].cpu(), vmin=0, vmax=1, cmap='turbo')
            axes[row, t].axis('off')
            if row == 0:
                axes[row, t].set_title(f'In {t+1}', fontsize=8)
        for t in range(6):
            # Top half of cell: prediction, bottom: truth
            axes[row, 6+t].imshow(pred[row, t].cpu(), vmin=0, vmax=1, cmap='turbo')
            axes[row, 6+t].axis('off')
            if row == 0:
                axes[row, 6+t].set_title(f'Pred {t+1}', fontsize=8)

    plt.suptitle(f'Epoch {epoch}: Input (1-6) → Predicted (7-12)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'predictions_epoch_{epoch:03d}.png'), dpi=100)
    plt.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data_dir', type=str, default=os.path.expanduser('~/projects/BOM/data'))
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Output dirs
    results_dir = os.path.expanduser(f'~/projects/BOM/results/{args.model}')
    ckpt_dir = os.path.expanduser(f'~/projects/BOM/checkpoints/{args.model}')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Data
    print("Loading data...")
    train_ds = RadarSequenceDataset(os.path.join(args.data_dir, 'train.npz'))
    val_ds = RadarSequenceDataset(os.path.join(args.data_dir, 'val.npz'))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Model
    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls().to(device)
    n_params = count_parameters(model)
    model_mb = n_params * 4 / 1e6  # float32
    print(f"Model: {args.model}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Size: {model_mb:.1f} MB (float32)")

    # Verify with a dummy input
    with torch.no_grad():
        dummy = torch.zeros(1, 6, 128, 128, device=device)
        out = model(dummy)
        print(f"  Input shape:  {dummy.shape}")
        print(f"  Output shape: {out.shape}")

    # Optimiser
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_mse': [],
               'csi_0.3': [], 'csi_0.5': [], 'csi_0.7': []}

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['weighted_mse'])
        history['val_mse'].append(val_metrics['mse'])
        history['csi_0.3'].append(val_metrics['csi_0.3'])
        history['csi_0.5'].append(val_metrics['csi_0.5'])
        history['csi_0.7'].append(val_metrics['csi_0.7'])

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train={train_loss:.6f} | val={val_metrics['weighted_mse']:.6f} | "
              f"mse={val_metrics['mse']:.6f} | "
              f"CSI@0.3={val_metrics['csi_0.3']:.3f} | "
              f"{elapsed:.1f}s")

        # Save best
        if val_metrics['weighted_mse'] < best_val_loss:
            best_val_loss = val_metrics['weighted_mse']
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best.pt'))

        # Save examples every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            save_example_predictions(model, val_loader, device, results_dir, epoch)

    # Save final model
    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'final.pt'))

    # Save TorchScript for deployment
    model.eval()
    scripted = torch.jit.script(model.cpu())
    script_path = os.path.join(ckpt_dir, f'{args.model}.pt')
    scripted.save(script_path)
    script_size = os.path.getsize(script_path) / 1e6
    print(f"\nTorchScript saved: {script_path} ({script_size:.1f} MB)")

    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Weighted MSE')
    axes[0].set_title('Loss')
    axes[0].legend()

    axes[1].plot(history['val_mse'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Validation MSE')

    axes[2].plot(history['csi_0.3'], label='CSI@0.3')
    axes[2].plot(history['csi_0.5'], label='CSI@0.5')
    axes[2].plot(history['csi_0.7'], label='CSI@0.7')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('CSI')
    axes[2].set_title('Critical Success Index')
    axes[2].legend()

    plt.suptitle(f'{args.model} — {n_params:,} params, {model_mb:.1f} MB', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=150)
    plt.close()

    print(f"\nDone! Results in {results_dir}")
    print(f"Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
