#!/usr/bin/env python3
"""
GAN training for radar nowcasting.
Uses ConvGRU as generator + PatchGAN discriminator.
Adversarial loss should produce sharper, more realistic predictions.
"""

import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import MODEL_REGISTRY
from models.discriminator import PatchDiscriminator


class RadarSequenceDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)['data']
        self.inputs = data[:, :6]
        self.targets = data[:, 6:]
        print(f"  Loaded {len(self)} sequences from {npz_path}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.targets[idx])


def weighted_mse_loss(pred, target, min_weight=0.01):
    weights = torch.clamp(target ** 2, min=min_weight)
    return (weights * (pred - target) ** 2).mean()


def compute_csi(pred, target, threshold):
    pred_pos = pred > threshold
    target_pos = target > threshold
    hits = (pred_pos & target_pos).float().sum()
    misses = (~pred_pos & target_pos).float().sum()
    false_alarms = (pred_pos & ~target_pos).float().sum()
    denom = hits + misses + false_alarms
    return (hits / denom).item() if denom > 0 else 0.0


def train_epoch(gen, disc, train_loader, opt_g, opt_d, device,
                lambda_adv=0.01, lambda_recon=1.0):
    gen.train()
    disc.train()
    g_losses, d_losses, recon_losses = [], [], []

    for x, y_real in train_loader:
        x, y_real = x.to(device), y_real.to(device)
        batch_size = x.size(0)

        # --- Train Discriminator ---
        y_fake = gen(x).detach()

        # Concatenate input context with output for conditioning
        real_input = y_real
        fake_input = y_fake

        d_real = disc(real_input)
        d_fake = disc(fake_input)

        # Hinge loss
        d_loss = (torch.relu(1 - d_real).mean() + torch.relu(1 + d_fake).mean()) * 0.5

        opt_d.zero_grad()
        d_loss.backward()
        nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
        opt_d.step()
        d_losses.append(d_loss.item())

        # --- Train Generator ---
        y_fake = gen(x)
        d_fake_for_g = disc(y_fake)

        # Adversarial loss (generator wants discriminator to think fake is real)
        g_adv_loss = -d_fake_for_g.mean()

        # Reconstruction loss
        recon_loss = weighted_mse_loss(y_fake, y_real)

        g_loss = lambda_recon * recon_loss + lambda_adv * g_adv_loss

        opt_g.zero_grad()
        g_loss.backward()
        nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt_g.step()

        g_losses.append(g_loss.item())
        recon_losses.append(recon_loss.item())

    return np.mean(g_losses), np.mean(d_losses), np.mean(recon_losses)


@torch.no_grad()
def evaluate(gen, loader, device):
    gen.eval()
    total_wmse = 0
    total_mse = 0
    csi_scores = {0.3: 0, 0.5: 0}
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = gen(x)
        total_wmse += weighted_mse_loss(pred, y).item() * x.size(0)
        total_mse += nn.functional.mse_loss(pred, y).item() * x.size(0)
        for t in csi_scores:
            csi_scores[t] += compute_csi(pred, y, t) * x.size(0)
        n += x.size(0)

    return {
        'wmse': total_wmse / n,
        'mse': total_mse / n,
        'csi_0.3': csi_scores[0.3] / n,
        'csi_0.5': csi_scores[0.5] / n,
    }


def save_visuals(gen, loader, device, output_dir, epoch, n=3):
    from matplotlib.colors import ListedColormap
    RADAR_COLORS = [
        (0, 0, 0), (0.96, 0.96, 1.0), (0.71, 0.71, 1.0), (0.47, 0.47, 1.0),
        (0.08, 0.08, 1.0), (0, 0.85, 0.76), (0, 0.59, 0.56), (0, 0.40, 0.40),
        (1.0, 1.0, 0), (1.0, 0.78, 0), (1.0, 0.59, 0), (1.0, 0.39, 0),
        (1.0, 0, 0), (0.78, 0, 0), (0.47, 0, 0),
    ]
    DIFF_COLORS = [(0,0,0),(0.2,0.1,0),(0.5,0.3,0),(0.8,0.6,0),(1,0.8,0),(1,0.5,0),(1,0.2,0),(1,0,0)]
    rcmap = ListedColormap(RADAR_COLORS)
    dcmap = ListedColormap(DIFF_COLORS)

    gen.eval()
    # Get interesting examples
    data = loader.dataset
    scores = []
    for i in range(len(data)):
        x, y = data[i]
        scores.append(y.sum().item())
    top_idx = np.argsort(scores)[-n*3:]
    np.random.seed(epoch)
    chosen = np.random.choice(top_idx, n, replace=False)

    for ex, idx in enumerate(chosen):
        x, y = data[idx]
        x_dev = x.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = gen(x_dev).cpu()[0]

        diff = torch.abs(pred - y)
        fig, axes = plt.subplots(4, 6, figsize=(18, 12))
        for t in range(6):
            axes[0, t].imshow(x[t], vmin=0, vmax=1, cmap=rcmap); axes[0, t].axis('off')
            axes[0, t].set_title(f't-{6-t}' if t < 5 else 't=0', fontsize=10)
            axes[1, t].imshow(pred[t], vmin=0, vmax=1, cmap=rcmap); axes[1, t].axis('off')
            axes[1, t].set_title(f't+{t+1}', fontsize=10)
            axes[2, t].imshow(y[t], vmin=0, vmax=1, cmap=rcmap); axes[2, t].axis('off')
            axes[3, t].imshow(diff[t], vmin=0, vmax=0.5, cmap=dcmap); axes[3, t].axis('off')
            axes[3, t].set_title(f'MSE:{(diff[t]**2).mean():.5f}', fontsize=8)

        for row, label in enumerate(['Input', 'Predicted', 'Truth', '|Diff|']):
            axes[row, 0].text(-0.15, 0.5, label, transform=axes[row, 0].transAxes,
                              fontsize=12, fontweight='bold', va='center', ha='right', rotation=90)

        mse = (diff**2).mean().item()
        plt.suptitle(f'GAN ConvGRU — Epoch {epoch}, Example {ex+1} (MSE: {mse:.5f})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'gan_epoch{epoch:03d}_ex{ex+1}.png'), dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator', type=str, default='convgru')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr_g', type=float, default=5e-4)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--lambda_adv', type=float, default=0.01)
    parser.add_argument('--pretrained', action='store_true',
                        help='Initialize generator from pretrained checkpoint')
    parser.add_argument('--data_dir', type=str, default=os.path.expanduser('~/projects/BOM/data'))
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    results_dir = os.path.expanduser(f'~/projects/BOM/results/gan_{args.generator}')
    ckpt_dir = os.path.expanduser(f'~/projects/BOM/checkpoints/gan_{args.generator}')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Data
    print("Loading data...")
    train_ds = RadarSequenceDataset(os.path.join(args.data_dir, 'train.npz'))
    val_ds = RadarSequenceDataset(os.path.join(args.data_dir, 'val.npz'))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Models
    gen_cls = MODEL_REGISTRY[args.generator]
    gen = gen_cls().to(device)
    disc = PatchDiscriminator().to(device)

    # Load pretrained generator weights
    if args.pretrained:
        pretrained_path = os.path.expanduser(f'~/projects/BOM/checkpoints/{args.generator}/best.pt')
        if os.path.exists(pretrained_path):
            gen.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
            print(f"  Loaded pretrained generator from {pretrained_path}")

    g_params = sum(p.numel() for p in gen.parameters())
    d_params = sum(p.numel() for p in disc.parameters())
    print(f"Generator: {args.generator} ({g_params:,} params, {g_params*4/1e6:.1f} MB)")
    print(f"Discriminator: PatchGAN ({d_params:,} params, {d_params*4/1e6:.1f} MB)")

    opt_g = torch.optim.AdamW(gen.parameters(), lr=args.lr_g, betas=(0.5, 0.999), weight_decay=1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr_d, betas=(0.5, 0.999), weight_decay=1e-4)

    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=args.epochs)

    best_val = float('inf')
    print(f"\nTraining GAN for {args.epochs} epochs (λ_adv={args.lambda_adv})...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        g_loss, d_loss, recon = train_epoch(
            gen, disc, train_loader, opt_g, opt_d, device,
            lambda_adv=args.lambda_adv)
        val = evaluate(gen, val_loader, device)
        sched_g.step()
        sched_d.step()
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"G={g_loss:.6f} D={d_loss:.4f} recon={recon:.6f} | "
              f"val_wmse={val['wmse']:.6f} mse={val['mse']:.6f} "
              f"CSI@0.3={val['csi_0.3']:.3f} | {elapsed:.0f}s")

        if val['wmse'] < best_val:
            best_val = val['wmse']
            torch.save(gen.state_dict(), os.path.join(ckpt_dir, 'best.pt'))

        if epoch % 5 == 0 or epoch == 1:
            save_visuals(gen, val_ds, device, results_dir, epoch)

    torch.save(gen.state_dict(), os.path.join(ckpt_dir, 'final.pt'))
    print(f"\nDone! Best val wmse: {best_val:.6f}")
    print(f"Results: {results_dir}")


if __name__ == "__main__":
    main()
