"""
Experiment 2: SimVP-Lite (CNN-only, ~1.5MB)

No recurrence - pure CNN. Encoder extracts spatial features,
Translator mixes temporal information, Decoder reconstructs.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class InceptionBlock(nn.Module):
    """Simplified Inception module for temporal mixing."""

    def __init__(self, ch):
        super().__init__()
        mid = ch // 4
        self.branch1 = nn.Conv2d(ch, mid, 1)
        self.branch3 = nn.Sequential(
            nn.Conv2d(ch, mid, 1),
            nn.Conv2d(mid, mid, 3, padding=1),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(ch, mid, 1),
            nn.Conv2d(mid, mid, 5, padding=2),
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(ch, mid, 1),
        )
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.act = nn.GELU()

    def forward(self, x):
        out = torch.cat([
            self.branch1(x),
            self.branch3(x),
            self.branch5(x),
            self.branch_pool(x),
        ], dim=1)
        return self.act(self.norm(out + x))


class SimVPLite(nn.Module):
    """
    Input: (B, 6, 128, 128) → Output: (B, 6, 128, 128)
    Frames stacked as channels.
    """

    def __init__(self, in_frames=6, out_frames=6, base_ch=32):
        super().__init__()
        # Encoder: spatial downsampling
        self.encoder = nn.Sequential(
            ConvBlock(in_frames, base_ch, stride=2),     # 64x64
            ConvBlock(base_ch, base_ch * 2, stride=2),   # 32x32
            ConvBlock(base_ch * 2, base_ch * 2, stride=2),  # 16x16
        )

        # Translator: temporal mixing at low resolution
        self.translator = nn.Sequential(
            InceptionBlock(base_ch * 2),
            InceptionBlock(base_ch * 2),
        )

        # Decoder: spatial upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1),  # 32x32
            nn.GroupNorm(8, base_ch * 2),
            nn.GELU(),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1),  # 64x64
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
            nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1),  # 128x128
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
            nn.Conv2d(base_ch, out_frames, 1),
        )

    def forward(self, x):
        enc = self.encoder(x)
        trans = self.translator(enc)
        out = self.decoder(trans)
        return torch.sigmoid(out)
