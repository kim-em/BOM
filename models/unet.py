"""
Experiment 3: Tiny U-Net (~0.5MB)

Simplest baseline. Stacks 6 input frames as channels, predicts 6 output frames.
Skip connections help preserve spatial detail.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """
    3-level U-Net. Input: (B, 6, 128, 128) → Output: (B, 6, 128, 128)
    6 input frames stacked as channels, 6 output frames as channels.
    """

    def __init__(self, in_frames=6, out_frames=6):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_frames, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(64, 64)

        # Decoder
        self.up3 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec3 = DoubleConv(128, 32)
        self.up2 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.dec2 = DoubleConv(64, 16)
        self.up1 = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.dec1 = DoubleConv(32, 16)

        # Output
        self.out_conv = nn.Conv2d(16, out_frames, 1)

    def forward(self, x):
        # x: (B, 6, 128, 128)
        e1 = self.enc1(x)           # (B, 16, 128, 128)
        e2 = self.enc2(self.pool(e1))  # (B, 32, 64, 64)
        e3 = self.enc3(self.pool(e2))  # (B, 64, 32, 32)

        b = self.bottleneck(self.pool(e3))  # (B, 64, 16, 16)

        d3 = self.up3(b)                     # (B, 64, 32, 32)
        d3 = self.dec3(torch.cat([d3, e3], 1))  # (B, 32, 32, 32)
        d2 = self.up2(d3)                    # (B, 32, 64, 64)
        d2 = self.dec2(torch.cat([d2, e2], 1))  # (B, 16, 64, 64)
        d1 = self.up1(d2)                    # (B, 16, 128, 128)
        d1 = self.dec1(torch.cat([d1, e1], 1))  # (B, 16, 128, 128)

        return torch.sigmoid(self.out_conv(d1))  # (B, 6, 128, 128)
