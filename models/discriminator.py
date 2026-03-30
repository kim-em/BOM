"""
PatchGAN discriminator for radar nowcasting.
Takes a sequence of 6 frames and classifies each 16x16 patch as real/fake.
"""

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """
    Input: (B, 6, 128, 128) — 6 frames (either real or predicted)
    Output: (B, 1, 8, 8) — real/fake score per patch
    """

    def __init__(self, in_frames=6):
        super().__init__()
        self.net = nn.Sequential(
            # 128→64
            nn.Conv2d(in_frames, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64→32
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
            # 32→16
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
            # 16→8
            nn.Conv2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2, inplace=True),
            # Patch score
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)
