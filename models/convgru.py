"""
Experiment 1: ConvGRU Encoder-Decoder (~2MB)

Recurrent model with good temporal modelling. Encoder compresses
spatial information, ConvGRU cells maintain temporal state,
decoder reconstructs predictions.
"""

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    """Single ConvGRU cell."""

    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_ch = hidden_ch
        # Gates: reset and update
        self.gates = nn.Conv2d(in_ch + hidden_ch, 2 * hidden_ch, kernel_size, padding=pad)
        # Candidate
        self.candidate = nn.Conv2d(in_ch + hidden_ch, hidden_ch, kernel_size, padding=pad)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=1)
        gates = torch.sigmoid(self.gates(combined))
        r, z = gates.chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([x, r * h], dim=1)))
        h_new = (1 - z) * h + z * candidate
        return h_new


class ConvGRUNet(nn.Module):
    """
    Encoder-decoder with ConvGRU.
    Input: (B, 6, 128, 128) → Output: (B, 6, 128, 128)
    Processes frames sequentially through encoder, maintains hidden state,
    then generates predictions autoregressively through decoder.
    """

    def __init__(self, in_frames=6, out_frames=6):
        super().__init__()
        self.in_frames = in_frames
        self.out_frames = out_frames

        # Spatial encoder: 128→64→32
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
        )

        # Recurrent core at 32x32 resolution
        self.gru = ConvGRUCell(32, 64)

        # Spatial decoder: 32→64→128
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(16, 1, 1)

    def encode(self, x):
        """Encode single frame: (B, 1, 128, 128) → (B, 32, 32, 32)"""
        return self.enc2(self.enc1(x))

    def decode(self, h):
        """Decode hidden state: (B, 64, 32, 32) → (B, 1, 128, 128)"""
        return torch.sigmoid(self.out_conv(self.dec1(self.dec2(h))))

    def forward(self, x):
        B = x.shape[0]
        device = x.device

        # Initialize hidden state
        h = torch.zeros(B, 64, 32, 32, device=device)

        # Encode input frames
        for t in range(self.in_frames):
            frame = x[:, t:t+1, :, :]  # (B, 1, 128, 128)
            enc = self.encode(frame)
            h = self.gru(enc, h)

        # Generate predictions
        outputs = []
        for t in range(self.out_frames):
            pred = self.decode(h)
            outputs.append(pred)
            # Feed prediction back
            enc = self.encode(pred)
            h = self.gru(enc, h)

        return torch.cat(outputs, dim=1)  # (B, 6, 128, 128)
