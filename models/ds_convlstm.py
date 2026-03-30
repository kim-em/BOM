"""
Experiment 4: Depthwise-Separable ConvLSTM (~1MB)

Like ConvGRU but uses LSTM cells with depthwise-separable convolutions
for ~60% parameter reduction.
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=pad, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DSConvLSTMCell(nn.Module):
    """ConvLSTM cell with depthwise-separable convolutions."""

    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        self.hidden_ch = hidden_ch
        # All 4 gates in one conv (input, forget, cell, output)
        self.gates = DepthwiseSeparableConv(in_ch + hidden_ch, 4 * hidden_ch, kernel_size)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class DSConvLSTMNet(nn.Module):
    """
    Encoder-decoder with depthwise-separable ConvLSTM.
    Input: (B, 6, 128, 128) → Output: (B, 6, 128, 128)
    """

    def __init__(self, in_frames=6, out_frames=6):
        super().__init__()
        self.in_frames = in_frames
        self.out_frames = out_frames

        # Spatial encoder
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
        )

        # Recurrent core
        self.lstm = DSConvLSTMCell(32, 48)

        # Spatial decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(48, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        B = x.shape[0]
        device = x.device

        h = torch.zeros(B, 48, 32, 32, device=device)
        c = torch.zeros(B, 48, 32, 32, device=device)

        # Encode input frames
        for t in range(self.in_frames):
            frame = x[:, t:t+1, :, :]
            enc = self.enc(frame)
            h, c = self.lstm(enc, h, c)

        # Generate predictions
        outputs = []
        for t in range(self.out_frames):
            pred = torch.sigmoid(self.dec(h))
            outputs.append(pred)
            enc = self.enc(pred)
            h, c = self.lstm(enc, h, c)

        return torch.cat(outputs, dim=1)
