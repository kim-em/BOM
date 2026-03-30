"""
Optical flow based nowcasting.

Architecture:
1. FlowEstimator: small U-Net that estimates optical flow between consecutive frames
2. Warp: use estimated flow to advect the last frame forward
3. Refinement: small CNN to correct warping artifacts

This is physically motivated — weather systems move, and the dominant
prediction mechanism should be advection (translating rain patterns).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowEstimator(nn.Module):
    """
    Estimate optical flow between two frames.
    Input: (B, 2, H, W) — two consecutive frames stacked
    Output: (B, 2, H, W) — (dx, dy) flow field
    """

    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # Decoder with skip connections
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # Flow output (2 channels: dx, dy)
        self.flow_out = nn.Conv2d(16, 2, 3, padding=1)
        # Initialize flow to near-zero
        nn.init.zeros_(self.flow_out.weight)
        nn.init.zeros_(self.flow_out.bias)

    def forward(self, x):
        e1 = self.enc1(x)   # (B, 16, 64, 64)
        e2 = self.enc2(e1)  # (B, 32, 32, 32)
        e3 = self.enc3(e2)  # (B, 32, 16, 16)

        d3 = self.dec3(e3)                      # (B, 32, 32, 32)
        d2 = self.dec2(torch.cat([d3, e2], 1))  # (B, 16, 64, 64)
        d1 = self.dec1(torch.cat([d2, e1], 1))  # (B, 16, 128, 128)

        return self.flow_out(d1)  # (B, 2, 128, 128)


def warp(frame, flow):
    """
    Warp a frame using optical flow via grid_sample.
    frame: (B, 1, H, W)
    flow: (B, 2, H, W) — pixel displacements (dx, dy)
    Returns: warped frame (B, 1, H, W)
    """
    B, _, H, W = frame.shape
    # Create base grid
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=frame.device),
        torch.linspace(-1, 1, W, device=frame.device),
        indexing='ij'
    )
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    # Normalise flow to [-1, 1] range (flow is in pixels, grid is [-1, 1])
    flow_norm = torch.stack([
        flow[:, 0] * 2.0 / W,
        flow[:, 1] * 2.0 / H,
    ], dim=-1)

    grid = grid + flow_norm
    return F.grid_sample(frame, grid, mode='bilinear', padding_mode='border', align_corners=True)


class Refiner(nn.Module):
    """Small CNN to refine warped predictions (fix artifacts, intensity changes)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),  # warped + last_input
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, warped, last_input):
        x = torch.cat([warped, last_input], dim=1)
        return torch.sigmoid(warped + self.net(x))  # residual


class FlowNowcaster(nn.Module):
    """
    Full optical flow nowcasting model.
    Input: (B, 6, 128, 128) — 6 input frames
    Output: (B, 6, 128, 128) — 6 predicted frames

    Strategy:
    1. Estimate flow from each consecutive pair in the input
    2. Average flows weighted by recency (recent flows matter more)
    3. Extrapolate: warp last frame by 1x flow, 2x flow, ... 6x flow
    4. Refine each warped prediction
    """

    def __init__(self, in_frames=6, out_frames=6):
        super().__init__()
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.flow_net = FlowEstimator()
        self.refiner = Refiner()

    def forward(self, x):
        B = x.shape[0]

        # Estimate flow from consecutive pairs
        flows = []
        for t in range(self.in_frames - 1):
            pair = torch.stack([x[:, t], x[:, t+1]], dim=1)  # (B, 2, H, W)
            flow = self.flow_net(pair)  # (B, 2, H, W)
            flows.append(flow)

        # Weighted average of flows (exponentially more weight to recent)
        weights = torch.tensor([2**i for i in range(len(flows))],
                               dtype=torch.float32, device=x.device)
        weights = weights / weights.sum()

        avg_flow = torch.zeros_like(flows[0])
        for w, flow in zip(weights, flows):
            avg_flow = avg_flow + w * flow

        # Generate predictions by extrapolating flow
        last_frame = x[:, -1:, :, :]  # (B, 1, H, W)
        outputs = []

        for t in range(1, self.out_frames + 1):
            # Extrapolate: multiply flow by step count
            warped = warp(last_frame, avg_flow * t)
            refined = self.refiner(warped, last_frame)
            outputs.append(refined)

        return torch.cat(outputs, dim=1)  # (B, 6, H, W)
