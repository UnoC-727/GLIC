"""
Graph neural network utilities for GLIC model.
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from .graph_basics import cossim, local_sampling, global_sampling


def gaussian_blur(x, k=5, sigma=1.0, mode='replicate'):
    """
    Apply Gaussian blur to input tensor.

    Args:
        x: Input tensor (B,H,W) or (B,C,H,W)
        k: Kernel size
        sigma: Standard deviation
        mode: Padding mode

    Returns:
        Blurred tensor with same shape as input
    """
    if x.dim() == 3:
        x = x.unsqueeze(1)  # (B,1,H,W)

    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    # 1D Gaussian kernel
    coords = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    g1 = torch.exp(-0.5 * (coords / sigma)**2)
    g1 = g1 / g1.sum()
    kx = g1.view(1, 1, 1, k).expand(C, 1, 1, k)   # horizontal kernel
    ky = g1.view(1, 1, k, 1).expand(C, 1, k, 1)   # vertical kernel

    pad = k // 2
    x = F.pad(x, (pad, pad, 0, 0), mode=mode)
    x = F.conv2d(x, kx, groups=C)
    x = F.pad(x, (0, 0, pad, pad), mode=mode)
    x = F.conv2d(x, ky, groups=C)
    return x.squeeze(1) if C == 1 else x


def compute_sobel_gradients(x, shape=(80, 80), scale=2, he=1):
    """
    Compute per-channel gradients using Sobel operators.

    Args:
        x: Input tensor (B, H*W, C)
        shape: Spatial shape (H, W)
        scale: Scale factor (unused in current implementation)
        he: Number of heads for averaging

    Returns:
        Gradient map (B, H, W)
    """
    B, _, C = x.shape
    H, W = shape

    # Reshape to spatial dimensions and average over heads
    x_rs = x.view(B, H, W, C // he, he).mean(-1).permute(0, 3, 1, 2)

    # Sobel kernels
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 4
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 4

    Cprime = x_rs.size(1)
    kx = sobel_x.repeat(Cprime, 1, 1, 1)  # (C′,1,3,3)
    ky = sobel_y.repeat(Cprime, 1, 1, 1)

    # Apply depthwise convolution
    x_padded = F.pad(x_rs, (1, 1, 1, 1), mode='replicate')
    gx = F.conv2d(x_padded, kx, padding=0, groups=Cprime)
    y_padded = F.pad(x_rs, (1, 1, 1, 1), mode='replicate')
    gy = F.conv2d(y_padded, ky, padding=0, groups=Cprime)

    # Compute gradient magnitude
    grad_mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-9)  # (B,C',H,W)
    grad_mag_s = gaussian_blur(grad_mag, k=3, sigma=0.8)  # Smooth per channel

    # RMS along channel dimension
    heat = torch.sqrt((grad_mag_s.pow(2)).mean(1))  # (B,H,W)
    return heat


class DropPath(nn.Module):
    """Stochastic depth (drop path) layer."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self._drop_path(x, self.drop_prob, self.training)

    def _drop_path(self, x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output