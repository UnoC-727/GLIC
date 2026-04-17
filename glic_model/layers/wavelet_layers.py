"""
Wavelet Linear Scaling layers for GLIC model.
"""

import torch
import torch.nn as nn
from ..utils.wavelet import DWT_2D, IDWT_2D
from .basic_layers import OLP


class WLS(nn.Module):
    """Wavelet Linear Scaling (analysis side)."""

    def __init__(self, in_dim: int, out_dim: int):
        super(WLS, self).__init__()
        self.dwt = DWT_2D(wave='haar')
        self.OLP = OLP(in_dim * 4, out_dim)
        self.scaling_factors = nn.Parameter(torch.cat((torch.zeros(1, 1, in_dim) + 0.5,
                                                       torch.zeros(1, 1, in_dim) + 0.5,
                                                       torch.zeros(1, 1, in_dim) + 0.5,
                                                       torch.zeros(1, 1, in_dim)), dim=2))

    def forward(self, x):
        """Forward pass through WLS."""
        x = self.dwt(x)
        b, _, h, w = x.shape
        x = x.view(b, -1, h * w).permute(0, 2, 1)  # (B, HW, 4C)
        x = x * torch.exp(self.scaling_factors)
        x = self.OLP(x)
        return x.view(b, h, w, -1).permute(0, 3, 1, 2)


class iWLS(nn.Module):
    """Inverse Wavelet Linear Scaling (synthesis side)."""

    def __init__(self, in_dim: int, out_dim: int):
        super(iWLS, self).__init__()
        self.idwt = IDWT_2D(wave='haar')
        self.OLP = OLP(in_dim, out_dim * 4)
        self.scaling_factors = nn.Parameter(torch.cat((torch.zeros(1, 1, out_dim) + 0.5,
                                                       torch.zeros(1, 1, out_dim) + 0.5,
                                                       torch.zeros(1, 1, out_dim) + 0.5,
                                                       torch.zeros(1, 1, out_dim)), dim=2))

    def forward(self, x):
        """Forward pass through inverse WLS."""
        b, _, h, w = x.shape
        x = x.view(b, -1, h * w).permute(0, 2, 1)  # (B, HW, C)
        x = self.OLP(x)
        x = x / torch.exp(self.scaling_factors)
        x = x.view(b, h, w, -1).permute(0, 3, 1, 2)
        x = self.idwt(x)

        return x