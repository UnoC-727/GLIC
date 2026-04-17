"""
GLIC encoder and decoder networks.
"""

import torch
import torch.nn as nn
from compressai.layers import subpel_conv3x3

from ..layers.basic_layers import GatedTransformCNN
from ..layers.wavelet_layers import WLS, iWLS
from ..layers.graph_feature_aggregation import GFA


class GLICAnalysisTransform(nn.Module):
    """GLIC Analysis Transform (Encoder)."""

    def __init__(self, N=192, M=320):
        super(GLICAnalysisTransform, self).__init__()
        embed_dim0 = 128
        embed_dim1 = 192
        embed_dim2 = 192

        # Auxiliary transform branch
        self.AuxT_enc = nn.Sequential(
            WLS(3, embed_dim0),
            WLS(embed_dim0, embed_dim1),
            WLS(embed_dim1, embed_dim2),
            WLS(embed_dim2, M),
        )

        # Main transform branch - first stage
        self.g1 = nn.Sequential(
            GatedTransformCNN(embed_dim0, embed_dim0),
            GatedTransformCNN(embed_dim0, embed_dim0),
            GatedTransformCNN(embed_dim0, embed_dim0),
        )

        # Main transform branch - graph blocks
        self.g2 = GFA(
            dim=embed_dim1,
            depth=5,
            num_heads=8,
            window_size=8,
            sample_size=16,
            graph_flags=True,
            top_k=64,
            diff_scales=1.5,
            stages=['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
        )

        self.g3 = GFA(
            dim=embed_dim2,
            depth=5,
            num_heads=8,
            window_size=8,
            sample_size=16,
            graph_flags=True,
            top_k=64,
            diff_scales=1.5,
            stages=['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
            mlp_ratio=2,
        )

        # Downsampling layers
        self.down0 = nn.Conv2d(3, embed_dim0, 3, stride=2, padding=1)
        self.down1 = nn.Conv2d(embed_dim0, embed_dim1, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(embed_dim1, embed_dim2, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(embed_dim2, M, 3, stride=2, padding=1)

    def forward(self, x):
        """Forward pass through encoder."""
        aux_x = x

        # Stage 0
        x = self.down0(x)
        x = self.g1(x)
        aux_x = self.AuxT_enc[0](aux_x)
        x += aux_x

        # Stage 1
        x = self.down1(x)
        B, C, H, W = x.shape
        x = self.g2(x, (H, W))
        aux_x = self.AuxT_enc[1](aux_x)
        x += aux_x

        # Stage 2
        x = self.down2(x)
        B, C, H, W = x.shape
        x = self.g3(x, (H, W))
        aux_x = self.AuxT_enc[2](aux_x)
        x += aux_x

        # Final stage
        x = self.down3(x)
        aux_x = self.AuxT_enc[3](aux_x)
        x += aux_x

        return x

    def forward_energy(self, x):
        """Forward pass with intermediate energy outputs."""
        aux_x = x
        energies = []

        # Stage 0
        x = self.down0(x)
        x = self.g1(x)
        aux_x = self.AuxT_enc[0](aux_x)
        x += aux_x
        energies.append(x)

        # Stage 1
        x = self.down1(x)
        B, C, H, W = x.shape
        x = self.g2(x, (H, W))
        aux_x = self.AuxT_enc[1](aux_x)
        x += aux_x
        energies.append(x)

        # Stage 2
        x = self.down2(x)
        B, C, H, W = x.shape
        x = self.g3(x, (H, W))
        aux_x = self.AuxT_enc[2](aux_x)
        x += aux_x
        energies.append(x)

        # Final stage
        x = self.down3(x)
        aux_x = self.AuxT_enc[3](aux_x)
        x += aux_x

        return x, *energies


class GLICSynthesisTransform(nn.Module):
    """GLIC Synthesis Transform (Decoder)."""

    def __init__(self, N=192, M=320):
        super(GLICSynthesisTransform, self).__init__()
        embed_dim1 = 128
        embed_dim2 = 192
        embed_dim3 = 192

        # Auxiliary transform branch
        self.AuxT_dec = nn.Sequential(
            iWLS(M, embed_dim3),
            iWLS(embed_dim3, embed_dim2),
            iWLS(embed_dim2, embed_dim1),
            iWLS(embed_dim1, 3),
        )

        # Main transform branch - graph blocks
        self.g1 = GFA(
            dim=embed_dim3,
            depth=5,
            num_heads=8,
            window_size=8,
            sample_size=16,
            graph_flags=True,
            top_k=64,
            diff_scales=1.5,
            stages=['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
            mlp_ratio=2,
        )

        self.g2 = GFA(
            dim=embed_dim2,
            depth=5,
            num_heads=8,
            window_size=8,
            sample_size=16,
            graph_flags=True,
            top_k=64,
            diff_scales=1.5,
            stages=['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
        )

        # Main transform branch - final stage
        self.g3 = nn.Sequential(
            GatedTransformCNN(embed_dim1, embed_dim1),
            GatedTransformCNN(embed_dim1, embed_dim1),
            GatedTransformCNN(embed_dim1, embed_dim1),
        )

        # Upsampling layers
        self.up0 = subpel_conv3x3(M, embed_dim3, 2)
        self.up1 = subpel_conv3x3(embed_dim3, embed_dim2, 2)
        self.up2 = subpel_conv3x3(embed_dim2, embed_dim1, 2)
        self.up3 = subpel_conv3x3(embed_dim1, 3, 2)

    def forward(self, x):
        """Forward pass through decoder."""
        aux_x = x

        # Stage 0
        x = self.up0(x)
        B, C, H, W = x.shape
        x = self.g1(x, (H, W))
        aux_x = self.AuxT_dec[0](aux_x)
        x += aux_x

        # Stage 1
        x = self.up1(x)
        B, C, H, W = x.shape
        x = self.g2(x, (H, W))
        aux_x = self.AuxT_dec[1](aux_x)
        x += aux_x

        # Stage 2
        x = self.up2(x)
        x = self.g3(x)
        aux_x = self.AuxT_dec[2](aux_x)
        x += aux_x

        # Final stage
        x = self.up3(x)
        aux_x = self.AuxT_dec[3](aux_x)
        x += aux_x

        return x


# Backward compatibility aliases
GLICEncoder = GLICAnalysisTransform
GLICDecoder = GLICSynthesisTransform