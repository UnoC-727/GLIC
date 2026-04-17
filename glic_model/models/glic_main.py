"""
Main GLIC model class
"""

import torch
import torch.nn as nn
from typing import cast
from torch import Tensor

from compressai.models.sensetime import Elic2022Official
from compressai.layers import (
    CheckerboardMaskedConv2d,
    conv1x1,
    conv3x3,
    sequential_channel_ramp,
    subpel_conv3x3,
)
from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    ChannelGroupsLatentCodec,
    CheckerboardLatentCodec,
    GaussianConditionalLatentCodec,
    HyperLatentCodec,
    HyperpriorLatentCodec,
)
from compressai.models.utils import conv

from .encoder_decoder import GLICAnalysisTransform, GLICSynthesisTransform
from ..layers.basic_layers import GatedTransformCNN, OLP


class ParameterAggregationBlock(nn.Module):
    """Parameter aggregation block with gating."""

    def __init__(self, dim, dim_out, expansion_factor=4, **layer_kwargs):
        super().__init__()
        from ..layers.basic_layers import LayerNorm2d, GatedFFN

        self.norm2 = LayerNorm2d(dim_out)
        self.mixer = nn.Conv2d(dim, dim_out, kernel_size=1, stride=1)
        self.mlp = GatedFFN(dim_out, expansion_factor=expansion_factor)

    def forward(self, x):
        x = self.mixer(x)
        x = x + self.mlp(self.norm2(x))
        return x


class GLICModel(Elic2022Official):
    """
    GLIC Model - Graph-based Learned Image Compression with Auxiliary Transforms.

    """

    def __init__(self, N=192, M=320, groups=None, **kwargs):
        super().__init__(**kwargs)

        self.g_a = GLICAnalysisTransform(N, M)
        self.g_s = GLICSynthesisTransform(N, M)

        # Hyperprior analysis and synthesis
        h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1),
            GatedTransformCNN(N, N, expansion_factor=2),
            GatedTransformCNN(N, N, expansion_factor=2),
            conv(N, N, kernel_size=3, stride=2),
            GatedTransformCNN(N, N, expansion_factor=2),
            GatedTransformCNN(N, N, expansion_factor=2),
            conv(N, N, kernel_size=3, stride=2),
        )

        h_s = nn.Sequential(
            subpel_conv3x3(N, N, 2),
            GatedTransformCNN(N, N, expansion_factor=2),
            GatedTransformCNN(N, N, expansion_factor=2),
            subpel_conv3x3(N, N, 2),
            GatedTransformCNN(N, N, expansion_factor=2),
            GatedTransformCNN(N, N, expansion_factor=2),
            conv(N, N * 2, kernel_size=3, stride=1),
        )

        # Channel context models
        channel_context = {
            f"y{k}": sequential_channel_ramp(
                sum(self.groups[:k]),
                self.groups[k] * 2,
                min_ch=N,
                num_layers=3,
                make_layer=GatedTransformCNN,
                make_act=lambda: nn.Identity(),
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(1, len(self.groups))
        }

        # Spatial context models
        spatial_context = [
            CheckerboardMaskedConv2d(
                self.groups[k],
                self.groups[k] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(len(self.groups))
        ]

        # Parameter aggregation models
        param_aggregation = [
            sequential_channel_ramp(
                self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + N * 2,
                self.groups[k] * 2,
                min_ch=N * 2,
                num_layers=3,
                make_layer=ParameterAggregationBlock,
                make_act=lambda: nn.Identity(),
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for k in range(len(self.groups))
        ]

        # Space-channel context model (SCCTX)
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(quantizer="ste"),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        # Hyperprior latent codec
        self.latent_codec = HyperpriorLatentCodec(
            latent_codec={
                "y": ChannelGroupsLatentCodec(
                    groups=self.groups,
                    channel_context=channel_context,
                    latent_codec=scctx_latent_codec,
                ),
                "hyper": HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_a,
                    h_s=h_s,
                    quantizer="ste",
                ),
            },
        )

    def ortho_loss(self) -> Tensor:
        """Compute orthogonality regularization loss."""
        loss = sum(m.loss() for m in self.modules() if isinstance(m, OLP))
        return cast(Tensor, loss)

    def forward(self, x):
        """Forward pass through the complete model."""
        y = self.g_a(x)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": y_out["likelihoods"],
        }

    def energy(self, x):
        """Forward pass with energy computation for analysis."""
        return self.g_a.forward_energy(x)


# Backward compatibility aliases
ParamGated = ParameterAggregationBlock


# Alias for backward compatibility
def graph_compression_AuxT():
    """Factory function to create GLIC model."""
    return GLICModel()