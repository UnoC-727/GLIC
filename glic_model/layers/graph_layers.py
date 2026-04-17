"""
Graph neural network layers for GLIC model.
"""

import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from ..utils.graph_utils import compute_sobel_gradients, DropPath
from ..utils.graph_basics import cossim as cosine_similarity, local_sampling, global_sampling

# Backward compatibility
cossim = cosine_similarity


class GraphAggregator(nn.Module):
    """Graph Aggregator module for feature aggregation."""

    def __init__(self, dim, window_size, num_heads, bias=True,
                 unfold_dict=None, inner_dim=None):
        super().__init__()
        self.dim = dim
        self.group_size = window_size
        self.num_heads = num_heads
        self.inner_dim = inner_dim or dim

        # Graph-related parameters
        self.unfold_dict = unfold_dict
        self.sample_size = unfold_dict['kernel_size']
        self.graph_switch = True

        # Learnable logit scale
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )

        # Projection layers
        self.proj_group = nn.Linear(dim, self.inner_dim, bias=bias)
        self.proj_sample = nn.Linear(dim, self.inner_dim * 2, bias=bias)
        self.proj = nn.Linear(self.inner_dim, dim)

        # Relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        # Get relative coordinates table
        relative_coords_h = torch.arange(
            -(self.sample_size[0] - 1), self.group_size[0], dtype=torch.float32
        )
        relative_coords_w = torch.arange(
            -(self.sample_size[1] - 1), self.group_size[1], dtype=torch.float32
        )
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w])
        ).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        relative_coords_table[:, :, :, 0] /= (self.group_size[0] - 1)
        relative_coords_table[:, :, :, 1] /= (self.group_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        relative_position_index = self._get_rel_pos_index()
        self.register_buffer("relative_position_index", relative_position_index)

        self.relative_position_bias_table = None

    def _get_rel_pos_index(self):
        """Generate relative position index."""
        group_size = self.group_size
        sample_size = self.unfold_dict['kernel_size']

        coords_grid = torch.stack(torch.meshgrid([
            torch.arange(group_size[0]),
            torch.arange(group_size[1])
        ]))
        coords_grid_flatten = torch.flatten(coords_grid, 1)

        coords_sample = torch.stack(torch.meshgrid([
            torch.arange(sample_size[0]),
            torch.arange(sample_size[1])
        ]))
        coords_sample_flatten = torch.flatten(coords_sample, 1)

        relative_coords = coords_sample_flatten[:, None, :] - coords_grid_flatten[:, :, None]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += group_size[0] - sample_size[0] + 1
        relative_coords[:, :, 0] *= group_size[1] + sample_size[1] - 1
        relative_coords[:, :, 1] += group_size[1] - sample_size[1] + 1

        return relative_coords.sum(-1)

    def _rel_pos_bias(self):
        """Compute relative position bias."""
        if self.training and self.relative_position_bias_table is not None:
            self.relative_position_bias_table = None  # clear

        if not self.training and self.relative_position_bias_table is not None:
            relative_position_bias_table = self.relative_position_bias_table
        else:
            relative_position_bias_table = self.cpb_mlp(
                self.relative_coords_table).view(-1, self.num_heads)

        # Store for inference
        if not self.training and self.relative_position_bias_table is None:
            self.relative_position_bias_table = relative_position_bias_table

        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.group_size[0] * self.group_size[1],
            self.sample_size[0] * self.sample_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        return relative_position_bias.unsqueeze(0)

    def _get_correlation(self, x1, x2, graph):
        """Compute correlation between query and key features."""
        scale = torch.exp(torch.clamp(self.logit_scale, max=4.6052))

        if self.graph_switch:
            assert (x1.size(-2) == graph.size(-2)) and (x2.size(-2) == graph.size(-1))

        sim = cossim(x1, x2, graph=graph if self.graph_switch else None)
        sim = sim * scale + self._rel_pos_bias()
        sim = F.softmax(sim, dim=-1)
        return sim

    def forward(self, x_complete, graph=None, sampling_method=0):
        """Forward pass through grapher."""
        if sampling_method == 0:
            x = local_sampling(x_complete, group_size=self.group_size,
                             unfold_dict=None, output=0, tp='bhwc')
        else:
            x = global_sampling(x_complete, group_size=self.group_size,
                              sample_size=None, output=0, tp='bhwc')

        b_, n, c = x.shape
        c = self.inner_dim
        x1 = einops.rearrange(self.proj_group(x), 'b n (h c) -> b h n c',
                            b=b_, n=n, h=self.num_heads)

        if sampling_method == 0:
            x_sampled = local_sampling(self.proj_sample(x_complete),
                                     group_size=self.group_size,
                                     unfold_dict=self.unfold_dict,
                                     output=1, tp='bhwc')
        else:
            x_sampled = global_sampling(self.proj_sample(x_complete),
                                      group_size=self.group_size,
                                      sample_size=self.sample_size,
                                      output=1, tp='bhwc')

        x2, feat = einops.rearrange(x_sampled, 'b n (div h c) -> div b h n c',
                                  div=2, h=self.num_heads,
                                  c=c // self.num_heads)

        corr = self._get_correlation(x1, x2, graph)
        x = (corr @ feat).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)

        return x


class GraphDepthwiseFeedForward(nn.Module):
    """Graph Depthwise Feed-Forward Network."""

    def __init__(self, channels, expansion_factor):
        super(GraphDepthwiseFeedForward, self).__init__()
        self.hidden_features = channels
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3,
                            padding=1, groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x, x_size):
        """Forward pass through GDFN."""
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features,
                                 x_size[0], x_size[1]).contiguous()
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer."""

    def __init__(self, dim, num_heads, unfold_dict, window_size=7, sampling_method=0,
                 expansion_factor=4., bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, inner_dim=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.sampling_method = sampling_method

        self.norm1 = norm_layer(dim)
        self.grapher = GraphAggregator(
            dim, window_size=self._to_2tuple(self.window_size), num_heads=num_heads,
            bias=bias, unfold_dict=unfold_dict, inner_dim=inner_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = GraphDepthwiseFeedForward(dim, expansion_factor=expansion_factor)

    def _to_2tuple(self, x):
        """Convert to 2-tuple."""
        if isinstance(x, int):
            return (x, x)
        return x

    def forward(self, x, x_size, graph):
        """Forward pass through GAL."""
        H, W = x_size
        B, _, C = x.shape

        shortcut = x
        x = x.view(B, H, W, C)
        x = self.grapher(self.norm1(x),
                       graph=graph[0] if self.sampling_method == 0 else graph[1],
                       sampling_method=self.sampling_method)

        # Regroup
        if self.sampling_method:
            x = einops.rearrange(x, '(b numh numw) (sh sw) c -> b (sh numh sw numw) c',
                               numh=H // self.window_size,
                               numw=W // self.window_size,
                               sh=self.window_size, sw=self.window_size)
        else:
            x = einops.rearrange(x, '(b numh numw) (sh sw) c -> b (numh sh numw sw) c',
                               numh=H // self.window_size,
                               numw=W // self.window_size,
                               sh=self.window_size, sw=self.window_size)

        x = shortcut + x
        # FFN
        x = x + self.mlp(self.norm2(x), x_size)

        return x


# Backward compatibility aliases
IPGGrapher = GraphAggregator
GDFN = GraphDepthwiseFeedForward
GAL = GraphAttentionLayer