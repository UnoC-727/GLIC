"""
Graph Feature Aggregation (GFA) for GLIC model.
"""

import time
import math
import numpy as np
import torch
import torch.nn as nn
import einops

from ..utils.graph_utils import compute_sobel_gradients
from ..utils.graph_basics import cossim as cosine_similarity, local_sampling, global_sampling
from .graph_layers import GraphAttentionLayer

# Backward compatibility alias
cossim = cosine_similarity


class GraphLayerStack(nn.Module):
    """Stack of graph attention layers."""

    def __init__(self, dim, depth, num_heads, window_size, stages, unfold_dict,
                 mlp_ratio=4., bias=True, drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, inner_dim=None):
        super().__init__()
        self.dim = dim
        self.depth = depth

        blocks = []
        for i in range(depth):
            if stages[i] == 'GN':
                block = GraphAttentionLayer(
                    dim=dim, num_heads=num_heads, window_size=window_size,
                    sampling_method=0, expansion_factor=mlp_ratio,
                    bias=bias, drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer, unfold_dict=unfold_dict, inner_dim=inner_dim
                )
            elif stages[i] == 'GS':
                block = GraphAttentionLayer(
                    dim=dim, num_heads=num_heads, window_size=window_size,
                    sampling_method=1, expansion_factor=mlp_ratio,
                    bias=bias, drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer, unfold_dict=unfold_dict, inner_dim=inner_dim
                )
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, x_size, graph):
        """Forward pass through basic layer."""
        for blk in self.blocks:
            x = blk(x, x_size, graph)
        return x


class FeatureReshape(nn.Module):
    """Feature reshaping for processing feature maps."""

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Flatten and transpose input."""
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x


class FeatureRestore(nn.Module):
    """Feature restoration from flattened format."""

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        """Restore spatial dimensions."""
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x


class GFA(nn.Module):
    """Graph Feature Aggregation - core component of the graph neural network."""

    def __init__(self, dim, depth, num_heads, window_size, sample_size,
                 graph_flags, top_k, diff_scales, stages, mlp_ratio=4.,
                 bias=True, drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 inner_dim=None):
        super(GFA, self).__init__()

        self.dim = dim
        self.window_size = window_size
        self.sample_size = sample_size
        self.padding_size = (self.sample_size - self.window_size) // 2

        # Unfold dictionary for local sampling
        self.unfold_dict = dict(
            kernel_size=(self.sample_size, self.sample_size),
            stride=(window_size, window_size),
            padding=(self.padding_size, self.padding_size)
        )

        # Graph-related parameters
        self.num_head = num_heads
        self.graph_flag = graph_flags
        self.dist_type = 'cossim'
        self.fast_graph = True
        self.dist = cossim
        self.top_k = top_k
        self.flex_type = 'interdiff_plain'
        self.graph_switch = True
        self.diff_scale = diff_scales

        # Main residual group
        self.residual_group = GraphLayerStack(
            dim=dim, depth=depth, num_heads=num_heads, window_size=window_size,
            mlp_ratio=mlp_ratio, bias=bias, drop=drop, drop_path=drop_path,
            norm_layer=norm_layer, unfold_dict=self.unfold_dict,
            stages=stages, inner_dim=inner_dim
        )

        self.patch_embed = FeatureReshape(in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = FeatureRestore(in_chans=0, embed_dim=dim, norm_layer=None)

        # For fast graph construction
        self.tensors = None
        self.tolerance = 5

    @torch.no_grad()
    def _calc_graph(self, x_, x_size):
        """Calculate graph structure for feature aggregation."""
        if not self.graph_switch:
            return None, None

        # Initialize tensors for fast graph construction
        if self.fast_graph and self.tensors is None:
            self.tensors = (
                torch.tensor([
                    [0.5, 1., 0.],
                    [0., 0., 0.],
                    [0.5, 0., 1.],
                ], dtype=torch.float32).to(x_.device),
                torch.tensor([
                    [0.5, 0., 1.],
                    [0.5, 1., 0.],
                    [0., 0., 0.],
                ], dtype=torch.float32).to(x_.device)
            )

        # Compute feature differences for graph construction
        X_diff = [None, None]
        if self.flex_type.startswith('interdiff'):
            X_diff = compute_sobel_gradients(x_, shape=x_size)

            if (self.diff_scale is not None) and (self.diff_scale != 0):
                # Affine transform for scaling
                mu = X_diff.mean(dim=(-2, -1), keepdim=True)
                X_diff = mu + (X_diff - mu) / self.diff_scale

            X_diff = [
                einops.rearrange(X_diff, 'b (numh wh) (numw ww)-> (b numh numw) (wh ww)',
                               wh=self.window_size, ww=self.window_size),
                einops.rearrange(X_diff, 'b (sh numh) (sw numw) -> (b numh numw) (sh sw)',
                               sh=self.window_size, sw=self.window_size)
            ]

        graph0 = self._calc_graph_(x_, x_size, sampling_method=0, X_diff=X_diff[0])
        graph1 = self._calc_graph_(x_, x_size, sampling_method=1, X_diff=X_diff[1])
        return (graph0, graph1)

    @torch.no_grad()
    def _calc_graph_(self, x_, x_size, sampling_method=0, X_diff=None):
        """Calculate graph for specific sampling method."""
        he = self.num_head
        x = einops.rearrange(x_, 'b (h w) c -> b c h w', h=x_size[0], w=x_size[1])

        # Sample features based on method
        if sampling_method:  # sparse global
            X_sample, Y_sample = global_sampling(
                x, group_size=self.window_size, sample_size=self.sample_size,
                output=2, tp='bchw'
            )
        else:  # dense local
            X_sample, Y_sample = local_sampling(
                x, group_size=self.window_size, unfold_dict=self.unfold_dict,
                output=2, tp='bchw'
            )

        assert X_sample.size(0) == Y_sample.size(0)

        # Compute similarity matrix
        D = self.dist(X_sample.unsqueeze(1), Y_sample.unsqueeze(1)).squeeze(1)

        if self.fast_graph:
            # Fast graph construction using adaptive thresholding
            if sampling_method == 1:
                cur_top_k = self.top_k
            else:
                cur_top_k = self.top_k

            maskarray = (X_diff / X_diff.sum(dim=-1, keepdim=True)) * D.size(1) * cur_top_k
            maskarray = torch.clamp(maskarray, 1, D.size(-1))

            # Binary search for threshold
            minbound = torch.min(D, dim=-1, keepdim=True)[0]
            maxbound = torch.ones_like(minbound)
            wall = D.mean(dim=-1, keepdim=True)
            MAT = torch.cat([wall, minbound, maxbound], dim=-1)

            for _ in range(self.tolerance):
                allocated = (D > MAT[..., 0:1]).sum(dim=-1)
                MAT = torch.where(
                    (allocated > maskarray).unsqueeze(-1),
                    MAT @ self.tensors[0],
                    MAT @ self.tensors[1],
                )

            graph = (D > MAT[..., 0:1]).unsqueeze(1)  # add head dim

        return graph

    def forward(self, x, x_size, prev_graph=None):
        """Forward pass through MGB."""
        x = x.flatten(2).transpose(1, 2)
        graph = self._calc_graph(x, x_size) if self.graph_flag else prev_graph
        x = self.residual_group(x, x_size, graph)
        return self.patch_unembed(x, x_size)