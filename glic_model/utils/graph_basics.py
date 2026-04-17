"""
Basic graph operations for GLIC model.
"""

import torch
import torch.nn.functional as F
import einops


def cossim(X_sample, Y_sample, graph=None):
    """
    Compute cosine similarity between sample tensors.

    Args:
        X_sample: Query samples (a, b, m, c)
        Y_sample: Key samples (a, b, n, c)
        graph: Optional graph mask

    Returns:
        Similarity matrix (a, b, m, n)
    """
    sim = torch.einsum('a b m c, a b n c -> a b m n',
                       F.normalize(X_sample, dim=-1),
                       F.normalize(Y_sample, dim=-1))

    if graph is not None:
        sim = sim + (-100.) * (~graph)

    return sim


def local_sampling(x, group_size, unfold_dict, output=0, tp='bhwc'):
    """
    Perform local sampling within windows.

    Args:
        x: Input tensor
        group_size: Size of groups (window size)
        unfold_dict: Dictionary with unfold parameters
        output: 0=grouped, 1=sampled, 2=both
        tp: Tensor format ('bhwc' or 'bchw')

    Returns:
        Sampled tensor(s) based on output parameter
    """
    if isinstance(group_size, int):
        group_size = (group_size, group_size)

    if output != 1:
        if tp == 'bhwc':
            x_grouped = einops.rearrange(
                x, 'b (numh sh) (numw sw) c-> (b numh numw) (sh sw) c',
                sh=group_size[0], sw=group_size[1]
            )
        elif tp == 'bchw':
            x_grouped = einops.rearrange(
                x, 'b c (numh sh) (numw sw)-> (b numh numw) (sh sw) c',
                sh=group_size[0], sw=group_size[1]
            )

        if output == 0:
            return x_grouped

    if tp == 'bhwc':
        x = einops.rearrange(x, 'b h w c -> b c h w')

    x_sampled = einops.rearrange(
        F.unfold(x, **unfold_dict),
        'b (c k0 k1) l -> (b l) (k0 k1) c',
        k0=unfold_dict['kernel_size'][0],
        k1=unfold_dict['kernel_size'][1]
    )

    if output == 1:
        return x_sampled

    assert x_grouped.size(0) == x_sampled.size(0)
    return x_grouped, x_sampled


def global_sampling(x, group_size, sample_size, output=0, tp='bhwc'):
    """
    Perform global sampling across the feature map.

    Args:
        x: Input tensor
        group_size: Size of groups
        sample_size: Size of sampling area
        output: 0=grouped, 1=sampled, 2=both
        tp: Tensor format ('bhwc' or 'bchw')

    Returns:
        Sampled tensor(s) based on output parameter
    """
    if isinstance(group_size, int):
        group_size = (group_size, group_size)
    if isinstance(sample_size, int):
        sample_size = (sample_size, sample_size)

    if output != 1:
        if tp == 'bchw':
            x_grouped = einops.rearrange(
                x, 'b c (sh numh) (sw numw) -> (b numh numw) (sh sw) c',
                sh=group_size[0], sw=group_size[1]
            )
        elif tp == 'bhwc':
            x_grouped = einops.rearrange(
                x, 'b (sh numh) (sw numw) c -> (b numh numw) (sh sw) c',
                sh=group_size[0], sw=group_size[1]
            )

        if output == 0:
            return x_grouped

    if tp == 'bchw':
        x_sampled = einops.rearrange(
            x, 'b c (sh extrah numh) (sw extraw numw) -> b extrah numh extraw numw c sh sw',
            sh=sample_size[0], sw=sample_size[1], extrah=1, extraw=1
        )
    elif tp == 'bhwc':
        x_sampled = einops.rearrange(
            x, 'b (sh extrah numh) (sw extraw numw) c -> b extrah numh extraw numw c sh sw',
            sh=sample_size[0], sw=sample_size[1], extrah=1, extraw=1
        )

    b_y, _, numh, _, numw, c_y, sh_y, sw_y = x_sampled.shape
    ratio_h, ratio_w = sample_size[0] // group_size[0], sample_size[1] // group_size[1]

    x_sampled = x_sampled.expand(
        b_y, ratio_h, numh, ratio_w, numw, c_y, sh_y, sw_y
    ).reshape(-1, c_y, sh_y * sw_y).permute(0, 2, 1)

    if output == 1:
        return x_sampled

    assert x_grouped.size(0) == x_sampled.size(0)
    return x_grouped, x_sampled