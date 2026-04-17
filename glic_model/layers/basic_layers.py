"""
Basic neural network layers for GLIC model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    """Custom LayerNorm function for 2D tensors."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (gx,
                (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
                grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
                None)


class LayerNorm2d(nn.Module):
    """2D Layer Normalization."""

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class GatedFFN(nn.Module):
    """Gated Feed-Forward Network."""

    def __init__(self, channels, expansion_factor=4):
        super(GatedFFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.project_in(x).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class DepthwiseConv(nn.Module):
    """Depthwise convolution with kernel size 5."""

    def __init__(self, in_ch, out_ch, slope=0.01, inplace=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )
        self.depth_conv = nn.Conv2d(in_ch, in_ch, 5, padding=2, groups=in_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 1)

        self.adaptor = None
        if in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)

        return out + identity


class GatedTransformCNN(nn.Module):
    """Gated transform CNN block."""

    def __init__(self, dim, dim_out, expansion_factor=4, **layer_kwargs):
        super().__init__()
        self.norm2 = LayerNorm2d(dim_out)
        self.mixer = DepthwiseConv(dim, dim_out)
        self.mlp = GatedFFN(dim_out, expansion_factor=expansion_factor)

    def forward(self, x):
        x = self.mixer(x)
        x = x + self.mlp(self.norm2(x))
        return x


class OLP(nn.Module):
    """Orthogonal Linear Projection with regularization."""

    def __init__(self, in_features: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_dim, bias=bias)
        self.in_dim = in_features
        self.out_dim = out_dim

        eye_dim = min(in_features, out_dim)
        # Match original implementation
        self.identity_matrix = torch.eye(eye_dim)

    def loss(self):
        """Compute orthogonality regularization loss."""
        kernel_matrix = self.linear.weight
        if self.in_dim > self.out_dim:
            gram_matrix = torch.mm(kernel_matrix, kernel_matrix.t())
        else:
            gram_matrix = torch.mm(kernel_matrix.t(), kernel_matrix)
        loss_ortho = F.mse_loss(gram_matrix, self.identity_matrix.to(gram_matrix.device))
        return loss_ortho

    def forward(self, x):
        return self.linear(x)