"""
Wavelet transform utilities for GLIC model.
Based on PyTorch implementations of DWT and IDWT.
"""

import time
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class DWT_Function(Function):
    """Discrete Wavelet Transform Function."""

    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        padding = w_ll.shape[-1] // 4

        x_ll = F.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim, padding=padding)
        x_lh = F.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim, padding=padding)
        x_hl = F.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim, padding=padding)
        x_hh = F.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim, padding=padding)

        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H//2, W//2)
            dx = dx.transpose(1, 2).reshape(B, -1, H//2, W//2)

            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            padding = w_ll.shape[-1] // 4
            dx = F.conv_transpose2d(dx, filters, stride=2, groups=C, padding=padding)

        return dx, None, None, None, None


class IDWT_Function(Function):
    """Inverse Discrete Wavelet Transform Function."""

    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = F.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = F.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = F.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = F.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = F.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class DWT_2D(nn.Module):
    """2D Discrete Wavelet Transform."""

    def __init__(self, wave='haar'):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0).to(torch.float32))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0).to(torch.float32))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0).to(torch.float32))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0).to(torch.float32))

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


class IDWT_2D(nn.Module):
    """2D Inverse Discrete Wavelet Transform."""

    def __init__(self, wave='haar'):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)

        self.register_buffer('filters', filters.to(torch.float32))

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)