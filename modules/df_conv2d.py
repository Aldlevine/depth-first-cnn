from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from modules.df_module import DfModule
from utils.size import (
    Size2_p,
    Size2_t,
    add_size,
    div_size,
    mul_size,
    to_size,
)


def conv2d_b(
    inp: Tensor,
    weight: Tensor,
    dilation: Size2_p = 1,
    padding: Size2_p = 0,
    stride: Size2_p = 1,
) -> Tensor:
    """
    Shape:
        inp: (*, InC, H, W)
        weight: (OutC, InC, KH, KW)
    """
    assert weight.dim() == 4
    kernel_size = weight.shape[-2:]
    dilation = to_size(2, dilation)
    padding = to_size(2, padding)
    stride = to_size(2, stride)

    blocks = F.unfold(inp, kernel_size, dilation, padding, stride).transpose(-2, -1)
    weight = weight.view(weight.shape[0], -1).t()
    out = blocks @ weight

    h_out = (
        inp.shape[-2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
    ) // stride[0] + 1
    w_out = (
        inp.shape[-1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
    ) // stride[1] + 1
    out = out.transpose(-2, -1).view(inp.shape[0], weight.shape[1], h_out, w_out)
    return out


def conv2d_d(
    inp: Tensor,
    weight: Tensor,
    dilation: Size2_p = 1,
    padding: Size2_p = 0,
    stride: Size2_p = 1,
    pos: Optional[Size2_p] = None,
) -> Tensor:
    """
    Shape:
        inp: (*, InC, H, W)
        weight: (OutC, InC, KH, KW)
    """
    assert weight.dim() == 4
    kernel_size = weight.shape[-2:]
    dilation = to_size(2, dilation)
    padding = to_size(2, padding)
    stride = to_size(2, stride)

    blocks = F.unfold(inp, kernel_size, dilation, padding, stride).transpose(-2, -1)
    weight = weight.view(weight.shape[0], -1).t()

    h_out = (
        inp.shape[-2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
    ) // stride[0] + 1
    w_out = (
        inp.shape[-1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
    ) // stride[1] + 1

    if pos == None:
        out_stack: list[Tensor] = []
        for i in range(blocks.shape[-2]):
            i_blocks = blocks[..., i : i + 1, :]
            i_out = i_blocks @ weight
            out_stack.append(i_out)
        out = torch.cat(out_stack, dim=-2)

        out = out.transpose(-2, -1).view(inp.shape[0], weight.shape[1], h_out, w_out)
        return out
    else:
        pos = to_size(2, pos)
        i = pos[0] * w_out + pos[1]
        i_blocks = blocks[..., i : i + 1, :]
        i_out = i_blocks @ weight
        return i_out.transpose(-2, -1).view(inp.shape[0], weight.shape[1], 1, 1)


class DfConv2d(DfModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2_p,
        dilation: Size2_p = 1,
        mask: Optional[Tensor] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        elif len(kernel_size) == 1:
            kernel_size = kernel_size * 2
        elif len(kernel_size) != 2:
            raise ValueError(f"kernel_size incorrect: {kernel_size}")

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = to_size(2, kernel_size)
        self._dilation = to_size(2, dilation)
        self._device = device
        self._dtype = dtype

        if mask == None:
            mask = torch.ones(kernel_size)
        if mask != None:
            assert mask.shape == self._kernel_size
        self._mask: Tensor
        self.register_buffer("_mask", mask, persistent=False)

        self.weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels,
                *kernel_size,
                dtype=dtype,
                device=device,
            )
        )

        nn.init.kaiming_normal_(self.weight.data, nonlinearity="relu")

    @property
    def _dilated_kernel_size(self) -> Size2_t:
        kernel_size = self._kernel_size
        dilation = self._dilation
        dk = add_size(
            kernel_size, mul_size(add_size(kernel_size, -1), add_size(dilation, -1))
        )
        return dk

    def b_forward(self, img: Tensor) -> Tensor:
        self.weight.data *= self._mask
        padding = div_size(self._dilated_kernel_size, 2)
        return conv2d_b(img, self.weight, self._dilation, padding)

    def d_forward(self, img: Tensor, pos: Size2_t, clear_cache: bool) -> Tensor:
        if clear_cache or self._cache == None:
            self._cache = torch.zeros(
                img.shape[0],
                self._out_channels,
                *img.shape[2:],
                dtype=img.dtype,
                device=img.device,
            )
        self.weight.data *= self._mask
        padding = div_size(self._dilated_kernel_size, 2)
        px = conv2d_d(img, self.weight, self._dilation, padding, pos=pos)
        self._cache[..., pos[0], pos[1]] = px.view(self._cache.shape[:-2])
        return self._cache
