from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from modules.df_conv2d import DfConv2d
from modules.df_module import DfModule


def stable_softmax(x: Tensor, dim: int = -1) -> Tensor:
    if x.numel() == 0:
        return torch.empty_like(x)
    x = x - x.max()
    return torch.log_softmax(x, dim)


class DfAttention(DfModule):
    def __init__(
        self,
        channels: int,
        key_dim: Optional[int] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self._channels = channels
        if key_dim == None:
            key_dim = channels // 8
        self._key_dim = key_dim

        self._conv = DfConv2d(
            channels, 2 * key_dim + channels, 1, device=device, dtype=dtype
        )
        self._gamma = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))

    def qet_qkv(
        self, img: Tensor, pos: Optional[tuple[int, int]], clear_cache: bool
    ) -> tuple[torch.Size, Tensor, Tensor, Tensor]:
        shape = img.shape
        qkv = self._conv(img, pos=pos, clear_cache=clear_cache)
        q, k, v = (
            t.view(*t.shape[:-2], -1)
            for t in qkv.split_with_sizes([self._key_dim] * 2 + [self._channels], -3)
        )

        return shape, q, k, v

    def b_forward(self, img: Tensor) -> Tensor:
        shape, q, k, v = self.qet_qkv(img, None, False)
        attn = stable_softmax(torch.bmm(q.transpose(-2, -1), k))
        attn.triu(1)
        out = torch.bmm(v, attn).view(*shape).contiguous()
        return out + img

    def d_forward(self, img: Tensor, pos: tuple[int, int], clear_cache: bool) -> Tensor:
        if self._cache == None or pos == (0, 0) or clear_cache:
            self._cache = torch.zeros_like(img)

        shape, q, k, v = self.qet_qkv(img, pos, clear_cache)
        i = (pos[0] * shape[-1]) + pos[1]
        attn = stable_softmax(
            torch.bmm(q[..., :i].transpose(-2, -1), k[..., i : i + 1])
        )
        px = torch.bmm(v[..., :i], attn)
        self._cache[..., pos[0], pos[1]] = img[..., pos[0], pos[1]] + px.squeeze()
        return self._cache
