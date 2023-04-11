from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from modules.df_conv2d import DfConv2d, conv2d_b, conv2d_d
from modules.df_elemwise import DfElemwise
from modules.df_module import DfModule
from utils.functions import stable_softmax
from utils.size import Size2_t


class DfAttention(DfModule):
    def __init__(
        self,
        in_channels: int,
        k_size: int,
        v_size: int,
    ) -> None:
        super().__init__()

        self._in_channels = in_channels
        self._k_size = k_size
        self._v_size = v_size

        self._weight = nn.Parameter(torch.randn(2 * k_size + v_size, in_channels, 1, 1))
        self._qkv: Optional[Tensor] = None

    @lru_cache
    def get_mask(
        self, shape: Size2_t, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        mask = torch.ones(shape, device=device, dtype=dtype).tril(0)
        return mask

    def b_forward(self, inp: Tensor) -> Tensor:
        *b, c, h, w = inp.shape
        qkv = conv2d_b(inp, self._weight)
        q, k, v = (
            t.view(*b, -1, h * w)
            for t in torch.split_with_sizes(
                qkv, [self._k_size, self._k_size, self._v_size], -3
            )
        )

        attn = q.transpose(-2, -1) @ k
        mask = self.get_mask(attn.shape[-2:], device=attn.device, dtype=attn.dtype)
        # attn = attn + mask.where(mask == 0, -torch.inf)
        attn = attn + mask.where(mask == 0, -9e15)
        # attn = stable_softmax(attn, -2).nan_to_num(0, 0, 0)
        attn = attn.softmax(-2).nan_to_num(0, 0, 0)
        attn = attn.triu(1)

        out = v @ attn
        # return F.normalize(out.view(*b, self._v_size, h, w), dim=-3)
        return out.view(*b, self._v_size, h, w)

    def d_forward(self, inp: Tensor, pos: Size2_t, clear_cache: bool) -> Tensor:
        *b, c, h, w = inp.shape

        if self._cache == None or clear_cache:
            self._cache = torch.zeros(
                *b, self._v_size, h, w, dtype=inp.dtype, device=inp.device
            )

        if self._qkv == None or clear_cache:
            qkv_size = 2 * self._k_size + self._v_size
            self._qkv = torch.zeros(
                *b, qkv_size, h, w, dtype=inp.dtype, device=inp.device
            )

        y, x = pos
        self._qkv[..., y, x] = conv2d_d(inp, self._weight, pos=(y, x)).squeeze()

        q, k, v = (
            t.view(*b, -1, h * w)
            for t in torch.split_with_sizes(
                self._qkv, [self._k_size, self._k_size, self._v_size], -3
            )
        )

        i = y * w + x
        q = q[..., :i]
        k = k[..., i : i + 1]
        v = v[..., :i]

        attn = q.transpose(-2, -1) @ k
        # attn = stable_softmax(attn, -2).nan_to_num(0, 0, 0)
        attn = attn.softmax(-2).nan_to_num(0, 0, 0)
        # self._cache[..., y, x] = F.normalize((v @ attn).squeeze(), dim=-1)
        self._cache[..., y, x] = (v @ attn).squeeze()
        return self._cache


# def stable_softmax(x: Tensor, dim: int = -1) -> Tensor:
#     if x.numel() == 0:
#         return torch.empty_like(x)
#     x = x - x.max()
#     # return torch.log_softmax(x, dim)
#     return torch.softmax(x, dim)


# class DfAttention(DfModule):
#     def __init__(
#         self,
#         channels: int,
#         key_dim: Optional[int] = None,
#         device: Optional[torch.device | str] = None,
#         dtype: Optional[torch.dtype] = None,
#     ) -> None:
#         super().__init__()
#         self._channels = channels
#         if key_dim == None:
#             key_dim = channels // 8
#         self._key_dim = key_dim

#         self._conv = DfConv2d(
#             channels, 2 * key_dim + channels, 1, device=device, dtype=dtype
#         )
#         # self._conv = DfElemwise(nn.Conv2d(channels, 2 * key_dim + channels, 1, bias=False, device=device, dtype=dtype))
#         self._gamma = nn.Parameter(torch.ones(1, device=device, dtype=dtype))

#     def qet_qkv(
#         self, img: Tensor, pos: Optional[tuple[int, int]], clear_cache: bool
#     ) -> tuple[torch.Size, Tensor, Tensor, Tensor]:
#         shape = img.shape
#         qkv = self._conv(img, pos=pos, clear_cache=clear_cache)
#         q, k, v = (
#             t.view(*t.shape[:-2], -1)
#             for t in qkv.split_with_sizes([self._key_dim] * 2 + [self._channels], -3)
#         )

#         return shape, q, k, v

#     def b_forward(self, img: Tensor) -> Tensor:
#         shape, q, k, v = self.qet_qkv(img, None, False)

#         attn = torch.bmm(q.transpose(-2, -1), k)
#         mask = torch.ones_like(attn).triu(1)
#         mask = mask.where(mask == 0, -torch.inf)
#         attn = stable_softmax(attn + mask).nan_to_num().triu(1)
#         out = torch.bmm(v, attn).view(*shape).contiguous()
#         return img + out * self._gamma

#     def d_forward(self, img: Tensor, pos: tuple[int, int], clear_cache: bool) -> Tensor:
#         if self._cache == None or clear_cache:
#             self._cache = torch.zeros_like(img)

#         shape, q, k, v = self.qet_qkv(img, pos, clear_cache)
#         i = (pos[0] * shape[-1]) + pos[1]
#         attn = stable_softmax(
#             torch.bmm(q[..., :i].transpose(-2, -1), k[..., i : i + 1])
#         )
#         px = torch.bmm(v[..., :i], attn)
#         self._cache[..., pos[0], pos[1]] = img[..., pos[0], pos[1]] + px.squeeze() * self._gamma
#         return self._cache

# # TODO: Needs a masked convolution
# # class DfAttention(DfModule):
# #     def __init__(
# #         self,
# #         channels: int,
# #         key_dim: Optional[int] = None,
# #         device: Optional[torch.device | str] = None,
# #         dtype: Optional[torch.dtype] = None,
# #     ) -> None:
# #         super().__init__()
# #         self._channels = channels
# #         if key_dim == None:
# #             key_dim = channels // 8
# #         self._key_dim = key_dim

# #         self._conv = DfConv2d(
# #             channels, 2 * key_dim + channels, 1, device=device, dtype=dtype
# #         )
# #         self._gamma = nn.Parameter(torch.ones(1, device=device, dtype=dtype))

# #     def qet_qkv(
# #         self, img: Tensor, pos: Optional[tuple[int, int]], clear_cache: bool
# #     ) -> tuple[torch.Size, Tensor, Tensor, Tensor]:
# #         shape = img.shape
# #         qkv = self._conv(img, pos=pos, clear_cache=clear_cache)
# #         # qkv = self._conv(img)
# #         q, k, v = (
# #             t.view(*t.shape[:-2], -1)
# #             for t in qkv.split_with_sizes([self._key_dim] * 2 + [self._channels], -3)
# #         )

# #         return shape, q, k, v

# #     def b_forward(self, img: Tensor) -> Tensor:
# #         shape, q, k, v = self.qet_qkv(img, None, False)
# #         attn = (q.transpose(-2, -1) @ k).triu(1)
# #         attn = attn.where(attn == 0, -torch.inf)
# #         attn = stable_softmax(attn).nan_to_num()
# #         attn.triu(1)
# #         out = (v @ attn).view(*shape).contiguous()
# #         return img + out * self._gamma

# #     def d_forward(self, img: Tensor, pos: tuple[int, int], clear_cache: bool) -> Tensor:
# #         if self._cache == None or clear_cache:
# #             self._cache = torch.zeros_like(img)

# #         shape, q, k, v = self.qet_qkv(img, pos, clear_cache)

# #         i = (pos[0] * shape[-1]) + pos[1]
# #         attn = q[..., :i].transpose(-2, -1) @ k[..., i : i + 1]
# #         attn = stable_softmax(attn)
# #         px = (v[..., :i] @ attn).view(*shape[:-2], 1, 1)
# #         self._cache[..., pos[0], pos[1]] = img[..., pos[0], pos[1]] + px[..., 0, 0] * self._gamma
# #         return self._cache
