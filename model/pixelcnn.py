import abc
import datetime
import timeit
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import matplotlib.colors as clr
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.version
from torch import Tensor
from torch.types import _size
from tqdm import tqdm

from modules.df_attention import DfAttention
from modules.df_conv2d import DfConv2d
from modules.df_elemwise import DfElemwise
from modules.df_module import DfModule
from modules.df_module_list import DfModuleList
from modules.df_residualize import DfResidualize
from modules.df_sequential import DfSequential
from utils.size import Size2_t, SizeN_t


class DfVerticalConv2d(DfConv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        mask_center: bool = False,
        dilation: int = 1,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size // 2 + 1 :, :] = 0
        if mask_center:
            mask[kernel_size // 2, :] = 0
        else:
            mask[kernel_size // 2, kernel_size // 2 + 1 :] = 0

        super().__init__(
            in_channels, out_channels, kernel_size, dilation, mask, device, dtype
        )


class DfHorizontalConv2d(DfConv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        mask_center: bool = False,
        dilation: int = 1,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        mask = torch.ones(1, kernel_size)
        mask[0, kernel_size // 2 + 1 :] = 0
        if mask_center:
            mask[0, kernel_size // 2] = 0

        super().__init__(
            in_channels, out_channels, (1, kernel_size), dilation, mask, device, dtype
        )


class DfGatedMaskedConv2d(DfModule):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        aux_channels: int = 0,
        noise: float = 0.1,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self._h_cache: Optional[Tensor] = None
        self._v_cache: Optional[Tensor] = None

        self._noise = noise

        self._v_conv = DfVerticalConv2d(
            in_channels,
            in_channels * 2,
            kernel_size,
            dilation=dilation,
            device=device,
            dtype=dtype,
        )

        self._h_conv = DfHorizontalConv2d(
            in_channels,
            in_channels * 2,
            kernel_size,
            dilation=dilation,
            device=device,
            dtype=dtype,
        )

        if aux_channels > 0:
            self._aux_conv = DfConv2d(aux_channels, in_channels * 2, 1)
        else:
            self._aux_conv = None

        self._v2h_conv = DfConv2d(in_channels * 2, in_channels * 2, kernel_size=1)

        self._hout_conv = DfConv2d(in_channels, in_channels, kernel_size=1)

    def b_forward(
        self, h_img: Tensor, v_img: Tensor, aux: Optional[Tensor]
    ) -> tuple[Tensor, Tensor]:
        if self.training and self._noise > 0:
            h_img = h_img + torch.randn_like(h_img) * self._noise
            v_img = v_img + torch.randn_like(v_img) * self._noise

        a_val, a_gate = 0, 0
        if aux != None:
            assert self._aux_conv != None
            a_feat = self._aux_conv(aux)
            a_val, a_gate = a_feat.chunk(2, dim=-3)

        v_feat = self._v_conv(v_img)
        v_val, v_gate = v_feat.chunk(2, dim=-3)
        v_out = torch.tanh(v_val + a_val) * torch.sigmoid(v_gate + a_gate)

        h_feat = self._h_conv(h_img) + self._v2h_conv(v_feat)
        h_val, h_gate = h_feat.chunk(2, dim=-3)
        h_out = torch.tanh(h_val + a_val) * torch.sigmoid(h_gate + a_gate)
        h_out = h_img + self._hout_conv(h_out)

        return h_out, v_out

    def d_forward(
        self,
        h_img: Tensor,
        v_img: Tensor,
        aux: Optional[Tensor],
        pos: tuple[int, int],
        clear_cache: bool,
    ) -> tuple[Tensor, Tensor]:
        y, x = pos

        if self._h_cache == None or clear_cache:
            self._h_cache = torch.zeros_like(h_img)
        if self._v_cache == None or clear_cache:
            self._v_cache = torch.zeros_like(v_img)

        if aux != None:
            assert self._aux_conv != None
            a_feat = self._aux_conv(aux, pos=pos, clear_cache=clear_cache)
            a_val, a_gate = a_feat[..., y, x].chunk(2, dim=-1)
        else:
            a_val, a_gate = 0, 0

        v_feat = self._v_conv(v_img, pos=pos, clear_cache=clear_cache)
        v_val, v_gate = v_feat[..., y, x].chunk(2, dim=-1)
        v_out = torch.tanh(v_val + a_val) * torch.sigmoid(v_gate + a_gate)

        h_feat = self._h_conv(h_img, pos=pos, clear_cache=clear_cache) + self._v2h_conv(
            v_feat, pos=pos, clear_cache=clear_cache
        )

        y, x = pos
        h_val, h_gate = h_feat[..., y : y + 1, x : x + 1].chunk(2, dim=-3)
        h_out = torch.tanh(h_val + a_val) * torch.sigmoid(h_gate + a_gate)
        h_out = (
            h_img[..., y, x]
            + self._hout_conv(h_out, pos=(0, 0), clear_cache=True)[..., 0, 0]
        )

        self._h_cache[..., y, x] = h_out
        self._v_cache[..., y, x] = v_out

        return self._h_cache, self._v_cache


class PixelCNN(DfModule):
    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def hidden_channels(self) -> int:
        return self._hidden_channels

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        aux_channels: int = 0,
        num_classes: int = 256,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self._in_channels = in_channels
        self._hidden_channels = hidden_channels
        self._aux_channels = aux_channels
        self._num_classes = num_classes

        self._continuous = DfElemwise(
            lambda t: (t.to(self.dtype) / (num_classes - 1)) * 2 - 1
        )

        self._h_conv = DfHorizontalConv2d(
            in_channels, hidden_channels, kernel_size, mask_center=True
        )
        self._v_conv = DfVerticalConv2d(
            in_channels, hidden_channels, kernel_size, mask_center=True
        )
        self._layers = DfModuleList(
            [
                # block 1
                DfGatedMaskedConv2d(hidden_channels, aux_channels=aux_channels),
                DfGatedMaskedConv2d(
                    hidden_channels, aux_channels=aux_channels, dilation=2
                ),
                DfGatedMaskedConv2d(hidden_channels, aux_channels=aux_channels),
                DfGatedMaskedConv2d(
                    hidden_channels, aux_channels=aux_channels, dilation=3
                ),
                DfGatedMaskedConv2d(hidden_channels, aux_channels=aux_channels),
                DfGatedMaskedConv2d(
                    hidden_channels, aux_channels=aux_channels, dilation=4
                ),
                DfGatedMaskedConv2d(hidden_channels, aux_channels=aux_channels),
                DfGatedMaskedConv2d(
                    hidden_channels, aux_channels=aux_channels, dilation=3
                ),
                DfGatedMaskedConv2d(hidden_channels, aux_channels=aux_channels),
                DfGatedMaskedConv2d(
                    hidden_channels, aux_channels=aux_channels, dilation=2
                ),
                DfGatedMaskedConv2d(hidden_channels, aux_channels=aux_channels),
                # DfResidualize(DfAttention(
                #     hidden_channels, hidden_channels // 8, hidden_channels
                # )),
                # block 2
                DfGatedMaskedConv2d(hidden_channels, aux_channels=aux_channels),
                DfGatedMaskedConv2d(
                    hidden_channels, aux_channels=aux_channels, dilation=2
                ),
                DfGatedMaskedConv2d(hidden_channels, aux_channels=aux_channels),
                DfGatedMaskedConv2d(
                    hidden_channels, aux_channels=aux_channels, dilation=3
                ),
                DfGatedMaskedConv2d(hidden_channels, aux_channels=aux_channels),
                DfGatedMaskedConv2d(
                    hidden_channels, aux_channels=aux_channels, dilation=4
                ),
                DfGatedMaskedConv2d(hidden_channels, aux_channels=aux_channels),
                DfGatedMaskedConv2d(
                    hidden_channels, aux_channels=aux_channels, dilation=3
                ),
                DfGatedMaskedConv2d(hidden_channels, aux_channels=aux_channels),
                DfGatedMaskedConv2d(
                    hidden_channels, aux_channels=aux_channels, dilation=2
                ),
                DfGatedMaskedConv2d(hidden_channels, aux_channels=aux_channels),
                # DfResidualize(DfAttention(
                #     hidden_channels, hidden_channels // 8, hidden_channels
                # )),
            ]
        )

        self._out_act = DfElemwise(nn.ELU())
        self._out_conv = DfConv2d(
            hidden_channels, in_channels * num_classes, kernel_size=1
        )

    def b_forward(self, img: Tensor, aux: Optional[Tensor]) -> Tensor:
        img = self._continuous(img)
        h = self._h_conv(img)
        v = self._v_conv(img)

        for layer in self._layers:
            if isinstance(layer, DfGatedMaskedConv2d):
                h, v = layer(h, v, aux)
            else:
                h = layer(h)

        h = self._out_act(h)
        out = self._out_conv(h)

        out = out.reshape(
            out.shape[0],
            self.num_classes,
            out.shape[1] // self.num_classes,
            *out.shape[2:]
        )
        return out

    def d_forward(
        self,
        img: Tensor,
        aux: Optional[Tensor],
        pos: tuple[int, int],
        clear_cache: bool,
    ) -> Tensor:
        img = self._continuous(img)
        h = self._h_conv(img, pos=pos, clear_cache=clear_cache)
        v = self._v_conv(img, pos=pos, clear_cache=clear_cache)

        for layer in self._layers:
            if isinstance(layer, DfGatedMaskedConv2d):
                h, v = layer(h, v, aux, pos=pos, clear_cache=clear_cache)
            else:
                h = layer(h, pos=pos, clear_cache=clear_cache)

        h = self._out_act(h, pos=pos, clear_cache=clear_cache)
        out = self._out_conv(h, pos=pos, clear_cache=clear_cache)

        out = out.reshape(
            out.shape[0],
            self.num_classes,
            out.shape[1] // self.num_classes,
            *out.shape[2:]
        )
        return out

    def get_class_aux(self, shape: Size2_t, classes: Tensor) -> Tensor:
        aux: Tensor = F.one_hot(classes, self._aux_channels)
        aux = aux.to(self.device)
        aux = aux.view(-1, self._aux_channels, 1, 1).repeat(1, 1, *shape)
        return aux / self._aux_channels

    def calc_likelihood(self, img: Tensor, aux: Optional[Tensor]) -> Tensor:
        pred = self(img, aux)
        nll = F.cross_entropy(pred, img, reduction="none")
        bpd = nll.mean(dim=[1, 2, 3]) * torch.log2(torch.exp(torch.tensor(1)))
        return bpd.mean()

    def sample(
        self,
        img: Optional[Tensor] = None,
        shape: Optional[_size] = None,
        aux: Optional[Tensor] = None,
        depth_first: bool = True,
    ) -> Tensor:
        device = next(self.parameters()).device
        if img == None:
            assert shape != None
            img = torch.full(shape, -1, dtype=torch.long, device=device)
        else:
            assert shape == None
            img = img.to(device)
            shape = img.shape

        if self._aux_channels > 0:
            if aux == None:
                aux = torch.randint(
                    0, self._aux_channels, [shape[0]], device=self.device
                )
                aux = self.get_class_aux(img.shape[-2:], aux)
            assert aux.shape[0] == shape[0] and aux.shape[-2:] == shape[-2:]
        else:
            assert aux == None

        prog = tqdm(total=shape[-2] * shape[-1], desc="Sampling", leave=False)
        for y in range(shape[-2]):
            for x in range(shape[-1]):
                if (img[..., y, x] != -1).all().item():
                    continue
                pos = (y, x) if depth_first else None
                pred = self(img, aux, pos=pos)
                probs = torch.softmax(pred[..., y, x], dim=-2)
                for c in range(shape[-3]):
                    img[..., c, y, x] = torch.multinomial(
                        probs[..., c], num_samples=1
                    ).squeeze(-1)
                prog.update()
        return img
