import abc
import datetime
import timeit
from dataclasses import dataclass
from typing import Literal, Optional

import matplotlib.colors as clr
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.version
from torch import Tensor
from modules.df_attention import DfAttention

from modules.df_conv2d import DfConv2d
from modules.df_elemwise import DfElemwise
from modules.df_module import DfModule
from modules.df_sequential import DfSequential


class PixelCNN(DfModule):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        hidden_layers: int,
        kernel_size: int = 3,
        max_dilation: int = 4,
        discretes: int = 256,
    ) -> None:
        super().__init__()
        self._channels = channels
        self._hidden_channels = hidden_channels
        self._max_dilation = max_dilation
        self._discretes = discretes
        self._in_layer = DfConv2d(channels, hidden_channels, kernel_size)

        self._hid_layers = DfSequential()

        for i in range(hidden_layers // 2):
            dilation = min(i+1, max_dilation)
            self._hid_layers.extend(
                [
                    DfConv2d(hidden_channels, hidden_channels, kernel_size),
                    DfElemwise(nn.SELU()),
                    DfConv2d(hidden_channels, hidden_channels, kernel_size, dilation),
                    DfElemwise(nn.SELU()),
                ]
            )

        self._hid_layers.extend(
            [
                DfConv2d(hidden_channels, hidden_channels, kernel_size),
                # DfElemwise(nn.SELU()),
                DfAttention(hidden_channels),
            ]
        )

        for i in reversed(range(hidden_layers // 2)):
            dilation = min(i+1, max_dilation)
            self._hid_layers.extend(
                [
                    DfConv2d(hidden_channels, hidden_channels, kernel_size, dilation),
                    DfElemwise(nn.SELU()),
                    DfConv2d(hidden_channels, hidden_channels, kernel_size),
                    DfElemwise(nn.SELU()),
                ]
            )

        self._out_layer = DfConv2d(
            hidden_channels, channels * discretes, kernel_size=1, masked=False
        )

    def b_forward(self, img: Tensor) -> Tensor:
        img = self._in_layer(img)
        img = self._hid_layers(img)
        out = self._out_layer(img)
        return out.view(-1, self._channels, self._discretes, *img.shape[-2:])
    
    def d_forward(self, img: Tensor, pos: tuple[int, int], clear_cache: bool) -> Tensor:
        img = self._in_layer(img, pos=pos, clear_cache=clear_cache)
        img = self._hid_layers(img, pos=pos, clear_cache=clear_cache)
        out = self._out_layer(img, pos=pos, clear_cache=clear_cache)
        return out.view(-1, self._channels, self._discretes, *img.shape[-2:])
