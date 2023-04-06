from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from modules.df_module import DfModule

class DfConv2d(DfModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        masked: bool = True,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._dilation = dilation
        self._masked = masked
        self._device = device
        self._dtype = dtype

        self.weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size,
                dtype=dtype,
                device=device,
            )
        )

        nn.init.kaiming_normal_(self.weight.data, nonlinearity="relu")

    @property
    def _dilated_kernel_size(self) -> int:
        kernel_size = self._kernel_size
        dilation = self._dilation
        dk = kernel_size + (kernel_size-1) * (dilation-1)
        return dk


    def b_forward(self, img: Tensor) -> Tensor:
        padding = self._dilated_kernel_size // 2
        dilation = self._dilation
        return F.conv2d(img, self.weight, padding=padding, dilation=dilation)

    def d_forward(self, img: Tensor, pos: tuple[int, int], clear_cache: bool) -> Tensor:
        dilation = self._dilation
        dilated_kernel_size = self._dilated_kernel_size
        padding = dilated_kernel_size // 2

        if clear_cache or self._cache == None or self._padded == None or pos == (0, 0):
            self._cache = torch.zeros(
                img.shape[0],
                self._out_channels,
                *img.shape[2:],
                dtype=img.dtype,
                device=img.device,
            )
            self._padded = F.pad(img, [padding] * 4, mode="constant", value=0)

        y, x = pos
        startp = None if padding == 0 else padding
        endp = None if padding == 0 else -padding
        self._padded[..., startp:endp, startp:endp] = img
        patch = self._padded[..., y : y + dilated_kernel_size : dilation, x : x + dilated_kernel_size : dilation]

        out: Tensor = F.conv2d(patch, self.weight)
        self._cache[..., y, x] = out.squeeze()
        return self._cache

    def forward(
        self,
        img: Tensor,
        pos: Optional[tuple[int, int]] = None,
        clear_cache: bool = False,
    ) -> Tensor:
        if self._masked:
            padding = self._kernel_size // 2
            self.weight.data[..., padding + 1 :, :] = 0
            self.weight.data[..., padding, padding:] = 0

        return super().forward(img, pos=pos, clear_cache=clear_cache)