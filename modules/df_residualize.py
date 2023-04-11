from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor

from modules.df_module import DfModule
from utils.size import Size2_t


class DfResidualize(DfModule):
    # def __init__(self, callable: Callable[[Tensor], Tensor], no_depth: bool = False, weight: bool = False) -> None:
    def __init__(self, module: DfModule, no_depth: bool = False, weight: bool = False) -> None:
        super().__init__()
        self._module = module
        self._no_depth = no_depth
        if weight:
            self._weight = nn.Parameter(torch.ones(1))
        else:
            self._weight = None

    @property
    def weight(self) -> Tensor | int:
        if self._weight == None:
            return 1
        return self._weight

    def b_forward(self, img: Tensor) -> Tensor:
        return self.weight * self._module(img) + img

    def d_forward(self, img: Tensor, pos: Size2_t, clear_cache: bool) -> Tensor:
        if self._no_depth:
            return self.b_forward(img)
        y, x = pos
        out = self._module(img, pos=pos, clear_cache=clear_cache)
        return self.weight * out + img
        # if clear_cache or self._cache == None:
        #     self._cache = torch.zeros(out.shape[0], out.shape[1], *img.shape[2:], dtype=out.dtype, device=img.device)
        # self._cache[..., y, x] = self.weight * out[..., 0, 0] + img[..., y, x]
        # return self._cache

