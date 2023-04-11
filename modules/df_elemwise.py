from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor

from modules.df_module import DfModule
from utils.size import Size2_t


class DfElemwise(DfModule):
    def __init__(self, callable: Callable[[Tensor], Tensor], no_depth: bool = False) -> None:
        super().__init__()
        self._callable = callable
        self._no_depth = no_depth

    def b_forward(self, img: Tensor) -> Tensor:
        return self._callable(img)

    def d_forward(self, img: Tensor, pos: Size2_t, clear_cache: bool) -> Tensor:
        if self._no_depth:
            return self.b_forward(img)
        y, x = pos
        out = self._callable(img[..., y:y+1, x:x+1])
        if clear_cache or self._cache == None:
            self._cache = torch.zeros(out.shape[0], out.shape[1], *img.shape[2:], dtype=out.dtype, device=img.device)
        self._cache[..., y, x] = out[..., 0, 0]
        return self._cache
