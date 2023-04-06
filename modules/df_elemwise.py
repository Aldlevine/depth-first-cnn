import torch
import torch.nn as nn
from torch import Tensor

from modules.df_module import DfModule


class DfElemwise(DfModule):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._module = module

    def b_forward(self, img: Tensor) -> Tensor:
        return self._module(img)

    def d_forward(self, img: Tensor, pos: tuple[int, int], clear_cache: bool) -> Tensor:
        if clear_cache or self._cache == None or pos == (0, 0):
            self._cache = torch.zeros_like(img)
        y, x = pos
        self._cache[..., y, x] = self._module(img[..., y, x])
        return self._cache
