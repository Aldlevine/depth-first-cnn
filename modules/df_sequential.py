from typing import Iterable, Optional
import torch
import torch.nn as nn
from torch import Tensor

from modules.df_module import DfModule


class DfSequential(nn.ModuleList, DfModule):
    def __init__(self, modules: Optional[Iterable[nn.Module]] = None) -> None:
        super().__init__()
        if modules:
            self.extend(modules)

    def b_forward(self, img: Tensor) -> Tensor:
        for module in self:
            img = module(img)
        return img

    def d_forward(self, img: Tensor, pos: tuple[int, int], clear_cache: bool) -> Tensor:
        for module in self:
            img = module(img, pos=pos, clear_cache=clear_cache)
        return img

