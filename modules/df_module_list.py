from typing import Generic, Iterable, Iterator, Optional, TypeVar

import torch
import torch.nn as nn
from torch import Tensor

from modules.df_module import DfModule
from utils.size import Size2_t

_T = TypeVar("_T", bound=DfModule)


class DfModuleList(Generic[_T], nn.ModuleList):
    def __init__(self, modules: Optional[Iterable[DfModule]] = None) -> None:
        super().__init__()
        if modules:
            self.extend(modules)

    def __iter__(self) -> Iterator[DfModule]:
        return super().__iter__() # type: ignore

    def __getitem__(self, idx: int | slice) -> DfModule:
        return super().__getitem__(idx)  # type: ignore

    def __setitem__(self, idx: int, module: DfModule) -> None:
        return super().__setitem__(idx, module)

    def extend(self, modules: Iterable[DfModule]) -> "DfModuleList":
        return super().extend(modules)  # type: ignore

    def append(self, module: DfModule) -> "DfModuleList":
        return super().append(module)  # type: ignore

    def insert(self, index: int, module: DfModule) -> None:
        return super().insert(index, module)

    def b_forward(self, img: Tensor) -> Tensor:
        for module in self:
            img = module(img)
        return img

    def d_forward(self, img: Tensor, pos: Size2_t, clear_cache: bool) -> Tensor:
        for module in self:
            img = module(img, pos=pos, clear_cache=clear_cache)
        return img
