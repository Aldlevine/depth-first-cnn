import abc
from typing import Literal, Optional
import torch

import torch.nn as nn
from torch import Tensor
from utils.debug import print_diff

from utils.size import Size2_p, Size2_t, to_size


class DfModule(abc.ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._cache: Optional[Tensor] = None

    @abc.abstractmethod
    def b_forward(self, *img: Optional[Tensor]) -> Tensor:
        raise

    @abc.abstractmethod
    def d_forward(
        self, *img: Optional[Tensor], pos: Size2_t, clear_cache: bool
    ) -> Tensor:
        raise

    def forward(
        self,
        *img: Optional[Tensor],
        pos: Optional[Size2_p] = None,
        clear_cache: bool = False,
    ) -> Tensor:
        if pos == None:
            return self.b_forward(*img)
        else:
            pos = to_size(2, pos)
            return self.d_forward(*img, pos=pos, clear_cache=clear_cache or pos == (0, 0))

    def __call__(
        self,
        *img: Optional[Tensor],
        pos: Optional[Size2_p] = None,
        clear_cache: bool = False,
    ) -> Tensor:
        return self.forward(*img, pos=pos, clear_cache=clear_cache)



@torch.no_grad()
def check_pixel_parity(
    module: DfModule,
    shape: tuple[int, int, int],
    batch_size: int = 64,
    method1: Literal["depth", "breadth"] = "depth",
    method2: Literal["depth", "breadth"] = "breadth",
    device1: Literal["cpu", "cuda"] = "cuda",
    device2: Literal["cpu", "cuda"] = "cuda",
    dtype1: torch.dtype = torch.float32,
    dtype2: torch.dtype = torch.float32,
) -> None:
    input = torch.randn(batch_size, *shape)

    seed = torch.seed()
    input1 = input.to(device1, dtype1)
    mod1 = module.to(device=torch.device(device1), dtype=dtype1)
    out1: Optional[Tensor] = None
    if method1 == "depth":
        for y in range(shape[1]):
            for x in range(shape[2]):
                out = mod1(input1, pos=(y, x))
                if out1 == None:
                    out1 = out.clone()
                else:
                    out1[..., y, x] = out[..., y, x]
    else:
        out1 = mod1(input1)
    assert out1 != None
    out1 = out1.to(device1, dtype1)

    torch.manual_seed(seed)
    input2 = input.to(device2, dtype2)
    mod2 = module.to(device=torch.device(device2), dtype=dtype2)
    out2: Optional[Tensor] = None
    if method2 == "depth":
        for y in range(shape[1]):
            for x in range(shape[2]):
                out = mod2(input2, pos=(y, x))
                if out2 == None:
                    out2 = out.clone()
                else:
                    out2[..., y, x] = out[..., y, x]
    else:
        out2 = mod2(input2)
    assert out2 != None
    out2 = out2.to(device1, dtype1)

    print(f"{module.__class__.__name__}")
    print_diff(out1, out2)
    print()
