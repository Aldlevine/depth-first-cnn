import abc
from typing import Literal, Optional
import torch

import torch.nn as nn
from torch import Tensor


class DfModule(abc.ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._cache: Optional[Tensor] = None

    @abc.abstractmethod
    def b_forward(self, *img: Tensor) -> Tensor:
        raise

    @abc.abstractmethod
    def d_forward(
        self, *img: Tensor, pos: tuple[int, int], clear_cache: bool
    ) -> Tensor:
        raise

    def forward(
        self,
        *img: Tensor,
        pos: Optional[tuple[int, int]] = None,
        clear_cache: bool = False,
    ) -> Tensor:
        if pos == None:
            return self.b_forward(*img)
        else:
            return self.d_forward(*img, pos=pos, clear_cache=clear_cache)

    def __call__(
        self,
        *img: Tensor,
        pos: Optional[tuple[int, int]] = None,
        clear_cache: bool = False,
    ) -> Tensor:
        return self.forward(*img, pos=pos, clear_cache=clear_cache)


@torch.no_grad()
def check_pixel_parity(
    module: DfModule,
    shape: tuple[int, int, int],
    position: tuple[int, int] = (0, 0),
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
    out1 = mod1(input1, pos=position if method1 == "depth" else None, clear_cache=True)[
        ..., position[0], position[1]
    ]
    out1 = out1.to(device1, dtype1)

    torch.manual_seed(seed)
    input2 = input.to(device2, dtype2)
    mod2 = module.to(device=torch.device(device2), dtype=dtype2)
    out2 = mod2(input2, pos=position if method2 == "depth" else None, clear_cache=True)[
        ..., position[0], position[1]
    ]
    out2 = out2.to(device1, dtype1)

    if torch.equal(out1, out2):
        print("Exactly equal")
    else:
        if torch.allclose(out1, out2):
            print("Close enough according to PyTorch defaults")
        else:
            print("Not close at all")
        max_diff = (out1.to(device1, dtype1) - out2.to(device1, dtype1)).abs().max()
        print(f"Maximum difference: {max_diff:.5f}")
