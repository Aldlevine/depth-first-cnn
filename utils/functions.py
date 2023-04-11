from torch import Tensor


def stable_softmax(x: Tensor, dim: int) -> Tensor:
    if x.numel() == 0:
        return x.clone()
    return (x - x.max()).softmax(dim)
