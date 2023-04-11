from typing import Literal, overload


Size1_t = tuple[int]
Size2_t = tuple[int, int]
Size3_t = tuple[int, int, int]
Size4_t = tuple[int, int, int, int]
SizeN_t = tuple[int, ...]
Size1_p = int | Size1_t
Size2_p = int | Size1_t | Size2_t
Size3_p = int | Size1_t | Size3_t
Size4_p = int | Size1_t | Size4_t
SizeN_p = int | SizeN_t


def _bin_sizes(a: SizeN_p, b: SizeN_p) -> tuple[SizeN_t, SizeN_t]:
    la, lb = 1, 1
    if isinstance(a, tuple):
        la = len(a)
    if isinstance(b, tuple):
        lb = len(b)
    if la == 1:
        la = lb
    if lb == 1:
        lb = la
    a = to_size(la, a)
    b = to_size(lb, b)
    assert len(a) == len(b)
    return a, b


@overload
def to_size(n: Literal[1], s: Size1_p) -> Size1_t:
    pass


@overload
def to_size(n: Literal[2], s: Size2_p) -> Size2_t:
    pass


@overload
def to_size(n: Literal[3], s: Size3_p) -> Size3_t:
    pass


@overload
def to_size(n: Literal[4], s: Size4_p) -> Size4_t:
    pass


@overload
def to_size(n: int, s: SizeN_p) -> SizeN_t:
    pass


def to_size(n: int, s: SizeN_p) -> SizeN_t:
    if isinstance(s, tuple):
        l = len(s)
        if l == 1:
            return s * n
        assert l == n
        return s
    return (s,) * n


@overload
def add_size(a: Size1_p, b: Size1_p) -> Size1_t:
    pass


@overload
def add_size(a: Size2_p, b: Size2_p) -> Size2_t:
    pass


@overload
def add_size(a: Size3_p, b: Size3_p) -> Size3_t:
    pass


@overload
def add_size(a: Size4_p, b: Size4_p) -> Size4_t:
    pass


def add_size(a: SizeN_p, b: SizeN_p) -> SizeN_t:
    a, b = _bin_sizes(a, b)
    return tuple(ai + bi for ai, bi in zip(a, b))


@overload
def mul_size(a: Size1_p, b: Size1_p) -> Size1_t:
    pass


@overload
def mul_size(a: Size2_p, b: Size2_p) -> Size2_t:
    pass


@overload
def mul_size(a: Size3_p, b: Size3_p) -> Size3_t:
    pass


@overload
def mul_size(a: Size4_p, b: Size4_p) -> Size4_t:
    pass


def mul_size(a: SizeN_p, b: SizeN_p) -> SizeN_t:
    a, b = _bin_sizes(a, b)
    return tuple(ai * bi for ai, bi in zip(a, b))


@overload
def div_size(a: Size1_p, b: Size1_p) -> Size1_t:
    pass


@overload
def div_size(a: Size2_p, b: Size2_p) -> Size2_t:
    pass


@overload
def div_size(a: Size3_p, b: Size3_p) -> Size3_t:
    pass


@overload
def div_size(a: Size4_p, b: Size4_p) -> Size4_t:
    pass


def div_size(a: SizeN_p, b: SizeN_p) -> SizeN_t:
    a, b = _bin_sizes(a, b)
    return tuple(ai // bi for ai, bi in zip(a, b))
