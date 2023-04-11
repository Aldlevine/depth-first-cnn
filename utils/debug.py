from textwrap import dedent, indent
import torch
from torch import Tensor

def stats(x: Tensor) -> str:
    return dedent(f"""\
    Range [{x.min().item():.2e}, {x.max().item():.2e}]
    Mean {x.mean().item():.2e}
    Var {x.var().item():.2e}\
    """)

def print_diff(a: Tensor, b: Tensor) -> None:
    equal = (a == b).count_nonzero().item() / a.numel() * 100
    close = torch.isclose(a, b).count_nonzero().item() / a.numel() * 100
    max_diff = (a - b).abs().max().item()
    print(dedent(f"""\
Equal {equal:.2f}%
Close {close:.2f}%
MaxDiff {max_diff:.2e}
A:
{indent(stats(a), " ")}
B:
{indent(stats(b), " ")}
Diff:
{indent(stats(a - b), " ")}
    """))
    # print(f"Equal {equal:.2f}%")
    # print(f"Close {close:.2f}%")
    # print("MaxDiff", max_diff)
    # print(f"A:")
    # print(f"    Range: [{a.min().item():.2e}, {a.max().item():.2e}]")
    # print(f"    Mean: {a.mean().item():.2e}")
    # print(f"    Std: {a.std().item():.2e}")
    # print(f"B:")
    # print(f"    Range: [{b.min().item():.2e}, {b.max().item():.2e}]")
    # print(f"    Mean: {b.mean().item():.2e}")
    # print(f"    Std: {b.std().item():.2e}")