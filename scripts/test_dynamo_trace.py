"""TorchDynamo miscompiles Ultralytics' `y.extend(<gen reading y[-1]>)` blocks.

CPU only, no GPU, ~1 min. Reproducer for the upstream reports.

Several Ultralytics blocks build a sequential chain like:

    y = [x]
    y.extend(m(y[-1]) for m in self.m)

`list.extend` consumes the generator one item at a time, so each `y[-1]` read
sees the element the previous step just appended. That is valid Python and is
correct in eager mode. Dynamo traces it wrongly: `HGBlock` raises a channel
mismatch, and every other affected block silently returns different numbers.

Run:  python scripts/test_dynamo_trace.py
Exit code is non-zero if any block mistraces.
"""

import sys
import types

import torch
import torch._dynamo
from ultralytics.nn.modules import block as B

TOL = 1e-6

CASES = [
    ("HGBlock",              lambda: B.HGBlock(32, 16, 64, n=6),           None),
    ("SPPF",                 lambda: B.SPPF(32, 64),                       None),
    ("C2f",                  lambda: B.C2f(32, 64, n=2),                   None),
    ("C2f.forward_split",    lambda: B.C2f(32, 64, n=2),                   "forward_split"),
    ("C3f",                  lambda: B.C3f(32, 64, n=2),                   None),
    ("SPPELAN",              lambda: B.SPPELAN(32, 64, 16),                None),
    ("RepNCSPELAN4",         lambda: B.RepNCSPELAN4(32, 64, 32, 16, n=1),  None),
    ("RepNCSPELAN4.forward_split",
     lambda: B.RepNCSPELAN4(32, 64, 32, 16, n=1), "forward_split"),
    ("A2C2f",                lambda: B.A2C2f(32, 64, n=1, a2=False),       None),
]


def compare(build, method):
    """max |eager - dynamo| for one block, or an exception string."""
    torch._dynamo.reset()
    mod = build().eval()
    x = torch.randn(1, 32, 32, 32)
    fn = getattr(mod, method) if method else mod
    with torch.no_grad():
        ref = fn(x)
        try:
            got = torch.compile(fn, backend="eager", dynamic=False)(x)
        except Exception as exc:
            return f"{type(exc).__name__}"
    return (ref - got).abs().max().item()


# --- explicit-loop rewrites, to prove the generator is what breaks ----------
def c2f_loop(self, x):
    y = list(self.cv1(x).chunk(2, 1))
    for m in self.m:
        y.append(m(y[-1]))
    return self.cv2(torch.cat(y, 1))


def sppf_loop(self, x):
    y = [self.cv1(x)]
    for _ in range(getattr(self, "n", 3)):
        y.append(self.m(y[-1]))
    return self.cv2(torch.cat(y, 1))


def isolate(build, loop_fn):
    """Swap only the loop form on one instance; everything else held fixed."""
    x = torch.randn(1, 32, 32, 32)
    mod = build().eval()
    with torch.no_grad():
        eager_gen = mod(x)
        torch._dynamo.reset()
        try:
            gen_delta = (eager_gen - torch.compile(
                mod, backend="eager", dynamic=False)(x)).abs().max().item()
        except Exception as exc:
            gen_delta = f"{type(exc).__name__}"

        mod.forward = types.MethodType(loop_fn, mod)
        eager_loop = mod(x)
        torch._dynamo.reset()
        loop_delta = (eager_loop - torch.compile(
            mod, backend="eager", dynamic=False)(x)).abs().max().item()
    return {
        "eager(gen) vs eager(loop)": (eager_gen - eager_loop).abs().max().item(),
        "eager(loop) vs dynamo(loop)": loop_delta,
        "eager(gen) vs dynamo(gen)": gen_delta,
    }


def main() -> int:
    import ultralytics
    print(f"torch {torch.__version__} / ultralytics {ultralytics.__version__}\n")

    bad = []
    print("eager vs dynamo, per block:")
    for name, build, method in CASES:
        d = compare(build, method)
        if isinstance(d, str):
            print(f"  RAISES     {name:28s} {d}")
            bad.append(name)
        elif d > TOL:
            print(f"  MISTRACES  {name:28s} max|delta| = {d:.3e}")
            bad.append(name)
        else:
            print(f"  ok         {name:28s} max|delta| = {d:.3e}")

    print("\nisolating the cause (same instance, weights and input):")
    for label, build, loop_fn in [("C2f", lambda: B.C2f(32, 64, n=2), c2f_loop),
                                  ("SPPF", lambda: B.SPPF(32, 64), sppf_loop)]:
        print(f"  {label}")
        for k, v in isolate(build, loop_fn).items():
            print(f"    {k:30s} {v if isinstance(v, str) else f'{v:.3e}'}")

    if bad:
        print(f"\n{len(bad)} block(s) mistrace: {', '.join(bad)}")
        return 1
    print("\nall blocks trace correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
