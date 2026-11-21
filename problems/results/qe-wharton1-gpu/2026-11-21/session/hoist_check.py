"""
P8 on a GPU: is autodiff's extra product, X times the identity, computed once
outside the Newton loop, as XLA's CPU backend does? It doesn't depend on β.
Compiles `mle`'s JAX solve (autodiff) and `mle_hand`'s (by hand) at `params`
k, as run_cell.py does, and lists each matrix product, `dot` or a cuBLAS call,
with its shape and whether it sits in the while loop.

    JAX_PLATFORMS=cuda CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
        JAX_ENABLE_X64=1 python hoist_check.py <problems folder> [k]
"""
import re
import sys

sys.path.insert(0, sys.argv[1])
import jax  # noqa: E402
import numpy as np  # noqa: E402

import mle  # noqa: E402

k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
m = mle.model("params", k, np.dtype("float64"))
devices = jax.devices()
print(f"{devices[0].platform} {devices[0].device_kind}, n = {m['X'].shape[0]:,}, k = {k}")

HEAD = re.compile(r"^(?:ENTRY )?%?([\w.\-]+) .*\{$")
CALLED = re.compile(r"(?:calls|to_apply|body|condition|branch_computations)=\{?([%\w.\-, ]+)\}?")
for autodiff in (True, False):
    fn, inputs = mle.jax_newton(m, devices, autodiff)
    text = fn.lower(*inputs).compile().as_text()
    calls, products, comp = {}, [], None
    for line in text.splitlines():
        head = HEAD.match(line)
        if head:
            comp = head.group(1)
            calls[comp] = set()
            continue
        if comp is None:
            continue
        for group in CALLED.findall(line):
            calls[comp] |= {c.strip().lstrip("%") for c in group.split(",") if c.strip()}
        if re.search(r"\bdot\(", line) or ("custom-call(" in line and re.search(r"gemm|matmul", line)):
            shape = re.search(r"= \(?(\w+\[[\d,]*\])", line)
            products.append((comp, shape.group(1) if shape else "?"))
    # Everything the while loops' bodies and conditions reach
    loop = set(re.findall(r"while\(.*?body=%?([\w.\-]+)", text))
    loop |= set(re.findall(r"while\(.*?condition=%?([\w.\-]+)", text))
    todo = list(loop)
    while todo:
        for c in calls.get(todo.pop(), ()):
            if c not in loop:
                loop.add(c)
                todo.append(c)
    print(f"{'autodiff' if autodiff else 'by hand'}:")
    for where, shape in products:
        print(f"  {shape:24s} {'in the loop' if where in loop else 'outside the loop'} ({where})")
