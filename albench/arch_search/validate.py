"""Static and runtime validation for model code written by an LLM.

The arch-search loop lets a model propose a torch Module as SOURCE, not as a
config dict, so it can reach architectures our hand-written space cannot express.
That only works if generated code is checked before it runs. Three layers:

  1. AST whitelist   -- imports, calls and attribute access are restricted to what
                        building a network needs. Anything touching the filesystem,
                        network, process table or interpreter internals is refused
                        at parse time, before any code executes.
  2. Restricted exec -- the module body runs with a namespace containing only
                        torch/nn/F/math, and builtins pared to the ones a layer
                        definition legitimately uses.
  3. Runtime budget  -- instantiate, check it maps (B,4,L) -> (B,), cap parameter
                        count and forward latency, so a pathological proposal
                        cannot wedge a GPU job.

The latency cap is a CRASH GUARD, not a performance filter, and it is deliberately
loose. It times a CPU forward, and CPU speed varies hugely between the login node
and a loaded GPU node -- an early version set it at 400ms and silently rejected
four sound architectures that measured 3.7ms at proposal time and 687ms on the
training node. Parameter count is the meaningful size bound; training has its own
early stopping. Reject here only what would hang.

A proposal that fails any layer is recorded with its reason and skipped. Nothing
is silently repaired: a model we had to fix is not the model that was proposed.
"""

from __future__ import annotations

import ast
import math
import time
from dataclasses import dataclass, field

ALLOWED_IMPORTS = {
    # anything useful for expressing an architecture
    "torch", "numpy", "math", "cmath", "random", "statistics",
    # plumbing a layer definition legitimately needs
    "typing", "dataclasses", "collections", "itertools", "functools",
    "abc", "enum", "copy", "warnings", "contextlib", "operator", "re",
}
BANNED_NAMES = {
    "exec", "eval", "compile", "__import__", "open", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "memoryview",
}
BANNED_ATTRS = {
    "__subclasses__", "__globals__", "__code__", "__closure__", "__builtins__",
    "__bases__", "__mro__", "__dict__", "__reduce__", "__class__",
}

SAFE_BUILTINS = {
    "range", "len", "enumerate", "zip", "min", "max", "abs", "sum", "int", "float",
    "bool", "str", "list", "tuple", "dict", "set", "sorted", "reversed", "round",
    "isinstance", "issubclass", "super", "print", "any", "all", "map", "filter",
    "ValueError", "TypeError", "RuntimeError", "NotImplementedError", "Exception",
    "property", "staticmethod", "classmethod", "object", "type", "repr", "divmod",
    "pow", "slice", "iter", "next", "hasattr", "format", "frozenset",
    # the `class` statement compiles to a __build_class__ call, so defining any
    # nn.Module needs it; it is not an escape route on its own because the AST
    # pass already refused the dunder attributes that would make it one
    "__build_class__", "__name__",
}


@dataclass
class Verdict:
    ok: bool
    reason: str = ""
    n_params: int = 0
    fwd_ms: float = 0.0
    layers: dict = field(default_factory=dict)


def static_check(src: str) -> str | None:
    """Return a refusal reason, or None if the source is acceptable."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return f"syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in {i.split(".")[0] for i in ALLOWED_IMPORTS}:
                    return f"import not allowed: {a.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.split(".")[0] not in {i.split(".")[0] for i in ALLOWED_IMPORTS}:
                return f"import-from not allowed: {mod}"
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return f"banned name: {node.id}"
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            return f"banned attribute: {node.attr}"
    return None


def classify_layers(src: str) -> dict:
    """What kinds of layer did this proposal actually use?

    This is the record of what the search EXPLORED, independent of how it scored —
    without it we would only know which architectures won, never which were tried.
    """
    kinds = {
        "conv": ("Conv1d", "Conv2d", "ConvTranspose1d"),
        "attention": ("MultiheadAttention", "scaled_dot_product_attention",
                      "TransformerEncoder", "TransformerEncoderLayer", "softmax"),
        "recurrent": ("LSTM", "GRU", "RNN"),
        "norm": ("BatchNorm1d", "LayerNorm", "GroupNorm", "InstanceNorm1d"),
        "pool": ("MaxPool1d", "AvgPool1d", "AdaptiveAvgPool1d", "AdaptiveMaxPool1d"),
        "dropout": ("Dropout", "Dropout1d"),
        "residual": ("+ residual", "+ identity", "+ x", "+= "),
        "dilation": ("dilation",),
        "se_block": ("Squeeze", "se_", "SEBlock", "excitation"),
        "linear": ("Linear",),
        "embedding": ("Embedding",),
    }
    out = {}
    for k, pats in kinds.items():
        n = sum(src.count(p) for p in pats)
        if n:
            out[k] = n
    return out


def build_and_check(src: str, *, seq_len: int = 200, n_channels: int = 4,
                    max_params: int = 60_000_000, max_fwd_ms: float = 8000.0,
                    batch: int = 8, class_name: str = "Net") -> tuple[object, Verdict]:
    """Static-check, exec in a restricted namespace, then instantiate and probe.

    Returns (module_instance_or_None, Verdict). The caller trains it only if ok.
    """
    reason = static_check(src)
    if reason:
        return None, Verdict(False, reason)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    safe_builtins = {k: __builtins__[k] if isinstance(__builtins__, dict)
                     else getattr(__builtins__, k)
                     for k in SAFE_BUILTINS
                     if (k in __builtins__ if isinstance(__builtins__, dict)
                         else hasattr(__builtins__, k))}
    import importlib

    def _gated_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root not in {i.split(".")[0] for i in ALLOWED_IMPORTS}:
            raise ImportError(f"import of {name!r} is not permitted in generated code")
        return importlib.__import__(name, globals, locals, fromlist, level)

    safe_builtins["__import__"] = _gated_import
    ns = {"__builtins__": safe_builtins, "torch": torch, "nn": nn, "F": F, "math": math}
    try:
        exec(compile(src, "<generated-arch>", "exec"), ns)          # noqa: S102
    except Exception as e:
        return None, Verdict(False, f"exec failed: {type(e).__name__}: {e}")

    cls = ns.get(class_name)
    if cls is None:
        cands = [v for k, v in ns.items()
                 if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module]
        if not cands:
            return None, Verdict(False, f"no class named {class_name} and no nn.Module found")
        cls = cands[-1]
    try:
        model = cls()
    except Exception as e:
        return None, Verdict(False, f"instantiation failed: {type(e).__name__}: {e}")

    n_params = sum(p.numel() for p in model.parameters())
    if n_params > max_params:
        return None, Verdict(False, f"too many parameters: {n_params:,} > {max_params:,}",
                             n_params=n_params)
    if n_params == 0:
        return None, Verdict(False, "model has no parameters")

    x = torch.randn(batch, n_channels, seq_len)
    model.eval()
    try:
        with torch.no_grad():
            model(x)                                   # warm up, then time
            t0 = time.perf_counter()
            y = model(x)
            fwd_ms = 1000 * (time.perf_counter() - t0)
    except Exception as e:
        return None, Verdict(False, f"forward failed: {type(e).__name__}: {e}", n_params=n_params)

    y = y.squeeze()
    if y.shape != (batch,):
        return None, Verdict(False, f"wrong output shape {tuple(y.shape)}, expected ({batch},)",
                             n_params=n_params)
    if not torch.isfinite(y).all():
        return None, Verdict(False, "forward produced non-finite output", n_params=n_params)
    if fwd_ms > max_fwd_ms:
        return None, Verdict(False, f"forward too slow: {fwd_ms:.0f}ms > {max_fwd_ms:.0f}ms",
                             n_params=n_params, fwd_ms=fwd_ms)
    # a model that ignores its input would score like a constant predictor and
    # quietly pollute the ensemble, so require the output to actually move
    with torch.no_grad():
        y2 = model(torch.randn(batch, n_channels, seq_len)).squeeze()
    if torch.allclose(y, y2, atol=1e-7):
        return None, Verdict(False, "output does not depend on input", n_params=n_params)

    # Train-mode forward+backward at a DIFFERENT batch size — the real training step.
    # The eval/batch-8 probe above misses bugs that only fire in train() (BatchNorm),
    # at another batch size (reshape/view that infers a dim), or in autograd at the
    # loss. ~1% of generated nets passed the forward probe but then died at train time
    # with a shape mismatch; catching them here frees the GPU slot instead.
    model.train()
    try:
        xb = torch.randn(max(2, batch * 2), n_channels, seq_len)
        ob = model(xb).squeeze()
        if ob.shape[0] != xb.shape[0]:
            return None, Verdict(False, f"train-mode output batch {tuple(ob.shape)} != {xb.shape[0]}",
                                 n_params=n_params)
        (((ob - torch.randn(ob.shape[0])) ** 2).mean()).backward()
    except Exception as e:
        return None, Verdict(False, f"train-step probe failed: {type(e).__name__}: {e}",
                             n_params=n_params)
    model.zero_grad(set_to_none=True)
    return model, Verdict(True, "", n_params=n_params, fwd_ms=fwd_ms, layers=classify_layers(src))
