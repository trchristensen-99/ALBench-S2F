#!/usr/bin/env python
"""Slide-ready architecture diagrams of the 20 configurations of menu v1 (the ones actually trained and stacked).

Each network is built by the repository's own builder, run once on a (1, 4, 200) input under a torch-function recorder plus forward hooks, and the recorded
dataflow (which tensor was produced by which layer and consumed by which later layer) is drawn: the main flow of the top-level stages, curved arrows for
every skip / concatenation / gating that the execution actually contains, and a detail panel with the internals of the most repeated block. Nothing is
inferred from names: structure comes from the executed graph, so what is drawn is what the code does.

    .venv/bin/python scripts/analysis/make_architecture_diagrams.py [--out slide_figures/architectures] [--only <id-substring>]

Outputs in --out: arch_<id>.png (one per entry), architectures_top6.png, architectures_overview.png, menu_hyperparameters.png, README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
import torch.nn as nn
from torch.overrides import TorchFunctionMode

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MENU = ROOT / "albench/ensemble/menus/menu_v1.json"
WEIGHTS = Path(
    "/private/tmp/claude-502/-Users-christen-Downloads-ALBench-S2F/8d197a6d-6a2f-448d-9021-2efc12a49072/scratchpad/prelim/data_oct32.json"
)

# ------------------------------------------------------------------------------------------------------------------------------ palette
COL = {
    "conv": "#1b6ca8",
    "norm": "#8a949c",
    "attn": "#c4571a",
    "rnn": "#7a5195",
    "pool": "#2a8f5b",
    "head": "#2b2f36",
    "mlp": "#5d6f80",
    "merge": "#b9a35a",
    "input": "#d9dee2",
}
KIND_LABEL = {
    "conv": "convolution",
    "norm": "norm / act.",
    "attn": "attention",
    "rnn": "recurrent",
    "pool": "pooling",
    "head": "readout head",
    "mlp": "linear / MLP",
    "merge": "add / concat / gate",
}

# ------------------------------------------------------------------------------------------------------------------------------ tracing
TRANSPARENT = {
    "view",
    "reshape",
    "permute",
    "transpose",
    "unsqueeze",
    "squeeze",
    "flatten",
    "contiguous",
    "expand",
    "expand_as",
    "getitem",
    "get",
    "float",
    "to",
    "type",
    "detach",
    "clone",
    "repeat",
    "unflatten",
    "movedim",
    "swapaxes",
    "t",
    "narrow",
    "select",
    "view_as",
    "type_as",
    "half",
    "double",
    "pad",
    "_pad",
    "unfold",
    "roll",
    "flip",
    "mT",
    "T",
    "chunk",
    "split",
    "unbind",
    "tensor_split",
    "cpu",
    "numpy",
}
ARITH = {
    "add",
    "__add__",
    "__radd__",
    "sub",
    "__sub__",
    "__rsub__",
    "mul",
    "__mul__",
    "__rmul__",
    "div",
    "__truediv__",
    "__rtruediv__",
    "neg",
    "__neg__",
    "pow",
    "__pow__",
}
POOL_OPS = {
    "avg_pool1d",
    "max_pool1d",
    "adaptive_avg_pool1d",
    "adaptive_max_pool1d",
    "avg_pool2d",
    "max_pool2d",
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
    "mean",
    "max",
    "amax",
    "sum",
    "topk",
    "sort",
    "logsumexp",
    "std",
    "var",
    "median",
    "kthvalue",
    "cumsum",
}
ACT_OPS = {
    "silu",
    "relu",
    "gelu",
    "sigmoid",
    "softplus",
    "tanh",
    "elu",
    "leaky_relu",
    "mish",
    "dropout",
    "layer_norm",
    "batch_norm",
    "group_norm",
    "softmax",
    "log_softmax",
    "normalize",
    "hardswish",
    "glu",
    "relu6",
    "selu",
    "exp",
    "log",
    "abs",
    "clamp",
    "sqrt",
    "rsqrt",
    "square",
}
MERGE_OPS = {
    "cat",
    "concat",
    "stack",
    "matmul",
    "bmm",
    "einsum",
    "mm",
    "__matmul__",
    "where",
    "lerp",
}


def flat_tensors(obj):
    out = []
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            out += flat_tensors(o)
    elif isinstance(obj, dict):
        for o in obj.values():
            out += flat_tensors(o)
    return out


_RENAME = {
    "radd": "add",
    "rmul": "mul",
    "rsub": "sub",
    "rtruediv": "div",
    "truediv": "div",
    "iadd": "add",
    "imul": "mul",
    "rmatmul": "matmul",
}


def clean_name(func):
    n = getattr(func, "__name__", None) or str(func)
    if n == "__getitem__":
        return "getitem"
    if n.startswith("__") and n.endswith("__"):
        n = n[2:-2]
    return _RENAME.get(n, n)


def caller_name():
    """Name of the user-code method that issued a torch call (forward, or a helper such as composition)."""
    f = sys._getframe(2)
    while f is not None and ("/torch/" in f.f_code.co_filename or f.f_code.co_filename == __file__):
        f = f.f_back
    return f.f_code.co_name if f is not None else "forward"


class Event:
    def __init__(self, kind, name, cls=None, module=None, path=""):
        self.kind = kind  # module | op
        self.name = name
        self.cls = cls
        self.module = module
        self.path = path
        self.children: list[Event] = []
        self.ins: list[int] = []
        self.outs: list[int] = []
        self.in_shapes: list[tuple] = []
        self.out_shapes: list[tuple] = []
        self.data = False
        self.fn = "forward"
        self.args: list = []
        self.kwargs: dict = {}


class Recorder(TorchFunctionMode):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.keep = []
        self.root = None
        self.stack: list[Event] = []
        self.paths = {m: n for n, m in net.named_modules()}
        self.leaf = {m: len(list(m.children())) == 0 for m in net.modules()}
        self.handles = []
        for m in net.modules():
            self.handles.append(m.register_forward_pre_hook(self._pre, with_kwargs=True))
            self.handles.append(m.register_forward_hook(self._post, with_kwargs=True))

    def _pre(self, m, args, kwargs):
        ev = Event(
            "module",
            self.paths.get(m, "?").split(".")[-1] or type(m).__name__,
            cls=type(m).__name__,
            module=m,
            path=self.paths.get(m, "?"),
        )
        ts = flat_tensors(args) + flat_tensors(kwargs)
        self.keep += ts
        ev.ins = [id(t) for t in ts]
        ev.in_shapes = [tuple(t.shape) for t in ts]
        if self.stack:
            self.stack[-1].children.append(ev)
        else:
            self.root = ev
        self.stack.append(ev)

    def _post(self, m, args, kwargs, out):
        ev = self.stack.pop()
        ts = flat_tensors(out)
        self.keep += ts
        ev.outs = [id(t) for t in ts]
        ev.out_shapes = [tuple(t.shape) for t in ts]

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if not self.stack or self.leaf.get(self.stack[-1].module, False):
            return out
        outs = flat_tensors(out)
        if not outs:
            return out
        name = clean_name(func)
        ins = flat_tensors(args) + flat_tensors(kwargs)
        self.keep += ins + outs
        ev = Event("op", name)
        ev.fn = caller_name()
        ev.args = [a for a in args if isinstance(a, (int, float, tuple, str))]
        ev.kwargs = {k: v for k, v in kwargs.items() if isinstance(v, (int, float, tuple, str))}
        ev.ins = [id(t) for t in ins]
        ev.outs = [id(t) for t in outs]
        ev.in_shapes = [tuple(t.shape) for t in ins]
        ev.out_shapes = [tuple(t.shape) for t in outs]
        self.stack[-1].children.append(ev)
        return out

    def close(self):
        for h in self.handles:
            h.remove()


def trace(net, seq_len=200):
    net.eval()
    rec = Recorder(net)
    x = torch.randn(1, 4, seq_len)
    rec.keep.append(x)
    with torch.no_grad(), rec:
        net(x)
    rec.close()
    mark_data(rec.root, {id(x)})
    return rec.root, id(x)


def mark_data(ev, data_ids):
    """Mark which events depend on the network input (constants such as position grids are not drawn)."""
    for c in ev.children:
        if any(i in data_ids for i in c.ins):
            c.data = True
            data_ids.update(c.outs)
            if c.kind == "module":
                mark_data(c, set(data_ids))
        elif c.kind == "module":
            mark_data(c, set(data_ids))
    ev.data = True


# ------------------------------------------------------------------------------------------------------------------------------ classification and description
def leaf_modules(m):
    return [x for x in m.modules() if len(list(x.children())) == 0]


def module_kind(m):
    cs = [type(x).__name__ for x in m.modules()]
    low = " ".join(cs).lower()
    if any(k in low for k in ("multiheadattention", "attention", "attn", "transformer")):
        return "attn"
    if any(k in low for k in ("lstm", "gru", "rnn")):
        return "rnn"
    if any(isinstance(x, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)) for x in m.modules()):
        return "conv"
    if any(isinstance(x, nn.Linear) for x in m.modules()):
        return "mlp"
    if any("pool" in c.lower() for c in cs):
        return "pool"
    return "norm"


def leaf_desc(m):
    t = type(m).__name__
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        k = (
            m.kernel_size[0]
            if m.kernel_size[0] == m.kernel_size[-1]
            else "x".join(map(str, m.kernel_size))
        )
        s = f"{t.replace('Conv', 'Conv')} {m.in_channels}→{m.out_channels}, k={k}"
        if m.groups > 1:
            s += f", groups={m.groups}"
        if m.dilation[0] > 1:
            s += f", dil={m.dilation[0]}"
        if m.stride[0] > 1:
            s += f", stride={m.stride[0]}"
        return s
    if isinstance(m, nn.ConvTranspose1d):
        return f"ConvTranspose1d {m.in_channels}→{m.out_channels}, k={m.kernel_size[0]}, stride={m.stride[0]}"
    if isinstance(m, nn.Linear):
        return f"Linear {m.in_features}→{m.out_features}"
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        return f"BatchNorm({m.num_features})"
    if isinstance(m, nn.LayerNorm):
        return f"LayerNorm({m.normalized_shape[0]})"
    if isinstance(m, nn.GroupNorm):
        return f"GroupNorm({m.num_groups},{m.num_channels})"
    if isinstance(m, nn.Dropout):
        return f"Dropout({m.p:g})"
    if isinstance(m, nn.MultiheadAttention):
        return f"MultiheadAttention(d={m.embed_dim}, heads={m.num_heads})"
    if isinstance(m, (nn.LSTM, nn.GRU)):
        return f"{t}(hidden={m.hidden_size}, layers={m.num_layers}{', bidirectional' if m.bidirectional else ''})"
    if isinstance(m, (nn.MaxPool1d, nn.AvgPool1d)):
        k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
        return f"{t}(k={k})"
    if isinstance(m, (nn.AdaptiveAvgPool1d, nn.AdaptiveMaxPool1d)):
        return f"{t}({m.output_size})"
    if isinstance(m, (nn.Embedding,)):
        return f"Embedding({m.num_embeddings},{m.embedding_dim})"
    return t


def leaf_short(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        k = m.kernel_size[-1]
        s = f"Conv {m.in_channels}→{m.out_channels} k{k}"
        if m.groups > 1:
            s += f" g{m.groups}"
        if m.dilation[-1] > 1:
            s += f" d{m.dilation[-1]}"
        if m.stride[-1] > 1:
            s += f" s{m.stride[-1]}"
        return s
    if isinstance(m, nn.Linear):
        return f"Linear {m.in_features}→{m.out_features}"
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        return "BN"
    if isinstance(m, nn.LayerNorm):
        return "LN"
    if isinstance(m, nn.Dropout):
        return f"Drop {m.p:g}"
    return leaf_desc(m)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def short_shape(sh):
    if not sh:
        return ""
    s = tuple(sh[1:]) if len(sh) > 1 else tuple(sh)
    return "(" + ", ".join(str(x) for x in s) + ")"


def run_length(items):
    out = []
    for it in items:
        if out and out[-1][0] == it:
            out[-1][1] += 1
        else:
            out.append([it, 1])
    return out


def describe_children(ev, limit=9):
    """One-line contents of a module event: its layers in execution order, run-length compressed."""
    parts = []
    for c in ev.children:
        if not c.data and c.kind == "op":
            continue
        if c.kind == "module":
            if len(list(c.module.children())) == 0:
                parts.append(leaf_short(c.module))
            else:
                parts.append(c.cls)
        elif c.name not in TRANSPARENT and c.name not in ARITH:
            parts.append(c.name)
    rl = run_length(parts)
    txt = [f"{n}×{p}" if n > 1 else p for p, n in rl]
    if len(txt) > limit:
        txt = txt[:limit] + ["…"]
    return " · ".join(txt)


# ------------------------------------------------------------------------------------------------------------------------------ flow extraction
class Item:
    """One box of a diagram: a module, a run of functional ops, or a repeated block."""

    def __init__(self, kind, title, events, sub=""):
        self.kind = kind
        self.title = title
        self.sub = sub
        self.events = events
        self.count = 1
        self.body: list[Event] | None = None
        self.params = 0
        self.out_shape: tuple = ()
        self.short = title
        self.tied = False
        self.openable = False
        self.attrs = ""
        self.outs: set[int] = set()


class Flow:
    def __init__(self, items, edges, raw=None):
        self.items = items
        self.edges = edges
        self.raw = (
            raw or items
        )  # boxes before the display cap merged neighbours; the detail panel opens one of these


PURE_ACT = {
    "silu",
    "relu",
    "gelu",
    "elu",
    "leaky_relu",
    "mish",
    "softplus",
    "dropout",
    "layer_norm",
    "batch_norm",
    "group_norm",
    "tanh",
    "hardswish",
    "relu6",
    "selu",
}
PRETTY = {
    "amax": "max",
    "cat": "concat",
    "topk": "top-k",
    "logsumexp": "logsumexp",
    "einsum": "einsum",
    "bmm": "bmm",
    "matmul": "matmul",
    "mean": "mean",
    "sum": "sum",
    "softmax": "softmax",
    "interpolate": "upsample",
    "mul": "multiply",
    "add": "add",
    "sub": "sub",
    "div": "div",
    "where": "where",
    "sigmoid": "sigmoid",
    "std": "std",
}


def struct_sig(m):
    parts = []
    for x in m.modules():
        t = type(x).__name__
        if isinstance(x, (nn.Conv1d, nn.Conv2d)):
            t += f"{x.kernel_size}{x.dilation}{x.stride}g{'1' if x.groups == 1 else 'n'}"
        parts.append(t)
    return "|".join(parts)


def sig(ev):
    return ("m", struct_sig(ev.module)) if ev.kind == "module" else ("o", ev.name)


def collect_alias(ev, alias):
    for c in ev.children:
        if c.kind == "op" and c.name in TRANSPARENT and c.ins:
            for o in c.outs:
                alias[o] = c.ins[0]
        if c.kind == "module":
            collect_alias(c, alias)


def resolve(i, alias):
    seen = 0
    while i in alias and seen < 50:
        i = alias[i]
        seen += 1
    return i


def walk(ev):
    for c in ev.children:
        yield c
        if c.kind == "module":
            yield from walk(c)


def uniq_params(events):
    mods = {}
    for ev in events:
        if ev.kind == "module":
            for x in ev.module.modules():
                mods[id(x)] = x
    return sum(p.numel() for x in mods.values() for p in x.parameters(recurse=False))


def conv_attrs(ev):
    """(kernel, dilation, groups) of a conv leaf module or a functional conv1d call, else None."""
    if ev.kind == "module" and isinstance(ev.module, (nn.Conv1d, nn.Conv2d)):
        m = ev.module
        return m.kernel_size[-1], m.dilation[-1], m.groups
    if ev.kind == "op" and ev.name in ("conv1d", "conv2d") and len(ev.in_shapes) > 1:
        dil = ev.kwargs.get("dilation", ev.args[2] if len(ev.args) > 2 else 1)
        grp = ev.kwargs.get("groups", ev.args[3] if len(ev.args) > 3 else 1)
        dil = dil[-1] if isinstance(dil, tuple) else dil
        return ev.in_shapes[1][-1], dil, grp
    return None


def ops_kind(names):
    n = set(names)
    if n and n <= {"conv1d", "conv2d"}:
        return "conv"
    if "softmax" in n and n & {"einsum", "bmm", "matmul", "sum", "mul"}:
        return "attn"
    if n & POOL_OPS:
        return "pool"
    if n & (MERGE_OPS | ARITH):
        return "merge"
    return "norm"


def module_title(ev):
    m = ev.module
    if len(list(m.children())) == 0:
        return leaf_desc(m)
    if ev.cls in ("Sequential", "ModuleList"):
        return ev.name if not ev.name.isdigit() else trunc(describe_children(ev, limit=4), 58)
    return (
        ev.cls
        if ev.name.isdigit() or ev.name.lower() == ev.cls.lower()
        else f"{ev.cls} ({ev.name})"
    )


def first_leaf(ev):
    for c in ev.children:
        if c.kind == "module":
            if len(list(c.module.children())) == 0:
                return leaf_desc(c.module)
            r = first_leaf(c)
            if r:
                return r
    return ""


def fmt_params(n):
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1000:
        return f"{n / 1e3:.0f}k"
    return str(n)


def inline_seq(events):
    """Replace plain nn.Sequential wrappers by their children so a block's own layers show up as boxes."""
    out = []
    for e in events:
        if e.kind == "module" and e.cls == "Sequential" and e.children:
            out += inline_seq(e.children)
        else:
            out.append(e)
    return out


def make_module_item(ev):
    m = ev.module
    leaf = len(list(m.children())) == 0
    unnamed_seq = ev.cls == "Sequential" and ev.name.isdigit()
    it = Item(
        module_kind(m),
        module_title(ev),
        [ev],
        "" if (leaf or unnamed_seq) else describe_children(ev),
    )
    it.params = uniq_params([ev])
    it.out_shape = ev.out_shapes[0] if ev.out_shapes else ()
    it.outs = set(ev.outs)
    it.openable = (not leaf) and any(c.data for c in ev.children)
    fl = "" if leaf else first_leaf(ev)
    it.short = (
        it.title
        if leaf
        else (f"{it.title} · {fl}" if it.title in (ev.name, "Sequential") and fl else it.title)
    )
    return it


def make_ops_item(evs):
    names = [e.name for e in evs if e.name not in TRANSPARENT]
    uniq = list(dict.fromkeys(names))
    pretty = [PRETTY.get(n, n.replace("_", " ")) for n in uniq]
    fn = evs[0].fn
    for e_ in evs:
        if e_.name in ("conv1d", "conv2d") and len(uniq) == 1:
            ca_ = conv_attrs(e_)
            pretty = [
                f"Conv {e_.in_shapes[0][1]}→{e_.out_shapes[0][1]} k{ca_[0]}"
                + (f" g{ca_[2]}" if ca_[2] > 1 else "")
                + " (functional)"
            ]
        if "pool1d" in e_.name and not e_.name.startswith("adaptive"):
            k_ = e_.kwargs.get("kernel_size", e_.args[0] if e_.args else "")
            pretty = (
                [f"{'avg' if 'avg' in e_.name else 'max'}-pool k={k_}"]
                if len(uniq) == 1
                else pretty
            )
    if fn != "forward" and len(pretty) > 2:
        title = f"{fn}()"
        sub = ", ".join(pretty[:7]) + (" …" if len(pretty) > 7 else "")
    elif fn != "forward":
        title = ", ".join(pretty)
        sub = f"in {fn}()"
    else:
        title = ", ".join(pretty[:5]) + (" …" if len(pretty) > 5 else "")
        sub = ""
    it = Item(ops_kind(uniq), title, list(evs), sub)
    it.out_shape = evs[-1].out_shapes[0] if evs[-1].out_shapes else ()
    for e in evs:
        it.outs |= set(e.outs)
    it.short = title
    return it


def block_title(body):
    mods = [e for e in body if e.kind == "module"]
    if len(body) == 1 and mods:
        e = mods[0]
        return e.cls if e.cls not in ("Sequential", "ModuleList") else "block"
    return "block"


def make_repeat_item(seq, i, p, r):
    body = seq[i : i + p]
    its = [seq[i + k * p : i + (k + 1) * p] for k in range(r)]
    it = Item("norm", "", seq[i : i + p * r])
    it.count = r
    it.body = body
    kinds = [make_module_item(e).kind if e.kind == "module" else None for e in body]
    kinds = [k for k in kinds if k]
    pri = ["attn", "rnn", "conv", "mlp", "pool", "norm"]
    it.kind = next(
        (k for k in pri if k in kinds), ops_kind([e.name for e in body if e.kind == "op"])
    )
    ttl = block_title(body)
    if len(body) == 1 and body[0].kind == "module" and body[0].cls in ("Sequential", "ModuleList"):
        names = [its[0][0].name, its[-1][0].name]
        ttl = f"{names[0]}…{names[1]}" if names[0] != names[1] else "block"
    chans = []
    for itr in its:
        e = itr[-1]
        if e.out_shapes and len(e.out_shapes[0]) == 3:
            chans.append(e.out_shapes[0][1])
    chans = list(dict.fromkeys(chans))
    attrs = []
    if chans:
        attrs.append("c=" + "→".join(map(str, chans[:5])))
    ks, ds = [], []
    for itr in its:
        ca = [
            conv_attrs(e) for ev in itr for e in ([ev, *walk(ev)] if ev.kind == "module" else [ev])
        ]
        ca = [c for c in ca if c]
        if ca:
            ks.append(max(c[0] for c in ca))
            ds.append(max(c[1] for c in ca))
    if ks:
        attrs.append("k=" + ",".join(map(str, dict.fromkeys(ks))))
    if ds and max(ds) > 1:
        attrs.append("dil=" + ",".join(map(str, dict.fromkeys(ds))))
    ids = [{id(x) for ev in itr if ev.kind == "module" for x in ev.module.modules()} for itr in its]
    it.tied = r > 1 and bool(ids[0] & ids[1])
    it.title = f"{r}× {ttl}" + (" (shared weights)" if it.tied else "")
    it.attrs = "  ".join(attrs)
    desc = (
        describe_children(body[0])
        if len(body) == 1 and body[0].kind == "module"
        else " · ".join(
            (
                leaf_short(e.module)
                if e.kind == "module" and not list(e.module.children())
                else (e.cls if e.kind == "module" else e.name)
            )
            for e in body
            if e.kind == "module" or e.name not in TRANSPARENT
        )
    )
    it.sub = (it.attrs + "\n" + desc).strip() if it.attrs else desc
    it.params = uniq_params(it.events)
    it.out_shape = it.events[-1].out_shapes[0] if it.events[-1].out_shapes else ()
    for e in it.events:
        it.outs |= set(e.outs)
    it.openable = True
    it.short = f"{it.title}  {it.attrs}".strip()
    return it


def chained(seq, i, p, alias):
    prod = {resolve(o, alias) for e in seq[i : i + p] for o in e.outs}
    return any(resolve(j, alias) in prod for e in seq[i + p : i + 2 * p] for j in e.ins)


def find_repeats(seq, alias):
    sigs = [sig(e) for e in seq]
    n, out, i = len(sigs), [], 0
    while i < n:
        best = None
        for p in range(1, (n - i) // 2 + 1):
            blk = sigs[i : i + p]
            r = 1
            while sigs[i + r * p : i + (r + 1) * p] == blk:
                r += 1
            if (
                r >= 2
                and any(s[0] == "m" for s in blk)
                and chained(seq, i, p, alias)
                and (best is None or p * r > best[0] * best[1])
            ):
                best = (p, r)
        if best:
            out.append(("rep", i, *best))
            i += best[0] * best[1]
        else:
            out.append(("one", i))
            i += 1
    return out


def build_items(events, alias):
    seq = [e for e in events if e.data and not (e.kind == "op" and e.name in TRANSPARENT)]
    segs = find_repeats(seq, alias)
    items: list[Item] = []
    run: list[Event] = []

    def flush():
        nonlocal run
        if run:
            items.append(make_ops_item(run))
            run = []

    for sg in segs:
        if sg[0] == "rep":
            flush()
            items.append(make_repeat_item(seq, sg[1], sg[2], sg[3]))
            continue
        ev = seq[sg[1]]
        if ev.kind == "module":
            flush()
            if (
                items
                and ev.module is not None
                and isinstance(ev.module, NORM_LEAVES)
                and items[-1].count == 1
                and items[-1].kind in ("conv", "mlp", "norm")
                and ev.ins
                and resolve(ev.ins[0], alias) in {resolve(o, alias) for o in items[-1].outs}
                and items[-1].events[0].kind == "module"
                and not list(items[-1].events[0].module.children())
            ):
                it = items[-1]
                it.events.append(ev)
                it.outs |= set(ev.outs)
                it.sub = (it.sub + " · " if it.sub else "") + leaf_short(ev.module)
                it.out_shape = ev.out_shapes[0] if ev.out_shapes else it.out_shape
                it.short = f"{it.title} · {it.sub}"
                continue
            items.append(make_module_item(ev))
            continue
        prev_outs = items[-1].outs if items and not run else set()
        if (
            ev.name in PURE_ACT
            and ev.ins
            and resolve(ev.ins[0], alias) in {resolve(o, alias) for o in prev_outs} | prev_outs
            and not run
        ):
            it = items[-1]
            it.events.append(ev)
            it.outs |= set(ev.outs)
            it.sub = (it.sub + f" · {ev.name}") if it.sub else it.sub
            continue
        if run and (run[0].fn != ev.fn or len(run) >= 10):
            flush()
        run.append(ev)
    flush()
    return items


NORM_LEAVES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.Dropout,
    nn.SiLU,
    nn.GELU,
    nn.ReLU,
    nn.ELU,
    nn.LeakyReLU,
    nn.Tanh,
    nn.Sigmoid,
    nn.Softplus,
    nn.Mish,
    nn.Hardswish,
    nn.ReLU6,
    nn.SELU,
    nn.Identity,
)


def merge_two(a, b):
    it = Item(
        a.kind if a.params >= b.params else b.kind,
        a.title if a.title == b.title else f"{a.title} + {b.title}",
        a.events + b.events,
    )
    it.params = a.params + b.params
    it.out_shape = b.out_shape
    it.outs = a.outs | b.outs
    it.short = it.title
    it.sub = ""
    return it


def shrink(items, cap):
    items = list(items)
    while len(items) > cap:
        costs = []
        for k in range(len(items) - 1):
            a, b = items[k], items[k + 1]
            c = (
                a.params
                + b.params
                + (0 if a.kind == b.kind else 3e5)
                + (2e6 if (a.count > 1 or b.count > 1) else 0)
            )
            costs.append(c)
        k = min(range(len(costs)), key=costs.__getitem__)
        items[k : k + 2] = [merge_two(items[k], items[k + 1])]
    return items


def compute_edges(items, in_ids, out_ids, alias):
    prod = {resolve(i, alias): 0 for i in in_ids}
    edges: dict[tuple[int, int], tuple] = {}
    for k, it in enumerate(items):
        if k == 0 or k == len(items) - 1:
            continue
        for ev in it.events:
            for j, i in enumerate(ev.ins):
                src = prod.get(resolve(i, alias))
                if src is not None and src != k and (src, k) not in edges:
                    edges[(src, k)] = ev.in_shapes[j]
            for o in ev.outs:
                prod[o] = k
    last = len(items) - 1
    for o in out_ids:
        src = prod.get(resolve(o, alias))
        if src is not None and src != last:
            edges[(src, last)] = items[src].out_shape
    return edges


def make_flow(
    events, in_ids, out_ids, alias, in_label, out_label, cap=12, in_shape=(), end_kind="head"
):
    raw = build_items(events, alias)
    core = shrink(raw, cap - 2)
    a = Item("input", in_label, [], "")
    a.out_shape = in_shape
    a.short = f"{in_label}  {short_shape((1, *in_shape)) if in_shape else ''}".strip()
    if end_kind == "head" and core and core[-1].kind == "mlp":
        core[-1].kind = "head"
    z = Item(end_kind, out_label, [], "")
    z.short = out_label
    items = [a, *core, z]
    return Flow(items, compute_edges(items, in_ids, out_ids, alias), raw)


def open_item(it):
    """Events inside a box, for the detail panel: the body of one repeated iteration, or the children of one module."""
    if it.body is not None:
        b = it.body
        if len(b) == 1 and b[0].kind == "module" and any(c.data for c in b[0].children):
            return b[0].children, b[0].ins, b[0].outs
        return b, b[0].ins, b[-1].outs
    if len(it.events) == 1 and it.events[0].kind == "module" and it.openable:
        e = it.events[0]
        return e.children, e.ins, e.outs
    return None


def custom_cls(m):
    return not type(m).__module__.startswith("torch")


def pick_block(root):
    """The user-defined block class with the most parameters across the network (e.g. an inverted-residual block)."""
    groups: dict = {}
    for e in walk(root):
        if e.kind == "module" and custom_cls(e.module) and any(c.data for c in e.children):
            g = groups.setdefault(
                type(e.module), {"mods": {}, "first": e, "calls": 0, "widths": [], "dils": []}
            )
            g["mods"][id(e.module)] = e.module
            g["calls"] += 1
            if e.out_shapes and len(e.out_shapes[0]) == 3:
                g["widths"].append(e.out_shapes[0][1])
            g["dils"].extend(c[1] for c in (conv_attrs(x) for x in walk(e)) if c)
    if not groups:
        return None
    cls = max(groups, key=lambda c: sum(nparams(m) for m in groups[c]["mods"].values()))
    return cls, groups[cls]


def make_detail(root, flow, alias):
    """Open the most parameter-heavy repeated unit: a user-defined block class, else the biggest repeated loop body or composite box."""
    kw = {"cap": 13, "end_kind": "input"}
    pick = pick_block(root)
    if pick:
        cls, g = pick
        e0 = g["first"]
        f2 = make_flow(
            inline_seq(e0.children),
            e0.ins,
            e0.outs,
            alias,
            "block input",
            "block output",
            in_shape=tuple(e0.in_shapes[0][1:]) if e0.in_shapes else (),
            **kw,
        )
        widths = list(dict.fromkeys(g["widths"]))
        attrs = (
            (
                "output channels "
                + ", ".join(map(str, widths[:6]))
                + (" …" if len(widths) > 6 else "")
            )
            if widths
            else ""
        )
        dils = sorted(set(g["dils"]))
        if dils and dils[-1] > 1:
            attrs += "   dilations used: " + ", ".join(map(str, dils))
        return {
            "flow": f2,
            "title": cls.__name__,
            "n": g["calls"],
            "distinct": len(g["mods"]),
            "params": nparams(e0.module),
            "attrs": attrs,
            "tied": g["calls"] > len(g["mods"]),
            "scope": "in this copy",
        }
    reps = [it for it in flow.raw if it.count > 1]
    cands = reps or [it for it in flow.raw if it.openable and it.kind in ("conv", "attn", "rnn")]
    if not cands:
        return None
    it = max(cands, key=lambda j: j.params)
    opened = open_item(it)
    if opened is None:
        return None
    evs, ins, outs = opened
    f2 = make_flow(
        inline_seq(evs),
        ins,
        outs,
        alias,
        "block input",
        "block output",
        in_shape=tuple(evs[0].in_shapes[0][1:]) if evs and evs[0].in_shapes else (),
        **kw,
    )
    title = it.title.split("× ", 1)[1] if "× " in it.title[:5] else it.title
    return {
        "flow": f2,
        "title": title.replace(" (shared weights)", ""),
        "n": it.count,
        "distinct": 1 if it.tied else it.count,
        "params": it.params // max(1, 1 if it.tied else it.count),
        "attrs": it.attrs,
        "tied": it.tied,
        "scope": "in all copies together" if it.tied else "in this copy",
    }


def skip_kinds(root, alias):
    """Skips found in the trace: an add/concat consuming a tensor that an earlier layer of the same module also consumed."""
    found = set()
    for m in [root, *[e for e in walk(root) if e.kind == "module"]]:
        seen: set[int] = set()
        for c in m.children:
            if not c.data:
                continue
            ids = {resolve(i, alias) for i in c.ins}
            if c.kind == "op" and len(c.ins) >= 2 and ids & seen:
                if c.name == "add":
                    found.add("residual add")
                elif c.name in ("cat", "concat"):
                    found.add("skip concat")
            seen |= ids
    return found


def arch_tags(root, flows, alias):
    tags = []
    evs = list(walk(root))
    mods = [e.module for e in evs if e.kind == "module"]
    ops = {e.name for e in evs if e.kind == "op"}
    if (
        ops & {"interpolate", "upsample"} or any(isinstance(m, nn.ConvTranspose1d) for m in mods)
    ) and "cat" in ops:
        tags.append("U-Net")
    ca = [c for c in (conv_attrs(e) for e in evs) if c]
    if any(c[1] > 1 for c in ca):
        tags.append("dilated")
    if any(c[2] > 1 for c in ca):
        tags.append("grouped conv")
    if any(isinstance(m, nn.Conv2d) for m in mods):
        tags.append("2-D pair map")
    if any(isinstance(m, nn.MultiheadAttention) for m in mods) or (
        "softmax" in ops and ops & {"einsum", "bmm", "matmul", "sum", "mul"}
    ):
        tags.append("attention")
    if "sigmoid" in ops and "mean" in ops and "mul" in ops:
        tags.append("SE gating")
    if "topk" in ops:
        tags.append("top-k pool")
    if any(isinstance(m, (nn.LSTM, nn.GRU)) for m in mods):
        tags.append("recurrent")
    if any(it.tied for f in flows for it in f.items):
        tags.append("shared weights")
    tags += sorted(skip_kinds(root, alias))
    return tags


# ------------------------------------------------------------------------------------------------------------------------------ drawing
FS = 11
SKIP = "#9a2f2f"
INK = "#1a1d21"
GREY = "#5b636b"
EDGE_INK = "#2b2f36"


def kind_fc(kind):
    return COL.get(kind, COL["norm"])


def kind_tc(kind):
    return INK if kind in ("merge", "input") else "white"


def trunc(s, n):
    return s if len(s) <= n else s[: max(1, n - 1)].rstrip() + "…"


def layout(n, y_top, y_bottom, gap=0.2, max_pitch=0.72):
    pitch = min(max_pitch, (y_bottom - y_top + gap) / max(1, n))
    return pitch, pitch - gap


def draw_flow(
    ax, flow, x, y0, w, pitch, bh, fs=FS, compact=False, shapes=True, lane0=0.3, lane_w=0.27
):
    from matplotlib.path import Path as MPath

    items, edges = flow.items, flow.edges
    n = len(items)
    cx = x + w / 2
    tops = [y0 + k * pitch for k in range(n)]
    cpl = int((w - 0.25) / (fs * 0.0086))
    for k, it in enumerate(items):
        y = tops[k]
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                bh,
                boxstyle="round,pad=0,rounding_size=0.07",
                fc=kind_fc(it.kind),
                ec="none",
                zorder=3,
            )
        )
        tc = kind_tc(it.kind)
        ptag = fmt_params(it.params) if (it.params and 0 < k < n - 1) else ""
        two = (not compact) and bool(it.sub) and bh >= 0.37
        stag = (
            "×".join(map(str, it.out_shape[1:]))
            if compact and 0 < k < n - 1 and 2 <= len(it.out_shape) <= 4
            else ""
        )
        room = cpl - (9 if ptag else 0) - (11 if stag else 0)
        if two:
            ax.text(
                x + 0.1,
                y + bh * 0.29,
                trunc(it.title, room),
                color=tc,
                fontsize=fs,
                fontweight="bold",
                va="center",
                ha="left",
                zorder=4,
            )
            ax.text(
                x + 0.1,
                y + bh * 0.73,
                trunc(it.sub.replace("\n", "  ·  "), cpl),
                color=tc,
                fontsize=fs,
                va="center",
                ha="left",
                zorder=4,
                alpha=0.92,
            )
        else:
            label = (
                it.short
                if compact
                else (it.title + ("  ·  " + it.sub.replace("\n", "  ·  ") if it.sub else ""))
            )
            ax.text(
                x + 0.1,
                y + bh / 2,
                trunc(label, room),
                color=tc,
                fontsize=fs,
                fontweight="normal" if compact else "bold",
                va="center",
                ha="left",
                zorder=4,
            )
        if ptag:
            ax.text(
                x + w - 0.1,
                y + (bh * 0.29 if two else bh / 2),
                ptag,
                color=tc,
                fontsize=fs,
                va="center",
                ha="right",
                zorder=4,
            )
        if stag:
            ax.text(
                x + w - 0.85,
                y + bh / 2,
                stag,
                color=tc,
                fontsize=fs,
                va="center",
                ha="right",
                zorder=4,
                alpha=0.75,
            )
    for k in range(n - 1):
        if (k, k + 1) in edges:
            ax.add_patch(
                FancyArrowPatch(
                    (cx, tops[k] + bh),
                    (cx, tops[k + 1]),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    lw=1.4,
                    color=EDGE_INK,
                    zorder=2,
                    shrinkA=0,
                    shrinkB=0,
                )
            )
            if shapes:
                ax.text(
                    cx + 0.12,
                    tops[k] + bh + (pitch - bh) / 2,
                    short_shape(edges[(k, k + 1)]),
                    fontsize=fs,
                    color=GREY,
                    style="italic",
                    va="center",
                    ha="left",
                    zorder=5,
                    bbox={"fc": "white", "ec": "none", "pad": 0.4},
                )
    skips = sorted((s, d) for (s, d) in edges if d != s + 1)
    lanes: list[list[tuple[int, int]]] = []
    for s, d in skips:
        lane = None
        for ln, occ in enumerate(lanes):
            if all(d < a or s > b for a, b in occ):
                occ.append((s, d))
                lane = ln
                break
        if lane is None:
            lanes.append([(s, d)])
            lane = len(lanes) - 1
        off = lane0 + lane_w * min(lane, 5)
        xs = x + w
        y_s, y_d = tops[s] + bh / 2, tops[d] + bh / 2
        verts = [(xs, y_s), (xs + off * 1.4, y_s), (xs + off * 1.4, y_d), (xs + 0.02, y_d)]
        path = MPath(verts, [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
        ax.add_patch(
            FancyArrowPatch(
                path=path,
                arrowstyle="-|>",
                mutation_scale=11,
                lw=1.5 if not compact else 1.2,
                color=SKIP,
                zorder=1,
                alpha=0.9,
                fc="none",
            )
        )
    return tops


def wrap_lines(txt, width, max_lines):
    lines = textwrap.wrap(txt, width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = trunc(lines[-1] + " …", width)
    return lines


def legend(
    ax,
    x,
    y,
    fs=FS,
    kinds=("conv", "norm", "attn", "rnn", "pool", "merge", "mlp", "head"),
    extra=True,
):
    cx = x
    for k in kinds:
        ax.add_patch(Rectangle((cx, y - 0.09), 0.26, 0.18, fc=kind_fc(k), ec="none"))
        lab = KIND_LABEL[k]
        ax.text(cx + 0.34, y, lab, fontsize=fs, va="center", color=INK)
        cx += 0.42 + len(lab) * fs * 0.0079
    if extra:
        ax.add_patch(
            FancyArrowPatch(
                (cx, y), (cx + 0.45, y), arrowstyle="-|>", mutation_scale=11, lw=1.5, color=SKIP
            )
        )
        ax.text(cx + 0.55, y, "skip / branch", fontsize=fs, va="center", color=INK)
    return cx


def recipe_text(tc):
    lr = f"{tc['lr']:g}"
    return f"{tc['optimizer']} · lr {lr} · wd {tc['weight_decay']:g} · batch {tc['batch_size']} · {tc['epochs']} epochs · {tc['lr_schedule']}"


def draw_entry(r, path):
    e, w = r["entry"], r["w"]
    fig = plt.figure(figsize=(16, 11), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(11, 0)
    ax.axis("off")
    ax.text(0.4, 0.42, e["id"], fontsize=22, fontweight="bold", color=INK, va="center")
    ax.text(
        15.6,
        0.42,
        f"menu-v1 entry {r['idx'] + 1} of 20",
        fontsize=13,
        color=GREY,
        va="center",
        ha="right",
    )
    tg = "  ·  ".join(r["tags"]) if r["tags"] else "plain conv stack"
    ax.text(
        0.4,
        0.86,
        f"origin search run: {e['cell']}   ·   search corpus: {e['family']}   ·   {tg}",
        fontsize=13,
        color=GREY,
        va="center",
    )
    lines = wrap_lines("LLM's idea: " + e["idea"], 125, 3)
    for k, ln in enumerate(lines):
        ax.text(0.4, 1.22 + 0.3 * k, ln, fontsize=14, style="italic", color=INK, va="center")
    y_s = 1.22 + 0.3 * len(lines) + 0.18
    wtxt = (
        f"30k stack weight {w['weight']:.3f}  (rank {r['rank']} of 20)"
        if w and w["weight"] > 1e-9
        else "30k stack weight 0  (not selected by the stack)"
    )
    ax.text(
        0.4,
        y_s,
        f"{r['n_built']:,} params   |   search val r {e['orig_val']:.4f}   |   {wtxt}",
        fontsize=15,
        fontweight="bold",
        color=INK,
        va="center",
    )
    ax.text(
        0.4,
        y_s + 0.36,
        "training recipe: " + recipe_text(e["train_config"]),
        fontsize=13,
        color=INK,
        va="center",
    )
    y_d = y_s + 0.64
    ax.plot([0.4, 15.6], [y_d, y_d], color="#cfd4d8", lw=1)
    y_h, y_f = y_d + 0.27, y_d + 0.56
    ax.text(
        0.4,
        y_h,
        "network, in execution order (one forward on a (1, 4, 200) input)",
        fontsize=12,
        color=GREY,
        va="center",
    )
    flow = r["flow"]
    pitch, bh = layout(len(flow.items), y_f, 10.3)
    draw_flow(ax, flow, 0.4, y_f, 5.7, pitch, bh)
    det = r["detail"]
    if det:
        cp = (
            "1 instance"
            if det["n"] == 1
            else f"{det['n']} calls, shared weights"
            if det["tied"]
            else f"{det['n']} instances"
        )
        ax.text(
            8.2,
            y_h,
            trunc(
                f"inside one {det['title']}   ({cp}; {fmt_params(det['params'])} params {det['scope']})",
                78,
            ),
            fontsize=12,
            fontweight="bold",
            color=INK,
            va="center",
        )
        y_df = y_f
        if det["attrs"] or det["tied"]:
            ax.text(
                8.2,
                y_h + 0.3,
                trunc(det["attrs"] + ("   weights shared across calls" if det["tied"] else ""), 80),
                fontsize=11,
                color=GREY,
                va="center",
            )
            y_df = y_f + 0.3
        f2 = det["flow"]
        pitch2, bh2 = layout(len(f2.items), y_df, 10.3)
        draw_flow(ax, f2, 8.2, y_df, 5.4, pitch2, bh2)
    else:
        ax.text(
            8.2,
            y_f + 0.1,
            "no repeated or composite block to open: the boxes at left are the whole network.",
            fontsize=12,
            color=GREY,
            va="top",
        )
    ax.text(
        0.4,
        10.56,
        "Boxes: layers as executed (hooks on every module, tensor-flow tracing for functional ops); right-hand number = parameters. Curved arrows: a tensor consumed by a non-adjacent layer.",
        fontsize=11,
        color=GREY,
        va="center",
    )
    legend(ax, 0.4, 10.86)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def draw_top6(results, path):
    fig = plt.figure(figsize=(20, 11.25), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 20)
    ax.set_ylim(11.25, 0)
    ax.axis("off")
    ax.text(
        0.4,
        0.35,
        "The six highest-weight configurations of the 30k genomic stack",
        fontsize=20,
        fontweight="bold",
        color=INK,
        va="center",
    )
    ax.text(
        0.4,
        0.72,
        "each diagram is the executed forward pass of that LLM-written network, simplified; right-hand numbers: output shape (channels×length) and parameters",
        fontsize=12,
        color=GREY,
        va="center",
    )
    cw, ch = 6.55, 4.85
    for k, r in enumerate(results[:6]):
        cx0 = 0.35 + (k % 3) * cw
        cy0 = 1.0 + (k // 3) * (ch + 0.1)
        e, w = r["entry"], r["w"]
        ax.text(
            cx0,
            cy0 + 0.17,
            trunc(f"{k + 1}. {e['id']}", 52),
            fontsize=12.5,
            fontweight="bold",
            color=INK,
            va="center",
        )
        ax.text(
            cx0,
            cy0 + 0.47,
            f"weight {w['weight']:.3f}  ·  {fmt_params(r['n_built'])} params  ·  search val r {e['orig_val']:.3f}",
            fontsize=11,
            color=GREY,
            va="center",
        )
        f = r["flow_c"]
        pitch, bh = layout(len(f.items), cy0 + 0.8, cy0 + ch - 0.1, gap=0.13, max_pitch=0.5)
        draw_flow(
            ax,
            f,
            cx0,
            cy0 + 0.8,
            5.0,
            pitch,
            bh,
            compact=True,
            shapes=False,
            lane0=0.22,
            lane_w=0.2,
        )
    legend(ax, 0.4, 11.0)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


FAM_SHORT = {
    "mixed corpus (genomic + random + motif)": "mixed G+R+M",
    "motif CT-enriched": "motif CT-enr.",
}


def draw_overview(results, path):
    fig = plt.figure(figsize=(20, 11.25), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 20)
    ax.set_ylim(11.25, 0)
    ax.axis("off")
    ax.text(
        0.4,
        0.33,
        "Menu v1: all 20 configurations, ordered by 30k stack weight",
        fontsize=20,
        fontweight="bold",
        color=INK,
        va="center",
    )
    ax.text(
        0.4,
        0.68,
        "each strip is the network from input (left) to scalar output (right), one block per box coloured by layer kind; red arcs are skip connections",
        fontsize=12,
        color=GREY,
        va="center",
    )
    tw, th = 3.92, 2.45
    for k, r in enumerate(results):
        col, row = k % 5, k // 5
        x0, y0 = 0.35 + col * tw, 1.0 + row * th
        e, w = r["entry"], r["w"]
        pos = w["weight"] > 1e-9
        ax.add_patch(
            Rectangle(
                (x0, y0), tw - 0.15, th - 0.15, fc="#f6f7f8" if pos else "white", ec="#cfd4d8", lw=1
            )
        )
        ax.text(
            x0 + 0.1,
            y0 + 0.22,
            trunc(e["id"], 36),
            fontsize=11,
            fontweight="bold",
            color=INK,
            va="center",
        )
        ax.text(
            x0 + 0.1,
            y0 + 0.52,
            f"{FAM_SHORT.get(e['family'], e['family'])} · {fmt_params(r['n_built'])} params",
            fontsize=11,
            color=GREY,
            va="center",
        )
        ax.text(
            x0 + 0.1,
            y0 + 0.82,
            f"weight {w['weight']:.3f}" if pos else "weight 0 (dropped)",
            fontsize=11,
            color=INK if pos else GREY,
            va="center",
            fontweight="bold" if pos else "normal",
        )
        f = r["flow_c"]
        n = len(f.items)
        sw = (tw - 0.45) / n
        sx, sy, sh = x0 + 0.15, y0 + 1.5, 0.42
        for j, it in enumerate(f.items):
            ax.add_patch(
                Rectangle((sx + j * sw, sy), sw * 0.82, sh, fc=kind_fc(it.kind), ec="none")
            )
            if it.count > 1:
                ax.text(
                    sx + j * sw + sw * 0.41,
                    sy + sh / 2,
                    str(it.count),
                    color="white",
                    fontsize=9,
                    ha="center",
                    va="center",
                    fontweight="bold",
                )
        for s, d in f.edges:
            if d != s + 1:
                xa, xb = sx + s * sw + sw * 0.41, sx + d * sw + sw * 0.41
                ax.add_patch(
                    FancyArrowPatch(
                        (xa, sy),
                        (xb, sy),
                        connectionstyle=f"arc3,rad={-0.5 if d - s > 2 else -0.8}",
                        arrowstyle="-|>",
                        mutation_scale=7,
                        lw=1.0,
                        color=SKIP,
                        alpha=0.85,
                    )
                )
        ax.text(
            x0 + 0.1,
            y0 + 2.14,
            trunc(" · ".join(r["tags"][:3]) or "plain conv stack", 42),
            fontsize=11,
            color=GREY,
            va="center",
            style="italic",
        )
    legend(ax, 0.4, 11.0, extra=False)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def draw_table(results, path):
    cols = [
        ("#", 0.45, "l"),
        ("menu entry", 3.75, "l"),
        ("origin run", 2.35, "l"),
        ("corpus", 1.55, "l"),
        ("params", 0.95, "r"),
        ("optim.", 0.75, "l"),
        ("lr", 0.7, "r"),
        ("wd", 0.8, "r"),
        ("batch", 0.65, "r"),
        ("epochs", 0.95, "r"),
        ("schedule", 0.95, "l"),
        ("search val r", 1.15, "r"),
        ("30k stack weight", 1.75, "l"),
    ]
    W = sum(c[1] for c in cols) + 0.7
    fig = plt.figure(figsize=(W, 11.25), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(11.25, 0)
    ax.axis("off")
    ax.text(
        0.35,
        0.35,
        "Menu v1: the 20 configurations and the recipe each is trained with",
        fontsize=20,
        fontweight="bold",
        color=INK,
        va="center",
    )
    ax.text(
        0.35,
        0.74,
        "sorted by stack weight at 30k genomic (non-negative ElasticNet on z-scored validation predictions); epochs: budget, with the number actually trained at 30k in brackets",
        fontsize=12,
        color=GREY,
        va="center",
    )
    y0, rh = 1.1, 0.45
    xs = [0.35]
    for c in cols:
        xs.append(xs[-1] + c[1])
    ax.add_patch(Rectangle((0.3, y0), W - 0.6, rh, fc=COL["head"], ec="none"))
    for (name, cw_, al), x in zip(cols, xs):
        tx = x + 0.08 if al == "l" else x + cw_ - 0.1
        ax.text(
            tx,
            y0 + rh / 2,
            name,
            color="white",
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="left" if al == "l" else "right",
        )
    wmax = max(r["w"]["weight"] for r in results) or 1
    for k, r in enumerate(results):
        y = y0 + rh * (k + 1)
        if k % 2 == 0:
            ax.add_patch(Rectangle((0.3, y), W - 0.6, rh, fc="#f3f5f6", ec="none"))
        e, w, tc = r["entry"], r["w"], r["entry"]["train_config"]
        ep = f"{tc['epochs']} ({w['epochs']})" if w and w.get("epochs") else str(tc["epochs"])
        vals = [
            str(k + 1),
            e["id"],
            e["cell"],
            FAM_SHORT.get(e["family"], e["family"]),
            f"{r['n_built']:,}",
            tc["optimizer"],
            f"{tc['lr']:g}",
            f"{tc['weight_decay']:g}",
            str(tc["batch_size"]),
            ep,
            tc["lr_schedule"],
            f"{e['orig_val']:.4f}",
            f"{w['weight']:.3f}" if w["weight"] > 1e-9 else "0",
        ]
        for (name, cw_, al), x, v in zip(cols, xs, vals):
            tx = x + 0.08 if al == "l" else x + cw_ - 0.1
            if name == "30k stack weight":
                if w["weight"] > 1e-9:
                    ax.add_patch(
                        Rectangle(
                            (x + 0.08, y + 0.11),
                            0.9 * w["weight"] / wmax,
                            rh - 0.22,
                            fc=COL["conv"],
                            ec="none",
                        )
                    )
                    ax.text(x + 1.05, y + rh / 2, v, fontsize=11, va="center", ha="left", color=INK)
                else:
                    ax.text(tx, y + rh / 2, v, fontsize=11, va="center", ha="left", color=GREY)
                continue
            ax.text(
                tx,
                y + rh / 2,
                trunc(v, 44),
                fontsize=11,
                va="center",
                ha="left" if al == "l" else "right",
                color=INK,
            )
    n_pos = sum(1 for r in results if r["w"]["weight"] > 1e-9)
    ax.text(
        0.35,
        y0 + rh * 21 + 0.3,
        f"{n_pos} of 20 receive positive weight at 30k genomic. All 20 take the same input (one-hot 4 × 200), return one scalar and go through the same training pipeline;",
        fontsize=12,
        color=GREY,
        va="center",
    )
    ax.text(
        0.35,
        y0 + rh * 21 + 0.6,
        "the network source is what differs, and (as the columns show) the optimiser settings are identical across the menu except for the epoch budget.",
        fontsize=12,
        color=GREY,
        va="center",
    )
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------------------------ driver
WEIGHT_CANDIDATES = [
    ROOT / "outputs/justification_oct22/data_oct32.json",
    WEIGHTS,
]


def load_weights(path):
    cands = [Path(path)] if path else WEIGHT_CANDIDATES
    for c in cands:
        if c.exists():
            d = json.load(open(c))["weights30k"]
            return {m["entry"]: m for m in d["members"]}
    raise SystemExit("no stack-weights file found; pass --weights <data_oct32.json>")


def build_result(idx, e, weights):
    from albench.arch_search.validate import build_and_check

    net, verdict = build_and_check(
        e["source"], seq_len=200, n_channels=4, max_params=10**9, max_fwd_ms=1e9
    )
    if net is None:
        raise RuntimeError(f"{e['id']}: build failed: {verdict}")
    n_built = sum(p.numel() for p in net.parameters())
    root, xid = trace(net)
    alias: dict[int, int] = {}
    collect_alias(root, alias)
    kw = {
        "alias": alias,
        "in_label": "one-hot DNA, 4 × 200",
        "out_label": "scalar prediction",
        "in_shape": (4, 200),
    }
    flow = make_flow(root.children, [xid], root.outs, cap=13, **kw)
    flow_c = make_flow(root.children, [xid], root.outs, cap=10, **kw)
    detail = make_detail(root, flow, alias)
    tags = arch_tags(root, [flow] + ([detail["flow"]] if detail else []), alias)
    partial = []
    if detail is None:
        partial.append(
            "no repeated or composite block to open: the main diagram is the whole network"
        )
    for it in flow.items[1:-1]:
        names = {e_.name for e_ in it.events if e_.kind == "op" and e_.name not in TRANSPARENT}
        if it.events and all(e_.kind == "op" for e_ in it.events) and len(names) >= 6:
            partial.append(
                f"box '{trunc(it.title, 40)}' is a run of {len(it.events)} functional ops summarised by name (see source)"
            )
    w = weights.get(e["id"], {"weight": 0.0, "epochs": None})
    return {
        "idx": idx,
        "entry": e,
        "n_built": n_built,
        "w": w,
        "flow": flow,
        "flow_c": flow_c,
        "detail": detail,
        "tags": tags,
        "partial": partial,
        "verdict": verdict,
    }


def write_readme(out, results, files):
    L = ["# Menu-v1 architecture diagrams", ""]
    L.append(
        "Produced by `scripts/analysis/make_architecture_diagrams.py` (local, CPU; no cluster access)."
    )
    L.append("")
    L.append("## How the diagrams are made")
    L.append("")
    L.append(
        "Each menu entry's `source` is built with the repository builder (`albench.arch_search.validate.build_and_check`, forward-time gate disabled), its parameter count is compared with the `n_params` stored in `menu_v1.json`, and one forward pass on a (1, 4, 200) tensor is recorded with forward hooks on every module plus a `TorchFunctionMode` that captures functional ops (pooling, concat, gating, add) and which tensor each layer consumed. The boxes, the curved skip arrows and the tensor shapes come from that recording (helper-method names such as `composition()` come from the Python call stack), not from reading the code by hand. Runs of identical blocks are collapsed to `N× block`; a box shows the layers inside it in execution order. The panel at right opens the box with the most parameters (a repeated block, or a composite module)."
    )
    L.append("")
    L.append(
        "Colour code: blue convolution, grey norm/activation, orange attention, purple recurrent, green pooling/reduction, gold add/concat/gate, dark linear head. Red curved arrows mark a tensor used by a non-adjacent layer (skip, branch, FiLM-style conditioning)."
    )
    L.append("")
    L.append("## Files")
    L.append("")
    for f, d in files:
        L.append(f"- `{f}`: {d}")
    L.append("")
    L.append("## Parameter-count check (built network vs `n_params` in the menu)")
    L.append("")
    bad = [r for r in results if r["n_built"] != r["entry"]["n_params"]]
    if bad:
        for r in bad:
            L.append(
                f"- MISMATCH `{r['entry']['id']}`: built {r['n_built']:,} vs menu {r['entry']['n_params']:,}"
            )
    else:
        L.append("All 20 built networks match the menu's `n_params` exactly.")
    L.append("")
    L.append("## Entries")
    L.append("")
    L.append(
        "| rank | entry | params | 30k weight | structure found in the trace | diagram notes |"
    )
    L.append("|---|---|---|---|---|---|")
    for k, r in enumerate(results):
        e = r["entry"]
        note = "; ".join(r["partial"]) if r["partial"] else "complete"
        L.append(
            f"| {k + 1} | `{e['id']}` | {r['n_built']:,} | {r['w']['weight']:.3f} | {', '.join(r['tags']) or 'plain conv stack'} | {note} |"
        )
    L.append("")
    L.append("## Entries whose diagram is partial")
    L.append("")
    part = [r for r in results if r["partial"]]
    if part:
        for r in part:
            L.append(f"- `{r['entry']['id']}`: " + "; ".join(r["partial"]))
    else:
        L.append("None.")
    L.append("")
    L.append("## Caveats")
    L.append("")
    L.append(
        "- 'corpus' / 'search corpus' is the `family` field of the menu entry: the training corpus of the AutoResearch run that produced the configuration, not an architecture family. Architecture tags are derived from the trace (e.g. U-Net = upsample + concat; SE gating = sigmoid-gated mean; attention = softmax with a weighted sum)."
    )
    L.append(
        "- Parameters held directly by the top-level `Net` (for example a shared grouped-conv weight used through `F.conv1d`) are in the network total but not in a box's parameter tag, which counts module parameters only."
    )
    L.append(
        "- Tensor shapes drop the batch dimension. For the window model (random_rep3_r03_arch_06) the batch axis carries the 5 overlapping windows of each sequence, so shapes after the unfold are per window."
    )
    L.append(
        "- Boxes are merged to fit one slide (cap of 13 boxes in the main diagram and 10 in the grid and tiles); a merged box is named after its parts, joined with `+`, and keeps all their parameters. The right-hand panel always shows one block at layer level."
    )
    L.append(
        "- All 20 entries share one training recipe (muon, lr 0.002, wd 1e-4, batch 256, onecycle); only the epoch budget (60 / 70 / 80) differs, so what distinguishes the menu is the network source."
    )
    L.append(
        "- Stack weights are those of the 30k genomic ensemble in `data_oct32.json` (`weights30k`); the 'epochs' figure in the table is the budget with the number actually trained at 30k in brackets."
    )
    open(out / "README.md", "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="slide_figures/architectures")
    ap.add_argument(
        "--only", default="", help="substring of entry id; draws only matching per-entry figures"
    )
    ap.add_argument("--weights", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = load_weights(a.weights)
    menu = json.load(open(MENU))
    results = [build_result(i, e, weights) for i, e in enumerate(menu)]
    results.sort(key=lambda r: (-r["w"]["weight"], -r["entry"]["orig_val"]))
    for k, r in enumerate(results):
        r["rank"] = k + 1
    files = []
    for r in results:
        name = f"arch_{r['entry']['id']}.png"
        if a.only and a.only not in r["entry"]["id"]:
            continue
        draw_entry(r, out / name)
        files.append(
            (
                name,
                f"rank {r['rank']}: {r['entry']['cell']}, {r['n_built']:,} params, 30k weight {r['w']['weight']:.3f}",
            )
        )
        print(
            "drew",
            name,
            "| params",
            r["n_built"],
            "vs",
            r["entry"]["n_params"],
            "| tags",
            r["tags"],
            "| partial",
            r["partial"],
            flush=True,
        )
    if not a.only:
        draw_top6(results, out / "architectures_top6.png")
        draw_overview(results, out / "architectures_overview.png")
        draw_table(results, out / "menu_hyperparameters.png")
        files += [
            (
                "architectures_top6.png",
                "the six highest-weight entries as simplified diagrams in a 2 × 3 grid (one slide)",
            ),
            (
                "architectures_overview.png",
                "all 20 entries as small tiles: name, corpus, parameters, weight, block strip",
            ),
            (
                "menu_hyperparameters.png",
                "table of all 20: origin run, corpus, parameters, optimiser, lr, weight decay, batch size, epochs, schedule, search val r, 30k stack weight",
            ),
        ]
        write_readme(out, results, files)
        print("wrote README.md and summary figures to", out)


if __name__ == "__main__":
    main()
