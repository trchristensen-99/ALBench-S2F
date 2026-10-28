"""Family and structural tagging of candidate code (spec 4.4, 4.5, 7 check 14, 14.1 'family label gaming').

The family is decided here, independently of the proposer: one LLM call reads the code (no tools), then AST rules check the claim. The LLM family is kept only
when the rules do not contradict it; otherwise the rule-derived family wins, so a proposer cannot move a model into an empty family by naming it, and a
residual CNN cannot hide behind a one-line attention layer. When the call fails, the rules alone tag the model.
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Any

# spec 4.4: exactly one of these, or other:<name> with a one-line justification
TAXONOMY_ROWS: tuple[tuple[str, str], ...] = (
    ("cnn_plain", "Stacked convolutions with pooling, no residual paths"),
    (
        "cnn_residual",
        "Residual convolution blocks, including squeeze-and-excite and inverted-residual variants (LegNet-like)",
    ),
    ("cnn_dilated", "Residual blocks with growing dilation, little or no pooling"),
    ("cnn_multiscale", "Parallel branches with different kernel sizes"),
    ("rnn_hybrid", "Convolution stem followed by a bidirectional LSTM or GRU (DREAM-RNN-like)"),
    ("tcn_tied", "Weight-tied or recurrent dilated convolution blocks"),
    ("transformer_hybrid", "Convolution stem followed by self-attention blocks"),
    ("transformer_pure", "Attention over k-mer or patch tokens with no convolution tower"),
    ("state_space", "Structured state-space or selective-scan sequence layers"),
    ("mixer_mlp", "MLP-mixer style token and channel mixing"),
    ("unet", "Encoder-decoder with skip connections before pooling"),
    (
        "motif_grammar",
        "An explicit motif-scanning first layer with a separate interaction or spacing module",
    ),
    ("equivariant", "Layers built to be reverse-complement equivariant"),
)
TAXONOMY: tuple[str, ...] = tuple(name for name, _ in TAXONOMY_ROWS)
TAG_KEYS = (
    "family",
    "core_block",
    "uses_attention",
    "uses_recurrence",
    "normalisation",
    "activation",
    "pooling",
    "positional_info",
    "downsampling_factor",
    "justification",
)


def taxonomy_text() -> str:
    lines = ["FAMILY TAXONOMY (choose exactly one, or other:<name> with a one-line justification)"]
    lines += [f"- {name}: {desc}" for name, desc in TAXONOMY_ROWS]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- AST features


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _const(node: ast.AST | None) -> Any:
    try:
        return ast.literal_eval(node) if node is not None else None
    except Exception:  # noqa: BLE001 - non-literal expression
        return None


class _Features:
    def __init__(self, code: str) -> None:
        self.code = code
        self.lower = code.lower()
        try:
            self.tree: ast.AST | None = ast.parse(code)
        except SyntaxError:
            self.tree = None
        self.calls: list[tuple[str, ast.Call]] = []
        self.names: set[str] = set()
        if self.tree is not None:
            for n in ast.walk(self.tree):
                if isinstance(n, ast.Call):
                    self.calls.append((_dotted(n.func), n))
                if isinstance(n, ast.Name):
                    self.names.add(n.id.lower())
                elif isinstance(n, ast.Attribute):
                    self.names.add(n.attr.lower())
                elif isinstance(n, (ast.ClassDef, ast.FunctionDef)):
                    self.names.add(n.name.lower())

    def count(self, *suffixes: str) -> int:
        return sum(1 for d, _ in self.calls if d.split(".")[-1] in suffixes)

    def has_name(self, *keys: str) -> bool:
        return any(any(k in nm for k in keys) for nm in self.names)

    # attention ------------------------------------------------------------------------------------------
    def n_attention(self) -> int:
        return self.count(
            "MultiheadAttention",
            "TransformerEncoderLayer",
            "TransformerEncoder",
            "scaled_dot_product_attention",
        ) + (
            1
            if self.has_name("selfattention", "self_attention", "attention_block", "attnblock")
            else 0
        )

    def uses_attention(self) -> bool:
        if self.n_attention() > 0:
            return True
        # hand-written attention: a softmax over a matmul of two projections of the same input
        return (
            self.count("softmax") > 0
            and ("transpose" in self.lower or "@" in self.code)
            and self.has_name("q", "query", "qkv")
        )

    def uses_recurrence(self) -> bool:
        return self.count("LSTM", "GRU", "RNN", "LSTMCell", "GRUCell") > 0

    # convolutions -----------------------------------------------------------------------------------------
    def conv_calls(self) -> list[ast.Call]:
        return [c for d, c in self.calls if d.split(".")[-1] in ("Conv1d", "Conv2d", "LazyConv1d")]

    def kernel_sizes(self) -> set[int]:
        ks: set[int] = set()
        for c in self.conv_calls():
            k = None
            for kw in c.keywords:
                if kw.arg == "kernel_size":
                    k = _const(kw.value)
            if k is None and len(c.args) >= 3:
                k = _const(c.args[2])
            if isinstance(k, int):
                ks.add(k)
            elif isinstance(k, (tuple, list)) and k and isinstance(k[-1], int):
                ks.add(int(k[-1]))
        return ks

    def has_dilation(self) -> bool:
        for c in self.conv_calls():
            for kw in c.keywords:
                if kw.arg == "dilation":
                    v = _const(kw.value)
                    if v is None or (isinstance(v, int) and v > 1):
                        return True  # non-literal dilation (e.g. 2 ** i) counts as growing
        return False

    def residual(self) -> bool:
        """A skip connection in forward(): ``x + f(x)`` (a name added to a call or another name) or ``h += f(h)``."""
        if self.tree is None:
            return False
        for fn in ast.walk(self.tree):
            if not (isinstance(fn, ast.FunctionDef) and fn.name == "forward"):
                continue
            for n in ast.walk(fn):
                if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
                    sides = (n.left, n.right)
                    if any(isinstance(s, ast.Name) for s in sides) and any(
                        isinstance(s, (ast.Call, ast.Name)) for s in sides
                    ):
                        return True
                if (
                    isinstance(n, ast.AugAssign)
                    and isinstance(n.op, ast.Add)
                    and isinstance(n.value, (ast.Call, ast.Name))
                ):
                    return True
        return False

    def weight_tied(self) -> bool:
        """A module attribute that is called inside a for/while loop without indexing (the same weights applied repeatedly)."""
        if self.tree is None:
            return False
        for n in ast.walk(self.tree):
            if isinstance(n, (ast.For, ast.While)):
                for m in ast.walk(n):
                    if (
                        isinstance(m, ast.Call)
                        and isinstance(m.func, ast.Attribute)
                        and isinstance(m.func.value, ast.Name)
                        and m.func.value.id == "self"
                        and m.func.attr not in ("append", "forward")
                        and re.search(r"(block|layer|conv|cell|step|tied|shared|body)", m.func.attr)
                    ):
                        return True
        return False

    def unet(self) -> bool:
        up = self.count("ConvTranspose1d", "Upsample", "interpolate") > 0
        cat = self.count("cat", "concat", "concatenate") > 0
        return up and cat

    def pool_factor(self) -> int:
        f = 1
        for d, c in self.calls:
            last = d.split(".")[-1]
            if last in ("MaxPool1d", "AvgPool1d", "max_pool1d", "avg_pool1d"):
                k = None
                for kw in c.keywords:
                    if kw.arg in ("kernel_size", "stride") and k is None:
                        k = _const(kw.value)
                if k is None and c.args:
                    k = _const(c.args[0])
                if isinstance(k, int) and k > 1:
                    f *= k
        return f


def _norm(f: _Features) -> str:
    for cls, name in (
        ("BatchNorm1d", "batchnorm"),
        ("BatchNorm2d", "batchnorm"),
        ("LayerNorm", "layernorm"),
        ("GroupNorm", "groupnorm"),
        ("RMSNorm", "rmsnorm"),
        ("InstanceNorm1d", "instancenorm"),
    ):
        if f.count(cls):
            return name
    return "rmsnorm" if "rmsnorm" in f.lower else "none"


def _act(f: _Features) -> str:
    for nm, key in (
        ("gelu", "GELU"),
        ("silu", "SiLU"),
        ("relu", "ReLU"),
        ("elu", "ELU"),
        ("leakyrelu", "LeakyReLU"),
        ("mish", "Mish"),
        ("tanh", "Tanh"),
    ):
        if f.count(key) or f"f.{nm}" in f.lower or f"torch.{nm}" in f.lower:
            return nm
    return "unknown"


def _pooling(f: _Features) -> str:
    if f.count("MaxPool1d", "max_pool1d", "AdaptiveMaxPool1d"):
        return "maxpool"
    if f.count("AvgPool1d", "avg_pool1d", "AdaptiveAvgPool1d"):
        return "avgpool"
    if f.has_name("attnpool", "attention_pool", "attn_pool"):
        return "attention_pool"
    if ".mean(" in f.code or "mean(dim" in f.code:
        return "global_mean"
    if f.count("flatten", "Flatten"):
        return "flatten"
    return "none"


def _positional(f: _Features) -> str:
    if f.has_name("rotary", "rope"):
        return "rotary"
    if f.has_name("relative", "rel_pos", "relpos", "alibi"):
        return "relative"
    if f.has_name("sinusoid", "sincos"):
        return "sinusoidal"
    if f.has_name("pos_emb", "positional", "pos_embed", "posembedding", "position"):
        return "learned"
    return "none"


def rule_family(code: str) -> tuple[str, dict[str, Any]]:
    """Family from the code alone, with the evidence that decided it."""
    f = _Features(code)
    ev: dict[str, Any] = {
        "n_attention": f.n_attention(),
        "uses_attention": f.uses_attention(),
        "uses_recurrence": f.uses_recurrence(),
        "n_conv": len(f.conv_calls()),
        "kernel_sizes": sorted(f.kernel_sizes()),
        "dilation": f.has_dilation(),
        "residual": f.residual(),
        "weight_tied": f.weight_tied(),
        "unet": f.unet(),
        "state_space_keyword": f.has_name(
            "ssm", "state_space", "selective_scan", "selectivescan", "mamba", "s4d", "hippo"
        ),
        "mixer_keyword": f.has_name("mixer", "mlpmixer", "token_mix", "tokenmix"),
        "equivariant_keyword": f.has_name(
            "equivariant", "rc_equivariant", "reverse_complement_equivariant"
        ),
        "motif_keyword": f.has_name("motif")
        and f.has_name("grammar", "spacing", "pairwise", "interaction", "cooccur"),
    }
    n_conv, n_attn = ev["n_conv"], ev["n_attention"]
    if ev["uses_recurrence"]:
        fam = "rnn_hybrid"
    elif ev["state_space_keyword"]:
        fam = "state_space"
    elif ev["uses_attention"] and n_conv <= 1 and n_attn >= 1:
        fam = "transformer_pure" if n_conv == 0 else "transformer_hybrid"
    elif ev["uses_attention"] and n_attn >= max(2, n_conv // 2):
        fam = "transformer_hybrid"
    elif ev["unet"]:
        fam = "unet"
    elif ev["equivariant_keyword"]:
        fam = "equivariant"
    elif ev["mixer_keyword"] and n_conv <= 2:
        fam = "mixer_mlp"
    elif ev["weight_tied"] and ev["dilation"]:
        fam = "tcn_tied"
    elif ev["dilation"]:
        fam = "cnn_dilated"
    elif ev["motif_keyword"]:
        fam = "motif_grammar"
    elif len(ev["kernel_sizes"]) >= 3 and f.count("cat", "concat") > 0 and not ev["residual"]:
        fam = "cnn_multiscale"
    elif ev["residual"]:
        fam = "cnn_residual"
    else:
        fam = "cnn_plain"
    return fam, ev


# what each family needs to be true of the code (a claim without it is contradicted)
_REQUIRES = {
    "rnn_hybrid": lambda ev: ev["uses_recurrence"],
    "transformer_hybrid": lambda ev: (
        ev["uses_attention"] and ev["n_attention"] >= max(1, ev["n_conv"] // 2)
    ),
    "transformer_pure": lambda ev: ev["uses_attention"],
    "cnn_dilated": lambda ev: ev["dilation"],
    "tcn_tied": lambda ev: ev["weight_tied"] or ev["dilation"],
    "unet": lambda ev: ev["unet"],
    "state_space": lambda ev: ev["state_space_keyword"],
    "mixer_mlp": lambda ev: ev["mixer_keyword"],
    "equivariant": lambda ev: ev["equivariant_keyword"],
    "cnn_multiscale": lambda ev: len(ev["kernel_sizes"]) >= 2,
    "motif_grammar": lambda ev: ev["motif_keyword"] or ev["n_conv"] >= 1,
}
_NO_ATTENTION_FAMILIES = {
    "cnn_plain",
    "cnn_residual",
    "cnn_dilated",
    "cnn_multiscale",
    "tcn_tied",
    "unet",
}


def consistent(family: str, ev: dict[str, Any]) -> bool:
    """True when the code does not contradict the claimed family."""
    if family.startswith("other:"):
        # an 'other' family is only credible when no specialised mechanism is present in the code
        return not (
            ev["uses_recurrence"] or ev["uses_attention"] or ev["state_space_keyword"] or ev["unet"]
        )
    if family not in TAXONOMY:
        return False
    req = _REQUIRES.get(family)
    if req is not None and not req(ev):
        return False
    if (
        family in _NO_ATTENTION_FAMILIES
        and ev["uses_attention"]
        and ev["n_attention"] >= max(2, ev["n_conv"] // 2)
    ):
        return False  # attention holds most of the depth: not a CNN family
    if family in _NO_ATTENTION_FAMILIES and ev["uses_recurrence"]:
        return False
    return True


def rule_check(code: str, tags: dict[str, Any] | None = None) -> dict[str, Any]:
    """Tags from the rules, merged with an optional LLM tag dict. Never executes the code.

    Returns the full tag dict plus ``source`` (llm | llm+rules_override | rules), ``rule_family`` and ``rule_evidence``. Structural fields that the code
    settles (attention, recurrence, pooling factor) always come from the rules; descriptive fields come from the LLM when it gave a value.
    """
    f = _Features(code)
    fam_rule, ev = rule_family(code)
    llm = dict(tags or {})
    claimed = str(llm.get("family") or "")
    if claimed and consistent(claimed, ev):
        family, source = claimed, "llm"
    elif claimed:
        family, source = fam_rule, "llm+rules_override"
    else:
        family, source = fam_rule, "rules"
    out: dict[str, Any] = {
        "family": family,
        "core_block": llm.get("core_block") or _core_block(family, ev),
        "uses_attention": bool(ev["uses_attention"]),
        "uses_recurrence": bool(ev["uses_recurrence"]),
        "normalisation": llm.get("normalisation") or _norm(f),
        "activation": llm.get("activation") or _act(f),
        "pooling": llm.get("pooling") or _pooling(f),
        "positional_info": llm.get("positional_info") or _positional(f),
        "downsampling_factor": int(f.pool_factor()),
        "justification": str(llm.get("justification") or f"rules: {family}"),
        "source": source,
        "rule_family": fam_rule,
        "rule_evidence": ev,
    }
    if source == "llm+rules_override":
        out["claimed_family"] = claimed
        out["justification"] = (
            f"LLM claimed {claimed}; the code says {fam_rule}. " + out["justification"]
        )
    return out


def _core_block(family: str, ev: dict[str, Any]) -> str:
    return {
        "cnn_plain": "conv-pool",
        "cnn_residual": "residual conv block",
        "cnn_dilated": "dilated residual block",
        "cnn_multiscale": "parallel multi-kernel block",
        "rnn_hybrid": "bidirectional recurrent layer",
        "tcn_tied": "weight-tied dilated block",
        "transformer_hybrid": "self-attention block",
        "transformer_pure": "self-attention block over tokens",
        "state_space": "state-space layer",
        "mixer_mlp": "token/channel mixing MLP",
        "unet": "encoder-decoder with skips",
        "motif_grammar": "motif scan plus interaction module",
        "equivariant": "RC-equivariant layer",
    }.get(family, "other")


# --------------------------------------------------------------------------------------------- LLM tag call


def tag(
    proposer: Any, code: str, taxonomy: list[str] | tuple[str, ...], cfg: dict, workdir: Path
) -> dict[str, Any]:
    """Tag one candidate: LLM call (no tools) then the rule check. ``taxonomy`` is only used to validate the LLM's family string."""
    from ar2 import views  # local import: views imports this module for TAXONOMY

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    timeout_s = int((cfg.get("proposer", cfg) or {}).get("timeout_s", 900))
    llm: dict[str, Any] | None = None
    err = None
    try:
        prompt = views.render("TAGGER", views.ViewContext(), {"code": code})
        res = proposer.call(prompt, workdir, "tags", timeout_s)
        if res.ok and isinstance(res.parsed, dict):
            llm = dict(res.parsed)
            fam = str(llm.get("family", ""))
            if fam not in taxonomy and not fam.startswith("other:"):
                err = f"LLM family {fam!r} is not in the taxonomy"
                llm = None
        else:
            err = res.error or "tag call failed"
    except Exception as ex:  # noqa: BLE001 - a failed tag call must never stop the search; the rules tag the model
        err = f"{type(ex).__name__}: {ex}"
    out = rule_check(code, llm)
    if llm is None:
        out["source"] = "rules"
        out["justification"] = f"rules only ({err})" if err else out["justification"]
    return out


# --------------------------------------------------------------------------------------------- distance

_CAT_FIELDS = (
    "core_block",
    "uses_attention",
    "uses_recurrence",
    "normalisation",
    "activation",
    "pooling",
    "positional_info",
)


def tag_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Distance in [0, 1] between two tag dicts: a different family counts 3, each other field 1, downsampling by its log2 ratio (capped at 1)."""
    w_family = 3.0
    total = w_family + len(_CAT_FIELDS) + 1.0
    d = w_family * (0.0 if a.get("family") == b.get("family") else 1.0)
    for k in _CAT_FIELDS:
        d += 0.0 if a.get(k) == b.get(k) else 1.0
    fa, fb = (
        max(1, int(a.get("downsampling_factor") or 1)),
        max(1, int(b.get("downsampling_factor") or 1)),
    )
    d += min(1.0, abs(math.log2(fa) - math.log2(fb)) / 3.0)
    return d / total
