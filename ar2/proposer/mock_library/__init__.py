"""A library of small, valid architectures across the 13 families, and planted failure transforms, for the mock proposer and for orchestrator tests.

Every file obeys the model contract (HPARAMS, TUNABLE, build_model(out_dim, **hp)), is pure torch and runs on CPU. Three of them are also the reference
models shown to the proposer (``ar2/proposer/reference_models``). ``make_variant`` rewrites the numeric defaults of ``HPARAMS`` so that variants of one
family have different normalised code (hence different code hashes) while staying valid.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE.parent / "reference_models"
FAMILY_FILES: dict[str, Path] = {
    "cnn_plain": HERE / "cnn_plain.py",
    "cnn_residual": REF / "cnn_residual.py",
    "cnn_dilated": HERE / "cnn_dilated.py",
    "cnn_multiscale": HERE / "cnn_multiscale.py",
    "rnn_hybrid": REF / "rnn_hybrid.py",
    "tcn_tied": HERE / "tcn_tied.py",
    "transformer_hybrid": REF / "transformer_hybrid.py",
    "transformer_pure": HERE / "transformer_pure.py",
    "state_space": HERE / "state_space.py",
    "mixer_mlp": HERE / "mixer_mlp.py",
    "unet": HERE / "unet.py",
    "motif_grammar": HERE / "motif_grammar.py",
    "equivariant": HERE / "equivariant.py",
}
FAMILIES = tuple(FAMILY_FILES)
_WIDTH_KEYS = ("width", "dim", "motifs", "hidden")
_DEPTH_KEYS = ("depth", "steps", "blocks")


def library_source(family: str) -> str:
    if family not in FAMILY_FILES:
        raise KeyError(f"no library model for family {family!r}")
    return FAMILY_FILES[family].read_text()


def make_variant(source: str, k: int) -> str:
    """Variant ``k`` of a model: widths + 8k, depths + (k % 2). k = 0 returns the source unchanged."""
    if k == 0:
        return source
    m = re.search(r"^HPARAMS = (\{.*\})$", source, re.M)
    if not m:
        return source
    hp = ast.literal_eval(m.group(1))
    for key, v in hp.items():
        if key in _WIDTH_KEYS and isinstance(v, int):
            hp[key] = v + 8 * k
        elif key in _DEPTH_KEYS and isinstance(v, int):
            hp[key] = v + (k % 2)
    return source[: m.start(1)] + repr(hp) + source[m.end(1) :]


def library_variant(family: str, k: int = 0) -> str:
    return make_variant(library_source(family), k)


# ------------------------------------------------------------------------------------------ planted failures

_WRAP = """

_orig_build_model = build_model


def build_model(out_dim, **hp):
    net = _orig_build_model(out_dim, **hp)

    class _Planted(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
{init}
        def forward(self, x):
{body}
    return _Planted()
"""

_BODIES = {
    "wrong_shape": ("", "            return self.net(x).repeat(1, 2)"),
    "constant": ("", "            return self.net(x) * 0.0 + 1.0"),
    "crops_input": ("", "            return self.net(x[:, :, 60:140])"),
    "batch_leak": ("", "            return self.net(x) + x.mean()"),
    "nonfinite": ("", "            return self.net(x) * float('inf')"),
    "too_large": (
        "            self.big = nn.Parameter(torch.zeros(45_000_000))\n",
        "            return self.net(x) + 0.0 * self.big[0]",
    ),
    "cannot_learn": ("", "            return self.net(x).detach()"),
    "slow": (
        "",
        "            y = self.net(x)\n            z = torch.ones(256, 256)\n            for _ in range(600):\n                z = z @ z / 256.0\n            return y + 0.0 * z.sum()",
    ),
}
PLANTED_KINDS = (
    "syntax_error",
    "forbidden_import",
    "duplicate",
    "collapse",
    "missing_card",
    "invalid_card",
    *_BODIES,
)


def broken_source(kind: str, base: str) -> str:
    """The base model with one planted defect (code-level kinds only; ``missing_card``, ``invalid_card``, ``duplicate`` and ``collapse`` are handled by the mock)."""
    if kind == "syntax_error":
        return base + "\ndef broken(:\n    pass\n"
    if kind == "forbidden_import":
        return "import os\n" + base
    if kind in _BODIES:
        init, body = _BODIES[kind]
        return base + _WRAP.format(init=init, body=body)
    raise KeyError(f"unknown planted kind {kind!r}")
