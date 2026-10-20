"""Ask a model to WRITE architectures, not to fill in a config schema.

The hand-written HP space can only express CNN variants, so it can never answer
"does architecture matter here?" -- the question is outside its vocabulary. This
path hands over a contract and a budget and lets the proposer write whatever it
can express in torch.

The prompt deliberately does NOT list architecture families. Naming CNN/RNN/
transformer would steer toward exactly the three everyone thinks of first, and
the point is to find out what it reaches for unprompted. The only hard statements
are the input/output contract and the resource budget, because violating those
produces a model that cannot be scored rather than an interesting one.
"""

from __future__ import annotations

import os
import re

CONTRACT = """\
CONTRACT (a proposal violating any of this is discarded unscored):
  - define a class `Net(nn.Module)` taking NO constructor arguments
  - forward(x) receives x of shape (batch, 4, {seq_len}) -- one-hot DNA, ACGT order
  - forward must return shape (batch,) -- one scalar regression output per sequence
  - <= {max_params:,} parameters, <= {max_fwd_ms:.0f} ms for a batch-8 CPU forward
  - importable: torch, numpy, math, random, functools, itertools, typing, re
    (no filesystem, network, subprocess or interpreter-internals access)
  - output must depend on the input
"""

SYSTEM = """\
You are designing neural network architectures for a genomics regression task, \
and you write them as working PyTorch code.

You are not filling in a hyperparameter schema. You are free to propose ANY \
architecture you can express in torch -- including ideas that have no name, that \
combine families, or that you are simply curious about. Unusual ideas that are \
plausibly motivated are more valuable here than another well-tuned version of \
whatever is currently winning, because the point of this search is to find out \
what the task actually rewards.

Return exactly {n} proposals. For each, emit:

### IDEA: <one line naming the mechanism you are betting on>
### WHY: <one or two sentences -- what property of the task makes this plausible>
```python
<complete, self-contained source defining Net>
```

Write real code. It will be executed and trained."""

TASK = """\
TASK: predict measured regulatory activity of a 200bp human DNA sequence in a \
massively parallel reporter assay (lentiMPRA, K562). Input is one-hot DNA; target \
is a single continuous activity value. Training set size for this run: {D:,} \
sequences. Local sequence motifs (transcription-factor binding sites) are known to \
drive activity, but so are their number, spacing and arrangement, and long-range \
composition effects are plausible but unestablished.
"""


def build_arch_prompt(
    n: int,
    history: list[dict],
    *,
    seq_len: int = 200,
    D: int = 30000,
    max_params: int = 60_000_000,
    max_fwd_ms: float = 400.0,
    note: str | None = None,
) -> tuple[str, str]:
    """-> (system, user). `history` entries: {idea, layers, n_params, score, ok, reason}."""
    sys_p = SYSTEM.format(n=n)
    parts = [
        TASK.format(D=D),
        CONTRACT.format(seq_len=seq_len, max_params=max_params, max_fwd_ms=max_fwd_ms),
    ]

    tried = [h for h in history if h.get("ok")]
    failed = [h for h in history if not h.get("ok")]
    if tried:
        parts.append("\nARCHITECTURES ALREADY TRIED (validation Pearson r, higher is better):")
        for h in sorted(tried, key=lambda h: -(h.get("score") or -9))[:18]:
            ts = h.get("train_s")
            tcol = f"  {ts:>4.0f}s" if ts else ""
            parts.append(
                f"  r={h.get('score', float('nan')):.4f}  "
                f"{h.get('n_params', 0):>9,}p{tcol}  {h.get('idea', '')[:66]}"
            )
    if failed:
        parts.append("\nPROPOSALS THAT WERE DISCARDED (do not repeat these mistakes):")
        for h in failed[:8]:
            parts.append(f"  {h.get('reason', '')[:88]}  <- {h.get('idea', '')[:50]}")
    if note:
        parts.append(f"\nNOTE: {note}")
    parts.append(f"\nPropose {n} architectures now.")
    return sys_p, "\n".join(parts)


_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
_IDEA = re.compile(r"###\s*IDEA:\s*(.+)")
_WHY = re.compile(r"###\s*WHY:\s*(.+)")


def parse_proposals(text: str) -> list[dict]:
    """Pull (idea, why, source) triples out of a response.

    Code blocks are the anchor: an idea line without code is not a proposal, and
    code without an idea line still counts -- losing a working architecture to a
    formatting slip would bias the search toward whichever model formats best.
    """
    out = []
    blocks = _BLOCK.findall(text)
    ideas = _IDEA.findall(text)
    whys = _WHY.findall(text)
    for i, src in enumerate(blocks):
        out.append(
            {
                "idea": (ideas[i].strip() if i < len(ideas) else f"unnamed proposal {i + 1}"),
                "why": (whys[i].strip() if i < len(whys) else ""),
                "source": src.strip(),
            }
        )
    return out


def call_model(system: str, user: str, model: str | None = None, max_tokens: int = 16000) -> str:
    """Reuse the existing AutoResearch transport (CLI when a token is set, else API)."""
    import sys

    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "experiments",
        ),
    )
    from llm_autoresearch import _call_claude_api, _call_claude_cli

    mdl = model or os.environ.get("LLM_MODEL", "claude-sonnet-4-5-20250929")
    use_cli = bool(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")) and not os.environ.get(
        "LLM_FORCE_API"
    )
    if use_cli:
        return _call_claude_cli(system, user, mdl)
    return _call_claude_api(system, user, mdl, max_tokens, temperature=None)
