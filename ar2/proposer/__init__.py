"""Proposer backends (spec 4.7 and 6): the headless Claude Code CLI and a deterministic mock."""

from __future__ import annotations

from typing import Any

from ar2.proposer.base import (
    Proposer,
    ProposerCapabilityError,
    ProposerHalt,
    ProposerPaused,
    ProposerResult,
)

__all__ = [
    "Proposer",
    "ProposerCapabilityError",
    "ProposerHalt",
    "ProposerPaused",
    "ProposerResult",
    "make_proposer",
]


def make_proposer(cfg: dict[str, Any], check: bool = True) -> Proposer:
    """Build the backend named by ``proposer.backend`` (``claude_code_cli`` or ``mock``).

    ``cfg`` is the whole resolved config or just its ``proposer`` section. For the CLI backend the capability check runs here (``ar2 init`` calls this),
    so a missing option fails before anything is spent; pass ``check=False`` only in tests that stub the binary.
    """
    p = cfg.get("proposer", cfg) if isinstance(cfg, dict) else {}
    backend = p.get("backend", "claude_code_cli")
    if backend == "mock":
        from ar2.proposer.mock import MockProposer

        return MockProposer(
            seed=int(p.get("seed", 0)),
            planted=p.get("planted"),
            fail_calls=set(p.get("fail_calls", []) or []),
            paused_calls=p.get("paused_calls"),
            tag_override=p.get("tag_override"),
            model_id=str(p.get("model") or "mock-model-1"),
        )
    if backend == "claude_code_cli":
        from ar2.proposer.claude_code_cli import ClaudeCodeCLIProposer

        prop = ClaudeCodeCLIProposer(p)
        if check:
            prop.check()
        return prop
    raise ValueError(f"unknown proposer backend {backend!r}")
