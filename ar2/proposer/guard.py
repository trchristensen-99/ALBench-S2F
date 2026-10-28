"""Last-line content guard for rendered prompts: held-out results and the final-scores table must never reach a proposer (spec 9.6, 11.1)."""

from __future__ import annotations

FORBIDDEN = ("val_select", "final_scores", "chr7", "chr13", "chrX")


def scan(text: str) -> None:
    low = text.lower()
    for w in FORBIDDEN:
        if w.lower() in low:
            raise ValueError(f"forbidden content {w!r} in a rendered prompt")
