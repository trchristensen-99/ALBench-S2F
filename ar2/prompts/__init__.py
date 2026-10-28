"""Frozen prompt templates (spec section 11). A template change is a new prompt version (spec 12.2), so this module only reads files and hashes them."""

from __future__ import annotations

import hashlib
from pathlib import Path

PROMPT_VERSION = "v1"
PROMPT_DIR = Path(__file__).resolve().parent / PROMPT_VERSION
PROMPT_FILES: tuple[str, ...] = (
    "core.md",
    "sketch.md",
    "explore.md",
    "refine.md",
    "repair.md",
    "tagger.md",
    "lessons.md",
    "model_contract.md",
)


def load_template(name: str) -> str:
    return (PROMPT_DIR / name).read_text()


def template_hash() -> str:
    """sha1 (first 12 hex digits) over the names and bytes of every template file, in the fixed PROMPT_FILES order."""
    h = hashlib.sha1()
    for name in PROMPT_FILES:
        h.update(name.encode())
        h.update(b"\0")
        h.update((PROMPT_DIR / name).read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:12]
