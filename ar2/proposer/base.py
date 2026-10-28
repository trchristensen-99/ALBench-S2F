"""Proposer contract (spec 4.7): one call = one fresh, isolated session that returns files or JSON."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

Expect = Literal["sketches", "candidate", "tags", "lesson"]

SKETCH_KEYS = (
    "family",
    "name",
    "mechanism",
    "why_for_mpra",
    "nearest_existing",
    "difference",
    "recipe_needs",
)
CARD_REQUIRED = ("family_claimed", "summary", "hypothesis")
CARD_OPTIONAL = {
    "what_changed": str,
    "parent_code_hash": str,
    "suggested_recipe": dict,
    "recipe_rationale": str,
    "expected_failure": str,
    "amp_safe": bool,
    "length_agnostic": bool,
}
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


class ProposerPaused(Exception):
    """Authentication or rate-limit error: the orchestrator pauses new LLM calls (training continues) and retries after ``retry_after_s``."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class ProposerHalt(Exception):
    """Five consecutive failed calls: halt the run with a clear message (state stays resumable)."""


class ProposerCapabilityError(RuntimeError):
    """The CLI lacks a required option; the message names it (spec 6.1, T15)."""


@dataclass
class ProposerResult:
    ok: bool
    files: dict[str, str] = field(
        default_factory=dict
    )  # file name -> text, for the files the call was expected to produce
    parsed: Any = None  # sketches: list[dict]; candidate: the card dict; tags / lesson: dict
    raw_stdout: str = ""
    raw_stderr: str = ""
    model_id: str | None = None
    cli_version: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    exit_code: int = 0
    error: str | None = None
    # additions beyond the contract (all defaulted, so callers that ignore them keep working)
    missing: list[str] = field(
        default_factory=list
    )  # what the follow-up REPAIR call should say is missing
    retries: int = 0
    proposer_changed: bool = (
        False  # reported model id differs from the pinned one or changed mid-run
    )


class Proposer(Protocol):
    def call(
        self, prompt: str, workdir: Path, expect: Expect, timeout_s: int
    ) -> ProposerResult: ...


# ------------------------------------------------------------------------------------------ validation


def validate_sketches(obj: Any) -> list[str]:
    if not isinstance(obj, list) or not obj:
        return ["sketches.json must be a non-empty list"]
    errs: list[str] = []
    for i, s in enumerate(obj):
        if not isinstance(s, dict):
            errs.append(f"sketch {i} is not an object")
            continue
        miss = [k for k in SKETCH_KEYS if k not in s]
        if miss:
            errs.append(f"sketch {i} lacks keys {miss}")
        elif not all(
            isinstance(s[k], str) and s[k].strip() for k in ("family", "name", "mechanism")
        ):
            errs.append(f"sketch {i}: family, name and mechanism must be non-empty strings")
    if not errs and len(obj) != 10:
        errs.append(f"expected 10 sketches, got {len(obj)}")
    return errs


def validate_card(obj: Any) -> list[str]:
    if not isinstance(obj, dict):
        return ["card.json must be an object"]
    errs = [
        f"card.json lacks {k!r}"
        for k in CARD_REQUIRED
        if not isinstance(obj.get(k), str) or not obj.get(k, "").strip()
    ]
    for k, typ in CARD_OPTIONAL.items():
        if k in obj and obj[k] is not None and not isinstance(obj[k], typ):
            errs.append(f"card.json key {k!r} must be {typ.__name__}")
    return errs


def validate_tags(obj: Any) -> list[str]:
    if not isinstance(obj, dict):
        return ["tags must be one JSON object"]
    errs = [f"tags lack {k!r}" for k in TAG_KEYS if k not in obj]
    if "family" in obj and not isinstance(obj["family"], str):
        errs.append("tags.family must be a string")
    return errs


def validate_lesson(obj: Any) -> list[str]:
    if (
        not isinstance(obj, dict)
        or not isinstance(obj.get("statement"), str)
        or not obj["statement"].strip()
    ):
        return ['lesson must be a JSON object {"statement": "..."}']
    return []


VALIDATORS = {
    "sketches": validate_sketches,
    "candidate": validate_card,
    "tags": validate_tags,
    "lesson": validate_lesson,
}
