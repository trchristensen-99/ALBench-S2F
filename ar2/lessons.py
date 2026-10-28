"""Lessons log helpers (spec 10.3): evidence records, numbers computed by code, wording by one LLM call, scope and retraction.

A lesson shapes every later proposal, so wrong ones are costly. Therefore:

* the only sources are (a) a refinement whose parent and child both have 3 confirmation seeds, (b) a result of the procedure screen, (c) an aggregate over at
  least 5 runs computed by code (for example a divergence rate);
* the effect, its standard error and n are computed here from the cited runs, never by the LLM; the LLM sees a record without numbers, writes one
  sentence, and the numbers are appended by this module (a sentence that contains a digit is discarded and replaced by template wording);
* each lesson has a scope (one family or ``global``); a family lesson is never shown as global;
* "no effect" is recorded, with its interval; an option that cannot be tested on noise-free oracle labels is recorded as ``not_testable_in_silico``;
* a later lesson of the same kind and subject that reverses the sign with an interval excluding the original effect retracts the earlier one.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

KINDS = ("refinement", "procedure", "aggregate")
MIN_SEEDS = 3
MIN_AGGREGATE_RUNS = 5
MIN_PROCEDURE_PAIRS = 4


def _mean(x: list[float]) -> float:
    return sum(x) / len(x)


def _var(x: list[float]) -> float:
    m = _mean(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - 1) if len(x) > 1 else 0.0


def compute_stats(
    kind: str, runs: dict[str, float], groups: dict[str, list[str]]
) -> dict[str, Any]:
    """effect, standard error, n and a description of n from the cited runs. Raises ValueError when the evidence is not allowed to become a lesson."""
    if kind == "refinement":
        p, c = [runs[i] for i in groups["parent"]], [runs[i] for i in groups["child"]]
        if len(p) < MIN_SEEDS or len(c) < MIN_SEEDS:
            raise ValueError(
                f"a refinement lesson needs {MIN_SEEDS} confirmation seeds on both sides; got {len(c)} and {len(p)}"
            )
        return {
            "effect": _mean(c) - _mean(p),
            "se": math.sqrt(_var(c) / len(c) + _var(p) / len(p)),
            "n": len(c) + len(p),
            "n_desc": f"{len(c)} vs {len(p)} seeds",
        }
    if kind == "procedure":
        o, b = [runs[i] for i in groups["option"]], [runs[i] for i in groups["baseline"]]
        if len(o) != len(b) or len(o) < MIN_PROCEDURE_PAIRS:
            raise ValueError(
                f"a procedure lesson needs at least {MIN_PROCEDURE_PAIRS} paired runs; got {len(o)} and {len(b)}"
            )
        d = [x - y for x, y in zip(o, b)]
        return {
            "effect": _mean(d),
            "se": math.sqrt(_var(d) / len(d)),
            "n": len(d),
            "n_desc": f"{len(d)} paired runs",
        }
    if kind == "aggregate":
        v = [float(runs[i]) for i in groups["runs"]]
        if len(v) < MIN_AGGREGATE_RUNS:
            raise ValueError(
                f"an aggregate lesson needs at least {MIN_AGGREGATE_RUNS} runs; got {len(v)}"
            )
        p = _mean(v)
        binary = all(x in (0.0, 1.0) for x in v)
        se = math.sqrt(p * (1 - p) / len(v)) if binary else math.sqrt(_var(v) / len(v))
        return {"effect": p, "se": se, "n": len(v), "n_desc": f"{len(v)} runs"}
    raise ValueError(f"unknown lesson kind {kind!r}; expected one of {KINDS}")


def direction_of(effect: float, se: float) -> str:
    if effect - 2 * se > 0:
        return "improved"
    if effect + 2 * se < 0:
        return "hurt"
    return "no clear effect"


def make_evidence(
    kind: str,
    *,
    scope: str,
    subject: str,
    runs: dict[str, float],
    groups: dict[str, list[str]],
    accepted: bool | None = None,
    noise_floor: float = 0.005,
    status: str = "confirmed",
) -> dict[str, Any]:
    """Structured evidence record. ``runs`` maps run_id to validation r (or 0/1 for an indicator); ``groups`` says which runs are the parent / child, the
    option / baseline pairs, or the aggregated runs. ``accepted`` is the controller's decision (default: gain above max(noise floor, 2 se))."""
    stats = compute_stats(kind, runs, groups)
    direction = direction_of(stats["effect"], stats["se"]) if kind != "aggregate" else "rate"
    if accepted is None and kind != "aggregate":
        accepted = stats["effect"] > max(noise_floor, 2 * stats["se"])
    ids = sorted({i for g in groups.values() for i in g})
    return {
        "kind": kind,
        "scope": scope,
        "subject": subject,
        "change": subject,
        "direction": direction,
        "accepted": accepted,
        "status": status,
        "groups": {k: list(v) for k, v in groups.items()},
        "evidence_run_ids": ids,
        **stats,
    }


def lesson_id(ev: dict[str, Any]) -> str:
    key = f"{ev['kind']}|{ev['scope']}|{ev['subject']}|{','.join(ev['evidence_run_ids'])}"
    return "L_" + hashlib.sha1(key.encode()).hexdigest()[:12]


def llm_record(ev: dict[str, Any]) -> dict[str, Any]:
    """What the wording call is allowed to see: no numbers."""
    return {
        k: ev[k] for k in ("kind", "scope", "subject", "change", "direction", "accepted") if k in ev
    }


def numbers_text(ev: dict[str, Any]) -> str:
    if ev["kind"] == "aggregate":
        return f"{ev['effect'] * 100:.0f}% +/- {ev['se'] * 100:.0f}% of {ev['n_desc']}"
    return f"{ev['effect']:+.3f} +/- {ev['se']:.3f} ({ev['n_desc']})"


def verdict_text(ev: dict[str, Any]) -> str:
    if ev.get("status") == "not_testable_in_silico":
        return "Not testable in silico (noise-free labels)."
    if ev["kind"] == "refinement":
        return "Accepted." if ev.get("accepted") else "Rejected."
    if ev["kind"] == "procedure":
        return "Adopted." if ev.get("accepted") else "Not adopted."
    return ""


def template_sentence(ev: dict[str, Any]) -> str:
    subj = str(ev["subject"]).strip().rstrip(".")
    subj = subj[:1].upper() + subj[1:]
    phrase = {
        "improved": "raised validation r",
        "hurt": "lowered validation r",
        "no clear effect": "made no clear difference",
        "rate": "occurred at this rate",
    }[ev["direction"]]
    return f"{subj} {phrase}"


def format_statement(ev: dict[str, Any], sentence: str) -> str:
    """``[scope] sentence: numbers. Verdict`` with the numbers computed by code."""
    sentence = sentence.strip().rstrip(".:")
    return f"[{ev['scope']}] {sentence}: {numbers_text(ev)}. {verdict_text(ev)}".rstrip()


def word(proposer: Any, evidence: dict[str, Any], cfg: dict, workdir: Path) -> str:
    """One LLM call for the wording; the statement returned already carries the numbers appended by code."""
    sentence = None
    try:
        from ar2 import views

        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        prompt = views.render("LESSON", views.ViewContext(), {"evidence": llm_record(evidence)})
        res = proposer.call(
            prompt, workdir, "lesson", int((cfg.get("proposer", cfg) or {}).get("timeout_s", 900))
        )
        if res.ok and isinstance(res.parsed, dict):
            s = str(res.parsed.get("statement", "")).strip()
            if s and not re.search(r"\d", s):  # the LLM must not add numbers
                sentence = s
    except Exception:  # noqa: BLE001 - wording is cosmetic; the template sentence is always available
        sentence = None
    return format_statement(evidence, sentence or template_sentence(evidence))


def to_row(ev: dict[str, Any], statement: str, created_at_cost: float = 0.0) -> dict[str, Any]:
    """A row for the derived ``lessons`` table (spec 10.2). ``evidence_run_ids`` holds JSON with the groups, so the numbers can be recomputed."""
    return {
        "lesson_id": lesson_id(ev),
        "scope": ev["scope"],
        "statement": statement,
        "effect": ev["effect"],
        "se": ev["se"],
        "n": ev["n"],
        "evidence_run_ids": json.dumps(
            {
                "kind": ev["kind"],
                "subject": ev["subject"],
                "status": ev.get("status", "confirmed"),
                "groups": ev["groups"],
            },
            sort_keys=True,
        ),
        "status": "confirmed",
        "created_at_cost": created_at_cost,
    }


def recompute(row: dict[str, Any], runs: dict[str, float]) -> dict[str, Any]:
    """Effect, se and n recomputed from the runs the row cites (T11)."""
    ev = json.loads(row["evidence_run_ids"])
    return compute_stats(ev["kind"], runs, ev["groups"])


def numbers_in_statement(statement: str) -> list[float]:
    return [float(x) for x in re.findall(r"[+-]?\d+\.\d+|\d+(?=%)", statement)]


def should_retract(old_row: dict[str, Any], new_row: dict[str, Any]) -> bool:
    """True when ``new_row`` has the same kind, scope and subject as ``old_row``, reverses the sign, and its interval (effect +/- 2 se) excludes the old effect."""
    if old_row["status"] != "confirmed" or old_row["scope"] != new_row["scope"]:
        return False
    a, b = json.loads(old_row["evidence_run_ids"]), json.loads(new_row["evidence_run_ids"])
    if a["kind"] != b["kind"] or a["subject"] != b["subject"] or a["kind"] == "aggregate":
        return False
    if old_row["effect"] * new_row["effect"] >= 0:
        return False
    return abs(new_row["effect"] - old_row["effect"]) > 2 * new_row["se"]


def select_for_prompt(
    rows: list[dict[str, Any]], family: str | None, limit: int = 15
) -> tuple[list[str], list[str]]:
    """(family lessons, global lessons) for a ViewContext: confirmed only, newest first, the assignment's family first, at most ``limit`` in all."""
    live = [r for r in rows if r["status"] == "confirmed"]
    live.sort(key=lambda r: -float(r.get("created_at_cost", 0.0)))
    fam = [r["statement"] for r in live if family and r["scope"] == family][:limit]
    glob = [r["statement"] for r in live if r["scope"] == "global"][: max(0, limit - len(fam))]
    return fam, glob
