"""Derived tables and the controller's snapshot, computed from the ledger only (spec 10.1, 10.3).

Everything here is a pure function of the ledger tables (``runs``, ``candidates``, ``calls``, ``decisions``, ``sketches``, ``lesson_text``): ``ar2 rebuild`` drops the derived
tables (``families``, ``lessons``) and recomputes them, and a test (T6) compares the result with the incrementally maintained ones. Facts the controller reports back
(refinement outcomes, deep-tune outcomes, family transitions) are read from NOTE rows of the ``decisions`` table, so they are part of the ledger.

Conventions the orchestrator follows when it writes ledger rows (documented here because the derivation depends on them):

* ``runs.purpose`` is one of first_look | tune | second_chance | refine | deep_tune | confirm | screen | procedure | final. A confirmation run carries ``meta.role`` in
  family | child | deep; a deep-tune run carries ``meta.batch``; an out-of-memory run that was replaced has ``meta.replaced_by``; a replacement has ``meta.replaces``.
* ``candidates.card`` holds the LLM's card plus ``_ar2: {origin: explore|refine|anchor, angle}``; ``candidates.parent_code_hash`` is set for refinement children.
* NOTE decisions have ``mode='NOTE'`` and ``payload.kind`` in refinement_outcome | deep_outcome | family_state | guard | phase | stop | menu | adaptation.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from typing import Any

from ar2.memory.db import DB
from ar2.state import ChildInfo, DeepInfo, FamilyState, RunInfo, Snapshot

TAXONOMY = [
    "cnn_plain",
    "cnn_residual",
    "cnn_dilated",
    "cnn_multiscale",
    "rnn_hybrid",
    "tcn_tied",
    "transformer_hybrid",
    "transformer_pure",
    "state_space",
    "mixer_mlp",
    "unet",
    "motif_grammar",
    "equivariant",
]
ANCHORS = ["cnn_residual", "rnn_hybrid", "transformer_hybrid"]
SCREENING = ("first_look", "tune", "second_chance")
BAD_STATUS = ("diverged", "exploded", "error", "oom")
OPEN = ("submitted", "running", "pending")
CHECKPOINTS = (0.5, 0.75)
PHASE_BUDGET_KEY = {"A": "anchors", "S": "screen", "P": "procedures", "M": "main", "D": "final"}


# ---------------------------------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------------------------------
def extract_hparams(source: str) -> tuple[dict, dict]:
    """HPARAMS and TUNABLE literals of a model.py, read with ``ast.literal_eval`` (the source is never executed)."""
    out: dict[str, dict] = {"HPARAMS": {}, "TUNABLE": {}}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}, {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in out
        ):
            try:
                v = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                continue
            if isinstance(v, dict):
                out[node.targets[0].id] = v
    return out["HPARAMS"], out["TUNABLE"]


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _sd(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _valid(r: dict) -> bool:
    return r["status"] not in OPEN and r["status"] not in BAD_STATUS and r["val_r"] is not None


def _meta(r: dict) -> dict:
    m = r.get("meta")
    return m if isinstance(m, dict) else {}


def _counted(r: dict) -> bool:
    """A run that holds a recipe slot: an out-of-memory run that has a replacement does not (the replacement does, spec 8.6)."""
    return not (r["status"] == "oom" and _meta(r).get("replaced_by"))


def run_info(r: dict) -> RunInfo:
    diag = r.get("diagnostics") if isinstance(r.get("diagnostics"), dict) else {}
    meta = _meta(r)
    recipe = r["recipe"] if isinstance(r["recipe"], dict) else {}
    st = r["status"]
    return RunInfo(
        run_id=r["run_id"],
        code_hash=r["code_hash"],
        family=r.get("family") or "",
        recipe_hash=r["recipe_hash"],
        recipe=recipe,
        schedule=r.get("schedule") or recipe.get("schedule", "onecycle"),
        seed=int(r["seed"]),
        purpose=r["purpose"],
        category=r.get("category") or "explore",
        phase=r.get("phase") or "M",
        status="pending" if st == "submitted" else st,
        val_r=r.get("val_r"),
        train_r=r.get("train_r"),
        cost_equiv=float(r.get("cost_equiv") or 0.0),
        epochs_run=int(r.get("epochs_run") or 0),
        max_epochs=int(recipe.get("max_epochs", 40)),
        best_epoch=int(diag.get("best_epoch", 0) or 0),
        still_improving=bool(diag.get("still_improving", False)),
        improvement_per_epoch=float(diag.get("improvement_per_epoch", 0.0) or 0.0),
        censored=(st == "time_cap") or r.get("stop_reason") == "time_cap",
        exploded=(st == "exploded") or bool((diag.get("stress") or {}).get("exploded")),
        checkpoint_best=meta.get("checkpoint_best", {}),
    )


def decisions_notes(db: DB, kind: str) -> list[dict]:
    out = []
    for d in db.decisions("NOTE"):
        p = d.get("payload")
        if isinstance(p, dict) and p.get("kind") == kind:
            out.append(p)
    if kind == "screen_adoption":
        # the controller emits the adoption as a SCREEN action whose target has kind "adopt"; without this the decision was never seen again and the screen repeated forever
        for d in db.decisions("SCREEN"):
            p = d.get("payload")
            t = p.get("target") if isinstance(p, dict) else None
            if isinstance(t, dict) and t.get("kind") == "adopt":
                eff = {
                    o: (e.get("mean") if isinstance(e, dict) else e)
                    for o, e in (t.get("effects") or {}).items()
                }
                out.append({**t, "kind": "screen_adoption", "effects": eff})
                break  # one adoption per search (the repeated decisions made before this fix are the same record)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# family derivation (the facts come from the ledger; the policy-level reading of them lives in ar2.controller.build_family_state)
# ---------------------------------------------------------------------------------------------------------------------------------
def _cand_info(c: dict) -> dict:
    card = c.get("card") if isinstance(c.get("card"), dict) else {}
    ar2 = card.get("_ar2", {}) if isinstance(card.get("_ar2"), dict) else {}
    hp = c.get("hparams") if isinstance(c.get("hparams"), dict) else {}
    return {
        "code_hash": c["code_hash"],
        "parent_code_hash": c.get("parent_code_hash"),
        "origin": ar2.get("origin", "explore"),
        "gate_status": "passed",
        "angle": ar2.get("angle") or "",
        "summary": str(card.get("summary", ""))[:200],
        "hparams": hp.get("hparams", {}) or {},
        "tunable": hp.get("tunable", {}) or {},
        "suggested_recipe": card.get("suggested_recipe", {}) or {},
    }


def _runinfo_all(db: DB) -> list[tuple[dict, RunInfo]]:
    out = []
    for r in db.runs_for():
        ri = run_info(r)
        ri.cap_multiplier = float(_meta(r).get("cap_multiplier", 1.0) or 1.0)
        ri.diagnostics = {
            k: v
            for k, v in (r["diagnostics"] if isinstance(r.get("diagnostics"), dict) else {}).items()
            if k
            in ("pred_sd_ratio", "mse_by_activity_quintile", "mse_by_gc_tercile", "stop_reason")
        }
        out.append((r, ri))
    return out


def derive_families(db: DB, cfg: dict, taxonomy: list[str] | None = None) -> dict[str, FamilyState]:
    """FamilyState per family from the ledger, through the controller's own fact builder, so the facts and the policy cannot drift apart."""
    from ar2 import controller as ctl

    taxonomy = taxonomy or TAXONOMY
    cands = [c for c in db.candidates() if c.get("family") and c.get("gate_status") == "ok"]
    by_family: dict[str, list[dict]] = {}
    for c in cands:
        by_family.setdefault(c["family"], []).append(c)
    runs = _runinfo_all(db)
    notes = [d.get("payload") for d in db.decisions("NOTE")]
    notes = [p for p in notes if isinstance(p, dict)]
    fams: dict[str, FamilyState] = {}
    for fam in sorted(set(by_family) | set(taxonomy)):
        fc = by_family.get(fam, [])
        if not fc:
            fams[fam] = FamilyState(family=fam, anchor=fam in ANCHORS)
            continue
        hashes = {c["code_hash"] for c in fc}
        fruns = [ri for r, ri in runs if r["code_hash"] in hashes]
        outcomes = [
            p
            for p in notes
            if p.get("kind") in ("refinement_outcome", "deep_outcome") and p.get("family") == fam
        ]
        fs = ctl.build_family_state(
            fam, fruns, [_cand_info(c) for c in fc], outcomes, cfg, anchor=fam in ANCHORS
        )
        fs.state = _cached_state(db, fam) or "first_look"
        fams[fam] = fs
    return fams


def _cached_state(db: DB, fam: str) -> str | None:
    last = None
    for n in decisions_notes(db, "family_state"):
        if n.get("family") == fam:
            last = n.get("to", n.get("state"))
    return last


def family_rows(db: DB, cfg: dict, taxonomy: list[str] | None = None) -> list[dict]:
    """Rows of the derived ``families`` table (only families that have a candidate). Deterministic: ``updated_at`` is the latest run finish time of the family."""
    fams = derive_families(db, cfg, taxonomy)
    runs = db.runs_for()
    rows = []
    for fam, fs in fams.items():
        if fs.n_candidates == 0:
            continue
        hashes = {c["code_hash"] for c in db.candidates(family=fam)}
        fin = [r["finished_at"] or "" for r in runs if r["code_hash"] in hashes]
        vals = [
            x["val_r"]
            for x in fs.recipe_results
            if x["val_r"] is not None and x["purpose"] in SCREENING
        ]  # single-seed screening results
        rows.append(
            {
                "family": fam,
                "state": fs.state,
                "n_candidates": fs.n_candidates,
                "n_runs": fs.n_runs,
                "best_code_hash": fs.best_code_hash,
                "best_recipe_hash": fs.best_recipe_hash,
                "screen_best": fs.screen_best,
                "confirmed_mean": fs.confirmed_mean,
                "confirmed_sd": fs.confirmed_sd,
                "n_confirm": fs.n_confirm,
                "recipe_spread": _sd(vals),
                "refinements": fs.refinements,
                "refinements_without_gain": fs.refinements_without_gain,
                "triage": fs.triage or None,
                "second_chance_used": int(fs.second_chance_used),
                "censored": int(fs.censored),
                "updated_at": max(fin) if fin else "",
                "detail": fs.to_dict(),
            }
        )
    return rows


# ---------------------------------------------------------------------------------------------------------------------------------
# noise control and lessons
# ---------------------------------------------------------------------------------------------------------------------------------
def sigma_seed(db: DB, cfg: dict, fams: dict[str, FamilyState] | None = None) -> float | None:
    """Pooled within-family SD of confirmation runs; None (so the prior is used) until 3 families are confirmed (9.5)."""
    fams = fams if fams is not None else derive_families(db, cfg)
    rows = [f for f in fams.values() if f.n_confirm >= 3]
    if len(rows) < 3:
        return None
    ss, df = 0.0, 0
    for f in rows:
        if f.confirmed_sd is not None:
            ss += f.confirmed_sd**2 * (f.n_confirm - 1)
            df += f.n_confirm - 1
    return math.sqrt(ss / df) if df else None


def evidence_key(kind: str, scope: str, key: str, run_ids: list[str]) -> str:
    return hashlib.sha1(json.dumps([kind, scope, key, sorted(run_ids)]).encode()).hexdigest()[:16]


def _template_statement(ev: dict) -> str:
    """Deterministic wording used when no LLM wording is on record, so rebuild never depends on a model call."""
    if ev["kind"] == "refinement":
        return f"[{ev['scope']}] {ev['key']}: {'accepted' if ev['accepted'] else 'rejected'}"
    if ev["kind"] == "screen":
        return f"[global] {ev['key']}: {'adopted' if ev['accepted'] else 'not adopted'}"
    return f"[{ev['scope']}] {ev['key']}"


def derive_lessons(db: DB, cfg: dict) -> list[dict]:
    """Lessons from the three allowed sources only (10.3): refinement outcomes where parent and child both have 3 confirmation seeds, the procedure screen, and
    aggregates over at least 5 runs computed by code (divergence rate). Numbers come from the ledger; wording comes from ``lesson_text`` or a fixed template.
    A later lesson with the same scope and key and the opposite sign whose interval excludes the earlier effect retracts it."""
    sig = sigma_seed(db, cfg) or cfg["controller"]["sigma_seed_prior"]
    se_diff = sig * math.sqrt(2 / 3)
    runs = db.runs_for()
    cost_cum = 0.0
    cost_at: dict[str, float] = {}
    for r in sorted(
        runs, key=lambda x: (x.get("finished_at") or "", x["run_id"])
    ):  # budget spent when the evidence was complete, in finish order
        if r["status"] not in OPEN:
            cost_cum += float(r["cost_equiv"] or 0.0)
        cost_at[r["run_id"]] = cost_cum
    conf: dict[tuple[str, str], list[dict]] = {}
    for r in runs:
        if r["purpose"] == "confirm" and _valid(r):
            conf.setdefault((r["code_hash"], r["recipe_hash"]), []).append(r)
    cand = {c["code_hash"]: c for c in db.candidates()}
    out: list[dict] = []
    for n in decisions_notes(db, "refinement_outcome"):
        child = n["child"]
        parent = n.get("parent") or (db.candidate(child) or {}).get("parent_code_hash")
        cc = next((v for (h, _), v in conf.items() if h == child and len(v) >= 3), None)
        pc = next(
            (
                v
                for (h, rh), v in conf.items()
                if h == parent
                and len(v) >= 3
                and (not n.get("parent_recipe_hash") or rh == n["parent_recipe_hash"])
            ),
            None,
        )
        if not cc or not pc:
            continue
        effect = _mean([r["val_r"] for r in cc]) - _mean([r["val_r"] for r in pc])
        ids = [r["run_id"] for r in cc + pc]
        angle = (
            (cand.get(child, {}).get("card") or {}).get("_ar2", {}).get("angle", "change")
            if isinstance(cand.get(child, {}).get("card"), dict)
            else "change"
        )
        ev = {
            "kind": "refinement",
            "scope": n["family"],
            "key": f"{angle}",
            "effect": effect,
            "se": se_diff,
            "n": 3,
            "accepted": bool(n["accepted"]),
            "run_ids": ids,
        }
        out.append(ev)
    for n in decisions_notes(db, "screen_adoption"):
        for opt, eff in (n.get("effects") or {}).items():
            out.append(
                {
                    "kind": "screen",
                    "scope": "global",
                    "key": f"procedure {opt}",
                    "effect": eff,
                    "se": n.get("se", se_diff),
                    "n": n.get("n", 6),
                    "accepted": opt in (n.get("adopted") or []),
                    "run_ids": n.get("run_ids", []),
                }
            )
    # aggregate: divergence rate per family over >= 5 runs
    fams: dict[str, list[dict]] = {}
    for r in runs:
        if (
            r["status"] not in OPEN
            and r.get("family")
            and r["purpose"] in SCREENING + ("tune", "deep_tune")
        ):
            fams.setdefault(r["family"], []).append(r)
    for fam, rs in sorted(fams.items()):
        if len(rs) >= 5:
            rate = sum(1 for r in rs if r["status"] == "diverged") / len(rs)
            if rate > 0:
                out.append(
                    {
                        "kind": "aggregate",
                        "scope": fam,
                        "key": "divergence rate",
                        "effect": rate,
                        "se": math.sqrt(rate * (1 - rate) / len(rs)),
                        "n": len(rs),
                        "accepted": True,
                        "run_ids": [r["run_id"] for r in rs],
                    }
                )
    rows = []
    for ev in out:
        ek = evidence_key(ev["kind"], ev["scope"], ev["key"], ev["run_ids"])
        stmt = db.lesson_text(ek) or _template_statement(ev)
        rows.append(
            {
                "lesson_id": ek,
                "scope": ev["scope"],
                "statement": f"{stmt} ({ev['effect']:+.3f} ± {ev['se']:.3f}, n={ev['n']})",
                "effect": ev["effect"],
                "se": ev["se"],
                "n": ev["n"],
                "evidence_run_ids": ev["run_ids"],
                "status": "confirmed",
                "created_at_cost": max((cost_at.get(i, 0.0) for i in ev["run_ids"]), default=0.0),
                "kind": ev["kind"],
                "key": ev["key"],
            }
        )
    # retraction
    groups: dict[tuple, list[dict]] = {}
    for r in sorted(rows, key=lambda x: (x["created_at_cost"], x["lesson_id"])):
        groups.setdefault((r["scope"], r["kind"], r["key"]), []).append(r)
    for rs in groups.values():
        for i, old in enumerate(rs[:-1]):
            for new in rs[i + 1 :]:
                if old["effect"] * new["effect"] < 0 and abs(
                    new["effect"] - old["effect"]
                ) > 2 * math.hypot(new["se"], old["se"]):
                    old["status"] = "retracted"
    return rows


# ---------------------------------------------------------------------------------------------------------------------------------
# rebuild and snapshot
# ---------------------------------------------------------------------------------------------------------------------------------
def rebuild(db: DB, cfg: dict) -> None:
    """Drop and recompute the derived tables from the ledger (``ar2 rebuild``)."""
    db.replace_families(family_rows(db, cfg))
    db.replace_lessons(derive_lessons(db, cfg))


def _recent_calls(db: DB) -> dict:
    out: dict[str, list] = {"EXPLORE": [], "SKETCH": [], "REFINE": []}
    for mode in out:
        for c in db.calls(mode)[-10:]:
            a = c.get("assignment") if isinstance(c.get("assignment"), dict) else {}
            o = a.get("outcome", {}) if isinstance(a, dict) else {}
            out[mode].append(
                {
                    "assigned_family": a.get("family") or o.get("assigned_family"),
                    "landed_family": o.get("landed_family"),
                    "landed_family_size": o.get("landed_family_size", 0),
                    "first_attempt_gate_ok": o.get(
                        "first_attempt_gate_ok", c.get("status") == "ok"
                    ),
                    "duplicate": o.get("duplicate", False),
                    "speed_fail": o.get("speed_fail", False),
                    "collapse": o.get("collapse", False),
                }
            )
    return out


def _new_family_outcomes(db: DB) -> list[str]:
    seen: dict[str, str] = {}
    order: list[str] = []
    for n in decisions_notes(db, "family_state"):
        fam, st = n.get("family"), n.get("to", n.get("state"))
        if fam in ANCHORS or st not in ("parked", "retired", "tuned_only", "confirmed", "menu"):
            continue
        if fam not in seen:
            order.append(fam)
        seen[fam] = st
    return [seen[f] for f in order]


def _exploit_outcomes(db: DB) -> list[dict]:
    out = []
    for d in db.decisions("NOTE"):
        p = d.get("payload")
        if isinstance(p, dict) and p.get("kind") in ("refinement_outcome", "deep_outcome"):
            out.append(
                {
                    "kind": "refine" if p["kind"] == "refinement_outcome" else "deep_tune",
                    "accepted": bool(p.get("accepted")),
                }
            )
    return out


def build_snapshot(
    db: DB,
    cfg: dict,
    step: int,
    *,
    calls_in_flight: int = 0,
    inflight_targets: list[dict] | None = None,
    stop_requested: bool = False,
    taxonomy: list[str] | None = None,
) -> Snapshot:
    """Facts for ``Controller.decide``. Spend counts finished runs only (cost_equiv); open runs are reported separately and counted 1.0 each by the controller."""
    runs = db.runs_for()
    families = derive_families(db, cfg, taxonomy)
    spend = {p: 0.0 for p in "ASPMD"}
    open_runs = {p: 0 for p in "ASPMD"}
    sbc = {"explore": 0.0, "refine": 0.0, "deep_tune": 0.0}
    obc = {"explore": 0, "refine": 0, "deep_tune": 0}
    pend = {
        k: 0 for k in ("first_look", "tune", "confirm", "refine", "deep_tune", "final", "screen")
    }
    since_sketch = 0.0
    last_sketch_at = (db.calls("SKETCH") or [{"created_at": ""}])[-1]["created_at"]
    for r in runs:
        ph, cat = r.get("phase") or "M", r.get("category") or "explore"
        if r["status"] in OPEN:
            open_runs[ph] = open_runs.get(ph, 0) + 1
            if ph == "M" and cat in obc:
                obc[cat] += 1
            key = "tune" if r["purpose"] in ("second_chance",) else r["purpose"]
            if key in pend:
                pend[key] += 1
        else:
            spend[ph] = spend.get(ph, 0.0) + float(r["cost_equiv"] or 0.0)
            if ph == "M" and cat in sbc:
                sbc[cat] += float(r["cost_equiv"] or 0.0)
            if (r.get("finished_at") or "") > last_sketch_at:
                since_sketch += float(r["cost_equiv"] or 0.0)
    sk = [
        {
            "id": s["sketch_id"],
            "family": s["family"],
            "name": s["name"],
            "mechanism": (s["body"] or {}).get("mechanism", "")
            if isinstance(s["body"], dict)
            else "",
        }
        for s in db.sketches("unused")
    ]
    unplanned, repair = [], []
    have_runs = {r["code_hash"] for r in runs}
    for c in db.candidates():
        card = c.get("card") if isinstance(c.get("card"), dict) else {}
        ar2 = card.get("_ar2", {})
        if (
            c.get("gate_status") in ("ok", "passed", "pass")
            and c.get("family")
            and c["code_hash"] not in have_runs
        ):
            hp = c.get("hparams") if isinstance(c.get("hparams"), dict) else {}
            unplanned.append(
                {
                    "code_hash": c["code_hash"],
                    "family": c["family"],
                    "origin": ar2.get("origin", "explore"),
                    "parent_code_hash": c.get("parent_code_hash"),
                    "suggested_recipe": card.get("suggested_recipe", {}) or {},
                    "hparams": hp.get("hparams", {}),
                    "tunable": hp.get("tunable", {}),
                }
            )
    for c in db.candidates(gate_status="repair"):
        if int(c.get("repair_attempts") or 0) < cfg["gate"]["max_repairs"]:
            repair.append(
                {
                    "code_hash": c["code_hash"],
                    "family": c.get("family_claimed") or c.get("family"),
                    "gate_code": c.get("gate_code"),
                    "attempts": int(c.get("repair_attempts") or 0),
                }
            )
    guard = {}
    try:
        guard = json.loads(db.meta_get("guard_state", "{}") or "{}")
    except ValueError:
        pass
    sig = sigma_seed(db, cfg, families)
    top = max(
        (f.confirmed_mean for f in families.values() if f.confirmed_mean is not None), default=None
    )
    defaults = {"loss": {"name": "mse"}, "ema": None, "screen_done": False}
    try:
        defaults = {**defaults, **json.loads(db.meta_get("recipe_defaults", "{}") or "{}")}
    except ValueError:
        pass
    scr = [r for r in runs if r["purpose"] == "screen"]
    sres, sbase = [], {}
    for r in scr:
        m = _meta(r)
        if r["status"] not in OPEN and r["val_r"] is not None:
            sres.append(
                {
                    "option": m.get("option"),
                    "anchor": m.get("anchor"),
                    "seed": int(r["seed"]),
                    "val_r": r["val_r"],
                }
            )
    for a in (
        ANCHORS
    ):  # the baseline of the procedure screen is each anchor's confirmation on seeds 1 and 2 (13.1)
        fs = families.get(a)
        if fs is not None and fs.best_code_hash:
            for r in runs:
                if (
                    r["purpose"] == "confirm"
                    and r["code_hash"] == fs.best_code_hash
                    and r["recipe_hash"] == fs.best_recipe_hash
                    and int(r["seed"]) in (1, 2)
                    and _valid(r)
                ):
                    sbase.setdefault(a, {})[str(r["seed"])] = r["val_r"]
    summaries = [
        {
            "family": c["family"],
            "summary": (c.get("card") or {}).get("summary", "")
            if isinstance(c.get("card"), dict)
            else "",
        }
        for c in db.candidates()
        if c.get("family")
    ]
    return Snapshot(
        step=step,
        phase="M",
        D=int(cfg["data"]["D"]),
        budgets=dict(cfg["controller"]["budget"]),
        spend=spend,
        open_runs=open_runs,
        spend_by_category=sbc,
        open_by_category=obc,
        balance=dict(cfg["balance"]),
        families=families,
        taxonomy=list(taxonomy or TAXONOMY),
        anchors=list(ANCHORS),
        pending_counts=pend,
        runs_in_flight=sum(1 for r in runs if r["status"] in OPEN),
        calls_in_flight=calls_in_flight,
        calls_in_flight_targets=list(inflight_targets or []),
        sketches_unused=sk,
        re_since_sketch=since_sketch,
        repair_queue=repair,
        new_candidates_unplanned=unplanned,
        candidate_summaries=summaries,
        recent_new_family_outcomes=_new_family_outcomes(db),
        recent_exploit_outcomes=_exploit_outcomes(db),
        recent_calls=_recent_calls(db),
        guard_state=guard,
        sigma_seed=sig,
        top_confirmed_mean=top,
        archive_hash=db.archive_hash(),
        recipe_defaults=defaults,
        screen_results=sres,
        screen_open=sum(1 for r in scr if r["status"] in OPEN),
        screen_baseline=sbase,
        screen_decision_done=bool(decisions_notes(db, "screen_adoption")),
        procedures=json.loads(db.meta_get("procedures", "{}") or "{}"),
        last_progress_spend=float(db.meta_get("last_progress_spend", "0") or 0),
        cum_main_spend=spend["M"],
        stop_requested=stop_requested,
        final_menu_done=db.meta_get("final_menu_done") == "1",
        anchors_budget_exhausted=spend["A"] >= cfg["controller"]["budget"]["anchors"],
    )


def menu_families(db: DB, cfg: dict) -> list[dict]:
    """Menu-eligible families (5.2): confirmed mean within ``menu_margin`` of the top; if fewer than ``menu_min_families`` qualify, the top confirmed families, flagged."""
    rows = [
        r for r in family_rows(db, cfg) if r["confirmed_mean"] is not None and r["n_confirm"] >= 3
    ]
    rows.sort(key=lambda r: -r["confirmed_mean"])
    if not rows:
        return []
    top = rows[0]["confirmed_mean"]
    elig = [
        dict(r, flagged=False)
        for r in rows
        if top - r["confirmed_mean"] <= cfg["controller"]["menu_margin"]
    ]
    k = cfg["controller"]["menu_min_families"]
    if len(elig) < k:
        elig = [
            dict(r, flagged=(top - r["confirmed_mean"] > cfg["controller"]["menu_margin"]))
            for r in rows[:k]
        ]
    return elig
