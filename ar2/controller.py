"""The controller (spec section 5): code decides what to explore, the LLM writes one model per assignment.

Three layers, all pure functions of plain data (see ar2/state.py), so the whole policy can be tested on a simulator without a database, a GPU or an LLM:

1. facts      ``build_family_state`` turns the ledger rows of one family (runs, candidates, recorded outcomes) into a ``FamilyState``;
              ``derive_family_state`` computes the lifecycle state from the facts (spec 5.2), ``menu_select`` the menu (5.2), ``acceptance`` the noise rule (9.5).
2. policy     ``Controller.decide(snapshot, free_job_slots, free_call_slots) -> list[Action]``: phases A/S/P/M/D with budgets in run-equivalents, the family life
              cycle, target choice, the explore-exploit balance with its guards (5.7), the stop rule (5.5) and the prompt adaptations of 12.1.
3. forecast   ``plan(cfg)`` prints the table of 5.7 and ``forecast(cfg)`` returns the numbers.

Determinism: the controller's randomness is a pure function of (cfg.controller.seed, snapshot.step, a tag), so a run replays exactly from its decisions. The only
state the controller keeps is ``guard_state`` (guard offsets and marks, hysteresis flags); it is returned in ``Controller.guard_state`` after each ``decide``, the
orchestrator stores it in meta and hands it back in ``Snapshot.guard_state``; it is also written into the decisions table as a NOTE (kind "guard_state"), and
``Controller.replay`` rebuilds it from those notes.
"""

from __future__ import annotations

import copy
import math
import zlib
from typing import Any

import numpy as np

from ar2 import tuner
from ar2.state import (
    CATEGORIES,
    REFINE_ANGLES,
    Action,
    ChildInfo,
    DeepInfo,
    FamilyState,
    RunInfo,
    Snapshot,
)

# --------------------------------------------------------------------------------------------------------------------------------- configuration

DEFAULT_CFG: dict = {
    "data": {"D": 100000, "primary_task": "K562", "extra_tasks": []},
    "evaluator": {
        "max_epochs": 40,
        "patience": 8,
        "base_cap_s": 1800,
        "time_cap_exponent": 0.72,
        "cap_extension": 1.5,
    },
    "tuner": {
        "recipes_per_family": 8,
        "first_look": 3,
        "early_term_margin": 0.03,
        "early_term_checkpoints": [0.5, 0.75],
        "screen_seed": 0,
        "confirm_seeds": [1, 2, 3],
        "final_seeds": [4, 5, 6],
    },
    "controller": {
        "seed": 20261023,
        "budget": {"anchors": 30, "screen": 36, "procedures": 12, "main": 102, "final": 20},
        "confirm_margin": 0.02,
        "menu_min_families": 5,
        "keep_margin": 0.03,
        "menu_margin": 0.01,
        "refine_margin": 0.02,
        "noise_floor": 0.005,
        "sigma_seed_prior": 0.005,
        "refine_temperature": 0.01,
        "plateau_after": 3,
        "sketches_per_call": 10,
        "sketches_kept": 2,
        "resketch_every": 20,
        "flat_stop": {"min_families_tuned": 8, "window": 50},
        "max_parallel_jobs": 4,
        "max_parallel_calls": 2,
        "queue_depth": 2,  # new LLM work starts only while open runs < queue_depth * max_parallel_jobs
        "anchor_source": "reference",  # reference: anchors are registered from the reference models, no LLM call; llm: EXPLORE with forced family
        "anchor_borrow": 6,  # run-equivalents phase A may borrow from the main budget to finish confirming the anchors
        "screen_borrow": 6,  # same for the combination runs of the procedure screen
        "refine_screen_margin": 0.005,  # a refinement child is confirmed if its best screening score is at most this far below its parent's
        "min_start": {
            "explore": 6.0,
            "refine": 4.0,
            "deep_tune": 4.0,
        },  # run-equivalents that must remain before new work of a category starts
        "expected_cost": {"explore": 8.0, "refine": 4.5, "deep_tune": 4.0},
        "deep_batch": 4,
        "expected_run_cost": 0.8,  # run-equivalents a not-yet-created run is expected to cost when setting budget aside for lineages in progress
        "final_uses_leftover": True,
    },
    "balance": {
        "mode": "scheduled",
        "explore_fraction": 0.6,
        "explore_start": 0.75,
        "explore_end": 0.40,
        "deep_tune_share": 0.3,
        "breadth_floor": 8,
        "guards": True,
        "guard_window": 4,
        "guard_step": 0.2,
        "explore_min": 0.2,
        "explore_max": 0.8,
    },
    "adaptation": {
        "window": 10,
        "collapse_family_size": 5,
        "novelty_min": 0.5,
        "gate_fail_on": 0.4,
        "gate_fail_off": 0.2,
    },
    "procedure_mode": {"enabled": True, "max_plugins": 2},
    "phases": {"screen": True, "procedures": True},
}

SCREEN_OPTIONS = ("huber", "hetero", "corr", "binned", "ema", "multitask")
NOT_TESTABLE_IN_SILICO = (
    "huber",
    "hetero",
)  # spec 13.3: labels carry no noise, so a null result does not reject these
TAXONOMY = (
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
)
ANCHORS = ("cnn_residual", "rnn_hybrid", "transformer_hybrid")
VALID_STATUSES = ("ok", "time_cap")
SCREENING_PURPOSES = ("first_look", "tune", "second_chance")
ALL_SCORING_PURPOSES = SCREENING_PURPOSES + ("refine", "deep_tune")
CHECK_FAILED = ("parked", "retired", "tuned_only")


def merged(cfg: dict | None) -> dict:
    """The user's config over the spec-15 defaults (so the controller also works on a bare dict)."""

    def merge(base: dict, new: dict) -> dict:
        out = copy.deepcopy(base)
        for k, v in (new or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = merge(out[k], v)
            else:
                out[k] = copy.deepcopy(v)
        return out

    if cfg and cfg.get("_merged"):
        return cfg  # already merged (the Controller passes its own copy around; merging again would deep-copy the defaults on every call)
    out = merge(DEFAULT_CFG, cfg or {})
    out["_merged"] = True
    return out


# --------------------------------------------------------------------------------------------------------------------------------- facts


def _valid(r: RunInfo) -> bool:
    return bool(
        r.finished and r.status in VALID_STATUSES and not r.exploded and r.val_r is not None
    )


def _rhash(r: RunInfo) -> str:
    return r.recipe_hash or (tuner.recipe_hash(r.recipe) if r.recipe else "")


def _best(runs: list[RunInfo]) -> RunInfo | None:
    v = [r for r in runs if _valid(r)]
    return (
        max(v, key=lambda r: r.val_r) if v else None
    )  # first of equal scores wins (max keeps the earliest)


def build_family_state(
    family: str,
    runs: list[RunInfo],
    candidates: list[dict],
    outcomes: list[dict],
    cfg: dict | None = None,
    *,
    anchor: bool = False,
) -> FamilyState:
    """Facts of one family from ledger rows.

    runs: every run of every candidate of the family, in submission order (RunInfo; open runs have status pending / running).
    candidates: [{code_hash, parent_code_hash, origin ("explore"|"anchor"|"refine"), gate_status ("passed"|"failed"|"pending"), angle, summary, hparams, tunable,
    suggested_recipe}] for the family.  outcomes: the recorded NOTE outcomes for the family in chronological order: {"kind": "refinement_outcome", "child", "accepted",
    "effect", "se"} and {"kind": "deep_outcome", "recipe_hash", "batch", "accepted", "effect"}.
    The family's current best is replayed from these facts: the best valid screening run of its first candidate(s), then every accepted refinement or deep-tune recipe in order.
    """
    c = merged(cfg)
    fs = FamilyState(family=family, anchor=anchor)
    cand_by = {x["code_hash"]: x for x in candidates}
    passed = [x for x in candidates if x.get("gate_status", "passed") == "passed"]
    fs.n_candidates = len(passed)
    fs.n_runs = len(runs)
    fs.gate_failed_open = sum(
        1 for x in candidates if x.get("gate_status") in ("failed", "repairing")
    )

    def done(rs: list[RunInfo]) -> list[RunInfo]:
        # an 'error' run reaches here only after the infrastructure retries are used up, so it is a deterministic failure of that recipe (for example sampled
        # hyperparameters the model cannot be built with): it counts toward the recipes of the family, like a diverged run, or the family would wait for it forever
        return [r for r in rs if r.finished and r.status != "oom"]

    def open_(rs: list[RunInfo]) -> list[RunInfo]:
        return [r for r in rs if not r.finished]

    screen = [r for r in runs if r.purpose in SCREENING_PURPOSES and r.seed == 0]
    fl = [r for r in screen if r.purpose == "first_look"]
    sc = [r for r in screen if r.purpose == "second_chance"]
    fs.first_look_done, fs.first_look_open = len(done(fl)), len(open_(fl))
    fb = _best(fl)
    fs.first_look_best = fb.val_r if fb else None
    fs.second_chance_used = bool(sc)
    fs.second_chance_done, fs.second_chance_open = len(done(sc)), len(open_(sc))
    fs.recipes_done, fs.recipes_open = len(done(screen)), len(open_(screen))
    fs.recipes_planned = fs.recipes_done + fs.recipes_open
    fs.recipe_results = [
        {
            "recipe_hash": _rhash(r),
            "val_r": r.val_r,
            "schedule": r.schedule,
            "status": r.status,
            "code_hash": r.code_hash,
            "seed": r.seed,
            "purpose": r.purpose,
            "recipe": copy.deepcopy(r.recipe),
            "train_r": r.train_r,
            "censored": r.censored,
            "still_improving": r.still_improving,
        }
        for r in done(screen)
    ]
    # --- the family's current best, replayed from the facts
    first = _best(screen)
    if first is not None:
        fs.best_code_hash, fs.best_recipe, fs.best_recipe_hash = (
            first.code_hash,
            copy.deepcopy(first.recipe),
            _rhash(first),
        )
    elif passed:
        fs.best_code_hash = passed[0]["code_hash"]
    accepted_children: set[str] = set()
    decided_children: set[str] = set()
    decided_batches: set[int] = set()
    for o in outcomes:
        if o.get("kind") == "refinement_outcome":
            child = o["child"]
            decided_children.add(child)
            fs.refinements += 1
            fs.refinement_history.append(
                {k: o.get(k) for k in ("child", "accepted", "effect", "se")}
            )
            if o.get("accepted"):
                accepted_children.add(child)
                fs.refinements_without_gain = 0
                cb = _best(
                    [
                        r
                        for r in runs
                        if r.code_hash == child and r.purpose == "refine" and r.seed == 0
                    ]
                )
                if cb is not None:
                    fs.best_code_hash, fs.best_recipe, fs.best_recipe_hash = (
                        child,
                        copy.deepcopy(cb.recipe),
                        _rhash(cb),
                    )
            else:
                fs.refinements_without_gain += 1
        elif o.get("kind") == "deep_outcome":
            decided_batches.add(int(o.get("batch", -1)))
            fs.deep.attempts += 1
            if o.get("accepted"):
                fs.deep.accepted += 1
                hit = next((r for r in runs if _rhash(r) == o.get("recipe_hash")), None)
                if hit is not None:
                    fs.best_recipe, fs.best_recipe_hash = copy.deepcopy(hit.recipe), _rhash(hit)
    bc = cand_by.get(fs.best_code_hash or "", {})
    fs.summary = bc.get("summary", "")
    fs.hparams, fs.tunable = dict(bc.get("hparams") or {}), dict(bc.get("tunable") or {})
    fs.suggested_recipe = dict(bc.get("suggested_recipe") or {})
    # --- scores of the best candidate
    own = [
        r
        for r in runs
        if r.code_hash == fs.best_code_hash and r.purpose in ALL_SCORING_PURPOSES and r.seed == 0
    ]
    sb = _best(own)
    fs.screen_best = sb.val_r if sb else None
    if sb is not None:
        fs.censored = bool(sb.censored)
        fs.last_diagnostics = {
            "train_r": sb.train_r,
            "val_r": sb.val_r,
            "best_epoch": sb.best_epoch,
            "epochs_run": sb.epochs_run,
            "max_epochs": sb.max_epochs,
            "still_improving": sb.still_improving,
            **(sb.diagnostics or {}),
        }
    # --- confirmation of the best (candidate, recipe)
    conf = [
        r
        for r in runs
        if r.purpose == "confirm"
        and r.code_hash == fs.best_code_hash
        and _rhash(r) == fs.best_recipe_hash
        and r.seed in (1, 2, 3)
    ]
    vals = {r.seed: r.val_r for r in conf if _valid(r)}
    fs.confirm_vals = [vals[s] for s in sorted(vals)]
    fs.confirm_open = len(open_(conf))
    fs.n_confirm = len(fs.confirm_vals)
    if fs.n_confirm >= 3:
        fs.confirmed_mean = float(np.mean(fs.confirm_vals))
        fs.confirmed_sd = float(np.std(fs.confirm_vals, ddof=1))
    fs.censored_extended = any(r.cap_multiplier > 1.0 for r in conf)
    fin = [
        r
        for r in runs
        if r.purpose == "final"
        and r.code_hash == fs.best_code_hash
        and _rhash(r) == fs.best_recipe_hash
    ]
    fs.final_vals = [r.val_r for r in fin if _valid(r)]
    fs.final_open = len(open_(fin))
    # --- refinement lineage
    refine_cands = [x for x in passed if x.get("origin") == "refine"]
    fs.angle_index = len([x for x in candidates if x.get("origin") == "refine"])
    pending = [
        x
        for x in refine_cands
        if x["code_hash"] not in decided_children and x["code_hash"] != fs.best_code_hash
    ]
    if pending:
        ch = pending[-1]["code_hash"]
        cr = [r for r in runs if r.code_hash == ch and r.purpose == "refine" and r.seed == 0]
        cc = [r for r in runs if r.code_hash == ch and r.purpose == "confirm"]
        parent = pending[-1].get("parent_code_hash")
        pscreen = _best(
            [
                r
                for r in runs
                if r.code_hash == parent and r.purpose in ALL_SCORING_PURPOSES and r.seed == 0
            ]
        )
        cb = _best(cr)
        fs.pending_refinement = ChildInfo(
            code_hash=ch,
            parent_code_hash=parent or "",
            angle=pending[-1].get("angle", ""),
            recipes_planned=len(done(cr)) + len(open_(cr)),
            screen_scores=[r.val_r for r in cr if _valid(r)],
            screen_open=len(open_(cr)),
            confirm_planned=len(done(cc)) + len(open_(cc)) >= 3,
            confirm_vals=[r.val_r for r in cc if _valid(r)],
            confirm_open=len(open_(cc)),
            best_recipe=copy.deepcopy(cb.recipe) if cb else {},
            outcome_recorded=False,
            parent_screen_best=pscreen.val_r if pscreen else fs.screen_best,
            parent_confirmed_mean=fs.confirmed_mean,
        )
    # --- deep tune: runs of purpose deep_tune in submission order, grouped in batches
    nb = int(c["controller"]["deep_batch"])
    deep_runs = [r for r in runs if r.purpose == "deep_tune" and r.seed == 0]
    fs.deep.recipes_done = len(done(deep_runs))
    batches = [deep_runs[i : i + nb] for i in range(0, len(deep_runs), nb)]
    fs.deep.batch_size = nb
    cur = next((i for i, b in enumerate(batches) if i not in decided_batches), None)
    fs.deep.batch_index = cur if cur is not None else len(batches)
    if cur is not None:
        b = batches[cur]
        bb = _best(b)
        fs.deep.batch_runs = len(b)
        fs.deep.batch_scores = [r.val_r for r in b if _valid(r)]
        fs.deep.batch_open = len(open_(b))
        fs.deep.batch_best_recipe = copy.deepcopy(bb.recipe) if bb else {}
        others = [r for r in own if r.run_id not in {x.run_id for x in b}] + [
            r
            for bi, bt in enumerate(batches)
            for r in bt
            if bi != cur and _valid(r) and r.code_hash == fs.best_code_hash
        ]
        pb = _best(others)
        fs.deep.prev_screen_best = pb.val_r if pb else fs.screen_best
        if bb is not None:
            dc = [
                r
                for r in runs
                if r.purpose == "confirm"
                and r.category == "deep_tune"
                and _rhash(r) == _rhash(bb)
                and r.code_hash == bb.code_hash
            ]
            if dc:
                fs.deep.confirm_recipe = copy.deepcopy(bb.recipe)
                fs.deep.confirm_recipe_hash = _rhash(bb)
                fs.deep.confirm_vals = [r.val_r for r in dc if _valid(r)]
                fs.deep.confirm_open = len(open_(dc))
    return fs


def derive_family_state(
    fs: FamilyState, top: float | None, cfg: dict | None = None, *, anchor: bool | None = None
) -> str:
    """Lifecycle state of a family from its facts and the best confirmed mean ``top`` (spec 5.2).

    The sticky states need no stored history: retired and tuned_only follow from the numbers, parked from ``second_chance_used`` and the first-look scores.
    """
    c = merged(cfg)["controller"]
    anchor = fs.anchor if anchor is None else anchor
    keep, cm, rm = c["keep_margin"], c["confirm_margin"], c["refine_margin"]
    first = int(merged(cfg)["tuner"]["first_look"])
    per_family = int(merged(cfg)["tuner"]["recipes_per_family"])
    if fs.n_candidates == 0 and fs.recipes_planned == 0:
        return "empty"
    if fs.n_confirm >= 3 and fs.confirmed_mean is not None:
        if fs.refinements_without_gain >= c["plateau_after"]:
            return "plateaued"
        if top is not None and fs.confirmed_mean < top - rm:
            return "retired"
        return "confirmed"
    if fs.first_look_done < first or fs.first_look_open > 0:
        return "first_look"
    if fs.second_chance_used:
        if fs.second_chance_done < first or fs.second_chance_open > 0:
            return "parked"
        if (
            not anchor
            and top is not None
            and (fs.screen_best is None or fs.screen_best < top - keep)
        ):
            return "retired"
    elif (
        not anchor
        and top is not None
        and (fs.first_look_best is None or fs.first_look_best < top - keep)
    ):
        return "parked"
    if fs.recipes_planned < per_family or fs.recipes_open > 0:
        return "tuning"
    if anchor or top is None or (fs.screen_best is not None and fs.screen_best >= top - cm):
        return "confirming"
    return "tuned_only"


def family_view(f: FamilyState, top: float | None, sigma_seed: float) -> dict:
    """What a REFINE prompt may show about a family: confirmed scores only (spec 11.1 and 14.1: nothing single-seed enters a prompt as established).

    Raises ValueError for a family that is not confirmed, so an unconfirmed family can never reach a REFINE prompt through this function.
    """
    if f.n_confirm < 3 or f.confirmed_mean is None:
        raise ValueError(f"{f.family} is not confirmed: no REFINE view")
    return {
        "family": f.family,
        "confirmed_mean": f.confirmed_mean,
        "confirmed_se": float(sigma_seed / math.sqrt(max(f.n_confirm, 1))),
        "n_seeds": f.n_confirm,
        "top_mean": top,
        "earlier_changes": [
            {"effect": h.get("effect"), "se": h.get("se"), "accepted": h.get("accepted")}
            for h in f.refinement_history
            if h.get("effect") is not None
        ],
    }


def top_confirmed(fams: dict[str, FamilyState]) -> float | None:
    v = [
        f.confirmed_mean for f in fams.values() if f.n_confirm >= 3 and f.confirmed_mean is not None
    ]
    return max(v) if v else None


def menu_select(fams: dict[str, FamilyState], cfg: dict | None = None) -> dict:
    """Menu of spec 5.2: confirmed families within ``menu_margin`` of the top; if fewer than ``menu_min_families`` qualify, the top N confirmed families, the ones outside the margin flagged."""
    c = merged(cfg)["controller"]
    conf = {n: f for n, f in fams.items() if f.n_confirm >= 3 and f.confirmed_mean is not None}
    if not conf:
        return {"menu": [], "flagged": [], "top": None, "within_margin": []}
    top = max(f.confirmed_mean for f in conf.values())
    ranked = sorted(conf, key=lambda n: (-conf[n].confirmed_mean, n))
    within = [n for n in ranked if conf[n].confirmed_mean >= top - c["menu_margin"]]
    if len(within) >= c["menu_min_families"]:
        return {"menu": within, "flagged": [], "top": top, "within_margin": within}
    menu = ranked[: c["menu_min_families"]]
    return {
        "menu": menu,
        "flagged": [n for n in menu if n not in within],
        "top": top,
        "within_margin": within,
    }


def estimate_sigma(fams: dict[str, FamilyState], cfg: dict | None = None) -> float:
    """sigma_seed: pooled within-family SD of confirmation runs; the prior (0.005) until 3 families are confirmed (spec 9.5)."""
    c = merged(cfg)["controller"]
    sds = [f.confirmed_sd for f in fams.values() if f.n_confirm >= 3 and f.confirmed_sd is not None]
    if len(sds) < 3:
        return float(c["sigma_seed_prior"])
    return float(math.sqrt(np.mean(np.square(sds))))


def acceptance(
    child_vals: list[float], parent_mean: float, sigma_seed: float, cfg: dict | None = None
) -> dict:
    """Accept a child only if its confirmed mean exceeds the parent's by more than max(noise_floor, 2 * se_diff), se_diff = sigma_seed * sqrt(2/3) (spec 9.5)."""
    c = merged(cfg)["controller"]
    se = float(sigma_seed * math.sqrt(2.0 / 3.0))
    thr = float(max(c["noise_floor"], 2.0 * se))
    eff = float(np.mean(child_vals) - parent_mean)
    return {"accepted": bool(eff > thr), "effect": eff, "se": se, "threshold": thr}


def decide_screen(results: list[dict], baseline: dict, cfg: dict | None = None) -> dict:
    """Adoption rule of the procedure screen (spec 13.2).

    results: [{option, anchor, seed, val_r}] (seeds 1 and 2); baseline: {anchor: {seed: val_r}} (the anchors' confirmation runs on the same seeds).
    effect(option, anchor) = mean(option seeds) - mean(baseline seeds); adopt an option whose mean over anchors exceeds 0.005 with the same sign on every anchor; at most 2.
    Returns {"effects": {option: {anchor: effect, "mean": m}}, "adopted": [...], "not_testable": [...], "needs_combination": bool}.
    """
    c = merged(cfg)["controller"]
    effects: dict[str, dict] = {}
    for opt in sorted({r["option"] for r in results}):
        per: dict[str, float] = {}
        for a in sorted({r["anchor"] for r in results}):
            v = [
                r["val_r"]
                for r in results
                if r["option"] == opt and r["anchor"] == a and r["val_r"] is not None
            ]
            b = [x for s, x in (baseline.get(a) or {}).items() if x is not None]
            if v and b:
                per[a] = float(np.mean(v) - np.mean(b))
        if per:
            per["mean"] = float(np.mean([x for k, x in per.items() if k != "mean"]))
            effects[opt] = per
    adopt = [
        o
        for o, e in effects.items()
        if e["mean"] > c["noise_floor"] and all(x > 0 for k, x in e.items() if k != "mean")
    ]
    adopt = sorted(adopt, key=lambda o: -effects[o]["mean"])[:2]
    notest = [
        o
        for o in NOT_TESTABLE_IN_SILICO
        if o in effects and o not in adopt and abs(effects[o]["mean"]) <= c["noise_floor"]
    ]
    return {
        "effects": effects,
        "adopted": adopt,
        "not_testable": notest,
        "needs_combination": len(adopt) == 2,
    }


def option_recipe(base: dict, option: str, cfg: dict | None = None) -> dict:
    """The anchor's best recipe with one procedure option applied (spec 13.1). multitask falls back to SWA over the last 25% of epochs when no extra label columns exist."""
    c = merged(cfg)
    r = copy.deepcopy(base)
    if option == "huber":
        r["loss"] = {"name": "huber", "delta": 1.0}
    elif option == "hetero":
        r["loss"] = {"name": "hetero"}
    elif option == "corr":
        r["loss"] = {"name": "mse_corr", "corr_weight": 0.5}
    elif option == "binned":
        r["loss"] = {"name": "binned", "n_bins": 32, "smooth": True}
    elif option == "ema":
        r["ema"] = tuner.EMA_DECAY
    elif option == "multitask":
        extra = list(c["data"].get("extra_tasks") or [])
        if extra:
            r["tasks"] = [c["data"].get("primary_task", "K562")] + extra
        else:
            r["swa_last_frac"] = 0.25
    else:
        raise ValueError(option)
    return r


def adopted_defaults(adopted: list[str], cfg: dict | None = None) -> dict:
    """Recipe defaults implied by the adopted options: {"loss": {...}, "ema": None | decay, "swa_last_frac": ...}; the orchestrator stores them as ``recipe_defaults`` (+ screen_done)."""
    out: dict = {"loss": {"name": "mse"}, "ema": None, "screen_done": True}
    for o in adopted:
        r = option_recipe({"loss": {"name": "mse"}, "ema": None, "tasks": []}, o, cfg)
        if r["loss"]["name"] != "mse":
            out["loss"] = r["loss"]
        if r.get("ema"):
            out["ema"] = r["ema"]
        if r.get("swa_last_frac"):
            out["swa_last_frac"] = r["swa_last_frac"]
        if len(r.get("tasks", [])) > 1:
            out["tasks"] = r["tasks"]
    return out


# --------------------------------------------------------------------------------------------------------------------------------- forecast


def forecast(
    cfg: dict | None = None,
    explore_fraction: float | None = None,
    deep_tune_share: float | None = None,
) -> dict:
    """Section 5.7 arithmetic: new families, refinements and deep-tune recipes the main budget buys at a given explore fraction (half of new families kept, half of refinements screen well)."""
    c = merged(cfg)
    b = c["balance"]
    B = float(c["controller"]["budget"]["main"])
    e = (
        explore_fraction
        if explore_fraction is not None
        else (
            b["explore_fraction"]
            if b["mode"] == "fixed"
            else (b["explore_start"] + b["explore_end"]) / 2
        )
    )
    d = deep_tune_share if deep_tune_share is not None else b["deep_tune_share"]
    cost = c["controller"]["expected_cost"]
    return {
        "explore_fraction": e,
        "new_families": e * B / cost["explore"],
        "refinements": (1 - e) * (1 - d) * B / cost["refine"],
        "deep_tune_recipes": (1 - e) * d * B / 1.0,
        "main_budget": B,
    }


def plan(cfg: dict | None = None) -> str:
    """The forecast table of spec 5.7 for explore fractions 0.8 / 0.6 / 0.4 / 0.2 and for the configured balance."""
    c = merged(cfg)
    lines = [
        f"Forecast for the main budget of {c['controller']['budget']['main']} run-equivalents "
        f"(new family ~{c['controller']['expected_cost']['explore']:g}, refinement ~{c['controller']['expected_cost']['refine']:g}, deep-tune recipe 1; deep_tune_share {c['balance']['deep_tune_share']:g})",
        "explore_fraction  new families  refinements  deep-tune recipes",
    ]
    for e in (0.8, 0.6, 0.4, 0.2):
        f = forecast(c, e)
        lines.append(
            f"{e:>16.1f}  {round(f['new_families']):>12d}  {round(f['refinements']):>11d}  {round(f['deep_tune_recipes']):>17d}"
        )
    f = forecast(c)
    lines.append(
        f"configured ({c['balance']['mode']}, mean explore {f['explore_fraction']:.2f}): {f['new_families']:.1f} new families, {f['refinements']:.1f} refinements, {f['deep_tune_recipes']:.1f} deep-tune recipes"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------------------------- novelty of sketches


def _tokens(text: str) -> set[str]:
    return {
        w for w in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(w) > 2
    }


def sketch_novelty(sketch: dict, fams: dict[str, FamilyState], summaries: list[dict]) -> float:
    """Spec 5.4: 2 if the sketch's family is empty, 1 if it has fewer than 2 candidates, plus the distance to the nearest existing candidate (1 - best Jaccard of word sets)."""
    fam = sketch.get("family", "")
    fs = fams.get(fam)
    n = fs.n_candidates if fs else 0
    base = 2.0 if n == 0 else (1.0 if n < 2 else 0.0)
    toks = _tokens(f"{sketch.get('name', '')} {sketch.get('mechanism', '')}")
    best = 0.0
    for s in summaries:
        o = _tokens(s.get("summary", ""))
        if toks and o:
            best = max(best, len(toks & o) / len(toks | o))
    return base + (1.0 - best)


# --------------------------------------------------------------------------------------------------------------------------------- the controller


class _Ctx:
    """Mutable per-decide bookkeeping: budget left per phase (all-or-nothing reservations), call slots, queue capacity."""

    def __init__(self, remaining: dict[str, float], calls: int, queue: int) -> None:
        self.remaining = remaining
        self.calls = calls
        self.queue = queue
        self.explored: set[str] = set()

    def reserve(self, phase: str, n: float) -> bool:
        if self.remaining.get(phase, 0.0) + 1e-9 >= n:
            self.remaining[phase] -= n
            return True
        return False


class Controller:
    """Owns the explore-exploit balance. Never asks the LLM what to do next; it gives the LLM one assignment and decides everything else from the ledger."""

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = merged(cfg)
        self.seed = int(self.cfg["controller"]["seed"])
        self.guard_state: dict = {}

    # ---------------------------------------------------------------- state and randomness

    def rng(self, step: int, tag: str) -> np.random.Generator:
        """Generator that is a pure function of (seed, step, tag): decisions replay exactly and nothing hidden accumulates."""
        return np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence([self.seed, int(step), zlib.crc32(tag.encode())])
            )
        )

    def state_dict(self) -> dict:
        return copy.deepcopy(self.guard_state)

    def load_state_dict(self, d: dict | None) -> None:
        self.guard_state = copy.deepcopy(d or {})

    def replay(self, decisions: list[dict]) -> None:
        """Rebuild ``guard_state`` from recorded decisions (NOTE actions of kind "guard_state", in order)."""
        for d in decisions:
            tgt = d.get("target", d)
            if d.get("mode", "NOTE") == "NOTE" and tgt.get("kind") == "guard_state":
                self.guard_state = copy.deepcopy(tgt["state"])

    # ---------------------------------------------------------------- derived views of a snapshot

    def _balance(self, snap: Snapshot) -> dict:
        return {**self.cfg["balance"], **(snap.balance or {})}

    def prepare(self, snap: Snapshot) -> dict[str, FamilyState]:
        """Families with derived lifecycle states (empty taxonomy families added)."""
        fams = {n: copy.copy(f) for n, f in snap.families.items()}
        for n in list(snap.taxonomy or TAXONOMY):
            fams.setdefault(n, FamilyState(family=n))
        for n in snap.anchors:
            fams.setdefault(n, FamilyState(family=n))
            fams[n].anchor = True
        fams = {
            n: fams[n] for n in sorted(fams)
        }  # a fixed order: nothing may depend on dict or set iteration order, or decisions would not replay
        top = top_confirmed(fams)
        for f in fams.values():
            f.state = derive_family_state(f, top, self.cfg)
        return fams

    def sigma(self, snap: Snapshot, fams: dict[str, FamilyState]) -> float:
        return float(snap.sigma_seed) if snap.sigma_seed else estimate_sigma(fams, self.cfg)

    def budget_remaining(self, snap: Snapshot, phase: str) -> float:
        """Run-equivalents left in a phase, counting every open run as 1.0 (a run's cost is only known when it finishes, and never exceeds 1)."""
        b, c = snap.budgets or self.cfg["controller"]["budget"], self.cfg["controller"]
        sp, op = snap.spend, snap.open_runs

        def used(p: str) -> float:
            return float(sp.get(p, 0.0)) + float(op.get(p, 0.0))

        if phase == "A":
            return float(b["anchors"] + c["anchor_borrow"]) - used("A")
        if phase == "S":
            return float(b["screen"] + c["screen_borrow"]) - used("S")
        if phase == "P":
            return float(b["procedures"]) - used("P")
        if phase == "M":
            eff = float(b["main"]) + sum(
                float(b[k]) - used(p)
                for k, p in (("anchors", "A"), ("screen", "S"), ("procedures", "P"))
            )
            return eff - used("M")
        if phase == "D":
            left = max(0.0, self.budget_remaining(snap, "M")) if c["final_uses_leftover"] else 0.0
            return float(b["final"]) + left - used("D")
        raise ValueError(phase)

    def main_budget_effective(self, snap: Snapshot) -> float:
        b = snap.budgets or self.cfg["controller"]["budget"]
        used = lambda p: float(snap.spend.get(p, 0.0)) + float(snap.open_runs.get(p, 0.0))  # noqa: E731
        return float(b["main"]) + sum(
            float(b[k]) - used(p)
            for k, p in (("anchors", "A"), ("screen", "S"), ("procedures", "P"))
        )

    # ---------------------------------------------------------------- phases and stop rule

    def _screen_enabled(self) -> bool:
        return (
            bool(self.cfg["phases"].get("screen"))
            and float(self.cfg["controller"]["budget"].get("screen", 0)) > 0
        )

    def _procedures_enabled(self) -> bool:
        return (
            bool(self.cfg["phases"].get("procedures"))
            and bool(self.cfg["procedure_mode"].get("enabled"))
            and float(self.cfg["controller"]["budget"].get("procedures", 0)) > 0
        )

    def anchors_done(self, snap: Snapshot, fams: dict[str, FamilyState]) -> bool:
        if all(fams[a].n_confirm >= 3 for a in snap.anchors):
            return True
        # stuck: no budget left to finish them and nothing in flight
        return (
            self.budget_remaining(snap, "A") < 3
            and snap.open_runs.get("A", 0) == 0
            and not snap.new_candidates_unplanned
        )

    def finish_cost(self, f: FamilyState) -> float:
        """Run-equivalents still needed to take a family's started lineage to its end (all 8 recipes, a second chance if parked, confirmation, an in-flight refinement child).

        New work is only started if the budget left, after setting this aside, still covers it: a family that is started is a family that finishes (a half-tuned family
        is wasted budget). Open runs are already counted as spent, so only runs not yet created are counted here.
        """
        per = float(self.cfg["tuner"]["recipes_per_family"])
        first = float(self.cfg["tuner"]["first_look"])
        rc = float(
            self.cfg["controller"]["expected_run_cost"]
        )  # a run costs less than 1 when it stops early; 1.0 is the budget ceiling used for open runs
        cost = 0.0
        pr = f.pending_refinement
        if pr is not None and not pr.outcome_recorded:
            cost += max(3.0 - pr.recipes_planned, 0.0) + (0.0 if pr.confirm_planned else 3.0)
        conf_left = max(3.0 - (f.confirm_open + len(f.confirm_vals)), 0.0)
        if f.state == "first_look":
            # kept (8 recipes + confirmation) or parked (second chance) is not known yet: weight them equally
            kept = max(per - max(f.recipes_planned, 0), 0.0) + conf_left
            parked = first + (0.0 if f.anchor else 0.0)
            cost += kept if f.anchor else 0.5 * (kept + parked)
        elif f.state == "tuning":
            cost += max(per - max(f.recipes_planned, 0), 0.0) + conf_left
        elif f.state == "parked":
            cost += (
                0.0 if f.second_chance_used else first
            )  # one second chance, then it is retired or tuned
        elif f.state == "confirming":
            cost += conf_left
        d = f.deep
        if f.state in ("confirmed", "plateaued") and d.confirm_recipe is not None:
            cost += max(3.0 - (d.confirm_open + len(d.confirm_vals)), 0.0)
        return cost * rc

    def committed_cost(self, fams: dict[str, FamilyState]) -> float:
        return float(sum(self.finish_cost(f) for f in fams.values() if f.state != "empty"))

    def _pending_work(self, snap: Snapshot, fams: dict[str, FamilyState], rem: float) -> bool:
        """True if some family can still take its next step (create runs that fit ``rem``, or write an outcome note) now that nothing is open."""
        per = int(self.cfg["tuner"]["recipes_per_family"])
        if any(u.get("origin") != "anchor" for u in snap.new_candidates_unplanned) and rem >= 3:
            return True
        for f in fams.values():
            if f.state == "empty":
                continue
            pr = f.pending_refinement
            if pr is not None and not pr.outcome_recorded:
                if pr.confirm_open == 0 and len(pr.confirm_vals) >= 3:
                    return True
                if pr.screen_open == 0 and pr.recipes_planned >= 3 and not pr.confirm_planned:
                    return True  # either a confirmation (3) or a rejection note
            d = f.deep
            if (
                f.state in ("confirmed", "plateaued", "retired")
                and d.confirm_recipe is None
                and d.batch_runs >= int(self.cfg["controller"]["deep_batch"])
                and d.batch_open == 0
            ):
                return True
            if (
                f.state in ("confirmed", "plateaued")
                and d.confirm_recipe is not None
                and d.confirm_open == 0
                and len(d.confirm_vals) >= 3
            ):
                return True
            if (
                f.state == "parked"
                and not f.second_chance_used
                and f.first_look_open == 0
                and rem >= 3
            ):
                return True
            if (
                f.state == "tuning"
                and f.recipes_open == 0
                and f.recipes_planned < per
                and rem >= (2 if f.recipes_planned < 5 else per - f.recipes_planned)
            ):
                return True
            if f.state == "confirming" and f.confirm_open + len(f.confirm_vals) < 3 and rem >= 3:
                return True
        return False

    def stop_check(self, snap: Snapshot, fams: dict[str, FamilyState]) -> tuple[bool, str, bool]:
        """(stop starting new work in the main search?, reason, flat). Spec 5.5: budget spent, or at least 8 families tuned and over the last 50 run-equivalents no
        new family became menu-eligible and no refinement was accepted (then the report states the space is flat at this data size).

        "Budget spent" means that what is left after setting aside the cost of finishing every started family is below the smallest start cost of any new work.
        """
        latched = self.guard_state.get("stopped")
        if latched:  # once the main search has stopped it stays stopped (a finishing run's saving must not reopen it)
            return True, latched["reason"], bool(latched["flat"])
        if snap.stop_requested:
            return True, "stopped by the operator", False
        ms = self.cfg["controller"]["min_start"]
        if self.budget_remaining(snap, "M") - self.committed_cost(fams) < min(ms.values()):
            return True, "main budget spent", False
        fl = self.cfg["controller"]["flat_stop"]
        tuned = sum(
            1
            for f in fams.values()
            if f.state in ("confirming", "confirmed", "plateaued", "retired", "tuned_only")
        )
        if (
            tuned >= fl["min_families_tuned"]
            and (snap.cum_main_spend - snap.last_progress_spend) >= fl["window"]
            and snap.cum_main_spend >= fl["window"]
        ):
            return (
                True,
                f"flat: {tuned} families tuned and no new menu family or accepted refinement in the last {fl['window']} run-equivalents",
                True,
            )
        return False, "", False

    def current_phase(self, snap: Snapshot, fams: dict[str, FamilyState] | None = None) -> str:
        fams = fams or self.prepare(snap)
        if snap.final_menu_done:
            return "DONE"
        if not self.anchors_done(snap, fams):
            return "A"
        if self._screen_enabled() and not snap.screen_decision_done:
            return "S"
        if self._procedures_enabled() and not snap.procedures.get("done"):
            return "P"
        stop, _, _ = self.stop_check(snap, fams)
        if not stop:
            return "M"
        if snap.open_runs.get("M", 0) > 0 or self._pending_work(
            snap, fams, self.budget_remaining(snap, "M")
        ):
            return "M_FINISH"
        return "D"

    def is_finished(self, snap: Snapshot) -> bool:
        return self.current_phase(snap) == "DONE"

    # ---------------------------------------------------------------- balance (5.7) and guards

    def explore_target(self, snap: Snapshot, fams: dict[str, FamilyState]) -> tuple[float, dict]:
        bal = self._balance(snap)
        g = self.guard_state
        if bal["mode"] == "fixed":
            e = float(bal["explore_fraction"])
        else:
            B = max(self.main_budget_effective(snap), 1e-9)
            p = min(max(snap.cum_main_spend / B, 0.0), 1.0)
            e = float(bal["explore_start"] + (bal["explore_end"] - bal["explore_start"]) * p)
        e += float(g.get("explore_offset", 0.0)) if bal["guards"] else 0.0
        tuned = sum(
            1
            for f in fams.values()
            if f.recipes_planned >= self.cfg["tuner"]["recipes_per_family"] and f.recipes_open == 0
        )
        floor_active = tuned < bal["breadth_floor"]
        if floor_active:
            e = max(e, 0.5)
        e = min(max(e, bal["explore_min"]), bal["explore_max"])
        d = min(
            max(
                float(bal["deep_tune_share"])
                + (float(g.get("deep_offset", 0.0)) if bal["guards"] else 0.0),
                0.0,
            ),
            0.6,
        )
        return e, {"deep_share": d, "floor_active": floor_active, "tuned": tuned}

    def targets(self, snap: Snapshot, fams: dict[str, FamilyState]) -> dict[str, float]:
        e, info = self.explore_target(snap, fams)
        d = info["deep_share"]
        return {"explore": e, "refine": (1 - e) * (1 - d), "deep_tune": (1 - e) * d}

    def _update_guards(self, snap: Snapshot, notes: list[Action]) -> None:
        """The four guards of 5.7 (window 4, step 0.2). Each firing is a NOTE with the outcomes that triggered it; a mark prevents re-firing on the same outcomes."""
        bal = self._balance(snap)
        g = self.guard_state
        if not bal["guards"]:
            return
        W, step = int(bal["guard_window"]), float(bal["guard_step"])
        new_o = list(snap.recent_new_family_outcomes)
        exp_o = list(snap.recent_exploit_outcomes)
        explore_drought = len(new_o) - g.get("explore_mark", 0) >= W and all(
            o in CHECK_FAILED for o in new_o[-W:]
        )
        exploit_drought = len(exp_o) - g.get("exploit_mark", 0) >= W and all(
            not o["accepted"] for o in exp_o[-W:]
        )
        deep_o = [o for o in exp_o if o["kind"] == "deep_tune"]
        recipes_matter = (
            len(deep_o) - g.get("deep_mark", 0) >= W
            and sum(1 for o in deep_o[-W:] if o["accepted"]) >= 2
        )
        fired: list[dict] = []
        if explore_drought and exploit_drought:
            g["explore_mark"], g["exploit_mark"] = len(new_o), len(exp_o)
            fired.append(
                {
                    "guard": "both_droughts",
                    "action": "none: the stop rule decides",
                    "outcomes": new_o[-W:],
                }
            )
        else:
            if explore_drought:
                g["explore_offset"] = round(g.get("explore_offset", 0.0) - step, 6)
                g["explore_mark"] = len(new_o)
                fired.append(
                    {
                        "guard": "explore_drought",
                        "action": f"explore share -{step}",
                        "outcomes": new_o[-W:],
                    }
                )
            if exploit_drought:
                g["explore_offset"] = round(g.get("explore_offset", 0.0) + step, 6)
                g["exploit_mark"] = len(exp_o)
                fired.append(
                    {
                        "guard": "exploit_drought",
                        "action": f"explore share +{step}",
                        "outcomes": [o["accepted"] for o in exp_o[-W:]],
                    }
                )
        if recipes_matter:
            g["deep_offset"] = min(
                round(g.get("deep_offset", 0.0) + step, 6), 0.6 - float(bal["deep_tune_share"])
            )
            g["deep_mark"] = len(deep_o)
            fired.append(
                {
                    "guard": "recipes_matter",
                    "action": f"deep_tune_share +{step} (max 0.6)",
                    "outcomes": [o["accepted"] for o in deep_o[-W:]],
                }
            )
        for f in fired:
            notes.append(
                Action(
                    "NOTE",
                    {
                        "kind": "guard",
                        **f,
                        "explore_offset": g.get("explore_offset", 0.0),
                        "deep_offset": g.get("deep_offset", 0.0),
                    },
                    reason=f"guard {f['guard']} fired: {f['action']}",
                    executable=False,
                )
            )

    # ---------------------------------------------------------------- 12.1 adaptation

    def adaptation(self, snap: Snapshot, fams: dict[str, FamilyState]) -> dict:
        a = self.cfg["adaptation"]
        g = self.guard_state
        out: dict[str, Any] = {
            "extra_constraints": [],
            "forced_families": None,
            "failure_view": False,
            "extend_do_not_repeat": None,
            "rotate_reference": False,
            "speed_note": False,
            "signals": {},
        }
        W = int(a["window"])
        ex = list(snap.recent_calls.get("EXPLORE", []))[-W:]
        allc = [c for m in ("EXPLORE", "REFINE", "SKETCH") for c in snap.recent_calls.get(m, [])]
        # collapse: EXPLORE candidates tagged into a family already holding >= collapse_family_size candidates, 2 in a row
        thr = int(a["collapse_family_size"])
        run = 0
        for c in reversed(ex):
            if c.get("landed_family_size", 0) >= thr:
                run += 1
            else:
                break
        out["signals"]["collapse_in_a_row"] = run
        if run >= 2:
            last = ex[-1]
            out["extra_constraints"] += [
                f"the mechanism that must do the integration: {last.get('mechanism', 'the sketched mechanism')}",
                f"the block that must not appear: the core block of {last.get('landed_family', 'the crowded family')}",
            ]
        if run >= 3:
            out["drop_sketch"] = True
        # low novelty: share of EXPLORE candidates that land in their assigned family
        if len(ex) >= 4:
            share = sum(1 for c in ex if c.get("assigned_family") == c.get("landed_family")) / len(
                ex
            )
            out["signals"]["landed_in_assigned_share"] = round(share, 3)
            if share < a["novelty_min"]:
                least = sorted(fams, key=lambda n: (fams[n].n_candidates, n))[:6]
                out["forced_families"] = least
        # gate failures: on above gate_fail_on, off below gate_fail_off (hysteresis)
        if allc:
            rate = sum(1 for c in allc if not c.get("first_attempt_gate_ok", True)) / len(allc)
            out["signals"]["gate_failure_rate"] = round(rate, 3)
            on = bool(g.get("failure_view_on", False))
            if rate > a["gate_fail_on"]:
                on = True
            elif rate < a["gate_fail_off"]:
                on = False
            g["failure_view_on"] = on
            out["failure_view"] = on
        dup = sum(1 for c in allc if c.get("duplicate"))
        out["signals"]["duplicates"] = dup
        if dup >= 3:
            out["extend_do_not_repeat"] = 20
            out["rotate_reference"] = True
        spd = sum(1 for c in allc if c.get("speed_fail"))
        out["signals"]["speed_failures"] = spd
        if spd >= 2:
            out["speed_note"] = True
        return out

    # ---------------------------------------------------------------- decide

    def decide(
        self, snap: Snapshot, free_job_slots: int | None = None, free_call_slots: int | None = None
    ) -> list[Action]:
        """One pass of the controller (spec 5.3 priority: pending CONFIRM, pending TUNE, pending first looks, then new work by the balance rule).

        free_job_slots: unused job slots (informative; new work is gated by the queue depth). free_call_slots: LLM calls that may start now.
        """
        if snap.guard_state:
            self.guard_state = copy.deepcopy(snap.guard_state)
        ctl = self.cfg["controller"]
        fams = self.prepare(snap)
        notes: list[Action] = []
        phase = self.current_phase(snap, fams)
        gs_before = copy.deepcopy(self.guard_state)
        if phase != self.guard_state.get("phase"):
            notes.append(
                Action(
                    "NOTE",
                    {"kind": "phase", "from": self.guard_state.get("phase"), "to": phase},
                    reason=f"phase {self.guard_state.get('phase')} -> {phase}",
                    executable=False,
                )
            )
            self.guard_state["phase"] = phase
        for n, f in fams.items():
            cached = snap.families.get(n)
            if cached is not None and cached.state != f.state:
                notes.append(
                    Action(
                        "NOTE",
                        {"kind": "family_state", "family": n, "from": cached.state, "to": f.state},
                        reason=f"family {n}: {cached.state} -> {f.state}",
                        executable=False,
                    )
                )
        calls = (
            ctl["max_parallel_calls"] - snap.calls_in_flight
            if free_call_slots is None
            else int(free_call_slots)
        )
        open_total = sum(int(v) for v in snap.open_runs.values())
        mpj = int(ctl["max_parallel_jobs"])
        free = (mpj - snap.runs_in_flight) if free_job_slots is None else int(free_job_slots)
        queue = (
            free + (int(ctl["queue_depth"]) - 1) * mpj - max(open_total - snap.runs_in_flight, 0)
        )
        rem = {p: self.budget_remaining(snap, p) for p in ("A", "S", "P", "M")}
        rem["D"] = self.budget_remaining(snap, "D") if phase in ("D", "DONE") else 0.0
        ctx = _Ctx(rem, max(calls, 0), queue)
        confirms: list[Action] = []
        tunes: list[Action] = []
        firsts: list[Action] = []
        newwork: list[Action] = []
        sigma = self.sigma(snap, fams)
        adapt = self.adaptation(snap, fams)
        if adapt["signals"] and (
            adapt["extra_constraints"]
            or adapt["forced_families"]
            or adapt["failure_view"]
            or adapt["speed_note"]
            or adapt["extend_do_not_repeat"]
        ):
            notes.append(
                Action(
                    "NOTE",
                    {"kind": "adaptation", **adapt},
                    reason="prompt adaptation (12.1): "
                    + ", ".join(f"{k}={v}" for k, v in adapt["signals"].items()),
                    executable=False,
                )
            )
        self._update_guards(snap, notes)

        if phase in ("M_FINISH", "D", "DONE") and not self.guard_state.get("stop_logged"):
            _, why, flat = self.stop_check(snap, fams)
            self.guard_state["stopped"] = {"reason": why, "flat": flat}
            notes.append(
                Action(
                    "NOTE",
                    {"kind": "stop", "reason": why, "flat": flat},
                    reason=f"main search stopped: {why}"
                    + (
                        "; the report must state that the space is flat at this data size"
                        if flat
                        else ""
                    ),
                    executable=False,
                )
            )
            self.guard_state["stop_logged"] = True
        if phase == "DONE":
            return notes + self._flush_state(notes, gs_before)
        if phase == "D":
            ctx.remaining["M"] = (
                0.0  # nothing new in the main search; only pending outcome notes can still be written
            )
            sigma = self.sigma(snap, fams)
            for f in fams.values():
                if f.pending_refinement is not None and not f.pending_refinement.outcome_recorded:
                    self._progress_child(snap, f, f.pending_refinement, ctx, [], sigma, notes)
                if f.state in ("confirmed", "plateaued", "retired"):
                    self._progress_deep(snap, f, ctx, [], sigma, notes)
            return notes + self._phase_d(snap, fams, ctx) + self._flush_state(notes, gs_before)

        # ---- 1. facts-driven progress of every family (CONFIRM > TUNE > first looks)
        order = sorted(fams, key=lambda n: (0 if fams[n].anchor else 1, n))
        unplanned = {u["code_hash"]: u for u in snap.new_candidates_unplanned}
        for u in snap.new_candidates_unplanned:
            self._plan_candidate(snap, fams, u, ctx, firsts, phase)
        for n in order:
            f = fams[n]
            if f.state == "empty":
                continue
            self._progress_family(snap, f, ctx, confirms, tunes, sigma, notes, phase, unplanned)

        # ---- 2. phase S / P work
        pre: list[Action] = []
        if phase == "S":
            pre += self._phase_s(snap, fams, ctx)
        if phase == "P":
            pre += self._phase_p(snap, fams, ctx)

        # ---- 3. new work
        if phase == "A":
            self._phase_a(snap, fams, ctx, newwork)
        elif phase == "M":
            self._repair(snap, ctx, newwork)
            self._new_work(snap, fams, ctx, newwork, adapt)
        elif phase == "M_FINISH":
            self._repair(
                snap, ctx, newwork
            )  # finish gate repairs already started; no new families or refinements
        for a in newwork:
            if a.mode in ("SKETCH", "EXPLORE", "REFINE", "REPAIR") and adapt["signals"]:
                a.adaptation = {k: v for k, v in adapt.items()}
        acts = notes + confirms + tunes + firsts + pre + newwork
        return acts + self._flush_state(notes + acts, gs_before)

    def _flush_state(self, acts: list[Action], before: dict) -> list[Action]:
        if self.guard_state != before:
            return [
                Action(
                    "NOTE",
                    {"kind": "guard_state", "state": copy.deepcopy(self.guard_state)},
                    reason="controller state changed",
                    executable=False,
                )
            ]
        return []

    # ---------------------------------------------------------------- run-creating helpers

    def _lineage_tag(self, f: FamilyState, phase: str) -> tuple[str, str]:
        """(phase, category) charged for a family's own screening and confirmation: anchors belong to phase A, every other family is exploration in phase M."""
        return ("A", "anchor") if f.anchor else ("M", "explore")

    def _tune_action(
        self,
        f: FamilyState,
        code_hash: str,
        recipes: list[dict],
        purpose: str,
        phase: str,
        category: str,
        reason: str,
        caps: list[float] | None = None,
        extra: dict | None = None,
    ) -> Action:
        t = {
            "code_hash": code_hash,
            "family": f.family,
            "recipe_ids": [tuner.recipe_hash(r) for r in recipes],
            "cap_multipliers": caps or [1.0] * len(recipes),
        }
        t.update(extra or {})
        return Action(
            "TUNE",
            t,
            reason=reason,
            recipes=recipes,
            seeds=[0] * len(recipes),
            purpose=purpose,
            category=category,
            phase=phase,
            expected_cost=float(len(recipes)),
        )

    def _plan_candidate(
        self,
        snap: Snapshot,
        fams: dict[str, FamilyState],
        u: dict,
        ctx: _Ctx,
        out: list[Action],
        phase: str,
    ) -> None:
        """A candidate that passed the gate and tagging gets its first recipes: 3 (first look) for a new family, the parent's best + 2 perturbations for a REFINE child,
        a single suggested recipe for an extra candidate in a family that is already past its first look."""
        f = fams.get(u["family"]) or FamilyState(family=u["family"])
        rng = self.rng(snap.step, f"plan:{u['code_hash']}")
        defaults = snap.recipe_defaults or {}
        hp, tun = u.get("hparams") or {}, u.get("tunable") or {}
        origin = u.get("origin", "explore")
        existing: set[str] = set()
        if origin == "refine":
            parent_best = f.best_recipe or tuner.default_recipe(f.family, self.cfg, hp, defaults)
            recs = tuner.refine_child_recipes(
                parent_best,
                f.family,
                self.cfg,
                rng,
                child_hparams=hp,
                tunable=tun,
                defaults=defaults,
                existing=existing,
            )
            if ctx.reserve("M", len(recs)):
                out.append(
                    self._tune_action(
                        f,
                        u["code_hash"],
                        recs,
                        "refine",
                        "M",
                        "refine",
                        f"REFINE child {u['code_hash'][:8]} of {f.family}: parent's best recipe plus two local perturbations (spec 8.2)",
                    )
                )
            return
        anchor = f.anchor or u.get("origin") == "anchor"
        ph, cat = ("A", "anchor") if anchor and f.n_confirm < 3 else ("M", "explore")
        if (
            f.n_candidates > 1
            and f.first_look_done >= int(self.cfg["tuner"]["first_look"])
            and not anchor
        ):
            recs = [
                tuner.fill_suggested(u.get("suggested_recipe"), f.family, self.cfg, hp, defaults)
            ]
            if ctx.reserve(ph, 1):
                out.append(
                    self._tune_action(
                        f,
                        u["code_hash"],
                        recs,
                        "tune",
                        ph,
                        cat,
                        f"extra candidate {u['code_hash'][:8]} in {f.family}: one run at its suggested recipe",
                    )
                )
            return
        recs = tuner.first_look_recipes(
            f.family,
            self.cfg,
            rng,
            suggested=u.get("suggested_recipe"),
            hparams=hp,
            tunable=tun,
            defaults=defaults,
        )
        if ctx.reserve(ph, len(recs)):
            out.append(
                self._tune_action(
                    f,
                    u["code_hash"],
                    recs,
                    "first_look",
                    ph,
                    cat,
                    f"first look of {f.family}: the proposer's suggested recipe, the family default and one draw from the prior (spec 5.2)",
                )
            )

    def _confirm_action(
        self,
        f: FamilyState,
        kind: str,
        recipe: dict,
        code_hash: str,
        phase: str,
        category: str,
        cap: float,
        reason: str,
    ) -> Action:
        seeds = tuner.seeds_for("confirm", self.cfg)
        return Action(
            "CONFIRM",
            {
                "code_hash": code_hash,
                "family": f.family,
                "recipe_hash": tuner.recipe_hash(recipe),
                "target": kind,
                "cap_multiplier": cap,
            },
            reason=reason,
            recipes=[copy.deepcopy(recipe) for _ in seeds],
            seeds=list(seeds),
            purpose="confirm",
            category=category,
            phase=phase,
            expected_cost=float(len(seeds)),
        )

    def _progress_family(
        self,
        snap: Snapshot,
        f: FamilyState,
        ctx: _Ctx,
        confirms: list[Action],
        tunes: list[Action],
        sigma: float,
        notes: list[Action],
        phase: str,
        unplanned: dict,
    ) -> None:
        c = self.cfg
        per_family = int(c["tuner"]["recipes_per_family"])
        ph, cat = self._lineage_tag(f, phase)
        defaults = snap.recipe_defaults or {}
        rng = self.rng(snap.step, f"progress:{f.family}")
        if f.best_code_hash in unplanned and f.recipes_planned == 0:
            return  # planned through _plan_candidate
        # ---- refinement child in flight
        pr = f.pending_refinement
        if pr is not None and not pr.outcome_recorded:
            self._progress_child(snap, f, pr, ctx, confirms, sigma, notes)
        # ---- deep tune in flight
        if f.state in ("confirmed", "plateaued", "retired"):
            self._progress_deep(snap, f, ctx, confirms, sigma, notes)
        if (
            f.state == "parked"
            and not f.second_chance_used
            and f.first_look_open == 0
            and f.first_look_done >= int(c["tuner"]["first_look"])
        ):
            fl_runs = [
                {
                    "recipe": r["recipe"],
                    "status": r["status"],
                    "train_r": r.get("train_r"),
                    "val_r": r["val_r"],
                    "censored": r.get("censored", False),
                    "still_improving": r.get("still_improving", False),
                }
                for r in self._first_look_details(f)
            ]
            recs, caps, tri = tuner.triage_recipes(
                fl_runs,
                f.family,
                c,
                rng,
                tunable=f.tunable,
                defaults=defaults,
                existing={x["recipe_hash"] for x in f.recipe_results},
            )
            if ctx.reserve(ph, len(recs)):
                tunes.append(
                    self._tune_action(
                        f,
                        f.best_code_hash or "",
                        recs,
                        "second_chance",
                        ph,
                        cat,
                        f"{f.family} parked (first look {f.first_look_best} more than {c['controller']['keep_margin']} below the top): one second chance with triage {tri['diagnoses']}",
                        caps,
                        {"triage": tri},
                    )
                )
            return
        if (
            f.state == "tuning"
            and f.recipes_open == 0
            and f.recipes_planned < per_family
            and f.first_look_done >= int(c["tuner"]["first_look"])
        ):
            existing = {x["recipe_hash"] for x in f.recipe_results}
            if f.recipes_planned < 5:
                n = min(5, per_family) - f.recipes_planned
                recs = tuner.prior_recipes(
                    f.family,
                    c,
                    rng,
                    n,
                    hparams=f.hparams,
                    tunable=f.tunable,
                    defaults=defaults,
                    existing=existing,
                )
                why = f"{f.family}: recipes {f.recipes_planned + 1}-{f.recipes_planned + n} are draws from the prior"
            else:
                n = per_family - f.recipes_planned
                best = f.best_recipe or tuner.default_recipe(f.family, c, f.hparams, defaults)
                recs = tuner.local_recipes(
                    best,
                    f.family,
                    c,
                    rng,
                    n,
                    tunable=f.tunable,
                    defaults=defaults,
                    existing=existing,
                )
                why = f"{f.family}: recipes {f.recipes_planned + 1}-{per_family} are local perturbations of the best recipe so far"
            if ctx.reserve(ph, len(recs)):
                tunes.append(
                    self._tune_action(f, f.best_code_hash or "", recs, "tune", ph, cat, why)
                )
            return
        if f.state == "confirming" and f.confirm_open + len(f.confirm_vals) < 3 and f.best_recipe:
            cap = (
                float(c["evaluator"].get("cap_extension", 1.5))
                if tuner.needs_cap_extension(
                    {"censored": f.censored, "improvement_per_epoch": self._best_improvement(f)},
                    f.censored_extended,
                )
                else 1.0
            )
            if ctx.reserve(ph, 3):
                confirms.append(
                    self._confirm_action(
                        f,
                        "family",
                        f.best_recipe,
                        f.best_code_hash or "",
                        ph,
                        cat,
                        cap,
                        f"{f.family}: tuning complete and within {c['controller']['confirm_margin']} of the best confirmed family{' (anchor)' if f.anchor else ''}: confirm on seeds 1-3",
                    )
                )

    def _best_improvement(self, f: FamilyState) -> float:
        return float(
            f.last_diagnostics.get(
                "improvement_per_epoch", 0.002 if f.last_diagnostics.get("still_improving") else 0.0
            )
        )

    def _first_look_details(self, f: FamilyState) -> list[dict]:
        """First-look runs as dicts for triage. FamilyState keeps only recipe_results; richer facts come from ``recipe_results[*]['recipe'/'train_r'/'censored']`` when the
        orchestrator adds them (it should: they are cheap to add), otherwise the diagnosis falls back to the status and val r."""
        out = []
        for r in f.recipe_results:
            if r.get("purpose") != "first_look":
                continue
            out.append(
                {
                    "recipe": r.get("recipe") or f.best_recipe or {},
                    "status": r["status"],
                    "train_r": r.get("train_r"),
                    "val_r": r["val_r"],
                    "censored": r.get("censored", False),
                    "still_improving": r.get("still_improving", False),
                }
            )
        return out

    def _progress_child(
        self,
        snap: Snapshot,
        f: FamilyState,
        ch: ChildInfo,
        ctx: _Ctx,
        confirms: list[Action],
        sigma: float,
        notes: list[Action],
    ) -> None:
        c = self.cfg["controller"]
        if ch.screen_open > 0 or ch.recipes_planned < 3:
            return
        if not ch.confirm_planned:
            best = max(ch.screen_scores) if ch.screen_scores else None
            ref = ch.parent_screen_best
            if best is None or (ref is not None and best < ref - c["refine_screen_margin"]):
                notes.append(
                    Action(
                        "NOTE",
                        {
                            "kind": "refinement_outcome",
                            "family": f.family,
                            "child": ch.code_hash,
                            "accepted": False,
                            "effect": None,
                            "se": None,
                            "reason": "screened poorly",
                            "screen_best": best,
                            "parent_screen_best": ref,
                        },
                        reason=f"refinement {ch.code_hash[:8]} of {f.family} rejected without confirmation: best screening score {best} against the parent's {ref}",
                        executable=False,
                    )
                )
            elif ctx.reserve("M", 3):
                confirms.append(
                    self._confirm_action(
                        f,
                        "child",
                        ch.best_recipe,
                        ch.code_hash,
                        "M",
                        "refine",
                        1.0,
                        f"refinement {ch.code_hash[:8]} of {f.family}: screening best {best:.4f} is within {c['refine_screen_margin']} of the parent's {ref}: confirm on seeds 1-3",
                    )
                )
            return
        if ch.confirm_open == 0 and len(ch.confirm_vals) >= 3:
            res = acceptance(
                ch.confirm_vals,
                ch.parent_confirmed_mean
                if ch.parent_confirmed_mean is not None
                else (f.confirmed_mean or 0.0),
                sigma,
                self.cfg,
            )
            notes.append(
                Action(
                    "NOTE",
                    {
                        "kind": "refinement_outcome",
                        "family": f.family,
                        "child": ch.code_hash,
                        "accepted": res["accepted"],
                        "effect": res["effect"],
                        "se": res["se"],
                        "threshold": res["threshold"],
                        "child_confirmed_mean": float(np.mean(ch.confirm_vals)),
                        "parent_confirmed_mean": ch.parent_confirmed_mean,
                    },
                    reason=f"refinement {ch.code_hash[:8]} of {f.family}: effect {res['effect']:+.4f} against threshold {res['threshold']:.4f} -> {'accepted' if res['accepted'] else 'rejected (stored as a negative result)'}",
                    executable=False,
                )
            )

    def _progress_deep(
        self,
        snap: Snapshot,
        f: FamilyState,
        ctx: _Ctx,
        confirms: list[Action],
        sigma: float,
        notes: list[Action],
    ) -> None:
        d = f.deep
        nb = int(self.cfg["controller"]["deep_batch"])
        if d.confirm_recipe is None:
            if (
                d.batch_runs >= nb and d.batch_open == 0
            ):  # a full batch of recipes has finished: compare with the previous best screening score
                prev = (
                    d.prev_screen_best if d.prev_screen_best is not None else (f.screen_best or 0.0)
                )
                best = max(d.batch_scores) if d.batch_scores else None
                rh = tuner.recipe_hash(d.batch_best_recipe) if d.batch_best_recipe else None
                if best is not None and best > prev + self.cfg["controller"]["noise_floor"]:
                    if ctx.reserve("M", 3):
                        confirms.append(
                            self._confirm_action(
                                f,
                                "deep",
                                d.batch_best_recipe,
                                f.best_code_hash or "",
                                "M",
                                "deep_tune",
                                1.0,
                                f"{f.family}: best deep-tune recipe {best:.4f} beats the previous best screening score {prev:.4f} by more than 0.005: confirm on seeds 1-3",
                            )
                        )
                else:
                    why = (
                        "no run finished validly"
                        if best is None
                        else f"best {best:.4f} does not beat {prev:.4f} + 0.005"
                    )
                    notes.append(
                        Action(
                            "NOTE",
                            {
                                "kind": "deep_outcome",
                                "family": f.family,
                                "batch": d.batch_index,
                                "recipe_hash": rh,
                                "accepted": False,
                                "effect": None if best is None else best - prev,
                                "reason": "no screening gain",
                            },
                            reason=f"{f.family}: deep-tune batch rejected ({why})",
                            executable=False,
                        )
                    )
        elif d.confirm_open == 0 and len(d.confirm_vals) >= 3:
            res = acceptance(d.confirm_vals, f.confirmed_mean or 0.0, sigma, self.cfg)
            notes.append(
                Action(
                    "NOTE",
                    {
                        "kind": "deep_outcome",
                        "family": f.family,
                        "batch": d.batch_index,
                        "recipe_hash": d.confirm_recipe_hash,
                        "accepted": res["accepted"],
                        "effect": res["effect"],
                        "se": res["se"],
                        "threshold": res["threshold"],
                    },
                    reason=f"{f.family}: deep-tune recipe confirmed {np.mean(d.confirm_vals):.4f} vs {f.confirmed_mean:.4f}: effect {res['effect']:+.4f} against {res['threshold']:.4f} -> {'accepted' if res['accepted'] else 'rejected'}",
                    executable=False,
                )
            )

    # ---------------------------------------------------------------- phase A

    def _phase_a(
        self, snap: Snapshot, fams: dict[str, FamilyState], ctx: _Ctx, out: list[Action]
    ) -> None:
        inflight = {t.get("family") for t in snap.calls_in_flight_targets}
        for a in snap.anchors:
            f = fams[a]
            if (
                f.n_candidates > 0
                or f.recipes_planned > 0
                or a in inflight
                or any(u["family"] == a for u in snap.new_candidates_unplanned)
                or f.gate_failed_open > 0
            ):
                continue
            src = self.cfg["controller"]["anchor_source"]
            if src == "llm" and ctx.calls <= 0:
                continue
            if src == "llm":
                ctx.calls -= 1
            out.append(
                Action(
                    "EXPLORE",
                    {
                        "family": a,
                        "sketch": None,
                        "anchor": True,
                        "source": src,
                        "extra_constraints": [],
                    },
                    reason=f"phase A: anchor family {a} (cnn_residual, rnn_hybrid, transformer_hybrid are always tuned and confirmed) from the {src} model",
                    category="anchor",
                    phase="A",
                )
            )

    # ---------------------------------------------------------------- phase S / P / D

    def _phase_s(self, snap: Snapshot, fams: dict[str, FamilyState], ctx: _Ctx) -> list[Action]:
        """Procedure screen (spec 13): 6 options x seeds 1, 2 x 3 anchors, paired with the anchors' confirmation seeds; adopt by decide_screen; a combination of two adopted
        options is run on the anchors and kept only if at least as good as the better single option."""
        if snap.open_runs.get("S", 0) > 0 or snap.screen_open > 0:
            return []
        seeds = [1, 2]
        have = {(r["anchor"], r["option"], r["seed"]) for r in snap.screen_results}
        out: list[Action] = []
        for a in snap.anchors:
            f = fams[a]
            if not f.best_recipe:
                continue
            recipes, sds, opts = [], [], []
            for opt in SCREEN_OPTIONS:
                for s_ in seeds:
                    if (a, opt, s_) not in have:
                        recipes.append(option_recipe(f.best_recipe, opt, self.cfg))
                        sds.append(s_)
                        opts.append(opt)
            if recipes and ctx.reserve("S", len(recipes)):
                out.append(
                    Action(
                        "SCREEN",
                        {
                            "kind": "run",
                            "anchor": a,
                            "code_hash": f.best_code_hash,
                            "options": opts,
                        },
                        reason=f"phase S: {len(SCREEN_OPTIONS)} options x seeds {seeds} on anchor {a}, paired with its confirmation seeds",
                        recipes=recipes,
                        seeds=sds,
                        purpose="screen",
                        category="screen",
                        phase="S",
                        expected_cost=float(len(recipes)),
                    )
                )
        if out or snap.screen_decision_done or not snap.screen_results:
            return out
        singles = [r for r in snap.screen_results if r["option"] in SCREEN_OPTIONS]
        combos = [r for r in snap.screen_results if r["option"] == "combo"]
        dec = decide_screen(singles, snap.screen_baseline, self.cfg)
        adopted = list(dec["adopted"])
        if dec["needs_combination"] and not combos:
            recipes, sds = [], []
            runs = []
            for a in snap.anchors:
                f = fams[a]
                for s_ in seeds:
                    rec = f.best_recipe
                    for opt in adopted:
                        rec = option_recipe(rec, opt, self.cfg)
                    recipes.append(rec)
                    sds.append(s_)
                    runs.append(a)
            if ctx.reserve("S", len(recipes)):
                return [
                    Action(
                        "SCREEN",
                        {
                            "kind": "run",
                            "anchor": None,
                            "options": ["combo"] * len(recipes),
                            "anchors_per_recipe": runs,
                            "adopted": adopted,
                        },
                        reason=f"phase S: the two adopted options {adopted} are combined and run on the three anchors (2 seeds); kept only if at least as good as the better single option",
                        recipes=recipes,
                        seeds=sds,
                        purpose="screen",
                        category="screen",
                        phase="S",
                        expected_cost=float(len(recipes)),
                    )
                ]
        if dec["needs_combination"] and combos:
            per = []
            for a in snap.anchors:
                v = [r["val_r"] for r in combos if r["anchor"] == a and r["val_r"] is not None]
                b = [x for x in (snap.screen_baseline.get(a) or {}).values() if x is not None]
                if v and b:
                    per.append(float(np.mean(v) - np.mean(b)))
            combo_eff = float(np.mean(per)) if per else -1.0
            best_single = max(dec["effects"][o]["mean"] for o in adopted)
            dec["combination_effect"] = combo_eff
            adopted = (
                adopted
                if combo_eff >= best_single
                else [max(adopted, key=lambda o: dec["effects"][o]["mean"])]
            )
        return [
            Action(
                "SCREEN",
                {
                    "kind": "adopt",
                    "adopted": adopted,
                    "effects": dec["effects"],
                    "not_testable_in_silico": dec["not_testable"],
                    "defaults": adopted_defaults(adopted, self.cfg),
                    "combination_effect": dec.get("combination_effect"),
                },
                reason=f"phase S decision: adopted {adopted or 'nothing'} (mean effect over anchors above 0.005 with the same sign on all three; at most 2); not testable in silico: {dec['not_testable']}",
                category="screen",
                phase="S",
            )
        ]

    def _phase_p(self, snap: Snapshot, fams: dict[str, FamilyState], ctx: _Ctx) -> list[Action]:
        p = snap.procedures or {}
        if ctx.calls <= 0:
            return []
        if not p.get("nominated"):
            ctx.calls -= 1
            return [
                Action(
                    "PROCEDURE",
                    {"step": "nominate", "idea": None},
                    reason="phase P: the LLM nominates 10 one-line loss / training-procedure ideas; the 2 that differ most from the six screened options are implemented",
                    category="procedure",
                    phase="P",
                )
            ]
        if int(p.get("implemented", 0)) < int(self.cfg["procedure_mode"]["max_plugins"]):
            ctx.calls -= 1
            i = int(p.get("implemented", 0))
            sel = p.get("selected") or p["nominated"]
            return [
                Action(
                    "PROCEDURE",
                    {"step": "implement", "idea": sel[min(i, len(sel) - 1)]},
                    reason=f"phase P: implement plug-in {i + 1}",
                    category="procedure",
                    phase="P",
                )
            ]
        return []

    def _phase_d(self, snap: Snapshot, fams: dict[str, FamilyState], ctx: _Ctx) -> list[Action]:
        menu = menu_select(fams, self.cfg)
        out: list[Action] = []
        seeds = tuner.seeds_for("final", self.cfg)
        names = menu["menu"]
        cap_n = int(
            max(
                self.budget_remaining(snap, "D")
                + sum(len(fams[n].final_vals) + fams[n].final_open for n in names),
                0,
            )
            // len(seeds)
        )
        names = names[: max(cap_n, 0)]
        if not names:
            return out
        all_done = True
        for n in names:
            f = fams[n]
            have = len(f.final_vals) + f.final_open
            if have < len(seeds):
                all_done = False
                if ctx.reserve("D", len(seeds)):
                    out.append(
                        Action(
                            "FINAL",
                            {
                                "family": n,
                                "code_hash": f.best_code_hash,
                                "recipe_hash": f.best_recipe_hash,
                                "val_select": True,
                            },
                            reason=f"phase D: reconfirm {n} (confirmed {f.confirmed_mean:.4f}) on seeds {seeds}, scored once on val_select",
                            recipes=[copy.deepcopy(f.best_recipe) for _ in seeds],
                            seeds=list(seeds),
                            purpose="final",
                            category="final",
                            phase="D",
                            expected_cost=float(len(seeds)),
                        )
                    )
            elif f.final_open > 0 or len(f.final_vals) < len(seeds):
                all_done = False
        if all_done and not snap.final_menu_done:
            out.append(
                Action(
                    "NOTE",
                    {
                        "kind": "menu",
                        "menu": names,
                        "flagged": menu["flagged"],
                        "top": menu["top"],
                        "within_margin": menu["within_margin"],
                        "done": True,
                    },
                    reason=f"phase D finished: menu {names}",
                    executable=False,
                )
            )
        return out

    # ---------------------------------------------------------------- phase M new work

    def _repair(self, snap: Snapshot, ctx: _Ctx, out: list[Action]) -> None:
        mx = int(self.cfg["gate"].get("max_repairs", 2)) if "gate" in self.cfg else 2
        inflight = {
            t.get("code_hash") for t in snap.calls_in_flight_targets if t.get("mode") == "REPAIR"
        }
        for q in snap.repair_queue:
            if ctx.calls <= 0:
                return
            if q.get("attempts", 0) >= mx or q["code_hash"] in inflight:
                continue
            ctx.calls -= 1
            out.append(
                Action(
                    "REPAIR",
                    {
                        "code_hash": q["code_hash"],
                        "family": q.get("family"),
                        "gate_code": q.get("gate_code"),
                        "attempt": q.get("attempts", 0) + 1,
                    },
                    reason=f"gate failure {q.get('gate_code')} of {q['code_hash'][:8]}: repair attempt {q.get('attempts', 0) + 1} of {mx}",
                    category="explore",
                    phase="M",
                )
            )

    def _new_work(
        self,
        snap: Snapshot,
        fams: dict[str, FamilyState],
        ctx: _Ctx,
        out: list[Action],
        adapt: dict,
    ) -> None:
        c = self.cfg["controller"]
        tgt = self.targets(snap, fams)
        proj = {
            k: float(snap.spend_by_category.get(k, 0.0)) + float(snap.open_by_category.get(k, 0.0))
            for k in CATEGORIES
        }
        sketches = list(snap.sketches_unused)
        inflight_fams = {t.get("family") for t in snap.calls_in_flight_targets}
        sketch_in_flight = any(t.get("mode") == "SKETCH" for t in snap.calls_in_flight_targets)
        top = top_confirmed(fams)
        guard = 0
        committed = self.committed_cost(fams)
        while guard < 8 and ctx.queue > 0:
            guard += 1
            total = sum(proj.values())
            share = {k: (proj[k] / total if total > 0 else 0.0) for k in CATEGORIES}
            deficit = {k: tgt[k] - share[k] for k in CATEGORIES}
            avail = self._available(snap, fams, ctx, sketches, inflight_fams, top, committed)
            if not avail:
                return
            cat = max(avail, key=lambda k: (deficit[k], tgt[k]))
            if cat == "deep_tune":
                a = self._deep_action(snap, fams, ctx, top)
                if a is None:
                    return
                out.append(a)
                proj["deep_tune"] += c["expected_cost"]["deep_tune"]
                ctx.queue -= len(a.recipes)
                continue
            if ctx.calls <= 0:
                return
            if cat == "refine":
                a = self._refine_action(snap, fams, ctx, top, inflight_fams)
                if a is None:
                    return
                out.append(a)
                ctx.calls -= 1
                proj["refine"] += c["expected_cost"]["refine"]
                ctx.queue -= 4
                continue
            # explore: need an unused sketch; otherwise request more
            if (
                len(sketches) < 4 or snap.re_since_sketch >= c["resketch_every"]
            ) and not sketch_in_flight:
                forced = adapt.get("forced_families")
                out.append(
                    Action(
                        "SKETCH",
                        {"forced_families": forced},
                        reason=f"{len(sketches)} unused sketches (request more below 4) / {snap.re_since_sketch:.0f} run-equivalents since the last SKETCH call",
                        category="explore",
                        phase="M",
                    )
                )
                ctx.calls -= 1
                sketch_in_flight = True
                if not sketches:
                    return
                continue
            sk = self._pick_sketch(snap, fams, sketches, inflight_fams | ctx.explored)
            if sk is None:
                return
            sketches.remove(sk)
            ctx.explored.add(sk["family"])
            out.append(
                Action(
                    "EXPLORE",
                    {
                        "family": sk["family"],
                        "sketch": sk,
                        "anchor": False,
                        "source": "llm",
                        "extra_constraints": list(adapt.get("extra_constraints") or []),
                    },
                    reason=f"balance: explore is furthest below its target share ({tgt['explore']:.2f}); implement the most novel sketch ({sk.get('name')}) in family {sk['family']}",
                    category="explore",
                    phase="M",
                    expected_cost=c["expected_cost"]["explore"],
                )
            )
            ctx.calls -= 1
            proj["explore"] += c["expected_cost"]["explore"]
            ctx.queue -= 3

    def _available(
        self,
        snap: Snapshot,
        fams: dict[str, FamilyState],
        ctx: _Ctx,
        sketches: list,
        inflight: set,
        top: float | None,
        committed: float = 0.0,
    ) -> list[str]:
        ms = self.cfg["controller"]["min_start"]
        rem = ctx.remaining["M"] - committed
        out = []
        if rem >= ms["explore"] and (sketches or True):
            out.append("explore")
        if rem >= ms["refine"] and self._refine_candidates(fams, top, inflight):
            out.append("refine")
        if rem >= ms["deep_tune"] and self._deep_candidates(fams, top):
            out.append("deep_tune")
        return out

    def _refine_candidates(
        self, fams: dict[str, FamilyState], top: float | None, inflight: set
    ) -> list[FamilyState]:
        rm = self.cfg["controller"]["refine_margin"]
        el = [
            f
            for f in fams.values()
            if f.state in ("confirmed", "plateaued")
            and f.pending_refinement is None
            and f.family not in inflight
            and f.deep.confirm_recipe is None
            and top is not None
            and f.confirmed_mean is not None
            and f.confirmed_mean >= top - rm
        ]
        live = [f for f in el if f.state != "plateaued"]
        return (
            live or el
        )  # plateaued families get more refinements only after every other family has plateaued

    def _deep_candidates(
        self, fams: dict[str, FamilyState], top: float | None
    ) -> list[FamilyState]:
        mm = self.cfg["controller"]["menu_margin"]
        return [
            f
            for f in fams.values()
            if f.state in ("confirmed", "plateaued")
            and top is not None
            and f.confirmed_mean is not None
            and f.confirmed_mean >= top - mm
            and f.pending_refinement is None
            and f.deep.batch_runs == 0
            and f.deep.confirm_recipe is None
        ]

    def _pick_sketch(
        self, snap: Snapshot, fams: dict[str, FamilyState], sketches: list[dict], busy: set
    ) -> dict | None:
        cands = [s for s in sketches if s.get("family") not in busy] or []
        if not cands:
            return None
        scored = sorted(
            cands,
            key=lambda s: (
                -sketch_novelty(s, fams, snap.candidate_summaries),
                str(s.get("id", s.get("name", ""))),
            ),
        )
        return scored[0]

    def _refine_action(
        self,
        snap: Snapshot,
        fams: dict[str, FamilyState],
        ctx: _Ctx,
        top: float | None,
        inflight: set,
    ) -> Action | None:
        el = self._refine_candidates(fams, top, inflight)
        if not el:
            return None
        lo = min(f.refinements for f in el)
        pool = [
            f for f in el if f.refinements == lo
        ]  # every eligible family gets one refinement before any gets a second
        T = float(self.cfg["controller"]["refine_temperature"])
        w = np.array(
            [
                math.exp((f.confirmed_mean - top) / T) / (1 + f.refinements_without_gain)
                for f in pool
            ]
        )
        rng = self.rng(
            snap.step, f"refine:{len(snap.families)}:{sum(f.refinements for f in fams.values())}"
        )
        i = int(rng.choice(len(pool), p=w / w.sum()))
        f = pool[i]
        shift = (
            1 if (f.refinements_without_gain > 0 and f.refinements_without_gain % 3 == 0) else 0
        )  # 12.1: three rejections in a row rotate to the next angle
        angle = REFINE_ANGLES[(f.angle_index + shift) % len(REFINE_ANGLES)]
        return Action(
            "REFINE",
            {
                "family": f.family,
                "parent_code_hash": f.best_code_hash,
                "angle": angle,
                "extra_constraints": [],
            },
            reason=f"balance: refine is furthest below its target share; {f.family} (confirmed {f.confirmed_mean:.4f}, top {top:.4f}, {f.refinements} refinements) sampled with weight exp((mean-top)/{T}) / (1 + rejections); angle: {angle}",
            category="refine",
            phase="M",
            expected_cost=self.cfg["controller"]["expected_cost"]["refine"],
        )

    def _deep_action(
        self, snap: Snapshot, fams: dict[str, FamilyState], ctx: _Ctx, top: float | None
    ) -> Action | None:
        el = self._deep_candidates(fams, top)
        if not el:
            return None
        nb = int(self.cfg["controller"]["deep_batch"])
        lo = min(f.deep.recipes_done for f in el)
        pool = sorted([f for f in el if f.deep.recipes_done == lo], key=lambda f: f.family)
        f = pool[int(self.rng(snap.step, "deep").integers(len(pool)))]
        if not ctx.reserve("M", nb):
            return None
        recs = tuner.local_recipes(
            f.best_recipe,
            f.family,
            self.cfg,
            self.rng(snap.step, f"deeprec:{f.family}"),
            nb,
            tunable=f.tunable,
            defaults=snap.recipe_defaults or {},
            existing={x["recipe_hash"] for x in f.recipe_results},
        )
        return Action(
            "TUNE_DEEP",
            {
                "family": f.family,
                "code_hash": f.best_code_hash,
                "batch": f.deep.batch_index,
                "recipe_ids": [tuner.recipe_hash(r) for r in recs],
                "cap_multipliers": [1.0] * nb,
            },
            reason=f"balance: deep tuning is furthest below its target share; {f.family} has the fewest deep-tune recipes ({f.deep.recipes_done}): {nb} local perturbations of its best recipe",
            recipes=recs,
            seeds=[0] * nb,
            purpose="deep_tune",
            category="deep_tune",
            phase="M",
            expected_cost=float(nb),
        )
