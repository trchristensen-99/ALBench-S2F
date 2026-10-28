"""Recipe tuner (spec section 8): prior, sampling order, local perturbation, triage, time caps and the failure table.

Every function is pure: randomness comes from the ``numpy.random.Generator`` passed in (the controller derives one per decision from the config seed, the step and a
tag, so a run replays exactly), and nothing reads the database. The tuner never calls an LLM: prior-informed random sampling matched LLM hyper-parameter proposals in
earlier tests (spec 8). Recipes are plain dicts in the shape of spec 4.3::

    {"optimizer", "lr", "weight_decay", "batch_size", "schedule", "warmup_frac", "max_epochs", "patience", "grad_clip",
     "loss": {"name": ...}, "rc_augment", "ema", "amp", "tasks", "hparams": {...}}

``recipe_hash`` is the identity of a recipe; the seed is a run field, not a recipe field.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

import numpy as np

ATTENTION_FAMILIES = ("transformer_hybrid", "transformer_pure")
RECURRENT_FAMILIES = ("rnn_hybrid", "tcn_tied")
LOW_LR_FAMILIES = ("transformer_hybrid", "transformer_pure", "state_space", "mixer_mlp")
BATCH_SIZES = (128, 256, 512, 1024)
SCHEDULES = ("onecycle", "cosine_warmup")
LR_RANGE = (1e-4, 1e-2)
LR_RANGE_LOW = (5e-5, 2e-3)
WD_RANGE = (1e-5, 1e-1)
EMA_DECAY = 0.999

# --------------------------------------------------------------------------------------------------------------------------------- identity


def recipe_hash(recipe: dict) -> str:
    """sha1[:12] of the canonical JSON (sorted keys, no spaces) of the recipe dict."""
    s = json.dumps(recipe, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def _json_default(o: Any):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")


# --------------------------------------------------------------------------------------------------------------------------------- family defaults


def _lr_range(family: str) -> tuple[float, float]:
    return LR_RANGE_LOW if family in LOW_LR_FAMILIES else LR_RANGE


def _forced_clip(family: str) -> bool:
    return family in ATTENTION_FAMILIES or family in RECURRENT_FAMILIES


def _warmup_range(family: str) -> tuple[float, float]:
    return (0.05, 0.15) if family in ATTENTION_FAMILIES else (0.0, 0.10)


def default_recipe(
    family: str, cfg: dict, hparams: dict | None = None, defaults: dict | None = None
) -> dict:
    """The family default of the tuner's prior (second recipe of the first look). ``defaults`` = adopted procedure defaults: {"loss": {...}, "ema": None | decay}."""
    ev = cfg.get("evaluator", {})
    defaults = defaults or {}
    if family in LOW_LR_FAMILIES:
        lr, wd, bs, warm = 5e-4, 0.05, 256, 0.10
    elif family in RECURRENT_FAMILIES:
        lr, wd, bs, warm = 1e-3, 0.01, 256, 0.05
    else:
        lr, wd, bs, warm = 2e-3, 0.01, 512, 0.05
    return {
        "optimizer": "adamw",
        "lr": lr,
        "weight_decay": wd,
        "batch_size": bs,
        "schedule": "onecycle",
        "warmup_frac": warm,
        "max_epochs": int(ev.get("max_epochs", 40)),
        "patience": int(ev.get("patience", 8)),
        "grad_clip": 1.0,
        "loss": copy.deepcopy(defaults.get("loss") or {"name": "mse"}),
        "rc_augment": True,
        "ema": defaults.get("ema"),
        "amp": True,
        "tasks": [cfg.get("data", {}).get("primary_task", "K562")],
        "hparams": dict(hparams or {}),
    }


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def fill_suggested(
    suggested: dict | None,
    family: str,
    cfg: dict,
    hparams: dict | None = None,
    defaults: dict | None = None,
) -> dict:
    """The proposer's suggested_recipe with missing (or invalid) fields filled from the family default. max_epochs and patience are fixed by the harness."""
    rec = default_recipe(family, cfg, hparams, defaults)
    s = suggested or {}
    if isinstance(s.get("lr"), (int, float)) and s["lr"] > 0:
        rec["lr"] = _clip(float(s["lr"]), 1e-5, 1e-1)
    if isinstance(s.get("weight_decay"), (int, float)) and s["weight_decay"] >= 0:
        rec["weight_decay"] = _clip(float(s["weight_decay"]), 0.0, 1.0)
    if isinstance(s.get("batch_size"), (int, float)) and s["batch_size"] > 0:
        rec["batch_size"] = int(
            min(BATCH_SIZES, key=lambda b: abs(math.log(b) - math.log(s["batch_size"])))
        )
    if isinstance(s.get("warmup_frac"), (int, float)):
        rec["warmup_frac"] = _clip(float(s["warmup_frac"]), 0.0, 0.3)
    if s.get("schedule") in SCHEDULES:
        rec["schedule"] = s["schedule"]
    if "grad_clip" in s and (s["grad_clip"] is None or isinstance(s["grad_clip"], (int, float))):
        rec["grad_clip"] = None if s["grad_clip"] is None else float(s["grad_clip"])
    if isinstance(s.get("rc_augment"), bool):
        rec["rc_augment"] = s["rc_augment"]
    if _forced_clip(family):
        rec["grad_clip"] = 1.0
    return rec


# --------------------------------------------------------------------------------------------------------------------------------- hparams


def _is_int_range(rng_pair: list, default: Any) -> bool:
    return all(isinstance(v, int) and not isinstance(v, bool) for v in rng_pair) and (
        default is None or isinstance(default, int)
    )


def _draw_hparams(
    hparams: dict, tunable: dict, rng: np.random.Generator, *, perturb_one: bool = False
) -> dict:
    """At most 2 TUNABLE keys per draw, log-uniform inside their range; the rest keep the default. ``perturb_one``: perturb a single key around its current value."""
    out = dict(hparams or {})
    keys = [
        k
        for k, v in (tunable or {}).items()
        if isinstance(v, (list, tuple)) and len(v) == 2 and v[0] > 0
    ]
    if not keys:
        return out
    if perturb_one:
        chosen = [keys[int(rng.integers(len(keys)))]]
    else:
        n = int(rng.integers(1, min(2, len(keys)) + 1))
        chosen = [keys[i] for i in rng.permutation(len(keys))[:n]]
    for k in chosen:
        lo, hi = tunable[k]
        if perturb_one and out.get(k):
            val = float(out[k]) * 10 ** float(rng.normal(0.0, 0.25))
        else:
            val = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        val = _clip(val, lo, hi)
        out[k] = int(round(val)) if _is_int_range([lo, hi], out.get(k)) else float(val)
        if isinstance(out[k], int):
            out[k] = int(_clip(out[k], lo, hi))
    return out


# --------------------------------------------------------------------------------------------------------------------------------- prior and perturbation


def sample_prior(
    family: str,
    cfg: dict,
    rng: np.random.Generator,
    *,
    hparams: dict | None = None,
    tunable: dict | None = None,
    defaults: dict | None = None,
) -> dict:
    """One draw from the recipe prior of spec 8.1 (loss and EMA stay at the adopted defaults until the procedure screen has finished)."""
    rec = default_recipe(family, cfg, hparams, defaults)
    lo, hi = _lr_range(family)
    rec["lr"] = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    rec["weight_decay"] = float(math.exp(rng.uniform(math.log(WD_RANGE[0]), math.log(WD_RANGE[1]))))
    rec["batch_size"] = int(BATCH_SIZES[int(rng.integers(len(BATCH_SIZES)))])
    rec["schedule"] = "onecycle" if rng.random() < 0.7 else "cosine_warmup"
    wlo, whi = _warmup_range(family)
    rec["warmup_frac"] = float(rng.uniform(wlo, whi))
    rec["grad_clip"] = 1.0 if (_forced_clip(family) or rng.random() < 0.5) else None
    rec["rc_augment"] = bool(rng.random() < 0.8)
    if (defaults or {}).get("screen_done"):
        rec["ema"] = EMA_DECAY if rng.random() < 0.3 else None
    rec["hparams"] = _draw_hparams(hparams or {}, tunable or {}, rng)
    return rec


def perturb(
    recipe: dict,
    family: str,
    cfg: dict,
    rng: np.random.Generator,
    *,
    tunable: dict | None = None,
    defaults: dict | None = None,
) -> dict:
    """Local perturbation of spec 8.2: lr x 10**N(0, 0.25), weight_decay x 10**N(0, 0.5), each categorical field flipped with p = 0.3, one TUNABLE key perturbed."""
    rec = copy.deepcopy(recipe)
    lo, hi = _lr_range(family)
    rec["lr"] = _clip(rec["lr"] * 10 ** float(rng.normal(0.0, 0.25)), lo / 2, hi * 2)
    rec["weight_decay"] = _clip(rec["weight_decay"] * 10 ** float(rng.normal(0.0, 0.5)), 1e-6, 0.5)
    if rng.random() < 0.3:
        rec["schedule"] = "cosine_warmup" if rec["schedule"] == "onecycle" else "onecycle"
    if rng.random() < 0.3:
        others = [b for b in BATCH_SIZES if b != rec["batch_size"]]
        rec["batch_size"] = int(others[int(rng.integers(len(others)))])
    if rng.random() < 0.3:
        rec["rc_augment"] = not rec["rc_augment"]
    if rng.random() < 0.3 and not _forced_clip(family):
        rec["grad_clip"] = None if rec["grad_clip"] else 1.0
    if (defaults or {}).get("screen_done") and rng.random() < 0.3:
        rec["ema"] = None if rec.get("ema") else EMA_DECAY
    if tunable:
        rec["hparams"] = _draw_hparams(rec.get("hparams", {}), tunable, rng, perturb_one=True)
    return rec


def _distinct(
    make, existing: set[str], tries: int = 30
) -> dict:  # make() -> recipe; returns one whose hash is not in ``existing`` (best effort)
    rec = make()
    for _ in range(tries):
        if recipe_hash(rec) not in existing:
            return rec
        rec = make()
    return rec


# --------------------------------------------------------------------------------------------------------------------------------- recipe sets


def first_look_recipes(
    family: str,
    cfg: dict,
    rng: np.random.Generator,
    *,
    suggested: dict | None = None,
    hparams: dict | None = None,
    tunable: dict | None = None,
    defaults: dict | None = None,
    existing: set[str] | None = None,
) -> list[dict]:
    """The first look: the proposer's suggested recipe, the family default, one draw from the prior (spec 5.2 and 8.2). Always three distinct recipes."""
    seen = set(existing or ())
    out = []
    a = fill_suggested(suggested, family, cfg, hparams, defaults)
    b = default_recipe(family, cfg, hparams, defaults)
    for rec in (a, b):
        if recipe_hash(rec) in seen:
            rec = _distinct(
                lambda: perturb(rec, family, cfg, rng, tunable=tunable, defaults=defaults), seen
            )
        seen.add(recipe_hash(rec))
        out.append(rec)
    c = _distinct(
        lambda: sample_prior(family, cfg, rng, hparams=hparams, tunable=tunable, defaults=defaults),
        seen,
    )
    out.append(c)
    n = int(cfg.get("tuner", {}).get("first_look", 3))
    return out[
        : max(1, n)
    ]  # tuner.first_look < 3 only for the "what does the three-recipe first look buy" experiment of spec 16.2


def prior_recipes(
    family: str,
    cfg: dict,
    rng: np.random.Generator,
    n: int,
    *,
    hparams: dict | None = None,
    tunable: dict | None = None,
    defaults: dict | None = None,
    existing: set[str] | None = None,
) -> list[dict]:
    """``n`` further draws from the prior (recipes 4 and 5)."""
    seen = set(existing or ())
    out = []
    for _ in range(n):
        rec = _distinct(
            lambda: sample_prior(
                family, cfg, rng, hparams=hparams, tunable=tunable, defaults=defaults
            ),
            seen,
        )
        seen.add(recipe_hash(rec))
        out.append(rec)
    return out


def local_recipes(
    best: dict,
    family: str,
    cfg: dict,
    rng: np.random.Generator,
    n: int,
    *,
    tunable: dict | None = None,
    defaults: dict | None = None,
    existing: set[str] | None = None,
) -> list[dict]:
    """``n`` local perturbations of the best recipe so far (recipes 6 to 8, and every deep-tune recipe)."""
    seen = set(existing or ()) | {recipe_hash(best)}
    out = []
    for _ in range(n):
        rec = _distinct(
            lambda: perturb(best, family, cfg, rng, tunable=tunable, defaults=defaults), seen
        )
        seen.add(recipe_hash(rec))
        out.append(rec)
    return out


def refine_child_recipes(
    parent_best: dict,
    family: str,
    cfg: dict,
    rng: np.random.Generator,
    *,
    child_hparams: dict | None = None,
    tunable: dict | None = None,
    defaults: dict | None = None,
    existing: set[str] | None = None,
) -> list[dict]:
    """A REFINE child starts from its parent's best recipe plus two local perturbations, not from the prior (spec 8.2)."""
    base = copy.deepcopy(parent_best)
    if child_hparams is not None:
        base["hparams"] = dict(child_hparams)
    seen = set(existing or ())
    out = [base]
    seen.add(recipe_hash(base))
    out += local_recipes(
        base, family, cfg, rng, 2, tunable=tunable, defaults=defaults, existing=seen
    )
    return out


def diagnose(run: dict) -> str:
    """Triage class of one run: 'optimisation' | 'overfit' | 'time_cap' | 'none' (spec 5.2).

    ``run``: {status, train_r, val_r, censored, still_improving}.
    """
    if run.get("status") == "diverged" or (run.get("train_r") is not None and run["train_r"] < 0.5):
        return "optimisation"
    tr, vr = run.get("train_r"), run.get("val_r")
    if tr is not None and vr is not None and tr - vr > 0.15:
        return "overfit"
    if run.get("censored") and run.get("still_improving"):
        return "time_cap"
    return "none"


def triage_recipes(
    runs: list[dict],
    family: str,
    cfg: dict,
    rng: np.random.Generator,
    *,
    tunable: dict | None = None,
    defaults: dict | None = None,
    existing: set[str] | None = None,
) -> tuple[list[dict], list[float], dict]:
    """The second-chance set of 3 recipes chosen from the diagnostics of the first look (spec 5.2).

    Each first-look run gets the fix for its own diagnosis: optimisation failure -> lr / 5, 10% warm-up, gradient clipping; overfitting -> weight decay x 10 and EMA;
    cut by the time cap while still improving -> the same recipe with the cap x 1.5. A run with no diagnosis gets a local perturbation. Returns (recipes,
    cap_multipliers aligned with recipes, {"diagnoses": [...]}).
    """
    seen = set(existing or ())
    recs: list[dict] = []
    caps: list[float] = []
    diag: list[str] = []
    for run in runs[:3]:
        d = diagnose(run)
        diag.append(d)
        rec = copy.deepcopy(run["recipe"])
        cap = 1.0
        if d == "optimisation":
            rec["lr"] = rec["lr"] / 5.0
            rec["warmup_frac"] = 0.10
            rec["grad_clip"] = 1.0
        elif d == "overfit":
            rec["weight_decay"] = _clip(rec["weight_decay"] * 10.0, 1e-6, 0.5)
            rec["ema"] = EMA_DECAY
        elif d == "time_cap":
            cap = float(cfg.get("evaluator", {}).get("cap_extension", 1.5))
        else:
            rec = perturb(rec, family, cfg, rng, tunable=tunable, defaults=defaults)
        if recipe_hash(rec) in seen and d != "time_cap":
            rec = _distinct(
                lambda: perturb(rec, family, cfg, rng, tunable=tunable, defaults=defaults), seen
            )
        seen.add(recipe_hash(rec))
        recs.append(rec)
        caps.append(cap)
    while (
        len(recs) < 3
    ):  # fewer than three first-look runs finished OK: pad with local perturbations of the best available
        base = recs[0] if recs else default_recipe(family, cfg, None, defaults)
        rec = _distinct(
            lambda: perturb(base, family, cfg, rng, tunable=tunable, defaults=defaults), seen
        )
        seen.add(recipe_hash(rec))
        recs.append(rec)
        caps.append(1.0)
        diag.append("pad")
    return recs, caps, {"diagnoses": diag}


# --------------------------------------------------------------------------------------------------------------------------------- seeds, caps, early termination


def seeds_for(purpose: str, cfg: dict) -> list[int]:
    """Seeds of a purpose: screening and tuning share seed 0 (so recipe comparisons share their randomness); confirmation 1-3; the final phase 4-6 (spec 8.3)."""
    t = cfg.get("tuner", {})
    if purpose == "confirm":
        return list(t.get("confirm_seeds", [1, 2, 3]))
    if purpose == "final":
        return list(t.get("final_seeds", [4, 5, 6]))
    return [int(t.get("screen_seed", 0))]


def time_cap_s(D: int, cfg: dict, *, multiplier: float = 1.0) -> float:
    """time_cap_s = base_cap_s * (D / 100000) ** 0.72 (spec 8.5); ``multiplier`` is 1.5 for the one extension."""
    ev = cfg.get("evaluator", {})
    base = float(ev.get("base_cap_s", 1800))
    exp = float(ev.get("time_cap_exponent", 0.72))
    return float(base * (D / 100000.0) ** exp * multiplier)


def needs_cap_extension(best_run: dict, already_extended: bool) -> bool:
    """The family's best run was censored by the time cap and was still improving by more than 0.001 r per epoch: its confirmation gets 1.5 x the cap, once."""
    return bool(
        not already_extended
        and best_run.get("censored")
        and float(best_run.get("improvement_per_epoch", 0.0)) > 0.001
    )


def reference_for(
    schedule: str,
    family_runs: list[dict],
    checkpoints: list[float] | tuple[float, ...] = (0.5, 0.75),
) -> dict:
    """Early-termination reference of spec 8.4: for ``schedule``, the best validation r at each checkpoint among the family's runs that used the SAME schedule.

    ``family_runs``: dicts with ``schedule`` and ``checkpoint_best`` ({"0.5": r, "0.75": r}); exploded runs are skipped. Returns {schedule: {"0.5": r, "0.75": r}} (empty
    when there is nothing to compare with).
    """
    ref: dict[str, float] = {}
    for run in family_runs:
        if run.get("schedule") != schedule or run.get("exploded"):
            continue
        for c in checkpoints:
            v = (run.get("checkpoint_best") or {}).get(str(c))
            if v is not None and v > ref.get(str(c), -1e9):
                ref[str(c)] = float(v)
    return {schedule: ref} if ref else {}


def early_term_spec(purpose: str, schedule: str, family_runs: list[dict], cfg: dict) -> dict | None:
    """The ``early_term`` block of an evaluator spec. Never in the first look (all 3 recipes finish), never in confirmation or the final phase; only tune / deep_tune."""
    if purpose not in ("tune", "deep_tune"):
        return None
    t = cfg.get("tuner", {})
    ref = reference_for(schedule, family_runs, t.get("early_term_checkpoints", [0.5, 0.75]))
    if not ref:
        return None
    return {
        "checkpoints": list(t.get("early_term_checkpoints", [0.5, 0.75])),
        "margin": float(t.get("early_term_margin", 0.03)),
        "reference": ref,
    }


# --------------------------------------------------------------------------------------------------------------------------------- failures (spec 8.6)


def handle_failure(
    status: str,
    recipe: dict,
    *,
    infra_retries_used: int = 0,
    is_replacement: bool = False,
    max_infra_retries: int = 2,
) -> dict:
    """What to do with a run that did not finish normally.

    diverged  -> no retry; the recipe counts toward the 8.
    oom       -> one retry with batch_size halved, logged as a replacement recipe (the failed one does not count; the replacement does). A replacement that fails again counts.
    error     -> infrastructure failure: resubmit up to ``max_infra_retries`` times, never counted toward the 8.
    other     -> a result, nothing to do (exploded and time_cap runs are results).
    Returns {"action": "none" | "replace" | "resubmit", "recipe": dict | None, "counts_toward_recipes": bool}.
    """
    if status == "oom" and not is_replacement and int(recipe.get("batch_size", 0)) > 16:
        new = copy.deepcopy(recipe)
        new["batch_size"] = max(16, int(recipe["batch_size"]) // 2)
        return {"action": "replace", "recipe": new, "counts_toward_recipes": False}
    if status == "error" and infra_retries_used < max_infra_retries:
        return {"action": "resubmit", "recipe": None, "counts_toward_recipes": False}
    return {"action": "none", "recipe": None, "counts_toward_recipes": True}
