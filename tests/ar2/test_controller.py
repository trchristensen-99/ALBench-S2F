"""Unit tests of the controller policy and the tuner (spec sections 5, 8, 9.5, 12.1, 13.2). Pure functions on plain data: no GPU, LLM or database."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from ar2 import tuner
from ar2.controller import (
    ANCHORS,
    TAXONOMY,
    Controller,
    acceptance,
    build_family_state,
    decide_screen,
    derive_family_state,
    estimate_sigma,
    family_view,
    forecast,
    menu_select,
    merged,
    plan,
    sketch_novelty,
)
from ar2.state import Action, FamilyState, RunInfo, Snapshot

CFG = merged(None)


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def fs(**kw) -> FamilyState:
    base = dict(
        family="unet",
        n_candidates=1,
        recipes_planned=3,
        recipes_done=3,
        first_look_done=3,
        first_look_best=0.70,
        screen_best=0.70,
    )
    base.update(kw)
    return FamilyState(**base)


# ------------------------------------------------------------------ tuner


def test_prior_respects_ranges_and_fixed_fields():
    for fam in ("cnn_residual", "transformer_hybrid", "state_space", "rnn_hybrid"):
        for i in range(200):
            r = tuner.sample_prior(
                fam,
                CFG,
                rng(i),
                hparams={"width": 256, "depth": 6},
                tunable={"width": [128, 512], "depth": [4, 10], "drop": [0.0, 0.5], "k": [3, 9]},
            )
            lo, hi = (5e-5, 2e-3) if fam in tuner.LOW_LR_FAMILIES else (1e-4, 1e-2)
            assert lo <= r["lr"] <= hi
            assert 1e-5 <= r["weight_decay"] <= 1e-1
            assert r["batch_size"] in (128, 256, 512, 1024)
            assert r["schedule"] in ("onecycle", "cosine_warmup")
            assert r["max_epochs"] == 40 and r["patience"] == 8
            assert r["ema"] is None and r["loss"] == {
                "name": "mse"
            }  # fixed until the procedure screen has finished
            if fam in tuner.ATTENTION_FAMILIES:
                assert 0.05 <= r["warmup_frac"] <= 0.15
            else:
                assert 0.0 <= r["warmup_frac"] <= 0.10
            if fam in tuner.ATTENTION_FAMILIES + tuner.RECURRENT_FAMILIES:
                assert r["grad_clip"] == 1.0
            changed = [
                k for k in r["hparams"] if r["hparams"][k] != {"width": 256, "depth": 6}.get(k)
            ]
            assert len(changed) <= 2
            assert 128 <= r["hparams"]["width"] <= 512 and isinstance(r["hparams"]["width"], int)


def test_prior_samples_ema_only_after_screen():
    seen = {
        tuner.sample_prior(
            "cnn_residual", CFG, rng(i), defaults={"screen_done": True, "ema": None}
        )["ema"]
        for i in range(200)
    }
    assert seen == {None, tuner.EMA_DECAY}


def test_first_look_is_suggested_default_prior():
    sug = {"lr": 3e-4, "weight_decay": 0.05, "batch_size": 256, "warmup_frac": 0.1}
    recs = tuner.first_look_recipes("transformer_hybrid", CFG, rng(1), suggested=sug)
    assert len(recs) == 3 and len({tuner.recipe_hash(r) for r in recs}) == 3
    assert recs[0]["lr"] == 3e-4 and recs[0]["weight_decay"] == 0.05
    assert recs[1] == tuner.default_recipe("transformer_hybrid", CFG)


def test_suggested_equal_to_default_still_gives_three_distinct():
    d = tuner.default_recipe("cnn_residual", CFG)
    recs = tuner.first_look_recipes(
        "cnn_residual",
        CFG,
        rng(2),
        suggested={
            "lr": d["lr"],
            "weight_decay": d["weight_decay"],
            "batch_size": d["batch_size"],
            "warmup_frac": d["warmup_frac"],
        },
    )
    assert len({tuner.recipe_hash(r) for r in recs}) == 3


def test_local_perturbation_moves_lr_and_wd_and_keeps_hparam_keys():
    best = tuner.default_recipe("cnn_residual", CFG, {"width": 256, "depth": 6})
    recs = tuner.local_recipes(
        best, "cnn_residual", CFG, rng(3), 3, tunable={"width": [128, 512], "depth": [4, 10]}
    )
    assert len(recs) == 3
    for r in recs:
        assert r != best and set(r["hparams"]) == {"width", "depth"}
        assert abs(math.log10(r["lr"] / best["lr"])) < 1.2  # 10**N(0, 0.25): within ~5 sigma


def test_refine_child_starts_from_parent_best_then_two_perturbations():
    parent = tuner.default_recipe("rnn_hybrid", CFG, {"width": 128})
    parent["lr"] = 4.2e-4
    recs = tuner.refine_child_recipes(
        parent,
        "rnn_hybrid",
        CFG,
        rng(4),
        child_hparams={"width": 192},
        tunable={"width": [64, 512]},
    )
    assert len(recs) == 3 and recs[0]["lr"] == 4.2e-4 and recs[0]["hparams"] == {"width": 192}
    assert len({tuner.recipe_hash(r) for r in recs}) == 3


def test_time_cap_formula_and_extension():
    assert tuner.time_cap_s(100_000, CFG) == pytest.approx(1800.0)
    assert tuner.time_cap_s(30_000, CFG) == pytest.approx(1800 * 0.3**0.72)
    assert tuner.time_cap_s(10_000, CFG, multiplier=1.5) == pytest.approx(1800 * 0.1**0.72 * 1.5)
    assert tuner.needs_cap_extension({"censored": True, "improvement_per_epoch": 0.002}, False)
    assert not tuner.needs_cap_extension(
        {"censored": True, "improvement_per_epoch": 0.002}, True
    )  # once
    assert not tuner.needs_cap_extension({"censored": True, "improvement_per_epoch": 0.0005}, False)
    assert not tuner.needs_cap_extension({"censored": False, "improvement_per_epoch": 0.01}, False)


def test_seeds_by_purpose():
    assert (
        tuner.seeds_for("first_look", CFG) == [0]
        and tuner.seeds_for("tune", CFG) == [0]
        and tuner.seeds_for("deep_tune", CFG) == [0]
    )
    assert tuner.seeds_for("confirm", CFG) == [1, 2, 3] and tuner.seeds_for("final", CFG) == [
        4,
        5,
        6,
    ]


def test_early_termination_reference_same_schedule_only():
    runs = [
        {"schedule": "onecycle", "checkpoint_best": {"0.5": 0.50, "0.75": 0.70}},
        {"schedule": "cosine_warmup", "checkpoint_best": {"0.5": 0.68, "0.75": 0.71}},
        {"schedule": "onecycle", "checkpoint_best": {"0.5": 0.55, "0.75": 0.69}, "exploded": True},
    ]
    ref = tuner.reference_for("onecycle", runs)
    assert ref == {
        "onecycle": {"0.5": 0.50, "0.75": 0.70}
    }  # a one-cycle run is never compared with cosine runs, exploded runs are skipped
    assert tuner.reference_for("constant", runs) == {}
    assert (
        tuner.early_term_spec("first_look", "onecycle", runs, CFG) is None
    )  # never in the first look
    assert tuner.early_term_spec("confirm", "onecycle", runs, CFG) is None  # never in confirmation
    spec = tuner.early_term_spec("tune", "onecycle", runs, CFG)
    assert (
        spec["margin"] == 0.03 and spec["checkpoints"] == [0.5, 0.75] and spec["reference"] == ref
    )


def test_triage_rules():
    base = tuner.default_recipe("cnn_residual", CFG)
    runs = [
        {"recipe": base, "status": "diverged", "train_r": None, "val_r": None},
        {"recipe": base, "status": "ok", "train_r": 0.90, "val_r": 0.70},  # overfit (gap 0.20)
        {
            "recipe": base,
            "status": "time_cap",
            "train_r": 0.7,
            "val_r": 0.69,
            "censored": True,
            "still_improving": True,
        },
    ]
    assert [tuner.diagnose(r) for r in runs] == ["optimisation", "overfit", "time_cap"]
    recs, caps, tri = tuner.triage_recipes(runs, "cnn_residual", CFG, rng(5))
    assert (
        recs[0]["lr"] == pytest.approx(base["lr"] / 5)
        and recs[0]["warmup_frac"] == 0.10
        and recs[0]["grad_clip"] == 1.0
    )
    assert (
        recs[1]["weight_decay"] == pytest.approx(base["weight_decay"] * 10)
        and recs[1]["ema"] == tuner.EMA_DECAY
    )
    assert recs[2] == base and caps == [1.0, 1.0, 1.5]
    assert tri["diagnoses"] == ["optimisation", "overfit", "time_cap"]


def test_failure_table():
    r = {"batch_size": 512}
    assert tuner.handle_failure("diverged", r) == {
        "action": "none",
        "recipe": None,
        "counts_toward_recipes": True,
    }
    oom = tuner.handle_failure("oom", r)
    assert (
        oom["action"] == "replace"
        and oom["recipe"]["batch_size"] == 256
        and not oom["counts_toward_recipes"]
    )
    assert tuner.handle_failure("oom", r, is_replacement=True)["action"] == "none"  # one retry only
    assert tuner.handle_failure("error", r, infra_retries_used=1)["action"] == "resubmit"
    assert tuner.handle_failure("error", r, infra_retries_used=2)["action"] == "none"
    assert tuner.handle_failure("exploded", r)["action"] == "none"


# ------------------------------------------------------------------ facts and lifecycle


def test_lifecycle_thresholds():
    top = 0.74
    assert derive_family_state(fs(first_look_best=0.70), top, CFG) == "parked"  # 0.04 below
    assert (
        derive_family_state(fs(first_look_best=0.72), top, CFG) == "tuning"
    )  # 0.02 below: within keep_margin 0.03
    assert (
        derive_family_state(fs(first_look_best=None), top, CFG) == "parked"
    )  # every first-look run failed
    assert (
        derive_family_state(
            fs(
                first_look_best=0.70,
                second_chance_used=True,
                second_chance_done=1,
                screen_best=0.70,
            ),
            top,
            CFG,
        )
        == "parked"
    )  # waiting for the second chance
    assert (
        derive_family_state(
            fs(
                second_chance_used=True,
                second_chance_done=3,
                recipes_planned=6,
                recipes_done=6,
                screen_best=0.70,
            ),
            top,
            CFG,
        )
        == "retired"
    )  # failed its second chance
    assert (
        derive_family_state(
            fs(
                second_chance_used=True,
                second_chance_done=3,
                recipes_planned=6,
                recipes_done=6,
                screen_best=0.725,
            ),
            top,
            CFG,
        )
        == "tuning"
    )
    done8 = dict(recipes_planned=8, recipes_done=8, first_look_best=0.73)
    assert (
        derive_family_state(fs(**done8, screen_best=0.725), top, CFG) == "confirming"
    )  # within 0.02
    assert (
        derive_family_state(fs(**done8, screen_best=0.715), top, CFG) == "tuned_only"
    )  # more than 0.02 below
    assert (
        derive_family_state(fs(**done8, screen_best=0.60, anchor=True), top, CFG) == "confirming"
    )  # anchors are always confirmed
    assert (
        derive_family_state(
            fs(**{**done8, "first_look_best": 0.70}, screen_best=0.70, anchor=True), None, CFG
        )
        == "confirming"
    )
    assert derive_family_state(FamilyState(family="x"), top, CFG) == "empty"
    conf = dict(**done8, screen_best=0.73, n_confirm=3, confirmed_mean=0.735)
    assert derive_family_state(fs(**conf), top, CFG) == "confirmed"
    assert (
        derive_family_state(fs(**{**conf, "confirmed_mean": 0.715}), top, CFG) == "retired"
    )  # more than 0.02 below the top
    assert (
        derive_family_state(fs(**conf, refinements=3, refinements_without_gain=3), top, CFG)
        == "plateaued"
    )
    assert (
        derive_family_state(fs(first_look_done=2, first_look_open=1, recipes_planned=3), top, CFG)
        == "first_look"
    )


def _run(
    i,
    code,
    recipe,
    seed,
    purpose,
    val,
    *,
    category="explore",
    phase="M",
    status="ok",
    family="unet",
    train=None,
):
    return RunInfo(
        run_id=f"r{i}",
        code_hash=code,
        family=family,
        recipe_hash=tuner.recipe_hash(recipe),
        recipe=recipe,
        schedule=recipe["schedule"],
        seed=seed,
        purpose=purpose,
        category=category,
        phase=phase,
        status=status,
        val_r=val,
        train_r=train,
        cost_equiv=1.0,
    )


def test_build_family_state_replays_accepted_refinement():
    ra, rb, rc = (tuner.default_recipe("unet", CFG, {"w": k}) for k in (1, 2, 3))
    runs = [
        _run(0, "p", ra, 0, "first_look", 0.70),
        _run(1, "p", rb, 0, "first_look", 0.72),
        _run(2, "p", rc, 0, "first_look", 0.71),
    ]
    runs += [_run(10 + s, "p", rb, s, "confirm", 0.72 + 0.001 * s) for s in (1, 2, 3)]
    cands = [
        {"code_hash": "p", "origin": "explore"},
        {
            "code_hash": "k",
            "origin": "refine",
            "parent_code_hash": "p",
            "angle": "capacity (width and depth)",
        },
    ]
    st = build_family_state("unet", runs, cands, [], CFG)
    assert (
        st.best_code_hash == "p"
        and st.best_recipe_hash == tuner.recipe_hash(rb)
        and st.screen_best == 0.72
    )
    assert (
        st.n_confirm == 3 and st.confirmed_mean == pytest.approx(0.722) and st.first_look_done == 3
    )
    assert (
        st.pending_refinement is not None
        and st.pending_refinement.code_hash == "k"
        and st.angle_index == 1
    )
    runs += [_run(20, "k", rb, 0, "refine", 0.735, category="refine")] + [
        _run(21 + s, "k", rb, s, "confirm", 0.735, category="refine") for s in (1, 2, 3)
    ]
    out = [
        {"kind": "refinement_outcome", "child": "k", "accepted": True, "effect": 0.013, "se": 0.004}
    ]
    st2 = build_family_state("unet", runs, cands, out, CFG)
    assert (
        st2.best_code_hash == "k"
        and st2.confirmed_mean == pytest.approx(0.735)
        and st2.pending_refinement is None
    )
    assert st2.refinements == 1 and st2.refinements_without_gain == 0


def test_build_family_state_excludes_exploded_and_counts_diverged():
    r = [tuner.default_recipe("unet", CFG, {"w": k}) for k in range(3)]
    runs = [
        _run(0, "p", r[0], 0, "first_look", 0.9, status="ok"),
        _run(1, "p", r[1], 0, "first_look", None, status="diverged"),
        _run(2, "p", r[2], 0, "first_look", 0.7),
    ]
    runs[0].exploded = True
    st = build_family_state("unet", runs, [{"code_hash": "p", "origin": "explore"}], [], CFG)
    assert (
        st.first_look_best == 0.7 and st.first_look_done == 3 and st.recipes_done == 3
    )  # an exploded run cannot be the best; a diverged recipe counts toward the 8


def test_menu_and_noise_rules():
    fams = {
        n: fs(family=n, n_confirm=3, confirmed_mean=m, confirmed_sd=0.003)
        for n, m in [("a", 0.75), ("b", 0.745), ("c", 0.70), ("d", 0.69), ("e", 0.68), ("f", 0.60)]
    }
    m = menu_select(fams, CFG)
    assert (
        m["within_margin"] == ["a", "b"]
        and m["menu"] == ["a", "b", "c", "d", "e"]
        and m["flagged"] == ["c", "d", "e"]
    )  # fewer than 5 qualify: top 5, outliers flagged
    fams2 = {
        n: fs(family=n, n_confirm=3, confirmed_mean=0.75 - 0.001 * i)
        for i, n in enumerate("abcdefg")
    }
    assert len(menu_select(fams2, CFG)["menu"]) == 7 and menu_select(fams2, CFG)["flagged"] == []
    assert estimate_sigma({}, CFG) == 0.005  # prior until 3 families are confirmed
    assert estimate_sigma(fams, CFG) == pytest.approx(0.003)


def test_acceptance_threshold_is_max_of_floor_and_two_se():
    a = acceptance([0.753, 0.754, 0.755], 0.745, 0.005, CFG)
    assert a["threshold"] == pytest.approx(max(0.005, 2 * 0.005 * math.sqrt(2 / 3)))  # 0.00816
    assert a["accepted"] and a["effect"] == pytest.approx(0.009)
    assert not acceptance([0.752, 0.753, 0.754], 0.745, 0.005, CFG)[
        "accepted"
    ]  # +0.008 is below 0.00816
    assert not acceptance([0.750, 0.750, 0.750], 0.745, 0.005, CFG)[
        "accepted"
    ]  # +0.005 does not exceed 0.00816
    assert (
        acceptance([0.750, 0.750, 0.750], 0.745, 0.001, CFG)["threshold"] == 0.005
    )  # low seed noise: the floor


def test_decide_screen_rule():
    res, base = [], {a: {1: 0.70, 2: 0.70} for a in ANCHORS}

    def put(opt, effects):
        for a, e in zip(ANCHORS, effects):
            for s in (1, 2):
                res.append({"option": opt, "anchor": a, "seed": s, "val_r": 0.70 + e})

    put("ema", (0.008, 0.007, 0.009))  # adopted
    put("corr", (0.010, 0.012, -0.001))  # mean 0.007 but one anchor negative: not adopted
    put("binned", (0.004, 0.004, 0.004))  # below 0.005
    put("huber", (0.0, 0.0, 0.0))
    put("hetero", (0.001, -0.001, 0.0))
    d = decide_screen(res, base, CFG)
    assert d["adopted"] == ["ema"] and not d["needs_combination"]
    assert set(d["not_testable"]) == {
        "huber",
        "hetero",
    }  # null results on noise-free labels are not_testable_in_silico, not rejected
    put("multitask", (0.006, 0.006, 0.006))
    d = decide_screen(res, base, CFG)
    assert d["adopted"] == ["ema", "multitask"] and d["needs_combination"]  # at most 2


def test_family_view_requires_confirmation():
    with pytest.raises(ValueError):
        family_view(fs(n_confirm=1), 0.7, 0.005)
    v = family_view(
        fs(
            n_confirm=3,
            confirmed_mean=0.7,
            refinement_history=[{"effect": 0.01, "se": 0.003, "accepted": True, "child": "x"}],
        ),
        0.72,
        0.005,
    )
    assert set(v) == {
        "family",
        "confirmed_mean",
        "confirmed_se",
        "n_seeds",
        "top_mean",
        "earlier_changes",
    }


# ------------------------------------------------------------------ forecast


def test_plan_matches_the_spec_table():
    rows = {e: forecast(CFG, e) for e in (0.8, 0.6, 0.4, 0.2)}
    assert [round(rows[e]["new_families"]) for e in (0.8, 0.6, 0.4, 0.2)] == [10, 8, 5, 3]
    assert [round(rows[e]["refinements"]) for e in (0.8, 0.6, 0.4, 0.2)] == [3, 6, 10, 13]
    assert [round(rows[e]["deep_tune_recipes"]) for e in (0.8, 0.6, 0.4, 0.2)] == [
        6,
        12,
        18,
        24,
    ]  # the spec table prints 25 for 0.2 (24.5 rounded up)
    assert "explore_fraction" in plan(CFG) and "configured" in plan(CFG)


# ------------------------------------------------------------------ decisions


def empty_snapshot(**kw) -> Snapshot:
    s = Snapshot(
        step=1,
        D=30_000,
        budgets=dict(CFG["controller"]["budget"]),
        taxonomy=list(TAXONOMY),
        balance=dict(CFG["balance"]),
    )
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_phase_a_registers_the_three_anchors_without_llm_calls():
    acts = Controller(CFG).decide(empty_snapshot(), 4, 2)
    ex = [a for a in acts if a.mode == "EXPLORE"]
    assert sorted(a.target["family"] for a in ex) == sorted(ANCHORS) and all(
        a.target["anchor"] and a.target["source"] == "reference" and a.phase == "A" for a in ex
    )
    assert not any(a.mode in ("SKETCH", "REFINE") for a in acts)


def test_first_look_for_an_unplanned_candidate_charged_to_phase_a_for_anchors():
    u = {
        "code_hash": "h1",
        "family": "cnn_residual",
        "origin": "anchor",
        "parent_code_hash": None,
        "suggested_recipe": {"lr": 2e-3},
        "hparams": {"width": 128},
        "tunable": {"width": [64, 256]},
    }
    snap = empty_snapshot(
        new_candidates_unplanned=[u],
        families={
            "cnn_residual": FamilyState(family="cnn_residual", n_candidates=1, best_code_hash="h1")
        },
    )
    acts = Controller(CFG).decide(snap, 4, 2)
    t = [a for a in acts if a.mode == "TUNE"]
    assert (
        len(t) == 1
        and t[0].purpose == "first_look"
        and len(t[0].recipes) == 3
        and t[0].phase == "A"
        and t[0].category == "anchor"
        and t[0].seeds == [0, 0, 0]
    )


def test_decisions_are_deterministic_and_replayable():
    snap = empty_snapshot(
        spend={"A": 20, "S": 30},
        families={
            a: fs(
                family=a,
                anchor=True,
                n_confirm=3,
                confirmed_mean=0.70 + 0.001 * i,
                recipes_planned=8,
                recipes_done=8,
                best_code_hash=a,
                best_recipe=tuner.default_recipe(a, CFG),
            )
            for i, a in enumerate(ANCHORS)
        },
        screen_decision_done=True,
        procedures={"done": True},
    )
    a1 = Controller(CFG).decide(copy.deepcopy(snap), 4, 2)
    a2 = Controller(CFG).decide(copy.deepcopy(snap), 4, 2)
    assert [x.to_dict() for x in a1] == [x.to_dict() for x in a2]
    c = Controller(CFG)
    c.guard_state = {"explore_offset": -0.2, "explore_mark": 4}
    replayed = Controller(CFG)
    replayed.replay([Action("NOTE", {"kind": "guard_state", "state": c.state_dict()}).to_dict()])
    assert replayed.guard_state == c.guard_state
    assert (
        Controller(CFG).rng(5, "x").random()
        == Controller(CFG).rng(5, "x").random()
        != Controller(CFG).rng(6, "x").random()
    )


def test_budget_is_all_or_nothing_and_never_exceeded():
    u = {
        "code_hash": "h1",
        "family": "cnn_residual",
        "origin": "anchor",
        "parent_code_hash": None,
        "suggested_recipe": {},
        "hparams": {},
        "tunable": {},
    }
    snap = empty_snapshot(
        new_candidates_unplanned=[u],
        families={
            "cnn_residual": FamilyState(family="cnn_residual", n_candidates=1, best_code_hash="h1")
        },
        spend={"A": 35.0},
    )  # 36 = 30 + borrow 6: only 1 left
    acts = Controller(CFG).decide(snap, 4, 2)
    assert not [
        a for a in acts if a.mode == "TUNE"
    ]  # a first look needs 3 and only 1 is left: it is not started at all


def test_balance_targets_scheduled_floor_and_bounds():
    c = Controller(CFG)
    fams = {n: FamilyState(family=n) for n in TAXONOMY}
    snap = empty_snapshot(cum_main_spend=0.0)
    e0, _ = c.explore_target(snap, fams)
    assert e0 == pytest.approx(0.75)
    snap = empty_snapshot(
        cum_main_spend=102.0, spend={"A": 30.0, "S": 36.0, "P": 12.0}
    )  # the unspent phase budgets are zero, so the effective main budget is 102
    e1, info = c.explore_target(snap, fams)
    assert (
        e1 == pytest.approx(0.5) and info["floor_active"]
    )  # end value 0.40, but fewer than 8 families are tuned: the explore share cannot fall below 0.5
    for f in list(fams.values())[:8]:
        f.recipes_planned, f.recipes_open = 8, 0
    assert c.explore_target(snap, fams)[0] == pytest.approx(0.40)
    c.guard_state = {"explore_offset": -0.8}
    assert c.explore_target(snap, fams)[0] == pytest.approx(0.2)  # explore_min
    c.guard_state = {"explore_offset": 0.8}
    assert c.explore_target(snap, fams)[0] == pytest.approx(0.8)  # explore_max
    snap.balance = {**CFG["balance"], "mode": "fixed", "explore_fraction": 0.3}
    c.guard_state = {}
    assert c.explore_target(snap, fams)[0] == pytest.approx(0.3)
    t = c.targets(snap, fams)
    assert t["explore"] + t["refine"] + t["deep_tune"] == pytest.approx(1.0) and t[
        "deep_tune"
    ] == pytest.approx(0.7 * 0.3)


def test_guards_fire_once_per_window_and_log_outcomes():
    c = Controller(CFG)
    snap = empty_snapshot(recent_new_family_outcomes=["parked", "retired", "tuned_only", "parked"])
    notes: list[Action] = []
    c._update_guards(snap, notes)
    assert (
        c.guard_state["explore_offset"] == pytest.approx(-0.2)
        and notes[0].target["guard"] == "explore_drought"
    )
    notes.clear()
    c._update_guards(snap, notes)  # same outcomes: no second firing
    assert not notes and c.guard_state["explore_offset"] == pytest.approx(-0.2)
    snap.recent_exploit_outcomes = [{"kind": "refine", "accepted": False}] * 3 + [
        {"kind": "deep_tune", "accepted": False}
    ]
    c._update_guards(snap, notes)
    assert notes[0].target["guard"] == "exploit_drought" and c.guard_state[
        "explore_offset"
    ] == pytest.approx(0.0)
    snap.recent_exploit_outcomes += [
        {"kind": "deep_tune", "accepted": True},
        {"kind": "deep_tune", "accepted": True},
        {"kind": "deep_tune", "accepted": False},
        {"kind": "deep_tune", "accepted": True},
    ]
    notes.clear()
    c._update_guards(snap, notes)
    assert any(n.target["guard"] == "recipes_matter" for n in notes) and c.guard_state[
        "deep_offset"
    ] == pytest.approx(0.2)
    c2 = Controller(CFG)  # both droughts at once: the stop rule decides, no adjustment
    snap2 = empty_snapshot(
        recent_new_family_outcomes=["parked"] * 4,
        recent_exploit_outcomes=[{"kind": "refine", "accepted": False}] * 4,
    )
    notes2: list[Action] = []
    c2._update_guards(snap2, notes2)
    assert (
        notes2[0].target["guard"] == "both_droughts"
        and c2.guard_state.get("explore_offset", 0.0) == 0.0
    )
    c3 = Controller(CFG)
    snap3 = empty_snapshot(
        recent_new_family_outcomes=["parked"] * 4, balance={**CFG["balance"], "guards": False}
    )
    c3._update_guards(snap3, [])
    assert not c3.guard_state


def test_adaptation_signals_and_hysteresis():
    c = Controller(CFG)
    fams = {n: FamilyState(family=n, n_candidates=i) for i, n in enumerate(TAXONOMY)}

    def ex(landed, size, ok=True, dup=False, spd=False, assigned="unet"):
        return {
            "assigned_family": assigned,
            "landed_family": landed,
            "landed_family_size": size,
            "first_attempt_gate_ok": ok,
            "duplicate": dup,
            "speed_fail": spd,
            "mechanism": "scan",
        }

    snap = empty_snapshot(recent_calls={"EXPLORE": [ex("cnn_residual", 5), ex("cnn_residual", 6)]})
    a = c.adaptation(snap, fams)
    assert (
        a["signals"]["collapse_in_a_row"] == 2
        and len(a["extra_constraints"]) == 2
        and not a.get("drop_sketch")
    )
    snap.recent_calls["EXPLORE"].append(ex("cnn_residual", 7))
    assert c.adaptation(snap, fams)["drop_sketch"]
    snap = empty_snapshot(
        recent_calls={"EXPLORE": [ex("cnn_residual", 6)] * 4}
    )  # none lands in its assigned family: low novelty
    a = c.adaptation(snap, fams)
    assert (
        a["forced_families"]
        and len(a["forced_families"]) == 6
        and a["forced_families"][0] == TAXONOMY[0]
    )  # the six least populated families
    snap = empty_snapshot(
        recent_calls={"EXPLORE": [ex("unet", 0, ok=False)] * 5 + [ex("unet", 0)] * 5}
    )  # 50% gate failures: failure view on
    assert c.adaptation(snap, fams)["failure_view"]
    snap = empty_snapshot(
        recent_calls={"EXPLORE": [ex("unet", 0, ok=False)] * 3 + [ex("unet", 0)] * 7}
    )  # 30%: between the thresholds, stays on
    assert c.adaptation(snap, fams)["failure_view"]
    snap = empty_snapshot(
        recent_calls={"EXPLORE": [ex("unet", 0, ok=False)] + [ex("unet", 0)] * 9}
    )  # 10%: off
    assert not c.adaptation(snap, fams)["failure_view"]
    snap = empty_snapshot(
        recent_calls={"EXPLORE": [ex("unet", 0, dup=True)] * 3 + [ex("unet", 0, spd=True)] * 2}
    )
    a = c.adaptation(snap, fams)
    assert a["extend_do_not_repeat"] == 20 and a["rotate_reference"] and a["speed_note"]


def test_sketch_novelty_prefers_empty_families_and_distant_mechanisms():
    fams = {
        "unet": FamilyState(family="unet", n_candidates=0),
        "cnn_plain": FamilyState(family="cnn_plain", n_candidates=1),
        "cnn_residual": FamilyState(family="cnn_residual", n_candidates=6),
    }
    summ = [{"family": "cnn_residual", "summary": "residual conv blocks with squeeze and excite"}]
    n_empty = sketch_novelty(
        {"family": "unet", "name": "u", "mechanism": "encoder decoder skip"}, fams, summ
    )
    n_thin = sketch_novelty(
        {"family": "cnn_plain", "name": "p", "mechanism": "stacked conv"}, fams, summ
    )
    n_crowded_same = sketch_novelty(
        {
            "family": "cnn_residual",
            "name": "r",
            "mechanism": "residual conv blocks with squeeze and excite",
        },
        fams,
        summ,
    )
    assert n_empty > n_thin > n_crowded_same >= 0


def test_stop_rule_flat_and_budget():
    c = Controller(CFG)
    fams = {f"f{i}": fs(family=f"f{i}", state="confirmed") for i in range(9)}
    snap = empty_snapshot(
        spend={"A": 25, "S": 30, "M": 70}, cum_main_spend=70.0, last_progress_spend=10.0
    )
    for f in fams.values():
        f.state = "confirmed"
    stop, why, flat = c.stop_check(snap, fams)
    assert (
        stop and flat and why.startswith("flat")
    )  # 9 tuned families, 60 run-equivalents without progress
    snap.last_progress_spend = 40.0
    assert not c.stop_check(snap, fams)[0]
    snap.spend["M"] = 200.0
    assert c.stop_check(snap, fams)[1] == "main budget spent" and not c.stop_check(snap, fams)[2]
