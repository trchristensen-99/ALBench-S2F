"""Prompt views: render tests for every mode, canary leakage test (T8), static test (T9), token limits, workdir preparation, template hash."""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ar2 import prompts, views
from ar2.proposer import mock_library as lib
from ar2.views import ViewContext, render

REPO = Path(__file__).resolve().parents[2]
CANARIES = [
    "CANARY_VALSEL_0.9871",
    "CANARY_LEGACY_0.7777",
    "CANARY_SINGLE_0.6543",
    "CANARY_FINAL_0.5555",
    "CANARY_SEEDSCORES_0.4444",
]


def canary_context(max_tokens: int = 12000) -> ViewContext:
    """A context that carries canary values in every place a careless view builder could leak them."""
    return ViewContext(
        D=30000,
        max_prompt_tokens=max_tokens,
        coverage=[
            {
                "family": "cnn_residual",
                "candidates": 14,
                "state": "confirmed",
                "note": "anchor",
                "score": CANARIES[2],
                "val_select_r": CANARIES[0],
            },
            {
                "family": "state_space",
                "candidates": 1,
                "state": "parked",
                "note": "optimisation failure",
                "legacy_score": CANARIES[1],
            },
        ],
        near=[
            {
                "family": "cnn_residual",
                "summary": "LegNet-like SE tower",
                "score": CANARIES[2],
                "legacy_score": CANARIES[1],
            }
        ],
        lessons_global=["[global] Dropout in the head: no effect"],
        lessons_family=[
            "[transformer_hybrid] Attention pooling: +0.007 +/- 0.003 (3 vs 3 seeds). Accepted."
        ],
        family={
            "family": "transformer_hybrid",
            "confirmed_mean": 0.741,
            "confirmed_se": 0.003,
            "top_mean": 0.748,
            "top_se": 0.002,
            "n_seeds": 3,
            "params": 4.2e6,
            "throughput": 21000,
            "screen_best": CANARIES[2],
            "seed_scores": CANARIES[4],
            "legacy_score": CANARIES[1],
            "val_select_r": CANARIES[0],
            "final_r": CANARIES[3],
            "diagnostics": {
                "train_r": 0.84,
                "val_r": 0.74,
                "best_epoch": 21,
                "max_epochs": 40,
                "stop_reason": "early_stop",
                "pred_sd_ratio": 0.88,
                "mse_by_activity_quintile": [0.11, 0.13, 0.17, 0.24, 0.42],
                "val_select_r": CANARIES[0],
                "single_seed_r": CANARIES[2],
            },
            "recipe_sensitivity": {
                "n": 8,
                "lo": 0.701,
                "hi": 0.739,
                "best_lr": 5e-4,
                "best_weight_decay": 0.05,
                "screen_scores": CANARIES[2],
            },
            "earlier_changes": [
                {
                    "text": "attention pooling in place of mean pooling",
                    "effect": 0.007,
                    "se": 0.003,
                    "accepted": True,
                },
                {
                    "text": "unconfirmed widening",
                    "effect": 0.0123456,
                    "se": 0.002,
                    "accepted": False,
                    "n_seeds": 1,
                },
            ],
        },
        failure_view=[{"code": "shape", "description": "output must be (B, out_dim)"}],
        extra_constraints=["The integration must be done by attention."],
        forced_families=["unet", "state_space"],
        speed_note="Parent throughput 21k sequences per second; cap 900 s.",
    )


ASSIGN = {
    "SKETCH": {},
    "EXPLORE": {
        "sketch": {
            "family": "unet",
            "name": "Encoder-decoder",
            "mechanism": "skips",
            "why_for_mpra": "multi-resolution",
        }
    },
    "REFINE": {"angle": "normalisation and activation"},
    "REPAIR": {
        "gate_code": "shape",
        "gate_description": "wrong output shape",
        "measured": {"shape": [2, 3]},
        "traceback": "Traceback ...",
    },
    "TAGGER": {"code": "import torch\nclass Net: pass\n"},
    "LESSON": {
        "evidence": {
            "kind": "refine",
            "scope": "transformer_hybrid",
            "change": "attention pooling",
            "direction": "improved",
            "accepted": True,
            "effect": 0.007,
            "se": 0.003,
        }
    },
}


@pytest.mark.parametrize("mode", views.MODES)
def test_every_mode_renders_clean(mode):
    text = render(mode, canary_context(), ASSIGN[mode])
    assert "{{" not in text
    assert views.estimate_tokens(text) <= 12000
    if mode in ("SKETCH", "EXPLORE", "REFINE", "REPAIR"):
        assert "You are designing neural network architectures" in text  # frozen core
        assert "30,000 sequences" in text
    if mode in ("SKETCH", "EXPLORE", "REFINE"):
        assert (
            text.index("OUTPUT:") > text.index("ASSIGNMENT:")
            and "OUTPUT:" in text.rstrip().splitlines()[-1]
            or mode == "SKETCH"
        )


@pytest.mark.parametrize("mode", views.MODES)
def test_t8_canaries_never_rendered(mode):
    text = render(mode, canary_context(), ASSIGN[mode])
    for c in CANARIES:
        assert c not in text, f"{c} leaked into the {mode} prompt"
    assert "0.0123456" not in text and "0.012" not in text  # an unconfirmed change shows no number


def test_scores_only_in_refine():
    ctx = canary_context()
    for mode in ("SKETCH", "EXPLORE"):
        text = render(mode, ctx, ASSIGN[mode])
        assert "0.741" not in text and "0.748" not in text and "confirmed 0." not in text
        assert "scores are withheld" in text if mode == "SKETCH" else True
    refine = render("REFINE", ctx, ASSIGN["REFINE"])
    assert "confirmed 0.741 +/- 0.003 (3 seeds)" in refine and "Top family: 0.748" in refine
    assert "attention pooling in place of mean pooling: +0.007 +/- 0.003, accepted" in refine
    assert "unconfirmed widening: rejected" in refine


def test_refine_refuses_unconfirmed_family():
    ctx = canary_context()
    ctx.family["n_seeds"] = 1
    with pytest.raises(ValueError, match="3 seeds"):
        render("REFINE", ctx, ASSIGN["REFINE"])
    ctx.family = None
    with pytest.raises(ValueError):
        render("REFINE", ctx, ASSIGN["REFINE"])


def test_forbidden_words_are_blocked():
    ctx = canary_context()
    ctx.lessons_global = ["Result on val_select was 0.7"]
    with pytest.raises(ValueError, match="forbidden"):
        render("EXPLORE", ctx, ASSIGN["EXPLORE"])


def test_placeholder_in_dynamic_text_does_not_trip_the_check():
    ctx = canary_context()
    ctx.near[0]["summary"] = "uses {{braces}} in its text"
    text = render("SKETCH", ctx, {})
    assert "{{" not in text and "{ {braces} }" in text


def test_missing_assignment_fields_raise():
    with pytest.raises(ValueError):
        render("EXPLORE", ViewContext(), {})
    with pytest.raises(ValueError):
        render("NOPE", ViewContext(), {})


def test_empty_taxonomy_families_are_listed_as_empty():
    text = render(
        "SKETCH",
        ViewContext(coverage=[{"family": "cnn_residual", "candidates": 3, "state": "tuning"}]),
        {},
    )
    assert re.search(r"unet\s+0\s+empty", text)
    assert re.search(r"cnn_residual\s+3\s+tuning", text)
    assert "FAMILY TAXONOMY" in text


def test_forced_families_extra_constraints_failure_view_in_assignment():
    text = render("SKETCH", canary_context(), {})
    assert "accepted only in these families: unet, state_space" in text
    assert "The integration must be done by attention." in text
    assert "RECENT AUTOMATIC-CHECK FAILURES" in text
    assert text.index("ASSIGNMENT:") > text.index(
        "COVERAGE"
    )  # reference material first, assignment last
    ex = render("EXPLORE", canary_context(), ASSIGN["EXPLORE"])
    assert "Parent throughput 21k" in ex


def test_token_limit_shrinks_views_not_the_assignment():
    ctx = ViewContext(
        max_prompt_tokens=3000,
        coverage=[
            {"family": f"other:fam{i}", "candidates": i, "state": "tuned", "note": "n" * 200}
            for i in range(300)
        ],
        near=[{"family": "cnn_residual", "summary": "s" * 190} for _ in range(20)],
        lessons_global=["[global] " + "lesson text " * 20 for _ in range(15)],
    )
    for mode in ("SKETCH", "EXPLORE"):
        text, meta = views.render_with_meta(mode, ctx, ASSIGN[mode])
        assert meta["tokens"] <= 3000 and meta["truncation_level"] >= 1
        assert "ASSIGNMENT:" in text and "OUTPUT:" in text and "WHAT THE DATA IS LIKE" in text
    refine_ctx = canary_context(3000)
    refine_ctx.family["earlier_changes"] = [
        {"text": "t" * 190, "effect": 0.001, "se": 0.002, "accepted": False} for _ in range(300)
    ]
    text, meta = views.render_with_meta("REFINE", refine_ctx, ASSIGN["REFINE"])
    assert meta["tokens"] <= 3000 and "OUTPUT:" in text and "Angle for this round" in text


def test_token_limit_impossible_raises():
    with pytest.raises(ValueError, match="over the limit"):
        render("SKETCH", ViewContext(max_prompt_tokens=100), {})


def test_large_default_prompts_fit():
    ctx = ViewContext(
        coverage=[
            {"family": f, "candidates": 20, "state": "confirmed", "note": "anchor"}
            for f in ("cnn_residual", "rnn_hybrid", "transformer_hybrid")
        ],
        near=[{"family": "cnn_residual", "summary": "x" * 150} for _ in range(20)],
        lessons_global=["[global] " + "w " * 40] * 15,
    )
    for mode in ("SKETCH", "EXPLORE"):
        assert views.estimate_tokens(render(mode, ctx, ASSIGN[mode])) < 12000


def test_t9_views_and_prompts_have_no_path_to_final_scores():
    banned = ("final_scores", "val_select")
    files = [REPO / "ar2" / "views.py", *sorted((REPO / "ar2" / "prompts").rglob("*"))]
    for f in files:
        if f.is_file() and f.suffix in (".py", ".md"):
            text = f.read_text()
            for b in banned:
                assert b not in text, f"{f.name} mentions {b}"
    tree = ast.parse((REPO / "ar2" / "views.py").read_text())
    imported = {
        a.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in n.names
    }
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any("memory" in m or "sqlite3" in m for m in mods | imported)


def test_template_hash_and_version(tmp_path, monkeypatch):
    v = views.prompt_version()
    assert re.fullmatch(r"v1\+[0-9a-f]{12}", v)
    assert views.prompt_version({"proposer": {"prompt_version": "v7"}}).startswith("v7+")
    h = prompts.template_hash()
    copy = tmp_path / "v1"
    shutil.copytree(prompts.PROMPT_DIR, copy)
    monkeypatch.setattr(prompts, "PROMPT_DIR", copy)
    assert prompts.template_hash() == h
    (copy / "core.md").write_text((copy / "core.md").read_text() + "\nextra line\n")
    assert prompts.template_hash() != h


def test_all_templates_exist_and_placeholders_are_known():
    known = {
        "core",
        "coverage_view",
        "do_not_repeat_view",
        "failure_view",
        "taxonomy",
        "forced_families",
        "extra_constraints",
        "lessons_view_global",
        "lessons_view_family",
        "family_view",
        "sketch_json",
        "speed_note",
        "angle",
        "gate_code",
        "gate_description",
        "measured",
        "traceback",
        "code",
        "evidence_json",
        "D",
        "n_recipes",
        "max_params",
        "time_cap_min",
        "noise_floor",
    }
    for name in prompts.PROMPT_FILES:
        text = (prompts.PROMPT_DIR / name).read_text()
        assert set(re.findall(r"\{\{(\w+)\}\}", text)) <= known, name


def test_prepare_workdir_modes(tmp_path):
    ref = views.reference_model_code(0)
    d = tmp_path / "w1"
    views.prepare_workdir(d, "EXPLORE", parent_code=None, reference_code=ref)
    assert {p.name for p in d.iterdir()} == {"CONTRACT.md", "selfcheck.py", "reference_model.py"}
    assert "build_model(out_dim: int, **hparams)" in (d / "CONTRACT.md").read_text()
    views.prepare_workdir(d, "REFINE", parent_code="# parent\n", reference_code=ref)
    assert (d / "parent_model.py").read_text() == "# parent\n"
    views.prepare_workdir(d, "SKETCH", parent_code=None, reference_code=ref)
    assert not (d / "parent_model.py").exists()
    with pytest.raises(ValueError):
        views.prepare_workdir(tmp_path / "w2", "REPAIR", parent_code=None, reference_code=ref)
    codes = {views.reference_model_code(i) for i in range(3)}
    assert len(codes) == 3 and views.reference_model_code(3) == views.reference_model_code(0)


def _run_selfcheck(workdir: Path):
    return subprocess.run(
        [sys.executable, "selfcheck.py"], cwd=workdir, capture_output=True, text=True, timeout=120
    )


def test_selfcheck_passes_reference_and_fails_wrong_shape(tmp_path):
    pytest.importorskip("torch")
    d = tmp_path / "w"
    views.prepare_workdir(
        d, "EXPLORE", parent_code=None, reference_code=views.reference_model_code(0)
    )
    (d / "model.py").write_text(lib.library_source("cnn_residual"))
    r = _run_selfcheck(d)
    assert r.returncode == 0 and "selfcheck passed" in r.stdout and "out_dim=2" in r.stdout
    (d / "model.py").write_text(lib.broken_source("wrong_shape", lib.library_source("cnn_plain")))
    r = _run_selfcheck(d)
    assert r.returncode == 1 and "expected output shape" in r.stdout
    (d / "model.py").write_text("x = 1\n")
    assert _run_selfcheck(d).returncode == 1
