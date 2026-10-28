"""B1: the ledger (idempotent runs, append-only), derived tables that rebuild exactly (T6), no path from the prompt builders to final_scores (T9), config freezing and
the live balance override rules (T16)."""

from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path

import b1_helpers as h
import pytest

from ar2 import config as cfgmod
from ar2.memory import archive, final
from ar2.memory.db import DB

ROOT = Path(__file__).resolve().parents[2]


def _db(tmp_path) -> DB:
    return DB(tmp_path / "ar2.db")


def _run(db, code="c1", rh="r1", seed=0, purpose="tune", **kw):
    row = {
        "run_id": f"r_{code}_{rh}_{seed}_{purpose}",
        "code_hash": code,
        "recipe": h.recipe(),
        "recipe_hash": rh,
        "pool_id": "p",
        "D": 1000,
        "seed": seed,
        "purpose": purpose,
        "phase": "M",
        "category": "explore",
        "family": "cnn_plain",
        "schedule": "onecycle",
        **kw,
    }
    return db.upsert_run(row), row["run_id"]


def test_runs_are_idempotent_and_ledger_rows_are_never_deleted(tmp_path):
    db = _db(tmp_path)
    ok1, rid = _run(db)
    ok2, _ = _run(
        db
    )  # same (code, recipe, pool, seed, purpose): the UNIQUE key makes resubmission a no-op
    assert ok1 and not ok2 and len(db.runs_for()) == 1
    db.finish_run(rid, status="ok", val_r=0.7, epochs_run=20, cost_equiv=0.5)
    assert db.run(rid)["status"] == "ok" and db.spend() == 0.5
    assert db.runs_for(pending=True) == []
    assert db.spend(phase="M", category="explore") == 0.5 and db.spend(phase="A") == 0.0
    with pytest.raises(
        sqlite3.IntegrityError
    ):  # a different run id with the same key is refused outright
        db.con.execute(
            "INSERT INTO runs(run_id,code_hash,recipe_hash,pool_id,seed,purpose) VALUES('x','c1','r1','p',0,'tune')"
        )


def test_wal_mode_and_meta(tmp_path):
    db = _db(tmp_path)
    assert db.con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    db.meta_set("k", {"a": 1})
    assert json.loads(db.meta_get("k")) == {"a": 1} and db.meta_get("schema_version") == "1"


def _refinement_ledger(tmp_path):
    """A parent and an accepted child, three confirmation seeds each, the refinement outcome as a NOTE decision, and one LLM wording on record."""
    db = _db(tmp_path)
    for ch, parent, origin in (("par", None, "explore"), ("kid", "par", "refine")):
        db.insert_candidate(
            {
                "code_hash": ch,
                "family": "cnn_plain",
                "family_claimed": "cnn_plain",
                "gate_status": "ok",
                "parent_code_hash": parent,
                "source": "x",
                "tags": {},
                "card": {
                    "summary": ch,
                    "_ar2": {"origin": origin, "angle": "capacity (width and depth)"},
                },
                "phase": "M",
            }
        )
    for ch, vals in (("par", [0.70, 0.71, 0.70]), ("kid", [0.73, 0.74, 0.735])):
        rec = h.recipe()
        for i, v in enumerate((0.69 if ch == "par" else 0.72, 0.68 if ch == "par" else 0.71)):
            _run(
                db,
                ch,
                f"s{i}",
                0,
                "first_look" if ch == "par" else "refine",
                status="ok",
                val_r=v,
                cost_equiv=1.0,
                epochs_run=40,
                recipe=rec,
                finished_at=f"2026-10-28T0{i}:00:00",
            )
        for s, v in zip((1, 2, 3), vals):
            _run(
                db,
                ch,
                "s0",
                s,
                "confirm",
                status="ok",
                val_r=v,
                cost_equiv=1.0,
                epochs_run=40,
                finished_at="2026-10-28T05:00:0%d" % s,
            )
    db.insert_decision(
        5,
        "NOTE",
        "x",
        "outcome",
        "h",
        {
            "kind": "refinement_outcome",
            "family": "cnn_plain",
            "child": "kid",
            "accepted": True,
            "effect": 0.034,
            "se": 0.004,
        },
    )
    ek = archive.evidence_key(
        "refinement",
        "cnn_plain",
        "capacity (width and depth)",
        [r["run_id"] for r in db.runs_for(purpose="confirm")],
    )
    db.put_lesson_text(ek, "[cnn_plain] widening the tower helped")
    return db


def _rows(db, table):
    return [tuple(r) for r in db.con.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()]


def test_T6_rebuild_reproduces_families_and_lessons_exactly(tmp_path):
    db = _refinement_ledger(tmp_path)
    cfg = cfgmod.load_config()
    archive.rebuild(db, cfg)
    fam1, les1 = _rows(db, "families"), _rows(db, "lessons")
    assert fam1 and les1
    lesson = db.lessons()[0]
    assert (
        "widening the tower helped" in lesson["statement"]
        and "+0.0" in lesson["statement"]
        and lesson["scope"] == "cnn_plain"
    )  # wording from the ledger, numbers from code
    assert lesson["n"] == 3 and abs(lesson["effect"] - (0.735 - 0.7033333)) < 0.002
    db.con.execute("DELETE FROM families")
    db.con.execute("DELETE FROM lessons")
    archive.rebuild(db, cfg)
    assert _rows(db, "families") == fam1 and _rows(db, "lessons") == les1
    # the accepted child replaced its parent as the family's best, and the archive score is the mean of its confirmation seeds only (T10)
    fam = db.families()[0]
    assert fam["best_code_hash"] == "kid" and fam["n_confirm"] == 3
    assert abs(fam["confirmed_mean"] - sum([0.73, 0.74, 0.735]) / 3) < 1e-9
    assert fam["refinements"] == 1 and fam["refinements_without_gain"] == 0


def test_lesson_without_three_seeds_is_not_written(tmp_path):
    db = _refinement_ledger(tmp_path)
    db.con.execute("DELETE FROM runs WHERE code_hash='par' AND purpose='confirm' AND seed=3")
    assert archive.derive_lessons(db, cfgmod.load_config()) == []


def test_lesson_is_retracted_by_later_opposite_evidence(tmp_path):
    cfg = cfgmod.load_config()
    db = _refinement_ledger(tmp_path)
    for ch, parent in (("par2", None), ("kid2", "par2")):
        db.insert_candidate(
            {
                "code_hash": ch,
                "family": "cnn_plain",
                "family_claimed": "cnn_plain",
                "gate_status": "ok",
                "parent_code_hash": parent,
                "source": "x",
                "tags": {},
                "card": {
                    "summary": ch,
                    "_ar2": {
                        "origin": "refine" if parent else "explore",
                        "angle": "capacity (width and depth)",
                    },
                },
                "phase": "M",
            }
        )
    for ch, vals in (("par2", [0.76, 0.76, 0.77]), ("kid2", [0.70, 0.71, 0.70])):
        for s, v in zip((1, 2, 3), vals):
            _run(
                db,
                ch,
                "z",
                s,
                "confirm",
                status="ok",
                val_r=v,
                cost_equiv=1.0,
                epochs_run=40,
                finished_at="2026-10-28T09:00:0%d" % s,
            )
    db.insert_decision(
        9,
        "NOTE",
        "x",
        "outcome",
        "h",
        {
            "kind": "refinement_outcome",
            "family": "cnn_plain",
            "child": "kid2",
            "accepted": False,
            "effect": -0.06,
            "se": 0.004,
        },
    )
    st = {x["effect"] > 0: x["status"] for x in archive.derive_lessons(db, cfg)}
    assert st[True] == "retracted" and st[False] == "confirmed"


def test_T9_prompt_builders_cannot_reach_final_scores():
    for rel in ("ar2/views.py", "ar2/tagger.py", "ar2/lessons.py"):
        p = ROOT / rel
        if not p.exists():
            continue
        src = p.read_text()
        assert (
            "final_scores" not in src
            and "memory.final" not in src
            and "memory import final" not in src
        ), rel
        tree = ast.parse(src)
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            assert not any(m.startswith("ar2.memory") for m in mods), f"{rel} imports {mods}"
    for f in (ROOT / "ar2" / "prompts").rglob("*"):
        if f.is_file():
            assert "final_scores" not in f.read_text(errors="ignore")
    assert "final_scores" not in (ROOT / "ar2/memory/db.py").read_text().replace(
        "never reads ``final_scores``", ""
    ).replace("``final_scores``", "")


def test_final_scores_roundtrip_through_final_module_only(tmp_path):
    db = _db(tmp_path)
    final.write_final_score(db.path, "c", "r", 4, 0.71, 0.72)
    final.write_final_score(db.path, "c", "r", 4, 0.71, 0.72)  # idempotent
    rows = final.read_final_scores(db.path)
    assert len(rows) == 1 and rows[0]["val_select_r"] == 0.71
    assert not hasattr(db, "final_scores") and not hasattr(db, "read_final_scores")


def test_config_hash_and_live_balance_rules_T16(tmp_path):
    ws = h.make_ws(tmp_path)
    cfg0 = cfgmod.load_frozen(ws)
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.set_live(ws, "controller.budget.main", 5)  # outside balance: refused
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.set_live(ws, "balance.no_such_key", 1)
    cfgmod.set_live(ws, "balance.explore_fraction", 0.3)
    eff, pending = cfgmod.effective_config(ws)
    assert eff["balance"]["explore_fraction"] == 0.3 and len(pending) == 1
    assert cfgmod.load_frozen(ws) == cfg0  # the frozen config (and its hash) never move
    assert cfgmod.config_hash(cfgmod.load_frozen(ws)) == DB(ws / "ar2.db").meta_get("config_hash")


def test_deployment_profile_is_not_applied_unless_asked():
    base = cfgmod.load_config()
    dep = cfgmod.load_config(profile="deployment")
    assert "deployment" not in base and "deployment" not in dep
    assert base["controller"]["budget"] == {
        "anchors": 30,
        "screen": 36,
        "procedures": 12,
        "main": 102,
        "final": 20,
    }
    assert dep["controller"]["budget"] == {
        "anchors": 30,
        "screen": 0,
        "procedures": 0,
        "main": 50,
        "final": 18,
    }
    assert (
        dep["proposer"]["model"] == "claude-sonnet-5-5"
        and dep["balance"]["mode"] == "scheduled"
        and dep["jobs"]["backend"] == "queue"
    )
    assert (
        base["proposer"]["model"] is None
        and base["evaluator"]["base_cap_s"] == 1800
        and base["gate"]["max_params"] == 40_000_000
    )
