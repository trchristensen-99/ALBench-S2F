"""The registry validator must catch the failures that the dispatcher would otherwise swallow."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "vr", ROOT / "orchestration" / "validate_registry.py"
)
vr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vr)


def cell(**kw) -> dict:
    c = {
        "name": "un30_x_s0",
        "gpus": ["h100", "v100"],
        "ext": [],
        "out_dir": "outputs/menu_retrain_v0/un30_x",
        "env": {"AL_PATIENCE": "8"},
        "needs": None,
        "cmd": "uv run --no-sync python scratchpad/menu_retrain.py --D 30000 --out_dir x",
        "done": {"type": "file", "path": "outputs/menu_retrain_v0/un30_x/COMPLETE_0of1"},
    }
    c.update(kw)
    return c


def test_valid_cell_passes():
    assert vr.schema_errors([cell()]) == []


def test_schema_errors_are_specific():
    bad = [
        cell(gpus=[]),
        cell(name="a b"),
        cell(env={"AL_X": 1}),
        cell(done={"type": "glob"}),
        cell(cmd="x --budget {BUDGT}"),
        cell(out_dir="/abs"),
        cell(extra=1),
    ]
    errs = vr.schema_errors(bad)
    for needle in (
        "gpus",
        "name must",
        "env must",
        "done.type",
        "BUDGT",
        "out_dir",
        "unknown keys",
    ):
        assert any(needle in e for e in errs), needle
    assert any("duplicate" in e for e in vr.schema_errors([cell(), cell()]))


def test_routing_uses_the_dispatcher_rules():
    disp = vr.load_dispatcher()
    fits = cell(name="zz98_x_s0")
    too_long = cell(
        name="zz99_x_s0", cmd="python scratchpad/menu_retrain.py --D 30000 --base_cap_min 900"
    )
    w = vr.routing_warnings([fits, too_long], disp)
    assert any(x.startswith("zz99_") and "2-day maximum" in x for x in w)
    assert not any(x.startswith("zz98_") and ("2-day" in x or "default" in x) for x in w)
    blocked = cell(name=disp.MIG_NO[0] + "x_s0")
    assert not any("MIG" in x for x in vr.routing_warnings([blocked], disp))


def test_unread_knob_is_reported(tmp_path):
    (tmp_path / "albench").mkdir()
    (tmp_path / "albench" / "m.py").write_text('import os\nos.environ.get("AL_READ")\n')
    w = vr.unread_knobs([cell(env={"AL_READ": "1", "AL_NOBODY": "1"})], tmp_path)
    assert w == ["AL_NOBODY set in 1 cells, read by no .py file"]


def test_pending_only_with_root(tmp_path):
    done = cell(name="zz99_done", done={"type": "file", "path": "D"})
    (tmp_path / "D").write_text("")
    todo = cell(name="zz98_todo", done={"type": "file", "path": "T"})
    assert vr.is_done(done, tmp_path) and not vr.is_done(todo, tmp_path)
    reg = tmp_path / "r.json"
    reg.write_text(json.dumps([done, todo]))
    assert vr.schema_errors(json.loads(reg.read_text())) == []


def test_menu_wall_follows_the_per_model_cap():
    disp = vr.load_dispatcher()

    def wall(name, cap, gpu, qos="slow_nice"):
        c = cell(name=name, cmd=f"python scratchpad/menu_retrain.py --D 30000 --base_cap_min {cap}")
        return disp.wall_and_budget(c, qos, gpu)[0]

    assert wall("zz_x", 25, "h100") == "03:30:00"  # floor
    assert wall("zz_x", 25, "v100") == "03:30:00"  # 125 + 45 min fits the floor
    assert wall("zz_x", 60, "v100") == "06:00:00"  # 300 + 45, rounded up to 30 min
    assert wall("un30_x", 60, "v100") == "06:00:00"  # a RETRAIN_PFX name no longer means 3:30
    assert wall("zz_x", 900, "v100") == "48:00:00"  # slow_nice maximum
    assert wall("a1m_x", 25, "v100") == "14:00:00"  # deliberate family rule kept
    assert wall("sw38_100k_x", 25, "v100") == "10:00:00"
    assert wall("zz_x", 60, "h100", qos="default") == disp.WALL["default"]  # other queues unchanged
    llm = cell(name="g30_x", cmd="python scratchpad/arch_autoresearch.py --D 30000")
    assert disp.wall_and_budget(llm, "slow_nice", "h100")[0] == "08:00:00"
