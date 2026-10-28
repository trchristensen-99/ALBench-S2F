"""Tests for the explore-then-exploit manager of the multi-agent loop (scripted fake LLM, tiny CPU data)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from test_agent_coordinator import CoordLLM  # noqa: E402
from test_agent_loop import make_args, make_harness  # noqa: E402

from albench.arch_search import agent_loop as al  # noqa: E402


def cfg(manager: str, frac: float = 0.4) -> str:
    return json.dumps(
        {
            "stages": ["propose", "coordinate", "code", "evaluate"],
            "ideas_per_round": 4,
            "manager": manager,
            "explore_frac": frac,
        }
    )


def test_manager_config_validation():
    assert al.AgentConfig.from_arg(cfg("schedule")).manager == "schedule"
    for bad in (
        '{"manager": "wild"}',
        '{"stages": ["propose", "code", "evaluate"], "manager": "schedule"}',
        '{"stages": ["propose", "coordinate", "code", "evaluate"], "manager": "schedule", "explore_frac": 1.5}',
    ):
        with pytest.raises(al.AgentConfigError):
            al.AgentConfig.from_arg(bad)
    assert (
        al.AgentConfig.from_arg("").manager == "none"
    )  # default leaves every earlier arm unchanged


def test_manager_mode_schedule():
    class C:  # minimal ctx stand-in
        pass

    c = C()
    c.cfg = al.AgentConfig.from_arg(cfg("schedule", 0.4))
    c.args = make_args(Path("."), rounds=15)
    assert [al.manager_mode(c, r) for r in range(15)] == ["explore"] * 6 + ["exploit"] * 9
    c.cfg = al.AgentConfig.from_arg(cfg("none"))
    assert al.manager_mode(c, 0) is None


def test_loop_with_manager_gives_mode_dependent_prompts(tmp_path):
    h, args, llm = (
        make_harness(tmp_path),
        make_args(tmp_path, rounds=4, agent_cfg=cfg("schedule_ens", 0.5)),
        CoordLLM(),
    )
    al.run_agent_loop(args, h, llm=llm)
    out = Path(args.out_dir)
    plans = [json.loads((out / "agent_logs" / f"plan_r{r:02d}.json").read_text()) for r in range(4)]
    assert [p["coordinator"].get("manager_mode") for p in plans] == [
        "explore",
        "explore",
        "exploit",
        "exploit",
    ]
    assert (
        "MANAGER MODE: EXPLORE" in llm.coord_prompts[0]
        and "COVERAGE of the 0 trained models" in llm.coord_prompts[0]
    )
    assert "MANAGER MODE: EXPLOIT" in llm.coord_prompts[-1]
    assert "ENSEMBLE STATUS" in llm.coord_prompts[-1]  # >= 4 trained models by round 3
    assert (
        "EXPLORATION ROUND" in llm.proposer_systems[0]
        and "REFINEMENT ROUND" in llm.proposer_systems[-1]
    )
    assert "weight" in llm.coord_prompts[-1]


def test_manager_off_leaves_prompts_unchanged(tmp_path):
    h, args, llm = (
        make_harness(tmp_path),
        make_args(tmp_path, rounds=3, agent_cfg=cfg("none")),
        CoordLLM(),
    )
    al.run_agent_loop(args, h, llm=llm)
    assert all("MANAGER MODE" not in p and "COVERAGE" not in p for p in llm.coord_prompts)
    assert all(
        "EXPLORATION ROUND" not in s and "REFINEMENT ROUND" not in s for s in llm.proposer_systems
    )
