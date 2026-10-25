"""Tests for the coordinator stage and the proposer_style option of the multi-agent loop (scripted fake LLM, tiny CPU data)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("torch")

from albench.arch_search import agent_loop as al  # noqa: E402
from test_agent_loop import FakeLLM, make_args, make_harness  # noqa: E402

CFG = '{"stages": ["propose", "coordinate", "code", "evaluate", "reflect"], "ideas_per_round": 4}'


class CoordLLM(FakeLLM):
    """Adds a coordinator that picks candidates 3 and 1 (1-based)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.n["coordinator"] = 0
        self.coord_prompts: list[str] = []
        self.proposer_systems: list[str] = []

    def __call__(self, system, user, model=None):
        if "You are the COORDINATOR" in system:
            self.n["coordinator"] += 1
            self.coord_prompts.append(system + "\n" + user)
            k = int(re.search(r"### SELECT: <(\d+) candidate", system).group(1))
            return "### SELECT: " + ", ".join(["3", "1"][:k]) + "\n### RATIONALE: because.\n"
        if "You are the PROPOSER" in system:
            self.proposer_systems.append(system)
        return super().__call__(system, user, model)


def test_parse_selection_is_robust():
    assert al.parse_selection("### SELECT: 3, 3, 9, 1\n### RATIONALE: x", 4, 2) == [2, 0]
    assert al.parse_selection("garbage", 4, 2) == [0, 1]
    assert al.parse_selection("### SELECT: 4", 4, 2) == [3, 0]
    assert al.parse_selection("### SELECT: 0, 5, 2", 4, 2) == [1, 0]


def test_config_validation_for_coordinator_and_style():
    cfg = al.AgentConfig.from_arg(CFG)
    assert cfg.stages[1] == "coordinate" and cfg.ideas_per_round == 4
    for bad in (
        '{"stages": ["coordinate", "propose", "code", "evaluate"]}',
        '{"stages": ["propose", "code", "coordinate", "evaluate"]}',
        '{"stages": ["code", "coordinate", "evaluate"]}',
        '{"proposer_style": "wild"}',
        '{"role_models": {"coordinator": "x", "judge": "y"}}',
    ):
        with pytest.raises(al.AgentConfigError):
            al.AgentConfig.from_arg(bad)
    assert al.AgentConfig.from_arg('{"role_models": {"coordinator": "claude-haiku-5-5"}}').role_models == {"coordinator": "claude-haiku-5-5"}


def test_coordinator_loop_only_codes_and_trains_selected(tmp_path):
    h, args, llm = make_harness(tmp_path), make_args(tmp_path, agent_cfg=CFG), CoordLLM()
    al.run_agent_loop(args, h, llm=llm)
    out = Path(args.out_dir)
    assert len(sorted(out.glob("r*_arch_*_meta.json"))) == 6  # same number of TRAINED models as the plain loop
    plan0 = json.loads((out / "agent_logs" / "plan_r00.json").read_text())
    assert len(plan0["items"]) == 4 and plan0["coordinator"]["selected"] == [2, 0]
    skipped = [it for it in plan0["items"] if it["status"] == "skipped"]
    assert len(skipped) == 2 and all(it["coordinator_skipped"] for it in skipped)
    assert all(it["coder_calls"] == 0 for it in skipped)  # unselected ideas never reach the coder
    assert "not selected" not in plan0["report"].split("RESULTS", 1)[1].split("ENSEMBLE")[0]  # not reported as failures
    use = al.summarize_usage(out)
    assert use["by_role"]["coordinator"]["calls"] == llm.n["coordinator"] >= 3
    # the proposer is told a coordinator picks, and was asked for 4 (the over-generation), the coordinator sees the candidates and no test numbers
    assert all("A COORDINATOR agent will choose" in s for s in llm.proposer_systems)
    assert all("Return exactly 4 proposals" in s for s in llm.proposer_systems[:2])
    assert "CANDIDATES this round (4)" in llm.coord_prompts[0] and "[3] IDEA:" in llm.coord_prompts[0]
    assert "test_oracle" not in llm.coord_prompts[0] and "test r" not in llm.coord_prompts[0].lower()  # validation-side numbers only


def test_explore_proposer_style_changes_only_the_proposer_prompt(tmp_path):
    cfg = '{"proposer_style": "explore"}'
    h, args, llm = make_harness(tmp_path), make_args(tmp_path, agent_cfg=cfg, rounds=1), CoordLLM()
    al.run_agent_loop(args, h, llm=llm)
    assert llm.proposer_systems and all("EXPLORATION STYLE" in s for s in llm.proposer_systems)
    h2, args2, llm2 = make_harness(tmp_path / "b"), make_args(tmp_path / "b", rounds=1), CoordLLM()
    (tmp_path / "b").mkdir()
    al.run_agent_loop(args2, h2, llm=llm2)
    assert all("EXPLORATION STYLE" not in s for s in llm2.proposer_systems)
