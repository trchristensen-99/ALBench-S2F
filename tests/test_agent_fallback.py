"""The reference-checked explore-then-exploit manager ('schedule_fb')."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from test_agent_coordinator import CoordLLM  # noqa: E402
from test_agent_loop import make_args, make_harness  # noqa: E402

from albench.arch_search import agent_loop as al  # noqa: E402


def cfg(**extra) -> str:
    return json.dumps(
        {"stages": ["propose", "coordinate", "code", "evaluate"], "ideas_per_round": 4, **extra}
    )


def _ctx(
    tmp_path: Path,
    vals,
    ref_text="BASELINE: the published MPRA-LegNet reaches validation Pearson r = 0.700.",
):
    f = tmp_path / ("baseline_legnet.txt" if ref_text is not None else "missing.txt")
    if ref_text is not None:
        f.write_text(ref_text)
    return SimpleNamespace(
        records=[SimpleNamespace(val=v) for v in vals], args=SimpleNamespace(baseline_file=str(f))
    )


def test_config_accepts_the_mode_and_still_rejects_unknown_ones():
    assert al.AgentConfig.from_arg(cfg(manager="schedule_fb")).manager == "schedule_fb"
    with pytest.raises(al.AgentConfigError):
        al.AgentConfig.from_arg(cfg(manager="schedule_xx"))
    with pytest.raises(al.AgentConfigError):  # needs the coordinate stage like every manager
        al.AgentConfig.from_arg(
            '{"stages": ["propose", "code", "evaluate"], "manager": "schedule_fb"}'
        )


def test_reference_check_names_the_incumbent_from_measured_numbers(tmp_path):
    win = al.reference_check_text(_ctx(tmp_path, [0.71, 0.74]))
    assert "beats the reference by +0.0400" in win and "best explored architecture" in win
    lose = al.reference_check_text(_ctx(tmp_path, [0.69, 0.703]))  # inside the 0.005 margin
    assert "No explored architecture beats the reference" in lose and "modify it slightly" in lose
    assert "0.7000" in lose and "0.7030" in lose
    neutral = al.reference_check_text(_ctx(tmp_path, [0.71], ref_text=None))
    assert neutral.startswith("INCUMBENT: the best model so far")
    assert al.reference_check_text(_ctx(tmp_path, [])).startswith("INCUMBENT: the best model")


def test_explore_rounds_unchanged_and_exploit_rounds_carry_the_check(tmp_path, monkeypatch):
    monkeypatch.delenv("AL_RECIPE_MODE", raising=False)
    base = tmp_path / "ref.txt"
    base.write_text("BASELINE: reaches validation Pearson r = 0.100.")  # every fake model beats it
    h = make_harness(tmp_path)
    args = make_args(
        tmp_path,
        rounds=4,
        agent_cfg=cfg(manager="schedule_fb", explore_frac=0.5),
        baseline_file=str(base),
    )
    llm = CoordLLM()
    al.run_agent_loop(args, h, llm=llm)
    first, last = llm.coord_prompts[0], llm.coord_prompts[-1]
    assert "MANAGER MODE: EXPLORE" in first and "reference check" not in first
    assert "MANAGER MODE: EXPLOIT with a reference check" in last and "REFERENCE CHECK" in last
    assert "EXPLORATION ROUND" in llm.proposer_systems[0]
    assert (
        "REFINEMENT ROUND" in llm.proposer_systems[-1] and "INCUMBENT" in llm.proposer_systems[-1]
    )
    # the plain schedule manager keeps its old wording
    assert (
        "reference check" not in al.MANAGER_EXPLOIT
        and "INCUMBENT" not in al.PROPOSER_MANAGER_EXPLOIT
    )
