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


def test_variants_manager_restricts_every_round_to_modifications(tmp_path):
    assert al.AgentConfig.from_arg(cfg("variants")).manager == "variants"
    h, args, llm = (
        make_harness(tmp_path),
        make_args(tmp_path, rounds=3, agent_cfg=cfg("variants")),
        CoordLLM(),
    )
    al.run_agent_loop(args, h, llm=llm)
    out = Path(args.out_dir)
    plans = [json.loads((out / "agent_logs" / f"plan_r{r:02d}.json").read_text()) for r in range(3)]
    assert [p["coordinator"].get("manager_mode") for p in plans] == ["variants"] * 3
    assert all("MANAGER MODE: VARIANTS" in p for p in llm.coord_prompts)
    assert all(
        "VARIANT ROUND" in s and "Do NOT propose a different mechanism" in s
        for s in llm.proposer_systems
    )
    assert "INCUMBENT: none yet" in llm.coord_prompts[0]
    assert "INCUMBENT: the best variant so far reaches validation r" in llm.coord_prompts[-1]
    assert all("EXPLORATION ROUND" not in s for s in llm.proposer_systems)


def test_training_facts_only_for_the_fixed_harness(tmp_path, monkeypatch):
    monkeypatch.delenv("AL_ONECYCLE_FULL", raising=False)
    assert al.training_facts_text() == ""
    monkeypatch.setenv("AL_ONECYCLE_FULL", "1")
    monkeypatch.setenv("AL_PATIENCE", "16")
    txt = al.training_facts_text()
    assert "TRAINING FACTS" in txt and "16 epochs without improvement" in txt
    h, args, llm = (
        make_harness(tmp_path),
        make_args(tmp_path, rounds=1, agent_cfg=cfg("none")),
        CoordLLM(),
    )
    al.run_agent_loop(
        args, h, llm=llm
    )  # the loop still runs with the paragraph appended to the setup
    assert llm.proposer_systems


def test_tuned_recipe_text_only_with_the_tuned_mode(monkeypatch):
    monkeypatch.delenv("AL_TUNED_RECIPE", raising=False)
    assert al.tuned_recipe_text() == ""
    monkeypatch.setenv("AL_TUNED_RECIPE", "adamw,0.01,0.2,1024,300,onecycle")
    txt = al.tuned_recipe_text()
    assert "RECIPE:" in txt and "300 epochs" in txt and "batch size 1024" in txt


def test_islands_manager_config_and_modes():
    c = al.AgentConfig.from_arg(cfg("islands", 0.4))
    assert c.manager == "islands"

    class C:  # minimal ctx stand-in
        pass

    x = C()
    x.cfg, x.args = c, make_args(Path("."), rounds=15)
    assert [al.manager_mode(x, r) for r in range(15)] == ["explore"] * 6 + ["island"] * 9


def test_island_of_separates_mechanism_and_trunk():
    legnet = "class EffBlock(nn.Module): pass\nclass Net(nn.Module): pass"
    gru = legnet + "\nself.rnn = nn.GRU(64, 64, bidirectional=True)"
    hand_attn = "class Net(nn.Module):\n  def forward(self, x):\n    a = torch.softmax(q @ k.transpose(-1, -2), -1)"
    unet = "class Net(nn.Module):\n  def __init__(self):\n    self.up = nn.ConvTranspose1d(8, 8, 2)"
    assert al.island_of(legnet) == "convolutional / LegNet trunk"
    assert al.island_of(gru) == "recurrent / LegNet trunk"
    assert al.island_of(hand_attn) == "attention / other trunk"  # hand-written attention counts
    assert al.island_of(unet) == "U-Net / other trunk"
    assert al.island_of("class Net(nn.Module): pass", "A LegNet with a wider middle").endswith(
        "LegNet trunk"
    )  # named in the idea but written without EffBlock


def test_island_for_alternates_the_main_island_and_other_families(tmp_path):
    def rec(i, val, src, idea):
        (tmp_path / f"r00_arch_{i:02d}_meta.json").write_text(
            json.dumps({"source": src, "idea": idea, "agent": {"design": f"design {i}"}})
        )
        return al.ModelRecord(f"r00_arch_{i:02d}", 0, idea, val, 1000, 1.0, 10, "adamw")

    class C:
        pass

    x = C()
    x.out_p, x.cfg, x.args = (
        tmp_path,
        al.AgentConfig.from_arg(cfg("islands", 0.4)),
        make_args(Path("."), rounds=15),
    )
    leg = "class EffBlock(nn.Module): pass"
    x.records = [
        rec(0, 0.910, leg, "LegNet wide"),
        rec(1, 0.905, leg + "\nnn.GRU(8, 8)", "LegNet + BiGRU"),
        rec(2, 0.860, "nn.ConvTranspose1d(8, 8, 2)", "U-Net"),
        rec(3, 0.840, "class Net(nn.Module): pass", "ConvNeXt tower"),
    ]
    chosen = []
    for rd in range(6, 15):
        plan = {"round": rd, "items": []}
        (tmp_path / "agent_logs").mkdir(exist_ok=True)
        chosen.append(al.island_for(x, plan, rd)["island"])
        assert al.island_for(x, plan, rd)["island"] == chosen[-1]  # fixed in the plan once decided
    main = "convolutional / LegNet trunk"
    assert chosen[0::2] == [main] * 5  # even exploit rounds: the best model overall
    assert chosen[1::2] == [  # odd ones: the other-trunk islands first, best first, in turn
        "U-Net / other trunk",
        "convolutional / other trunk",
        "recurrent / LegNet trunk",
        "U-Net / other trunk",
    ]
    text = al.island_status_text(x, al.island_for(x, {"round": 7, "items": []}, 7))
    assert "below the best model overall" in text and "U-Net / other trunk" in text
    assert al.island_status_text(x, None).startswith("ISLAND: none yet")


def test_islands_loop_gives_island_prompts_after_exploration(tmp_path):
    h, args, llm = (
        make_harness(tmp_path),
        make_args(tmp_path, rounds=4, agent_cfg=cfg("islands", 0.5)),
        CoordLLM(),
    )
    al.run_agent_loop(args, h, llm=llm)
    out = Path(args.out_dir)
    plans = [json.loads((out / "agent_logs" / f"plan_r{r:02d}.json").read_text()) for r in range(4)]
    assert [p["coordinator"].get("manager_mode") for p in plans] == [
        "explore",
        "explore",
        "island",
        "island",
    ]
    assert (
        "MANAGER MODE: EXPLORE" in llm.coord_prompts[0]
        and "ENSEMBLE STATUS" in llm.coord_prompts[-1]
    )
    assert (
        "MANAGER MODE: ISLAND" in llm.coord_prompts[-1]
        and "ISLAND ROUND" in llm.proposer_systems[-1]
    )
    assert "island" in plans[2] and "island" not in plans[0]
