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


def _tuned_harness(tmp_path):
    h = make_harness(tmp_path)

    def sample_train(spec):  # what the launcher's sample_train does in tuned mode
        o, lr, wd, bs, ep, sch = __import__("os").environ["AL_TUNED_RECIPE"].split(",")
        if __import__("os").environ.get("AL_RECIPE_MODE") == "tuned":
            return {
                "optimizer": o,
                "lr": float(lr),
                "weight_decay": float(wd),
                "batch_size": int(bs),
                "epochs": int(ep),
                "lr_schedule": sch,
            }
        return {
            "optimizer": "muon",
            "lr": 2e-3,
            "weight_decay": 1e-4,
            "batch_size": 256,
            "epochs": 80,
            "lr_schedule": "onecycle",
        }

    h.sample_train = sample_train
    return h


def _harness_cfg(manager: str) -> str:
    return json.dumps(
        {
            "stages": ["propose", "coordinate", "code", "evaluate"],
            "ideas_per_round": 4,
            "manager": manager,
            "explore_frac": 0.5,
            "recipe_source": "harness",
        }
    )


def test_tuned_recipe_mode_survives_the_loop_and_is_applied(tmp_path, monkeypatch):
    """Regression: run_agent_loop used to overwrite AL_RECIPE_MODE=tuned with 'harness' (Nov 3 ar40/ia40 trained Muon for 80 epochs)."""
    monkeypatch.setenv("AL_RECIPE_MODE", "tuned")
    monkeypatch.setenv("AL_TUNED_RECIPE", "adamw,0.01,0.2,1024,3,onecycle")
    h, args, llm = (
        _tuned_harness(tmp_path),
        make_args(tmp_path, rounds=2, agent_cfg=_harness_cfg("schedule_ens")),
        CoordLLM(),
    )
    al.run_agent_loop(args, h, llm=llm)
    assert __import__("os").environ["AL_RECIPE_MODE"] == "tuned"
    metas = [
        json.loads(p.read_text()) for p in sorted(Path(args.out_dir).glob("r*_arch_*_meta.json"))
    ]
    assert metas and all(
        m["train_config"]["epochs"] == 3 and m["train_config"]["optimizer"] == "adamw"
        for m in metas
    )
    assert all("TUNED RECIPE" in s.upper() or "tuned" in s.lower() for s in llm.proposer_systems)


def test_tuned_recipe_guard_refuses_a_harness_that_ignores_it(tmp_path, monkeypatch):
    monkeypatch.setenv("AL_RECIPE_MODE", "tuned")
    monkeypatch.setenv("AL_TUNED_RECIPE", "adamw,0.01,0.2,1024,3,onecycle")
    h = make_harness(tmp_path)  # its sample_train knows nothing about the tuned mode
    args = make_args(tmp_path, rounds=1, agent_cfg=_harness_cfg("none"))
    with pytest.raises(al.AgentConfigError):  # tuned mode needs the harness recipe source
        al.run_agent_loop(make_args(tmp_path, rounds=1, agent_cfg=cfg("none")), h, llm=CoordLLM())
    h.sample_train = lambda spec: {
        "optimizer": "muon",
        "lr": 2e-3,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "epochs": 80,
        "lr_schedule": "onecycle",
    }
    with pytest.raises(
        SystemExit
    ) as ex:  # the loop aborts (exit 3) instead of training with the wrong recipe
        al.run_agent_loop(args, h, llm=CoordLLM())
    assert ex.value.code == 3
    assert not list(Path(args.out_dir).glob("r*_arch_*_meta.json"))  # nothing was trained


class LaneLLM(CoordLLM):
    """A proposer that marks lanes and a coordinator that (wrongly) picks two lane-A candidates."""

    def __call__(self, system, user, model=None):
        if "You are the COORDINATOR" in system:
            self.n["coordinator"] += 1
            self.coord_prompts.append(system + "\n" + user)
            return "### SELECT: 1, 2\n### RATIONALE: both from lane A.\n"
        return super(CoordLLM, self).__call__(system, user, model)


def test_lane_helpers():
    items = [
        {"slot": i, "idea": t}
        for i, t in enumerate(["A: x", "B: y", "plain", "plain", "[B] z", "A- w"])
    ]
    assert [al.lane_of(it, 6) for it in items] == [
        "A",
        "B",
        "A",
        "B",
        "B",
        "A",
    ]  # marker first, else position (first half is lane A)
    assert al.enforce_lanes([0, 5], ["A", "B", "A", "B", "B", "A"], 2) == [
        0,
        1,
    ]  # two lane-A picks become one per lane
    assert al.enforce_lanes([3, 1], ["A", "B", "A", "B", "B", "A"], 2) == [
        0,
        3,
    ] or al.enforce_lanes([3, 1], ["A", "B", "A", "B", "B", "A"], 2) == [0, 1]
    assert al.enforce_lanes([2], ["A", "A", "A"], 2) == [
        2,
        0,
    ]  # no lane-B proposal: padded in order


def test_lanes_loop_trains_one_model_per_lane_and_rejects_reused_blocks(tmp_path):
    assert al.AgentConfig.from_arg(cfg("lanes")).manager == "lanes"
    h, args, llm = (
        make_harness(tmp_path),
        make_args(tmp_path, rounds=3, agent_cfg=cfg("lanes", 0.4)),
        LaneLLM(),
    )
    al.run_agent_loop(args, h, llm=llm)
    out = Path(args.out_dir)
    plans = [json.loads((out / "agent_logs" / f"plan_r{r:02d}.json").read_text()) for r in range(3)]
    for p in plans:
        lanes = [it["lane"] for it in p["items"]]
        picked = [it["lane"] for it in p["items"] if not it.get("coordinator_skipped")]
        assert sorted(picked) == ["A", "B"], (
            lanes,
            picked,
        )  # one per lane although the coordinator picked two from lane A
    assert all(
        "MANAGER MODE: LANES" in c and "(lane A)" in c and "(lane B)" in c
        for c in llm.coord_prompts
    )
    assert all("LANE ROUND" in s and "LANE B (independent trunk" in s for s in llm.proposer_systems)
    assert (
        "INCUMBENT" not in llm.coord_prompts[0] and "INCUMBENT" in llm.coord_prompts[-1]
    )  # the last round is a refinement round


def test_fixed_epochs_mode_fixes_only_the_epochs(tmp_path, monkeypatch):
    monkeypatch.setenv("AL_FIXED_EPOCHS", "3")
    monkeypatch.setenv("AL_PATIENCE", "9999")
    monkeypatch.delenv("AL_RECIPE_MODE", raising=False)
    h, args, llm = (
        make_harness(tmp_path),
        make_args(tmp_path, rounds=2, agent_cfg=cfg("lanes", 0.5)),
        LaneLLM(),
    )
    al.run_agent_loop(args, h, llm=llm)
    metas = [
        json.loads(p.read_text()) for p in sorted(Path(args.out_dir).glob("r*_arch_*_meta.json"))
    ]
    assert metas and all(
        m["train_config"]["epochs"] == 3 for m in metas
    )  # the harness's epochs, whatever the proposer wrote
    assert any(
        m["train_config"]["optimizer"] for m in metas
    )  # the rest of the recipe is still the proposer's
    assert all("exactly 3 epochs" in sx for sx in llm.proposer_systems)
    assert all("same long schedule" in sx for sx in llm.proposer_systems)


def test_fixed_epochs_rejects_a_harness_recipe(tmp_path, monkeypatch):
    monkeypatch.setenv("AL_FIXED_EPOCHS", "300")
    h = make_harness(tmp_path)
    with pytest.raises(al.AgentConfigError):
        al.run_agent_loop(
            make_args(tmp_path, rounds=1, agent_cfg=_harness_cfg("none")), h, llm=CoordLLM()
        )


def test_fixed_epochs_requires_patience_off(tmp_path, monkeypatch):
    """Regression: with patience 16 a proposer-chosen cosine schedule stopped at 69-105 of 300 epochs (AL_ONECYCLE_FULL only protects one-cycle)."""
    monkeypatch.setenv("AL_FIXED_EPOCHS", "300")
    monkeypatch.setenv("AL_PATIENCE", "16")
    monkeypatch.delenv("AL_RECIPE_MODE", raising=False)
    with pytest.raises(al.AgentConfigError):
        al.run_agent_loop(
            make_args(tmp_path, rounds=1, agent_cfg=cfg("none")),
            make_harness(tmp_path),
            llm=CoordLLM(),
        )


def test_proposer_ensemble_signal_and_extra_references_are_opt_in(tmp_path, monkeypatch):
    """Both new prompt sections must be off unless their env var is set, and must render when it is."""
    import json

    from albench.arch_search import agent_loop as al

    refs = [
        {"name": "C219 (collaborator)", "val_r": 0.9083, "source": "class Net(nn.Module): pass"}
    ]
    f = tmp_path / "refs.json"
    f.write_text(json.dumps(refs))

    monkeypatch.delenv("AL_EXTRA_REFERENCES", raising=False)
    monkeypatch.setenv("AL_EXTRA_REFERENCES", str(f))
    txt = al.extra_references_text()
    assert "C219 (collaborator)" in txt and "0.9083" in txt and "class Net" in txt

    monkeypatch.setenv("AL_EXTRA_REFERENCES", str(tmp_path / "missing.json"))
    assert "unavailable" in al.extra_references_text()  # a bad path must not crash a run

    assert "uncorrelated" in al.PROPOSER_ENSEMBLE_NOTE
