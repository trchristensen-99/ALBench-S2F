"""Tests for the quota manager and harness-assigned recipes of the multi-agent loop (scripted fake LLM, tiny CPU data)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from test_agent_coordinator import CoordLLM  # noqa: E402
from test_agent_loop import make_args, make_harness  # noqa: E402

from albench.arch_search import agent_loop as al  # noqa: E402


def cfg(**extra) -> str:
    return json.dumps(
        {"stages": ["propose", "coordinate", "code", "evaluate"], "ideas_per_round": 4, **extra}
    )


def test_config_validation_and_defaults():
    assert (
        al.AgentConfig.from_arg(cfg(manager="quota", recipe_source="harness")).recipe_source
        == "harness"
    )
    assert al.AgentConfig.from_arg("").recipe_source == "proposal"  # earlier arms unchanged
    with pytest.raises(al.AgentConfigError):
        al.AgentConfig.from_arg(cfg(recipe_source="random"))
    with pytest.raises(al.AgentConfigError):  # the quota is applied by the coordinator
        al.AgentConfig.from_arg('{"stages": ["propose", "code", "evaluate"], "manager": "quota"}')


def test_quota_mode_every_round():
    class C:
        pass

    c = C()
    c.cfg = al.AgentConfig.from_arg(cfg(manager="quota"))
    c.args = make_args(Path("."), rounds=10)
    assert {al.manager_mode(c, r) for r in range(10)} == {"quota"}


def test_quota_and_harness_recipe_prompts(tmp_path, monkeypatch):
    monkeypatch.delenv("AL_RECIPE_MODE", raising=False)
    h, args, llm = (
        make_harness(tmp_path),
        make_args(tmp_path, rounds=3, agent_cfg=cfg(manager="quota", recipe_source="harness")),
        CoordLLM(),
    )
    al.run_agent_loop(args, h, llm=llm)
    assert all("MANAGER MODE: QUOTA" in p and "COVERAGE of the" in p for p in llm.coord_prompts)
    assert all("NOT YET TRIED" in p for p in llm.coord_prompts[1:])
    assert all("TRAIN:" not in p for p in llm.coord_prompts)  # recipes are not shown or proposed
    assert all("### TRAIN" not in s and "QUOTA ROUND" in s for s in llm.proposer_systems)
    import os

    assert os.environ["AL_RECIPE_MODE"] == "harness"
    meta = json.loads(sorted(Path(args.out_dir).glob("r*_arch_*_meta.json"))[0].read_text())
    assert meta["agent"]["recipe_source"] == "harness" and meta["agent"]["manager"] == "quota"


def test_default_prompts_keep_the_train_line(tmp_path):
    h, args, llm = (
        make_harness(tmp_path),
        make_args(tmp_path, rounds=2, agent_cfg=cfg()),
        CoordLLM(),
    )
    al.run_agent_loop(args, h, llm=llm)
    assert all("### TRAIN" in s for s in llm.proposer_systems)
    assert all("QUOTA" not in p for p in llm.coord_prompts)


def test_feature_coverage_counts_source(tmp_path):
    class R:
        def __init__(self, i):
            self.model_id, self.recipe = f"r00_arch_{i:02d}", "x"

    class C:
        pass

    c = C()
    c.cfg = al.AgentConfig.from_arg(cfg(manager="quota", recipe_source="harness"))
    c.out_p = tmp_path
    c.records = [R(0), R(1)]
    (tmp_path / "r00_arch_00_meta.json").write_text(
        json.dumps({"source": "x = nn.GRU(4, 8)\ny = torch.cat([a, b], 1)"})
    )
    (tmp_path / "r00_arch_01_meta.json").write_text(
        json.dumps({"source": "nn.MultiheadAttention(8, 2)"})
    )
    t = al.coverage_text_quota(c)
    assert (
        "recurrent: 1" in t and "self-attention: 1" in t and "dense / concatenation skips: 1" in t
    )
    assert "NOT YET TRIED" in t and "state-space" in t and "training recipes used" not in t


class UserSpy(CoordLLM):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.proposer_users: list[str] = []

    def __call__(self, system, user, model=None):
        if "You are the PROPOSER" in system:
            self.proposer_users.append(user)
        return super().__call__(system, user, model)


@pytest.mark.parametrize("recipe_source,kept", [("harness", False), ("proposal", True)])
def test_training_note_is_dropped_only_for_harness_recipes(
    tmp_path, monkeypatch, recipe_source, kept
):
    """The real launcher puts the input-context paragraph after the training note; a suffix test on the note missed it."""
    monkeypatch.delenv("AL_RECIPE_MODE", raising=False)
    h = make_harness(tmp_path)
    h.guidance_note = lambda rd: "GUIDANCE-BLOCK\n\n" + h.train_note + "\n\nINPUT CONTEXT: L = 240"
    args = make_args(
        tmp_path, rounds=1, agent_cfg=cfg(manager="quota", recipe_source=recipe_source)
    )
    llm = UserSpy()
    al.run_agent_loop(args, h, llm=llm)
    assert llm.proposer_users
    for u in llm.proposer_users:
        assert "GUIDANCE-BLOCK" in u and "INPUT CONTEXT: L = 240" in u
        assert ("TRAINNOTE" in u) is kept
