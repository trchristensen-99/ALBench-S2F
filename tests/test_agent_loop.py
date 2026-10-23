"""Behavioural tests for the multi-agent AutoResearch loop, using a scripted fake LLM.

The fake LLM is keyed on the role (from the system prompt) and on markers in the idea text, so the
tests exercise the real prompt builders, parsers, validator, repair path, training and screens on
a tiny CPU dataset without spending any LLM budget.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from albench.arch_search import agent_loop as al  # noqa: E402
from albench.arch_search import arch_autoresearch as aa  # noqa: E402

GOOD = """\
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Sequential(nn.Conv1d(4, 8, 5, padding=2), nn.BatchNorm1d(8), nn.SiLU())
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        return self.fc(self.c(x).mean(-1)).squeeze(-1)
"""
BAD = GOOD.replace("nn.Linear(8, 1)", "nn.Linear(9, 1)")  # fails the forward probe


class RateLimitExceeded(Exception):  # matched by type name, like the real one
    pass


class FakeLLM:
    """Scripted agents. Markers in an idea: 'bad1' = coder fails once then repairs, 'never' = always fails."""

    def __init__(self, break_at_coder_call: int | None = None, marks: dict | None = None):
        self.marks = {1: " bad1", 2: " never"} if marks is None else marks
        self.n = {"proposer": 0, "coder": 0, "reflector": 0}
        self.break_at = break_at_coder_call
        self.prompts: list[str] = []

    def __call__(self, system: str, user: str, model=None) -> str:
        self.prompts.append(system + "\n" + user)
        if "PROPOSER" in system:
            self.n["proposer"] += 1
            k = self.n["proposer"]
            n = int(re.search(r"Return exactly (\d+) proposals", system).group(1))
            mark = self.marks.get(k, "")
            out = []
            for i in range(n):
                out.append(
                    f"### IDEA: idea{k}_{i}{mark if i == 0 else ''}\n### WHY: because {k}\n"
                    f"### DESIGN: a small conv net, variant {k}_{i}.\n"
                    '### TRAIN: {"optimizer":"adamw","lr":0.003,"batch_size":16,"epochs":2,'
                    '"lr_schedule":"cosine"}\n'
                )
            return "\n".join(out)
        if "CODING agent" in system:
            self.n["coder"] += 1
            if self.break_at is not None and self.n["coder"] == self.break_at:
                raise RateLimitExceeded("session limit")
            idea = re.search(r"IDEA: (.*)", user).group(1)
            repair = "REJECTED by the validator" in user
            if "never" in idea or ("bad1" in idea and not repair):
                return f"```python\n{BAD}\n```"
            return f"```python\n{GOOD}\n```"
        self.n["reflector"] += 1
        return f"- [WORKS] note after call {self.n['reflector']}: conv nets train fine (val r 0.1)."


def make_harness(tmp: Path):
    rng = np.random.default_rng(0)

    def oh(n):
        idx = rng.integers(0, 4, size=(n, 200))
        x = np.zeros((n, 4, 200), np.float32)
        for c in range(4):
            x[:, c, :] = idx == c
        return x

    Xtr, Xva, Xte = oh(64), oh(32), oh(16)
    ytr = (Xtr[:, 0, :].mean(-1) * 5 + rng.normal(size=64) * 0.1).astype(np.float32)
    yva = (Xva[:, 0, :].mean(-1) * 5 + rng.normal(size=32) * 0.1).astype(np.float32)
    te_oracle = rng.normal(size=16).astype(np.float32)

    def asz(path, **kw):
        tmpf = Path(str(path) + ".tmp.npz")
        np.savez(tmpf, **kw)
        tmpf.replace(path)

    def awt(path, text):
        Path(path).write_text(text)

    train_note = "TRAINNOTE"
    return SimpleNamespace(
        Xtr=Xtr,
        ytr=ytr,
        Xva=Xva,
        yva=yva,
        Xte=Xte,
        Xsets={"x": oh(8)},
        te_oracle=te_oracle,
        gpu="cpu",
        cap_min=1.0,
        train_one=aa.train_one,
        sample_train=aa.sample_train,
        predict_rc=aa.predict_rc,
        pear=aa.pear,
        atomic_savez=asz,
        atomic_write_text=awt,
        guidance_note=lambda rd: "GUIDANCE-BLOCK\n\n" + train_note,
        train_note=train_note,
    )


def make_args(tmp: Path, **kw):
    d = dict(
        out_dir=str(tmp / "run"),
        rounds=3,
        per_round=2,
        seq_len=200,
        D=64,
        max_params=40_000_000,
        budget_hours=1.0,
        resume=False,
        full_control=False,
        style="none",
        agent_cfg="",
    )
    d.update(kw)
    return SimpleNamespace(**d)


def calls_by_role(out: Path) -> dict:
    return {r: v["calls"] for r, v in al.summarize_usage(out)["by_role"].items()}


# ----------------------------------------------------------------------------- parsing


def test_parse_ideas_fields_and_train_json():
    text = (
        "preamble\n### IDEA: A one-liner\n### WHY: it is plausible\n"
        "### DESIGN: stem k=9 x 64\n- 6 blocks\n```python\nnot code\n```\n"
        '### TRAIN: {"optimizer": "muon", "lr": 0.002}\n'
        "### IDEA: no design here\n### WHY: x\n"
        "### IDEA: Second\n### DESIGN: d2\n### TRAIN: not json\n"
    )
    ideas = al.parse_ideas(text)
    assert [i["idea"] for i in ideas] == [
        "A one-liner",
        "Second",
    ]  # idea without DESIGN dropped
    assert ideas[0]["train_spec"] == {"optimizer": "muon", "lr": 0.002}
    assert "6 blocks" in ideas[0]["design"]
    assert ideas[1]["train_spec"] == {}
    assert len(al.parse_ideas(text, max_n=1)) == 1


def test_extract_code_prefers_last_net_block():
    t = "```python\nx = 1\n```\ntext\n```python\nclass Net:\n    pass\n```\n```\nclass Other: pass\n```"
    assert "class Net" in al.extract_code(t)
    assert al.extract_code("no fences class Net") is None


def test_clean_lessons_caps_and_normalises():
    raw = "Here you go:\n```\n- [WORKS] a\n* [HURTS] b\n1. [TODO] c\n```\n" + "\n".join(
        f"- bullet {i} " + "x" * 80 for i in range(30)
    )
    out = al.clean_lessons(raw, 400)
    assert len(out) <= 400 and out.startswith("- [WORKS] a") and "[TODO] c" in out
    assert al.clean_lessons("no bullets here at all", 400) == ""


def test_config_validation():
    assert al.AgentConfig.from_arg("").stages == al.DEFAULT_STAGES
    assert al.AgentConfig.from_arg('{"stages": "propose,code,evaluate"}').stages == (
        "propose",
        "code",
        "evaluate",
    )
    for bad in (
        '{"nope": 1}',
        '{"stages": ["code", "evaluate"]}',
        '{"stages": ["propose", "code"]}',
        '{"stages": ["propose", "propose", "code", "evaluate"]}',
        '{"stages": ["evaluate", "propose", "code"]}',
        '{"role_models": {"critic": "x"}}',
    ):
        with pytest.raises(al.AgentConfigError):
            al.AgentConfig.from_arg(bad)


def test_freedom_paragraph_is_the_single_loop_one():
    assert "not filling in a hyperparameter schema" in al._FREEDOM


# ----------------------------------------------------------------------------- full loop


def test_full_loop_layout_repairs_and_no_test_leak(tmp_path):
    h, args, llm = make_harness(tmp_path), make_args(tmp_path), FakeLLM()
    al.run_agent_loop(args, h, llm=llm)
    out = Path(args.out_dir)
    metas = sorted(out.glob("r*_arch_*_meta.json"))
    assert len(metas) == 6  # exactly rounds x per_round trained models
    assert (out / "COMPLETE").read_text().startswith("rounds_run=")
    base_keys = {
        "model_id",
        "strategy",
        "round",
        "idea",
        "why",
        "n_params",
        "train_config",
        "optimizer_used",
        "val_pearson",
        "best_val_pearson",
        "test_oracle_pearson",
        "val_mse",
        "epochs_trained",
        "train_time_sec",
        "gpu",
        "source",
    }
    for m in metas:
        meta = json.loads(m.read_text())
        assert base_keys <= set(meta) and meta["strategy"] == "arch_search"
        z = np.load(str(m).replace("_meta.json", ".npz"))
        assert set(z.files) == {"val_pred", "test_pred", "test_pred_x"}
    # coder stage accounting: one idea repaired, one rejected after 2 repairs
    use = al.summarize_usage(out)
    assert use["coder"]["repaired_ok"] == 1 and use["coder"]["rejected_after_repairs"] == 1
    assert use["coder"]["repair_calls"] == 1 + 2
    assert calls_by_role(out) == {
        "proposer": llm.n["proposer"],
        "coder": llm.n["coder"],
        "reflector": llm.n["reflector"],
    }
    # the evaluator report tells the reflector WHY a repair was needed
    plan0 = json.loads((out / "agent_logs" / "plan_r00.json").read_text())
    assert "coder needed 1 repair(s); first validator error: forward failed" in plan0["report"]
    # the rejected idea is fed back to the proposer as a discarded proposal
    assert any("PROPOSALS THAT WERE DISCARDED" in p and "idea2_0 never" in p for p in llm.prompts)
    # lessons written after round 0 reach the round-1 proposer prompt
    prop = [p for p in llm.prompts if "You are the PROPOSER" in p]
    assert "LESSONS FROM EARLIER ROUNDS" not in prop[0] and "note after call 1" in prop[1]
    assert (
        "GUIDANCE-BLOCK" in prop[0] and "TRAINNOTE" not in prop[0]
    )  # guidance kept, single-loop TRAIN note swapped
    # nothing about the test sets reaches any agent: the test r of every model is absent from prompts
    blob = "\n".join(llm.prompts)
    for m in metas:
        r = json.loads(m.read_text())["test_oracle_pearson"]
        assert f"{r:.5f}" not in blob and "test_oracle" not in blob
    # one log file per round, with prompt size, response and token estimate on every call line
    lines = [
        json.loads(x)
        for f in sorted((out / "agent_logs").glob("round_*.jsonl"))
        for x in f.read_text().splitlines()
    ]
    assert lines and all(
        {
            "role",
            "system_chars",
            "user_chars",
            "response",
            "approx_tokens_in",
            "approx_tokens_out",
            "seconds",
        }
        <= set(x)
        for x in lines
    )
    assert (out / "lessons.md").exists() and (out / "prompts" / "prompt_r00.txt").exists()


def test_target_count_is_exact_even_when_last_round_is_short(tmp_path):
    h, args = make_harness(tmp_path), make_args(tmp_path, rounds=2, per_round=3)
    al.run_agent_loop(args, h, llm=FakeLLM())
    assert len(list(Path(args.out_dir).glob("r*_arch_*_meta.json"))) == 6


def test_resume_after_rate_limit_does_not_repay_the_proposer(tmp_path):
    h, args = (
        make_harness(tmp_path),
        make_args(tmp_path, resume=True, rounds=1, per_round=2),
    )
    first = FakeLLM(break_at_coder_call=2, marks={})  # rate-limited on the 2nd idea of round 0
    with pytest.raises(SystemExit) as ei:
        al.run_agent_loop(args, h, llm=first)
    assert ei.value.code == 75 and first.n["proposer"] == 1
    out = Path(args.out_dir)
    plan = json.loads((out / "agent_logs" / "plan_r00.json").read_text())
    assert [i["status"] for i in plan["items"]] == ["coded", "idea"]
    second = FakeLLM(marks={})
    second.n["proposer"] = 10  # a new proposer call would be visible as 11
    al.run_agent_loop(args, h, llm=second)
    assert second.n["proposer"] == 10  # round 0 was finished from the plan, no new proposer call
    assert len(list(out.glob("r*_arch_*_meta.json"))) == 2 and (out / "COMPLETE").exists()
    assert second.n["coder"] == 1  # only the idea that had not been coded yet


def test_stage_order_and_dropping_the_reflector(tmp_path):
    # reflect first: lessons from round r-1 are written at the start of round r
    h = make_harness(tmp_path)
    args = make_args(tmp_path, agent_cfg='{"stages": ["reflect", "propose", "code", "evaluate"]}')
    llm = FakeLLM()
    al.run_agent_loop(args, h, llm=llm)
    n_rounds = len(list((Path(args.out_dir) / "agent_logs").glob("plan_r*.json")))
    assert llm.n["reflector"] == n_rounds - 1
    # no reflector stage: no lessons ever, zero reflector calls
    h2 = make_harness(tmp_path)
    args2 = make_args(
        tmp_path,
        out_dir=str(tmp_path / "run2"),
        agent_cfg='{"stages": ["propose", "code", "evaluate"]}',
    )
    llm2 = FakeLLM()
    al.run_agent_loop(args2, h2, llm=llm2)
    assert llm2.n["reflector"] == 0 and not (Path(args2.out_dir) / "lessons.md").exists()
    assert all("LESSONS FROM EARLIER ROUNDS" not in p for p in llm2.prompts)


def test_report_contains_only_validation_numbers(tmp_path):
    h, args, llm = make_harness(tmp_path), make_args(tmp_path, rounds=1), FakeLLM()
    al.run_agent_loop(args, h, llm=llm)
    plan = json.loads((Path(args.out_dir) / "agent_logs" / "plan_r00.json").read_text())
    assert "val r" in plan["report"] and "test" not in plan["report"].lower()


def test_single_loop_screens_are_still_mirrored():
    """The agent loop copies the baseline's exploded-prediction screens; fail loudly if they drift apart."""
    src_a = Path(aa.__file__).read_text()
    src_b = Path(al.__file__).read_text()
    for needle in (
        "exploded predictions (|pred|>1e3)",
        "exploded predictions on",
        "torch.manual_seed(1000 + ",
    ):
        assert needle in src_a and needle in src_b, needle


def test_parallel_coder_workers_and_role_models(tmp_path):
    seen: list = []

    class Spy(FakeLLM):
        def __call__(self, system, user, model=None):
            seen.append(model)
            return super().__call__(system, user, model)

    h = make_harness(tmp_path)
    args = make_args(
        tmp_path,
        rounds=2,
        per_round=2,
        agent_cfg='{"coder_workers": 2, "role_models": {"coder": "coder-model-x"}}',
    )
    llm = Spy(marks={})
    al.run_agent_loop(args, h, llm=llm)
    assert len(list(Path(args.out_dir).glob("r*_arch_*_meta.json"))) == 4
    assert seen.count("coder-model-x") == llm.n["coder"] and llm.n["coder"] == 4
    assert all(m is None for m in seen if m != "coder-model-x")  # other roles use LLM_MODEL
