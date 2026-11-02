"""The Claude CLI transport of the agent loop: isolation from the repository, and what counts as a usage limit (audit of Nov 2, F1 and F7)."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

llm = pytest.importorskip("experiments.llm_autoresearch")


def fake_run(captured: dict, stdout: str = "", stderr: str = "", rc: int = 0):
    def run(cmd, **kw):
        captured["cmd"], captured["kw"] = cmd, kw
        return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=rc)

    return run


def test_call_is_tool_free_and_runs_outside_the_repository(monkeypatch):
    cap: dict = {}
    monkeypatch.setattr(subprocess, "run", fake_run(cap, stdout="### SELECT: 1, 2\n"))
    monkeypatch.delenv("LLM_CLI_ISOLATE", raising=False)
    llm._call_claude_cli("sys", "user", "claude-sonnet-5-5")
    assert ["--tools", ""] == cap["cmd"][
        cap["cmd"].index("--tools") : cap["cmd"].index("--tools") + 2
    ]
    assert cap["cmd"][cap["cmd"].index("--system-prompt") + 1] == "sys" and cap["cmd"][2] == "user"
    assert cap["cmd"][cap["cmd"].index("--effort") + 1] == "medium"
    assert cap["kw"]["cwd"] and "llm_cli_" in cap["kw"]["cwd"]


def test_effort_can_be_changed_or_left_at_the_default(monkeypatch):
    cap: dict = {}
    monkeypatch.setattr(subprocess, "run", fake_run(cap, stdout="### SELECT: 1\n"))
    monkeypatch.setenv("LLM_CLI_EFFORT", "high")
    llm._call_claude_cli("sys", "user", "sonnet")
    assert cap["cmd"][cap["cmd"].index("--effort") + 1] == "high"
    monkeypatch.setenv("LLM_CLI_EFFORT", "default")
    llm._call_claude_cli("sys", "user", "sonnet")
    assert "--effort" not in cap["cmd"]


def test_isolation_can_be_switched_off(monkeypatch):
    cap: dict = {}
    monkeypatch.setattr(subprocess, "run", fake_run(cap, stdout="### SELECT: 1\n"))
    monkeypatch.setenv("LLM_CLI_ISOLATE", "0")
    llm._call_claude_cli("sys", "user", "sonnet")
    assert "--tools" not in cap["cmd"] and cap["kw"]["cwd"] is None
    assert cap["cmd"][2].startswith("<system>\nsys\n</system>")


def test_a_real_answer_that_mentions_quota_is_not_a_usage_limit(monkeypatch):
    answer = (
        "### SELECT: 1, 3\n### RATIONALE:\n1. Keeps the quota of exploit models; the recipe resets the schedule; "
        "error 429 in the log is unrelated.\n"
    )
    monkeypatch.setattr(subprocess, "run", fake_run({}, stdout=answer))
    assert llm._call_claude_cli("s", "u", "sonnet") == answer


def test_a_short_limit_notice_is_still_a_usage_limit(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", fake_run({}, stdout="You've hit your session limit · resets 5pm")
    )
    with pytest.raises(llm.RateLimitExceeded):
        llm._call_claude_cli("s", "u", "sonnet")


def test_timeouts_retry_once_and_then_fail_without_claiming_a_limit(monkeypatch):
    calls = []

    def run(cmd, **kw):
        calls.append(1)
        raise subprocess.TimeoutExpired(cmd, 300)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError) as e:
        llm._call_claude_cli("s", "u", "sonnet")
    assert not isinstance(e.value, llm.RateLimitExceeded) and len(calls) == 2
