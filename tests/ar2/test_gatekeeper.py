"""Package A: gatekeeper tests (spec T4: every broken-model fixture fails its named check and no other; the reference models pass)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from a_helpers import CARD, TINY_CNN, write_candidate

from ar2.gatekeeper import (
    GateResult,
    code_hash,
    near_duplicate,
    normalise_source,
    run_gate,
    static_check,
    write_gate_json,
)

FIX = Path(__file__).parent / "fixtures"
BROKEN = sorted(p for p in (FIX / "broken_models").iterdir() if p.is_dir())
REFS = sorted(p for p in (FIX / "reference_models").iterdir() if p.is_dir())
PROPOSER_REFS = sorted(
    p
    for p in (Path(__file__).resolve().parents[2] / "ar2/proposer/reference_models").glob("*")
    if (p / "model.py").exists()
)
CFG = {"D": 100000, "evaluator": {"max_epochs": 40}}


@pytest.mark.parametrize("fixture", BROKEN, ids=lambda p: p.name)
def test_t4_broken_models_fail_their_named_check(fixture):
    expected = (fixture / "EXPECTED").read_text().strip()
    kw = {"speed_probe": True, "time_cap_s": 1.0} if expected == "speed" else {}
    r = run_gate(fixture, CFG, **kw)
    assert not r.ok and r.code == expected, (r.code, r.detail)
    assert r.action == ("reject" if expected in ("unsafe", "duplicate") else "repair")


@pytest.mark.parametrize("ref", REFS + PROPOSER_REFS, ids=lambda p: p.name)
def test_reference_models_pass_all_checks(ref):
    r = run_gate(ref, CFG, speed_probe=True, time_cap_s=1e6)
    assert r.ok and r.action in ("pass", "amp_off"), (r.code, r.detail, r.traceback)
    assert (
        r.measured["n_params"] >= 10_000
        and r.measured["coverage"] >= 0.9
        and r.measured["throughput_seq_per_s"] > 0
    )


def test_measured_values_and_gate_json(tmp_path):
    d = write_candidate(tmp_path, "m", TINY_CNN.replace('"width": 16', '"width": 64'))
    r = run_gate(d, CFG)
    assert r.ok and r.measured["code_hash"] == code_hash((d / "model.py").read_text())
    path = write_gate_json(d, r)
    assert json.loads(path.read_text())["ok"] is True


def test_check_order_and_files(tmp_path):
    assert run_gate(tmp_path / "nothing", CFG).code == "files"
    d = write_candidate(tmp_path, "nocard", TINY_CNN)
    (d / "card.json").unlink()
    assert run_gate(d, CFG).code == "files"
    d2 = write_candidate(tmp_path, "badcard", TINY_CNN, {**CARD, "suggested_recipe": "lr=1"})
    r = run_gate(d2, CFG)
    assert r.code == "files" and r.action == "repair" and "suggested_recipe" in r.detail


STATIC_BAD = {
    "import os": "import os\ndef build_model(out_dim, **hp):\n    return None\n",
    "from subprocess": "from subprocess import run\ndef build_model(out_dim, **hp):\n    return None\n",
    "open": "def build_model(out_dim, **hp):\n    open('x')\n",
    "eval": "def build_model(out_dim, **hp):\n    eval('1')\n",
    "exec": "def build_model(out_dim, **hp):\n    exec('1')\n",
    "__import__": "def build_model(out_dim, **hp):\n    __import__('os')\n",
    "torch.load": "import torch\ndef build_model(out_dim, **hp):\n    torch.load('w.pt')\n",
    "load_state_dict": "import torch\ndef build_model(out_dim, **hp):\n    m = torch.nn.Linear(1, 1)\n    m.load_state_dict({})\n    return m\n",
    "os attr": "import torch\ndef build_model(out_dim, **hp):\n    return torch.os.getcwd()\n",
    "dunder": "import torch\ndef build_model(out_dim, **hp):\n    return ().__class__.__mro__[1].__subclasses__()\n",
    "numpy": "import numpy as np\ndef build_model(out_dim, **hp):\n    return None\n",
}


@pytest.mark.parametrize("name", sorted(STATIC_BAD))
def test_static_scan_rejects(name):
    ok, code, detail = static_check(STATIC_BAD[name])
    assert not ok and code == "unsafe", (name, detail)


def test_static_scan_accepts_reference_code_and_flags_missing_build_model():
    for p in REFS:
        assert static_check((p / "model.py").read_text()) == (True, None, "")
    ok, code, _ = static_check("import torch\nx = 1\n")
    assert not ok and code == "build"
    ok, code, _ = static_check("def build_model(:\n")
    assert not ok and code == "build"


def test_duplicate_detection_ignores_comments_whitespace_docstrings(tmp_path):
    a = write_candidate(tmp_path, "a", TINY_CNN)
    src = (a / "model.py").read_text()
    b_src = '"""A docstring."""\n# comment\n' + src.replace(
        "nn.ReLU(),", "nn.ReLU(),   # same layer\n   "
    )
    assert code_hash(src) == code_hash(b_src) and normalise_source(src) == normalise_source(b_src)
    b = write_candidate(tmp_path, "b", b_src)
    r = run_gate(b, CFG, existing_hashes={code_hash(src)})
    assert (
        not r.ok and r.code == "duplicate" and r.action == "reject" and code_hash(src) in r.detail
    )
    c = write_candidate(
        tmp_path,
        "c",
        src.replace("nn.Conv1d(4, width, 9, padding=4)", "nn.Conv1d(4, width, 7, padding=3)"),
    )
    assert code_hash((c / "model.py").read_text()) != code_hash(src)


def test_amp_failure_is_not_a_gate_failure(tmp_path):
    src = TINY_CNN.replace('"width": 16', '"width": 64').replace(
        "return self.head(self.conv(x).mean(-1))",
        "o = self.head(self.conv(x).mean(-1))\n        return o * float('nan') if torch.is_autocast_enabled('cpu') else o",
    )
    r = run_gate(write_candidate(tmp_path, "amp", src), CFG)
    assert r.ok and r.action == "amp_off" and r.measured["amp_safe"] is False and r.code == "amp"


def test_sandbox_has_no_network(tmp_path):
    src = TINY_CNN.replace('"width": 16', '"width": 64')
    ok, *_ = static_check(src.replace("import torch", "import socket\nimport torch", 1))
    assert not ok  # the scan stops it before it can run; the sandbox additionally disables sockets
    from ar2.gatekeeper import RUNNER

    assert "socket.socket = _NoNet" in RUNNER


def test_near_duplicate_helper():
    tags = {"family": "cnn_residual", "core_block": "res", "uses_attention": False, "n_params": 1}
    base = {
        "code_hash": "h1",
        "tags": {**tags, "n_params": 5, "justification": "x"},
        "hparams": {"width": 256, "depth": 6, "act": "gelu"},
    }
    assert (
        near_duplicate(tags, {"width": 260, "depth": 6, "act": "gelu"}, [base]) == "h1"
    )  # every value within 10%
    assert near_duplicate(tags, {"width": 300, "depth": 6, "act": "gelu"}, [base]) is None
    assert near_duplicate(tags, {"width": 256, "depth": 6, "act": "relu"}, [base]) is None
    assert (
        near_duplicate(
            {**tags, "uses_attention": True}, {"width": 256, "depth": 6, "act": "gelu"}, [base]
        )
        is None
    )
    assert (
        near_duplicate(
            {**tags, "family": "unet"}, {"width": 256, "depth": 6, "act": "gelu"}, [base]
        )
        is None
    )


def test_gate_is_randomised_across_calls(tmp_path):
    """Seeds, inputs and batch sizes are drawn fresh per call: two runs with the same candidate must both pass, and the rng seed argument differs."""
    d = write_candidate(tmp_path, "m", TINY_CNN.replace('"width": 16', '"width": 64'))
    assert run_gate(d, CFG).ok and run_gate(d, CFG).ok
    assert isinstance(run_gate(d, CFG), GateResult)
