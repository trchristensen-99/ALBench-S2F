"""Shared helpers for the package-A tests (data, evaluator, gatekeeper): synthetic pools, manifests, tiny candidate models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ar2.data.manifest import make_manifest, save_manifest  # noqa: E402

TINY_CNN = """
import torch
import torch.nn as nn

HPARAMS = {"width": 16}
TUNABLE = {"width": [8, 32]}


class Net(nn.Module):
    def __init__(self, out_dim, width):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(4, width, 9, padding=4), nn.BatchNorm1d(width), nn.ReLU(),
                                  nn.Conv1d(width, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU())
        self.head = nn.Linear(width, out_dim)

    def forward(self, x):
        return self.head(self.conv(x).mean(-1))


def build_model(out_dim, **hp):
    return Net(out_dim, **hp)
"""

CARD = {
    "family_claimed": "cnn_plain",
    "summary": "tiny test model",
    "hypothesis": "none",
    "what_changed": "none",
    "parent_code_hash": None,
    "suggested_recipe": {"lr": 0.002},
    "recipe_rationale": "none",
    "expected_failure": "none",
    "amp_safe": True,
    "length_agnostic": True,
}


def write_candidate(root: Path, name: str, source: str, card: dict | None = None) -> Path:
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.py").write_text(source)
    (d / "card.json").write_text(json.dumps(card or CARD))
    return d


def synth_data(n_train=400, n_val=200, seed=0, scale=1.0, shift=0.0):
    """Sequences with a planted motif (GATAAG anywhere, either strand counts) and labels = count + noise."""
    rng = np.random.default_rng(seed)
    alpha = np.array(list("ACGT"))
    motif, rc = "GATAAG", "CTTATC"

    def make(n):
        seqs, y = [], []
        for _ in range(n):
            s = list("".join(alpha[rng.integers(4, size=200)]))
            k = int(rng.integers(0, 4))
            for _ in range(k):
                p = int(rng.integers(0, 190))
                s[p : p + 6] = list(motif if rng.random() < 0.5 else rc)
            seq = "".join(s)
            seqs.append(seq)
            y.append(seq.count(motif) + seq.count(rc) + rng.normal(0, 0.2))
        return seqs, np.array(y, np.float32) * scale + shift

    tr, ytr = make(n_train)
    va, yva = make(n_val)
    return tr, ytr, va, yva


def write_pool(tmp: Path, seqs, y, role, name, extra: dict | None = None) -> str:
    npz = tmp / f"{name}.npz"
    np.savez(npz, sequences=np.asarray(seqs, dtype=object), oracle_labels=np.asarray(y, np.float32))
    man = make_manifest(str(npz), role, pool_id=name, D=len(seqs), **(extra or {}))
    mp = tmp / f"{name}.manifest.json"
    save_manifest(mp, man)
    return str(mp)


def make_spec(tmp: Path, code_dir: Path, tr_manifest: str, va_manifest: str, **over) -> dict:
    run_id = over.pop("run_id", "r_test")
    spec = {
        "run_id": run_id,
        "code_dir": str(code_dir),
        "recipe": {
            "lr": 0.01,
            "batch_size": 64,
            "max_epochs": 4,
            "patience": 3,
            "weight_decay": 0.0,
            "amp": False,
            "schedule": "onecycle",
            "warmup_frac": 0.1,
        },
        "seed": 0,
        "purpose": "tune",
        "pool_id": "p",
        "D": 400,
        "out_dir": str(tmp / "runs" / run_id),
        "train_manifest": tr_manifest,
        "val_loop_manifest": va_manifest,
        "val_select_manifest": None,
        "time_cap_s": None,
        "speed_factor": 1.0,
        "stress_path": str(tmp / "stress" / "stress_seqs.npy"),
        "save_preds": True,
        "save_ckpt": False,
        "label_noise_sd": 0.0,
        "rc_average": True,
        "device": "cpu",
    }
    recipe_over = over.pop("recipe", {})
    spec.update(over)
    spec["recipe"] = {**spec["recipe"], **recipe_over}
    return spec
