#!/usr/bin/env python
"""Evaluator parity (spec 16.1, milestone 1): train one architecture from an existing menu entry through the ar2 evaluator and print its validation r, so it
can be compared with the same architecture, pool and seed in the existing loop (menu_retrain.py / albench.ensemble.menu).

    python scripts/ar2_parity.py --menu albench/ensemble/menus/menu_v1.json --entry motif_ct_enriched_rep3_r05_arch_10 \
        --pool outputs/grid_pools/k562_reps/genomic_r1_d10000.npz --val outputs/val_variants/val_V0.npz --D 10000 --seed 0 --reference 0.7373

The entry's `source` defines an nn.Module (class Net by convention, built with no arguments, output (B,) or (B, 1)); it is wrapped into a model.py with
build_model(out_dim, **hp) for out_dim 1. The entry's train_config is mapped onto an ar2 recipe (optimizer, lr, weight_decay, batch_size, epochs -> max_epochs,
lr_schedule -> schedule: onecycle -> onecycle, cosine and plateau -> cosine_warmup). Differences from the existing loop that no mapping removes, and which
explain a gap: the existing loop adds random shifts of up to 10 bp (ar2 excludes them, spec 13.4), uses its own one-cycle schedule, and stops on patience 8 or
16 depending on the experiment.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ar2.data.manifest import make_manifest, save_manifest  # noqa: E402
from ar2.evaluator.train import run  # noqa: E402

WRAPPER = '''

HPARAMS = {}


class _Wrap(nn.Module):
    """Adapts the entry's Net() (output (B,) or (B, 1)) to the ar2 contract (B, out_dim) with out_dim 1."""

    def __init__(self):
        super().__init__()
        self.net = %(cls)s()

    def forward(self, x):
        return self.net(x).reshape(x.shape[0], -1)[:, :1]


def build_model(out_dim, **hp):
    assert out_dim == 1, "the parity wrapper supports out_dim 1 only"
    return _Wrap()
'''


def load_entry(menu_path: str, entry_id: str) -> dict:
    data = json.load(open(menu_path))
    entries = data["entries"] if isinstance(data, dict) and "entries" in data else data
    for e in entries:
        if e.get("id") == entry_id or e.get("name") == entry_id:
            return e
    raise SystemExit(
        f"entry {entry_id!r} not found in {menu_path}; ids: {[e.get('id') for e in entries][:8]} ..."
    )


def net_class_name(source: str) -> str:
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    for n in classes:
        if n.name == "Net":
            return "Net"
    if not classes:
        raise SystemExit("the entry's source defines no class")
    return classes[-1].name


def recipe_from_train_config(
    tc: dict, patience: int, epochs: int | None, optimizer: str | None
) -> dict:
    sched = {"onecycle": "onecycle", "cosine": "cosine_warmup", "plateau": "cosine_warmup"}.get(
        str(tc.get("lr_schedule", "onecycle")), "onecycle"
    )
    return {
        "optimizer": optimizer or tc.get("optimizer", "adamw"),
        "lr": float(tc.get("lr", 2e-3)),
        "weight_decay": float(tc.get("weight_decay", 1e-4)),
        "batch_size": int(tc.get("batch_size", 256)),
        "schedule": sched,
        "warmup_frac": 0.05,
        "max_epochs": int(epochs or tc.get("epochs", 40)),
        "patience": patience,
        "grad_clip": 1.0,
        "loss": {"name": "mse"},
        "rc_augment": True,
        "ema": None,
        "amp": True,
        "tasks": ["K562"],
        "hparams": {},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--menu", default=str(ROOT / "albench/ensemble/menus/menu_v1.json"))
    ap.add_argument("--entry", required=True)
    ap.add_argument("--pool", required=True, help="training pool npz (sequences, oracle_labels)")
    ap.add_argument(
        "--val", required=True, help="validation npz (e.g. outputs/val_variants/val_V0.npz)"
    )
    ap.add_argument("--D", type=int, default=None, help="use the first D rows of the pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=None, help="override the entry's epochs")
    ap.add_argument(
        "--optimizer",
        default=None,
        help="override (e.g. adamw when the muon package is not installed)",
    )
    ap.add_argument("--time-cap-s", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--reference",
        type=float,
        default=None,
        help="validation r of the same entry, pool and seed in the existing loop",
    )
    a = ap.parse_args()

    e = load_entry(a.menu, a.entry)
    out = Path(a.out or ROOT / "outputs/ar2_parity" / f"{a.entry}_D{a.D or 'all'}_s{a.seed}")
    code = out / "candidate"
    code.mkdir(parents=True, exist_ok=True)
    src = e["source"]
    (code / "model.py").write_text(
        "import torch\nimport torch.nn as nn\n" + src + WRAPPER % {"cls": net_class_name(src)}
    )
    (code / "card.json").write_text(
        json.dumps(
            {
                "family_claimed": "other:parity",
                "summary": e.get("idea", "")[:200],
                "suggested_recipe": {},
            }
        )
    )
    tm, vm = out / "train.manifest.json", out / "val.manifest.json"
    save_manifest(tm, make_manifest(a.pool, "train", D=a.D, pool_id=Path(a.pool).stem))
    save_manifest(vm, make_manifest(a.val, "val_loop", pool_id=Path(a.val).stem))
    spec = {
        "run_id": f"parity_{a.entry}",
        "code_dir": str(code),
        "recipe": recipe_from_train_config(
            e.get("train_config", {}), a.patience, a.epochs, a.optimizer
        ),
        "seed": a.seed,
        "purpose": "screen",
        "pool_id": Path(a.pool).stem,
        "D": a.D,
        "out_dir": str(out / "run"),
        "train_manifest": str(tm),
        "val_loop_manifest": str(vm),
        "val_select_manifest": None,
        "time_cap_s": a.time_cap_s,
        "speed_factor": 1.0,
        "stress_path": str(out / "stress_seqs.npy"),
        "save_preds": True,
        "save_ckpt": False,
        "rc_average": True,
        "device": a.device,
    }
    m = run(spec)
    print(
        json.dumps(
            {
                k: m.get(k)
                for k in (
                    "status",
                    "stop_reason",
                    "val_r",
                    "train_r",
                    "best_epoch",
                    "epochs_run",
                    "n_params",
                    "wall_time_s",
                    "stress",
                )
            },
            indent=1,
        )
    )
    if a.reference is not None and m.get("val_r") is not None:
        d = m["val_r"] - a.reference
        print(
            f"ar2 val_r {m['val_r']:.4f} vs existing loop {a.reference:.4f}: difference {d:+.4f} ({'within' if abs(d) <= 0.005 else 'outside'} the 0.005 parity tolerance)"
        )


if __name__ == "__main__":
    main()
