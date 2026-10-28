"""A stand-in for ``python -m ar2.evaluator.train --spec`` that needs no torch: deterministic synthetic metrics, written like the real evaluator (curves.csv, then
metrics.json last by temporary name and rename). Used by the orchestrator and job tests (B1). Behaviours keyed on the spec:

* ``recipe.batch_size >= 1024`` -> status oom (the orchestrator must replace the recipe with batch_size halved);
* ``recipe.lr == 0.0123``       -> status error the first time (infrastructure failure), ok on the retry;
* ``recipe.lr == 0.0456``       -> status diverged.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path


def val_r_for(spec: dict) -> float:
    fam = Path(spec["code_dir"]).name
    h = int(hashlib.sha1(f"{fam}|{spec['seed']}".encode()).hexdigest()[:6], 16) / 16**6
    lr = float(spec["recipe"].get("lr", 0.002))
    base = 0.70 + 0.05 * (int(hashlib.sha1(fam.encode()).hexdigest()[:4], 16) / 16**4)
    return round(base - 0.01 * abs(__import__("math").log10(lr / 0.002)) + 0.004 * (h - 0.5), 6)


def run(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text())
    out = Path(spec["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rec = spec["recipe"]
    max_epochs = int(rec.get("max_epochs", 40))
    marker = out / "attempt1"
    status, epochs = "ok", 30
    if int(rec.get("batch_size", 512)) >= 1024:
        status, epochs = "oom", 0
    elif rec.get("lr") == 0.0123 and not marker.exists():
        marker.write_text("x")
        status, epochs = "error", 0
    elif rec.get("lr") == 0.0456:
        status, epochs = "diverged", 3
    v = val_r_for(spec)
    with open(out / "curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "val_r", "train_r", "loss"])
        for e in range(1, epochs + 1):
            w.writerow([e, round(v * (1 - 0.5 ** (e / 4)), 6), round(v + 0.05, 6), 1.0])
    m = {
        "run_id": spec["run_id"],
        "status": status,
        "stop_reason": "early_stop" if status == "ok" else status,
        "val_r": v if status == "ok" else None,
        "val_mse": 0.2,
        "val_spearman": v - 0.02,
        "train_r": v + 0.1,
        "best_epoch": epochs,
        "epochs_run": epochs,
        "still_improving": False,
        "pred_sd_ratio": 0.9,
        "mse_by_activity_quintile": [0.1] * 5,
        "mse_by_gc_tercile": [0.2] * 3,
        "stress": {"finite": True, "max_abs_z": 3.0, "exploded": False},
        "throughput_seq_per_s": 20000,
        "peak_mem_gb": 2.0,
        "wall_time_s": 1.0,
        "n_params": 1000,
        "cost_equiv": epochs / max_epochs,
        "cap_s_used": spec["time_cap_s"],
    }
    if status == "error":
        m["error"] = "synthetic infrastructure failure"
    if spec.get("val_select_manifest"):
        m["val_select_r"] = v - 0.003
    tmp = out / "metrics.json.tmp"
    tmp.write_text(json.dumps(m))
    os.replace(tmp, out / "metrics.json")


if __name__ == "__main__":
    run(sys.argv[sys.argv.index("--spec") + 1])
