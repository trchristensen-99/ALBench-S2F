#!/usr/bin/env python
"""Extract best HP configs from RayTune trial results."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
rt_dir = REPO / "outputs" / "raytune_results"
best_dir = REPO / "outputs" / "raytune_best"

for exp_dir in sorted(rt_dir.iterdir()):
    if not exp_dir.is_dir():
        continue

    best_val = -1.0
    best_config = None

    for trial_dir in exp_dir.iterdir():
        if not trial_dir.is_dir():
            continue
        result_file = trial_dir / "result.json"
        if not result_file.exists():
            continue
        try:
            d = json.loads(result_file.read_text())
            val_p = d.get("val_pearson", -1)
            if val_p > best_val:
                best_val = val_p
                params_file = trial_dir / "params.json"
                if params_file.exists():
                    best_config = json.loads(params_file.read_text())
        except Exception:
            continue

    if best_config and best_val > 0:
        out_path = best_dir / exp_dir.name
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "best_config.json", "w") as f:
            json.dump(
                {"config": best_config, "val_pearson": best_val, "experiment": exp_dir.name},
                f,
                indent=2,
            )
        lr = best_config.get("lr", "?")
        bs = best_config.get("batch_size", "?")
        lr_str = f"{lr:.5f}" if isinstance(lr, float) else str(lr)
        print(f"{exp_dir.name}: val={best_val:.4f} lr={lr_str} bs={bs}")
