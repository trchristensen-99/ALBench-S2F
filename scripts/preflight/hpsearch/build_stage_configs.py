"""Build configs.json for Stage 2/3/4 of the standardized HP-search procedure.

Auto-generated "AutoResearch surrogate" — since the LLM-driven AutoResearch
can't live inside a SLURM job, we use deterministic ablation/sweep logic
that mimics what the subagents would propose:

  stage2_ablate: take top-K from Stage 1, generate ~6 perturbations per
                 anchor (vary one HP at a time: lr, dropout, optimizer,
                 block_class, aug, shape). 5 anchors × 6 = 30 configs.

  stage3_aug:    take top-K from Stage 1+2 leaderboard, sweep all
                 (aug × max_shift × evoaug_intensity) combos. 3 anchors ×
                 9 combos = 27 configs.

  stage4_seeds:  take top-2 from Stage 3 leaderboard, replicate with 3
                 different seeds each. 6 configs.

Usage (called by SLURM job from standardized_procedure.sh):
  python -m scripts.preflight.hpsearch.build_stage_configs \\
    --stage stage2_ablate \\
    --in_dirs results/preflight/hpsearch/std_legnet_d20000/stage1_random \\
              results/preflight/hpsearch/std_legnet_d20000/stage1_optuna \\
              results/preflight/hpsearch/std_legnet_d20000/stage1_pbt \\
    --out_dir results/preflight/hpsearch/std_legnet_d20000/stage2_ablate \\
    --top_k 5 --arch legnet --d_train 20000 --epochs 60 --patience 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _walk_results(in_dirs: list[Path]) -> list[dict]:
    """Walk Ray Tune + subprocess result.jsons under each in_dir."""
    rows = []
    for d in in_dirs:
        d = Path(d)
        if not d.exists():
            continue
        # Ray Tune layout: in_dir/<run_name>/trainable_*/run/result.json
        for rf in d.rglob("result.json"):
            try:
                r = json.loads(rf.read_text())
            except Exception:
                continue
            if "best_val_mse" not in r:
                continue
            # rf may be relative or absolute depending on how the caller invoked
            # us; either way just store as-is (the trial_path is informational).
            rows.append(
                {
                    "val": r["best_val_mse"],
                    "test": r.get("test_mse_at_best_val", float("nan")),
                    "hp": r.get("hp", {}),
                    "aug": r.get("augmentations", "rev_complement"),
                    "trial_path": str(rf.parent),
                }
            )
    rows.sort(key=lambda r: r["val"])
    return rows


def _trim_hp(hp: dict, aug: str) -> dict:
    """Pick the subset of hp that we'll vary; defaults for missing keys."""
    return {
        "lr": hp.get("lr", 0.001),
        "batch_size": int(hp.get("batch_size", 256)),
        "weight_decay": hp.get("weight_decay", 0.01),
        "block_sizes": hp.get("block_sizes", [256] * 4),
        "ks": int(hp.get("ks", 5)),
        "optimizer": hp.get("optimizer", "adamw"),
        "block_class": hp.get("block_class", "eff"),
        "conv_dropout": hp.get("conv_dropout", hp.get("dropout", 0.1)),
        "dense_dropout": hp.get("dense_dropout", 0.0),
        "dense_dims": hp.get("dense_dims", []),
        "aug": aug,
    }


def stage2_ablate(top_anchors: list[dict]) -> list[dict]:
    """7 perturbations per anchor — vary one HP axis at a time."""
    proposals = []
    for i, anchor in enumerate(top_anchors[:5]):
        base = _trim_hp(anchor["hp"], anchor["aug"])
        # 1. lr ablations
        for lr_mult, tag in [(0.5, "halfLR"), (2.0, "doubleLR")]:
            p = dict(base)
            p["lr"] = max(1e-6, min(1e-2, base["lr"] * lr_mult))
            proposals.append((f"s2_a{i}_{tag}", p))
        # 2. conv_dropout flip
        p = dict(base)
        p["conv_dropout"] = 0.2 if base["conv_dropout"] < 0.1 else 0.0
        proposals.append((f"s2_a{i}_convdrop_flip", p))
        # 3. block_class change
        new_bc = {"eff": "ag", "ag": "plain", "plain": "eff"}[base["block_class"]]
        p = dict(base)
        p["block_class"] = new_bc
        # AG often needs lower LR
        if new_bc == "ag":
            p["lr"] = min(p["lr"], 3e-4)
        proposals.append((f"s2_a{i}_block_{new_bc}", p))
        # 4. optimizer change
        new_opt = {"adamw": "muon", "adam": "adamw", "muon": "adamw"}.get(
            base["optimizer"], "adamw"
        )
        p = dict(base)
        p["optimizer"] = new_opt
        proposals.append((f"s2_a{i}_opt_{new_opt}", p))
        # 5. aug switch
        cur_aug = base["aug"]
        new_aug = "rc_shift" if cur_aug != "rc_shift" else "rev_complement"
        p = dict(base)
        p["aug"] = new_aug
        proposals.append((f"s2_a{i}_aug_{new_aug}", p))
        # 6. ks rotation — Stage 1 strategies tend to collapse on the default
        # ks=5; force kernel-size diversity.
        new_ks = 3 if base["ks"] != 3 else 7
        p = dict(base)
        p["ks"] = new_ks
        proposals.append((f"s2_a{i}_ks_{new_ks}", p))
    return _to_run_configs(proposals)


def stage3_aug(top_anchors: list[dict]) -> list[dict]:
    """9 augmentation variants per anchor — full aug × max_shift × intensity grid."""
    grid = [
        ("rev_complement", 0, 0),
        ("rc_shift", 5, 0),
        ("rc_shift", 15, 0),
        ("rc_shift", 25, 0),
        ("rc_shift_evoaug", 15, 1),
        ("rc_shift_evoaug", 15, 2),
        ("rc_shift_evoaug", 15, 4),
        ("rc_shift_evoaug", 25, 2),
        ("rc_shift_evoaug", 25, 4),
    ]
    proposals = []
    for i, anchor in enumerate(top_anchors[:3]):
        base = _trim_hp(anchor["hp"], anchor["aug"])
        for aug, max_shift, evoaug in grid:
            p = dict(base)
            p["aug"] = aug
            p["max_shift"] = max_shift
            p["evoaug_intensity"] = evoaug
            tag = f"{aug}_ms{max_shift}_ei{evoaug}"
            proposals.append((f"s3_a{i}_{tag}", p))
    return _to_run_configs(proposals)


def stage4_seeds(top_anchors: list[dict], n_seeds: int = 3) -> list[dict]:
    """N seeds per anchor for variance analysis."""
    seeds = [42, 43, 44, 45, 46][:n_seeds]
    proposals = []
    for i, anchor in enumerate(top_anchors[:2]):
        base = _trim_hp(anchor["hp"], anchor["aug"])
        for s in seeds:
            p = dict(base)
            p["__seed"] = s
            proposals.append((f"s4_a{i}_seed{s}", p))
    return _to_run_configs(proposals)


def _to_run_configs(proposals: list[tuple]) -> list[dict]:
    """Convert (label, hp dict) pairs into parallel_gpu_runner config records."""
    out = []
    for label, hp in proposals:
        overrides = []
        for k in ("lr", "batch_size", "weight_decay", "ks", "optimizer", "block_class"):
            overrides.append(f"{k}={hp[k]}")
        # block_sizes (list literal)
        bs = ",".join(str(int(x)) for x in hp["block_sizes"])
        overrides.append(f"block_sizes=[{bs}]")
        overrides.append(f"conv_dropout={hp['conv_dropout']}")
        overrides.append(f"dense_dropout={hp['dense_dropout']}")
        dd = hp.get("dense_dims", [])
        dd_str = "[" + ",".join(str(int(x)) for x in dd) + "]" if isinstance(dd, list) else str(dd)
        overrides.append(f"dense_dims={dd_str}")
        cfg = {
            "label": label,
            "arch": hp.get("__arch", "legnet"),
            "d_train": hp.get("__d_train", 20000),
            "seed": int(hp.get("__seed", 42)),
            "epochs": hp.get("__epochs", 60),
            "patience": hp.get("__patience", 15),
            "aug": hp.get("aug", "rev_complement"),
            "output_dir": hp.get("__out_dir_placeholder", "TBD"),  # filled by caller
            "hp_overrides": overrides,
        }
        if hp.get("max_shift", 0) > 0:
            cfg["max_shift"] = int(hp["max_shift"])
        if hp.get("evoaug_intensity", 0) > 0:
            cfg["evoaug_intensity"] = int(hp["evoaug_intensity"])
        out.append(cfg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage", required=True, choices=["stage2_ablate", "stage3_aug", "stage4_seeds"]
    )
    ap.add_argument(
        "--in_dirs",
        nargs="+",
        required=True,
        help="Dirs to read prior stage results from (multiple OK).",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--arch", default="legnet")
    ap.add_argument("--d_train", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--n_seeds", type=int, default=3)
    args = ap.parse_args()

    in_dirs = [Path(d) for d in args.in_dirs]
    rows = _walk_results(in_dirs)
    if not rows:
        print(f"No completed trials found in {args.in_dirs}")
        raise SystemExit(2)
    print(f"Read {len(rows)} prior trials; best val={rows[0]['val']:.5f}")

    top_anchors = rows[: args.top_k]
    if args.stage == "stage2_ablate":
        configs = stage2_ablate(top_anchors)
    elif args.stage == "stage3_aug":
        configs = stage3_aug(top_anchors)
    else:
        configs = stage4_seeds(top_anchors, n_seeds=args.n_seeds)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Fill in per-config output dirs + arch + d_train
    for c in configs:
        c["arch"] = args.arch
        c["d_train"] = args.d_train
        c["epochs"] = args.epochs
        c["patience"] = args.patience
        trial_out = out_dir / c["label"]
        c["output_dir"] = (
            str(trial_out.relative_to(REPO)) if trial_out.is_absolute() else str(trial_out)
        )

    out_json = out_dir / "configs.json"
    out_json.write_text(json.dumps(configs, indent=2))
    print(f"Wrote {len(configs)} configs → {out_json}")


if __name__ == "__main__":
    main()
