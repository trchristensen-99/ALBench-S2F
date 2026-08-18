"""Pick the FM hyperparameter config from the sweep, then launch the rank-stability check.

Selection: highest held-out val_pearson (NEVER a battery/test metric) on the reservoir-balanced
mixture. Ties broken toward the FASTER config, since thousands of cells remain to run.

Rank-stability: the chosen config is only defensible if the RESERVOIR RANKING does not depend on it.
So we take the top-K configs and run each across every reservoir; a later analysis compares the
per-config reservoir orderings (Spearman). Stable ordering => the reservoir comparison is robust to
HP choice, which is what justifies freezing ONE config instead of per-cell HP search.
"""

import argparse
import glob
import json
import os
import subprocess

SBATCH = "/cm/shared/apps/slurm/current/bin/sbatch"
RESERVOIRS = [
    ("genomic", "--genomic_train outputs/chr_split_cache/chr_train_ref_only.npz"),
    ("evoaug_heavy", None),
    ("motif_planted_v2", None),
    ("phylogenetic_zoonomia", None),
    ("random", None),
    ("dinuc_shuffle", None),
]


def flags_from_hp(hp):
    f = [
        f"--lr {hp['lr']}",
        f"--encoder_lr_mult {hp['encoder_lr_mult']}",
        f"--stage1_frac {hp['stage1_frac']}",
        f"--weight_decay {hp['weight_decay']}",
        f"--head_hidden {hp['head_hidden']}",
        f"--head_dropout {hp['head_dropout']}",
        f"--batch_size {hp['batch_size']}",
        f"--epochs {hp['epochs']}",
        f"--pooling {hp['pooling']}",
    ]
    if hp.get("center_bins"):
        f.append(f"--center_bins {hp['center_bins']}")
    return " ".join(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/fm_hpsearch")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--D", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_root", default="outputs/fm_rankstab")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.root, "cfg*", "fm_scaling_point.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        v = d.get("val_pearson")
        if v is None:
            continue
        rows.append(
            (v, d.get("train_sec", 9e9), os.path.basename(os.path.dirname(f)), d.get("hp", {}))
        )
    if not rows:
        print("[select] no completed configs with val_pearson yet — nothing to do")
        return

    # best val first; on near-ties (<0.002, i.e. noise) prefer the faster config
    rows.sort(key=lambda r: (-r[0], r[1]))
    print(f"{'cfg':<8} {'val_r':>8} {'sec':>7}  hp")
    for v, s, name, hp in rows:
        print(f"{name:<8} {v:>8.4f} {s:>7.0f}  {hp}")
    best_v = rows[0][0]
    near = [r for r in rows if best_v - r[0] < 0.002]
    near.sort(key=lambda r: r[1])
    print(
        f"\n[select] best val_r={best_v:.4f}; {len(near)} within noise (0.002) -> fastest = {near[0][2]}"
    )
    json.dump(
        {
            "chosen": near[0][2],
            "chosen_hp": near[0][3],
            "val_pearson": near[0][0],
            "ranked": [
                {"cfg": n, "val_pearson": v, "train_sec": s, "hp": h} for v, s, n, h in rows
            ],
        },
        open(os.path.join(args.root, "selection.json"), "w"),
        indent=2,
    )

    # rank-stability: top-K configs x every reservoir
    for v, s, name, hp in rows[: args.topk]:
        for rname, src in RESERVOIRS:
            src = (
                src
                or f"--reservoir_cache outputs/reservoir_cache/k562_{rname}_d{args.D}_seed{args.seed}.npz"
            )
            out = f"{args.out_root}/{name}_{rname}_d{args.D}"
            if os.path.exists(os.path.join(out, "fm_scaling_point.json")):
                continue
            cmd = (
                f"cd {os.getcwd()}; export TF_CPP_MIN_LOG_LEVEL=3 TQDM_DISABLE=1 "
                f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
                f"uv run --no-sync python scripts/fm_scaling_driver.py --model borzoi "
                f"--head full_encoder --input_len 512 {src} --D {args.D} --seed {args.seed} "
                f"--battery_dir data/k562/test_sets_ag_s2_chrsplit --val_frac 0.1 "
                f"{flags_from_hp(hp)} --out_dir {out}"
            )
            sb = [
                SBATCH,
                "--parsable",
                "--qos=slow_nice",
                "--time=08:00:00",
                f"--job-name=rs_{name}_{rname[:8]}",
                "--partition=gpuq",
                "--gres=gpu:h100:1",
                "--cpus-per-task=6",
                "--mem=64G",
                f"--output=logs/rs_{name}_{rname}_%j.out",
                f"--wrap={cmd}",
            ]
            if args.dry_run:
                print(f"  DRY {name} x {rname}")
            else:
                jid = subprocess.run(
                    sb, capture_output=True, text=True, stdin=subprocess.DEVNULL
                ).stdout.strip()
                print(f"  submitted {name} x {rname} jid={jid}")


if __name__ == "__main__":
    main()
