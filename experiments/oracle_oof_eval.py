"""Out-of-fold oracle evaluation: score each battery sequence with ONLY the fold that held it out.

The deployed oracle label is `oracle_mean`, the average over all 10 fold models. But
`train_oracle_s2_fullcv.py` splits a RANDOM permutation of the full 856,252-sequence pool with no
chromosome exclusion, so every battery sequence sat in the TRAINING split of 9 folds and was held
out by exactly 1. The published battery numbers are therefore 9/10 in-sample -- and the r = 0.974 on
absolute activity actually EXCEEDS the assay's own reliability ceiling of 0.943 (computed from Gosai
lfcSE), which is only possible if the ensemble has partly fit the measurement noise.

The split is deterministic, so nothing needs retraining:
    perm       = np.random.default_rng(seed=42).permutation(856252)
    fold k val = perm[k*fold_size : (k+1)*fold_size]     fold_size = 856252 // 10
    (the final fold absorbs the remainder)

Run as a 10-task array; each task predicts only the ~11k battery sequences its own fold held out,
so the whole evaluation costs one tenth of a full battery rescore.

    python experiments/oracle_oof_eval.py --stage foldmap                  # once, CPU
    python experiments/oracle_oof_eval.py --stage predict --fold_id $K     # array 0-9, GPU
    python scripts/analysis/oracle_oof_report.py                          # metrics
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

SEED = 42
N_FOLDS = 10
# battery file -> sequence keys to score
BATTERY = {
    "genomic": ("genomic_oracle.npz", ["sequences"]),
    "ood": ("ood_oracle.npz", ["sequences"]),
    "ctrl_neg": ("ctrl_neg_oracle.npz", ["sequences"]),
    "snv": ("snv_oracle.npz", ["ref_sequences", "alt_sequences"]),
}


def owner_fold(pool, n_folds=N_FOLDS):
    """Invert the training split: which fold held out each row of the pool."""
    perm = np.random.default_rng(seed=SEED).permutation(pool)
    fold_size = pool // n_folds
    owner = np.full(pool, -1, dtype=np.int8)
    for k in range(n_folds):
        s = k * fold_size
        e = s + fold_size if k < n_folds - 1 else pool
        owner[perm[s:e]] = k
    assert owner.min() >= 0
    return owner


def stage_foldmap(args):
    from scripts.build_full_oracle_cache import load_all_sequences

    all_seqs, _ = load_all_sequences()
    pool = len(all_seqs)
    print(f"[pool] {pool:,} sequences, fold_size={pool // N_FOLDS:,}", flush=True)
    owner_by_row = owner_fold(pool)

    # sequence -> fold that held it out. Duplicates: a repeated sequence may appear in several
    # folds' training data, so it is only truly held out if EVERY copy landed in one fold's val
    # slice. Record the set of owners and mark ambiguous ones -- otherwise a duplicate would be
    # scored as out-of-fold while a copy of it was in that fold's training set.
    first, ambiguous = {}, set()
    for i, s in enumerate(all_seqs):
        s = str(s)
        f = int(owner_by_row[i])
        if s in first:
            if first[s] != f:
                ambiguous.add(s)
        else:
            first[s] = f
    print(
        f"[pool] {len(first):,} unique; {len(ambiguous):,} duplicated across folds (excluded)",
        flush=True,
    )

    out, summary = {}, {}
    for name, (fn, keys) in BATTERY.items():
        path = Path(args.battery_dir) / fn
        if not path.exists():
            print(f"[skip] {fn} missing")
            continue
        z = np.load(path, allow_pickle=True)
        for key in keys:
            seqs = [str(x) for x in z[key]]
            owner = np.full(len(seqs), -1, dtype=np.int8)
            for j, s in enumerate(seqs):
                if s in first and s not in ambiguous:
                    owner[j] = first[s]
            tag = f"{name}.{key}"
            out[tag] = owner
            ok = owner >= 0
            summary[tag] = {
                "n": len(seqs),
                "scorable_oof": int(ok.sum()),
                "not_in_pool_or_ambiguous": int((~ok).sum()),
                "per_fold": np.bincount(owner[ok], minlength=N_FOLDS).tolist(),
            }
            print(
                f"  {tag:<22} n={len(seqs):>7,} oof={int(ok.sum()):>7,} "
                f"skip={int((~ok).sum()):>6,} per-fold={summary[tag]['per_fold']}",
                flush=True,
            )

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(Path(args.out_dir) / "foldmap.npz", **out)
    with open(Path(args.out_dir) / "foldmap.json", "w") as f:
        json.dump(
            {
                "pool": pool,
                "seed": SEED,
                "n_folds": N_FOLDS,
                "n_unique": len(first),
                "n_ambiguous": len(ambiguous),
                "sets": summary,
            },
            f,
            indent=2,
        )
    tot = sum(v["scorable_oof"] for v in summary.values())
    print(f"\n[oof] {tot:,} forward passes total (~{tot / N_FOLDS:,.0f} per fold task)")


def stage_predict(args):
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from alphagenome_ft import create_model_with_heads

    from experiments.generate_stage2_pseudolabels_single_fold import _predict_strings
    from models.alphagenome_heads import register_s2f_head
    from models.embedding_cache import reinit_head_params

    k = int(args.fold_id)
    stem = "allfold" if args.all_sequences else "oof_fold"
    suffix = "" if args.n_shards == 1 else f"_s{args.shard_id}"
    out_path = Path(args.out_dir) / f"{stem}_{k}{suffix}.npz"
    if out_path.exists() and not args.overwrite:
        print(f"SKIP: {out_path} exists")
        return

    fm = (
        None
        if args.all_sequences
        else np.load(Path(args.out_dir) / "foldmap.npz", allow_pickle=True)
    )
    todo = {}
    for name, (fn, keys) in BATTERY.items():
        path = Path(args.battery_dir) / fn
        if not path.exists():
            continue
        z = np.load(path, allow_pickle=True)
        for key in keys:
            tag = f"{name}.{key}"
            if args.all_sequences:
                sel = np.arange(len(z[key]))
            else:
                if tag not in fm.files:
                    continue
                sel = np.where(fm[tag] == k)[0]
            if args.n_shards > 1:
                sel = sel[args.shard_id :: args.n_shards]  # strided, so shards stay balanced
            if len(sel):
                todo[tag] = (sel, [str(z[key][i]) for i in sel])
    n_tot = sum(len(v[0]) for v in todo.values())
    print(f"[fold {k}] {n_tot:,} sequences across {len(todo)} sets", flush=True)
    if n_tot == 0:
        return

    unique_head_name = f"{args.head_name}_{args.head_arch.replace('-', '_')}_v4"
    register_s2f_head(
        head_name=unique_head_name,
        arch=args.head_arch,
        task_mode="human",
        num_tracks=args.num_tracks,
        dropout_rate=0.1,
    )
    model = create_model_with_heads(
        "all_folds",
        heads=[unique_head_name],
        checkpoint_path=args.weights_path,
        use_encoder_output=True,
        detach_backbone=False,
    )
    reinit_head_params(model, unique_head_name, num_tokens=5, dim=1536)

    @jax.jit
    def predict_step(params, state, sequences):
        return model._predict(
            params,
            state,
            sequences,
            jnp.zeros(len(sequences), dtype=jnp.int32),
            # the installed alphagenome_ft makes requested_outputs a REQUIRED keyword-only arg;
            # the older reference script omitted it and now fails with a TypeError
            requested_outputs=[unique_head_name],
            negative_strand_mask=jnp.zeros(len(sequences), dtype=bool),
            strand_reindexing=None,
        )[unique_head_name]

    # orbax requires an ABSOLUTE path: a relative one raises
    # "Checkpoint path should be absolute" from build_kvstore_tspec.
    ckpt = Path(args.oracle_dir).expanduser().resolve() / f"fold_{k}" / "best_model" / "checkpoint"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    loaded_params, _ = ocp.StandardCheckpointer().restore(ckpt)
    model._params = jax.device_put(loaded_params)
    _ = predict_step(
        model._params, model._state, jnp.zeros((args.batch_size, 600, 4), dtype=jnp.float32)
    )
    _.block_until_ready()
    print("[jit] compiled", flush=True)

    res = {"fold_id": k}
    for tag, (sel, seqs) in todo.items():
        p = _predict_strings(predict_step, model._params, model._state, seqs, args.batch_size)
        res[f"{tag}|idx"] = sel.astype(np.int64)
        res[f"{tag}|pred"] = np.asarray(p, dtype=np.float32).ravel()
        print(f"  {tag}: {len(sel):,} done", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(out_path, **res)
    print(f"[fold {k}] saved {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["foldmap", "predict"], required=True)
    ap.add_argument("--fold_id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--battery_dir", default="data/k562/test_sets_ag_s2_chrsplit")
    ap.add_argument("--oracle_dir", default="outputs/oracle_full856k_clean/s2")
    ap.add_argument("--out_dir", default="outputs/oracle_oof")
    ap.add_argument("--head_name", default="oracle_k562_fullcv")
    ap.add_argument("--head_arch", default="boda-flatten-512-512")
    ap.add_argument("--num_tracks", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument(
        "--weights_path",
        default="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1",
    )
    ap.add_argument(
        "--all_sequences",
        action="store_true",
        help="predict EVERY sequence with this fold, not just the ones it held out. Averaging the "
        "10 resulting files reproduces the deployed all-folds ensemble on the same pairs, which is "
        "what makes an inflation number comparable.",
    )
    ap.add_argument(
        "--n_shards",
        type=int,
        default=1,
        help="split each fold's sequences across this many tasks. Wall-clock per task falls ~1/N, "
        "but every task re-pays the ~6 min JAX init + weight load, so sharding only wins when "
        "predict time already dominates setup (roughly >20k sequences per fold).",
    )
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    (stage_foldmap if a.stage == "foldmap" else stage_predict)(a)
