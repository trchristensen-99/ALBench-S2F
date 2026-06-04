"""Train AG S1 head on chr-split K562 train data, with either real or oracle labels.

Loads N sequences from chr_train pool, encodes them with AG, trains a linear
head, and evaluates on chr-test (both oracle and real labels).

Uses the same encoder + head training infrastructure as train_ag_s1_fold_comparison.py
but with chr-split data and subsample/seed controls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Defaults from the AG_S1 fold comparison work
LR = 0.0003
WD = 1e-6
DROPOUT = 0.1
EPOCHS = 50
PATIENCE = 7
BATCH_SIZE = 256
HEAD_SEED_BASE = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True, help="dataset subsample + head init seed")
    ap.add_argument("--label_source", required=True, choices=["oracle", "real"])
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        print(f"  already done: {out}")
        return

    os.environ["ALPHAGENOME_WEIGHTS"] = (
        "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"
    )

    # Load chr-split data
    z_train = np.load(REPO / "outputs/chr_split_cache/chr_train_ref_only.npz", allow_pickle=True)
    z_val = np.load(REPO / "outputs/chr_split_cache/chr_val_ref_only.npz", allow_pickle=True)

    label_key = f"{args.label_source}_labels"
    train_seqs_full = z_train["sequences"]
    train_labels_full = z_train[label_key].astype(np.float32)
    finite = np.isfinite(train_labels_full)
    train_seqs_full = train_seqs_full[finite]
    train_labels_full = train_labels_full[finite]

    rng = np.random.default_rng(args.seed)
    if args.N > len(train_seqs_full):
        N_eff = len(train_seqs_full)
    else:
        N_eff = args.N
    idx = rng.choice(len(train_seqs_full), size=N_eff, replace=False)
    train_seqs = list(train_seqs_full[idx])
    train_labels = train_labels_full[idx]

    val_seqs = list(z_val["sequences"])
    val_labels = z_val[label_key].astype(np.float32)
    finite_v = np.isfinite(val_labels)
    val_seqs = [s for s, f in zip(val_seqs, finite_v) if f]
    val_labels = val_labels[finite_v]

    # Test set with both oracle and real labels
    test_z = np.load(
        REPO / "data/k562/test_sets_ag_s2_chrsplit/genomic_oracle.npz", allow_pickle=True
    )
    test_seqs = list(test_z["sequences"])
    test_oracle = test_z["oracle_mean"].astype(np.float32)
    test_real = test_z["true_label"].astype(np.float32)
    # SNV + OOD extras
    snv = np.load(REPO / "data/k562/test_sets_ag_s2_chrsplit/snv_oracle.npz", allow_pickle=True)
    snv_ref_seqs = list(snv["ref_sequences"])
    snv_alt_seqs = list(snv["alt_sequences"])
    snv_delta_oracle = snv["delta_mean"].astype(np.float32)
    snv_delta_real = snv["true_delta"].astype(np.float32)
    ood_z = np.load(REPO / "data/k562/test_sets_ag_s2_chrsplit/ood_oracle.npz", allow_pickle=True)
    ood_seqs = list(ood_z["sequences"])
    ood_oracle_y = ood_z["oracle_mean"].astype(np.float32)
    ood_real_y = ood_z["true_label"].astype(np.float32)

    print(f"  N={N_eff}  seed={args.seed}  label_source={args.label_source}", flush=True)
    print(
        f"  train={len(train_seqs)}  val={len(val_seqs)}  test={len(test_seqs)}  snv={len(snv_ref_seqs)}  ood={len(ood_seqs)}",
        flush=True,
    )

    # Encode with AG
    from experiments.exp1_1_scaling import (
        _AG_MODEL_CACHE,
        _encode_sequences_for_ag,
        _get_ag_model_and_encoder,
    )

    _AG_MODEL_CACHE.clear()
    ag = _get_ag_model_and_encoder("k562")
    model = ag["model"]
    head_name = ag["head_name"]

    print("  encoding train...", flush=True)
    t0 = time.time()
    train_embs = _encode_sequences_for_ag(train_seqs, "k562", ag["encoder_fn"]).astype(np.float32)
    print(f"    {train_embs.shape}  {time.time() - t0:.0f}s", flush=True)
    print("  encoding val...", flush=True)
    t0 = time.time()
    val_embs = _encode_sequences_for_ag(val_seqs, "k562", ag["encoder_fn"]).astype(np.float32)
    print(f"    {val_embs.shape}  {time.time() - t0:.0f}s", flush=True)
    print("  encoding test...", flush=True)
    t0 = time.time()
    test_embs = _encode_sequences_for_ag(test_seqs, "k562", ag["encoder_fn"]).astype(np.float32)
    print(f"    {test_embs.shape}  {time.time() - t0:.0f}s", flush=True)
    print("  encoding snv_ref...", flush=True)
    t0 = time.time()
    snv_ref_embs = _encode_sequences_for_ag(snv_ref_seqs, "k562", ag["encoder_fn"]).astype(
        np.float32
    )
    print(f"    {snv_ref_embs.shape}  {time.time() - t0:.0f}s", flush=True)
    print("  encoding snv_alt...", flush=True)
    t0 = time.time()
    snv_alt_embs = _encode_sequences_for_ag(snv_alt_seqs, "k562", ag["encoder_fn"]).astype(
        np.float32
    )
    print(f"    {snv_alt_embs.shape}  {time.time() - t0:.0f}s", flush=True)
    print("  encoding ood...", flush=True)
    t0 = time.time()
    ood_embs = _encode_sequences_for_ag(ood_seqs, "k562", ag["encoder_fn"]).astype(np.float32)
    print(f"    {ood_embs.shape}  {time.time() - t0:.0f}s", flush=True)

    # Train head
    from models.embedding_cache import reinit_head_params

    reinit_head_params(model, head_name, num_tokens=5, dim=1536, rng=HEAD_SEED_BASE + args.seed)
    head_train_fn = ag["head_train_fn"]
    head_predict_fn = ag["head_predict_fn"]
    optimizer = optax.adamw(learning_rate=LR, weight_decay=WD)
    opt_state = optimizer.init(model._params)

    @jax.jit
    def train_step(params, opt_state, rng_key, emb, targets, org_idx):
        def loss_fn(p):
            preds = head_train_fn(p, rng_key, emb, org_idx)
            return jnp.mean((preds.squeeze() - targets) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_state, loss

    @jax.jit
    def eval_step(params, emb, org_idx):
        return head_predict_fn(params, emb, org_idx).squeeze()

    org_idx = jnp.array([0])
    params = model._params
    n_train = len(train_labels)
    best_val = -1.0
    best_params = None
    no_improve = 0
    rng_e = np.random.default_rng(args.seed)
    for epoch in range(EPOCHS):
        perm = rng_e.permutation(n_train)
        for i in range(0, n_train, BATCH_SIZE):
            batch = perm[i : i + BATCH_SIZE]
            params, opt_state, _ = train_step(
                params,
                opt_state,
                jax.random.PRNGKey(epoch * 10000 + i),
                jnp.array(train_embs[batch]),
                jnp.array(train_labels[batch]),
                org_idx,
            )
        # Val
        v_preds = []
        for i in range(0, len(val_labels), BATCH_SIZE):
            v_preds.append(
                np.array(eval_step(params, jnp.array(val_embs[i : i + BATCH_SIZE]), org_idx))
            )
        v_preds = np.concatenate(v_preds)
        m = np.isfinite(v_preds) & np.isfinite(val_labels)
        val_r = float(pearsonr(v_preds[m], val_labels[m])[0]) if m.sum() > 8 else -1.0
        if val_r > best_val:
            best_val = val_r
            best_params = jax.tree_util.tree_map(lambda x: x, params)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    # Test predictions: genomic, snv_ref, snv_alt, ood
    def predict_all(embs):
        out_chunks = []
        for i in range(0, len(embs), BATCH_SIZE):
            out_chunks.append(
                np.array(eval_step(best_params, jnp.array(embs[i : i + BATCH_SIZE]), org_idx))
            )
        return np.concatenate(out_chunks)

    t_preds = predict_all(test_embs)
    snv_ref_preds = predict_all(snv_ref_embs)
    snv_alt_preds = predict_all(snv_alt_embs)
    snv_delta_preds = snv_alt_preds - snv_ref_preds
    ood_preds = predict_all(ood_embs)

    def m_against(pred, y):
        mm = np.isfinite(pred) & np.isfinite(y)
        if mm.sum() < 8:
            return None
        r = float(pearsonr(pred[mm], y[mm])[0])
        mse = float(((pred[mm] - y[mm]) ** 2).mean())
        return {"pearson_r": r, "mse": mse, "n": int(mm.sum())}

    summary = {
        "N": N_eff,
        "seed": args.seed,
        "label_source": args.label_source,
        "hp": {"lr": LR, "wd": WD, "dropout": DROPOUT, "epochs_run": epoch + 1},
        "val_pearson": float(best_val),
        "test_vs_oracle": m_against(t_preds, test_oracle),
        "test_vs_real": m_against(t_preds, test_real),
        "ood_vs_oracle": m_against(ood_preds, ood_oracle_y),
        "ood_vs_real": m_against(ood_preds, ood_real_y),
        "snv_delta_vs_oracle": m_against(snv_delta_preds, snv_delta_oracle),
        "snv_delta_vs_real": m_against(snv_delta_preds, snv_delta_real),
    }
    # Re-predict on val with best_params to save val_pred (for post-hoc affine recal)
    val_preds_final = []
    for i in range(0, len(val_labels), BATCH_SIZE):
        val_preds_final.append(
            np.array(eval_step(best_params, jnp.array(val_embs[i : i + BATCH_SIZE]), org_idx))
        )
    val_preds_final = np.concatenate(val_preds_final)
    np.savez(
        out / "model.npz",
        val_pred=val_preds_final.astype(np.float32),
        val_labels=val_labels.astype(np.float32),
        test_pred=t_preds.astype(np.float32),
        test_pred_snv_ref=snv_ref_preds.astype(np.float32),
        test_pred_snv_alt=snv_alt_preds.astype(np.float32),
        test_pred_ood=ood_preds.astype(np.float32),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  saved: {out}  val_pearson={best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
