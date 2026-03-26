#!/usr/bin/env python3
"""Re-evaluate SNV metrics for HepG2/SK-N-SH using cell-specific labels.

Loads each model checkpoint, predicts on cell-specific SNV test pairs,
and patches snv_abs + snv_delta in the existing result JSONs.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CELL_LINE_LABEL_COLS = {
    "hepg2": "HepG2_log2FC",
    "sknsh": "SKNSH_log2FC",
}


def _safe_corr(pred, true, fn):
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() < 3:
        return 0.0
    return float(fn(pred[mask], true[mask])[0])


def _compute_metrics(pred, true):
    mask = np.isfinite(pred) & np.isfinite(true)
    return {
        "pearson_r": _safe_corr(pred, true, pearsonr),
        "spearman_r": _safe_corr(pred, true, spearmanr),
        "mse": float(np.mean((pred[mask] - true[mask]) ** 2)),
        "n": int(mask.sum()),
    }


def patch_json(path, snv_abs_metrics, snv_delta_metrics):
    """Patch snv_abs and snv_delta into result JSON."""
    data = json.loads(path.read_text())
    container = data.get("test_metrics", data)
    if snv_abs_metrics:
        container["snv_abs"] = snv_abs_metrics
    if snv_delta_metrics:
        container["snv_delta"] = snv_delta_metrics
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    print(f"    Patched {path}", flush=True)


def predict_batched(predict_fn, sequences, batch_size=256):
    """Predict in batches."""
    preds = []
    for i in range(0, len(sequences), batch_size):
        preds.append(predict_fn(sequences[i : i + batch_size]))
    return np.concatenate(preds)


# ── Model-specific loading and prediction ─────────────────────────────────


def eval_malinois(result_dir, snv_df, cell):
    """Load Malinois and predict on SNV sequences."""
    import torch

    from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
    from models.basset_branched import BassetBranched

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flank_5 = MPRA_UPSTREAM[-200:]
    flank_3 = MPRA_DOWNSTREAM[:200]
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

    # Load model
    ckpt_path = result_dir / "best_model.pt"
    if not ckpt_path.exists():
        for p in result_dir.rglob("best_model.pt"):
            ckpt_path = p
            break
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = BassetBranched(input_channels=4, seq_length=600)
    model.load_state_dict(ckpt)
    model.to(device).eval()

    def encode_and_predict(sequences):
        encoded = []
        for seq in sequences:
            full = flank_5 + seq[:200] + flank_3
            oh = np.zeros((4, 600), dtype=np.float32)
            for j, c in enumerate(full[:600]):
                if c in mapping:
                    oh[mapping[c], j] = 1.0
            encoded.append(oh)
        x = torch.tensor(np.stack(encoded), device=device)
        with torch.no_grad():
            return model(x).cpu().numpy().reshape(-1)

    ref_preds = predict_batched(encode_and_predict, snv_df["sequence_ref"].tolist())
    alt_preds = predict_batched(encode_and_predict, snv_df["sequence_alt"].tolist())
    return ref_preds, alt_preds


def eval_foundation_s1(result_dir, snv_df, encoder_name, cell):
    """Load foundation S1 (frozen encoder + head) and predict."""
    import torch

    ckpt_path = result_dir / "best_model.pt"
    if not ckpt_path.exists():
        for p in result_dir.rglob("best_model.pt"):
            ckpt_path = p
            break
    if not ckpt_path.exists():
        print(f"    No best_model.pt in {result_dir}")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoder
    if encoder_name == "enformer":
        from models.enformer_wrapper import EnformerWrapper

        encoder = EnformerWrapper()
    elif encoder_name == "borzoi":
        from models.borzoi_wrapper import BorzoiWrapper

        encoder = BorzoiWrapper()
    elif encoder_name == "ntv3_post":
        from models.nt_v3_wrapper import NTv3Wrapper

        encoder = NTv3Wrapper()
    else:
        print(f"    Unknown encoder: {encoder_name}")
        return None, None

    # Load head
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    head_state = ckpt.get("model_state_dict", ckpt)
    # Infer head architecture from state dict
    layers = []
    keys = sorted(head_state.keys())
    for k in keys:
        if "weight" in k and head_state[k].dim() == 2:
            in_f, out_f = head_state[k].shape[1], head_state[k].shape[0]
            layers.append((in_f, out_f))

    head_modules = []
    for i, (in_f, out_f) in enumerate(layers):
        head_modules.append(torch.nn.Linear(in_f, out_f))
        if i < len(layers) - 1:
            head_modules.append(torch.nn.ReLU())
            head_modules.append(torch.nn.Dropout(0.1))
    head = torch.nn.Sequential(*head_modules)

    # Strip "net." prefix if present
    cleaned = {k.removeprefix("net."): v for k, v in head_state.items()}
    head.load_state_dict(cleaned, strict=False)
    head.to(device).eval()

    def predict_fn(sequences):
        emb = encoder.extract_embeddings(sequences)
        emb_t = torch.tensor(emb, device=device, dtype=torch.float32)
        with torch.no_grad():
            return head(emb_t).cpu().numpy().reshape(-1)

    ref_preds = predict_batched(predict_fn, snv_df["sequence_ref"].tolist(), batch_size=4)
    alt_preds = predict_batched(predict_fn, snv_df["sequence_alt"].tolist(), batch_size=4)
    return ref_preds, alt_preds


def eval_ag(result_dir, snv_df, cell, stage=1):
    """Load AG S1 or S2 and predict on SNV sequences."""
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from alphagenome_ft import create_model_with_heads

    from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
    from models.alphagenome_heads import register_s2f_head
    from models.embedding_cache import reinit_head_params

    # Read config
    for jn in ("test_metrics.json", "result.json"):
        jp = result_dir / jn
        if jp.exists():
            rdata = json.loads(jp.read_text())
            break
    else:
        print(f"    No result JSON in {result_dir}")
        return None, None

    head_name = rdata.get("head_name", "alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4")
    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten-512-512",
        task_mode="human",
        num_tracks=1,
        dropout_rate=0.1,
    )

    weights_path = (
        "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"
    )
    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=(stage == 1),
    )

    # Load checkpoint
    ckpt_path = (result_dir / "best_model" / "checkpoint").resolve()
    if ckpt_path.exists():
        from collections.abc import Mapping

        checkpointer = ocp.PyTreeCheckpointer()
        restored = checkpointer.restore(str(ckpt_path), item=model._params)
        if isinstance(restored, Mapping):
            model._params = dict(restored)
        else:
            model._params = restored

    # Prediction
    flank_5 = MPRA_UPSTREAM[-200:]
    flank_3 = MPRA_DOWNSTREAM[:200]
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

    def encode_one(seq):
        if len(seq) < 200:
            pad = 200 - len(seq)
            seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
        elif len(seq) > 200:
            start = (len(seq) - 200) // 2
            seq = seq[start : start + 200]
        full = flank_5 + seq + flank_3
        oh = np.zeros((600, 5), dtype=np.float32)
        for i, c in enumerate(full):
            if c in mapping:
                oh[i, mapping[c]] = 1.0
        oh[:, 4] = 1.0
        return oh

    from experiments.train_oracle_alphagenome_hashfrag_cached import _predict_sequences

    def predict_fn(sequences):
        return _predict_sequences(
            lambda p, s, st, seqs, *a, **kw: model._predict(p, s, seqs, *a, **kw)[head_name],
            model._params,
            model._state,
            sequences,
        )

    # Actually, use the simpler approach via model.encode + head
    @jax.jit
    def predict_step(params, state, sequences):
        return model._predict(
            params,
            state,
            sequences,
            jnp.zeros(len(sequences), dtype=jnp.int32),
            negative_strand_mask=jnp.zeros(len(sequences), dtype=bool),
            strand_reindexing=None,
        )[head_name]

    def predict_batch(sequences):
        oh = np.stack([encode_one(s) for s in sequences])
        preds = predict_step(model._params, model._state, jnp.array(oh))
        return np.array(jnp.squeeze(preds)).reshape(-1)

    ref_preds = predict_batched(predict_batch, snv_df["sequence_ref"].tolist(), batch_size=64)
    alt_preds = predict_batched(predict_batch, snv_df["sequence_alt"].tolist(), batch_size=64)
    return ref_preds, alt_preds


# ── Main dispatch ─────────────────────────────────────────────────────────


# (cell, model_type, encoder_name, result_dirs, stage)
EVAL_TASKS = []
for cell in ["hepg2", "sknsh"]:
    EVAL_TASKS.extend(
        [
            (
                cell,
                "malinois",
                None,
                [
                    f"outputs/malinois_{cell}_3seeds/seed_0/seed_0",
                    f"outputs/malinois_{cell}_3seeds/seed_1/seed_1",
                    f"outputs/malinois_{cell}_3seeds/seed_2/seed_2",
                ],
                1,
            ),
            (
                cell,
                "foundation_s1",
                "enformer",
                [
                    f"outputs/enformer_{cell}_cached/seed_0/seed_0",
                    f"outputs/enformer_{cell}_cached/seed_1/seed_1",
                    f"outputs/enformer_{cell}_cached/seed_2/seed_2",
                ],
                1,
            ),
            (
                cell,
                "foundation_s1",
                "borzoi",
                [
                    f"outputs/borzoi_{cell}_cached/seed_0/seed_0",
                    f"outputs/borzoi_{cell}_cached/seed_1/seed_1",
                    f"outputs/borzoi_{cell}_cached/seed_2/seed_2",
                ],
                1,
            ),
            (
                cell,
                "foundation_s1",
                "ntv3_post",
                [
                    f"outputs/ntv3_post_{cell}_cached/seed_0/seed_0",
                    f"outputs/ntv3_post_{cell}_cached/seed_1/seed_1",
                    f"outputs/ntv3_post_{cell}_cached/seed_2/seed_2",
                ],
                1,
            ),
            (
                cell,
                "ag",
                None,
                [
                    f"outputs/ag_hashfrag_{cell}_cached/seed_0",
                    f"outputs/ag_hashfrag_{cell}_cached/seed_1",
                    f"outputs/ag_hashfrag_{cell}_cached/seed_2",
                ],
                1,
            ),
            (cell, "ag", None, [f"outputs/ag_{cell}_stage2/seed_0"], 2),
        ]
    )


def main():
    for cell, model_type, encoder_name, result_dirs, stage in EVAL_TASKS:
        fc_col = CELL_LINE_LABEL_COLS[cell]
        alt_col = f"{fc_col}_alt"
        delta_col = f"delta_{fc_col}"

        snv_path = REPO / "data" / cell / "test_sets" / "test_snv_pairs_hashfrag.tsv"
        if not snv_path.exists():
            print(f"SKIP {cell}: {snv_path} not found")
            continue
        snv_df = pd.read_csv(snv_path, sep="\t")
        if alt_col not in snv_df.columns:
            print(f"SKIP {cell}: {alt_col} not in SNV file")
            continue

        for rd_str in result_dirs:
            rd = REPO / rd_str
            if not rd.exists():
                continue

            # Find result JSON
            rj = None
            for jn in ("result.json", "test_metrics.json"):
                if (rd / jn).exists():
                    rj = rd / jn
                    break
            if rj is None:
                continue

            print(f"\n=== {model_type} {cell} stage={stage}: {rd_str} ===", flush=True)

            try:
                if model_type == "malinois":
                    ref_preds, alt_preds = eval_malinois(rd, snv_df, cell)
                elif model_type == "foundation_s1":
                    ref_preds, alt_preds = eval_foundation_s1(rd, snv_df, encoder_name, cell)
                elif model_type == "ag":
                    ref_preds, alt_preds = eval_ag(rd, snv_df, cell, stage=stage)
                else:
                    print(f"    Unknown model type: {model_type}")
                    continue

                if ref_preds is None:
                    continue

                # Compute metrics
                alt_true = snv_df[alt_col].to_numpy(dtype=np.float32)
                delta_true = snv_df[delta_col].to_numpy(dtype=np.float32)
                delta_pred = alt_preds - ref_preds

                snv_abs = _compute_metrics(alt_preds, alt_true)
                snv_delta = _compute_metrics(delta_pred, delta_true)

                print(f"    snv_abs: r={snv_abs['pearson_r']:.4f}")
                print(f"    snv_delta: r={snv_delta['pearson_r']:.4f}")

                patch_json(rj, snv_abs, snv_delta)

            except Exception as e:
                print(f"    ERROR: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    main()
