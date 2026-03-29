#!/usr/bin/env python
"""Build SNV embedding caches and re-evaluate SNV metrics using cached embeddings.

For foundation model S1 heads trained on cached (float16) embeddings, the SNV
re-evaluation must also use cached embeddings to avoid train/eval mismatch.
This script:

1. Loads the cell-specific SNV test file
2. Runs ref and alt sequences through the encoder (same pipeline as cache builder)
3. Saves embeddings as {prefix}_canonical.npy / {prefix}_rc.npy
4. Loads the trained MLP head
5. Computes RC-averaged predictions and SNV metrics
6. Patches the result.json with corrected metrics

Usage::

    python scripts/build_and_eval_snv_cache.py \
        --model borzoi \
        --cell-line hepg2 \
        --result-dir outputs/borzoi_hepg2_cached/seed_0/seed_0

    python scripts/build_and_eval_snv_cache.py \
        --model enformer \
        --cell-line sknsh \
        --result-dir outputs/enformer_sknsh_cached/seed_1/seed_1

    python scripts/build_and_eval_snv_cache.py \
        --model ntv3_post \
        --cell-line hepg2 \
        --result-dir outputs/ntv3_post_hepg2_cached/seed_0/seed_0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM  # noqa: E402
from data.utils import one_hot_encode  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
CELL_LINE_LABEL_COLS = {
    "k562": "K562_log2FC",
    "hepg2": "HepG2_log2FC",
    "sknsh": "SKNSH_log2FC",
}

MODEL_EMBED_DIMS = {
    "borzoi": 1536,
    "enformer": 3072,
    "ntv3_post": 1536,
}

# ── MPRA flanking sequences ──────────────────────────────────────────────────
_FLANK_5_STR = MPRA_UPSTREAM[-200:]
_FLANK_3_STR = MPRA_DOWNSTREAM[:200]

_MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}

_FLANK_5_ENC = np.zeros((4, 200), dtype=np.float32)
for _i, _c in enumerate(_FLANK_5_STR):
    if _c in _MAPPING:
        _FLANK_5_ENC[_MAPPING[_c], _i] = 1.0

_FLANK_3_ENC = np.zeros((4, 200), dtype=np.float32)
for _i, _c in enumerate(_FLANK_3_STR):
    if _c in _MAPPING:
        _FLANK_3_ENC[_MAPPING[_c], _i] = 1.0

# Reverse complement helper
_RC_MAP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def _reverse_complement(seq: str) -> str:
    return seq.translate(_RC_MAP)[::-1]


def _standardize_to_200bp(sequence: str) -> str:
    target_len = 200
    curr_len = len(sequence)
    if curr_len == target_len:
        return sequence
    if curr_len < target_len:
        pad = target_len - curr_len
        return "N" * (pad // 2) + sequence + "N" * (pad - pad // 2)
    start = (curr_len - target_len) // 2
    return sequence[start : start + target_len]


def _seq_to_one_hot_600(seq: str) -> np.ndarray:
    """Convert sequence to (4, 600) one-hot with MPRA flanks."""
    core = _standardize_to_200bp(seq)
    oh = one_hot_encode(core, add_singleton_channel=False)  # (4, 200)
    return np.concatenate([_FLANK_5_ENC, oh, _FLANK_3_ENC], axis=1)  # (4, 600)


def _make_600bp_str(seq: str) -> str:
    """Pad to 200bp then add MPRA flanks for 600bp total (for NTv3)."""
    core = _standardize_to_200bp(seq)
    return _FLANK_5_STR + core + _FLANK_3_STR


# ── Encoding functions per model ──────────────────────────────────────────────


def _encode_borzoi(
    model,
    sequences: list[str],
    cache_dir: Path,
    prefix: str,
    device: torch.device,
    batch_size: int = 2,
    dtype: np.dtype = np.float16,
) -> None:
    """Encode sequences with Borzoi (same as build_borzoi_embedding_cache.py)."""
    BORZOI_SEQ_LEN = 196_608

    can_path = cache_dir / f"{prefix}_canonical.npy"
    rc_path = cache_dir / f"{prefix}_rc.npy"

    if can_path.exists() and rc_path.exists():
        print(f"  {prefix}: cache already exists -- skipping.")
        return

    N = len(sequences)
    D = 1536
    cache_dir.mkdir(parents=True, exist_ok=True)

    can_buf = np.lib.format.open_memmap(can_path, mode="w+", dtype=dtype, shape=(N, D))
    rc_buf = np.lib.format.open_memmap(rc_path, mode="w+", dtype=dtype, shape=(N, D))

    for i in tqdm(range(0, N, batch_size), desc=f"  {prefix}"):
        end = min(i + batch_size, N)
        batch_seqs = sequences[i:end]

        oh_batch = np.stack([_seq_to_one_hot_600(s) for s in batch_seqs])  # (B, 4, 600)
        oh_rc = oh_batch[:, ::-1, ::-1].copy()

        can_t = torch.from_numpy(oh_batch).float()
        rc_t = torch.from_numpy(oh_rc).float()

        pad_total = BORZOI_SEQ_LEN - 600
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        can_padded = torch.nn.functional.pad(can_t, (pad_left, pad_right), value=0.0)
        rc_padded = torch.nn.functional.pad(rc_t, (pad_left, pad_right), value=0.0)

        can_padded = can_padded.to(device)
        rc_padded = rc_padded.to(device)

        for attempt, use_autocast in enumerate([device.type == "cuda", False]):
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_autocast):
                emb_can = model.get_embs_after_crop(can_padded)  # (B, 1536, 6144)
                emb_rc = model.get_embs_after_crop(rc_padded)

            emb_can = emb_can.mean(dim=2)  # (B, 1536)
            emb_rc = emb_rc.mean(dim=2)

            has_nan = torch.isnan(emb_can).any() or torch.isnan(emb_rc).any()
            if not has_nan:
                break
            if attempt == 0:
                print(f"    WARNING: NaN in batch {i}-{end}, retrying without autocast...")

        emb_can_np = emb_can.cpu().float().numpy()
        emb_rc_np = emb_rc.cpu().float().numpy()

        np.nan_to_num(emb_can_np, copy=False)
        np.nan_to_num(emb_rc_np, copy=False)

        if dtype == np.float16:
            can_buf[i:end] = np.clip(emb_can_np, -65504, 65504).astype(dtype)
            rc_buf[i:end] = np.clip(emb_rc_np, -65504, 65504).astype(dtype)
        else:
            can_buf[i:end] = emb_can_np.astype(dtype)
            rc_buf[i:end] = emb_rc_np.astype(dtype)

    can_buf.flush()
    rc_buf.flush()
    del can_buf, rc_buf
    print(f"  {prefix}: saved {N} embeddings ({D}D) -> {can_path.name}, {rc_path.name}")


def _encode_enformer(
    model,
    sequences: list[str],
    cache_dir: Path,
    prefix: str,
    device: torch.device,
    batch_size: int = 4,
    n_center_bins: int = 4,
    dtype: np.dtype = np.float16,
) -> None:
    """Encode sequences with Enformer (same as build_enformer_embedding_cache.py)."""
    ENFORMER_SEQ_LEN = 196_608
    ENFORMER_TARGET_LEN = 896

    can_path = cache_dir / f"{prefix}_canonical.npy"
    rc_path = cache_dir / f"{prefix}_rc.npy"

    if can_path.exists() and rc_path.exists():
        print(f"  {prefix}: cache already exists -- skipping.")
        return

    N = len(sequences)
    D = 3072
    cache_dir.mkdir(parents=True, exist_ok=True)

    can_buf = np.lib.format.open_memmap(can_path, mode="w+", dtype=dtype, shape=(N, D))
    rc_buf = np.lib.format.open_memmap(rc_path, mode="w+", dtype=dtype, shape=(N, D))

    center = ENFORMER_TARGET_LEN // 2  # 448
    half = n_center_bins // 2

    for i in tqdm(range(0, N, batch_size), desc=f"  {prefix}"):
        end = min(i + batch_size, N)
        batch_seqs = sequences[i:end]

        oh_batch = np.stack([_seq_to_one_hot_600(s) for s in batch_seqs])  # (B, 4, 600)
        oh_rc = oh_batch[:, ::-1, ::-1].copy()

        # Enformer: channels-last (B, L, 4)
        can_cl = torch.from_numpy(oh_batch.transpose(0, 2, 1)).float()
        rc_cl = torch.from_numpy(oh_rc.transpose(0, 2, 1)).float()

        pad_total = ENFORMER_SEQ_LEN - 600
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        can_padded = torch.nn.functional.pad(can_cl, (0, 0, pad_left, pad_right), value=0.0)
        rc_padded = torch.nn.functional.pad(rc_cl, (0, 0, pad_left, pad_right), value=0.0)

        can_padded = can_padded.to(device)
        rc_padded = rc_padded.to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            emb_can = model(can_padded, return_only_embeddings=True)  # (B, 896, 3072)
            emb_rc = model(rc_padded, return_only_embeddings=True)

        emb_can = emb_can[:, center - half : center + half, :].mean(dim=1)  # (B, 3072)
        emb_rc = emb_rc[:, center - half : center + half, :].mean(dim=1)

        emb_can_np = emb_can.cpu().float().numpy()
        emb_rc_np = emb_rc.cpu().float().numpy()

        if dtype == np.float16:
            can_buf[i:end] = np.clip(emb_can_np, -65504, 65504).astype(dtype)
            rc_buf[i:end] = np.clip(emb_rc_np, -65504, 65504).astype(dtype)
        else:
            can_buf[i:end] = emb_can_np.astype(dtype)
            rc_buf[i:end] = emb_rc_np.astype(dtype)

    can_buf.flush()
    rc_buf.flush()
    del can_buf, rc_buf
    print(f"  {prefix}: saved {N} embeddings ({D}D) -> {can_path.name}, {rc_path.name}")


def _encode_ntv3(
    ntv3_wrapper,
    sequences: list[str],
    cache_dir: Path,
    prefix: str,
    batch_size: int = 32,
    dtype: np.dtype = np.float16,
) -> None:
    """Encode sequences with NTv3 (same as build_ntv3_embedding_cache.py)."""
    can_path = cache_dir / f"{prefix}_canonical.npy"
    rc_path = cache_dir / f"{prefix}_rc.npy"

    if can_path.exists() and rc_path.exists():
        print(f"  {prefix}: cache already exists -- skipping.")
        return

    N = len(sequences)
    D = ntv3_wrapper.embed_dim
    cache_dir.mkdir(parents=True, exist_ok=True)

    can_buf = np.lib.format.open_memmap(can_path, mode="w+", dtype=dtype, shape=(N, D))
    rc_buf = np.lib.format.open_memmap(rc_path, mode="w+", dtype=dtype, shape=(N, D))

    for i in tqdm(range(0, N, batch_size), desc=f"  {prefix}"):
        batch_seqs = sequences[i : i + batch_size]

        can_seqs = [_make_600bp_str(s) for s in batch_seqs]
        rc_seqs = [_reverse_complement(_make_600bp_str(s)) for s in batch_seqs]

        emb_can = ntv3_wrapper.extract_embeddings(can_seqs)  # (B, D) float32
        emb_rc = ntv3_wrapper.extract_embeddings(rc_seqs)

        end = min(i + batch_size, N)
        if dtype == np.float16:
            can_buf[i:end] = np.clip(emb_can, -65504, 65504).astype(dtype)
            rc_buf[i:end] = np.clip(emb_rc, -65504, 65504).astype(dtype)
        else:
            can_buf[i:end] = emb_can.astype(dtype)
            rc_buf[i:end] = emb_rc.astype(dtype)

    can_buf.flush()
    rc_buf.flush()
    del can_buf, rc_buf
    print(f"  {prefix}: saved {N} embeddings ({D}D) -> {can_path.name}, {rc_path.name}")


# ── Evaluation ────────────────────────────────────────────────────────────────


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


def predict_from_cache(
    head: torch.nn.Module,
    cache_dir: Path,
    prefix: str,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Load cached embeddings and compute RC-averaged predictions."""
    can = np.load(cache_dir / f"{prefix}_canonical.npy", mmap_mode="r")
    rc = np.load(cache_dir / f"{prefix}_rc.npy", mmap_mode="r")
    preds = []
    head.eval()
    with torch.no_grad():
        for i in range(0, len(can), batch_size):
            end = min(i + batch_size, len(can))
            can_t = torch.from_numpy(can[i:end].astype(np.float32)).to(device)
            rc_t = torch.from_numpy(rc[i:end].astype(np.float32)).to(device)
            p_can = head(can_t)
            p_rc = head(rc_t)
            preds.append(((p_can + p_rc) / 2.0).cpu().numpy())
    return np.concatenate(preds)


def load_mlp_head(result_dir: Path, embed_dim: int, device: torch.device) -> torch.nn.Module:
    """Load trained MLPHead from checkpoint."""
    from experiments.train_foundation_cached import MLPHead

    ckpt_path = result_dir / "best_model.pt"
    if not ckpt_path.exists():
        # Search subdirectories
        for p in result_dir.rglob("best_model.pt"):
            ckpt_path = p
            break
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best_model.pt in {result_dir}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)

    head = MLPHead(embed_dim=embed_dim, hidden_dim=512, dropout=0.1)
    head.load_state_dict(state_dict)
    head.to(device).eval()

    print(f"  Loaded head from {ckpt_path}")
    return head


def patch_result_json(result_dir: Path, snv_abs_metrics: dict, snv_delta_metrics: dict) -> None:
    """Patch SNV metrics into the result JSON."""
    rj = None
    for jn in ("result.json", "test_metrics.json"):
        if (result_dir / jn).exists():
            rj = result_dir / jn
            break
    if rj is None:
        print(f"  WARNING: No result JSON found in {result_dir}")
        return

    data = json.loads(rj.read_text())
    container = data.get("test_metrics", data)
    if snv_abs_metrics:
        container["snv_abs"] = snv_abs_metrics
    if snv_delta_metrics:
        container["snv_delta"] = snv_delta_metrics
    rj.write_text(json.dumps(data, indent=2, default=str) + "\n")
    print(f"  Patched {rj}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=["borzoi", "enformer", "ntv3_post"],
        help="Foundation model to use for encoding.",
    )
    parser.add_argument(
        "--cell-line",
        required=True,
        choices=["hepg2", "sknsh", "k562"],
        help="Cell line (determines which SNV labels to use).",
    )
    parser.add_argument(
        "--result-dir",
        required=True,
        type=Path,
        help="Path to trained S1 model directory (contains best_model.pt and result.json).",
    )
    parser.add_argument(
        "--snv-cache-dir",
        type=Path,
        default=None,
        help="Override SNV cache directory. Default: same as the model's embedding cache.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Encoding batch size.")
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip cache building (assume caches already exist).",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation (only build caches).",
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    model_name = args.model
    cell_line = args.cell_line
    result_dir = args.result_dir.resolve()
    embed_dim = MODEL_EMBED_DIMS[model_name]

    # Determine cache directory for SNV embeddings
    if args.snv_cache_dir is not None:
        snv_cache_dir = args.snv_cache_dir.resolve()
    else:
        # Use the model's embedding cache dir (same as train/val caches)
        # The K562 cache is shared across cell lines (same sequences)
        if model_name == "ntv3_post":
            snv_cache_dir = REPO / "outputs" / "ntv3_post_k562_cached" / "embedding_cache"
        else:
            snv_cache_dir = REPO / "outputs" / f"{model_name}_k562_cached" / "embedding_cache"

    fc_col = CELL_LINE_LABEL_COLS[cell_line]
    alt_col = f"{fc_col}_alt"
    delta_col = f"delta_{fc_col}"

    # Load SNV test file (cell-specific: may have fewer rows due to NaN filtering)
    snv_path = REPO / "data" / cell_line / "test_sets" / "test_snv_pairs_hashfrag.tsv"
    if not snv_path.exists():
        # Fall back to K562 SNV file
        snv_path = REPO / "data" / "k562" / "test_sets" / "test_snv_pairs_hashfrag.tsv"
    if not snv_path.exists():
        print(f"ERROR: No SNV test file found for {cell_line}")
        sys.exit(1)

    snv_df = pd.read_csv(snv_path, sep="\t")
    print(f"SNV test file: {snv_path} ({len(snv_df)} pairs)")

    ref_sequences = snv_df["sequence_ref"].tolist()
    alt_sequences = snv_df["sequence_alt"].tolist()

    # Use cell-line-specific cache prefix to avoid collisions when cell lines
    # have different SNV subsets (NaN filtering)
    cache_prefix_ref = f"test_snv_ref_{cell_line}"
    cache_prefix_alt = f"test_snv_alt_{cell_line}"

    # ── Phase 1: Build SNV caches ─────────────────────────────────────────────
    if not args.skip_cache:
        print(f"\n=== Building SNV caches for {model_name} ({cell_line}) ===")
        print(f"  Cache dir: {snv_cache_dir}")
        print(f"  Ref sequences: {len(ref_sequences):,}")
        print(f"  Alt sequences: {len(alt_sequences):,}")

        if model_name == "borzoi":
            from borzoi_pytorch import Borzoi
            from borzoi_pytorch.modeling_borzoi import Borzoi as BorzoiModel

            if not hasattr(BorzoiModel, "all_tied_weights_keys"):
                BorzoiModel.all_tied_weights_keys = {}

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Loading Borzoi replicate 0 on {device}...")
            encoder = Borzoi.from_pretrained("johahi/borzoi-replicate-0")
            encoder.eval().to(device)
            for p in encoder.parameters():
                p.requires_grad = False

            bs = args.batch_size or 2
            _encode_borzoi(encoder, ref_sequences, snv_cache_dir, cache_prefix_ref, device, bs)
            _encode_borzoi(encoder, alt_sequences, snv_cache_dir, cache_prefix_alt, device, bs)
            del encoder
            torch.cuda.empty_cache()

        elif model_name == "enformer":
            from enformer_pytorch import Enformer

            if not hasattr(Enformer, "all_tied_weights_keys"):
                Enformer.all_tied_weights_keys = {}

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Loading Enformer on {device}...")
            encoder = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
            encoder.eval().to(device)
            for p in encoder.parameters():
                p.requires_grad = False

            bs = args.batch_size or 4
            _encode_enformer(encoder, ref_sequences, snv_cache_dir, cache_prefix_ref, device, bs)
            _encode_enformer(encoder, alt_sequences, snv_cache_dir, cache_prefix_alt, device, bs)
            del encoder
            torch.cuda.empty_cache()

        elif model_name == "ntv3_post":
            from models.nt_v3_wrapper import NTv3Wrapper

            print("Loading NTv3 650M post-trained...")
            encoder = NTv3Wrapper(
                model_name="NTv3_650M_post",
                model_variant="post",
                use_bfloat16=True,
            )

            bs = args.batch_size or 32
            _encode_ntv3(encoder, ref_sequences, snv_cache_dir, cache_prefix_ref, bs)
            _encode_ntv3(encoder, alt_sequences, snv_cache_dir, cache_prefix_alt, bs)
            del encoder

        print("SNV cache build complete.")

    # ── Phase 2: Evaluate ─────────────────────────────────────────────────────
    if not args.skip_eval:
        print(f"\n=== Evaluating SNV metrics for {model_name} ({cell_line}) ===")
        print(f"  Result dir: {result_dir}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        head = load_mlp_head(result_dir, embed_dim, device)

        ref_pred = predict_from_cache(head, snv_cache_dir, cache_prefix_ref, device)
        alt_pred = predict_from_cache(head, snv_cache_dir, cache_prefix_alt, device)

        # Compute metrics
        if alt_col not in snv_df.columns:
            print(f"  WARNING: {alt_col} not in SNV file, using K562_log2FC_alt as fallback")
            alt_col = "K562_log2FC_alt"
        if delta_col not in snv_df.columns:
            delta_col = "delta_log2FC"

        alt_true = snv_df[alt_col].to_numpy(dtype=np.float32)
        delta_true = snv_df[delta_col].to_numpy(dtype=np.float32)
        delta_pred = alt_pred - ref_pred

        snv_abs = _compute_metrics(alt_pred, alt_true)
        snv_delta = _compute_metrics(delta_pred, delta_true)

        print(
            f"  snv_abs:   pearson_r={snv_abs['pearson_r']:.4f}, "
            f"spearman_r={snv_abs['spearman_r']:.4f}, mse={snv_abs['mse']:.4f}"
        )
        print(
            f"  snv_delta: pearson_r={snv_delta['pearson_r']:.4f}, "
            f"spearman_r={snv_delta['spearman_r']:.4f}, mse={snv_delta['mse']:.4f}"
        )

        patch_result_json(result_dir, snv_abs, snv_delta)

    print("\nDone!")


if __name__ == "__main__":
    main()
