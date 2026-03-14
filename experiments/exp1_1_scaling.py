#!/usr/bin/env python
"""Experiment 1.1: Reservoir sampling scaling laws.

For each (reservoir strategy x training size), generates sequences using the
reservoir, labels them with the oracle, trains a student with HP sweep, and
evaluates on the oracle-labeled test panel.

Usage::

    # Single run
    uv run python experiments/exp1_1_scaling.py \\
        --task yeast --student dream_rnn --reservoir random

    # All Phase 1 reservoirs
    uv run python experiments/exp1_1_scaling.py \\
        --task yeast --student dream_rnn \\
        --reservoir random genomic prm_5pct

    # Override training sizes
    uv run python experiments/exp1_1_scaling.py \\
        --task k562 --student dream_rnn --reservoir random \\
        --training-sizes 1000 5000 10000

    # Quick smoke test (1 HP config, 1 replicate, 1 size)
    uv run python experiments/exp1_1_scaling.py \\
        --task yeast --student dream_rnn --reservoir random \\
        --training-sizes 1000 --n-replicates 1 --no-hp-sweep
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from albench.model import SequenceModel  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TRAINING_SIZES = [1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]

HP_GRIDS = {
    "dream_rnn": {
        "learning_rate": [0.003, 0.005],
        "batch_size": [512, 1024],
    },
    "alphagenome_k562_s1": {
        "learning_rate": [3e-4, 1e-3],
        "batch_size": [128, 256],
    },
    "alphagenome_yeast_s1": {
        "learning_rate": [3e-4, 1e-3],
        "batch_size": [128, 256],
    },
    # S2: joint encoder + head fine-tuning (encoder_lr is the encoder LR;
    # head LR is fixed at 1e-3 based on Exp 0 s2c results)
    "alphagenome_k562_s2": {
        "learning_rate": [1e-4],  # encoder LR (head_lr=1e-3 fixed)
        "batch_size": [128],
    },
    "alphagenome_yeast_s2": {
        "learning_rate": [1e-4, 5e-5],
        "batch_size": [64, 128],
    },
}

# For large training sizes (n >= LARGE_N_THRESHOLD), use only the fastest HP
# configs to stay within SLURM time limits. These keep the largest batch size
# from the full grid so epoch time is minimized, while still covering the LR
# range.  Falls back to full HP_GRIDS when the student isn't listed here.
LARGE_N_THRESHOLD = 100_000

HP_GRIDS_LARGE_N = {
    "dream_rnn": {
        "learning_rate": [0.003, 0.005],
        "batch_size": [1024],  # drop bs=512 — 2× fewer batches/epoch
    },
    "alphagenome_k562_s1": {
        "learning_rate": [3e-4, 1e-3],
        "batch_size": [256],  # drop bs=128
    },
    "alphagenome_yeast_s1": {
        "learning_rate": [3e-4, 1e-3],
        "batch_size": [256],
    },
}

# S2 config: which encoder blocks to unfreeze (from Exp 0 best: blocks 4,5)
S2_CONFIG = {
    "k562": {
        "unfreeze_blocks": [4, 5],
        "head_lr": 1e-3,
        "weight_decay": 1e-6,
        "max_shift": 15,
        "epochs": 30,
        "early_stop_patience": 7,
        "warmup_epochs": 3,  # head-only warmup before unfreezing encoder
    },
    "yeast": {
        "unfreeze_blocks": [4, 5],
        "head_lr": 1e-3,
        "weight_decay": 1e-6,
        "max_shift": 0,  # no shift aug for yeast (80bp is the full region)
        "epochs": 30,
        "early_stop_patience": 7,
        "warmup_epochs": 3,
    },
}

TASK_CONFIGS = {
    "k562": {
        "sequence_length": 200,
        "input_channels": 5,
        "task_mode": "k562",
        "random_region_length": 200,
        "data_root": "data/k562",
        "test_set_dir": "data/k562/test_sets",
    },
    "yeast": {
        "sequence_length": 150,
        "input_channels": 6,
        "task_mode": "yeast",
        "random_region_length": 80,
        "data_root": "data/yeast",
        "test_set_dir": "data/yeast/test_sets",
    },
}


@dataclass
class RunResult:
    """Result of one (reservoir, size, HP, seed) training run."""

    reservoir: str
    task: str
    student: str
    n_train: int
    hp_config: dict
    seed: int
    val_pearson_r: float
    test_metrics: dict
    wall_seconds: float
    output_dir: str


# ---------------------------------------------------------------------------
# Reservoir instantiation
# ---------------------------------------------------------------------------


def _load_reservoir(name: str, seed: int):
    """Load a reservoir sampler config and instantiate."""
    cfg_path = REPO / "configs" / "reservoir" / f"{name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No reservoir config: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())

    target = cfg.pop("_target_")
    # Replace ${seed} placeholder
    for k, v in cfg.items():
        if v == "${seed}":
            cfg[k] = seed

    # Import and instantiate
    parts = target.rsplit(".", 1)
    mod = __import__(parts[0], fromlist=[parts[1]])
    cls = getattr(mod, parts[1])
    return cls(**cfg)


def _load_pool_sequences(task: str) -> tuple[list[str], np.ndarray | None]:
    """Load genomic pool sequences for the task."""
    if task == "k562":
        from data.k562 import K562Dataset

        ds = K562Dataset(data_path=str(REPO / "data" / "k562"), split="train")
        return list(ds.sequences), ds.labels.astype(np.float32)
    else:
        from data.yeast import YeastDataset

        ds = YeastDataset(
            data_path=str(REPO / "data" / "yeast"),
            split="train",
            context_mode="dream150",
        )
        return list(ds.sequences), ds.labels.astype(np.float32)


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------


def _load_oracle(task: str, oracle_type: str = "default") -> SequenceModel:
    """Load the oracle ensemble for labeling.

    Args:
        task: ``"k562"`` or ``"yeast"``.
        oracle_type: ``"default"``, ``"ag"``, or ``"dream_rnn"``.
            - ``"default"``: AG oracle for K562, DREAM-RNN for yeast.
            - ``"ag"``: AlphaGenome oracle (K562 only).
            - ``"dream_rnn"``: DREAM-RNN oracle.
    """
    if oracle_type == "default":
        oracle_type = "ag" if task == "k562" else "dream_rnn"

    loaders = {
        ("k562", "ag"): _load_k562_ag_oracle,
        ("k562", "dream_rnn"): _load_k562_dream_oracle,
        ("yeast", "dream_rnn"): _load_yeast_dream_oracle,
        ("yeast", "ag"): _load_yeast_ag_oracle,
    }

    key = (task, oracle_type)
    if key not in loaders:
        raise ValueError(
            f"No oracle available for task={task}, oracle_type={oracle_type}. "
            f"Available: {list(loaders.keys())}"
        )

    try:
        return loaders[key]()
    except Exception as e:
        raise RuntimeError(f"Failed to load {oracle_type} oracle for {task}: {e}") from e


def _load_k562_ag_oracle():
    """Load K562 AlphaGenome 10-fold oracle."""
    from collections.abc import Mapping

    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from alphagenome_ft import create_model_with_heads

    from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
    from models.alphagenome_heads import register_s2f_head

    oracle_dir = REPO / "outputs" / "ag_hashfrag_oracle_cached"
    if not oracle_dir.exists():
        oracle_dir = REPO / "outputs" / "stage2_k562_full_train"

    ckpt_paths = sorted(
        [
            p / "best_model" / "checkpoint"
            for p in sorted(oracle_dir.glob("oracle_*"))
            if (p / "best_model" / "checkpoint").exists()
        ]
    )
    if not ckpt_paths:
        raise FileNotFoundError(f"No AG oracle checkpoints in {oracle_dir}")

    flank_5 = MPRA_UPSTREAM[-200:]
    flank_3 = MPRA_DOWNSTREAM[:200]
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

    def _encode_flanks(flank: str) -> np.ndarray:
        enc = np.zeros((200, 4), dtype=np.float32)
        for i, c in enumerate(flank):
            if c in mapping:
                enc[i, mapping[c]] = 1.0
        return enc

    f5_enc = _encode_flanks(flank_5)
    f3_enc = _encode_flanks(flank_3)

    head_name = "alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4"
    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten-512-512",
        task_mode="human",
        num_tracks=1,
        dropout_rate=0.1,
    )
    weights_path = os.environ.get(
        "ALPHAGENOME_WEIGHTS",
        "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1",
    )
    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=True,
    )

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

    def _merge(base, override):
        if not isinstance(override, Mapping):
            return override
        if not isinstance(base, Mapping):
            return override
        merged = dict(base)
        for k, v in override.items():
            if k in merged and isinstance(merged[k], Mapping) and isinstance(v, Mapping):
                merged[k] = _merge(merged[k], v)
            else:
                merged[k] = v
        return merged

    checkpointer = ocp.StandardCheckpointer()
    params_list = []
    for ckpt_path in ckpt_paths:
        loaded_params, _ = checkpointer.restore(ckpt_path)
        model._params = jax.device_put(_merge(model._params, loaded_params))
        params_list.append(jax.device_put(model._params))
    model_state = model._state

    class _AGOracle(SequenceModel):
        def predict(self, sequences: list[str]) -> np.ndarray:
            n = len(sequences)
            x_fwd = np.stack([self._encode(s) for s in sequences])
            x_rev = x_fwd[:, ::-1, ::-1]
            all_p = []
            for params in params_list:
                pf, pr = [], []
                for i in range(0, n, 128):
                    cf = jnp.array(x_fwd[i : i + 128])
                    cr = jnp.array(x_rev[i : i + 128])
                    pf.append(np.array(predict_step(params, model_state, cf)).reshape(-1))
                    pr.append(np.array(predict_step(params, model_state, cr)).reshape(-1))
                all_p.append((np.concatenate(pf) + np.concatenate(pr)) / 2.0)
            return np.stack(all_p).mean(axis=0).astype(np.float32)

        def _encode(self, seq: str) -> np.ndarray:
            seq = seq.upper()
            if len(seq) < 200:
                pad = 200 - len(seq)
                seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
            elif len(seq) > 200:
                start = (len(seq) - 200) // 2
                seq = seq[start : start + 200]
            core = np.zeros((200, 4), dtype=np.float32)
            for i, c in enumerate(seq):
                if c in mapping:
                    core[i, mapping[c]] = 1.0
            return np.concatenate([f5_enc, core, f3_enc], axis=0)

    logger.info(f"Loaded K562 AG oracle with {len(params_list)} folds")
    return _AGOracle()


def _load_k562_dream_oracle():
    """Load K562 DREAM-RNN oracle ensemble."""
    import torch

    from data.utils import one_hot_encode
    from models.dream_rnn import create_dream_rnn

    oracle_dir = REPO / "outputs" / "oracle_dream_rnn_k562_ensemble"
    runs = []
    for run_dir in sorted(oracle_dir.glob("oracle_*")):
        best = run_dir / "best_model.pt"
        last = run_dir / "last_model.pt"
        if best.exists():
            runs.append(best)
        elif last.exists():
            runs.append(last)

    if not runs:
        raise FileNotFoundError(f"No K562 DREAM-RNN oracle checkpoints in {oracle_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for ckpt_path in runs:
        try:
            state = torch.load(ckpt_path, map_location="cpu")
        except Exception:
            # Try fallback (last_model.pt if best was corrupted, or vice versa)
            alt = ckpt_path.parent / (
                "last_model.pt" if ckpt_path.name == "best_model.pt" else "best_model.pt"
            )
            if alt.exists():
                try:
                    state = torch.load(alt, map_location="cpu")
                    logger.warning(f"Loaded fallback {alt.name} for {ckpt_path.parent.name}")
                except Exception:
                    logger.warning(f"Skipping corrupt checkpoint: {ckpt_path.parent.name}")
                    continue
            else:
                logger.warning(f"Skipping corrupt checkpoint: {ckpt_path.parent.name}")
                continue
        m = create_dream_rnn(
            input_channels=5,
            sequence_length=200,
            task_mode="k562",
            hidden_dim=320,
            cnn_filters=160,
            dropout_cnn=0.1,
            dropout_lstm=0.1,
        )
        m.load_state_dict(state["model_state_dict"], strict=True)
        m.to(device).eval()
        models.append(m)

    def _encode_k562(seq: str) -> np.ndarray:
        seq = seq.upper()
        if len(seq) < 200:
            pad = 200 - len(seq)
            seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
        elif len(seq) > 200:
            start = (len(seq) - 200) // 2
            seq = seq[start : start + 200]
        oh = one_hot_encode(seq, add_singleton_channel=False)
        rc = np.zeros((1, oh.shape[1]), dtype=np.float32)
        return np.concatenate([oh, rc], axis=0)

    class _DREAMOracleK562(SequenceModel):
        def predict(self, sequences: list[str], batch_size: int = 512) -> np.ndarray:
            encoded = np.stack([_encode_k562(s) for s in sequences])
            all_preds = []
            for m in models:
                fold_preds = []
                for i in range(0, len(encoded), batch_size):
                    batch = torch.from_numpy(encoded[i : i + batch_size]).float().to(device)
                    with torch.no_grad():
                        p = m.predict(batch, use_reverse_complement=True)
                        fold_preds.append(p.cpu().numpy().reshape(-1))
                all_preds.append(np.concatenate(fold_preds))
            return np.stack(all_preds).mean(axis=0).astype(np.float32)

    logger.info(f"Loaded K562 DREAM-RNN oracle with {len(models)} folds")
    return _DREAMOracleK562()


def _load_yeast_dream_oracle():
    """Load yeast DREAM-RNN 10-fold oracle."""
    import torch

    from models.dream_rnn import create_dream_rnn

    oracle_dir = REPO / "outputs" / "oracle_dream_rnn_yeast_kfold_v2"
    runs = []
    for run_dir in sorted(oracle_dir.glob("oracle_*")):
        ckpt = run_dir / "best_model.pt"
        if ckpt.exists():
            runs.append(ckpt)

    if not runs:
        raise FileNotFoundError(f"No yeast DREAM oracle checkpoints in {oracle_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for ckpt_path in runs:
        m = create_dream_rnn(
            input_channels=6,
            sequence_length=150,
            task_mode="yeast",
            cnn_filters=256,
            hidden_dim=320,
        )
        state = torch.load(ckpt_path, map_location="cpu")
        m.load_state_dict(state["model_state_dict"], strict=True)
        m.to(device).eval()
        models.append(m)

    from data.utils import one_hot_encode

    class _DREAMOracle(SequenceModel):
        def predict(self, sequences: list[str], batch_size: int = 512) -> np.ndarray:
            encoded = []
            for seq in sequences:
                base = one_hot_encode(seq, add_singleton_channel=False)
                rc = np.zeros((1, len(seq)), dtype=np.float32)
                singleton = np.zeros((1, len(seq)), dtype=np.float32)
                encoded.append(np.concatenate([base, rc, singleton], axis=0))
            encoded = np.stack(encoded)

            all_preds = []
            for m in models:
                fold_preds = []
                for i in range(0, len(encoded), batch_size):
                    batch = torch.from_numpy(encoded[i : i + batch_size]).float().to(device)
                    with torch.no_grad():
                        p = m.predict(batch, use_reverse_complement=True)
                        fold_preds.append(p.cpu().numpy().reshape(-1))
                all_preds.append(np.concatenate(fold_preds))
            return np.stack(all_preds).mean(axis=0).astype(np.float32)

    logger.info(f"Loaded yeast DREAM oracle with {len(models)} folds")
    return _DREAMOracle()


def _load_yeast_ag_oracle():
    """Load yeast AlphaGenome oracle (S1 head-only, cached)."""
    import jax
    import jax.numpy as jnp
    from alphagenome_ft import create_model_with_heads

    from models.alphagenome_heads import register_s2f_head
    from models.embedding_cache import (
        build_encoder_fn,
        build_head_only_predict_fn,
        reinit_head_params,
    )

    oracle_dir = REPO / "outputs" / "oracle_alphagenome_yeast_ensemble"
    ckpt_dirs = sorted([d for d in oracle_dir.glob("oracle_*") if (d / "best_model").exists()])
    if not ckpt_dirs:
        raise FileNotFoundError(f"No yeast AG oracle checkpoints in {oracle_dir}")

    head_name = "ag_yeast_boda_flatten_512_512_v4"
    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten-512-512",
        task_mode="yeast",
        num_tracks=18,
        dropout_rate=0.1,
    )
    weights_path = os.environ.get(
        "ALPHAGENOME_WEIGHTS",
        "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1",
    )
    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=True,
    )
    reinit_head_params(model, head_name, num_tokens=3, dim=1536)

    encoder_fn = build_encoder_fn(model)
    head_predict_fn = build_head_only_predict_fn(model, head_name)

    # Load each oracle fold's head params
    from collections.abc import Mapping

    import orbax.checkpoint as ocp

    def _merge(base, override):
        if not isinstance(override, Mapping) or not isinstance(base, Mapping):
            return override
        merged = dict(base)
        for k, v in override.items():
            if k in merged and isinstance(merged[k], Mapping) and isinstance(v, Mapping):
                merged[k] = _merge(merged[k], v)
            else:
                merged[k] = v
        return merged

    checkpointer = ocp.StandardCheckpointer()
    params_list = []
    for ckpt_dir in ckpt_dirs:
        loaded_params, _ = checkpointer.restore(ckpt_dir / "best_model" / "checkpoint")
        model._params = jax.device_put(_merge(model._params, loaded_params))
        params_list.append(jax.device_put(model._params))

    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    yeast_f5 = "GCTAGCGCCGATATCCTAACGAAGTCACTACTACGTACTGCCCTGCACGATAGC"
    yeast_f3 = (
        "CCTGCAGCAGACGTCGACACGCGTCGTAAAGTGACGTTGTCCGAAACCCTT"
        "GCATTCGACACCAAACATTCTCTCAGTGCGTGCCCATGAAC"
    )

    def _encode_yeast(seq: str) -> np.ndarray:
        seq = seq.upper()[:150] if len(seq) >= 150 else seq.upper()
        full = yeast_f5 + seq + yeast_f3
        out = np.zeros((384, 4), dtype=np.float32)
        start = max(0, (384 - len(full)) // 2)
        for i, c in enumerate(full):
            if i + start < 384 and c in mapping:
                out[i + start, mapping[c]] = 1.0
        return out

    class _AGOracleYeast(SequenceModel):
        def predict(self, sequences: list[str]) -> np.ndarray:
            all_preds = []
            for params in params_list:
                preds = []
                for i in range(0, len(sequences), 128):
                    batch = sequences[i : i + 128]
                    x = np.stack([_encode_yeast(s) for s in batch])
                    emb = encoder_fn(jnp.array(x), jnp.zeros(len(batch), dtype=jnp.int32))
                    emb_f32 = jnp.array(emb, dtype=jnp.float32)
                    org = jnp.zeros(len(batch), dtype=jnp.int32)
                    p = head_predict_fn(params, emb_f32, org)
                    # For yeast 18-bin: convert logits to expected expression
                    p_np = np.array(p)
                    if p_np.ndim == 2 and p_np.shape[1] == 18:
                        # Softmax → expected bin index
                        from scipy.special import softmax

                        probs = softmax(p_np, axis=1)
                        bins = np.arange(18)
                        p_np = (probs * bins).sum(axis=1)
                    preds.append(p_np.reshape(-1))
                all_preds.append(np.concatenate(preds))
            return np.stack(all_preds).mean(axis=0).astype(np.float32)

    logger.info(f"Loaded yeast AG oracle with {len(params_list)} folds")
    return _AGOracleYeast()


# ---------------------------------------------------------------------------
# AG S1 student (head-only, on-the-fly encoding)
# ---------------------------------------------------------------------------

# Lazy-loaded AG model + encoder (shared across all training runs)
_AG_MODEL_CACHE: dict[str, Any] = {}


def _get_ag_model_and_encoder(task: str):
    """Load AG model and build encoder/head functions once, cache for reuse."""
    if task in _AG_MODEL_CACHE:
        return _AG_MODEL_CACHE[task]

    import jax
    import jax.numpy as jnp
    import optax
    from alphagenome_ft import create_model_with_heads

    from models.alphagenome_heads import register_s2f_head
    from models.embedding_cache import (
        build_encoder_fn,
        build_head_only_predict_fn,
        build_head_only_train_fn,
        reinit_head_params,
    )

    if task == "k562":
        head_name = "exp1_s1_k562"
        num_tokens, num_tracks = 5, 1
        task_mode = "human"
    else:
        head_name = "exp1_s1_yeast"
        num_tokens, num_tracks = 3, 18
        task_mode = "yeast"

    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten-512-512",
        task_mode=task_mode,
        num_tracks=num_tracks,
        dropout_rate=0.1,
    )
    weights_path = os.environ.get(
        "ALPHAGENOME_WEIGHTS",
        "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1",
    )
    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=True,
    )
    reinit_head_params(model, head_name, num_tokens=num_tokens, dim=1536)

    encoder_fn = build_encoder_fn(model)
    head_predict_fn = build_head_only_predict_fn(model, head_name)
    head_train_fn = build_head_only_train_fn(model, head_name)

    result = {
        "model": model,
        "head_name": head_name,
        "encoder_fn": encoder_fn,
        "head_predict_fn": head_predict_fn,
        "head_train_fn": head_train_fn,
        "num_tokens": num_tokens,
        "num_tracks": num_tracks,
    }
    _AG_MODEL_CACHE[task] = result
    logger.info(f"AG S1 model loaded for {task} (T={num_tokens}, tracks={num_tracks})")
    return result


def _encode_sequences_for_ag(
    sequences: list[str], task: str, encoder_fn, batch_size: int = 128
) -> np.ndarray:
    """Encode sequences with frozen AG encoder. Returns (N, T, D) float16 array."""
    import jax.numpy as jnp

    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

    if task == "k562":
        from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM

        flank_5 = MPRA_UPSTREAM[-200:]
        flank_3 = MPRA_DOWNSTREAM[:200]

        def _encode_one(seq: str) -> np.ndarray:
            seq = seq.upper()
            if len(seq) < 200:
                pad = 200 - len(seq)
                seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
            elif len(seq) > 200:
                start = (len(seq) - 200) // 2
                seq = seq[start : start + 200]
            full = flank_5 + seq + flank_3
            out = np.zeros((600, 4), dtype=np.float32)
            for i, c in enumerate(full):
                if c in mapping:
                    out[i, mapping[c]] = 1.0
            return out

    else:  # yeast
        # Yeast plasmid flanks for AlphaGenome (54bp 5' + 89bp 3')
        yeast_f5 = "GCTAGCGCCGATATCCTAACGAAGTCACTACTACGTACTGCCCTGCACGATAGC"
        yeast_f3 = (
            "CCTGCAGCAGACGTCGACACGCGTCGTAAAGTGACGTTGTCCGAAACCCTT"
            "GCATTCGACACCAAACATTCTCTCAGTGCGTGCCCATGAAC"
        )

        def _encode_one(seq: str) -> np.ndarray:
            seq = seq.upper()
            # Extract 150bp core (or use as-is)
            core_str = seq[:150] if len(seq) >= 150 else seq
            full_str = yeast_f5 + core_str + yeast_f3
            full_len = len(full_str)
            # Pad/center to 384bp
            out = np.zeros((384, 4), dtype=np.float32)
            start = max(0, (384 - full_len) // 2)
            for i, c in enumerate(full_str):
                if i + start < 384 and c in mapping:
                    out[i + start, mapping[c]] = 1.0
            return out

    # Encode all sequences in batches
    all_embs = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        x = np.stack([_encode_one(s) for s in batch])
        emb = np.array(encoder_fn(jnp.array(x), jnp.zeros(len(batch), dtype=jnp.int32)))
        all_embs.append(emb.astype(np.float16))
    return np.concatenate(all_embs, axis=0)


def _train_ag_s1_student(
    task: str,
    sequences: list[str],
    labels: np.ndarray,
    lr: float,
    batch_size: int,
    seed: int,
    pre_encoded_embs: np.ndarray | None = None,
) -> SequenceModel:
    """Train an AG S1 student (frozen encoder, head-only).

    Args:
        pre_encoded_embs: If provided, skip encoding and use these embeddings
            directly. Shape ``(N, T, 1536)`` in float16. This avoids re-encoding
            the same sequences for each HP config / replicate.
    """
    import jax
    import jax.numpy as jnp
    import optax

    ag = _get_ag_model_and_encoder(task)
    model = ag["model"]
    head_train_fn = ag["head_train_fn"]
    head_predict_fn = ag["head_predict_fn"]
    encoder_fn = ag["encoder_fn"]

    from models.embedding_cache import reinit_head_params

    # Re-init head for each training run (fresh weights)
    reinit_head_params(model, ag["head_name"], num_tokens=ag["num_tokens"], dim=1536, rng=seed)

    # Use pre-encoded embeddings if available, otherwise encode now
    if pre_encoded_embs is not None:
        train_embs = pre_encoded_embs
        logger.info(f"  Using pre-encoded embeddings: {train_embs.shape}")
    else:
        logger.info(f"  Encoding {len(sequences):,} sequences with AG encoder...")
        train_embs = _encode_sequences_for_ag(sequences, task, encoder_fn)
        logger.info(f"  Embeddings shape: {train_embs.shape}")

    # Setup optimizer
    optimizer = optax.adamw(learning_rate=lr, weight_decay=1e-6)
    opt_state = optimizer.init(model._params)
    jax_rng = jax.random.PRNGKey(seed)

    @jax.jit
    def train_step(params, current_opt_state, step_rng, emb, targets, org_idx):
        def loss_func(p):
            preds = head_train_fn(p, step_rng, emb, org_idx)
            pred = jnp.squeeze(preds, axis=-1) if preds.ndim > 1 else preds
            return jnp.mean((pred - targets) ** 2)

        loss, grads = jax.value_and_grad(loss_func)(params)
        updates, next_opt_state = optimizer.update(grads, current_opt_state, params)
        return optax.apply_updates(params, updates), next_opt_state, loss

    @jax.jit
    def eval_step(params, emb, org_idx):
        preds = head_predict_fn(params, emb, org_idx)
        return jnp.squeeze(preds, axis=-1) if preds.ndim > 1 else preds

    # Training loop
    n_train = len(sequences)
    rng_perm = np.random.default_rng(seed)
    best_loss = float("inf")
    patience, patience_counter = 5, 0

    for epoch in range(50):
        perm = rng_perm.permutation(n_train)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            emb = jnp.array(train_embs[idx].astype(np.float32))
            targets = jnp.array(labels[idx])
            org_idx = jnp.zeros(len(idx), dtype=jnp.int32)
            jax_rng, step_rng = jax.random.split(jax_rng)
            model._params, opt_state, loss = train_step(
                model._params, opt_state, step_rng, emb, targets, org_idx
            )

        # Quick train loss check for early stopping
        epoch_loss = float(loss)
        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  AG S1 early stop at epoch {epoch + 1}")
                break

    # Return a SequenceModel wrapper
    frozen_params = jax.device_get(model._params)

    class _AGS1Student(SequenceModel):
        def predict(self, seqs: list[str]) -> np.ndarray:
            embs = _encode_sequences_for_ag(seqs, task, encoder_fn)
            preds = []
            for i in range(0, len(seqs), 256):
                emb = jnp.array(embs[i : i + 256].astype(np.float32))
                org = jnp.zeros(emb.shape[0], dtype=jnp.int32)
                p = eval_step(frozen_params, emb, org)
                preds.append(np.array(p).reshape(-1))
            return np.concatenate(preds)

    return _AGS1Student()


def _train_ag_s2_student(
    task: str,
    sequences: list[str],
    labels: np.ndarray,
    encoder_lr: float,
    batch_size: int,
    seed: int,
) -> SequenceModel:
    """Train an AG S2 student (joint encoder + head fine-tuning).

    Uses differential learning rates: encoder_lr for unfrozen encoder blocks,
    head_lr (from S2_CONFIG) for the head, and zero for frozen layers.
    """
    import jax
    import jax.numpy as jnp
    import optax
    from alphagenome_ft import create_model_with_heads

    from models.alphagenome_heads import register_s2f_head
    from models.embedding_cache import reinit_head_params

    s2_cfg = S2_CONFIG[task]
    head_lr = s2_cfg["head_lr"]
    wd = s2_cfg["weight_decay"]
    unfreeze_blocks = s2_cfg["unfreeze_blocks"]
    epochs = s2_cfg["epochs"]
    patience = s2_cfg["early_stop_patience"]
    warmup_epochs = s2_cfg["warmup_epochs"]

    task_cfg = TASK_CONFIGS[task]
    head_name = f"s2f_exp1_s2_{task}_{seed}"
    num_tracks = 18 if task == "yeast" else 1
    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten-512-512",
        task_mode=task_cfg["task_mode"],
        num_tracks=num_tracks,
        dropout_rate=0.1,
    )

    weights_path = os.environ.get(
        "ALPHAGENOME_WEIGHTS",
        "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1",
    )
    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=False,  # allow encoder gradients for S2
    )
    num_tokens = 5 if task == "k562" else 3
    reinit_head_params(model, head_name, num_tokens=num_tokens, dim=1536, rng=seed)

    # Per-group optimizer: head / encoder (unfrozen blocks) / frozen
    unfreeze_set = {f"downres_block_{b}" for b in unfreeze_blocks}

    def _label_fn(path, _leaf):
        key_strs = [p.key if hasattr(p, "key") else str(p) for p in path]
        s = "/".join(str(k) for k in key_strs)
        if head_name in s:
            return "head"
        elif "sequence_encoder" in s:
            for block_name in unfreeze_set:
                if block_name in s:
                    return "encoder"
            return "frozen"
        return "frozen"

    param_labels = jax.tree_util.tree_map_with_path(_label_fn, model._params)

    # During warmup: encoder gets zero updates
    def _make_optimizer(enc_lr: float):
        return optax.multi_transform(
            {
                "head": optax.adamw(learning_rate=head_lr, weight_decay=wd),
                "encoder": optax.adamw(learning_rate=enc_lr, weight_decay=wd),
                "frozen": optax.set_to_zero(),
            },
            param_labels,
        )

    optimizer = _make_optimizer(0.0)  # start with frozen encoder (warmup)
    opt_state = optimizer.init(model._params)

    # Prepare one-hot encoding function
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    if task == "k562":
        from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM

        flank_5 = MPRA_UPSTREAM[-200:]
        flank_3 = MPRA_DOWNSTREAM[:200]

        def _encode_one(seq_str: str) -> np.ndarray:
            seq_str = seq_str.upper()
            if len(seq_str) < 200:
                pad = 200 - len(seq_str)
                seq_str = "N" * (pad // 2) + seq_str + "N" * (pad - pad // 2)
            elif len(seq_str) > 200:
                start = (len(seq_str) - 200) // 2
                seq_str = seq_str[start : start + 200]
            full = flank_5 + seq_str + flank_3
            out = np.zeros((600, 4), dtype=np.float32)
            for i, c in enumerate(full):
                if c in mapping:
                    out[i, mapping[c]] = 1.0
            return out
    else:
        yeast_f5 = "GCTAGCGCCGATATCCTAACGAAGTCACTACTACGTACTGCCCTGCACGATAGC"
        yeast_f3 = (
            "CCTGCAGCAGACGTCGACACGCGTCGTAAAGTGACGTTGTCCGAAACCCTT"
            "GCATTCGACACCAAACATTCTCTCAGTGCGTGCCCATGAAC"
        )

        def _encode_one(seq_str: str) -> np.ndarray:
            seq_str = seq_str.upper()
            core = seq_str[:150] if len(seq_str) >= 150 else seq_str
            full_str = yeast_f5 + core + yeast_f3
            out = np.zeros((384, 4), dtype=np.float32)
            start = max(0, (384 - len(full_str)) // 2)
            for i, c in enumerate(full_str):
                if i + start < 384 and c in mapping:
                    out[i + start, mapping[c]] = 1.0
            return out

    # Pre-encode all training sequences (one-hot, not embeddings)
    logger.info(f"  Encoding {len(sequences):,} sequences to one-hot...")
    all_onehot = np.stack([_encode_one(s) for s in sequences])

    @jax.jit
    def train_step(params, current_opt_state, seqs, targets):
        def loss_func(p):
            preds = model._predict(
                p,
                model._state,
                seqs,
                jnp.zeros(len(seqs), dtype=jnp.int32),
                negative_strand_mask=jnp.zeros(len(seqs), dtype=bool),
                strand_reindexing=None,
            )[head_name]
            pred = jnp.squeeze(preds, axis=-1) if preds.ndim > 1 else preds
            if task == "yeast" and pred.ndim == 2 and pred.shape[1] == 18:
                # Yeast: softmax → expected bin index
                probs = jax.nn.softmax(pred, axis=1)
                pred = (probs * jnp.arange(18)).sum(axis=1)
            return jnp.mean((pred - targets) ** 2)

        loss, grads = jax.value_and_grad(loss_func)(params)
        updates, next_opt_state = optimizer.update(grads, current_opt_state, params)
        return optax.apply_updates(params, updates), next_opt_state, loss

    @jax.jit
    def predict_step(params, seqs):
        preds = model._predict(
            params,
            model._state,
            seqs,
            jnp.zeros(len(seqs), dtype=jnp.int32),
            negative_strand_mask=jnp.zeros(len(seqs), dtype=bool),
            strand_reindexing=None,
        )[head_name]
        pred = jnp.squeeze(preds, axis=-1) if preds.ndim > 1 else preds
        if task == "yeast" and pred.ndim == 2 and pred.shape[1] == 18:
            probs = jax.nn.softmax(pred, axis=1)
            pred = (probs * jnp.arange(18)).sum(axis=1)
        return pred

    # Training loop
    n_train = len(sequences)
    rng_perm = np.random.default_rng(seed)
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Switch from warmup to full S2 after warmup_epochs
        if epoch == warmup_epochs:
            logger.info(f"  S2: Unfreezing encoder blocks {unfreeze_blocks} (epoch {epoch + 1})")
            optimizer = _make_optimizer(encoder_lr)
            opt_state = optimizer.init(model._params)

        perm = rng_perm.permutation(n_train)
        epoch_losses = []
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            seqs = jnp.array(all_onehot[idx])
            targets = jnp.array(labels[idx])
            model._params, opt_state, loss = train_step(model._params, opt_state, seqs, targets)
            epoch_losses.append(float(loss))

        epoch_loss = float(np.mean(epoch_losses))
        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            best_params = jax.device_get(model._params)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= warmup_epochs:
                logger.info(f"  AG S2 early stop at epoch {epoch + 1}")
                break

    # Return a SequenceModel wrapper using best params
    class _AGS2Student(SequenceModel):
        def predict(self, seqs: list[str]) -> np.ndarray:
            x = np.stack([_encode_one(s) for s in seqs])
            preds = []
            for i in range(0, len(seqs), 64):
                batch_x = jnp.array(x[i : i + 64])
                p = predict_step(best_params, batch_x)
                preds.append(np.array(p).reshape(-1))
            return np.concatenate(preds)

    return _AGS2Student()


# ---------------------------------------------------------------------------
# Student training
# ---------------------------------------------------------------------------


def _train_student(
    task: str,
    student_type: str,
    sequences: list[str],
    labels: np.ndarray,
    lr: float,
    batch_size: int,
    seed: int,
    pre_encoded_embs: np.ndarray | None = None,
    ensemble_size: int = 5,
    epochs: int = 80,
    early_stopping_patience: int | None = None,
) -> SequenceModel:
    """Train a student model and return it."""
    if student_type == "dream_rnn":
        from models.dream_rnn_student import DREAMRNNStudent, TrainConfig

        cfg = TASK_CONFIGS[task]
        np.random.seed(seed)
        import torch

        torch.manual_seed(seed)

        student = DREAMRNNStudent(
            input_channels=cfg["input_channels"],
            sequence_length=cfg["sequence_length"],
            task_mode=cfg["task_mode"],
            ensemble_size=ensemble_size,
            train_config=TrainConfig(
                batch_size=batch_size,
                lr=lr,
                lr_lstm=lr,
                epochs=epochs,
                early_stopping_patience=early_stopping_patience,
            ),
        )
        student.fit(sequences, labels)
        return student
    elif student_type in ("alphagenome_k562_s1", "alphagenome_yeast_s1"):
        return _train_ag_s1_student(
            task,
            sequences,
            labels,
            lr,
            batch_size,
            seed,
            pre_encoded_embs=pre_encoded_embs,
        )
    elif student_type in ("alphagenome_k562_s2", "alphagenome_yeast_s2"):
        return _train_ag_s2_student(task, sequences, labels, lr, batch_size, seed)
    else:
        raise ValueError(f"Unknown student type: {student_type}")


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run_scaling_experiment(
    task: str,
    student_type: str,
    reservoir_name: str,
    training_sizes: list[int],
    hp_sweep: bool,
    n_replicates: int,
    output_base: Path,
    seed: int = 42,
    oracle_type: str = "default",
    ensemble_size: int = 5,
    epochs: int = 80,
    early_stopping_patience: int | None = None,
) -> list[RunResult]:
    """Run one reservoir scaling experiment."""
    from evaluation.exp1_eval import evaluate_on_exp1_test_panel

    task_cfg = TASK_CONFIGS[task]

    # Oracle-specific test set directories:
    # Default test sets use the primary oracle (AG for K562, DREAM for yeast).
    # Non-default oracle types use separate test set dirs with their own labels.
    resolved_oracle = (
        oracle_type if oracle_type != "default" else ("ag" if task == "k562" else "dream_rnn")
    )
    default_oracle = "ag" if task == "k562" else "dream_rnn"
    if resolved_oracle != default_oracle:
        test_set_dir = REPO / "data" / task / f"test_sets_{resolved_oracle}"
    else:
        test_set_dir = REPO / task_cfg["test_set_dir"]

    if not test_set_dir.exists():
        # Fall back to default test sets (AG labels for K562, DREAM for yeast).
        # This means training labels come from one oracle but test labels from another,
        # which is valid: test labels serve as a fixed ground-truth benchmark.
        fallback_dir = REPO / task_cfg["test_set_dir"]
        if fallback_dir.exists():
            logger.warning(
                f"Test set dir {test_set_dir} not found. "
                f"Falling back to default test sets: {fallback_dir}"
            )
            test_set_dir = fallback_dir
        else:
            logger.warning(
                f"Test set dir {test_set_dir} does not exist. "
                "Test evaluation will be skipped for missing NPZ files."
            )

    logger.info(f"Loading oracle for task={task}, oracle_type={oracle_type}...")
    oracle = _load_oracle(task, oracle_type=oracle_type)

    # Load pool for reservoir types that need base/pool sequences
    _NEEDS_POOL = {
        "genomic",
        "gc_matched",
        "dinuc_shuffle",
        "prm_1pct",
        "prm_5pct",
        "prm_10pct",
        "prm_20pct",
        "prm_50pct",
        "prm_uniform_1_10",
        "recombination_uniform",
        "recombination_2pt",
        "evoaug_structural",
        "evoaug_heavy",
        "ise_maximize",
        "ise_diverse_targets",
        "ise_target_high",
        "snv",
    }
    pool_seqs, pool_labels = None, None
    if reservoir_name in _NEEDS_POOL:
        logger.info("Loading genomic pool sequences...")
        pool_seqs, pool_labels = _load_pool_sequences(task)
        logger.info(f"Pool size: {len(pool_seqs):,}")

    def _build_hp_configs(n_train: int) -> list[dict]:
        """Build HP grid, using faster configs for large training sizes."""
        if not hp_sweep:
            if student_type.startswith("alphagenome"):
                return [{"learning_rate": 1e-3, "batch_size": 128}]
            return [{"learning_rate": 0.005, "batch_size": 1024}]

        # Use large-N grid for big training sizes (drops slow batch sizes)
        if n_train >= LARGE_N_THRESHOLD and student_type in HP_GRIDS_LARGE_N:
            grid = HP_GRIDS_LARGE_N[student_type]
        elif student_type in HP_GRIDS:
            grid = HP_GRIDS[student_type]
        else:
            if student_type.startswith("alphagenome"):
                return [{"learning_rate": 1e-3, "batch_size": 128}]
            return [{"learning_rate": 0.005, "batch_size": 1024}]

        return [dict(zip(grid.keys(), vals)) for vals in itertools.product(*grid.values())]

    results: list[RunResult] = []

    # ── Pre-label all training sizes to cache oracle labels ──────────────
    # This allows freeing the oracle from GPU before student training,
    # which is critical when oracle (JAX AG, ~80GB) and student (PyTorch
    # DREAM, ~15GB) together exceed the 93GB H100 GPU memory.
    _ISE_TYPES_PRE = {"ise_maximize", "ise_diverse_targets", "ise_target_high"}
    _POOL_MUTAGENESIS_PRE = {
        "recombination_uniform",
        "recombination_2pt",
        "evoaug_structural",
        "evoaug_heavy",
    }
    for n_train in training_sizes:
        label_cache_dir = output_base / reservoir_name / f"n{n_train}"
        label_cache_path = label_cache_dir / "oracle_labels.npz"
        if label_cache_path.exists():
            continue  # Already cached
        logger.info(f"[pre-label] Generating + labeling n={n_train:,} for {reservoir_name}")
        label_cache_dir.mkdir(parents=True, exist_ok=True)
        _res = _load_reservoir(reservoir_name, seed=seed)
        if reservoir_name == "random":
            _seqs, _ = _res.generate(n_train, task=task)
        elif reservoir_name == "dinuc_shuffle":
            _seqs, _ = _res.generate(
                n_train, task=task, method="dinuc_shuffle", reference_sequences=pool_seqs
            )
        elif reservoir_name == "genomic":
            _seqs, _ = _res.generate(n_train, pool_sequences=pool_seqs, pool_labels=pool_labels)
        elif reservoir_name == "gc_matched":
            _seqs, _ = _res.generate(n_train, pool_sequences=pool_seqs, task=task)
        elif reservoir_name.startswith("prm") or reservoir_name == "snv":
            _seqs, _ = _res.generate(n_train, base_sequences=pool_seqs, task=task)
        elif reservoir_name in _POOL_MUTAGENESIS_PRE or reservoir_name.startswith("recombination"):
            _seqs, _ = _res.generate(n_train, base_sequences=pool_seqs, task=task)
        elif reservoir_name in _ISE_TYPES_PRE:
            _seqs, _ = _res.generate(
                n_train, base_sequences=pool_seqs, task=task, student_model=oracle
            )
        else:
            _seqs, _ = _res.generate(n_train, task=task)
        _labels = oracle.predict(_seqs)
        np.savez_compressed(
            label_cache_path, sequences=np.array(_seqs, dtype=object), labels=_labels
        )
        logger.info(f"[pre-label] Cached {len(_seqs):,} labels to {label_cache_path}")
        del _seqs, _labels, _res

    # Free oracle from GPU to make room for student training
    logger.info("Freeing oracle model from GPU memory...")
    del oracle
    oracle = None
    import gc

    gc.collect()
    try:
        import jax

        jax.clear_caches()
        # Force JAX to release GPU memory
        backend = jax.lib.xla_bridge.get_backend()
        for buf in backend.live_buffers():
            buf.delete()
    except Exception:
        pass
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("Oracle freed. Proceeding with student training from cached labels.")

    for n_train in training_sizes:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training size: {n_train:,}")
        logger.info(f"{'=' * 60}")

        # Load sequences + labels from pre-labeling cache, or generate on the fly
        label_cache_dir = output_base / reservoir_name / f"n{n_train}"
        label_cache_dir.mkdir(parents=True, exist_ok=True)
        label_cache_path = label_cache_dir / "oracle_labels.npz"

        if label_cache_path.exists():
            # Pre-labeling phase already cached sequences + labels
            logger.info(f"Loading cached sequences + labels from {label_cache_path}")
            cached = np.load(label_cache_path, allow_pickle=True)
            sequences = cached["sequences"].tolist()
            labels = cached["labels"]
            logger.info(f"Cache hit: {len(labels):,} sequences + labels loaded")
        else:
            # No cache — generate sequences and label with oracle (must still be alive)
            if oracle is None:
                raise RuntimeError(
                    f"No label cache at {label_cache_path} and oracle already freed. "
                    "Pre-labeling should have cached this. Delete output dir and rerun."
                )
            reservoir = _load_reservoir(reservoir_name, seed=seed)
            _POOL_MUTAGENESIS = {
                "recombination_uniform",
                "recombination_2pt",
                "evoaug_structural",
                "evoaug_heavy",
            }
            _ISE_TYPES = {"ise_maximize", "ise_diverse_targets", "ise_target_high"}

            if reservoir_name == "random":
                sequences, meta = reservoir.generate(n_train, task=task)
            elif reservoir_name == "dinuc_shuffle":
                sequences, meta = reservoir.generate(
                    n_train, task=task, method="dinuc_shuffle", reference_sequences=pool_seqs
                )
            elif reservoir_name == "genomic":
                sequences, meta = reservoir.generate(
                    n_train, pool_sequences=pool_seqs, pool_labels=pool_labels
                )
            elif reservoir_name == "gc_matched":
                sequences, meta = reservoir.generate(n_train, pool_sequences=pool_seqs, task=task)
            elif reservoir_name.startswith("prm") or reservoir_name == "snv":
                sequences, meta = reservoir.generate(n_train, base_sequences=pool_seqs, task=task)
            elif reservoir_name in _POOL_MUTAGENESIS or reservoir_name.startswith("recombination"):
                sequences, meta = reservoir.generate(n_train, base_sequences=pool_seqs, task=task)
            elif reservoir_name in _ISE_TYPES:
                sequences, meta = reservoir.generate(
                    n_train, base_sequences=pool_seqs, task=task, student_model=oracle
                )
            elif reservoir_name in (
                "motif_planted",
                "motif_grammar",
                "motif_grammar_tight",
            ):
                sequences, meta = reservoir.generate(n_train, task=task)
            else:
                sequences, meta = reservoir.generate(n_train, task=task)

            logger.info(f"Labeling {len(sequences):,} sequences with oracle...")
            label_start = time.perf_counter()
            labels = oracle.predict(sequences)
            logger.info(f"Oracle labeling took {time.perf_counter() - label_start:.1f}s")
            np.savez_compressed(
                label_cache_path,
                sequences=np.array(sequences, dtype=object),
                labels=labels,
            )
            logger.info(f"Cached oracle labels to {label_cache_path}")

        # Validation split (10% of generated data)
        n_val = max(100, int(0.1 * n_train))
        rng = np.random.default_rng(seed)
        val_idx = rng.choice(len(sequences), size=n_val, replace=False)
        train_mask = np.ones(len(sequences), dtype=bool)
        train_mask[val_idx] = False
        train_seqs = [sequences[i] for i in range(len(sequences)) if train_mask[i]]
        train_labels = labels[train_mask]
        val_seqs = [sequences[i] for i in val_idx]
        val_labels = labels[val_idx]

        # Pre-encode embeddings once for AG S1 (avoid redundant encoder passes)
        train_embs_cached = None
        if student_type in ("alphagenome_k562_s1", "alphagenome_yeast_s1"):
            ag = _get_ag_model_and_encoder(task)
            logger.info(f"  Pre-encoding {len(train_seqs):,} training sequences for AG S1...")
            enc_start = time.perf_counter()
            train_embs_cached = _encode_sequences_for_ag(train_seqs, task, ag["encoder_fn"])
            logger.info(
                f"  Pre-encoding took {time.perf_counter() - enc_start:.1f}s, "
                f"shape={train_embs_cached.shape}"
            )

        hp_configs = _build_hp_configs(n_train)
        # Filter out configs where batch_size > training samples (causes 0 steps with drop_last)
        actual_n_train = len(train_seqs)
        hp_configs = [hp for hp in hp_configs if hp["batch_size"] <= actual_n_train]
        logger.info(f"HP configs for n={n_train:,}: {len(hp_configs)} configs")

        best_hp = None
        best_val_r = -1.0

        for hp_idx, hp in enumerate(hp_configs):
            hp_val_rs = []

            for rep in range(n_replicates):
                rep_seed = seed + rep * 1000 + hp_idx * 100
                run_dir = (
                    output_base / reservoir_name / f"n{n_train}" / f"hp{hp_idx}" / f"seed{rep_seed}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)

                logger.info(
                    f"  HP {hp_idx + 1}/{len(hp_configs)}, "
                    f"rep {rep + 1}/{n_replicates}, "
                    f"lr={hp['learning_rate']}, bs={hp['batch_size']}"
                )

                run_start = time.perf_counter()
                try:
                    student = _train_student(
                        task=task,
                        student_type=student_type,
                        sequences=train_seqs,
                        labels=train_labels,
                        lr=hp["learning_rate"],
                        batch_size=hp["batch_size"],
                        seed=rep_seed,
                        pre_encoded_embs=train_embs_cached,
                        ensemble_size=ensemble_size,
                        epochs=epochs,
                        early_stopping_patience=early_stopping_patience,
                    )

                    # Validation evaluation
                    val_preds = student.predict(val_seqs)
                    val_r = float(np.corrcoef(val_preds, val_labels)[0, 1])
                    if np.isnan(val_r):
                        val_r = 0.0
                    hp_val_rs.append(val_r)

                    # Test evaluation
                    test_metrics = evaluate_on_exp1_test_panel(student, task, test_set_dir)

                    wall_s = time.perf_counter() - run_start

                    result = RunResult(
                        reservoir=reservoir_name,
                        task=task,
                        student=student_type,
                        n_train=n_train,
                        hp_config=hp,
                        seed=rep_seed,
                        val_pearson_r=val_r,
                        test_metrics=test_metrics,
                        wall_seconds=wall_s,
                        output_dir=str(run_dir),
                    )
                    results.append(result)

                    # Save result
                    (run_dir / "result.json").write_text(
                        json.dumps(asdict(result), indent=2, default=str)
                    )
                    logger.info(f"    val_r={val_r:.4f}, wall={wall_s:.1f}s")

                    # W&B logging
                    try:
                        import wandb

                        if wandb.run is not None:
                            log_data = {
                                "n_train": n_train,
                                "val/pearson_r": val_r,
                                "hp/lr": hp["learning_rate"],
                                "hp/batch_size": hp["batch_size"],
                            }
                            for tname, tmetrics in test_metrics.items():
                                for mname, mval in tmetrics.items():
                                    log_data[f"test/{tname}/{mname}"] = mval
                            wandb.log(log_data)
                    except ImportError:
                        pass

                except Exception as e:
                    logger.error(f"    FAILED: {e}")
                    continue

            # Track best HP
            if hp_val_rs:
                mean_val_r = float(np.mean(hp_val_rs))
                if mean_val_r > best_val_r:
                    best_val_r = mean_val_r
                    best_hp = hp

        logger.info(f"  Best HP for n={n_train:,}: {best_hp}, val_r={best_val_r:.4f}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Experiment 1.1: Reservoir scaling laws")
    parser.add_argument("--task", required=True, choices=["k562", "yeast"])
    parser.add_argument(
        "--student",
        required=True,
        choices=[
            "dream_rnn",
            "alphagenome_k562_s1",
            "alphagenome_yeast_s1",
            "alphagenome_k562_s2",
            "alphagenome_yeast_s2",
        ],
    )
    parser.add_argument("--reservoir", nargs="+", required=True, help="Reservoir strategy name(s)")
    parser.add_argument(
        "--oracle",
        default="default",
        choices=["default", "ag", "dream_rnn"],
        help="Oracle type: 'default' (AG for K562, DREAM for yeast), 'ag', or 'dream_rnn'",
    )
    parser.add_argument("--training-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--n-replicates", type=int, default=3)
    parser.add_argument("--no-hp-sweep", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=5,
        help="Number of ensemble members per DREAM-RNN student (default: 5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Max training epochs for DREAM-RNN (default: 80)",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Early stopping patience (epochs without improvement). Default: None (no early stop)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    training_sizes = args.training_sizes or DEFAULT_TRAINING_SIZES
    oracle_suffix = f"_{args.oracle}" if args.oracle != "default" else ""
    output_base = (
        Path(args.output_dir)
        if args.output_dir
        else (REPO / "outputs" / "exp1_1" / args.task / f"{args.student}{oracle_suffix}")
    )

    # W&B init
    try:
        import wandb

        wandb.init(
            project="albench-s2f",
            name=f"exp1_1_{args.task}_{args.student}_{'-'.join(args.reservoir)}",
            tags=["exp1", "scaling", args.task, args.student] + args.reservoir,
            group="exp1_1",
            config={
                "task": args.task,
                "student": args.student,
                "oracle": args.oracle,
                "reservoirs": args.reservoir,
                "training_sizes": training_sizes,
                "n_replicates": args.n_replicates,
                "hp_sweep": not args.no_hp_sweep,
            },
        )
        wandb.define_metric("test/*/pearson_r", step_metric="n_train")
    except (ImportError, Exception) as e:
        logger.info(f"wandb not available — skipping logging ({e})")

    all_results = []
    for reservoir_name in args.reservoir:
        logger.info(f"\n{'#' * 70}")
        logger.info(f"Reservoir: {reservoir_name}")
        logger.info(f"{'#' * 70}")

        results = run_scaling_experiment(
            task=args.task,
            student_type=args.student,
            reservoir_name=reservoir_name,
            training_sizes=training_sizes,
            hp_sweep=not args.no_hp_sweep,
            n_replicates=args.n_replicates,
            output_base=output_base,
            seed=args.seed,
            oracle_type=args.oracle,
            ensemble_size=args.ensemble_size,
            epochs=args.epochs,
            early_stopping_patience=args.early_stop_patience,
        )
        all_results.extend(results)

    # Save summary
    summary_dir = output_base / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = [asdict(r) for r in all_results]
    (summary_dir / "all_results.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"\nSaved {len(all_results)} results to {summary_dir}")

    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
