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

    head_name = "boda_flatten_v4"
    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten",
        task_mode="human",
        num_tracks=1,
        dropout_rate=0.0,
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
        ckpt = run_dir / "best_model.pt"
        if ckpt.exists():
            runs.append(ckpt)

    if not runs:
        raise FileNotFoundError(f"No K562 DREAM-RNN oracle checkpoints in {oracle_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for ckpt_path in runs:
        m = create_dream_rnn(
            input_channels=5,
            sequence_length=200,
            task_mode="k562",
            hidden_dim=320,
            cnn_filters=160,
            dropout_cnn=0.1,
            dropout_lstm=0.1,
        )
        state = torch.load(ckpt_path, map_location="cpu")
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
        def predict(self, sequences: list[str]) -> np.ndarray:
            encoded = np.stack([_encode_k562(s) for s in sequences])
            x = torch.from_numpy(encoded).float().to(device)
            all_preds = []
            for m in models:
                with torch.no_grad():
                    p = m.predict(x, use_reverse_complement=True)
                    all_preds.append(p.cpu().numpy().reshape(-1))
            return np.stack(all_preds).mean(axis=0).astype(np.float32)

    logger.info(f"Loaded K562 DREAM-RNN oracle with {len(models)} folds")
    return _DREAMOracleK562()


def _load_yeast_dream_oracle():
    """Load yeast DREAM-RNN 10-fold oracle."""
    import torch

    from models.dream_rnn import create_dream_rnn

    oracle_dir = REPO / "outputs" / "oracle_yeast_dream_rnn_v2"
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
        m = create_dream_rnn(input_channels=6, sequence_length=150, task_mode="yeast")
        state = torch.load(ckpt_path, map_location="cpu")
        m.load_state_dict(state["model_state_dict"], strict=True)
        m.to(device).eval()
        models.append(m)

    from data.utils import one_hot_encode

    class _DREAMOracle(SequenceModel):
        def predict(self, sequences: list[str]) -> np.ndarray:
            encoded = []
            for seq in sequences:
                base = one_hot_encode(seq, add_singleton_channel=False)
                rc = np.zeros((1, len(seq)), dtype=np.float32)
                singleton = np.zeros((1, len(seq)), dtype=np.float32)
                encoded.append(np.concatenate([base, rc, singleton], axis=0))
            x = torch.from_numpy(np.stack(encoded)).float().to(device)

            all_preds = []
            for m in models:
                with torch.no_grad():
                    p = m.predict(x, use_reverse_complement=True)
                    all_preds.append(p.cpu().numpy().reshape(-1))
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

    head_name = "yeast_oracle_head"
    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten",
        task_mode="yeast",
        num_tracks=18,
        dropout_rate=0.0,
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
        arch="boda-flatten",
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
) -> SequenceModel:
    """Train an AG S1 student (frozen encoder, head-only)."""
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

    # Encode training sequences
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
            ensemble_size=5,
            train_config=TrainConfig(
                batch_size=batch_size,
                lr=lr,
                lr_lstm=lr,
                epochs=80,
            ),
        )
        student.fit(sequences, labels)
        return student
    elif student_type in ("alphagenome_k562_s1", "alphagenome_yeast_s1"):
        return _train_ag_s1_student(task, sequences, labels, lr, batch_size, seed)
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
        logger.warning(
            f"Test set dir {test_set_dir} does not exist. "
            "Test evaluation will be skipped for missing NPZ files."
        )

    logger.info(f"Loading oracle for task={task}, oracle_type={oracle_type}...")
    oracle = _load_oracle(task, oracle_type=oracle_type)

    # Load pool for genomic/PRM samplers
    pool_seqs, pool_labels = None, None
    if reservoir_name in ("genomic", "prm_1pct", "prm_5pct", "prm_10pct", "prm_uniform_1_10"):
        logger.info("Loading genomic pool sequences...")
        pool_seqs, pool_labels = _load_pool_sequences(task)
        logger.info(f"Pool size: {len(pool_seqs):,}")

    # HP grid
    if hp_sweep and student_type in HP_GRIDS:
        grid = HP_GRIDS[student_type]
        hp_configs = [dict(zip(grid.keys(), vals)) for vals in itertools.product(*grid.values())]
    else:
        # Default single config
        if student_type.startswith("alphagenome"):
            hp_configs = [{"learning_rate": 1e-3, "batch_size": 128}]
        else:
            hp_configs = [{"learning_rate": 0.005, "batch_size": 1024}]

    results: list[RunResult] = []

    for n_train in training_sizes:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training size: {n_train:,}")
        logger.info(f"{'=' * 60}")

        # Generate sequences via reservoir
        reservoir = _load_reservoir(reservoir_name, seed=seed)

        if reservoir_name == "random":
            sequences, meta = reservoir.generate(n_train, task=task)
        elif reservoir_name == "genomic":
            sequences, meta = reservoir.generate(
                n_train, pool_sequences=pool_seqs, pool_labels=pool_labels
            )
        elif reservoir_name.startswith("prm"):
            sequences, meta = reservoir.generate(n_train, base_sequences=pool_seqs, task=task)
        else:
            sequences, meta = reservoir.generate(n_train, task=task)

        # Label with oracle
        logger.info(f"Labeling {len(sequences):,} sequences with oracle...")
        label_start = time.perf_counter()
        labels = oracle.predict(sequences)
        logger.info(f"Oracle labeling took {time.perf_counter() - label_start:.1f}s")

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
        choices=["dream_rnn", "alphagenome_k562_s1", "alphagenome_yeast_s1"],
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
    except ImportError:
        logger.info("wandb not available — skipping logging")

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
