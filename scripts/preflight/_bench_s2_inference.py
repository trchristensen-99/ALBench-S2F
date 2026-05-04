"""Tiny benchmark: time AG-S2 inference per-batch on H100 to verify rate.

Loads fold 0 ckpt, JITs, runs 100 batches with explicit per-batch
``block_until_ready`` calls, prints individual + median timing. This
isolates the GPU forward time from any host-side overhead.

If the median per-batch time is ~0.05 sec (matching training), then my
production loop has overhead. If it's ~8 sec, then the real bug is in
the model setup itself (precision, block usage, etc.).
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from alphagenome_ft import create_model_with_heads

from models.alphagenome_heads import register_s2f_head
from models.embedding_cache import reinit_head_params

REPO = Path(__file__).resolve().parents[2]


def main():
    head_name = "alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4"
    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten-512-512",
        task_mode="human",
        num_tracks=1,
        dropout_rate=0.1,
    )
    weights = "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"
    print("Loading model …", flush=True)
    t0 = time.time()
    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=weights,
        use_encoder_output=True,
        detach_backbone=False,
    )
    reinit_head_params(model, head_name, num_tokens=5, dim=1536)
    print(f"  Model load: {time.time() - t0:.1f}s", flush=True)

    ckpt = REPO / "outputs" / "stage2_k562_oracle" / "fold_0" / "best_model" / "checkpoint"
    print(f"Loading S2 ckpt from {ckpt}", flush=True)
    t0 = time.time()
    loaded_params, _ = ocp.StandardCheckpointer().restore(ckpt)
    model._params = jax.device_put(loaded_params)
    print(f"  Ckpt load: {time.time() - t0:.1f}s", flush=True)

    @jax.jit
    def predict_step(params, state, sequences):
        return model._predict(
            params,
            state,
            sequences,
            jnp.zeros(len(sequences), dtype=jnp.int32),
            requested_outputs=[head_name],
            negative_strand_mask=jnp.zeros(len(sequences), dtype=bool),
            strand_reindexing=None,
        )[head_name]

    BS = 256
    L = 600
    dummy = jnp.zeros((BS, L, 4), dtype=jnp.float32)

    # JIT compile (first call always slow)
    print("JIT compiling …", flush=True)
    t0 = time.time()
    _ = predict_step(model._params, model._state, dummy).block_until_ready()
    print(f"  JIT + first call: {time.time() - t0:.2f}s", flush=True)

    # Now time 20 batches with sync
    print("Timing 20 hot batches …", flush=True)
    times = []
    for i in range(20):
        t = time.time()
        out = predict_step(model._params, model._state, dummy)
        out.block_until_ready()
        dt = time.time() - t
        times.append(dt)
        print(f"  batch {i}: {dt * 1000:.1f}ms", flush=True)
    print(
        f"\nMedian per-batch: {1000 * np.median(times):.1f}ms  "
        f"min: {1000 * min(times):.1f}ms  max: {1000 * max(times):.1f}ms"
    )
    print(
        f"At median rate, 1M seqs / 256 / median = {1_000_000 / 256 / np.median(times) / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
