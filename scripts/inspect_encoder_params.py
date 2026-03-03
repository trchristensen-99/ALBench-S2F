#!/usr/bin/env python
"""Print AlphaGenome encoder parameter tree structure for selective freezing.

Outputs the sequence_encoder layer names, parameter shapes, and sizes.
Run on HPC (needs GPU for model creation):

    uv run --no-sync python scripts/inspect_encoder_params.py \
        --weights-path /grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax

from models.alphagenome_heads import register_s2f_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-path", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    from alphagenome_ft import create_model_with_heads

    register_s2f_head(
        head_name="tmp_inspect_boda_flatten_512_512_v4",
        arch="boda-flatten-512-512",
        task_mode="human",
        num_tracks=1,
        dropout_rate=0.0,
    )

    weights_path = str(Path(args.weights_path).expanduser().resolve())
    model = create_model_with_heads(
        "all_folds",
        heads=["tmp_inspect_boda_flatten_512_512_v4"],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=True,
    )

    print("=" * 80)
    print("AlphaGenome Parameter Tree — Top-Level Groups")
    print("=" * 80)

    # Top-level breakdown
    group_counts: dict[str, int] = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(model._params):
        keys = [p.key if hasattr(p, "key") else str(p) for p in path]
        top = keys[1] if len(keys) > 1 else keys[0]  # skip 'alphagenome' wrapper
        group_counts[top] = group_counts.get(top, 0) + leaf.size
    for group, count in sorted(group_counts.items(), key=lambda x: -x[1]):
        print(f"  {group}: {count:>15,} params")

    print("\n" + "=" * 80)
    print("Sequence Encoder — Layer Details")
    print("=" * 80)

    encoder_layers: dict[str, int] = {}
    encoder_shapes: dict[str, list[str]] = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(model._params):
        keys = [p.key if hasattr(p, "key") else str(p) for p in path]
        full_path = "/".join(str(k) for k in keys)
        if "sequence_encoder" not in full_path:
            continue
        # Group by the first 4 path components (e.g., alphagenome/sequence_encoder/layer_0/sublayer)
        depth = min(4, len(keys))
        layer_key = "/".join(str(k) for k in keys[:depth])
        encoder_layers[layer_key] = encoder_layers.get(layer_key, 0) + leaf.size
        if layer_key not in encoder_shapes:
            encoder_shapes[layer_key] = []
        encoder_shapes[layer_key].append(
            f"{'/'.join(str(k) for k in keys[depth:])}={list(leaf.shape)}"
        )

    for layer, count in sorted(encoder_layers.items()):
        print(f"\n  {layer}: {count:,} params")
        for shape_info in encoder_shapes[layer][:5]:  # show first 5 param tensors
            print(f"    {shape_info}")
        if len(encoder_shapes[layer]) > 5:
            print(f"    ... and {len(encoder_shapes[layer]) - 5} more tensors")

    print("\n" + "=" * 80)
    print("Head Parameters")
    print("=" * 80)
    for path, leaf in jax.tree_util.tree_leaves_with_path(model._params):
        keys = [p.key if hasattr(p, "key") else str(p) for p in path]
        full_path = "/".join(str(k) for k in keys)
        if "tmp_inspect" in full_path:
            print(f"  {full_path}: {list(leaf.shape)} ({leaf.size:,})")


if __name__ == "__main__":
    main()
