"""Dump the param-tree leaf keys of one AG_S2 oracle fold checkpoint.

Diagnostic for the `Unable to retrieve parameter 'scale' for module
'.../~predict/norm'` crash: prints every leaf path so we can see whether the
saved checkpoint actually contains the head LayerNorm `scale`/`offset` params,
and under what module path (e.g. `.../~predict/norm` vs `.../norm`).
"""

from __future__ import annotations

import sys

import orbax.checkpoint as ocp


def walk(tree, prefix=""):
    if isinstance(tree, dict):
        for k in sorted(tree.keys()):
            walk(tree[k], f"{prefix}/{k}" if prefix else str(k))
    else:
        try:
            shape = getattr(tree, "shape", None)
        except Exception:
            shape = None
        print(f"{prefix}\t{shape}")


def main() -> None:
    ckpt_path = sys.argv[1]
    checkpointer = ocp.StandardCheckpointer()
    params, _ = checkpointer.restore(ckpt_path)
    print(f"=== TOP-LEVEL KEYS for {ckpt_path} ===")
    if isinstance(params, dict):
        for k in sorted(params.keys()):
            print(f"  {k}")
    print("=== ALL LEAVES CONTAINING 'norm' (case-insensitive) ===")
    # Re-walk but only print norm-related and head-related leaves
    import io

    buf = io.StringIO()
    _stdout = sys.stdout
    sys.stdout = buf
    walk(params)
    sys.stdout = _stdout
    lines = buf.getvalue().splitlines()
    for ln in lines:
        low = ln.lower()
        if "norm" in low or "boda_flatten" in low or "~predict" in low or "head" in low:
            print(ln)
    print(f"=== TOTAL LEAVES: {len(lines)} ===")


if __name__ == "__main__":
    main()
