#!/usr/bin/env python
"""Add --val_cache to scratchpad/menu_retrain.py and albench/arch_search/arch_autoresearch.py (+ identical scratchpad copy).

Targeted string replacement, never a whole-file overwrite: other agents edit these files, so every edit anchors on a
line that must occur EXACTLY ONCE in the file as it is right now, and the write is atomic (tmp + os.replace) and
aborted if the file changed between our read and our write. Re-run safe: an already-patched file is left alone.

usage: apply_val_cache_patch.py menu FILE            patch a menu_retrain.py (FILE may be a copy)
       apply_val_cache_patch.py arch FILE [FILE2..]  patch arch_autoresearch.py file(s); all must end up identical
"""

from __future__ import annotations

import hashlib
import os
import sys
import time

ARG_LINE = (
    '    ap.add_argument("--val_cache", default="", help="npz (sequences, oracle_labels, oracle_id) replacing the chr19/21/X validation set '
    'for early stopping, val_pred, val_labels and proposer feedback; default empty = unchanged behaviour")\n'
)
LOAD_ANCHOR = "        args.D, ref_only=True, val_frac=0.1, seed=0, reservoir_cache=args.reservoir_cache, chr_val=True)\n"
LOAD_BLOCK = """    if args.val_cache:  # alternative validation set (albench/arch_search/val_cache.py); the default chr19/21/X path above is untouched
        from albench.arch_search.val_cache import guard_out_dir, load_val_cache
        va_s, va_y, _vc = load_val_cache(args.val_cache, train_seqs=tr_s)
        guard_out_dir(args.out_dir, _vc)
        print(f"  val_cache={_vc['path']} variant={_vc['variant']} n={_vc['n']:,} sha={_vc['sha1_16']} oracle={_vc['oracle_id']} "
              f"test_overlap={_vc['n_test_overlap']} train_overlap={_vc['n_train_overlap']}", flush=True)
    elif os.path.exists(os.path.join(args.out_dir, "val_cache.json")):
        raise SystemExit(f"{args.out_dir} was started with --val_cache; pass the same --val_cache or use a fresh --out_dir")
"""
META_ADD = """            if args.val_cache:
                meta["val_cache"] = {"path": _vc["path"], "sha1_16": _vc["sha1_16"], "variant": _vc["variant"], "n": _vc["n"]}
"""
EXTRA_ARG = (
    '    ap.add_argument("--extra_val", default="", help="comma-separated NAME=PATH val caches whose predictions are ALSO saved (valpred_NAME in each model npz, '
    'val_labels_NAME in labels.npz) for cross-scoring a model on validation sets it was not trained against; default off = unchanged outputs")\n'
)
EXTRA_LOAD = """    extra = {}
    if args.extra_val:  # cross-scoring validation sets (albench/arch_search/val_cache.py); nothing here runs by default
        from albench.arch_search.val_cache import guard_extra, load_val_cache
        _infos = {}
        for _item in args.extra_val.split(","):
            _nm, _pth = _item.split("=", 1)
            _s, _y, _inf = load_val_cache(_pth, train_seqs=tr_s)
            if args.smoke:
                _s, _y = _s[:3000], _y[:3000]
            extra[_nm] = (one_hot(_s, args.seq_len), np.asarray(_y, np.float32))
            _infos[_nm] = {"path": _inf["path"], "sha1_16": _inf["sha1_16"], "n": len(_y)}
        guard_extra(args.out_dir, _infos)
        print("  extra_val: " + ", ".join(f"{k}(n={v['n']:,} sha={v['sha1_16']})" for k, v in _infos.items()), flush=True)
    elif os.path.exists(os.path.join(args.out_dir, "extra_val.json")):
        raise SystemExit(f"{args.out_dir} was started with --extra_val; pass the same --extra_val or use a fresh --out_dir")
"""
LABELS_ANCHOR = '    labels = {"val_labels": yva, "test_oracle": te_oracle, "test_true": te_true}\n'
SAVE_ANCHOR = '    _atomic_savez(out / "labels.npz", **labels)\n'
TESTLOOP_ANCHOR = '            for kk, Xk in Xsets.items():\n                res[f"test_pred_{kk}"] = predict_rc(model, Xk)\n'
EDITS = {
    "menu": [
        ("    args = ap.parse_args()\n", ARG_LINE + "    args = ap.parse_args()\n"),
        (LOAD_ANCHOR, LOAD_ANCHOR + LOAD_BLOCK),
        (
            '"cap_hit": bool(info.get("epochs_trained", 0) and (time.time() - t0) > cap_min * 60 * 0.98)}\n',
            '"cap_hit": bool(info.get("epochs_trained", 0) and (time.time() - t0) > cap_min * 60 * 0.98)}\n'
            + META_ADD,
        ),
        # --extra_val (cross-scoring); menu_retrain only
        ("    args = ap.parse_args()\n", EXTRA_ARG + "    args = ap.parse_args()\n"),
        (LABELS_ANCHOR, EXTRA_LOAD + LABELS_ANCHOR),
        (
            SAVE_ANCHOR,
            '    labels.update({f"val_labels_{_k}": _v[1] for _k, _v in extra.items()})\n'
            + SAVE_ANCHOR,
        ),
        (
            TESTLOOP_ANCHOR,
            TESTLOOP_ANCHOR
            + '            for _k, _v in extra.items():\n                res[f"valpred_{_k}"] = predict_rc(model, _v[0])\n',
        ),
    ],
    "arch": [
        ("    args = ap.parse_args()\n", ARG_LINE + "    args = ap.parse_args()\n"),
        (LOAD_ANCHOR, LOAD_ANCHOR + LOAD_BLOCK),
        (
            '"train_time_sec": time.time() - t0, "gpu": gpu, "source": p["source"]}\n',
            '"train_time_sec": time.time() - t0, "gpu": gpu, "source": p["source"]}\n' + META_ADD,
        ),
    ],
}


def patch(path: str, kind: str) -> str:
    st0 = os.stat(path)
    src = open(path).read()
    if "--val_cache" in src:
        return "already patched"
    for anchor, new in EDITS[kind]:
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(
                f"{path}: anchor occurs {n}x (need exactly 1): {anchor.strip()[:80]!r}"
            )
        src = src.replace(anchor, new)
    st1 = os.stat(path)
    if (st1.st_mtime_ns, st1.st_size) != (st0.st_mtime_ns, st0.st_size):
        raise SystemExit(f"{path} changed while patching; re-run")
    tmp = f"{path}.vcpatch{os.getpid()}"
    open(tmp, "w").write(src)
    os.replace(tmp, path)
    return "patched"


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


if __name__ == "__main__":
    kind, files = sys.argv[1], sys.argv[2:]
    if kind == "arch" and len(files) > 1:
        if len({md5(f) for f in files}) != 1:
            raise SystemExit(
                f"copies differ before patching: {[(f, md5(f)) for f in files]}; sync them first"
            )
    for f in files:
        print(f, patch(f, kind), time.strftime("%H:%M:%S"))
    if kind == "arch" and len(files) > 1:
        assert len({md5(f) for f in files}) == 1, "copies diverged after patching"
        print("copies identical:", md5(files[0]))
