"""Load an alternative VALIDATION set (a "val cache") for the architecture loop and the menu retrain.

Why a loader and not three lines in each script: the validation set is what early stopping, the ElasticNet ensemble and
the LLM proposer's feedback are all fitted against, so a wrong one fails silently (plausible numbers, wrong selection).
Every guard that matters is therefore in one place: the oracle stamp must be the canonical one (an unstamped or superseded
label space makes val-vs-train comparisons meaningless), no sequence may be a test-chromosome genomic sequence, and the
content is hashed so the hash can be stamped next to every result that used it.

Default behaviour of the callers is unchanged: nothing here is imported unless ``--val_cache`` is given.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np


def _canon(s: str) -> str:
    r = s[::-1].translate(str.maketrans("ACGTacgt", "TGCAtgca"))
    return s if s <= r else r


def load_val_cache(path: str | Path, train_seqs=None) -> tuple[list[str], np.ndarray, dict]:
    """Return (sequences, oracle_labels, info) for a val-cache npz (keys: sequences, oracle_labels, oracle_id).

    Raises on: missing keys, a non-canonical oracle stamp (override: HP_ALLOW_UNSTAMPED=1), or any sequence that also
    appears (either strand) in the chr7/chr13 test genomic set (override: HP_ALLOW_VAL_OVERLAP=1). Overlap with
    ``train_seqs`` is reported, not fatal: a validation sequence that is also a training sequence inflates val r.
    """
    from albench.core.paths import REPO_ROOT

    p = Path(path)
    z = np.load(p, allow_pickle=True)
    for k in ("sequences", "oracle_labels"):
        if k not in z.files:
            raise RuntimeError(f"val cache {p.name} lacks '{k}' (has {z.files})")
    stamp = str(z["oracle_id"]) if "oracle_id" in z.files else "UNSTAMPED"
    if os.environ.get("HP_ALLOW_UNSTAMPED") != "1":
        from experiments.test_set_guards import CANONICAL_ORACLE_ID

        if stamp != CANONICAL_ORACLE_ID:
            raise RuntimeError(
                f"val cache {p.name} has oracle_id={stamp!r}, expected {CANONICAL_ORACLE_ID!r}. Re-label it with "
                f"scripts/label_pool.py, or set HP_ALLOW_UNSTAMPED=1 for a deliberate ad-hoc run."
            )
    seqs = [str(s) for s in z["sequences"]]
    y = np.asarray(z["oracle_labels"], np.float32)
    fin = np.isfinite(y)
    if not fin.all():
        seqs = [s for s, ok in zip(seqs, fin) if ok]
        y = y[fin]
    test = np.load(
        Path(REPO_ROOT) / "data/k562/test_sets_ag_s2_chrsplit/genomic_oracle.npz", allow_pickle=True
    )["sequences"]
    test_c = {_canon(str(s)) for s in test}
    n_test = sum(_canon(s) in test_c for s in seqs)
    if n_test and os.environ.get("HP_ALLOW_VAL_OVERLAP") != "1":
        raise RuntimeError(
            f"val cache {p.name}: {n_test} sequences are chr7/chr13 TEST genomic sequences (either strand); refusing."
        )
    n_train = 0
    if train_seqs is not None:
        tr_c = {_canon(str(s)) for s in train_seqs}
        n_train = sum(_canon(s) in tr_c for s in seqs)
    h = hashlib.sha1()
    for s in seqs:
        h.update(s.encode())
    h.update(y.tobytes())
    info = {
        "path": str(p),
        "n": len(seqs),
        "oracle_id": stamp,
        "sha1_16": h.hexdigest()[:16],
        "n_test_overlap": n_test,
        "n_train_overlap": n_train,
        "variant": "",
    }
    if "provenance" in z.files:
        import json

        try:
            info["variant"] = json.loads(str(z["provenance"])).get("variant", "")
        except Exception:  # noqa: BLE001 - provenance is informational only
            pass
    return seqs, y, info


def _write_marker(m: Path, payload: dict) -> None:
    """Atomic: three shards of one cell start together and must never read a half-written marker."""
    import json
    import os

    tmp = m.with_name(m.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, m)


def _read_marker(m: Path) -> dict:
    import json
    import time

    for _ in range(10):
        try:
            return json.loads(m.read_text())
        except (
            json.JSONDecodeError
        ):  # a concurrent writer; the write itself is atomic, so this only guards odd filesystems
            time.sleep(1.0)
    raise SystemExit(f"{m} is unreadable")


def guard_out_dir(out_dir: str | Path, info: dict) -> None:
    """Refuse to mix validation sets inside one result directory (resume safety) when a val cache is in use.

    First use writes ``val_cache.json`` (path, hash, n, oracle stamp); later uses must present the same content hash. A
    directory that already holds results from the default validation set is refused. The reverse case (default run
    pointed at a val-cache directory) is checked inline by the callers so the default path imports nothing new.
    """
    m = Path(out_dir) / "val_cache.json"
    if m.exists():
        old = _read_marker(m)
        if old.get("sha1_16") != info["sha1_16"]:
            raise SystemExit(
                f"{out_dir} holds results validated on a different set ({old.get('path')} sha {old.get('sha1_16')}); "
                f"refusing to mix with {info['path']} sha {info['sha1_16']}"
            )
    else:
        if list(Path(out_dir).glob("r*_meta.json")):
            raise SystemExit(
                f"{out_dir} already has results from the default validation set; use a fresh --out_dir for --val_cache"
            )
        _write_marker(m, info)


def guard_extra(out_dir: str | Path, infos: dict) -> None:
    """Same idea as guard_out_dir for the cross-scoring sets (--extra_val): every model in a directory must carry the same valpred_* arrays."""
    m = Path(out_dir) / "extra_val.json"
    if m.exists():
        if _read_marker(m) != infos:
            raise SystemExit(
                f"{out_dir}: --extra_val differs from the one this directory was started with ({m}); use a fresh --out_dir"
            )
    else:
        if list(Path(out_dir).glob("r*_meta.json")):
            raise SystemExit(
                f"{out_dir} already has models without --extra_val outputs; use a fresh --out_dir"
            )
        _write_marker(m, infos)
