"""Shared pieces of the Oct-28 benchmark of our genomic ensembles against single and many MPRA-LegNet and AlphaGenome-FT-MPRA students.

Everything is scored the way the menu retrains are, so one analysis script (scripts/analysis/analyze_bench33.py) scores all arms with the same stacking
code: a model directory holds labels.npz plus r{seed:02d}_menu_{idx:03d}.npz / _meta.json with `val_pred` and `test_pred_<set>` arrays on the 18 sets of
scaling_hp_search.load_all_test_sets(). Predictions are reverse-complement averaged (the menu students are); forward-only predictions are kept as `fwd_*`.

Validation is the V0c asset (outputs/val_variants/val_V0.npz, oracle_v3 labels). The menu retrains (sc31_*) validated on outputs/chr_split_cache/
chr_val_ref_only.npz, the same 29,449 sequences with different labels (r = 0.983), so their val labels are kept next to ours as `val_labels_cache` and the
analysis re-fits every stack on V0c.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

R = Path(os.environ.get("ALBENCH_ROOT", Path(__file__).resolve().parents[2]))
for p in (R, R / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

NEST = R / "outputs/grid_pools/k562_nested"
OUT_ROOT = R / "outputs/menu_retrain_v0"
_LUT = np.full(256, 255, np.uint8)
for _i, _c in enumerate("ACGT"):
    _LUT[ord(_c)] = _i
    _LUT[ord(_c.lower())] = _i


def onehot_core200(seqs, length: int = 200) -> np.ndarray:
    """(N, 200, 4) float32, ACGT channel order. A sequence shorter than 200 is padded with all-zero columns, centred; a longer one is cropped around its
    centre. This is how oracle_v3 encoded the sequences it labelled (exp1_1_scaling._load_k562_ag_s2_oracle._encode)."""
    out = np.zeros((len(seqs), length, 4), np.float32)
    for n, s in enumerate(seqs):
        b = _LUT[np.frombuffer(s.encode("ascii", "ignore"), np.uint8)]
        L = len(b)
        if L > length:
            st = (L - length) // 2
            b = b[st : st + length]
            L = length
        left = (length - L) // 2
        pos = np.nonzero(b < 4)[0]
        out[n, left + pos, b[pos]] = 1.0
    return out


def flank_onehot(seq: str) -> np.ndarray:
    out = np.zeros((len(seq), 4), np.float32)
    for i, c in enumerate(seq.upper()):
        j = "ACGT".find(c)
        if j >= 0:
            out[i, j] = 1.0
    return out


def mpra_flanks(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The n bases of the lentiMPRA plasmid immediately 5' and 3' of the 200 bp insert, one-hot (n, 4) each."""
    from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM

    return flank_onehot(MPRA_UPSTREAM[-n:]), flank_onehot(MPRA_DOWNSTREAM[:n])


def pearson(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def load_data(D: int, pool_path: str | None = None, smoke: bool = False) -> dict:
    """Training pool (canonical nested genomic pool), V0c validation, and the 18 test sets, all with oracle_v3 labels."""
    from scaling_hp_search import load_all_test_sets, load_chr_test_genomic

    from albench.paths import resolve

    pool = Path(pool_path or NEST / f"genomic_d{D}.npz")
    z = np.load(pool, allow_pickle=True)
    assert str(z["oracle_id"]) == "oracle_v3", (pool, z["oracle_id"])
    tr_s = [str(s) for s in z["sequences"]]
    tr_y = np.asarray(z["oracle_labels"], np.float32)
    assert len(tr_s) == D == len(tr_y) and np.isfinite(tr_y).all(), (len(tr_s), D)

    vz = np.load(resolve("val_v0c"), allow_pickle=True)
    assert str(vz["oracle_id"]) == "oracle_v3", vz["oracle_id"]
    va_s = [str(s) for s in vz["sequences"]]
    va_y = np.asarray(vz["oracle_labels"], np.float32)
    assert np.isfinite(va_y).all()
    cz = np.load(R / "outputs/chr_split_cache/chr_val_ref_only.npz", allow_pickle=True)
    assert [str(s) for s in cz["sequences"]] == va_s, (
        "V0c and the menu's validation file hold different sequences"
    )
    va_y_cache = np.asarray(cz["oracle_labels"], np.float32)
    assert not (set(tr_s) & set(va_s)), "training pool overlaps the validation set"

    te_s, te_oracle, te_true = load_chr_test_genomic()
    tests = load_all_test_sets()
    if smoke:
        tr_s, tr_y, va_s, va_y, va_y_cache = (
            tr_s[:512],
            tr_y[:512],
            va_s[:512],
            va_y[:512],
            va_y_cache[:512],
        )
        te_s, te_oracle, te_true = te_s[:512], te_oracle[:512], te_true[:512]
        tests = {k: (v[0][:512], v[1][:512]) for k, v in tests.items()}
    return dict(
        train_seqs=tr_s,
        train_y=tr_y,
        val_seqs=va_s,
        val_y=va_y,
        val_y_cache=va_y_cache,
        test_seqs=te_s,
        test_oracle=te_oracle,
        test_true=te_true,
        tests=tests,
        pool=str(pool),
    )


def atomic_savez(path: Path, **arrays) -> None:
    tmp = str(path).replace(".npz", f".tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    Path(tmp).write_text(text)
    os.replace(tmp, path)


def save_labels(out_dir: Path, d: dict) -> None:
    """labels.npz in the menu-retrain layout; written once (it is identical for every model of the directory)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "labels.npz"
    if path.exists():
        return
    lab = {
        "val_labels": d["val_y"],
        "val_labels_cache": d["val_y_cache"],
        "test_oracle": d["test_oracle"],
        "test_true": d["test_true"],
    }
    lab.update({f"oracle_{k}": v[1] for k, v in d["tests"].items()})
    atomic_savez(path, **lab)


def save_model(out_dir: Path, name: str, preds: dict, meta: dict) -> None:
    """`name` = r{seed_index:02d}_menu_{idx:03d}; the npz goes first so a meta file always has its arrays."""
    bad = [
        k for k, v in preds.items() if (not np.isfinite(v).all()) or float(np.abs(v).max()) > 1e3
    ]
    if bad:  # the exploded-prediction guard of the menu retrains
        raise RuntimeError(f"non-finite or exploded predictions on {bad[:3]}; not saved")
    atomic_savez(
        out_dir / f"{name}.npz", **{k: np.asarray(v, np.float32) for k, v in preds.items()}
    )
    meta = dict(meta, model_id=name, saved_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    atomic_write_text(out_dir / f"{name}_meta.json", json.dumps(meta, indent=2, default=str))


def eval_report(prefix_preds: dict, d: dict) -> dict:
    """Pearson of reverse-complement-averaged predictions on every set (+ experimental labels on the genomic test set)."""
    out = {"val_pearson": pearson(prefix_preds["val_pred"], d["val_y"])}
    for k, (_s, y) in d["tests"].items():
        out[f"test_{k}"] = pearson(prefix_preds[f"test_pred_{k}"], y)
    out["test_genomic_vs_experiment"] = pearson(prefix_preds["test_pred"], d["test_true"])
    return out
