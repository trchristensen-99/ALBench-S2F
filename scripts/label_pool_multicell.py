"""Label sequences with the K562 AND HepG2 oracle_v3 ensembles, plus their difference.

Why this exists: the end application is cell-type SPECIFICITY, so every pool and evaluation
set needs a K562 label, a HepG2 label and a differential (K562 - HepG2), all produced by
oracles trained under the same protocol (same chromosome-CV fold map, same recipe).

Design decisions that are not obvious from the code:

  PER-FOLD PREDICTIONS ARE KEPT. The sidecar stores the (10, N) per-fold matrix for each cell.
  Routing (out-of-fold vs ensemble) is then a re-indexing, not a GPU re-run, and the cross-fold
  std (the epistemic-disagreement / extrapolation detector) comes for free.

  TWO ROUTING MODES, chosen per input file.
    ensemble  mean over all 10 folds. Correct for synthetic sequences, which no fold has seen,
              and for reservoir pools (matches how every stored K562 pool label was made).
    oof       a sequence that is in the oracle's 856,252-row training pool is scored ONLY by a
              fold that did not train on it: its test-fold model, else (duplicates spanning two
              folds) a model whose val fold covers every copy. Sequences not in the pool fall
              back to the ensemble mean. Use for held-out-chromosome REAL sets (genomic, snv,
              ood): the ensemble mean is in-sample for 8 of 10 folds there. Membership is
              decided by exact sequence match, not by set name, so it cannot be mislabelled.
    A sequence the cell's oracle never saw (e.g. the 22,962 designed rows under the partial
    HepG2 oracle, whose pool is 833,290 rows) is clean for every fold and gets the ensemble.

  SIDECAR, NOT IN-PLACE. Inputs (including the battery, which has a provenance guard) are never
  modified; output goes to <outdir>/<stem>.multicell.npz, written atomically.

  FIDELITY CHECK IS BUILT IN. If the input carries a stored K562 label (`oracle_mean` or
  `oracle_labels`), the K562 ensemble mean is compared with it and the correlation is stamped
  into the provenance. That is the measurement that this re-implementation of the live loader
  in experiments/exp1_1_scaling.py reproduces the labels already used everywhere.

  PAD MODE. `legacy` reproduces the K562 live loader (centred N pad for <200 bp, centre crop for
  >200 bp). Training used `context` (real plasmid flank fills the gap). Identical for 200 bp
  sequences, which is every generated pool. Both cells always use the same mode.

Usage::

    python scripts/label_pool_multicell.py label --in A.npz B.npz --outdir D [--mode oof]
    python scripts/label_pool_multicell.py relabel-eval --outdir D          # the 18 eval sets
    python scripts/label_pool_multicell.py relabel-reps --outdir D          # 30k replicate pools
    python scripts/label_pool_multicell.py status                           # oracle readiness

Output keys (SNV pairs get `ref_`/`alt_` prefixes plus `*_delta`): `{k562,hepg2}_label`,
`diff_label` (= k562 - hepg2), `{k562,hepg2}_ens|std|folds|source`, `diff_ens`, `pool_first_idx`,
`pool_fold_mask`, `provenance` (JSON). `source`: 0 ensemble, 1 OOF test-fold, 2 OOF val-fold,
3 in-pool but no clean fold (leaky ensemble fallback), 4 pool row unseen by this cell's oracle.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

N_FOLDS = 10
POOLMAP = REPO / "data" / "k562" / "oracle_poolmap_v2.npy"
BATTERY = REPO / "data" / "k562" / "test_sets_ag_s2_chrsplit"

# Oracle ensembles, keyed by cell. Overridable per call (--k562-dir / --hepg2-dir).
DEFAULT_ORACLES = {
    "k562": REPO / "outputs" / "oracle_v3" / "s2",
    "hepg2": REPO / "outputs" / "oracle_v3_hepg2" / "s2",
}
# The earlier HepG2 oracle trained on ref+alt only (no designed rows). Provisional stand-in.
PARTIAL_HEPG2 = REPO / "outputs" / "oracle_hepg2_v3" / "s2"

# The 18 evaluation sets of the program: experiments/scaling_hp_search.py TEST_SET_NAMES with
# `snv` split into snv_ref / snv_alt (one file, paired schema). Copied here rather than imported
# because that module drags in the whole training stack.
EVAL_SET_NAMES = [
    "genomic", "ood", "snv", "random_32k", "gc_sweep", "dinuc_shuffle",
    "sub_low", "sub_med", "sub_high", "ins_low", "ins_med", "ins_high",
    "del_low", "del_med", "del_high", "translocation", "inversion",
]  # fmt: skip
# Held-out-chromosome REAL sets: sequences sit inside the oracle pool, so score them out-of-fold.
OOF_EVAL_SETS = {"genomic", "ood", "snv"}


# ---------------------------------------------------------------- oracle bookkeeping


def oracle_info(oracle_dir: Path, need_all: bool = True) -> dict:
    """Describe an oracle ensemble from its on-disk artefacts (never from its name)."""
    folds, metrics = [], {}
    for k in range(N_FOLDS):
        ck = oracle_dir / f"fold_{k}" / "best_model" / "checkpoint"
        tm = oracle_dir / f"fold_{k}" / "test_metrics.json"
        if ck.exists() and tm.exists():
            folds.append(k)
            m = json.loads(tm.read_text())
            metrics[k] = {
                "test_pearson": (m.get("test_metrics") or {}).get("pearson"),
                "test_n": (m.get("test_metrics") or {}).get("n"),
                "best_val_pearson": m.get("best_val_pearson"),
                "best_epoch": m.get("best_epoch"),
                "pool": m.get("pool"),
                "pad_mode": m.get("pad_mode"),
                "head_name": m.get("head_name"),
                "ckpt_mtime_utc": datetime.fromtimestamp(
                    ck.stat().st_mtime, timezone.utc
                ).isoformat(),
            }
    if need_all and len(folds) != N_FOLDS:
        raise FileNotFoundError(
            f"{oracle_dir}: {len(folds)}/{N_FOLDS} folds complete (need best_model/checkpoint AND "
            f"test_metrics.json). Train the rest, or pass an explicit --k562-dir/--hepg2-dir "
            f"(e.g. {PARTIAL_HEPG2.relative_to(REPO)} for the provisional partial HepG2 oracle)."
        )
    pools = {m["pool"] for m in metrics.values()}
    pads = {m["pad_mode"] for m in metrics.values()}
    if len(pools) > 1 or len(pads) > 1:
        raise ValueError(f"{oracle_dir}: folds disagree on pool {pools} / pad_mode {pads}")
    return {
        "dir": str(oracle_dir.relative_to(REPO))
        if oracle_dir.is_relative_to(REPO)
        else str(oracle_dir),
        "oracle_id": oracle_dir.parent.name,
        "folds": folds,
        "pool_rows_trained": int(pools.pop()) if pools else None,
        "train_pad_mode": pads.pop() if pads else None,
        "mean_test_pearson": float(np.mean([m["test_pearson"] for m in metrics.values()]))
        if metrics
        else None,
        "per_fold": metrics,
    }


# ---------------------------------------------------------------- encoding

_LUT = np.full(256, 4, dtype=np.uint8)  # 4 = N / anything else -> all-zero one-hot
for _i, _c in enumerate("ACGT"):
    _LUT[ord(_c)] = _i
    _LUT[ord(_c.lower())] = _i


def _flank_idx(seq: str) -> np.ndarray:
    return _LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def encode_indices(seqs: list[str], pad_mode: str) -> np.ndarray:
    """(N, 600) int8 base indices (0-3 = ACGT, 4 = N). One-hot happens on device.

    Index form is 600 B/sequence, so a million sequences fit in RAM and the encoding is done once
    for all 10 folds x 2 strands x 2 cells.
    """
    from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM

    if pad_mode not in ("legacy", "context"):
        raise ValueError(f"pad_mode must be legacy|context, got {pad_mode!r}")
    up = _flank_idx(MPRA_UPSTREAM[-300:])
    dn = _flank_idx(MPRA_DOWNSTREAM[:300])
    assert len(up) == 300 and len(dn) == 300
    out = np.full((len(seqs), 600), 4, dtype=np.int8)
    for i, s in enumerate(seqs):
        core = _flank_idx(s.upper())
        if pad_mode == "legacy":
            # experiments/exp1_1_scaling.py::_AGS2Oracle._encode: centred N pad / centre crop.
            L = len(core)
            if L < 200:
                left = (200 - L) // 2
                core = np.concatenate(
                    [np.full(left, 4, np.uint8), core, np.full(200 - L - left, 4, np.uint8)]
                )
            elif L > 200:
                st = (L - 200) // 2
                core = core[st : st + 200]
            out[i, :200] = up[-200:]
            out[i, 200:400] = core
            out[i, 400:] = dn[:200]
        else:
            # train_oracle_s2_v2.py collate(): centred 600 bp window of UP[300]+core[:200]+DOWN[300].
            core = core[:200]
            wide = np.concatenate([up, core, dn])
            st = (len(wide) - 600) // 2
            out[i] = wide[st : st + 600]
    return out


# ---------------------------------------------------------------- inference


class _Engine:
    """Per-fold AlphaGenome inference, one fold resident on the GPU at a time."""

    def __init__(self, chunk: int):
        import jax
        import jax.numpy as jnp
        import orbax.checkpoint as ocp
        from alphagenome_ft import create_model_with_heads

        from albench.paths import resolve
        from models.alphagenome_heads import register_s2f_head

        self.jax, self.jnp, self.chunk = jax, jnp, chunk
        self._ocp = ocp
        self._create = create_model_with_heads
        self._register = register_s2f_head
        self._weights = str(resolve("ag_weights"))
        self._models: dict[str, tuple] = {}

    def _model_for(self, params) -> tuple:
        """One model object per head name; the head name must equal the one in the checkpoint."""
        names = sorted({k.split("/")[1] for k in params if k.startswith("head/") and "/" in k[5:]})
        if len(names) != 1:
            raise ValueError(f"expected exactly one head in checkpoint, found {names}")
        head = names[0]
        if head not in self._models:
            jax, jnp = self.jax, self.jnp
            self._register(
                head_name=head, arch="boda-flatten-512-512", task_mode="human",
                num_tracks=1, dropout_rate=0.1,
            )  # fmt: skip
            model = self._create(
                "all_folds", heads=[head], checkpoint_path=self._weights,
                use_encoder_output=True, detach_backbone=False,
            )  # fmt: skip

            @jax.jit
            def step(p, state, idx):
                x = jax.nn.one_hot(idx.astype(jnp.int32), 4, dtype=jnp.float32)  # N (=4) -> zeros
                out = model._predict(
                    p, state, x, jnp.zeros(x.shape[0], dtype=jnp.int32),
                    requested_outputs=[head],
                    negative_strand_mask=jnp.zeros(x.shape[0], dtype=bool),
                    strand_reindexing=None,
                )[head]  # fmt: skip
                return jnp.reshape(out, (-1,))

            self._models[head] = (model, step, head)
        return self._models[head]

    def predict_fold(self, ckpt: Path, idx: np.ndarray) -> np.ndarray:
        """Forward + reverse-complement mean for every row of `idx` under one fold checkpoint."""
        jax = self.jax
        params, _ = self._ocp.StandardCheckpointer().restore(str(ckpt.resolve()))
        params = jax.device_put(params)
        model, step, _ = self._model_for(params)
        n, c = len(idx), self.chunk
        out = np.zeros(n, dtype=np.float32)
        for i in range(0, n, c):
            b = idx[i : i + c]
            m = len(b)
            if m < c:  # pad to a fixed shape so XLA compiles once, not once per tail length
                b = np.concatenate([b, np.full((c - m, 600), 4, dtype=np.int8)])
            rc = np.where(b < 4, 3 - b, b)[:, ::-1]  # complement ACGT<->TGCA, reverse positions
            pf = np.asarray(step(params, model._state, jax.numpy.asarray(b)))
            pr = np.asarray(step(params, model._state, jax.numpy.asarray(np.ascontiguousarray(rc))))
            out[i : i + m] = ((pf + pr) / 2.0)[:m]
        del params
        return out


def predict_folds(engine: _Engine, oracle_dir: Path, idx: np.ndarray, folds, log) -> np.ndarray:
    """(len(folds), N) float32 per-fold predictions."""
    res = np.zeros((len(folds), len(idx)), dtype=np.float32)
    for j, k in enumerate(folds):
        t = time.time()
        res[j] = engine.predict_fold(oracle_dir / f"fold_{k}" / "best_model" / "checkpoint", idx)
        log(f"    {oracle_dir.parent.name} fold {k}: {len(idx):,} seq in {time.time() - t:,.0f}s")
    return res


# ---------------------------------------------------------------- pool membership / routing


def pool_index() -> tuple[dict, np.ndarray]:
    """seq -> [fold bitmask over every pool copy, first pool row]. The pool order is the
    oracle's training order (ref, alt, OOD), exactly what train_oracle_s2_v2.py indexes."""
    from scripts.build_full_oracle_cache import load_all_sequences

    seqs, _ = load_all_sequences()
    fmap = np.load(POOLMAP)
    if len(seqs) != len(fmap):
        raise ValueError(f"pool/fold-map length mismatch {len(seqs)} vs {len(fmap)}")
    d: dict[str, list[int]] = {}
    for i, (s, f) in enumerate(zip(seqs, fmap.tolist())):
        e = d.get(s.upper())
        if e is None:
            d[s.upper()] = [1 << f, i]
        else:
            e[0] |= 1 << f
    return d, fmap


def _weights_for_mask(mask: int) -> tuple[np.ndarray, int]:
    """Fold weights for a pool sequence whose copies sit in the folds of `mask`."""
    w = np.zeros(N_FOLDS, dtype=np.float32)
    for j in range(N_FOLDS):  # test-fold model: every copy is in that model's held-out fold
        if mask == 1 << j:
            w[j] = 1.0
            return w, 1
    ok = [j for j in range(N_FOLDS) if mask & ~((1 << j) | (1 << ((j + 1) % N_FOLDS))) == 0]
    if ok:  # copies split over a model's test + val folds: never trained on, val used only to stop
        w[ok] = 1.0 / len(ok)
        return w, 2
    return np.full(N_FOLDS, 1.0 / N_FOLDS, dtype=np.float32), 3  # no clean model exists


def route(masks: np.ndarray, first: np.ndarray, mode: str, pool_rows_trained: int | None):
    """(N, 10) fold weights and (N,) source codes. See module docstring for the codes."""
    n = len(masks)
    W = np.full((n, N_FOLDS), 1.0 / N_FOLDS, dtype=np.float32)
    src = np.zeros(n, dtype=np.int8)
    if mode == "ensemble":
        return W, src
    cache = {int(m): _weights_for_mask(int(m)) for m in np.unique(masks[masks > 0])}
    for m, (w, code) in cache.items():
        sel = masks == m
        if pool_rows_trained is not None:
            unseen = sel & (first >= pool_rows_trained)
            if unseen.any():
                src[unseen] = 4
            sel = sel & ~unseen
        W[sel] = w
        src[sel] = code
    return W, src


# ---------------------------------------------------------------- file handling


def _groups(z) -> dict[str, str]:
    if "sequences" in z.files:
        return {"": "sequences"}
    if "ref_sequences" in z.files and "alt_sequences" in z.files:
        return {"ref_": "ref_sequences", "alt_": "alt_sequences"}
    raise KeyError(f"no `sequences` or `ref_/alt_sequences` key; keys={z.files}")


def _stored_k562(z, prefix: str):
    """The K562 label the input already carries: `oracle_mean`/`oracle_labels`, or ref_mean/alt_mean."""
    keys = [f"{prefix.rstrip('_')}_mean"] if prefix else ["oracle_mean", "oracle_labels"]
    for k in keys:
        if k in z.files:
            return np.asarray(z[k], dtype=np.float64)
    return None


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, timeout=20
        ).strip()
    except Exception:
        return "unknown"


def _corr(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else float("nan")


def label_files(
    jobs: list[tuple[Path, str, Path]],
    cells=("k562", "hepg2"),
    k562_dir: Path | None = None,
    hepg2_dir: Path | None = None,
    pad_mode: str = "legacy",
    chunk: int = 128,
    overwrite: bool = False,
    log=print,
    debug_folds: list[int] | None = None,
) -> list[Path]:
    """jobs = [(input_npz, mode, output_npz)]. All sequences of all jobs are deduplicated and
    scored once per (cell, fold), so model loading is amortised over the whole batch."""
    t0 = time.time()
    dirs = {
        "k562": k562_dir or DEFAULT_ORACLES["k562"],
        "hepg2": hepg2_dir or DEFAULT_ORACLES["hepg2"],
    }
    info = {c: oracle_info(Path(dirs[c])) for c in cells}
    if debug_folds:  # smoke tests only: score a subset of folds; stamped into the provenance
        for c in cells:
            info[c]["folds"] = [k for k in info[c]["folds"] if k in debug_folds]
    for c in cells:
        if info[c]["train_pad_mode"] != "context":
            log(f"WARNING {c} oracle trained with pad_mode={info[c]['train_pad_mode']!r}")

    todo = [(i, m, o) for i, m, o in jobs if overwrite or not (o.exists() and o.stat().st_size > 0)]
    for i, m, o in jobs:
        if (i, m, o) not in todo:
            log(f"SKIP existing {o}")
    if not todo:
        return [o for _, _, o in jobs]

    loaded = {}
    for src, mode, _ in todo:
        z = np.load(src, allow_pickle=True)
        loaded[src] = (z, {p: [str(s) for s in z[k]] for p, k in _groups(z).items()})
    uniq: dict[str, int] = {}
    for _, g in loaded.values():
        for seqs in g.values():
            for s in seqs:
                uniq.setdefault(s.upper(), len(uniq))
    useqs = list(uniq)
    log(f"{len(todo)} files, {sum(len(s) for _, g in loaded.values() for s in g.values()):,} sequences, "
        f"{len(useqs):,} unique; pad_mode={pad_mode}")  # fmt: skip

    t = time.time()
    idx = encode_indices(useqs, pad_mode)
    log(f"encoded in {time.time() - t:.0f}s")
    pool, _ = pool_index()
    umask = np.zeros(len(useqs), dtype=np.int16)
    ufirst = np.full(len(useqs), -1, dtype=np.int64)
    for j, s in enumerate(useqs):
        e = pool.get(s)
        if e is not None:
            umask[j], ufirst[j] = e
    log(f"{int((umask > 0).sum()):,}/{len(useqs):,} unique sequences are in the oracle pool")

    engine = _Engine(chunk)
    ufolds = {}
    for c in cells:
        log(f"scoring cell {c} with {info[c]['dir']}")
        ufolds[c] = predict_folds(engine, Path(dirs[c]), idx, info[c]["folds"], log)
    score_s = time.time() - t0

    outs = []
    for src, mode, dst in todo:
        z, g = loaded[src]
        o: dict[str, np.ndarray | str] = {}
        stored_check, counts = {}, {}
        for p, seqs in g.items():
            rows = np.array([uniq[s.upper()] for s in seqs])
            masks, first = umask[rows].astype(np.int64), ufirst[rows]
            o[f"{p}pool_fold_mask"], o[f"{p}pool_first_idx"] = (
                masks.astype(np.int16),
                first.astype(np.int32),
            )
            for c in cells:
                F = ufolds[c][:, rows]  # (10, n)
                W, code = route(masks, first, mode, info[c]["pool_rows_trained"])
                ens, std = F.mean(0), F.std(0)
                o[f"{p}{c}_folds"], o[f"{p}{c}_ens"], o[f"{p}{c}_std"] = F, ens, std
                Wu = W[:, info[c]["folds"]]  # all 10 folds unless --debug-folds; renormalised
                o[f"{p}{c}_label"] = ((Wu.T * F).sum(0) / Wu.sum(1)).astype(np.float32)
                o[f"{p}{c}_source"] = code
                counts[f"{p}{c}"] = {
                    int(k): int(v) for k, v in zip(*np.unique(code, return_counts=True))
                }
            if set(cells) == {"k562", "hepg2"}:
                o[f"{p}diff_label"] = (o[f"{p}k562_label"] - o[f"{p}hepg2_label"]).astype(
                    np.float32
                )
                o[f"{p}diff_ens"] = (o[f"{p}k562_ens"] - o[f"{p}hepg2_ens"]).astype(np.float32)
            st = _stored_k562(z, p) if "k562" in cells else None
            if st is not None and len(st) == len(seqs):
                e = o[f"{p}k562_ens"]
                stored_check[p or "all"] = {
                    "r_ens_vs_stored": _corr(e, st),
                    "max_abs_diff": float(np.nanmax(np.abs(e - st))),
                    "n_short_lt200": int(sum(len(s) < 200 for s in seqs)),
                }
        if (
            "ref_" in g and "alt_" in g
        ):  # SNV pairs: delta only meaningful when both alleles share a route
            for c in cells:
                o[f"{c}_delta"] = (o[f"alt_{c}_label"] - o[f"ref_{c}_label"]).astype(np.float32)
            same = np.ones(len(g["ref_"]), dtype=bool)
            for c in cells:
                same &= o[f"ref_{c}_source"] == o[f"alt_{c}_source"]
            o["delta_same_route"] = same
            if set(cells) == {"k562", "hepg2"}:
                o["diff_delta"] = (o["k562_delta"] - o["hepg2_delta"]).astype(np.float32)
        prov = {
            "script": "scripts/label_pool_multicell.py",
            "git_commit": _git_commit(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "input": str(src), "input_sha256": _sha256(src),
            "mode": mode, "pad_mode": pad_mode, "chunk": chunk,
            "diff_definition": "diff_label = k562_label - hepg2_label",
            "source_codes": "0 ensemble, 1 OOF test-fold, 2 OOF val-fold, 3 in-pool no clean fold, 4 pool row unseen by this cell's oracle",
            "route_counts": counts,
            "k562_stored_check": stored_check,
            "oracles": {c: {k: v for k, v in info[c].items()} for c in cells},
            "poolmap": str(POOLMAP.relative_to(REPO)),
            "debug_folds": debug_folds,
            "n_files_in_batch": len(todo), "n_unique_in_batch": len(useqs),
            "batch_scoring_seconds": round(score_s),
        }  # fmt: skip
        for k in z.files:  # pass-through of everything the input already carried
            o[k if (k not in o and k != "provenance") else f"input_{k}"] = z[k]
        o["provenance"] = json.dumps(prov, default=str)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp.npz")
        np.savez_compressed(tmp, **o)
        os.replace(tmp, dst)  # atomic: a killed job never leaves a truncated file behind
        outs.append(dst)
        for k, v in stored_check.items():
            log(f"  {dst.name} [{k}] K562 ensemble vs stored label: r={v['r_ens_vs_stored']:.6f} "
                f"max|d|={v['max_abs_diff']:.4g} (n<200bp={v['n_short_lt200']})")  # fmt: skip
        log(f"  wrote {dst}")
    log(f"done in {time.time() - t0:,.0f}s")
    return outs


# ---------------------------------------------------------------- CLI


def _eval_jobs(outdir: Path):
    jobs = []
    for name in EVAL_SET_NAMES:
        f = BATTERY / f"{name}_oracle.npz"
        if not f.exists() and name == "random_32k":
            f = BATTERY / "random_10k_oracle.npz"
        if f.exists():
            jobs.append(
                (
                    f,
                    "oof" if name in OOF_EVAL_SETS else "ensemble",
                    outdir / f"{f.stem}.multicell.npz",
                )
            )
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["label", "relabel-eval", "relabel-reps", "status"])
    ap.add_argument("--in", dest="inputs", nargs="*", default=[], help="input .npz files (label)")
    ap.add_argument("--glob", default=str(REPO / "outputs/grid_pools/k562_reps/*_r1_d30000.npz"))
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--mode", choices=["ensemble", "oof"], default="ensemble")
    ap.add_argument("--pad-mode", choices=["legacy", "context"], default="legacy")
    ap.add_argument("--cells", default="k562,hepg2")
    ap.add_argument("--k562-dir", type=Path, default=None)
    ap.add_argument("--hepg2-dir", type=Path, default=None)
    ap.add_argument("--chunk", type=int, default=int(os.environ.get("AG_ORACLE_CHUNK", "128")))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-files", type=int, default=None, help="debug: first N files only")
    ap.add_argument("--debug-folds", default=None, help="debug: comma list of folds to score")
    a = ap.parse_args()

    if a.cmd == "status":
        # Readiness + the per-fold held-out Pearson table (a fold counts once checkpoint AND
        # test_metrics.json exist). Pass --hepg2-dir to inspect another ensemble.
        dirs = {
            "k562": a.k562_dir or DEFAULT_ORACLES["k562"],
            "hepg2": a.hepg2_dir or DEFAULT_ORACLES["hepg2"],
        }
        infos = {}
        for c, d in dirs.items():
            try:
                infos[c] = oracle_info(Path(d), need_all=False)
                i = infos[c]
                print(f"{c}: {i['dir']}  folds {len(i['folds'])}/10  pool_rows={i['pool_rows_trained']}  "
                      f"mean test r={i['mean_test_pearson']}")  # fmt: skip
            except Exception as e:
                print(f"{c}: {d}: {e}")
        if len(infos) == 2:
            print("fold  k562_test_r  hepg2_test_r  hepg2_n_test  hepg2_best_epoch")
            for k in range(N_FOLDS):
                kr, hr = infos["k562"]["per_fold"].get(k), infos["hepg2"]["per_fold"].get(k)
                print(f"{k:>4}  {kr['test_pearson'] if kr else float('nan'):>11.4f}  "
                      f"{hr['test_pearson'] if hr else float('nan'):>12.4f}  "
                      f"{hr['test_n'] if hr else 0:>12}  {hr['best_epoch'] if hr else '-':>16}")  # fmt: skip
        return 0

    if a.outdir is None:
        ap.error("--outdir is required")
    if a.cmd == "label":
        jobs = [(Path(p), a.mode, a.outdir / f"{Path(p).stem}.multicell.npz") for p in a.inputs]
    elif a.cmd == "relabel-eval":
        jobs = _eval_jobs(a.outdir)
    else:
        jobs = [
            (Path(p), "ensemble", a.outdir / f"{Path(p).stem}.multicell.npz")
            for p in sorted(glob.glob(a.glob))
        ]
    if a.max_files:
        jobs = jobs[: a.max_files]
    if not jobs:
        raise SystemExit("no input files")
    label_files(
        jobs,
        tuple(a.cells.split(",")),
        a.k562_dir,
        a.hepg2_dir,
        a.pad_mode,
        a.chunk,
        a.overwrite,
        log=lambda s: print(s, flush=True),
        debug_folds=[int(x) for x in a.debug_folds.split(",")] if a.debug_folds else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
