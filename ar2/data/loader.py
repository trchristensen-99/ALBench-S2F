"""Split loading, label scaling and leakage checks (spec 2.2, 2.4, 14.3).

Pools and validation sets are npz files described by a manifest. ``load_split`` refuses unknown roles, wrong sequence length and non-finite labels;
``verify_no_overlap`` checks, by sequence (and reverse complement), that no training sequence reaches a validation set.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .encode import SEQ_LEN, SequenceError, check_sequences, revcomp
from .manifest import ALLOWED_ROLES, ManifestError, load_manifest, validate_manifest


class LeakageError(ValueError):
    """A training sequence (or its reverse complement) also appears in a validation set."""


@dataclass
class Split:
    seqs: list[str]
    labels: np.ndarray  # (N, T) float32, columns in manifest `tasks` order
    meta: dict
    chrom: np.ndarray | None = None
    tasks: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.seqs)


@dataclass
class LabelScaler:
    """z-scoring with TRAINING statistics only; ``inverse`` restores the original scale before any metric is computed."""

    mu: np.ndarray  # (T,)
    sd: np.ndarray  # (T,)

    @classmethod
    def fit(cls, y_train: np.ndarray) -> LabelScaler:
        y = np.asarray(y_train, np.float64)
        if y.ndim == 1:
            y = y[:, None]
        sd = y.std(0)
        return cls(y.mean(0), np.where(sd > 0, sd, 1.0))

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, np.float64)
        return ((y - self.mu) / self.sd if y.ndim == 2 else (y - self.mu[0]) / self.sd[0]).astype(
            np.float32
        )

    def inverse(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, np.float64)
        return (z * self.sd + self.mu if z.ndim == 2 else z * self.sd[0] + self.mu[0]).astype(
            np.float32
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(
    manifest: str | Path | dict, expect_role: str | None = None, seq_len: int = SEQ_LEN
) -> Split:
    """Load the npz a manifest points to.

    ``manifest`` is a path to a manifest JSON or an already-parsed dict. ``expect_role`` (when given) must equal the manifest's role. Rows are selected by
    ``index_file`` when present, otherwise the first ``D`` rows are taken when the manifest has a ``D`` smaller than the file (the standard way to read a
    nested 10k/30k prefix out of a bigger pool).
    """
    m = validate_manifest(dict(manifest)) if isinstance(manifest, dict) else load_manifest(manifest)
    if expect_role is not None and m["role"] != expect_role:
        raise ManifestError(f"expected role {expect_role!r}, manifest has {m['role']!r}")
    path = Path(m["path"])
    if m.get("sha256"):
        got = _sha256(path)
        if got != m["sha256"]:
            raise ManifestError(f"sha256 mismatch for {path}: {got} != {m['sha256']}")
    z = np.load(path, allow_pickle=True)
    seqs_raw = z[m.get("seq_key", "sequences")]
    idx = None
    if m.get("index_file"):
        idx = np.load(m["index_file"])
    elif m.get("D") and len(seqs_raw) > int(m["D"]) and m["role"] == "train":
        idx = np.arange(int(m["D"]))
    cols = [
        np.asarray(z[m["label_keys"][t]], np.float32).reshape(len(seqs_raw), -1)[:, 0]
        for t in m["tasks"]
    ]
    y = np.stack(cols, 1)
    chrom = np.asarray(z[m["chrom_key"]]) if m.get("chrom_key") else None
    if idx is not None:
        seqs_raw, y = seqs_raw[idx], y[idx]
        chrom = chrom[idx] if chrom is not None else None
    seqs = [str(s).upper() for s in seqs_raw]
    check_sequences(seqs, seq_len)
    if not np.isfinite(y).all():
        raise ManifestError(f"{path}: non-finite labels")
    if m.get("n") and int(m["n"]) != len(seqs) and idx is None:
        raise ManifestError(f"{path}: manifest says n={m['n']} but the file has {len(seqs)} rows")
    return Split(seqs=seqs, labels=y, meta=m, chrom=chrom, tasks=list(m["tasks"]))


def _canon(s: str) -> str:
    r = revcomp(s)
    return s if s <= r else r


def verify_no_overlap(
    train_seqs: list[str], *others: list[str], names: tuple[str, ...] = ()
) -> None:
    """Raise :class:`LeakageError` if any training sequence or its reverse complement occurs in any of ``others``."""
    tr = {_canon(s) for s in train_seqs}
    for i, o in enumerate(others):
        hit = sum(1 for s in o if _canon(s) in tr)
        if hit:
            nm = names[i] if i < len(names) else f"set {i}"
            raise LeakageError(
                f"{hit} sequences of {nm} also occur (as is or reverse-complemented) in the training pool"
            )


def load_all(
    train_manifest: str | dict,
    val_loop_manifest: str | dict,
    val_select_manifest: str | dict | None = None,
    check_overlap: bool = True,
) -> tuple[Split, Split, Split | None]:
    """Load train / val_loop / (optionally) val_select with the role and overlap checks of spec 2.4."""
    tr = load_split(train_manifest, "train")
    vl = load_split(val_loop_manifest, "val_loop")
    vs = load_split(val_select_manifest, "val_select") if val_select_manifest else None
    if tr.tasks[:1] != vl.tasks[:1]:
        raise ManifestError(
            f"primary task differs: train {tr.tasks[:1]} vs val_loop {vl.tasks[:1]}"
        )
    if check_overlap:
        others = [vl.seqs] + ([vs.seqs] if vs is not None else [])
        verify_no_overlap(tr.seqs, *others, names=("val_loop", "val_select"))
        if vs is not None:
            verify_no_overlap(vl.seqs, vs.seqs, names=("val_select (against val_loop)",))
    return tr, vl, vs


def add_label_noise(y_train: np.ndarray, sd: float, seed: int = 0) -> np.ndarray:
    """Optional simulated measurement noise (``label_noise_sd``, in original label units) on TRAINING labels only."""
    if not sd:
        return y_train
    rng = np.random.default_rng(seed)
    return (y_train + rng.normal(0, sd, y_train.shape)).astype(np.float32)


__all__ = [
    "ALLOWED_ROLES",
    "LabelScaler",
    "LeakageError",
    "SequenceError",
    "Split",
    "add_label_noise",
    "load_all",
    "load_split",
    "verify_no_overlap",
]
