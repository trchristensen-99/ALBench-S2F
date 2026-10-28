"""Label-free stress screen (spec 9.4): 4,000 unlabelled sequences generated once per search from a fixed seed. A run fails (status exploded) if any
prediction is non-finite or more than ``stress_max_abs_z`` training-label SDs from the training mean.

Composition: 1,000 uniform random; 500 at GC 0.2 and 500 at GC 0.8; 1,000 dinucleotide shuffles of val_loop sequences; 500 low-complexity (homopolymers,
long homopolymer runs, short tandem repeats); 500 val_loop sequences with 10% substitutions.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ALPHA = np.array(list("ACGT"))
STRESS_SEED = 20261023


def _random_seqs(rng, n: int, length: int = 200, gc: float = 0.5) -> list[str]:
    p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
    return ["".join(ALPHA[rng.choice(4, length, p=p)]) for _ in range(n)]


def dinuc_shuffle(seq: str, rng: np.random.Generator) -> str:
    """Altschul-Erickson dinucleotide-preserving shuffle (Eulerian path with a random arborescence of last exits)."""
    if len(seq) < 3:
        return seq
    edges: dict[str, list[str]] = {}
    for a, b in zip(seq[:-1], seq[1:], strict=True):
        edges.setdefault(a, []).append(b)
    last = seq[-1]
    for _ in range(200):
        shuf = {v: list(rng.permutation(e)) for v, e in edges.items()}
        tree = {v: e[-1] for v, e in shuf.items() if v != last}
        ok = True
        for v in tree:
            seen, u = {v}, v
            while u != last:
                u = tree.get(u)
                if u is None or u in seen:
                    ok = False
                    break
                seen.add(u)
            if not ok:
                break
        if ok:
            break
    else:  # pragma: no cover - vanishingly unlikely for 4 letters
        return "".join(rng.permutation(list(seq)))
    out, v = [seq[0]], seq[0]
    for _ in range(len(seq) - 1):
        v = shuf[v].pop(0)
        out.append(v)
    return "".join(out)


def _low_complexity(rng, n: int, length: int = 200) -> list[str]:
    out = []
    for i in range(n):
        kind = i % 3
        if kind == 0:  # pure homopolymer
            out.append(str(ALPHA[rng.integers(4)]) * length)
        elif kind == 1:  # long homopolymer run inside random sequence
            run = int(rng.integers(80, 160))
            start = int(rng.integers(0, length - run))
            s = list(_random_seqs(rng, 1, length)[0])
            s[start : start + run] = str(ALPHA[rng.integers(4)]) * run
            out.append("".join(s))
        else:  # short tandem repeat
            unit = "".join(ALPHA[rng.integers(4, size=int(rng.integers(2, 7)))])
            out.append((unit * (length // len(unit) + 1))[:length])
    return out


def _substitute(rng, seq: str, frac: float = 0.10) -> str:
    s = list(seq)
    for i in rng.choice(len(s), int(round(frac * len(s))), replace=False):
        s[i] = str(rng.choice([c for c in "ACGT" if c != s[i]]))
    return "".join(s)


def make_stress_set(val_seqs: list[str], seed: int = STRESS_SEED) -> list[str]:
    """The 4,000-sequence stress set; deterministic for a given seed and validation set."""
    rng = np.random.default_rng(seed)
    pick = lambda n: [val_seqs[i] for i in rng.choice(len(val_seqs), n, replace=len(val_seqs) < n)]  # noqa: E731
    out = _random_seqs(rng, 1000)
    out += _random_seqs(rng, 500, gc=0.2) + _random_seqs(rng, 500, gc=0.8)
    out += [dinuc_shuffle(s, rng) for s in pick(1000)]
    out += _low_complexity(rng, 500)
    out += [_substitute(rng, s) for s in pick(500)]
    assert len(out) == 4000
    return out


def save_stress_set(path: str | Path, seqs: list[str]) -> None:
    """Store as a fixed-width unicode array (loadable without pickle)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f"{path.stem}.{os.getpid()}.tmp.npy"
    )  # unique per process: several workers may create the file at once
    np.save(tmp, np.array(seqs, dtype="U200"))
    tmp.replace(path)


def load_stress_set(path: str | Path) -> list[str]:
    return [str(s) for s in np.load(path, allow_pickle=False)]


def stress_check(pred_z: np.ndarray, max_abs_z: float = 25.0) -> dict:
    """Spec rule: every prediction finite and within ``max_abs_z`` training SDs of the training mean (``pred_z`` is on the z-scored scale)."""
    p = np.asarray(pred_z, np.float64).ravel()
    finite = bool(np.isfinite(p).all())
    mx = float(np.max(np.abs(p))) if finite and len(p) else float("inf")
    return {"finite": finite, "max_abs_z": mx, "exploded": (not finite) or mx > max_abs_z}
