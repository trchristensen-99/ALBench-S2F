"""Real motif vocabulary from JASPAR PFMs, replacing the 9 hardcoded consensus strings.

WHY THIS REPLACES THE OLD LIST
`albench/reservoir/motif_planted_v2.py` planted from nine hardcoded consensus strings, one of which
('CTCFCC') is not valid DNA. Two problems beyond the typo:

  1. NINE MOTIFS IS NOT A VOCABULARY. A 200 bp oligo holds ~4 motifs, so nine motifs give almost no
     combinatorial space and cannot teach a model to generalise to the motifs that actually appear
     in the evaluation sets.
  2. PLANTING A FIXED CONSENSUS IS THE LEAKAGE MODE WE WERE WARNED ABOUT. If every FOXA1 site is
     the identical string at a predictable spacing, a model can memorise that string and its
     effect rather than learning binding preference. Sampling each instance FROM THE PWM gives the
     natural variation real sites have, so the model must learn the preference to benefit.

So instances are sampled from the position probability matrix, not stamped from a consensus.

SELECTING THE VOCABULARY
Which motifs to include is a scientific choice, not a default, and is deliberately parameterised:
  --taxon        restrict to a JASPAR collection (human CORE by default, via the API)
  --max_motifs   cap the vocabulary size, taking the most informative first
  --cluster_at   collapse near-duplicate PFMs above a similarity threshold, since JASPAR carries
                 many closely related versions of the same factor and treating them as distinct
                 inflates the apparent vocabulary
  --expressed    optional list of TF genes to keep (e.g. expressed in K562), for the cell-type
                 vocabulary arms

Information content is used for ranking because a low-IC motif is nearly a random string: planting
it teaches nothing and it would be found by chance in any background.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MEME_DEFAULT = "/grid/koo/home/shared/cl_procap/annotations/JASPAR2022_CORE_pfms.meme"
_ACGT = np.array(list("ACGT"))


@dataclass
class Motif:
    mid: str
    name: str
    pwm: np.ndarray  # (L, 4) probabilities

    @property
    def length(self) -> int:
        return self.pwm.shape[0]

    @property
    def info_content(self) -> float:
        """Total information content in bits: 2L minus the per-position entropy."""
        p = np.clip(self.pwm, 1e-9, 1.0)
        return float(np.sum(2.0 + np.sum(p * np.log2(p), axis=1)))

    @property
    def consensus(self) -> str:
        return "".join(_ACGT[np.argmax(self.pwm, axis=1)])

    @property
    def per_position_ic(self) -> np.ndarray:
        """Information content in bits at each position, shape (L,)."""
        p = np.clip(self.pwm, 1e-9, 1.0)
        return 2.0 + np.sum(p * np.log2(p), axis=1)

    def sample(self, rng: np.random.Generator, n: int = 1) -> list[str]:
        """Draw instances FROM the PWM, so planted sites carry realistic variation.

        Vectorised by inverse-CDF: reservoirs are generated at the 1M+ scale, where a
        per-position rng.choice loop dominates generation time.
        """
        cum = np.cumsum(self.pwm, axis=1)[:, :3]  # (L, 3)
        u = rng.random((n, self.length))
        idx = (u[:, :, None] >= cum[None, :, :]).sum(axis=2)
        chars = _ACGT[idx]  # (n, L)
        return ["".join(row) for row in chars]

    def trim(self, min_position_ic: float) -> "Motif":
        """Trim terminal positions carrying less than `min_position_ic` bits.

        Many JASPAR PFMs pad the informative core with low-information flanks. Those
        flanks lengthen the motif without adding specificity, and motif length bounds
        how many sites fit on a 200bp oligo -- so the padding silently caps the
        motifs-per-sequence knob. Only the ends are trimmed; interior low-IC
        positions are part of the motif (e.g. the spacer in a dimeric site).
        """
        ic = self.per_position_ic
        keep = np.where(ic >= min_position_ic)[0]
        if len(keep) == 0:
            return self
        lo, hi = int(keep[0]), int(keep[-1]) + 1
        if lo == 0 and hi == self.length:
            return self
        return Motif(self.mid, self.name, self.pwm[lo:hi].copy())

    def revcomp(self) -> "Motif":
        return Motif(self.mid + "_rc", self.name + "(-)", self.pwm[::-1, ::-1].copy())


def parse_meme(path: str = MEME_DEFAULT) -> list[Motif]:
    motifs, cur, rows, width = [], None, [], None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("MOTIF"):
                if cur and rows:
                    motifs.append(Motif(cur[0], cur[1], np.array(rows, dtype=np.float64)))
                parts = line.split()
                mid = parts[1]
                name = parts[2] if len(parts) > 2 else mid
                name = name.split(".")[-1]  # "MA0004.1.Arnt" -> "Arnt"
                cur, rows, width = (mid, name), [], None
            elif line.startswith("letter-probability"):
                m = re.search(r"w=\s*(\d+)", line)
                width = int(m.group(1)) if m else None
                rows = []
            elif cur is not None and width is not None and len(rows) < width:
                vals = line.split()
                if len(vals) == 4:
                    try:
                        rows.append([float(v) for v in vals])
                    except ValueError:
                        pass
    if cur and rows:
        motifs.append(Motif(cur[0], cur[1], np.array(rows, dtype=np.float64)))
    for m in motifs:  # renormalise defensively
        m.pwm /= np.clip(m.pwm.sum(axis=1, keepdims=True), 1e-9, None)
    return motifs


def human_core_ids(timeout: int = 60) -> set[str] | None:
    """JASPAR CORE matrix IDs for Homo sapiens. Returns None if the API is unreachable."""
    ids, url = (
        set(),
        (
            "https://jaspar.elixir.no/api/v1/matrix/"
            "?collection=CORE&tax_id=9606&page_size=1000&format=json"
        ),
    )
    try:
        while url:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                d = json.load(r)
            ids.update(x["matrix_id"] for x in d.get("results", []))
            url = d.get("next")
    except Exception:
        return None
    return ids or None


def _pwm_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Best-offset Pearson correlation between two PFMs, taking the better strand."""
    best = -1.0
    for bb in (b, b[::-1, ::-1]):
        la, lb = len(a), len(bb)
        for off in range(-(lb - 1), la):
            s, e = max(0, off), min(la, off + lb)
            if e - s < 4:  # need real overlap to compare
                continue
            x = a[s:e].ravel()
            y = bb[s - off : e - off].ravel()
            if x.std() < 1e-9 or y.std() < 1e-9:
                continue
            best = max(best, float(np.corrcoef(x, y)[0, 1]))
    return best


def cluster(motifs: list[Motif], threshold: float = 0.90) -> list[Motif]:
    """Greedy: keep the highest-IC motif of each near-duplicate group.

    JASPAR ships several versions of many factors, plus paralogues with nearly identical PFMs.
    Counting those as distinct would overstate the vocabulary and skew any per-motif coverage
    calculation.
    """
    order = sorted(motifs, key=lambda m: -m.info_content)
    kept: list[Motif] = []
    for m in order:
        if all(_pwm_similarity(m.pwm, k.pwm) < threshold for k in kept):
            kept.append(m)
    return kept


def build(
    meme: str = MEME_DEFAULT,
    human_only: bool = True,
    cluster_at: float | None = 0.90,
    max_motifs: int | None = None,
    min_ic: float = 6.0,
    expressed: set[str] | None = None,
    trim_ic: float = 0.0,
    max_len: int | None = None,
) -> list[Motif]:
    """Assemble a motif vocabulary from a MEME file.

    Args:
        meme: Path to a MEME-format PFM file (JASPAR2022 CORE by default).
        human_only: Restrict to Homo sapiens CORE matrices via the JASPAR API.
        cluster_at: Collapse near-duplicate PFMs above this correlation. This is the
            filter that does the real work -- roughly two-thirds of human CORE is
            near-duplicate, so skipping it inflates every coverage number.
        max_motifs: Cap vocabulary size, most-informative first.
        min_ic: Drop motifs below this total information content. JASPAR CORE is
            already curated, so at the default of 6 bits this removes nothing; it
            matters only for uncurated PFM sources.
        expressed: Optional TF gene symbols to keep (the cell-type vocabulary arms).
        trim_ic: Trim terminal positions below this per-position IC, in bits.
            Applied BEFORE clustering, since trimming can make two padded PFMs
            resolve as the same motif.
        max_len: Drop motifs longer than this after trimming. Length bounds how many
            sites fit on an oligo, so this and `trim_ic` together set the ceiling on
            the motifs-per-sequence knob.

    Returns:
        Motifs sorted by descending information content.
    """
    key = _cache_key(meme, human_only, cluster_at, max_motifs, min_ic, expressed, trim_ic, max_len)
    cached = _cache_load(key)
    if cached is not None:
        return cached

    ms = parse_meme(meme)
    if human_only:
        ids = human_core_ids()
        if ids:
            ms = [
                m
                for m in ms
                if m.mid in ids or m.mid.split(".")[0] in {i.split(".")[0] for i in ids}
            ]
        else:
            # Silently keeping all taxa would change the vocabulary from ~242 motifs
            # to 700+ without anything in the output saying so, and every coverage
            # number downstream would then be wrong. Fail loudly instead.
            raise RuntimeError(
                "human_only=True but the JASPAR API is unreachable, so Homo sapiens "
                "IDs could not be fetched. Skipping the filter would silently change "
                "the vocabulary, so this is an error. Retry with network access or "
                "pass human_only=False deliberately."
            )
    if expressed:
        up = {e.upper() for e in expressed}
        ms = [m for m in ms if any(t.upper() in up for t in re.split(r"[:\-_.]", m.name))]
    ms = [m for m in ms if m.info_content >= min_ic]
    if trim_ic > 0:
        ms = [m.trim(trim_ic) for m in ms]
    if max_len:
        ms = [m for m in ms if m.length <= max_len]
    if cluster_at:
        ms = cluster(ms, cluster_at)
    ms.sort(key=lambda m: -m.info_content)
    ms = ms[:max_motifs] if max_motifs else ms
    _cache_store(key, ms)
    return ms


# --- vocabulary cache ---------------------------------------------------------
# Clustering is O(n^2) in the number of motifs, each pair needing a both-strand
# best-offset alignment, so building one vocabulary from human CORE takes minutes.
# Vocabulary source and filters are swept factors, so the same handful of
# configurations get rebuilt repeatedly; cache them keyed on the full parameter set
# (including the MEME file's mtime, so editing the PFM source invalidates the cache).

_CACHE_DIR = Path(
    os.environ.get("MOTIF_VOCAB_CACHE", str(Path.home() / ".cache/albench/motif_vocab"))
)


def _cache_key(
    meme, human_only, cluster_at, max_motifs, min_ic, expressed, trim_ic, max_len
) -> str:
    try:
        stamp = str(int(Path(meme).stat().st_mtime))
    except OSError:
        stamp = "0"
    payload = repr(
        (
            Path(meme).name,
            stamp,
            bool(human_only),
            cluster_at,
            max_motifs,
            min_ic,
            tuple(sorted(expressed)) if expressed else None,
            trim_ic,
            max_len,
        )
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _cache_load(key: str) -> list[Motif] | None:
    path = _CACHE_DIR / f"{key}.npz"
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        mids = [str(s) for s in z["mids"]]
        names = [str(s) for s in z["names"]]
        flat, lens = z["pwm_flat"], z["lens"]
        out, off = [], 0
        for i in range(int(z["n"])):
            length = int(lens[i])
            out.append(Motif(mids[i], names[i], flat[off : off + length * 4].reshape(length, 4)))
            off += length * 4
        return out
    except Exception:  # a corrupt cache must never break a run
        return None


def _cache_store(key: str, motifs: list[Motif]) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Must end in .npz: np.savez silently appends the suffix otherwise, and the
        # rename below would then target a file that was never written.
        tmp = _CACHE_DIR / f"{key}.tmp{os.getpid()}.npz"
        np.savez(
            tmp,
            n=len(motifs),
            mids=np.array([m.mid for m in motifs]),
            names=np.array([m.name for m in motifs]),
            lens=np.array([m.length for m in motifs], dtype=np.int32),
            pwm_flat=(np.concatenate([m.pwm.ravel() for m in motifs]) if motifs else np.zeros(0)),
        )
        tmp.replace(_CACHE_DIR / f"{key}.npz")  # atomic, so concurrent jobs are safe
    except Exception:
        pass  # caching is an optimisation, never a requirement
