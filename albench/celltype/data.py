"""The long-format (sequence, cell, label) view the conditioning trainer consumes.

Two things this has to get right that a single-cell loader does not.

*Holdouts are now two-dimensional.* The chromosome holdout is unchanged and still real --
chr7 and chr13 are test, chr19, chr21 and chrX are validation, and nothing may train on
them. On top of that a conditioning study holds out whole CELL TYPES, and the two splits
are independent: a held-out cell is held out on every chromosome, and a held-out chromosome
is held out in every cell. Reporting a held-out-cell number on training chromosomes would
measure cell transfer on memorised sequence, so the default test cut is the intersection.

*Sequences are stored as base indices, not one-hot.* The Gosai library is 798,064 rows and
a 260-base one-hot in float32 is 3.3 GB before any batching; the indices are 207 MB and the
one-hot is made on device per batch. The 260 is not arbitrary: it is the 200-bp insert plus
``FLANK_CTX + SHIFT_MAX`` = 30 real plasmid bases each side, so the standard +-10 shift
always lands on real sequence (see ``arch_search.harness_standard``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from albench.arch_search.harness_standard import FLANK_CTX, INSERT_LEN, SHIFT_MAX

TEST_CHROMS = ("chr7", "chr13")
VAL_CHROMS = ("chr19", "chr21", "chrX")
GOSAI_CELLS = ("K562", "HepG2", "SK-N-SH")
_COL = {"K562": "K562", "HepG2": "HepG2", "SK-N-SH": "SKNSH"}
_BASE = {b: i for i, b in enumerate("ACGT")}
MARGIN = FLANK_CTX + SHIFT_MAX
CHROM_NAMES = {str(i) for i in range(1, 23)} | {"X", "Y", "M", "MT"}


def normalise_chrom(values) -> np.ndarray:
    """``7`` -> ``chr7``. Gosai's table omits the prefix; the holdout constants carry it.

    Only values that LOOK like a chromosome are prefixed. A designed oligo has no genomic
    origin and is labelled ``designed``; blindly prefixing turned that into ``chrdesigned``,
    which matched no holdout and no exclusion, so the designed rows -- the cross-cell eval
    set -- went straight into training and nothing in the output said so.
    """
    out = []
    for v in values:
        s = str(v).strip()
        if s.lower().startswith("chr"):
            out.append(s)
        elif s.upper() in CHROM_NAMES:
            out.append(f"chr{s.upper() if not s.isdigit() else s}")
        else:
            out.append(s)
    return np.array(out, dtype=object)


def canonical_cell(name: str) -> str:
    """``sknsh`` -> ``SK-N-SH``, so a command line need not match the corpus spelling."""
    from albench.celltype.descriptions import canonical

    try:
        return canonical(name)
    except KeyError:
        return str(name).strip()


@dataclass
class MultiCellData:
    """Sequences once, labels per cell, and the mask that says which were measured."""

    idx: np.ndarray  # (N, INSERT_LEN + 2*MARGIN) uint8 base indices, 4 = N/pad
    y: np.ndarray  # (N, C) float32
    se: np.ndarray  # (N, C) float32
    mask: np.ndarray  # (N, C) bool
    chrom: np.ndarray  # (N,) object
    cells: list[str]
    ids: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.idx)

    def subset(self, sel: np.ndarray) -> MultiCellData:
        return MultiCellData(
            self.idx[sel],
            self.y[sel],
            self.se[sel],
            self.mask[sel],
            self.chrom[sel],
            list(self.cells),
            None if self.ids is None else self.ids[sel],
        )

    def split(self) -> dict[str, np.ndarray]:
        """Row selectors for the chromosome split. Anything not held out is training.

        Raises rather than returning an empty holdout. Gosai's Table S2 writes chromosomes
        as bare ``7``/``13``, so an unnormalised ``isin(chrom, ("chr7", ...))`` matches
        nothing, every holdout silently becomes training data, and the run still finishes
        and still reports numbers -- which is precisely the failure the "chromosome
        holdouts are real, check this" rule exists to catch. Normalisation happens at load;
        this is the assertion that it worked.
        """
        te = np.isin(self.chrom, TEST_CHROMS)
        va = np.isin(self.chrom, VAL_CHROMS)
        designed = self.chrom == "designed"
        if designed.all():
            return {"train": ~(te | va), "val": va, "test": te}
        if not (te.any() and va.any()):
            seen = sorted(set(self.chrom.tolist()))[:8]
            raise ValueError(
                f"chromosome holdout matched nothing (test={int(te.sum())}, "
                f"val={int(va.sum())}); labels look like {seen}, expected {TEST_CHROMS}"
            )
        # A corpus that mixes genomic and designed oligos must keep the designed ones OUT of
        # training: they carry no chromosome, so the chromosome holdout cannot protect them,
        # and they are the cross-cell eval set. Training on them would contaminate the one
        # endpoint where the cell-type difference has a real measurement ceiling.
        return {"train": ~(te | va | designed), "val": va, "test": te, "designed": designed}


def encode_sequences(seqs, margin: int = MARGIN) -> np.ndarray:
    """Base indices for the insert inside its real plasmid flank, vectorised.

    A per-sequence Python loop is fine at n=1,000 and does not finish at n=800,000, so the
    whole library is encoded with one bytes join and one frombuffer.
    """
    from albench.arch_search.flank_context import DOWNSTREAM_FULL, UPSTREAM_FULL

    if margin > len(UPSTREAM_FULL):
        raise ValueError(f"margin {margin} exceeds the {len(UPSTREAM_FULL)} stored flank bases")
    up = UPSTREAM_FULL[-margin:] if margin else ""
    dn = DOWNSTREAM_FULL[:margin] if margin else ""

    lut = np.full(256, 4, dtype=np.uint8)
    for b, i in _BASE.items():
        lut[ord(b)] = i
        lut[ord(b.lower())] = i

    n, width = len(seqs), INSERT_LEN + 2 * margin
    flat = bytearray()
    for s in seqs:
        s = str(s).upper()
        if len(s) != INSERT_LEN:
            s = s[:INSERT_LEN].ljust(INSERT_LEN, "N")
        flat += (up + s + dn).encode("ascii", "replace")
    arr = np.frombuffer(bytes(flat), dtype=np.uint8).reshape(n, width)
    return lut[arr]


def onehot_from_idx(idx, device=None):
    """(B, L) uint8 indices -> (B, 4, L) float one-hot, with 4 mapping to all-zeros."""
    import torch

    t = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=device)
    oh = torch.zeros(t.shape[0], 5, t.shape[1], device=t.device)
    oh.scatter_(1, t.unsqueeze(1), 1.0)
    return oh[:, :4]


def load_gosai(path: str | None = None, *, cells=GOSAI_CELLS) -> MultiCellData:
    """The 798k Gosai library with all three cell lines side by side.

    This is the only corpus where the same sequences carry a measurement in more than one
    cell, which is what makes a conditioning scheme trainable at all today. The Tewhey
    variant libraries add four more cell types but need their sequences reconstructed from
    hg38 first.
    """
    import pandas as pd

    from albench.core import paths

    if path is None:
        cand = [
            os.environ.get("ALBENCH_GOSAI_TABLE_S2", ""),
            str(paths.REPO_ROOT / "data/k562/DATA-Table_S2__MPRA_dataset.txt"),
        ]
        path = next((c for c in cand if c and os.path.exists(c)), "")
        if not path:
            raise FileNotFoundError(
                "Gosai Table S2 not found; set ALBENCH_GOSAI_TABLE_S2 to "
                "DATA-Table_S2__MPRA_dataset.txt"
            )

    d = pd.read_csv(path, sep="\t", low_memory=False)
    cells = list(cells)
    y = np.stack([d[f"{_COL[c]}_log2FC"].to_numpy(np.float32) for c in cells], 1)
    se = np.stack([d[f"{_COL[c]}_lfcSE"].to_numpy(np.float32) for c in cells], 1)
    mask = np.isfinite(y) & np.isfinite(se)
    return MultiCellData(
        idx=encode_sequences(d["sequence"].to_numpy()),
        y=np.nan_to_num(y),
        se=np.nan_to_num(se, nan=1.0),
        mask=mask,
        chrom=normalise_chrom(d["chr"].to_numpy()),
        cells=cells,
        ids=d["IDs"].astype(str).to_numpy(dtype=object),
    )


TEWHEY_CELLS = ("K562", "HepG2", "SK-N-SH", "A549", "GM12878", "HCT116", "Jurkat")


def load_tewhey(path: str | None = None, *, cells=TEWHEY_CELLS, require_chrom: bool = True):
    """The 7-cell Tewhey/Siraj corpus: 1.48M oligos, 4.52M (oligo, cell) observations.

    This is the corpus that makes a conditioning scheme answerable. With Gosai's three cells
    a text embedding has only three points to learn from, so nothing constrains the mapping
    off those points and `text` cannot be told apart from `onehot` -- which is exactly what
    the first bake-off measured. Seven cells give the scheme somewhere to interpolate.

    The sequences come from the ENCODE ``elements_references`` FASTAs rather than from hg38:
    reconstructing them from coordinates fails outright (the base at ``chr:pos`` agrees with
    ``ref_allele`` at chance, and the variant's offset inside the oligo is not identifiable
    from the reference at all).

    ``require_chrom`` refuses a corpus built before chromosomes were attached, because the
    holdout cannot be honoured without them and the resulting run would look fine.
    """
    from albench.core import paths

    if path is None:
        cand = [
            os.environ.get("ALBENCH_TEWHEY_CORPUS", ""),
            str(paths.REPO_ROOT / "outputs/tewhey/corpus_chrom.npz"),
        ]
        path = next((c for c in cand if c and os.path.exists(c)), "")
        if not path:
            raise FileNotFoundError(
                "Tewhey corpus not found; build it with scratchpad/nov20/build_tewhey_corpus.py "
                "then attach chromosomes with scratchpad/nov20/add_corpus_chrom.py"
            )

    z = np.load(path, allow_pickle=True)
    if require_chrom and "chrom" not in z.files:
        raise ValueError(
            f"{path} carries no chromosome column, so chr7/chr13 cannot be held out; "
            "run scratchpad/nov20/add_corpus_chrom.py on it first"
        )
    have = [str(c) for c in z["cells"]]
    cells = list(cells)
    missing = [c for c in cells if c not in have]
    if missing:
        raise ValueError(f"corpus has no column for {missing}; it carries {have}")
    col = [have.index(c) for c in cells]

    y = z["y"][:, col].astype(np.float32)
    se = z["se"][:, col].astype(np.float32)
    mask = z["mask"][:, col] if "mask" in z.files else (np.isfinite(y) & np.isfinite(se))
    chrom = normalise_chrom(z["chrom"]) if "chrom" in z.files else np.array(["designed"] * len(y))
    return MultiCellData(
        idx=encode_sequences([str(s) for s in z["sequences"]]),
        y=np.nan_to_num(y),
        se=np.nan_to_num(se, nan=1.0),
        mask=mask & np.isfinite(y) & np.isfinite(se),
        chrom=chrom,
        cells=cells,
        ids=np.array([str(i) for i in z["oligo_id"]], dtype=object),
    )


def load_designed_crosscell(
    designed_dir: str, crosscell_npz: str, *, cells=GOSAI_CELLS
) -> MultiCellData:
    """The designed libraries with all three cells' measurements attached.

    These are the eval set the ENCODE deposition unlocked: 131,743 machine-designed oligos
    with a real measurement in every cell, spanning four log2 units between cells. Unlike
    the genomic library, the cell-type DIFFERENCE here has a measurement ceiling near 0.9,
    so it is the one place a conditioning scheme can be scored on specificity rather than
    on level. Carries no chromosome, so it is an eval set only -- never a training split.
    """
    seqs: list[str] = []
    ids: list[str] = []
    for cell in ("k562", "hepg2", "sknsh"):
        p = os.path.join(designed_dir, f"designed_{cell}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        seqs += [str(s) for s in z["sequences"]]
        ids += [str(i) for i in z["ids"]]

    cc = np.load(crosscell_npz, allow_pickle=True)
    pos = {str(v): i for i, v in enumerate(cc["ids"])}
    keep = [k for k, i in enumerate(ids) if i in pos]
    rows = np.array([pos[ids[k]] for k in keep])
    seqs = [seqs[k] for k in keep]

    # Only three cells were ever measured on the designed libraries. A 7-cell run still asks
    # for all seven, so the four that were never assayed come back as NaN and are masked out
    # rather than raising -- the designed set is an eval set, and losing it should cost those
    # four columns, not the whole evaluation.
    col = {"K562": "k562", "HepG2": "hepg2", "SK-N-SH": "sknsh"}
    blank = np.full(len(rows), np.nan, np.float32)
    y = np.stack(
        [cc[f"measured_{col[c]}"][rows].astype(np.float32) if c in col else blank for c in cells],
        1,
    )
    se = np.stack(
        [
            cc[f"measured_se_{col[c]}"][rows].astype(np.float32) if c in col else blank
            for c in cells
        ],
        1,
    )
    return MultiCellData(
        idx=encode_sequences(seqs),
        y=np.nan_to_num(y),
        se=np.nan_to_num(se, nan=1.0),
        mask=np.isfinite(y) & np.isfinite(se),
        chrom=np.array(["designed"] * len(seqs), dtype=object),
        cells=list(cells),
        ids=np.array([ids[k] for k in keep], dtype=object),
    )


def long_format(data: MultiCellData, rows: np.ndarray, train_cells: list[str]):
    """Expand (sequence, C labels) into one example per observed (sequence, cell) pair.

    Only ``train_cells`` are expanded, so a held-out cell contributes nothing to training
    even though its column is present in the array.
    """
    keep = np.flatnonzero(rows)
    ci = [data.cells.index(c) for c in train_cells]
    seq_ix, cell_ix = [], []
    for j, c in zip(ci, train_cells, strict=True):
        sel = keep[data.mask[keep, j]]
        seq_ix.append(sel)
        cell_ix.append(np.full(len(sel), data.cells.index(c), dtype=np.int64))
    si = np.concatenate(seq_ix) if seq_ix else np.zeros(0, np.int64)
    ki = np.concatenate(cell_ix) if cell_ix else np.zeros(0, np.int64)
    return si, ki, data.y[si, ki].astype(np.float32), data.se[si, ki].astype(np.float32)
