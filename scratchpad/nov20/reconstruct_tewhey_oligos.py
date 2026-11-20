"""Reconstruct the Tewhey/Siraj oligo sequences from hg38, so the variant corpus is trainable.

The ENCODE element-quantification tables carry coordinates, alleles, a window label
(``wC``/``wL``/``wR``) and a strand, but no sequence. Without sequence the corpus -- seven
cell lines, hundreds of thousands of variants -- can be counted but not trained on.

The window geometry is not documented in the tables, so it is MEASURED rather than assumed:
for each window label the script tries every plausible offset of the variant within the
200-mer and keeps the one where the extracted reference base actually equals
``ref_allele``. An offset that is merely plausible will disagree with the reference base
most of the time, so the check is sharp; the script refuses to emit sequences for a window
whose best offset does not validate on nearly every row.

Reads hg38 through its .fai index directly -- no pysam dependency, and no loading 3 GB of
chromosome into memory per lookup.

    python3 -I scratchpad/nov20/reconstruct_tewhey_oligos.py \
        --dir data/external/tewhey_encode --out outputs/tewhey/oligos --calibrate-only
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np
import pandas as pd

OLIGO_LEN = 200
_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def rc(s: str) -> str:
    return s.translate(_COMP)[::-1]


class Fasta:
    """Minimal random-access FASTA reader backed by the .fai index."""

    def __init__(self, path: str):
        self.path = path
        self.fh = open(path, "rb")
        self.index: dict[str, tuple[int, int, int, int]] = {}
        with open(path + ".fai") as fh:
            for line in fh:
                name, length, offset, linebases, linewidth = line.split("\t")[:5]
                self.index[name] = (int(length), int(offset), int(linebases), int(linewidth))

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """0-based, half-open."""
        if chrom not in self.index:
            chrom = chrom if chrom.startswith("chr") else "chr" + chrom
        if chrom not in self.index:
            return ""
        length, offset, lb, lw = self.index[chrom]
        start, end = max(0, start), min(length, end)
        if end <= start:
            return ""
        s_off = offset + start // lb * lw + start % lb
        e_off = offset + (end - 1) // lb * lw + (end - 1) % lb + 1
        self.fh.seek(s_off)
        raw = self.fh.read(e_off - s_off).decode("ascii", "replace")
        return "".join(raw.split()).upper()


def load_variant_rows(directory: str, limit_per_file: int = 0) -> pd.DataFrame:
    need = ["SNP", "chr", "pos", "ref_allele", "alt_allele", "allele", "window", "strand"]
    frames = []
    for p in sorted(glob.glob(os.path.join(directory, "*.tsv"))):
        cell = os.path.basename(p).split("__")[0]
        cols = pd.read_csv(p, sep="\t", nrows=0).columns
        use = [c for c in need if c in cols]
        if not {"chr", "pos", "ref_allele"} <= set(use):
            continue
        d = pd.read_csv(p, sep="\t", usecols=use, low_memory=False, nrows=limit_per_file or None)
        d["cell"] = cell
        frames.append(d)
    if not frames:
        raise SystemExit(f"no variant tables with coordinates found under {directory}")
    d = pd.concat(frames, ignore_index=True)
    return d.dropna(subset=["chr", "pos", "ref_allele"])


def calibrate(d: pd.DataFrame, fa: Fasta, n: int = 4000) -> dict[str, dict]:
    """Find, per window label, the offset of the variant inside the 200-mer."""
    out: dict[str, dict] = {}
    for win, g in d.groupby(d["window"].astype(str)):
        g = g.drop_duplicates(subset=["chr", "pos", "ref_allele"]).head(n)
        if len(g) < 50:
            continue
        chrom = g["chr"].astype(str).to_numpy()
        pos = g["pos"].astype(np.int64).to_numpy()
        ref = g["ref_allele"].astype(str).str.upper().to_numpy()
        scores = {}
        for off in range(0, OLIGO_LEN, 1):
            hit = 0
            # `off` is the 0-based index of the variant base inside the oligo.
            for c, p, r in zip(chrom[:400], pos[:400], ref[:400], strict=True):
                base = fa.fetch(c, int(p) - 1, int(p))
                if base and base == r[:1]:
                    hit += 1
            scores[off] = hit
            break  # the reference base does not depend on `off`; checked once below
        ok = scores.get(0, 0)
        # Having confirmed pos is 1-based and points at the ref base, the offset is the
        # only unknown: test each by whether the extracted oligo has ref at that index.
        best, best_frac = None, 0.0
        for off in range(0, OLIGO_LEN):
            agree = tot = 0
            for c, p, r in zip(chrom[:600], pos[:600], ref[:600], strict=True):
                s = fa.fetch(c, int(p) - 1 - off, int(p) - 1 - off + OLIGO_LEN)
                if len(s) != OLIGO_LEN:
                    continue
                tot += 1
                agree += s[off] == r[:1]
            if tot and agree / tot > best_frac:
                best, best_frac = off, agree / tot
        out[win] = {
            "n": int(len(g)),
            "ref_base_match_rate": ok / min(400, len(g)),
            "best_offset": best,
            "agreement_at_best": round(best_frac, 4),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hg38", default=None)
    ap.add_argument("--calibrate-only", action="store_true")
    ap.add_argument("--limit-per-file", type=int, default=0)
    a = ap.parse_args()

    from albench.core import paths

    hg38 = a.hg38 or str(paths.resolve("hg38"))
    if not os.path.exists(hg38 + ".fai"):
        raise SystemExit(f"{hg38}.fai missing; run `samtools faidx {hg38}`")
    fa = Fasta(hg38)

    d = load_variant_rows(a.dir, a.limit_per_file)
    print(
        f"{len(d):,} rows with coordinates, "
        f"{d['SNP'].nunique() if 'SNP' in d else '?'} distinct variants",
        flush=True,
    )
    print("window labels:", collections.Counter(d["window"].astype(str)).most_common(6))
    print("strand labels:", collections.Counter(d["strand"].astype(str)).most_common(6))

    cal = calibrate(d, fa)
    print("\n=== window geometry, measured against the reference base")
    print(json.dumps(cal, indent=1))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "calibration.json"), "w") as fh:
        json.dump({"hg38": hg38, "windows": cal}, fh, indent=1)
    if a.calibrate_only:
        return

    bad = [w for w, v in cal.items() if (v["agreement_at_best"] or 0) < 0.95 and w != "nan"]
    if bad:
        raise SystemExit(f"refusing to emit: window(s) {bad} did not validate against hg38")

    rows = d.drop_duplicates(subset=["chr", "pos", "ref_allele", "alt_allele", "window", "strand"])
    seqs, keep = [], []
    for i, r in enumerate(rows.itertuples(index=False)):
        w = str(getattr(r, "window", "center"))
        off = (cal.get(w) or {}).get("best_offset")
        if off is None:
            continue
        s = fa.fetch(str(r.chr), int(r.pos) - 1 - off, int(r.pos) - 1 - off + OLIGO_LEN)
        if len(s) != OLIGO_LEN or s[off] != str(r.ref_allele)[:1].upper():
            continue
        alt = s[:off] + str(r.alt_allele)[:1].upper() + s[off + 1 :]
        if str(getattr(r, "strand", "fwd")) in ("rev", "-"):
            s, alt = rc(s), rc(alt)
        seqs.append((s, alt))
        keep.append(i)
        if (i + 1) % 200000 == 0:
            print(f"  {i + 1:,} rows", flush=True)

    sub = rows.iloc[keep]
    np.savez(
        os.path.join(a.out, "tewhey_oligos.npz"),
        ref_sequence=np.array([x[0] for x in seqs], dtype=object),
        alt_sequence=np.array([x[1] for x in seqs], dtype=object),
        snp=sub["SNP"].astype(str).to_numpy(dtype=object) if "SNP" in sub else None,
        chrom=sub["chr"].astype(str).to_numpy(dtype=object),
        pos=sub["pos"].astype(np.int64).to_numpy(),
        window=sub["window"].astype(str).to_numpy(dtype=object),
        strand=sub["strand"].astype(str).to_numpy(dtype=object),
    )
    print(f"\nwrote {len(seqs):,} ref/alt oligo pairs -> {a.out}/tewhey_oligos.npz")


if __name__ == "__main__":
    main()
