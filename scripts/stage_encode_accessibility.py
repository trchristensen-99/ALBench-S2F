"""Stage cell-type-specific accessibility peaks for K562 and HepG2 from the ENCODE portal.

Why this is needed. The cCRE registry already on the cluster
(/grid/koo/home/shared/d3/data/zoonomia/GRCh38-cCREs.bed, 2,348,854 regions) is the UNION across
biosamples: its six columns are chrom/start/end/accessionD/accessionE/class, where class is the
regulatory category (pELS, dELS, PLS, CA-CTCF, ...) and NOT per-cell-type activity. It therefore
supports the "broad accessible regions" reservoir but cannot distinguish open-in-K562 from
open-in-HepG2, which is exactly the shared-vs-specific contrast the design needs.

Approach: query the ENCODE REST API rather than hardcoding accessions, so the selection is
reproducible and auditable. We take DNase-seq (and optionally ATAC-seq) released, non-audit-error
peak BEDs on GRCh38 for each biosample, preferring "pseudoreplicated peaks" / IDR-thresholded
conservative sets over raw replicate peaks.

Produces, per cell type, a merged peak BED, then the three-way partition the reservoir arms need:
    shared      open in BOTH K562 and HepG2
    k562_only   open in K562, not HepG2
    hepg2_only  open in HepG2, not K562
The partition is written as BED plus a summary, so the "how much does accessibility actually differ
between these lines?" question becomes a measured number rather than an assumption.
"""

import argparse
import json
import os
import urllib.parse
import urllib.request

API = "https://www.encodeproject.org"
PREFERRED = (
    "pseudoreplicated peaks",
    "conservative IDR thresholded peaks",
    "replicated peaks",
    "peaks",
)


def search_files(biosample, assay, assembly="GRCh38", limit=50):
    q = {
        "type": "File",
        "assay_term_name": assay,
        "biosample_ontology.term_name": biosample,
        "assembly": assembly,
        "file_format": "bed",
        "status": "released",
        "limit": str(limit),
        "format": "json",
    }
    url = f"{API}/search/?{urllib.parse.urlencode(q)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r).get("@graph", [])


def pick(files):
    """Prefer the most processed peak product; fall back down the list."""
    for want in PREFERRED:
        hits = [f for f in files if f.get("output_type") == want]
        if hits:
            # newest first
            hits.sort(key=lambda f: f.get("date_created", ""), reverse=True)
            return hits[0], want
    return (files[0], files[0].get("output_type")) if files else (None, None)


def read_bed(path, gz=False):
    """chrom -> sorted list of (start, end), primary chromosomes only."""
    import gzip

    opener = gzip.open if gz else open
    keep = {f"chr{c}" for c in list(range(1, 23)) + ["X"]}
    out = {}
    with opener(path, "rt") as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            f = line.split("\t")
            if len(f) < 3 or f[0] not in keep:
                continue
            out.setdefault(f[0], []).append((int(f[1]), int(f[2])))
    return {c: sorted(v) for c, v in out.items()}


def merge(iv):
    """Union of overlapping intervals in a sorted list."""
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _mask_ops(a, b):
    """Return (intersection, a_minus_b) for two merged interval lists via a sweep."""
    inter, only = [], []
    j = 0
    for s, e in a:
        cur = s
        hit = False
        while j > 0 and b[j - 1][1] > s:
            j -= 1
        k = j
        while k < len(b) and b[k][0] < e:
            bs, be = b[k]
            if be <= s:
                k += 1
                continue
            ov_s, ov_e = max(s, bs), min(e, be)
            if ov_s < ov_e:
                hit = True
                inter.append((ov_s, ov_e))
                if cur < ov_s:
                    only.append((cur, ov_s))
                cur = max(cur, ov_e)
            k += 1
        if cur < e and (hit or True):
            only.append((cur, e))
    return inter, only


def write_bed(path, per_chrom):
    n = bp = 0
    with open(path, "w") as fh:
        for c in sorted(per_chrom):
            for s, e in per_chrom[c]:
                fh.write(f"{c}\t{s}\t{e}\n")
                n += 1
                bp += e - s
    return n, bp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/encode_accessibility")
    ap.add_argument("--cells", nargs="+", default=["K562", "HepG2"])
    ap.add_argument("--assays", nargs="+", default=["DNase-seq", "ATAC-seq"])
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    chosen = {}
    for cell in args.cells:
        for assay in args.assays:
            if cell in chosen:
                break
            try:
                files = [f for f in search_files(cell, assay) if f.get("href")]
            except Exception as e:
                print(f"  [{cell} {assay}] query failed: {e}")
                continue
            f, kind = pick(files)
            print(
                f"  {cell:<6} {assay:<10} {len(files):>3} beds -> "
                f"{f['accession'] if f else 'NONE'} ({kind})"
            )
            if f:
                chosen[cell] = (assay, f, kind)
    missing = set(args.cells) - set(chosen)
    if missing:
        print(f"\nWARNING no peaks found for: {sorted(missing)}")

    peaks = {}
    for cell, (assay, f, kind) in chosen.items():
        url = API + f["href"]
        raw = os.path.join(args.out_dir, f"{cell}_{f['accession']}.bed.gz")
        print(f"\n[{cell}] {assay} / {kind} / {f['accession']}\n  {url}")
        if args.dry_run:
            continue
        if not os.path.exists(raw):
            urllib.request.urlretrieve(url, raw)
        per = {c: merge(v) for c, v in read_bed(raw, gz=True).items()}
        n, bp = write_bed(os.path.join(args.out_dir, f"{cell}_peaks_merged.bed"), per)
        print(f"  merged: {n:,} peaks, {bp / 1e6:.1f} Mb")
        peaks[cell] = per

    if len(peaks) == 2 and not args.dry_run:
        a, b = args.cells[0], args.cells[1]
        shared, a_only, b_only = {}, {}, {}
        for c in sorted(set(peaks[a]) | set(peaks[b])):
            A, B = peaks[a].get(c, []), peaks[b].get(c, [])
            if not A:
                b_only[c] = B
                continue
            if not B:
                a_only[c] = A
                continue
            inter, aonly = _mask_ops(A, B)
            _, bonly = _mask_ops(B, A)
            shared[c] = merge(sorted(inter))
            a_only[c] = merge(sorted(aonly))
            b_only[c] = merge(sorted(bonly))
        print("\n=== accessibility partition (the open question, measured) ===")
        summ = {}
        for name, d in (
            ("shared_open_both", shared),
            (f"{a.lower()}_only", a_only),
            (f"{b.lower()}_only", b_only),
        ):
            n, bp = write_bed(os.path.join(args.out_dir, f"{name}.bed"), d)
            summ[name] = {"n_regions": n, "bp": bp}
            print(f"  {name:<18} {n:>8,} regions  {bp / 1e6:>7.1f} Mb")
        tot_bp = sum(v["bp"] for v in summ.values())
        if tot_bp:
            print(f"\n  shared fraction (bp) = {summ['shared_open_both']['bp'] / tot_bp:.1%}")
            print(
                "  Carl predicted this is HIGH, because shared TFs (MYC, AP1) drive most "
                "activity. If the cell-type-specific fractions are small, the differential arm "
                "must be drawn BALANCED-BY-CONSTRUCTION rather than proportionally, or its "
                "specific strata will be too thin to analyse."
            )
        with open(os.path.join(args.out_dir, "partition_summary.json"), "w") as fh:
            json.dump(summ, fh, indent=2)


if __name__ == "__main__":
    main()
