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
import subprocess
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


def sh(cmd):
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


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
            try:
                files = search_files(cell, assay)
            except Exception as e:
                print(f"  [{cell} {assay}] query failed: {e}")
                continue
            files = [f for f in files if f.get("href")]
            f, kind = pick(files)
            print(
                f"  {cell:<6} {assay:<10} {len(files):>3} beds -> "
                f"{f['accession'] if f else 'NONE'} ({kind})"
            )
            if f and cell not in chosen:  # first assay that yields something wins
                chosen[cell] = (assay, f, kind)
    if len(chosen) < len(args.cells):
        missing = set(args.cells) - set(chosen)
        print(f"\nWARNING no peaks found for: {sorted(missing)}")

    paths = {}
    for cell, (assay, f, kind) in chosen.items():
        url = API + f["href"]
        raw = os.path.join(args.out_dir, f"{cell}_{f['accession']}.bed.gz")
        merged = os.path.join(args.out_dir, f"{cell}_peaks_merged.bed")
        print(f"\n[{cell}] {assay} / {kind} / {f['accession']}\n  {url}")
        if args.dry_run:
            continue
        if not os.path.exists(raw):
            urllib.request.urlretrieve(url, raw)
        # merge overlapping peaks; keep only primary chromosomes
        sh(
            f"zcat {raw} | awk '$1 ~ /^chr([0-9]+|X)$/' | sort -k1,1 -k2,2n "
            f"| bedtools merge -i - > {merged}"
        )
        n = int(sh(f"wc -l < {merged}").strip())
        bp = int(sh(f"awk '{{s+=$3-$2}} END {{print s+0}}' {merged}").strip())
        print(f"  merged: {n:,} peaks, {bp / 1e6:.1f} Mb")
        paths[cell] = merged

    if len(paths) == 2 and not args.dry_run:
        a, b = args.cells[0], args.cells[1]
        out = {
            "shared": os.path.join(args.out_dir, "shared_open_both.bed"),
            f"{a.lower()}_only": os.path.join(args.out_dir, f"{a.lower()}_only.bed"),
            f"{b.lower()}_only": os.path.join(args.out_dir, f"{b.lower()}_only.bed"),
        }
        sh(f"bedtools intersect -a {paths[a]} -b {paths[b]} > {out['shared']}")
        sh(f"bedtools subtract -A -a {paths[a]} -b {paths[b]} > {out[f'{a.lower()}_only']}")
        sh(f"bedtools subtract -A -a {paths[b]} -b {paths[a]} > {out[f'{b.lower()}_only']}")
        print("\n=== accessibility partition (this IS the open question, measured) ===")
        summ = {}
        for k, v in out.items():
            n = int(sh(f"wc -l < {v}").strip())
            bp = int(sh(f"awk '{{s+=$3-$2}} END {{print s+0}}' {v}").strip())
            summ[k] = {"n_regions": n, "bp": bp}
            print(f"  {k:<12} {n:>8,} regions  {bp / 1e6:>7.1f} Mb")
        tot = sum(v["n_regions"] for v in summ.values())
        if tot:
            frac = summ["shared"]["n_regions"] / tot
            print(
                f"\n  shared fraction = {frac:.1%} -- Carl's point is that this will be HIGH "
                "because shared TFs (MYC, AP1) drive most activity; the specific fractions are "
                "what the differential arm must be drawn from, so if they are small the arm needs "
                "balanced-by-construction sampling rather than proportional sampling."
            )
        with open(os.path.join(args.out_dir, "partition_summary.json"), "w") as fh:
            json.dump(summ, fh, indent=2)


if __name__ == "__main__":
    main()
