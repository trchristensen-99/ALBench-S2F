"""Fetch published quantifications for the experiments that share HMEC's oligo library.

HMEC's two element references (ENCSR091QEV, ENCSR935CNL) are also used by experiments in
K562, HepG2, SK-N-SH, GM12878 and HEK293 -- the SAME 101 bp oligos, under the SAME rsID
names. Tewhey published activity for those; ENCODE published nothing for HMEC, which is why
we counted it from fastq ourselves.

That makes these tables the external check our HMEC numbers otherwise lack. The 7-cell
corpus cannot serve: it is a different library generation whose IDs are coordinates
(``chr:pos:ref:alt:allele:window``) rather than rsIDs, so nothing joins.

    python3 -I fetch_sibling_quant.py --out data/external/hmec_siblings
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

API = "https://www.encodeproject.org"
EXPERIMENTS = [
    "ENCSR147BIK",
    "ENCSR181PJG",
    "ENCSR812PWW",
    "ENCSR885ULM",
    "ENCSR481QRH",
    "ENCSR781PSG",
]
WANTED = {"element quantifications", "variant effect quantifications"}


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as fh:
        return json.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    manifest = []
    for acc in EXPERIMENTS:
        d = get(f"{API}/experiments/{acc}/?format=json")
        cell = (d.get("biosample_ontology") or {}).get("term_name", "unknown")
        ctrl = d.get("control_type")
        picked = 0
        for f in d.get("files", []):
            if not isinstance(f, dict):
                continue
            ot = (f.get("output_type") or "").lower()
            if ot not in WANTED or f.get("status") != "released":
                continue
            safe = cell.replace(" ", "-")
            dest = os.path.join(a.out, f"{safe}__{acc}__{f['accession']}.tsv")
            if not os.path.exists(dest):
                urllib.request.urlretrieve(API + f["href"], dest)
            manifest.append(
                {
                    "experiment": acc,
                    "cell": cell,
                    "control_type": ctrl,
                    "file": f["accession"],
                    "output_type": ot,
                    "path": dest,
                    "bytes": os.path.getsize(dest),
                }
            )
            picked += 1
        print(f"  {acc}  {cell:26s} control={ctrl}  {picked} quantification file(s)", flush=True)

    with open(os.path.join(a.out, "MANIFEST.json"), "w") as fh:
        json.dump({"files": manifest}, fh, indent=1)
    print(f"\n{len(manifest)} files -> {a.out}")


if __name__ == "__main__":
    main()
