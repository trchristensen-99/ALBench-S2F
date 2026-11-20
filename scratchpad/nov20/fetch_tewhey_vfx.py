"""Stage the Tewhey `variant effect quantifications` tables from ENCODE.

The element-quantification files we already have carry a per-oligo activity, and only the
2024-04-25 batch carries Log2Skew. ENCODE publishes a separate, small output type -- 42
files, 0.76 GB in all -- that is the per-VARIANT effect table. If it covers the 2023
depositions it gives measured allelic skew in A549, GM12878, HCT116, HepG2, K562 and
SK-N-SH rather than only the three 2024 cell lines.

Own directory, manifest written, nothing executed from the download path.

    python3 -I scratchpad/fetch_tewhey_vfx.py --out data/external/tewhey_vfx
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

API = "https://www.encodeproject.org"
SEARCH = (
    f"{API}/search/?type=FunctionalCharacterizationExperiment"
    "&lab.title=Ryan+Tewhey%2C+JAX&assay_title=MPRA&limit=all&format=json"
)


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as fh:
        return json.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    g = get_json(SEARCH)["@graph"]
    todo = []
    for i, exp in enumerate(g):
        acc = exp["accession"]
        d = get_json(f"{API}/experiments/{acc}/?format=json")
        cell = (d.get("biosample_ontology") or {}).get("term_name", "unknown")
        for f in d.get("files", []):
            if f.get("output_type") != "variant effect quantifications":
                continue
            if f.get("status") != "released":
                continue
            todo.append(
                {
                    "experiment": acc,
                    "cell": cell,
                    "file": f["accession"],
                    "bytes": f.get("file_size"),
                    "released": d.get("date_released"),
                    "format": f.get("file_format"),
                    "url": API + f["href"],
                }
            )
        if (i + 1) % 15 == 0:
            print(f"  scanned {i + 1}/{len(g)}", flush=True)

    print(
        f"{len(todo)} variant-effect files, {sum(r['bytes'] or 0 for r in todo) / 1e9:.2f} GB",
        flush=True,
    )
    with open(os.path.join(a.out, "MANIFEST.json"), "w") as fh:
        json.dump(
            {"source": SEARCH, "output_type": "variant effect quantifications", "files": todo},
            fh,
            indent=1,
        )

    for i, r in enumerate(todo):
        ext = {"tsv": "tsv", "bed": "bed.gz"}.get(r["format"], r["format"] or "dat")
        dest = os.path.join(
            a.out,
            f"{r['cell'].replace('/', '_')}__{r['released']}__{r['experiment']}__{r['file']}.{ext}",
        )
        if os.path.exists(dest) and os.path.getsize(dest) == r["bytes"]:
            continue
        urllib.request.urlretrieve(r["url"], dest)
        print(f"  {i + 1}/{len(todo)} {os.path.basename(dest)}", flush=True)
    print(f"done -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
