"""Plan the HMEC primary-cell MPRA: which fastqs, which design library, which control.

HMEC is the first biosample in the corpus that is not an immortalised cancer line, so it is
the only place the `immortalised` / `malignant` fields in the cell descriptions can vary --
without it those fields are constant across training and a conditioned model cannot learn
what they mean. ENCODE released reads but no processed quantification and no
barcode-to-oligo association map, so the activity has to be counted from the fastqs.

Writes a manifest and prints the plan; downloads nothing.

    python3 -I hmec_manifest.py --out data/external/hmec_mpra
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

API = "https://www.encodeproject.org"
RNA = "ENCSR616NNF"  # HMEC MPRA, 5 biological replicates
DNA = "ENCSR720LEA"  # pooled clone sequencing of the same library (the DNA denominator)


def get(acc: str) -> dict:
    url = f"{API}/experiments/{acc}/?format=json"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as fh:
        return json.load(fh)


def describe(acc: str) -> dict:
    d = get(acc)
    bs = d.get("biosample_ontology") or {}
    out = {
        "accession": acc,
        "description": d.get("description", ""),
        "biosample": bs.get("term_name"),
        "classification": bs.get("classification"),
        "control_type": d.get("control_type"),
        "elements_references": [
            r.get("accession") if isinstance(r, dict) else str(r)
            for r in (d.get("elements_references") or [])
        ],
        "fastqs": [],
    }
    for f in d.get("files", []):
        fo = f if isinstance(f, dict) else None
        if fo is None:
            continue
        if fo.get("file_format") != "fastq" or fo.get("status") != "released":
            continue
        out["fastqs"].append(
            {
                "accession": fo["accession"],
                "bytes": fo.get("file_size"),
                "read": fo.get("paired_end"),
                "pair": (fo.get("paired_with") or "").strip("/").split("/")[-1],
                "replicate": (fo.get("replicate") or {}).get("biological_replicate_number"),
                "run_type": fo.get("run_type"),
                "read_length": fo.get("read_length"),
                "url": API + fo["href"],
            }
        )
    out["fastqs"].sort(key=lambda r: (r["replicate"] or 0, r["read"] or "0"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    man = {"rna": describe(RNA), "dna": describe(DNA)}
    for key, d in man.items():
        tot = sum(f["bytes"] or 0 for f in d["fastqs"])
        print(f"\n=== {key.upper()} {d['accession']}  {d['biosample']} [{d['classification']}]")
        print(f"    {d['description'][:100]}")
        print(f"    control_type={d['control_type']}  elements={d['elements_references']}")
        print(f"    {len(d['fastqs'])} fastq files, {tot / 1e9:.1f} GB")
        for f in d["fastqs"][:12]:
            print(
                f"      rep{f['replicate']} R{f['read']} {f['accession']} "
                f"{(f['bytes'] or 0) / 1e9:.2f} GB  {f['run_type']} {f['read_length']}bp"
            )
        if len(d["fastqs"]) > 12:
            print(f"      ... {len(d['fastqs']) - 12} more")

    refs = set(man["rna"]["elements_references"]) & set(man["dna"]["elements_references"])
    print(f"\nshared elements reference(s): {sorted(refs) or 'NONE -- check before counting'}")
    with open(os.path.join(a.out, "MANIFEST.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    print(f"wrote {a.out}/MANIFEST.json")


if __name__ == "__main__":
    main()
