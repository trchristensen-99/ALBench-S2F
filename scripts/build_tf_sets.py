"""Build the ``tf_sets`` asset: which TFs are active in K562/HepG2, and which are shared.

The motif vocabulary arms need a principled, reproducible answer to "which TFs count
as cell-type-enriched". We take it from ENCODE TF ChIP-seq experiment metadata:
a factor with a released ChIP-seq experiment in a cell type has direct evidence of
binding there. That is stronger than RNA expression (a TF can be expressed without
binding) and, critically, it is independent of our oracle, our labels and our
evaluation sets -- which was the constraint agreed for motif selection.

    ct_enriched   ChIP'd in K562 or HepG2 but NOT in most other cell types
    shared_core   ChIP'd across many cell types, including ours
    k562_only / hepg2_only   the per-cell-type breakdowns, for the ct_bias arm

"Shared" is defined by breadth of cell types, not by our data. The threshold is a
parameter because there is no natural cut point; the default keeps the two sets
roughly balanced, and the counts are printed so the choice is visible.

Usage:
    python scripts/build_tf_sets.py                       # queries the ENCODE API
    python scripts/build_tf_sets.py --out data/motifs/tf_sets.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ENCODE = "https://www.encodeproject.org/search/"
TARGET_CELLS = ("K562", "HepG2")


def query(cell: str, limit: int = 2000, timeout: int = 120) -> set[str]:
    """Gene symbols of TFs with a released ChIP-seq experiment in `cell`."""
    params = {
        "type": "Experiment",
        "assay_title": "TF ChIP-seq",
        "biosample_ontology.term_name": cell,
        "status": "released",
        "format": "json",
        "limit": str(limit),
        "field": "target.label",
    }
    url = ENCODE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    out = set()
    for e in d.get("@graph", []):
        lab = (e.get("target") or {}).get("label")
        if lab:
            out.add(str(lab).upper())
    return out


def query_breadth(timeout: int = 180, limit: int = 8000) -> dict[str, set[str]]:
    """target -> set of cell types with a released TF ChIP-seq experiment."""
    params = {
        "type": "Experiment",
        "assay_title": "TF ChIP-seq",
        "status": "released",
        "format": "json",
        "limit": str(limit),
        "field": "target.label",
    }
    url = ENCODE + "?" + urllib.parse.urlencode(params) + "&field=biosample_ontology.term_name"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    breadth: dict[str, set[str]] = defaultdict(set)
    for e in d.get("@graph", []):
        lab = (e.get("target") or {}).get("label")
        cell = (e.get("biosample_ontology") or {}).get("term_name")
        if lab and cell:
            breadth[str(lab).upper()].add(str(cell))
    return breadth


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="default: <ALBENCH_DATA>/motifs/tf_sets.yaml")
    # Absolute cell-type counts make a lopsided split because ENCODE depth varies
    # enormously by factor: at >=8 cell types only 44 TFs qualify as shared. Splitting
    # on PERCENTILES of the breadth distribution (restricted to TFs present in our two
    # cell types) keeps both arms usable and adapts as ENCODE grows.
    ap.add_argument(
        "--definition",
        choices=("both_cells", "breadth"),
        default="both_cells",
        help="both_cells: shared = ChIP'd in BOTH K562 and HepG2 (direct evidence, "
        "default). breadth: shared = ChIP'd across many cell types (weak signal -- "
        "the median TF here has only one cell type).",
    )
    ap.add_argument(
        "--shared-pct",
        type=float,
        default=66.0,
        help="TFs above this percentile of cell-type breadth are shared_core",
    )
    ap.add_argument(
        "--specific-pct",
        type=float,
        default=33.0,
        help="TFs below this percentile of cell-type breadth are ct_enriched",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from albench.paths import ASSETS, data_root

    out = Path(args.out) if args.out else data_root() / ASSETS["tf_sets"].default_relpath

    print("querying ENCODE for TF ChIP-seq targets ...")
    per_cell = {c: query(c) for c in TARGET_CELLS}
    for c, s in per_cell.items():
        print(f"  {c}: {len(s)} TFs with a released ChIP-seq experiment")
    ours = set().union(*per_cell.values())

    breadth = query_breadth()
    print(
        f"  breadth table: {len(breadth)} TFs across "
        f"{len({c for v in breadth.values() for c in v})} cell types"
    )

    import numpy as np

    # PRIMARY DEFINITION (default). "Shared" and "specific" are questions about OUR
    # two cell types, so answer them with direct evidence in those two rather than
    # with ENCODE's coverage of 218 others. Breadth turns out to be a weak signal:
    # the median TF here has been ChIP'd in ONE cell type, so a percentile split lands
    # on "1 vs 2 cell types", which does not mean shared vs specific.
    k562, hepg2 = per_cell["K562"], per_cell["HepG2"]
    if args.definition == "both_cells":
        shared = k562 & hepg2
        specific = k562 ^ hepg2
        print(f"\n  definition=both_cells")
        print(f"    ChIP'd in BOTH K562 and HepG2  -> shared_core   : {len(shared)}")
        print(f"    ChIP'd in exactly one of them  -> ct_enriched   : {len(specific)}")
        print(f"      K562-only {len(k562 - hepg2)}, HepG2-only {len(hepg2 - k562)}")
        sets_extra = {"k562_only": sorted(k562 - hepg2), "hepg2_only": sorted(hepg2 - k562)}
    else:
        sets_extra = {}

    width = {t: len(breadth.get(t, ())) for t in ours}
    vals = np.array(list(width.values()), float)
    hi = np.percentile(vals, args.shared_pct)
    lo = np.percentile(vals, args.specific_pct)
    print(
        f"  breadth among our TFs: median {np.median(vals):.0f} cell types, "
        f"p{args.specific_pct:.0f}={lo:.0f}, p{args.shared_pct:.0f}={hi:.0f}"
    )
    if args.definition == "breadth":
        shared = {t for t, w in width.items() if w >= hi}
        specific = {t for t, w in width.items() if w <= lo}

    sets = {
        "ct_enriched": sorted(specific),
        "shared_core": sorted(shared),
        "k562": sorted(per_cell["K562"]),
        "hepg2": sorted(per_cell["HepG2"]),
        **sets_extra,
        "_meta": {
            "source": "ENCODE TF ChIP-seq experiment metadata (released)",
            "definition": args.definition,
            "shared_pct": args.shared_pct,
            "specific_pct": args.specific_pct,
            "note": (
                "ChIP evidence, not expression: a TF can be expressed without binding. "
                "Independent of our oracle, labels and evaluation sets."
            ),
        },
    }
    print(f"\n  ct_enriched (breadth <= p{args.specific_pct:.0f}): {len(specific)}")
    print(f"  shared_core (breadth >= p{args.shared_pct:.0f}): {len(shared)}")
    print(f"  overlap (should be 0): {len(specific & shared)}")
    if not specific or not shared:
        print("\nWARNING: one of the sets is empty; adjust the thresholds.", file=sys.stderr)

    import yaml

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(sets, sort_keys=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
