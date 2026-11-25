"""Read the two acquisition follow-ups: does the candidate POOL change the answer?

ab43 -- pool size and pool content, holding the rule fixed at `epistemic`:

    cap50k / cap500k / cap970k   the same motif_shared_core pool, scored at three candidate
                                 counts. Until Nov 24 every published arm silently used
                                 `--cap 200000`, so the "1M pools" were never 1M and the
                                 size axis had never actually been varied.
    cap970k_rand                 random selection at the largest cap. The required null: if
                                 the curve is flat for random too, the axis is measuring
                                 pool composition drift, not selection.
    union_epi / union_rand       shared_core + ct_enriched pooled (1.97M candidates), the
                                 "more than one reservoir on the menu" case.

ab44 -- the rules that were registered but unreachable from the AL loop until Nov 24,
run on both motif reservoirs at Arm B's recipe and seeds so they drop into that table.

Both read with read_acq52's stacking path, so the numbers are comparable to every other
acquisition arm rather than to each other only.

    python scratchpad/nov25/read_ab43_ab44.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# read_acq52 expects `ens_util` on sys.path under REPO/scratchpad, where it used to live; the
# repo consolidation moved it to orchestration/ and left the import to be satisfied by the
# caller's path. Put the real directory on first so this works from a clean checkout.
sys.path.insert(0, str(REPO / "orchestration"))
sys.path.insert(0, str(REPO / "scripts/analysis"))
from read_acq52 import SETS, read_run  # noqa: E402

PRETTY = {"genomic": "genomic", "ood": "designed", "random_32k": "random"}

GROUPS = {
    "ab43 -- candidate pool size and content (rule fixed = epistemic)": [
        ("cap50k_epi", "50k candidates"),
        ("cap500k_epi", "500k candidates"),
        ("cap970k_epi", "970k candidates"),
        ("cap970k_rand", "970k, RANDOM (null)"),
        ("union_epi", "1.97M union, epistemic"),
        ("union_rand", "1.97M union, RANDOM"),
    ],
    "ab44 -- acquisition rules that were never reachable from the loop": [
        ("motif_shared_core_activity_normalized", "shared_core  activity_norm"),
        ("motif_shared_core_expected_activity", "shared_core  expected_act"),
        ("motif_ct_enriched_activity_normalized", "ct_enriched  activity_norm"),
        ("motif_ct_enriched_expected_activity", "ct_enriched  expected_act"),
    ],
}
ROOTS = {"ab43": REPO / "outputs/acq43", "ab44": REPO / "outputs/acq44"}


def find(root: Path, name: str) -> Path | None:
    d = root / name
    if not d.is_dir():
        return None
    # The run lands one level down under bench38's own naming; take the single child.
    kids = [k for k in sorted(d.iterdir()) if k.is_dir()]
    return kids[0] if len(kids) == 1 else d


def main() -> int:
    for title, entries in GROUPS.items():
        root = ROOTS["ab43" if title.startswith("ab43") else "ab44"]
        print(f"\n=== {title} ===")
        hdr = f"{'arm':<30}{'K':>3}" + "".join(f"{PRETTY[s]:>22}" for s in SETS)
        print(hdr)
        print("-" * len(hdr))
        for name, label in entries:
            d = find(root, name)
            if d is None:
                print(f"{label:<30}  --  (not built)")
                continue
            got = read_run(d)
            if got is None:
                print(f"{label:<30}  --  (no finished members)")
                continue
            single, stack, k = got
            cells = "".join(f"{single[s]:>10.4f}/{stack[s]:<11.4f}" for s in SETS)
            print(f"{label:<30}{k:>3}{cells}")
    print("\nsingle = mean of individual members; stack = enet_all_pos over the members.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
