"""Width-scaled copies of the best menu configurations, to test whether larger models pay off at larger dataset sizes.

The search proposes models of about 1M parameters (median 0.9M, 90th percentile under 2.5M, maximum 12M; the 40M cap never binds), and across 0.4M to 1.8M parameters the
retrained menu shows only a weak, mostly insignificant gain with size. That range is too narrow to say what happens at 4x to 9x the parameters, so the best configurations
are rewritten with every channel / hidden width multiplied by f (f = 2 and 3: about 4x and 9x the parameters for convolutional towers), keeping the architecture and the
training recipe. The proposer model does the rewrite (it is a mechanical edit of code that is different in each of the 30 configurations); each result is rebuilt and kept only if
its parameter count grew by a plausible factor.

    python scripts/program_oct22/scale_configs.py --out outputs/scaleup [--top 8] [--factors 2 3]
writes outputs/scaleup/menu_x{f}.json (top 8) and menu_x{f}_top4.json (best 4), and outputs/scaleup/DONE.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
MR = "outputs/menu_retrain_v0"
RETRAINS = {
    "genomic": "nest_genomic_r1_d100k",
    "random": "lad_random_r1_d100k",
    "motif": "lad_motif_ct_enriched_r1_d100k",
}

SYSTEM = (
    "You are a careful PyTorch engineer. You receive the source of a 1D CNN-style sequence model (class Net(nn.Module), input (B, 4, 200), output (B,)) and a width factor f. "
    "Rewrite it so that every channel count, hidden size and embedding width is multiplied by f (round to a multiple of 8; keep the number of attention heads dividing the width). "
    "Do NOT change the depth, the connectivity, kernel sizes, dilations, pooling, activations, readout or anything else. Do not add or remove layers. Keep the class name Net and "
    "the constructor signature (no required arguments). Return exactly one ```python code block with the complete rewritten source, and nothing else."
)


def val_rank_table(skip):
    rows: dict[str, list[float]] = {}
    for d in RETRAINS.values():
        for f in glob.glob(f"{MR}/{d}/r00_menu_*_meta.json"):
            m = json.load(open(f))
            if m.get("val_pearson") is not None:
                rows.setdefault(m["menu_entry"], []).append(m["val_pearson"])
    return {
        k: float(np.mean(v)) for k, v in rows.items() if len(v) == len(RETRAINS) and k not in skip
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/scaleup")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--factors", type=int, nargs="+", default=[2, 3])
    args = ap.parse_args()
    from albench.arch_search.propose import call_model, parse_proposals
    from albench.arch_search.validate import build_and_check
    from albench.ensemble.menu import load_menu

    menu = load_menu("v0")
    table = val_rank_table(set(menu.skipped))
    best = [k for k, _ in sorted(table.items(), key=lambda kv: -kv[1])][: args.top]
    print(
        "base configurations (mean validation r over the three 100k reservoirs):",
        [(k, round(table[k], 4)) for k in best],
        flush=True,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in args.factors:
        rows = []
        for eid in best:
            e = menu[eid]
            base_params = None
            net, info = build_and_check(e.source, max_params=40_000_000, max_fwd_ms=1e9)
            if net is not None:
                base_params = int(info.n_params)
            src_new = None
            for attempt in range(3):
                reply = call_model(
                    SYSTEM, f"Width factor f = {f}.\n\n```python\n{e.source}\n```", max_tokens=16000
                )
                got = parse_proposals(reply)
                if not got:
                    continue
                cand = got[0]["source"]
                net2, v = build_and_check(cand, max_params=200_000_000, max_fwd_ms=1e9)
                if net2 is None:
                    print(
                        f"  {eid} x{f} attempt {attempt}: build failed ({v.reason[:80]})",
                        flush=True,
                    )
                    continue
                ratio = v.n_params / base_params if base_params else float("nan")
                if base_params and not (1.3 <= ratio <= 1.6 * f * f):
                    print(
                        f"  {eid} x{f} attempt {attempt}: parameter ratio {ratio:.2f} outside [1.3, {1.6 * f * f:.1f}]",
                        flush=True,
                    )
                    continue
                src_new = (cand, int(v.n_params), ratio)
                break
            if src_new is None:
                print(f"  {eid} x{f}: no valid rewrite; left out", flush=True)
                continue
            print(
                f"  {eid} x{f}: {base_params:,} -> {src_new[1]:,} parameters (x{src_new[2]:.2f})",
                flush=True,
            )
            rows.append(
                {
                    "id": f"{eid}__w{f}",
                    "cell": f"scaled x{f}",
                    "idea": f"width x{f} of {e.idea[:80]}",
                    "orig_val": None,
                    "n_params": src_new[1],
                    "source": src_new[0],
                    "train_config": e.train_config,
                    "base_id": eid,
                    "base_params": base_params,
                }
            )
        json.dump(rows, open(out / f"menu_x{f}.json", "w"), indent=1)
        json.dump(rows[:4], open(out / f"menu_x{f}_top4.json", "w"), indent=1)
        print(f"menu x{f}: {len(rows)} configurations", flush=True)
    (out / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
