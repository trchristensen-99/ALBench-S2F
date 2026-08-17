"""Collect FM scaling-driver results into a table (non-CL curves + CL arms)."""

import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/fm_scaling")
    ap.add_argument("--pattern", default="*")
    args = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(args.root, args.pattern, "fm_scaling_point.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        m = d.get("metrics", {})
        rows.append(
            (
                os.path.basename(os.path.dirname(f)),
                d.get("D"),
                d.get("cl"),
                d.get("replay_lambda"),
                d.get("genomic_preservation"),
                m.get("genomic"),
                m.get("ood"),
                m.get("sub_med"),
                d.get("train_sec"),
            )
        )
    hdr = f"{'cell':<42} {'D':>7} {'cl':<11} {'lam':>4} {'preserv':>8} {'genomic':>8} {'ood':>7} {'snv':>7} {'sec':>8}"
    print(hdr)
    print("-" * len(hdr))

    def fmt(v, n=4):
        return ("%.*f" % (n, v)) if isinstance(v, (int, float)) else "-"

    for c, D, cl, lam, pres, gen, ood, snv, sec in rows:
        print(
            f"{c:<42} {D if D else '-':>7} {str(cl):<11} {fmt(lam, 1):>4} {fmt(pres):>8} "
            f"{fmt(gen):>8} {fmt(ood):>7} {fmt(snv):>7} {fmt(sec, 0):>8}"
        )


if __name__ == "__main__":
    main()
