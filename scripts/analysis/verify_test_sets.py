"""Guard: assert every K562 eval set is the intended one before any number is reported.

Written because two silent problems were found by hand. First, 7.9% of "SNV" pairs in the canonical
monoallelic file have ref and alt of DIFFERENT LENGTHS - they are indels or truncations, not
substitutions - and the wider 45,543-pair file is only 66% true SNVs, with thousands of pairs whose
ref and alt are largely unrelated (Hamming > 40). Second, filenames are not trustworthy on their
own: a file labelled "hashfrag" turned out to be chromosome-clean, and several similarly named SNV
files differ in size and construction. Both classes of error are invisible in a metric.

Every check is a hard assertion, so this can be run in CI or as a preflight step. Exits non-zero on
any failure.
"""

import argparse
import sys

import numpy as np

HELD_OUT = {"7", "13"}
FAIL = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def hamming(r, a):
    return np.array(
        [sum(1 for x, y in zip(u, v) if x != y) if len(u) == len(v) else -1 for u, v in zip(r, a)]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="data/k562/test_sets_ag_s2_chrsplit")
    ap.add_argument(
        "--require_snv_filter",
        action="store_true",
        help="require the SNV file to contain ONLY Hamming==1 pairs",
    )
    args = ap.parse_args()
    d = args.battery.rstrip("/")

    print("SNV pairs")
    z = np.load(f"{d}/snv_oracle.npz", allow_pickle=True)
    r = [str(x) for x in z["ref_sequences"]]
    a = [str(x) for x in z["alt_sequences"]]
    ham = hamming(r, a)
    n = len(r)
    n_snv = int((ham == 1).sum())
    print(
        f"  n={n:,}  true SNVs (Hamming==1)={n_snv:,} ({n_snv / n:.1%})  "
        f"length-mismatched={int((ham == -1).sum()):,}  Hamming>3={int((ham > 3).sum()):,}"
    )
    check("test_set_version" in z.files, "carries a test_set_version stamp")
    if "test_set_version" in z.files:
        print(f"       version = {str(z['test_set_version'])}")
    check(bool(z["monoallelic"]) if "monoallelic" in z.files else False, "flagged monoallelic")
    if "pair_keys" in z.files:
        ch = {str(k).split(":")[0].replace("chr", "") for k in z["pair_keys"]}
        check(ch <= HELD_OUT, f"all pairs on held-out chromosomes (found {sorted(ch)[:6]})")
        vk = [str(k) for k in z["pair_keys"]]
        check(len(set(vk)) == len(vk), "one pair per variant (no repeats)")
    check(int((ham > 3).sum()) == 0, "no grossly mismatched pairs (Hamming>3)")
    if args.require_snv_filter:
        check(n_snv == n, "SNV file contains ONLY true substitutions")
    else:
        print(
            f"  NOTE  {n - n_snv:,} non-substitution pairs present -> any delta metric MUST "
            f"filter to Hamming==1"
        )
    td = np.asarray(z["true_delta"], float)
    chk = np.asarray(z["true_alt_label"], float) - np.asarray(z["true_ref_label"], float)
    check(float(np.nanmax(np.abs(td - chk))) < 1e-4, "true_delta == alt - ref")

    print("\nAbsolute-activity sets")
    for fn, tag in (
        ("genomic_oracle.npz", "genomic reference"),
        ("ood_oracle.npz", "designed high-activity"),
        ("ctrl_neg_oracle.npz", "negative controls"),
    ):
        try:
            y = np.load(f"{d}/{fn}", allow_pickle=True)
        except FileNotFoundError:
            check(False, f"{tag}: {fn} present")
            continue
        has = "true_label" in y.files
        check(has, f"{tag}: carries real labels")
        if has:
            lab = np.asarray(y["true_label"], float)
            print(
                f"       {tag}: n={len(lab):,}  mean={np.nanmean(lab):.3f}  sd={np.nanstd(lab):.3f}"
            )
        seqs = [str(s) for s in y["sequences"]]
        check(len(set(seqs)) == len(seqs), f"{tag}: no duplicate sequences")

    print()
    if FAIL:
        print(f"{len(FAIL)} CHECK(S) FAILED:")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
