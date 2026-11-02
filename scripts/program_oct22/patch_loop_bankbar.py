"""AutoResearch launcher: the reference bar comes from the 20-seed LegNet banks instead of from a LegNet trained inside the loop.

The in-run baseline (a LegNet trained once in the loop's own harness, 100 epochs) is noisy and off the scale of the banks: the same recipe scored 0.644 and 0.784 validation r in two 10k runs, and a tuned in-harness
baseline scored 0.618 at 10k where the 20-seed bank scores 0.809. The bar shown to the proposer is now the bank's mean validation r on V0c (docs/TUNED_LEGNET.md): the tuned per-size recipe first, the out-of-the-box recipe
beside it, with a note that networks trained by the loop's harness can score lower than the reference script for the same architecture. Nothing is trained for the bar, which also saves the GPU time. The first
"validation Pearson r = X" in baseline_legnet.txt (read by the manager's reference check and the target sentence) is the tuned bank mean. Sizes without a bank get no baseline text.

    python scripts/program_oct22/patch_loop_bankbar.py scratchpad/arch_autoresearch.py      (idempotent; backup .bak_bankbar)
"""

import ast
import shutil
import sys

START = "    if args.reference_legnet and not args.baseline_file:"
END = '    if args.loop == "agent":  # opt-in; everything below this branch (the default loop) is untouched\n'
BLOCK = """    if args.reference_legnet and not args.baseline_file:  # bank-derived reference bar (docs/TUNED_LEGNET.md)
        _bl = Path(args.out_dir) / "baseline_legnet.txt"
        _bank = {  # D: (tuned recipe description, tuned bank mean val r, out-of-the-box bank mean val r); 20 seeds each, V0c
            10000: ("weight decay 0.4, 800 epochs", 0.809, 0.756),
            30000: ("weight decay 0.2, 500 epochs", 0.905, 0.882),
            100000: ("weight decay 0.1, 500 epochs", 0.945, 0.940),
        }.get(int(args.D))
        if not _bl.exists() and _bank is not None:
            _bl.write_text(
                f"BASELINE: the MPRA-LegNet (shown below), trained 20 times with its reference script on {len(tr_s):,} sequences of this kind, reaches validation Pearson r = {_bank[1]:.3f} (mean over seeds; tuned recipe: "
                f"AdamW, learning rate 0.01, {_bank[0]}, batch 1024, one-cycle). Its out-of-the-box recipe (weight decay 0.1, 100 epochs) reaches {_bank[2]:.3f}. Networks trained by this harness can score lower than the reference "
                f"script does for the same architecture, so treat the first value as the level a single model should approach; what is scored is the ensemble of everything you train."
            )
        if _bl.exists():
            args.baseline_file = str(_bl)

"""

p = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/arch_autoresearch.py"
s = open(p).read()
if "bank-derived reference bar" in s:
    print("already patched")
    sys.exit(0)
i, j = s.index(START), s.index(END)
assert i < j, "block order"
s = s[:i] + BLOCK + s[j:]
ast.parse(s)
shutil.copy(p, p + ".bak_bankbar")
open(p, "w").write(s)
print("patched", p)
