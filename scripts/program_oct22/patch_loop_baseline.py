"""AutoResearch launcher: with --reference_legnet and no --baseline_file, train the published MPRA-LegNet once in the run's own harness (same pool, same input window, its published recipe) and show the proposer its validation
Pearson r as the bar to beat. The text is written to <out_dir>/baseline_legnet.txt and reused when the run resumes.

    python scripts/program_oct22/patch_loop_baseline.py scratchpad/arch_autoresearch.py      (idempotent; backup .bak_blr)
"""

import ast
import shutil
import sys

BLOCK = """    if args.reference_legnet and not args.baseline_file:  # measure the baseline in this very harness so the bar is on the same scale as every proposal
        _bl = Path(args.out_dir) / "baseline_legnet.txt"
        if not _bl.exists():
            try:
                from albench.arch_search.mpra_legnet_ref import SOURCE as _LSRC

                _m, _v = build_and_check(_LSRC, seq_len=args.seq_len, max_params=args.max_params)
                _m, _info, _ = train_one(_wrap(_m), Xtr, ytr, Xva, yva, {"optimizer": "adamw", "lr": 0.01, "weight_decay": 0.1, "batch_size": 256, "epochs": 60, "lr_schedule": "onecycle"}, max_sec=cap_min * 60)
                if _m is not None:
                    _r = pear(predict_rc(_m, Xva), yva)
                    _bl.write_text(
                        f"BASELINE: the published MPRA-LegNet (shown below), trained by this harness on this training set ({len(tr_s):,} sequences) with its published recipe (AdamW, learning rate 0.01, weight decay 0.1, one-cycle), "
                        f"reaches validation Pearson r = {_r:.3f}. Beating it is the bar for a single model; what is scored is the ensemble of everything you train."
                    )
                    print(f"  baseline LegNet val r = {_r:.4f}", flush=True)
            except Exception as _ex:  # noqa: BLE001 - the bar is advice, never a reason to stop the search
                print(f"  baseline LegNet failed: {type(_ex).__name__}: {str(_ex)[:100]}", flush=True)
        if _bl.exists():
            args.baseline_file = str(_bl)

"""

p = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/arch_autoresearch.py"
s = open(p).read()
if "baseline_legnet.txt" in s:
    print("already patched")
    sys.exit(0)
anchor = '    if args.loop == "agent":  # opt-in; everything below this branch (the default loop) is untouched\n'
assert anchor in s, "anchor not found"
s = s.replace(anchor, BLOCK + anchor, 1)
ast.parse(s)
shutil.copy(p, p + ".bak_blr")
open(p, "w").write(s)
print("patched", p)
