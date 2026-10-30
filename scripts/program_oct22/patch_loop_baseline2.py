"""AutoResearch launcher: the in-run LegNet baseline is the BEST of the two tuned recipes (e100: batch 1024, 100 epochs; b256: batch 256, 60 epochs; both AdamW, lr 0.01, weight decay 0.1, one-cycle) instead of
only b256 called "published". With the context input the 100-epoch recipe is 0.007 higher on validation (20-seed means 0.882 against 0.875 at 30k), and the reference check of the explore-then-exploit manager
and the bar shown to the proposer must be the best known version of the model, not a weak one. The text names the recipe that won. Also adds one factual sentence on capacity to the reference note.

    python scripts/program_oct22/patch_loop_baseline2.py scratchpad/arch_autoresearch.py      (idempotent; backup .bak_blr2)
"""

import ast
import shutil
import sys

START = "    if args.reference_legnet and not args.baseline_file:"
END = '    if args.loop == "agent":  # opt-in; everything below this branch (the default loop) is untouched\n'
BLOCK = """    if args.reference_legnet and not args.baseline_file:  # measure the baseline in this very harness so the bar is on the same scale as every proposal
        _bl = Path(args.out_dir) / "baseline_legnet.txt"
        if not _bl.exists():
            try:
                from albench.arch_search.mpra_legnet_ref import SOURCE as _LSRC

                _recipes = {
                    "e100": {"optimizer": "adamw", "lr": 0.01, "weight_decay": 0.1, "batch_size": 1024, "epochs": 100, "lr_schedule": "onecycle"},
                    "b256": {"optimizer": "adamw", "lr": 0.01, "weight_decay": 0.1, "batch_size": 256, "epochs": 60, "lr_schedule": "onecycle"},
                }
                _best = None
                for _name, _tc in _recipes.items():
                    _m, _v = build_and_check(_LSRC, seq_len=args.seq_len, max_params=args.max_params)
                    _m, _info, _ = train_one(_wrap(_m), Xtr, ytr, Xva, yva, _tc, max_sec=cap_min * 60)
                    if _m is not None:
                        _r = pear(predict_rc(_m, Xva), yva)
                        print(f"  baseline LegNet ({_name}) val r = {_r:.4f}", flush=True)
                        if _best is None or _r > _best[0]:
                            _best = (_r, _name, _tc)
                if _best is not None:
                    _r, _name, _tc = _best
                    _bl.write_text(
                        f"BASELINE: the MPRA-LegNet (shown below), trained by this harness on this training set ({len(tr_s):,} sequences) with the best of its two tuned recipes ({_name}: AdamW, learning rate 0.01, weight decay 0.1, "
                        f"batch {_tc['batch_size']}, {_tc['epochs']} epochs, one-cycle), reaches validation Pearson r = {_r:.3f}. Beating it is the bar for a single model; what is scored is the ensemble of everything you train."
                    )
            except Exception as _ex:  # noqa: BLE001 - the bar is advice, never a reason to stop the search
                print(f"  baseline LegNet failed: {type(_ex).__name__}: {str(_ex)[:100]}", flush=True)
        if _bl.exists():
            args.baseline_file = str(_bl)

"""
NOTE_OLD = "It is a starting point and a bar, not a template: the goal is a set of strong architectures that differ from each other and from it, so propose different designs as well as improvements of it."
NOTE_NEW = (
    "It has about 1.3 million parameters; whether a larger or a smaller model is better can depend on the training-set size (in a first test, doubling the channel widths of our earlier configurations at 100k sequences changed the genomic-test r "
    "by -0.002 to +0.004, within noise), so capacity is worth testing, not assuming. " + NOTE_OLD
)

p = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/arch_autoresearch.py"
s = open(p).read()
if "best of its two tuned recipes" in s:
    print("already patched")
    sys.exit(0)
i, j = s.index(START), s.index(END)
assert i < j, "block order"
s = s[:i] + BLOCK + s[j:]
if NOTE_OLD in s and "capacity is worth testing" not in s:
    s = s.replace(NOTE_OLD, NOTE_NEW, 1)
ast.parse(s)
shutil.copy(p, p + ".bak_blr2")
open(p, "w").write(s)
print("patched", p, "| note updated:", "capacity is worth testing" in s)
