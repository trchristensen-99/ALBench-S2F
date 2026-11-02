"""NOT APPLIED (reverted Nov 2): the in-harness baseline is not on the scale of the seed-bank LegNet (10k: 0.639 / 0.618 here against 0.756 / 0.808 for the 20-seed banks, and the same recipe gave 0.644 and 0.784 in two earlier 10k runs),
and the manually launched cells had no LLM credentials. Use a bar taken from the seed banks instead of retraining the baseline inside the loop.

AutoResearch launcher: the in-run LegNet baseline is the TUNED per-size recipe of docs/TUNED_LEGNET.md, and the out-of-the-box recipe is reported beside it.

Until Nov 2 the bar shown to the proposer (and used by the manager's reference check) was the best of the published recipe at 100 epochs and a 60-epoch batch-256 variant. That recipe is under-trained: with 500 to 800
epochs and per-size weight decay a single LegNet gains +0.066 (10k), +0.024 (30k) and +0.007 (100k) genomic-test r, so the bar was too low and the loop could look better than it is. The baseline is now trained twice in the
run's own harness on the run's own pool: the out-of-the-box recipe (AdamW, lr 0.01, wd 0.1, batch 1024, 100 epochs) and the tuned recipe for the size (10k: 800 epochs, wd 0.4; 30k: 500 epochs, wd 0.2; otherwise 500 epochs,
wd 0.1). The text names the better one first, so the first "validation Pearson r = X" in baseline_legnet.txt (which the manager and the target sentence read) is the bar, and reports the other beside it. Early stopping is
switched off for these two trainings (the long schedules improve until the learning rate has annealed), then restored.

    python scripts/program_oct22/patch_loop_tuned.py scratchpad/arch_autoresearch.py      (idempotent; backup .bak_tuned)
"""

import ast
import shutil
import sys

START = "    if args.reference_legnet and not args.baseline_file:"
END = '    if args.loop == "agent":  # opt-in; everything below this branch (the default loop) is untouched\n'
BLOCK = """    if args.reference_legnet and not args.baseline_file:  # measure the baseline in this very harness so the bar is on the same scale as every proposal
        _bl = Path(args.out_dir) / "baseline_legnet.txt"
        if not _bl.exists():
            _pat = os.environ.get("AL_PATIENCE")
            try:
                from albench.arch_search.mpra_legnet_ref import SOURCE as _LSRC

                _tep, _twd = {10000: (800, 0.4), 30000: (500, 0.2)}.get(int(args.D), (500, 0.1))  # tuned per-size recipe (docs/TUNED_LEGNET.md)
                _recipes = {
                    "out-of-the-box": {"optimizer": "adamw", "lr": 0.01, "weight_decay": 0.1, "batch_size": 1024, "epochs": 100, "lr_schedule": "onecycle"},
                    "tuned": {"optimizer": "adamw", "lr": 0.01, "weight_decay": _twd, "batch_size": 1024, "epochs": _tep, "lr_schedule": "onecycle"},
                }
                os.environ["AL_PATIENCE"] = "100000"  # the long schedules are meant to run to the end of the one-cycle
                _res = {}
                for _name, _tc in _recipes.items():
                    _m, _v = build_and_check(_LSRC, seq_len=args.seq_len, max_params=args.max_params)
                    _m, _info, _ = train_one(_wrap(_m), Xtr, ytr, Xva, yva, _tc, max_sec=cap_min * 60)
                    if _m is not None:
                        _res[_name] = (pear(predict_rc(_m, Xva), yva), _tc)
                        print(f"  baseline LegNet ({_name}) val r = {_res[_name][0]:.4f}", flush=True)
                if _res:
                    _order = sorted(_res, key=lambda k: -_res[k][0])
                    _b, _o = _order[0], (_order[1] if len(_order) > 1 else None)
                    _r, _tc = _res[_b]
                    _txt = (
                        f"BASELINE: the MPRA-LegNet (shown below), trained by this harness on this training set ({len(tr_s):,} sequences) with its best recipe ({_b}: AdamW, learning rate 0.01, "
                        f"weight decay {_tc['weight_decay']}, batch {_tc['batch_size']}, {_tc['epochs']} epochs, one-cycle), reaches validation Pearson r = {_r:.3f}."
                    )
                    if _o is not None:
                        _ro, _tco = _res[_o]
                        _txt += f" The other recipe ({_o}: weight decay {_tco['weight_decay']}, {_tco['epochs']} epochs) reaches {_ro:.3f}."
                    _txt += " Beating the first value is the bar for a single model; what is scored is the ensemble of everything you train."
                    _bl.write_text(_txt)
            except Exception as _ex:  # noqa: BLE001 - the bar is advice, never a reason to stop the search
                print(f"  baseline LegNet failed: {type(_ex).__name__}: {str(_ex)[:100]}", flush=True)
            finally:
                if _pat is None:
                    os.environ.pop("AL_PATIENCE", None)
                else:
                    os.environ["AL_PATIENCE"] = _pat
        if _bl.exists():
            args.baseline_file = str(_bl)

"""

p = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/arch_autoresearch.py"
s = open(p).read()
if "tuned per-size recipe (docs/TUNED_LEGNET.md)" in s:
    print("already patched")
    sys.exit(0)
i, j = s.index(START), s.index(END)
assert i < j, "block order"
s = s[:i] + BLOCK + s[j:]
ast.parse(s)
shutil.copy(p, p + ".bak_tuned")
open(p, "w").write(s)
print("patched", p)
