"""Nov-2 evening: the live launcher (scratchpad/arch_autoresearch.py, the one every loop cell runs) stopped every model after 8 epochs without improvement, hard-coded, whatever AL_PATIENCE said and whatever
schedule the proposal asked for. Measured on the 1,498 models of the standard-input loop runs: 90% stop before 90% of their requested schedule (100% at 10k), and a model that asks for >= 200 epochs trains a median of
26 to 36. A one-cycle schedule only anneals at its end, so those models are cut at the high-learning-rate phase; the tuned LegNet recipe (e500 to e800) cannot run at all. The repository copy
(albench/arch_search/arch_autoresearch.py, used by the menu library) reads AL_PATIENCE; the launcher did not.

This patch makes the launcher read the same variables, with the old behaviour as the default so that every earlier arm and every running cell is unchanged:

  AL_PATIENCE          epochs without improvement before stopping (default 8, the old hard-coded value; the menu standard is 16)
  AL_ONECYCLE_FULL=1   a one-cycle schedule is never stopped by patience (the per-model time cap still applies); the best epoch by validation is still the one kept

Both are stamped into config.json (`harness`). Idempotent; keeps scratchpad/arch_autoresearch.py.bak_patience.

    python scripts/program_oct22/patch_loop_patience.py        (run in the repo root on the cluster)
"""

import py_compile
import shutil
import sys

P = "scratchpad/arch_autoresearch.py"
OLD = """            bad += 1
            if bad >= 8:
                break"""
NEW = """            bad += 1
            # AL_PATIENCE (default 8 as before); AL_ONECYCLE_FULL=1: a one-cycle schedule only anneals at its end, so patience never stops it (the time cap below still does)
            if bad >= int(os.environ.get("AL_PATIENCE", "8")) and not (
                os.environ.get("AL_ONECYCLE_FULL") == "1" and tc["lr_schedule"] == "onecycle"
            ):
                break"""
STAMP_OLD = (
    '("AL_EMA", "AL_BN_RECALC", "AL_FLANK_SHIFT", "AL_SHIFT_MAX", "AL_PATIENCE", "AL_PRECISION")'
)
STAMP_NEW = '("AL_EMA", "AL_BN_RECALC", "AL_FLANK_SHIFT", "AL_SHIFT_MAX", "AL_PATIENCE", "AL_PRECISION", "AL_ONECYCLE_FULL")'

s = open(P).read()
if "AL_ONECYCLE_FULL" in s:
    print("already patched")
    sys.exit(0)
assert s.count(OLD) == 1, "patience block not found exactly once"
assert s.count(STAMP_OLD) == 1, "harness stamp not found exactly once"
shutil.copy(P, P + ".bak_patience")
open(P, "w").write(s.replace(OLD, NEW).replace(STAMP_OLD, STAMP_NEW))
py_compile.compile(P, doraise=True)
print("patched", P)
