"""Nov-2 evening (audit finding F5): under AL_PRECISION=fp16 the live launcher aborted a model at the FIRST non-finite training loss, before the GradScaler could act (it only protects gradients, not the loss).
A proposal whose forward overflows fp16 on one batch was therefore dropped from the history and the ensemble although the same network trains in fp32: a precision-dependent selection that would confound every
"39" arm (fixed harness, fp16) against the "38" arms (fp32).

AL_NONFINITE_SKIP=1 (default off: nothing earlier changes) makes a non-finite loss under the fp16 scaler skip that batch and aborts only after 20 skipped batches in a row; the number of skipped batches is stamped
into the model's info (`nonfinite_skipped`). The variable is stamped into config.json (`harness`). Idempotent; keeps scratchpad/arch_autoresearch.py.bak_nonfinite.

    python scripts/program_oct22/patch_loop_nonfinite.py        (run in the repo root on the cluster)
"""

import py_compile
import shutil
import sys

P = "scratchpad/arch_autoresearch.py"
OLD = """            if not torch.isfinite(loss):
                return None, {"error": f"non-finite loss ep{ep}"}, used
"""
NEW = """            if not torch.isfinite(loss):
                if scaler is not None and os.environ.get("AL_NONFINITE_SKIP") == "1":  # fp16: skip the batch, give up after 20 in a row
                    _nf_run += 1
                    _nf_total += 1
                    if _nf_run > 20:
                        return None, {"error": f"non-finite loss ep{ep} (20 batches in a row)"}, used
                    continue
                return None, {"error": f"non-finite loss ep{ep}"}, used
            _nf_run = 0
"""
INIT_OLD = "    best, best_state, bad = np.inf, None, 0\n"
INIT_NEW = "    best, best_state, bad = np.inf, None, 0\n    _nf_run = _nf_total = 0\n"
RET_OLD = """    return model, {"best_val_mse": best, "epochs_trained": ep + 1, "optimizer_used": used, **extra}, used"""
RET_NEW = """    if _nf_total:
        extra["nonfinite_skipped"] = _nf_total
    return model, {"best_val_mse": best, "epochs_trained": ep + 1, "optimizer_used": used, **extra}, used"""
STAMP_OLD = '"AL_PRECISION", "AL_ONECYCLE_FULL")'
STAMP_NEW = '"AL_PRECISION", "AL_ONECYCLE_FULL", "AL_NONFINITE_SKIP")'

s = open(P).read()
if "AL_NONFINITE_SKIP" in s:
    print("already patched")
    sys.exit(0)
for name, old in (
    ("loss block", OLD),
    ("init", INIT_OLD),
    ("return", RET_OLD),
    ("stamp", STAMP_OLD),
):
    assert s.count(old) == 1, f"{name} not found exactly once"
shutil.copy(P, P + ".bak_nonfinite")
s = (
    s.replace(OLD, NEW)
    .replace(INIT_OLD, INIT_NEW)
    .replace(RET_OLD, RET_NEW)
    .replace(STAMP_OLD, STAMP_NEW)
)
open(P, "w").write(s)
py_compile.compile(P, doraise=True)
print("patched", P)
