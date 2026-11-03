"""Nov-3: AL_RECIPE_MODE=tuned for the live launcher (scratchpad/arch_autoresearch.py).

With recipe_source "harness" the loop's proposer writes architectures only and `sample_train` assigns every model a recipe. The existing assignment (AL_RECIPE_MODE=harness) is Muon with a small spread; for the diversity arms (`ar40_`)
every architecture is trained with the recipe that makes the reference LegNet strong at this size instead, given as AL_TUNED_RECIPE="optimizer,lr,weight_decay,batch_size,epochs,schedule" (for example
"adamw,0.01,0.2,1024,300,onecycle" at 30k), so that a non-LegNet architecture is judged under a recipe known to work, not under a small-batch Muon recipe that suits other networks. Default behaviour unchanged.
Stamped into config.json (`harness`). Idempotent; keeps scratchpad/arch_autoresearch.py.bak_tuned_recipe.

    python scripts/program_oct22/patch_loop_tuned_recipe.py        (repo root on the cluster)
"""

import py_compile
import shutil
import sys

P = "scratchpad/arch_autoresearch.py"
OLD = """    if os.environ.get("AL_RECIPE_MODE") == "harness":  # the harness assigns the recipe"""
NEW = """    if os.environ.get("AL_RECIPE_MODE") == "tuned":  # every model trains with the same, size-specific recipe (AL_TUNED_RECIPE), the one that makes the reference LegNet strong
        _o, _lr, _wd, _bs, _ep, _sch = os.environ["AL_TUNED_RECIPE"].split(",")
        return {"optimizer": _o, "lr": float(_lr), "weight_decay": float(_wd), "batch_size": int(_bs), "epochs": int(_ep), "lr_schedule": _sch}
    if os.environ.get("AL_RECIPE_MODE") == "harness":  # the harness assigns the recipe"""
STAMP_OLD = '"AL_ONECYCLE_FULL", "AL_NONFINITE_SKIP")'
STAMP_NEW = (
    '"AL_ONECYCLE_FULL", "AL_NONFINITE_SKIP", "AL_RC_MODE", "AL_RECIPE_MODE", "AL_TUNED_RECIPE")'
)

s = open(P).read()
if "AL_TUNED_RECIPE" in s:
    print("already patched")
    sys.exit(0)
assert s.count(OLD) == 1, "recipe branch not found exactly once"
assert s.count(STAMP_OLD) == 1, "harness stamp not found exactly once"
shutil.copy(P, P + ".bak_tuned_recipe")
open(P, "w").write(s.replace(OLD, NEW).replace(STAMP_OLD, STAMP_NEW))
py_compile.compile(P, doraise=True)
print("patched", P)
