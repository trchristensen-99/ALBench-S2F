"""Add absolute recipe overrides to scratchpad/menu_retrain.py: --optimizer, --lr_abs, --wd_abs (the existing --lr_scale / --bs_scale / --epoch_scale stay). Defaults leave every
existing run, including the train_override.json comparison of resumed directories, unchanged. Idempotent; writes a .bak_hp backup.

    python scripts/program_oct22/patch_menu_retrain_hp.py scratchpad/menu_retrain.py
"""

import ast
import shutil
import sys

p = sys.argv[1]
s = open(p).read()
if "--lr_abs" in s:
    print("already patched")
    sys.exit(0)
a = '    ap.add_argument("--bs_scale", type=float, default=1.0, help="multiply every entry\'s batch size; 1 = unchanged")\n'
assert a in s, "bs_scale arg not found"
s = s.replace(
    a,
    a
    + """    ap.add_argument("--optimizer", default="", choices=["", "muon", "adamw", "adam"], help="replace every entry's optimizer; empty = keep the entry's own")
    ap.add_argument("--lr_abs", type=float, default=0.0, help="replace every entry's peak learning rate (applied before --lr_scale); 0 = keep")
    ap.add_argument("--wd_abs", type=float, default=0.0, help="replace every entry's weight decay; 0 = keep")
""",
    1,
)
a = """    _ov = {"epoch_scale": args.epoch_scale, "lr_scale": args.lr_scale, "bs_scale": args.bs_scale, "al_patience": os.environ.get("AL_PATIENCE", "8")}
"""
assert a in s, "override dict not found"
s = s.replace(
    a,
    a
    + """    _new_hp = {"optimizer": args.optimizer, "lr_abs": args.lr_abs, "wd_abs": args.wd_abs}
    if any(_new_hp.values()):
        _ov.update(_new_hp)  # only when used, so override files of earlier runs still compare equal
""",
    1,
)
a = '    if (args.epoch_scale, args.lr_scale, args.bs_scale) != (1.0, 1.0, 1.0) or _ov["al_patience"] != "8":\n'
assert a in s, "override write condition not found"
s = s.replace(
    a,
    '    if (args.epoch_scale, args.lr_scale, args.bs_scale) != (1.0, 1.0, 1.0) or _ov["al_patience"] != "8" or any(_new_hp.values()):\n',
    1,
)
a = '            tc = dict(e["train_config"])\n'
assert a in s, "tc line not found"
s = s.replace(
    a,
    a
    + """            if args.optimizer:
                tc["optimizer"] = args.optimizer
            if args.lr_abs:
                tc["lr"] = args.lr_abs
            if args.wd_abs:
                tc["weight_decay"] = args.wd_abs
""",
    1,
)
ast.parse(s)
shutil.copy(p, p + ".bak_hp")
open(p, "w").write(s)
print("patched", p)
