"""menu_retrain.py: oracle-like shift augmentation and shift test-time averaging for the flank-context students (new flags, defaults reproduce the current behaviour).

--shift_per_sample   draw the training shift per sample instead of one shift per batch
--shift_prob P       apply the shift to a fraction P of the samples (the oracle shifts +-15 on half of its samples); default 1
--tta K              at evaluation average the predictions over the real-flank windows at offsets -K, -K/2, 0, K/2, K (K <= --shift_max)

python scripts/program_oct22/patch_menu_retrain_aug.py [scratchpad/menu_retrain.py]      (idempotent; backup .bak_aug)
"""

import ast
import shutil
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/menu_retrain.py"
s = open(p).read()
if "--shift_per_sample" in s:
    print("already patched")
    sys.exit(0)


def sub(old, new):
    global s
    assert old in s, f"anchor not found: {old[:70]}"
    s = s.replace(old, new, 1)


sub(
    '    ap.add_argument("--save_models"',
    '    ap.add_argument("--shift_per_sample", action="store_true", help="flank-context path: draw the training shift per sample (default: one shift per batch)")\n'
    '    ap.add_argument("--shift_prob", type=float, default=1.0, help="flank-context path: fraction of samples that are shifted (the oracle uses 0.5)")\n'
    '    ap.add_argument("--tta", type=int, default=0, help="flank-context path: average evaluation predictions over real-flank windows at offsets -K, -K/2, 0, K/2, K")\n'
    '    ap.add_argument("--save_models"',
)
sub(
    "net = wrap_ctx(net, ctx_F, ctx_S)",
    "net = wrap_ctx(net, ctx_F, ctx_S, per_sample=args.shift_per_sample, prob=args.shift_prob, tta=args.tta)",
)
sub(
    '        _new_harness.update({"flank_ctx": ctx_F, "shift_max": ctx_S})',
    '        _new_harness.update({"flank_ctx": ctx_F, "shift_max": ctx_S})\n'
    "        if args.shift_per_sample or args.shift_prob != 1.0 or args.tta:\n"
    '            _new_harness.update({"shift_per_sample": bool(args.shift_per_sample), "shift_prob": args.shift_prob, "tta": args.tta})',
)
sub(
    '                meta["harness"].update({"flank_ctx": ctx_F, "shift_max": ctx_S})',
    '                meta["harness"].update({"flank_ctx": ctx_F, "shift_max": ctx_S, "shift_per_sample": bool(args.shift_per_sample), "shift_prob": args.shift_prob, "tta": args.tta})',
)
ast.parse(s)
shutil.copy(p, p + ".bak_aug")
open(p, "w").write(s)
print("patched", p)
