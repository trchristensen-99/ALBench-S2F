"""menu_retrain.py: flank-context students and harness stamping (new flags default off, so existing cells and their resume guards are untouched).

    --flank_ctx F   the student gets F real plasmid bases on each side of the 200-bp insert as input (architecture built for 200 + 2F bases); the training shift then moves the window over real bases
    --shift_max S   the +-S shift of the training window (default 10 when --flank_ctx is given); with --flank_ctx 0 this is the shift-on-real-bases arm with a different range
Either flag switches to albench.arch_search.flank_context (encoding margin F + S per side, WideCrop wrapper; the harness's own zero-padded shift is turned off). The saved model files carry the wrapper's state.
Also stamps the harness switches AL_EMA / AL_BN_RECALC and the context flags into train_override.json and each model's meta, so a run's provenance says what was on.

    python scripts/program_oct22/patch_menu_retrain_ctx.py [scratchpad/menu_retrain.py]       (idempotent; backup .bak_ctx)
"""

import ast
import re
import shutil
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/menu_retrain.py"
s = open(p).read()
if "--flank_ctx" in s:
    print("already patched")
    sys.exit(0)


def sub(old, new, count=1):
    global s
    assert s.count(old) >= 1, f"anchor not found: {old[:70]}"
    s = s.replace(old, new, count)


s, n_sub = re.subn(r"one_hot\(([A-Za-z_\[\]0-9]+), args\.seq_len\)", r"_enc(\1)", s)
assert n_sub == 6, n_sub
sub(
    '    ap.add_argument("--save_models"',
    '    ap.add_argument("--flank_ctx", type=int, default=-1, help="F real plasmid bases of input context on each side of the 200-bp insert (shift then runs over real bases); -1 = off, the zero-padded shift as before")\n'
    '    ap.add_argument("--shift_max", type=int, default=-1, help="+-shift of the training window in the flank-context path (default 10 when --flank_ctx is given); -1 = off")\n'
    '    ap.add_argument("--save_models"',
)
sub(
    "    from albench.arch_search.train_generated import one_hot\n",
    "    from albench.arch_search.train_generated import one_hot\n"
    "    from albench.arch_search.flank_context import wide_onehot, wrap as wrap_ctx\n",
)
sub(
    "    out = Path(args.out_dir)\n    out.mkdir(parents=True, exist_ok=True)\n    k, n = ",
    "    wide = args.flank_ctx >= 0 or args.shift_max >= 0\n"
    "    ctx_F = max(args.flank_ctx, 0)\n"
    "    ctx_S = args.shift_max if args.shift_max >= 0 else 10\n"
    "    if wide:\n"
    '        if os.environ.get("AL_FLANK_SHIFT") == "1":\n'
    '            raise SystemExit("--flank_ctx/--shift_max replace AL_FLANK_SHIFT; unset it")\n'
    '        os.environ["AL_SHIFT_MAX"] = "0"  # the WideCrop wrapper does the shifting, on real bases\n'
    "\n"
    "    def _enc(seqs):\n"
    "        return wide_onehot(seqs, ctx_F + ctx_S) if wide else one_hot(seqs, args.seq_len)\n"
    "\n"
    "    out = Path(args.out_dir)\n    out.mkdir(parents=True, exist_ok=True)\n    k, n = ",
)
sub(
    "seq_len=args.seq_len, max_params=args.max_params",
    "seq_len=args.seq_len + 2 * ctx_F, max_params=args.max_params",
)
sub(
    '            tc = dict(e["train_config"])\n',
    "            if wide:\n                net = wrap_ctx(net, ctx_F, ctx_S)\n"
    '            tc = dict(e["train_config"])\n',
)
sub(
    "    if any(_new_hp.values()):\n        _ov.update(_new_hp)",
    '    _new_harness = {k_.lower(): os.environ[k_] for k_ in ("AL_EMA", "AL_BN_RECALC") if os.environ.get(k_)}\n'
    "    if wide:\n"
    '        _new_harness.update({"flank_ctx": ctx_F, "shift_max": ctx_S})\n'
    "    _ov.update(_new_harness)  # only when used, like the line below\n"
    "    if any(_new_hp.values()):\n        _ov.update(_new_hp)",
)
sub(
    "or any(_new_hp.values()):\n        _ovp.write_text",
    "or any(_new_hp.values()) or _new_harness:\n        _ovp.write_text",
)
sub(
    '            if args.val_cache:\n                meta["val_cache"]',
    '            meta["harness"] = {k_: os.environ.get(k_) for k_ in ("AL_EMA", "AL_BN_RECALC", "AL_FLANK_SHIFT", "AL_SHIFT_MAX", "AL_PATIENCE") if os.environ.get(k_) is not None}\n'
    "            if wide:\n"
    '                meta["harness"].update({"flank_ctx": ctx_F, "shift_max": ctx_S})\n'
    '            meta.update({k_: info[k_] for k_ in ("ema_decay", "val_mse_before_bn_recalc", "bn_layers_recalculated") if k_ in info})\n'
    '            if args.val_cache:\n                meta["val_cache"]',
)
ast.parse(s)
shutil.copy(p, p + ".bak_ctx")
open(p, "w").write(s)
print("patched", p)
