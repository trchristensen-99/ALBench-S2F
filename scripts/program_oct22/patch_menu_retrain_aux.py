"""menu_retrain.py: auxiliary-target (multi-task) training (new flags, default off).

    --aux_labels PATH   npz with `sequences` and `oracle_labels` (e.g. the HepG2 oracle labels of the same sequences, outputs/grid_pools/k562_hepg2/hepg2/...); matched to the training sequences by string
    --aux_weight W      weight of the auxiliary squared error relative to the primary one (default 0.5)

The architecture keeps its single scalar output for every evaluation and prediction; in training an extra linear head on the input of its final readout predicts the auxiliary label (albench.arch_search.multitask).

    python scripts/program_oct22/patch_menu_retrain_aux.py [scratchpad/menu_retrain.py]      (idempotent; backup .bak_aux)
"""

import ast
import shutil
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/menu_retrain.py"
s = open(p).read()
if "--aux_labels" in s:
    print("already patched")
    sys.exit(0)


def sub(old, new):
    global s
    assert old in s, f"anchor not found: {old[:70]}"
    s = s.replace(old, new, 1)


sub(
    '    ap.add_argument("--save_models"',
    '    ap.add_argument("--aux_labels", default="", help="npz (sequences, oracle_labels) of an auxiliary target for the same sequences (e.g. HepG2); trained as a second output, never used for evaluation")\n'
    '    ap.add_argument("--aux_weight", type=float, default=0.5, help="weight of the auxiliary squared error")\n'
    '    ap.add_argument("--save_models"',
)
sub(
    "    _enc(tr_s), np.asarray(tr_y, np.float32)",
    "    _enc(tr_s), np.asarray(tr_y, np.float32)",
) if False else None
sub(
    "    Xva, yva = _enc(va_s), np.asarray(va_y, np.float32)\n",
    "    Xva, yva = _enc(va_s), np.asarray(va_y, np.float32)\n"
    "    if args.aux_labels:\n"
    "        _az = np.load(args.aux_labels, allow_pickle=True)\n"
    '        _amap = dict(zip([str(x) for x in _az["sequences"]], np.asarray(_az["oracle_labels"], np.float32)))\n'
    "        _miss = [x for x in tr_s if x not in _amap]\n"
    "        if _miss:\n"
    '            raise SystemExit(f"{len(_miss)} training sequences have no auxiliary label in {args.aux_labels}")\n'
    "        _ay = np.asarray([_amap[x] for x in tr_s], np.float32)\n"
    "        ytr = np.stack([ytr, np.sqrt(args.aux_weight) * _ay], axis=1)\n"
    '        print(f"  aux labels: {args.aux_labels} weight {args.aux_weight} corr(primary, aux) = {np.corrcoef(ytr[:, 0], _ay)[0, 1]:.3f}", flush=True)\n',
)
sub(
    "            if wide:\n                net = wrap_ctx(",
    "            if args.aux_labels:\n"
    "                from albench.arch_search.multitask import wrap as mt_wrap\n\n"
    "                try:\n"
    "                    net = mt_wrap(net, 1, args.aux_weight)\n"
    "                except ValueError as ex:\n"
    "                    failed += 1\n"
    '                    print(f"    {mid} AUX-FAIL {ex}", flush=True)\n'
    "                    continue\n"
    "            if wide:\n                net = wrap_ctx(",
)
sub(
    "    _ov.update(_new_harness)  # only when used, like the line below",
    '    if args.aux_labels:\n        _new_harness.update({"aux_labels": os.path.basename(args.aux_labels), "aux_weight": args.aux_weight})\n    _ov.update(_new_harness)  # only when used, like the line below',
)
sub(
    '            meta.update({k_: info[k_] for k_ in ("ema_decay"',
    '            if args.aux_labels:\n                meta["harness"].update({"aux_labels": args.aux_labels, "aux_weight": args.aux_weight})\n            meta.update({k_: info[k_] for k_ in ("ema_decay"',
)
ast.parse(s)
shutil.copy(p, p + ".bak_aux")
open(p, "w").write(s)
print("patched", p)
