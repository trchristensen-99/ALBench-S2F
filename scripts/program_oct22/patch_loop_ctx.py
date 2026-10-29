"""AutoResearch loop: give the students the real plasmid flank as input (opt-in; default unchanged).

    --flank_ctx F   the student sees 200 + 2F bases: the 200-bp insert plus F real constant plasmid bases on each side (what the assay and the oracle see); the training window shifts by +-S over real bases
    --shift_max S   the +-S shift (default 10 when --flank_ctx is given)

Patches the launcher (scratchpad/arch_autoresearch.py) and albench/arch_search/agent_loop.py. The proposer and the coder see the input length L = 200 + 2F in the contract and a note that explains the
context and asks for length-agnostic architectures; every architecture is checked at L; the harness wraps each model in flank_context.WideCrop for training and evaluation; the run directory is stamped
(flank_ctx.json) so a run cannot be resumed with a different input; every model's meta records the harness switches.

    python scripts/program_oct22/patch_loop_ctx.py scratchpad/arch_autoresearch.py albench/arch_search/agent_loop.py      (idempotent; backups .bak_loopctx)
"""

import ast
import re
import shutil
import sys

LAUNCHER_NOTE = """
FLANK_NOTE = (
    "INPUT CONTEXT: x has length L = {L}: the 200-bp variable insert in the centre plus {F} real bases of the constant plasmid sequence on EACH side. "
    "The assay and the oracle that labels the sequences see the insert inside this plasmid, so a motif that spans the insert boundary is part of the label; the flank bases are identical for every "
    "sequence. During training the window is shifted by up to +-{S} bases over real sequence; evaluation uses the centred window. Write the architecture for any L divisible by 8: do not hard-code 200 "
    "or {L} (use global or adaptive pooling and operations that do not depend on the length). A model that cannot run at L = {L} is discarded."
)

"""


def patch_launcher(p):
    s = open(p).read()
    if "--flank_ctx" in s:
        print("already patched", p)
        return

    def sub(old, new):
        nonlocal s
        assert old in s, f"anchor not found: {old[:60]}"
        s = s.replace(old, new, 1)

    s, n = re.subn(
        r"one_hot\(([A-Za-z_\[\]0-9]+), args\.seq_len\)", lambda m: f"_enc({m.group(1)})", s
    )
    assert n == 4, n
    sub(
        "\n\ndef _round_note(args, rd, cap_min, lessons=()):",
        "\n" + LAUNCHER_NOTE + "\ndef _round_note(args, rd, cap_min, lessons=()):",
    )
    sub(
        "    base = (FULL_NOTE if args.full_control else TRAIN_NOTE).format(max_min=cap_min, max_params=args.max_params)\n",
        "    base = (FULL_NOTE if args.full_control else TRAIN_NOTE).format(max_min=cap_min, max_params=args.max_params)\n"
        '    if getattr(args, "wide", False):\n'
        '        base += "\\n\\n" + FLANK_NOTE.format(L=200 + 2 * args.ctx_F, F=args.ctx_F, S=args.ctx_S)\n',
    )
    sub(
        "    args = ap.parse_args()\n\n    from scaling_hp_search import (",
        '    ap.add_argument("--flank_ctx", type=int, default=-1, help="F real plasmid bases of input context on each side of the 200-bp insert (shift over real bases); -1 = off")\n'
        '    ap.add_argument("--shift_max", type=int, default=-1, help="+-shift of the training window with --flank_ctx (default 10); -1 = off")\n'
        "    args = ap.parse_args()\n\n    from scaling_hp_search import (",
    )
    sub(
        "    te_s, te_oracle, te_true = load_chr_test_genomic()\n    tests = load_all_test_sets()\n",
        "    te_s, te_oracle, te_true = load_chr_test_genomic()\n    tests = load_all_test_sets()\n"
        "    args.wide = args.flank_ctx >= 0 or args.shift_max >= 0\n"
        "    args.ctx_F, args.ctx_S = max(args.flank_ctx, 0), (args.shift_max if args.shift_max >= 0 else 10)\n"
        '    _cp = Path(args.out_dir) / "flank_ctx.json"\n'
        "    if args.wide:\n"
        "        from albench.arch_search.flank_context import wide_onehot, wrap as wrap_ctx\n\n"
        '        if os.environ.get("AL_FLANK_SHIFT") == "1":\n'
        '            raise SystemExit("--flank_ctx/--shift_max replace AL_FLANK_SHIFT; unset it")\n'
        '        if not _cp.exists() and list(Path(args.out_dir).glob("r*_arch_*_meta.json")):\n'
        '            raise SystemExit(f"{args.out_dir} already holds models trained without flank context; use a fresh --out_dir")\n'
        '        if _cp.exists() and json.load(open(_cp)) != {"F": args.ctx_F, "S": args.ctx_S}:\n'
        '            raise SystemExit(f"{args.out_dir} was started with {open(_cp).read()}; pass the same flags")\n'
        '        _cp.write_text(json.dumps({"F": args.ctx_F, "S": args.ctx_S}))\n'
        '        os.environ["AL_SHIFT_MAX"] = "0"  # WideCrop shifts over real bases\n'
        "        args.seq_len = 200 + 2 * args.ctx_F  # the contract the proposer and the coder see\n"
        "    elif _cp.exists():\n"
        '        raise SystemExit(f"{args.out_dir} was started with flank context; pass the same flags or use a fresh --out_dir")\n\n'
        "    def _enc(seqs):\n"
        "        return wide_onehot(seqs, args.ctx_F + args.ctx_S) if args.wide else one_hot(seqs, args.seq_len)\n\n"
        "    def _wrap(m):\n"
        "        return wrap_ctx(m, args.ctx_F, args.ctx_S) if args.wide else m\n\n"
        '    args.harness = {k: os.environ[k] for k in ("AL_EMA", "AL_BN_RECALC", "AL_FLANK_SHIFT", "AL_SHIFT_MAX", "AL_PATIENCE", "AL_PRECISION") if os.environ.get(k)}\n'
        "    if args.wide:\n"
        '        args.harness.update({"flank_ctx": args.ctx_F, "shift_max": args.ctx_S})\n\n',
    )
    sub(
        "            pear=pear, atomic_savez=_atomic_savez,",
        "            wrap_model=_wrap, harness=args.harness,\n            pear=pear, atomic_savez=_atomic_savez,",
    )
    sub(
        "model, info, used = train_one(model, Xtr, ytr, Xva, yva, tc, max_sec=cap_min * 60)",
        "model, info, used = train_one(_wrap(model), Xtr, ytr, Xva, yva, tc, max_sec=cap_min * 60)",
    )
    sub(
        '"train_time_sec": time.time() - t0, "gpu": gpu, "source": p["source"]}',
        '"train_time_sec": time.time() - t0, "gpu": gpu, "source": p["source"], "harness": args.harness}',
    )
    ast.parse(s)
    shutil.copy(p, p + ".bak_loopctx")
    open(p, "w").write(s)
    print("patched", p)


def patch_agent(p):
    s = open(p).read()
    if "wrap_model" in s:
        print("already patched", p)
        return

    def sub(old, new):
        nonlocal s
        assert old in s, f"anchor not found: {old[:60]}"
        s = s.replace(old, new, 1)

    sub(
        '    tc = h.sample_train(item["train_spec"])\n    torch.manual_seed(1000 + ctx.midx)\n',
        '    if getattr(h, "wrap_model", None) is not None:  # flank-context students: WideCrop around the checked architecture\n        model = h.wrap_model(model)\n'
        '    tc = h.sample_train(item["train_spec"])\n    torch.manual_seed(1000 + ctx.midx)\n',
    )
    sub(
        '        "source": item["source"],\n        # agent-loop extras',
        '        "source": item["source"],\n        "harness": getattr(h, "harness", {}),\n        # agent-loop extras',
    )
    ast.parse(s)
    shutil.copy(p, p + ".bak_loopctx")
    open(p, "w").write(s)
    print("patched", p)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        (patch_agent if p.endswith("agent_loop.py") else patch_launcher)(p)
