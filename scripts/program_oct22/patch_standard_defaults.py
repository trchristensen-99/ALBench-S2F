"""Make the 20-base context standard the default of the retraining script and the AutoResearch launcher, add the minimal-guidance prompt and the harness-assigned recipe (all other behaviour unchanged).

scratchpad/menu_retrain.py        no --flank_ctx / --shift_max given: a FRESH output directory trains with the standard (albench.arch_search.harness_standard: 20 real flank bases each side, shift +-10);
                                  a directory that already holds models trained without context keeps its old behaviour (requeued jobs resume as they were started); --no_context forces the bare 200-bp insert
scratchpad/arch_autoresearch.py   the same rule; the input-length note defaults to the fixed-length version (position-specific designs are allowed); --length_agnostic restores the hedged note;
                                  --guidance minimal (facts and negative results only, no positive prescriptions); AL_RECIPE_MODE=harness makes sample_train draw each model's recipe from a safe set

python scripts/program_oct22/patch_standard_defaults.py scratchpad/menu_retrain.py scratchpad/arch_autoresearch.py      (idempotent; backup .bak_std)
"""

import ast
import shutil
import sys

MINIMAL = """
MINIMAL_GUIDANCE = (
    "WHAT EARLIER SEARCHES ON THIS TASK SHOWED (facts, not instructions):\\n"
    "- The ensemble of your models is what is scored, so a model whose predictions are decorrelated from the best ones is worth more than a slightly better copy of a design that is already trained "
    "(a 0.002 gain on a near-duplicate adds nothing to the ensemble).\\n"
    "- Convolutional models, most often with squeeze-excitation blocks, have been the strongest family: validation Pearson 0.77 to 0.82 at this size. Pure transformer, MLP-mixer, spectral or wavelet "
    "and hand-built motif-grammar models have scored clearly lower (0.43 to 0.74), as have models that discard position entirely.\\n"
    "- Reverse-complement and shift augmentation are applied for you and the input includes real plasmid context (see below); do not re-implement them.\\n"
    "- Retraining the same design with another seed moves its validation Pearson by about 0.005 to 0.01: do not read a smaller gap between two designs as real."
)

"""

SAFE_RECIPE = """    if os.environ.get("AL_RECIPE_MODE") == "harness":  # the harness assigns the recipe: Muon with a small spread around the settings that every recipe study found equivalent
        r = random.Random()
        return {
            "optimizer": "muon",
            "lr": r.choice([1e-3, 2e-3, 2e-3, 4e-3]),
            "weight_decay": r.choice([1e-4, 1e-4, 3e-4, 1e-3]),
            "batch_size": r.choice([128, 256, 256, 512]),
            "epochs": 80,
            "lr_schedule": r.choice(["onecycle", "onecycle", "cosine"]),
        }
"""


def patch_launcher(p):
    s = open(p).read()
    if "MINIMAL_GUIDANCE" in s:
        print("already patched", p)
        return

    def sub(old, new):
        nonlocal s
        assert old in s, f"anchor not found: {old[:70]}"
        s = s.replace(old, new, 1)

    sub(
        '    """Fill missing training HPs with randomly-varied defaults."""\n',
        '    """Fill missing training HPs with randomly-varied defaults."""\n' + SAFE_RECIPE,
    )
    sub(
        "\ndef _round_note(args, rd, cap_min, lessons=()):",
        "\n" + MINIMAL + "\ndef _round_note(args, rd, cap_min, lessons=()):",
    )
    sub(
        '    if args.guidance != "known_good":\n        return',
        '    if args.guidance not in ("known_good", "minimal"):\n        return',
    )
    sub(
        "    parts = [KNOWN_GOOD] + ([ctx] if ctx else [])\n",
        '    parts = [MINIMAL_GUIDANCE if args.guidance == "minimal" else KNOWN_GOOD] + ([ctx] if ctx else [])\n',
    )
    sub('choices=["none", "known_good"]', 'choices=["none", "known_good", "minimal"]')
    sub(
        '    ap.add_argument("--shift_max"',
        '    ap.add_argument("--no_context", action="store_true", help="train on the bare 200-bp insert (an experiment that deliberately differs from the standard: 20 real flank bases each side, shift +-10)")\n'
        '    ap.add_argument("--length_agnostic", action="store_true", help="with context: ask for length-agnostic architectures instead of telling the proposer the length is fixed")\n'
        '    ap.add_argument("--shift_max"',
    )
    sub(
        "    args.wide = args.flank_ctx >= 0 or args.shift_max >= 0\n",
        "    from albench.arch_search.harness_standard import FLANK_CTX as STD_F, SHIFT_MAX as STD_S\n\n"
        "    if args.no_context:\n"
        "        args.flank_ctx = args.shift_max = -1\n"
        "    elif args.flank_ctx < 0 and args.shift_max < 0:  # the standard, unless this directory already holds models trained without context\n"
        '        _legacy = bool(list(Path(args.out_dir).glob("r*_arch_*_meta.json"))) and not (Path(args.out_dir) / "flank_ctx.json").exists()\n'
        "        if not _legacy:\n"
        "            args.flank_ctx, args.shift_max = STD_F, STD_S\n"
        "    args.wide = args.flank_ctx >= 0 or args.shift_max >= 0\n"
        "    if args.wide and not args.length_agnostic:\n"
        "        args.fixed_length = True\n",
    )
    ast.parse(s)
    shutil.copy(p, p + ".bak_std")
    open(p, "w").write(s)
    print("patched", p)


def patch_retrain(p):
    s = open(p).read()
    if "--no_context" in s:
        print("already patched", p)
        return

    def sub(old, new):
        nonlocal s
        assert old in s, f"anchor not found: {old[:70]}"
        s = s.replace(old, new, 1)

    sub(
        '    ap.add_argument("--save_models"',
        '    ap.add_argument("--no_context", action="store_true", help="train on the bare 200-bp insert (an experiment that deliberately differs from the standard: 20 real flank bases each side, shift +-10)")\n'
        '    ap.add_argument("--save_models"',
    )
    sub(
        "    wide = args.flank_ctx >= 0 or args.shift_max >= 0\n",
        "    from albench.arch_search.harness_standard import FLANK_CTX as STD_F, SHIFT_MAX as STD_S\n\n"
        "    if args.no_context:\n"
        "        args.flank_ctx = args.shift_max = -1\n"
        "    elif args.flank_ctx < 0 and args.shift_max < 0:  # the standard, unless this directory already holds models trained without context\n"
        "        _od = Path(args.out_dir)\n"
        '        _legacy = any(_od.glob("r*_menu_*_meta.json")) or (_od / "train_override.json").exists() and "flank_ctx" not in (_od / "train_override.json").read_text()\n'
        "        if not _legacy:\n"
        "            args.flank_ctx, args.shift_max = STD_F, STD_S\n"
        "    wide = args.flank_ctx >= 0 or args.shift_max >= 0\n",
    )
    ast.parse(s)
    shutil.copy(p, p + ".bak_std")
    open(p, "w").write(s)
    print("patched", p)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        (patch_launcher if p.endswith("arch_autoresearch.py") else patch_retrain)(p)
