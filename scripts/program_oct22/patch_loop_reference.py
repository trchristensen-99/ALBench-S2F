"""AutoResearch loop: show the proposer the published MPRA-LegNet as the baseline, allow fixed-length designs, and give it the baseline's measured validation r (all opt-in; defaults reproduce the current prompt).

--reference_legnet   add the published MPRA-LegNet, written to the contract (albench/arch_search/mpra_legnet_ref.py), to the guidance block, with a line saying it is a bar and a starting point, not a template
--baseline_file F    text appended to the guidance block (for example the baseline's validation r on this training set)
--fixed_length       with --flank_ctx: tell the proposer that the input length L is fixed by the harness for every model in the study, so position-specific designs sized for L are allowed (the default note asks for
                     length-agnostic code, which is only needed while the context size is still being chosen)

python scripts/program_oct22/patch_loop_reference.py scratchpad/arch_autoresearch.py      (idempotent; backup .bak_ref)
"""

import ast
import shutil
import sys

FIXED = """
FLANK_NOTE_FIXED = (
    "INPUT CONTEXT: x has length L = {L} for every model in this study, fixed by the harness and never changed: the 200-bp variable insert in the centre plus {F} real bases of the constant plasmid sequence on EACH side. "
    "The assay and the oracle that labels the sequences see the insert inside this plasmid, so a motif that spans the insert boundary is part of the label; the flank bases are identical for every sequence. "
    "During training the window is shifted by up to +-{S} bases over real sequence; evaluation uses the centred window. Because L is fixed you may use position-specific designs sized for L "
    "(position-resolved readouts, fixed windows, learned positional parameters); you are free to explore any architecture."
)

REFERENCE_LEGNET_NOTE = (
    "REFERENCE: THE PUBLISHED MPRA-LegNet (the standard baseline for this task), written to this contract. A model of this kind trained by this harness on this training set reaches the validation r given with it. "
    "It is a starting point and a bar, not a template: the goal is a set of strong architectures that differ from each other and from it, so propose different designs as well as improvements of it.\\n```python\\n"
)

"""

p = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/arch_autoresearch.py"
s = open(p).read()
if "--reference_legnet" in s:
    print("already patched")
    sys.exit(0)


def sub(old, new):
    global s
    assert old in s, f"anchor not found: {old[:70]}"
    s = s.replace(old, new, 1)


sub(
    "\ndef _round_note(args, rd, cap_min, lessons=()):",
    FIXED + "\ndef _round_note(args, rd, cap_min, lessons=()):",
)
sub(
    'base += "\\n\\n" + FLANK_NOTE.format(L=200 + 2 * args.ctx_F, F=args.ctx_F, S=args.ctx_S)',
    'base += "\\n\\n" + (FLANK_NOTE_FIXED if getattr(args, "fixed_length", False) else FLANK_NOTE).format(L=200 + 2 * args.ctx_F, F=args.ctx_F, S=args.ctx_S)',
)
sub(
    "    parts = [KNOWN_GOOD] + ([ctx] if ctx else [])\n",
    "    parts = [KNOWN_GOOD] + ([ctx] if ctx else [])\n"
    '    if getattr(args, "reference_legnet", False):\n'
    "        from albench.arch_search.mpra_legnet_ref import SOURCE as LEGNET_SRC\n\n"
    '        bl = open(args.baseline_file).read().strip() + "\\n\\n" if getattr(args, "baseline_file", "") else ""\n'
    '        parts.append(bl + REFERENCE_LEGNET_NOTE + LEGNET_SRC + "```")\n'
    '    elif getattr(args, "baseline_file", ""):\n'
    "        parts.append(open(args.baseline_file).read().strip())\n",
)
sub(
    '    ap.add_argument("--shift_max"',
    '    ap.add_argument("--reference_legnet", action="store_true", help="show the proposer the published MPRA-LegNet (as a bar and a starting point)")\n'
    '    ap.add_argument("--baseline_file", default="", help="text file appended to the guidance block (e.g. the baseline\'s validation r on this training set)")\n'
    '    ap.add_argument("--fixed_length", action="store_true", help="with --flank_ctx: the input length is fixed by the harness, position-specific designs are allowed")\n'
    '    ap.add_argument("--shift_max"',
)
ast.parse(s)
shutil.copy(p, p + ".bak_ref")
open(p, "w").write(s)
print("patched", p)
