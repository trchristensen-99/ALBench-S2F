"""AutoResearch launcher: an OPEN prompt. The collaborator removed LegNet from his prompt and the loop stopped converging to LegNet variants (new best singles with grouped convolutions, ResNeXt blocks, second-order
convolutions, pairwise maps). Our known-good and minimal guidance both tell the proposer that convolutional models with squeeze-excitation are the strongest family, and the reference note shows LegNet's code in every
round, so our loop is anchored the same way (99.5% of its candidates contained squeeze-excitation). This patch adds, without changing any existing arm:

  --guidance open            the minimal facts (ensemble is what is scored; augmentation and context are applied; seed noise) WITHOUT the sentence naming a best architecture family
  --reference_from_round N   the reference LegNet code and the in-run baseline are shown only from round N on (default 0 = every round, as before); the baseline is still trained at the start, so the
                             reference check of the 'schedule_fb' manager works while exploration rounds see no LegNet
  --target_val R             optional sentence: models with validation Pearson r of about R have been trained on this task at this training-set size (a known-achievable target); default off

    python scripts/program_oct22/patch_loop_open.py scratchpad/arch_autoresearch.py      (idempotent; backup .bak_open)
"""

import ast
import re
import shutil
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/arch_autoresearch.py"
s = open(p).read()
if "OPEN_GUIDANCE" in s:
    print("already patched")
    sys.exit(0)


def must(cond, what):
    if not cond:
        raise SystemExit(f"anchor missing: {what}")


# 1. the open guidance text, before _round_note
OPEN = """OPEN_GUIDANCE = (
    "WHAT EARLIER SEARCHES ON THIS TASK SHOWED (facts, not instructions):\\n"
    "- The ensemble of your models is what is scored, so a model whose predictions are decorrelated from the best ones is worth more than a slightly better copy of a design that is already trained "
    "(a 0.002 gain on a near-duplicate adds nothing to the ensemble).\\n"
    "- No architecture family is known to be best on this task; unusual mechanisms have not been ruled out, so look widely before refining.\\n"
    "- Reverse-complement and shift augmentation are applied for you and the input includes real plasmid context (see below); do not re-implement them.\\n"
    "- Retraining the same design with another seed moves its validation Pearson by about 0.005 to 0.01: do not read a smaller gap between two designs as real."
)


"""
i = s.index("def _round_note(")
s = s[:i] + OPEN + s[i:]

# 2. guidance selection in _round_note
a = 'if args.guidance not in ("known_good", "minimal"):'
must(a in s, "guidance check")
s = s.replace(a, 'if args.guidance not in ("known_good", "minimal", "open"):', 1)
b = 'parts = [MINIMAL_GUIDANCE if args.guidance == "minimal" else KNOWN_GOOD] + ([ctx] if ctx else [])'
must(b in s, "guidance parts")
s = s.replace(
    b,
    'parts = [{"minimal": MINIMAL_GUIDANCE, "open": OPEN_GUIDANCE}.get(args.guidance, KNOWN_GOOD)] + ([ctx] if ctx else [])\n'
    '    if getattr(args, "target_val", 0):\n'
    '        parts.append(f"TARGET: models with validation Pearson r of about {args.target_val:.3f} have been trained on this task at this training-set size.")',
    1,
)

# 3. reference gating by round
c = '    if getattr(args, "reference_legnet", False):\n        from albench.arch_search.mpra_legnet_ref import SOURCE as LEGNET_SRC'
must(c in s, "reference block")
s = s.replace(
    c,
    '    if getattr(args, "reference_legnet", False) and rd >= getattr(args, "reference_from_round", 0):\n        from albench.arch_search.mpra_legnet_ref import SOURCE as LEGNET_SRC',
    1,
)
d = '    elif getattr(args, "baseline_file", ""):\n        parts.append(open(args.baseline_file).read().strip())'
must(d in s, "baseline-only branch")
s = s.replace(
    d,
    '    elif getattr(args, "baseline_file", "") and not getattr(args, "reference_legnet", False):\n        parts.append(open(args.baseline_file).read().strip())',
    1,
)

# 4. arguments
m = re.search(r'^    ap\.add_argument\("--reference_legnet"', s, re.M)
must(m is not None, "--reference_legnet argument")
new_args = (
    '    ap.add_argument("--reference_from_round", type=int, default=0, help="show the reference LegNet code and baseline only from this round on (0 = every round)")\n'
    '    ap.add_argument("--target_val", type=float, default=0.0, help="optional known-achievable validation r stated in the prompt (0 = off)")\n'
)
s = s[: m.start()] + new_args + s[m.start() :]
g = re.search(r'^    ap\.add_argument\("--guidance"[^\n]*\n', s, re.M)
must(g is not None, "--guidance argument")
line = g.group(0)
print("guidance argument line:", line.strip()[:200])
if "choices" in line and '"open"' not in line:
    new = line.replace('"minimal"', '"minimal", "open"', 1)
    must(new != line, "could not extend the guidance choices")
    s = s.replace(line, new, 1)

ast.parse(s)
shutil.copy(p, p + ".bak_open")
open(p, "w").write(s)
print("patched", p)
