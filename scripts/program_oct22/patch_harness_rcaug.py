"""Student harness: reverse-complement augmentation the way the oracle does it (opt-in, env AL_RC_MODE=aug; default unchanged).

Default (consistency): every step the model is run on the sequence and on its reverse complement and the loss is the mean of the two squared errors. With AL_RC_MODE=aug each sample is flipped to its reverse
complement with probability 0.5 and the model runs once (half the compute per step), as in the oracle's training (train_oracle_s2_v2.py). Validation and test predictions still average both strands.

    python scripts/program_oct22/patch_harness_rcaug.py [path ...]      (idempotent; backup .bak_rcaug)
"""

import ast
import shutil
import sys

OLD = """                out = model(xb).squeeze()
                # RC consistency: average fwd + reverse-complement prediction in the loss
                out_rc = model(rc(xb)).squeeze()
"""
NEW = """                if os.environ.get("AL_RC_MODE", "consistency") == "aug":
                    # oracle-style augmentation: flip half of the samples, one forward pass
                    _flip = torch.rand(xb.shape[0], device=xb.device) < 0.5
                    out = model(torch.where(_flip[:, None, None], rc(xb), xb)).squeeze()
                    out_rc = out
                else:
                    out = model(xb).squeeze()
                    # RC consistency: average fwd + reverse-complement prediction in the loss
                    out_rc = model(rc(xb)).squeeze()
"""
for p in sys.argv[1:] or ["albench/arch_search/arch_autoresearch.py"]:
    s = open(p).read()
    if "AL_RC_MODE" in s:
        print("already patched", p)
        continue
    assert OLD in s, f"anchor not found in {p}"
    s = s.replace(OLD, NEW, 1)
    ast.parse(s)
    shutil.copy(p, p + ".bak_rcaug")
    open(p, "w").write(s)
    print("patched", p)
