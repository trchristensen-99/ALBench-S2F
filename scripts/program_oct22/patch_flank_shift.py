"""Make the student harness optionally shift with REAL plasmid flank instead of zeros (env AL_FLANK_SHIFT=1; default off, so every existing cell behaves exactly as before).

Patches albench/arch_search/arch_autoresearch.py in place (a backup is written next to it); needs albench/arch_search/plasmid_flank.py. Idempotent.

    python scripts/program_oct22/patch_flank_shift.py [path/to/arch_autoresearch.py]
"""

import ast
import shutil
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "albench/arch_search/arch_autoresearch.py"
s = open(p).read()
if "AL_FLANK_SHIFT" in s:
    print("already patched")
    sys.exit(0)
a = """    y = torch.zeros_like(x)
    if s > 0:
        y[:, :, s:] = x[:, :, :-s]
    else:
        y[:, :, :s] = x[:, :, -s:]
    return y
"""
assert a in s, "shift body not found"
b = '''    y = torch.zeros_like(x)
    flank = os.environ.get("AL_FLANK_SHIFT") == "1"  # real plasmid bases instead of zeros where the window moves past the insert
    if s > 0:
        y[:, :, s:] = x[:, :, :-s]
        if flank:
            y[:, :, :s] = _flank(x.device, x.dtype)[0][:, -s:]
    else:
        y[:, :, :s] = x[:, :, -s:]
        if flank:
            y[:, :, s:] = _flank(x.device, x.dtype)[1][:, : -s]
    return y


_FLANK_CACHE: dict = {}


def _flank(device, dtype):
    """(upstream tail, downstream head) one-hot (4, N) tensors in ACGT order."""
    key = (str(device), dtype)
    if key not in _FLANK_CACHE:
        import torch

        from albench.arch_search.plasmid_flank import DOWNSTREAM_HEAD, UPSTREAM_TAIL

        def oh(seq):
            t = torch.zeros(4, len(seq), device=device, dtype=dtype)
            for i, c in enumerate(seq):
                t["ACGT".index(c), i] = 1.0
            return t

        _FLANK_CACHE[key] = (oh(UPSTREAM_TAIL), oh(DOWNSTREAM_HEAD))
    return _FLANK_CACHE[key]
'''
s = s.replace(a, b, 1)
ast.parse(s)
shutil.copy(p, p + ".bak_flank")
open(p, "w").write(s)
print("patched", p)
