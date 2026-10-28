"""Constant plasmid sequence on each side of the 200-bp MPRA insert (the context the oracle sees), trimmed to what a shift of up to +-{N} bp can pull in.

Source: data/k562_full.py MPRA_UPSTREAM (last 40 bases) and MPRA_DOWNSTREAM (first 40 bases), the same constants experiments/train_oracle_s2_v2.py uses. Used by the student harness when
AL_FLANK_SHIFT=1, so that a shifted training window takes real flank bases in instead of zeros.
"""

N = 40
UPSTREAM_TAIL = "GGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG"
DOWNSTREAM_HEAD = "CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCT"
