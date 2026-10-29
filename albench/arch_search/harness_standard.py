"""The student harness standard: what every student sees and how it is augmented, fixed in one place.

Students are shown the 200-bp insert inside its real plasmid context: ``FLANK_CTX`` = 20 real bases on each side (input length ``INPUT_LEN`` = 240), as the oracle that labels every sequence reads it. Training slides the
window by up to +-``SHIFT_MAX`` = 10 bases over the real plasmid sequence; evaluation uses the centred window; reverse-complement predictions are averaged. Measured on 30k genomic training sequences: 20 bases per side
raise a single model's test r by about +0.07 over the bare insert, 12 to 20 bases give the whole gain, 60 to 200 add nothing, and a shift of +-10 beats none or +-30 (see the Evidence Ledger).

This is the default everywhere (library, retraining scripts, AutoResearch loop, prompts). It is a standard, not a hyperparameter: proposals cannot change it, and it changes only when a sweep shows something better
(the context and shift sweeps ``sw38_`` are running); then this file is the one place to edit. An experiment that deliberately differs passes explicit flags (``--flank_ctx``, ``--shift_max``) or ``--no_context``.
"""

FLANK_CTX = 20
SHIFT_MAX = 10
INSERT_LEN = 200
INPUT_LEN = INSERT_LEN + 2 * FLANK_CTX  # 240, a multiple of 8
