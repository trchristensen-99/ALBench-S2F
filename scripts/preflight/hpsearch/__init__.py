"""HP search infrastructure: Ray Tune + AutoResearch subagents.

Shared modules:
    hp_space.py  — HP search space + per-arch translation (width/depth → arch HPs).
    trainable.py — Ray Tune Trainable wrapping run_single.train().
    raytune_search.py — driver for Random / Optuna / HyperOpt / BOHB / PBT.
    autoresearch.py   — spawns Claude subagents to iterate on HPs.
"""
