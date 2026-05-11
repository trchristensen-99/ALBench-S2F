"""HP search space + per-architecture HP translation.

Shared by raytune_search.py and autoresearch.py. Defines:
- The 6-dim search space (width, depth, lr, batch_size, weight_decay, dropout)
- Per-arch ranges (LegNet, DREAM-RNN, DREAM-ATTN have different valid ranges)
- to_run_single_overrides(): translates abstract HP dict → run_single.py --hp args
"""

from __future__ import annotations

from typing import Any

# Boundaries chosen 2026-05-11.
# Each arch has independent (width, depth, dropout) ranges based on what is
# computationally feasible and architecturally meaningful. LR/BS/WD use the
# same wide range across archs — the optimizer can carve out per-arch regions.
SHARED_RANGES: dict[str, Any] = {
    "lr": ("loguniform", 1e-5, 1e-2),
    "batch_size": ("choice", [64, 128, 256, 512, 1024]),
    "weight_decay": ("loguniform", 1e-6, 1e-1),
    "dropout": ("choice", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
}

ARCH_RANGES: dict[str, dict[str, Any]] = {
    "legnet": {
        # block_sizes = [width] * depth
        "width": ("choice", [128, 256, 512, 1024, 2000]),
        "depth": ("choice", [2, 3, 4, 5, 6, 7]),
    },
    "dream_rnn": {
        # hidden_dim = width (LSTM hidden per direction)
        # num_lstm_layers = depth
        # cnn_filters scales with width (clamped to 320 max)
        "width": ("choice", [64, 128, 256, 512]),
        "depth": ("choice", [1, 2, 3, 4]),
    },
    "dream_attn": {
        # embedding_dim = width, num_blocks = depth
        "width": ("choice", [128, 256, 512]),
        "depth": ("choice", [2, 4, 6, 8]),
    },
}


def get_full_space(arch: str) -> dict[str, Any]:
    """Return the full HP search space for an arch as a {name: spec} dict.

    Spec is a tuple: (type, *args). Types: "choice", "loguniform", "uniform".
    """
    space = dict(SHARED_RANGES)
    space.update(ARCH_RANGES[arch])
    return space


def to_run_single_overrides(arch: str, hp: dict[str, Any]) -> list[str]:
    """Translate an abstract HP dict into run_single.py --hp k=v strings.

    The abstract HP dict has keys: lr, batch_size, weight_decay, dropout,
    width, depth. This translates to arch-specific HP names that
    run_single.py's ARCH_PRIORS expects.
    """
    lr = hp["lr"]
    bs = hp["batch_size"]
    wd = hp["weight_decay"]
    dr = hp["dropout"]
    w = hp["width"]
    d = hp["depth"]

    overrides = [f"lr={lr}", f"batch_size={bs}", f"weight_decay={wd}"]

    if arch == "legnet":
        block_sizes = [int(w)] * int(d)
        overrides.append(f"block_sizes={block_sizes}")
        overrides.append(f"dropout={dr}")
    elif arch == "dream_rnn":
        overrides.append(f"hidden_dim={int(w)}")
        # cnn_filters scales with width (clamped to keep params reasonable)
        cnn_filters = min(int(w), 320)
        overrides.append(f"cnn_filters={cnn_filters}")
        overrides.append(f"num_lstm_layers={int(d)}")
        overrides.append(f"dropout_cnn={dr}")
        overrides.append(f"dropout_lstm={dr}")
    elif arch == "dream_attn":
        overrides.append(f"embedding_dim={int(w)}")
        overrides.append(f"num_blocks={int(d)}")
        # All 3 dropouts get the same value (single dropout knob)
        overrides.append(f"first_block_dropout={dr}")
        overrides.append(f"core_dropout={dr}")
        overrides.append(f"head_dropout={dr}")
    else:
        raise ValueError(f"Unknown arch: {arch}")

    return overrides


def sample_random(arch: str, rng) -> dict[str, Any]:
    """Sample one random config from the search space (for seeding / baselines)."""
    import math

    space = get_full_space(arch)
    out: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "choice":
            out[name] = rng.choice(spec[1])
        elif kind == "loguniform":
            lo, hi = spec[1], spec[2]
            out[name] = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        elif kind == "uniform":
            lo, hi = spec[1], spec[2]
            out[name] = float(rng.uniform(lo, hi))
        else:
            raise ValueError(f"Unknown spec type: {kind}")
    return out


def to_ray_space(arch: str) -> dict[str, Any]:
    """Convert the search space to a Ray Tune config dict.

    Returns a dict of {name: tune.<type>(...)} suitable for passing to
    tune.Tuner's param_space.
    """
    from ray import tune

    space = get_full_space(arch)
    out: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "choice":
            out[name] = tune.choice(spec[1])
        elif kind == "loguniform":
            out[name] = tune.loguniform(spec[1], spec[2])
        elif kind == "uniform":
            out[name] = tune.uniform(spec[1], spec[2])
        else:
            raise ValueError(f"Unknown spec type: {kind}")
    return out
