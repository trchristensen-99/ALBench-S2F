"""HP search space + per-architecture HP translation.

Shared by raytune_search.py and autoresearch.py. Defines:
- The 6-dim search space (width, depth, lr, batch_size, weight_decay, dropout)
- Per-arch ranges (LegNet, DREAM-RNN, DREAM-ATTN have different valid ranges)
- to_run_single_overrides(): translates abstract HP dict → run_single.py --hp args
"""

from __future__ import annotations

from typing import Any


def expand_block_sizes(width: int, depth: int, shape: str = "flat") -> list[int]:
    """Expand (width, depth, shape) into a concrete block_sizes list.

    Per Peter's feedback (May 12 2026): "change [width] by layer" — search
    over different shape patterns instead of forcing homogeneous widths.

    Shapes:
      flat         — [W, W, ..., W]                          uniform
      pyramid      — pairs of equal widths, halving every 2  (legnet default style)
      decreasing   — linear ramp from W → max(W/8, 16)       smooth narrowing
      increasing   — linear ramp from max(W/4, 16) → W       widening
    """
    width = int(width)
    depth = int(depth)
    if depth < 1:
        return [width]
    if shape == "flat":
        return [width] * depth
    if shape == "pyramid":
        sizes = []
        cur = width
        for i in range(depth):
            sizes.append(int(cur))
            if (i + 1) % 2 == 0 and cur > 16:
                cur = max(16, cur // 2)
        return sizes
    if shape == "decreasing":
        end = max(width // 8, 16)
        return [int(round(width + (end - width) * i / max(1, depth - 1))) for i in range(depth)]
    if shape == "increasing":
        start = max(width // 4, 16)
        return [int(round(start + (width - start) * i / max(1, depth - 1))) for i in range(depth)]
    raise ValueError(f"unknown shape {shape!r}; expected one of flat/pyramid/decreasing/increasing")


# Boundaries chosen 2026-05-11.
# Each arch has independent (width, depth, dropout) ranges based on what is
# computationally feasible and architecturally meaningful. LR/BS/WD use the
# same wide range across archs — the optimizer can carve out per-arch regions.
SHARED_RANGES: dict[str, Any] = {
    "lr": ("loguniform", 1e-5, 1e-2),
    "batch_size": ("choice", [64, 128, 256, 512, 1024]),
    "weight_decay": ("loguniform", 1e-6, 1e-1),
    # `dropout` is a legacy single-knob default. For LegNet it's overridden by
    # the explicit conv_dropout/dense_dropout below; for dream_attn it still
    # acts as a single dropout knob applied to all 3 sites.
    "dropout": ("choice", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    # Added per Peter's feedback (May 11 2026): optimizer choice
    "optimizer": ("choice", ["adam", "adamw"]),  # muon added later (needs pip)
}

ARCH_RANGES: dict[str, dict[str, Any]] = {
    "legnet": {
        # Per-layer widths picked from a shape pattern + global width/depth.
        # See expand_block_sizes() — shape ∈ {flat, pyramid, decreasing, increasing}.
        "width": ("choice", [128, 256, 512, 1024, 2000]),
        "depth": ("choice", [2, 3, 4, 5, 6, 7]),
        "shape": ("choice", ["flat", "pyramid", "decreasing", "increasing"]),
        # Added per Peter: alternative block classes (vanilla conv / AlphaGenome-style)
        # See models/legnet.BLOCK_CLASSES
        "block_class": ("choice", ["eff", "plain", "ag"]),
        # Peter (May 12 2026): split conv vs dense dropout. Conv layers need less
        # dropout than dense layers — search them independently.
        "conv_dropout": ("choice", [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]),
        "dense_dropout": ("choice", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        # Dense head shape. [] = original 1x1-conv + GAP head (no dense layers).
        # Single or two hidden layers map the GAP output → 1 scalar.
        "dense_dims": ("choice", [[], [256], [512], [256, 128], [512, 256]]),
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
    if "optimizer" in hp:
        overrides.append(f"optimizer={hp['optimizer']}")

    if arch == "legnet":
        shape = hp.get("shape", "flat")
        block_sizes = expand_block_sizes(int(w), int(d), shape)
        overrides.append(f"block_sizes={block_sizes}")
        if "shape" in hp:
            overrides.append(f"shape={shape}")
        # If conv_dropout/dense_dropout are present (new search space), forward
        # them; otherwise fall back to the legacy shared `dropout`.
        if "conv_dropout" in hp:
            overrides.append(f"conv_dropout={hp['conv_dropout']}")
        else:
            overrides.append(f"dropout={dr}")
        if "dense_dropout" in hp:
            overrides.append(f"dense_dropout={hp['dense_dropout']}")
        if "dense_dims" in hp:
            # Pass as list literal — parse_overrides understands [a,b]
            dd = hp["dense_dims"]
            if isinstance(dd, (list, tuple)):
                dd_str = "[" + ",".join(str(int(x)) for x in dd) + "]"
            else:
                dd_str = str(dd)
            overrides.append(f"dense_dims={dd_str}")
        if "block_class" in hp:
            overrides.append(f"block_class={hp['block_class']}")
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
