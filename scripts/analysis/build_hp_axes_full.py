"""Complete HP-axis inventory: every axis an HP strategy may vary.

Sourced from the code, not from the July slide: experiments/hp_strategies.py (core search space),
experiments/scaling_hp_search.py (the off-menu "novel axes" registry that AutoResearch may propose
when LLM_ALLOW_NOVEL_AXES=1), and experiments/scaling_hp_search_ms.py (batch-size menu, LR schedules).
Split into three tables so each fits a slide.
"""

import os

import pandas as pd

SRC = os.path.expanduser("~/Downloads/pi_meeting_figs/editable_tables")
C = ["Axis", "Type", "Range / choices", "Default or note", "Group"]

core_arch = [
    ["n_layers", "int", "2 – 12", "network depth", "architecture"],
    [
        "width_base",
        "categorical",
        "16, 32, 64, 128, 256",
        "PI feedback: extend to 512/1024",
        "architecture",
    ],
    [
        "width_ratio",
        "float (log)",
        "0.125 – 4.0",
        "layer-to-layer width growth/shrink",
        "architecture",
    ],
    [
        "width_jitter_i",
        "float × n_layers",
        "0.5 – 2.0 per layer",
        "per-layer width multiplier (up to 12)",
        "architecture",
    ],
    ["block_class", "categorical", "eff, ag, plain", "conv block family", "architecture"],
    ["ks", "categorical", "3, 5, 7, 9, 11", "PI feedback: extend to 25", "architecture"],
    ["pool_downsample", "categorical", "0, 1, 2, 3, 4", "number of pooling stages", "architecture"],
    [
        "outer_skip_style",
        "categorical *",
        "concat, add, none",
        "residual/skip wiring — PI: 'residuals greatly improve performance'",
        "architecture",
    ],
    ["skip_stride", "int *", "1 – 4", "how far a skip spans", "architecture"],
    ["activation", "categorical *", "silu, relu, gelu, …", "off-menu axis", "architecture"],
    ["se_reduction", "int *", "2 – 16", "squeeze-excite ratio", "architecture"],
]
core_opt = [
    ["lr", "float (log)", "1e-5 – 1e-2", "env HP_LR_MIN / HP_LR_MAX", "optimisation"],
    ["optimizer", "categorical", "adam, adamw, muon", "", "optimisation"],
    [
        "weight_decay",
        "float (log)",
        "1e-6 – 1e-2",
        "PI feedback: widen to 1e-8 – 1e-3",
        "optimisation",
    ],
    [
        "lr_schedule",
        "categorical",
        "plateau, onecycle, cosine",
        "plateau is correct under early stopping",
        "optimisation",
    ],
    [
        "pct_start",
        "categorical",
        "0.1, 0.2, 0.3, 0.4",
        "warmup fraction (OneCycle)",
        "optimisation",
    ],
    [
        "batch_size",
        "categorical",
        "D-aware menu",
        "window ¼·B_crit – 2·B_crit; B_crit ∝ D^0.301",
        "optimisation",
    ],
    ["conv_dropout", "float", "0.0 – 0.3", "", "regularisation"],
    ["dense_dropout", "float", "0.0 – 0.5", "", "regularisation"],
    ["loss", "categorical *", "mse, huber, …", "off-menu axis", "optimisation"],
    ["huber_delta", "float *", "0.1 – 5.0", "only if loss=huber", "optimisation"],
]
core_aug = [
    ["use_shift_aug", "bool", "False, True", "random shift augmentation", "augmentation"],
    ["shift_max", "categorical", "5, 10, 15, 20 bp", "shift magnitude", "augmentation"],
    ["use_evoaug", "bool", "False, True", "EvoAug on/off", "augmentation"],
    [
        "evoaug_intensity",
        "float *",
        "off-menu",
        "PI feedback: expose #augs explicitly",
        "augmentation",
    ],
    ["evoaug_prob", "float *", "0.05 – 1.0", "per-op probability", "augmentation"],
    ["use_reverse_complement", "bool *", "False, True", "RC augmentation", "augmentation"],
]
fm_axes = [
    [
        "branch",
        "categorical",
        "conv, res_tower, unet1, maxpool, transformer, full",
        "where the MPRA head attaches",
        "FM-specific",
    ],
    ["head_arch", "categorical", "linear, mlp, attn, conv", "readout head", "FM-specific"],
    ["freeze_transformer", "bool", "False, True", "recommended True", "FM-specific"],
    ["freeze_encoder", "bool", "False, True", "head-only control", "FM-specific"],
    [
        "bn_mode",
        "categorical",
        "frozen, dual, train",
        "frozen is the fix; train caused BN drift",
        "FM-specific",
    ],
    ["encoder_lr_mult", "float", "0.01 – 0.3", "stage-2 encoder LR = lr × this", "FM-specific"],
    ["stage1_frac", "float", "0.0 – 0.5", "frozen-warmup fraction", "FM-specific"],
    [
        "mpra_context",
        "categorical",
        "zeros, plasmid, dinuc",
        "how the 200 bp element is padded",
        "FM-specific",
    ],
    [
        "input_len / center_bins / pooling",
        "mixed",
        "512; all or N bins; mean/sum/max",
        "window and pooling",
        "FM-specific",
    ],
    [
        "cl mode",
        "categorical",
        "none, distill, replay_real, distill_replay",
        "continual-learning objective",
        "FM-specific",
    ],
    ["replay_lambda / distill_lambda", "float", "1 – 100", "constraint strength", "FM-specific"],
    ["replay_every", "int", "1, 4, 16", "anchor frequency (independent of λ)", "FM-specific"],
    [
        "anchor_len / anchor_n / anchor_batch",
        "int",
        "512 – 524,288; 16 – 512; 1 – 8",
        "CL anchor geometry",
        "FM-specific",
    ],
    [
        "init",
        "categorical",
        "pretrained, scratch",
        "+ optional --warmstart on genomic MPRA",
        "FM-specific",
    ],
    ["early_stop_patience", "int", "0 (off) – 10", "with generous --epochs", "training budget"],
]
tables = {
    "HP_1_architecture": pd.DataFrame(core_arch, columns=C),
    "HP_2_optimisation": pd.DataFrame(core_opt + core_aug, columns=C),
    "HP_3_FM_specific": pd.DataFrame(fm_axes, columns=C),
}
for n, df in tables.items():
    df.to_csv(f"{SRC}/{n}.csv", index=False)
    print(f"  {n}.csv  ({len(df)} axes)")

strategies = pd.DataFrame(
    [
        ["random", "uniform sampling over the space", "baseline"],
        ["optuna_tpe", "Tree-structured Parzen Estimator", ""],
        ["optuna_cmaes", "CMA-ES", ""],
        ["optuna_gp", "Gaussian-process Bayesian opt", ""],
        ["optuna_qmc", "quasi-Monte-Carlo (Sobol)", ""],
        ["optuna_parallel", "parallel TPE", ""],
        [
            "evo_single / evo_batch",
            "evolutionary: mutate best config(s)",
            "renamed from autoresearch_*",
        ],
        ["evo_explore / evo_exploit", "evolutionary, bias to exploration / refinement", ""],
        ["evo_adaptive", "adapts explore-exploit balance over rounds", "strong performer"],
        ["evo_massive / evo_knowledgeable", "large batches / prior-informed", ""],
        [
            "llm_autoresearch",
            "LLM proposes configs from history + schema",
            "personas: default/explore/exploit/critic/diverse/neutral/blank/misguided",
        ],
        [
            "llm_autoresearch + novel axes",
            "may invent OFF-MENU axes",
            "LLM_ALLOW_NOVEL_AXES=1; registry gates what is accepted",
        ],
    ],
    columns=["Strategy", "How it proposes configs", "Notes"],
)
strategies.to_csv(f"{SRC}/HP_4_strategies.csv", index=False)
print(f"  HP_4_strategies.csv  ({len(strategies)} strategies)")
print("\n* = off-menu 'novel axis': only proposable by AutoResearch with LLM_ALLOW_NOVEL_AXES=1")
