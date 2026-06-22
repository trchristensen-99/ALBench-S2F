"""HP search v2 — chr-split + cached data + 10% holdout val + optional compile + TF32.

Key changes from v1:
  - Uses pre-built chr_split_cache (faster startup, no DATA-Table_S2 reload)
  - 10% random holdout from train pool serves as val (oracle-labeled, consistent
    with other reservoir strategy protocols)
  - Disables torch.compile via TrainConfig.use_compile=False to skip autotune
  - Sets torch.set_float32_matmul_precision('high') for TF32 on H100/A100
  - Eval target = chr-test genomic with ORACLE labels (primary)
    + report real labels too as secondary metric
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

from experiments.test_set_guards import assert_mono_snv, read_battery_provenance

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
CACHE = REPO / "outputs/chr_split_cache"

# TF32 — fast on H100 with negligible accuracy loss
torch.set_float32_matmul_precision("high")


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically (tmp file + os.replace). A resumed job — or any
    reader — never sees a half-written checkpoint if this process is killed
    mid-write (walltime SIGKILL, preemption, node crash). os.replace is atomic
    on POSIX, so the target is always either the old file or the complete new one."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_savez(path: Path, **arrays) -> None:
    """np.savez to a tmp .npz then atomic-replace (see _atomic_write_text)."""
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# Canonical seed for the reserved-eval partition split. Independent of any
# data_seed so the SEARCH (load_chr_train_pool) and the fair-comparison retrain
# harness agree on exactly which pool rows are held out vs searchable.
POOL_RESERVE_SEED = 20260618


def pool_partition(n: int, reserve_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Split [0, n) into (searchable, reserved_eval) by the canonical permutation.

    The reserved-eval rows are NEVER sampled by any search data_seed, so a retrain
    drawing train+val from them is disjoint from all search data. Returns
    (searchable_idx, reserved_eval_idx); reserve_frac<=0 → (all, empty).
    """
    if reserve_frac <= 0:
        return np.arange(n), np.empty(0, dtype=int)
    perm = np.random.default_rng(POOL_RESERVE_SEED).permutation(n)
    n_reserve = int(reserve_frac * n)
    return perm[n_reserve:], perm[:n_reserve]


# ── Regime stamping (no-mixing guarantee) ──────────────────────────────────────
# Bump when the meaning of a stamped field changes so old results can be told apart.
REGIME_SCHEMA_VERSION = "v1"
BATTERY_DIR = REPO / "data/k562/test_sets_ag_s2_chrsplit"


def load_battery_provenance() -> dict:
    """Read the oracle/test-set provenance stamp written by the re-scoring pipeline.

    Returns {'oracle_id', 'test_set_version'}. Defaults to 'unstamped' until the
    battery has been re-scored + stamped (see PROVENANCE.json in BATTERY_DIR).
    Surfacing these here lets every HP result record exactly which oracle and
    test-set version produced its labels."""
    return read_battery_provenance(BATTERY_DIR)


def build_regime(args, patience: int, min_delta: float, battery_prov: dict) -> dict:
    """The training/eval regime every result in an out_dir is stamped with.

    Two runs whose regimes differ (e.g. epochs=15 vs 100, or a different oracle)
    MUST NOT be mixed in one out_dir or one deploy aggregation — their rankings
    are not comparable. The out-dir-level guard in run_search and the per-meta
    stamp downstream both key off this dict."""
    return {
        "schema_version": REGIME_SCHEMA_VERSION,
        "epochs": int(args.epochs),
        "early_stop_patience": int(patience),
        "min_delta": float(min_delta),
        "metric_for_best": "pearson_r",
        "val_protocol": (
            "reservoir_val_chr_heldout"
            if getattr(args, "reservoir_val_cache", None)
            else "chr_val"
            if getattr(args, "chr_val", False)
            else "per_combo_10pct"
        ),
        "oracle_id": battery_prov["oracle_id"],
        "test_set_version": battery_prov["test_set_version"],
    }


def regime_key(regime: dict | None) -> str:
    """Stable string key for comparing two regimes. None/empty → 'legacy_unstamped'."""
    if not regime:
        return "legacy_unstamped"
    return json.dumps(regime, sort_keys=True)


# ── Data loading (cached) ─────────────────────────────────────────────────────


def load_chr_train_pool(
    D: int | None,
    ref_only: bool = True,
    val_frac: float = 0.1,
    seed: int = 0,
    reservoir_cache: str | Path | None = None,
    chr_val: bool = False,
    reservoir_val_cache: str | Path | None = None,
):
    """Load chr-train pool with oracle labels; carve out val.

    Returns (train_seqs, train_labels, val_seqs, val_labels) — all oracle labels.

    When chr_val=False (default), val = 10% random holdout from the training pool.
    When chr_val=True, val = real chr19+21+X genomic sequences from
    outputs/chr_split_cache/chr_val_ref_only.npz (the "production" val protocol).
    For genomic-based strategies this matches what's used at inference time.
    For synthetic strategies (e.g. random, motif_planted) chr_val still works —
    train comes from the reservoir cache, val from real chr-val.
    """
    if reservoir_cache is not None:
        cache_path = Path(reservoir_cache)
        z = np.load(cache_path, allow_pickle=True)
        fname = cache_path.name
        # Contamination guard: refuse any reservoir cache not stamped with the
        # canonical oracle. Re-score with scripts/rescore_reservoir_cache.py.
        # Override only for deliberate ad-hoc runs via HP_ALLOW_UNSTAMPED=1.
        if os.environ.get("HP_ALLOW_UNSTAMPED") != "1":
            stamp = str(z["oracle_id"]) if "oracle_id" in z.files else "UNSTAMPED"
            if stamp != "full856k_clean":
                raise RuntimeError(
                    f"reservoir cache {fname} has oracle_id={stamp!r}, expected "
                    "'full856k_clean'. Re-score it (scripts/rescore_reservoir_cache.py) "
                    "or set HP_ALLOW_UNSTAMPED=1 to bypass."
                )
    else:
        fname = "chr_train_ref_only.npz" if ref_only else "chr_train_all_alleles.npz"
        z = np.load(CACHE / fname, allow_pickle=True)
    seqs = z["sequences"]
    labels = z["oracle_labels"]
    n = len(seqs)
    print(f"  Loaded {fname}: n={n:,}, μ={labels.mean():.3f} σ={labels.std():.3f}")

    # Reserved-eval partition (fair R×A comparison): carve a canonical tail of the
    # pool — using a FIXED seed independent of data_seed — that the SEARCH never
    # samples. The fair-comparison retrain (deployed menu / genomic-transferred /
    # native, all locked) draws its fresh train+val from that reserved tail, so it
    # is disjoint from every search seed's data (data_seed draws only overlap, they
    # are not disjoint). Off by default (frac=0 → full pool searchable, legacy).
    reserve_frac = float(os.environ.get("HP_POOL_RESERVE_EVAL_FRAC", "0") or 0)
    universe, reserved = pool_partition(n, reserve_frac)
    if reserve_frac > 0:
        print(
            f"  Reserved-eval partition: {len(reserved):,} held out "
            f"(frac={reserve_frac}); searchable universe={len(universe):,}"
        )

    if D is not None and D < len(universe):
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(universe), size=D, replace=False)
        idx = universe[pick]
        seqs = seqs[idx]
        labels = labels[idx]
        print(f"  Subsampled to D={D:,}")
    elif D is not None and D > len(universe):
        seqs = seqs[universe]
        labels = labels[universe]
        print(f"  WARN: D={D:,} > searchable universe {len(universe):,}; using all searchable")

    if reservoir_val_cache is not None:
        # Held-out, transform-matched val: same reservoir transform applied to the
        # chr19/21/X backgrounds (disjoint from the chr-train pool), oracle-labeled.
        # Subsample to val_frac*D so val stays proportional to the train size.
        vpath = Path(reservoir_val_cache)
        vz = np.load(vpath, allow_pickle=True)
        if os.environ.get("HP_ALLOW_UNSTAMPED") != "1":
            vstamp = str(vz["oracle_id"]) if "oracle_id" in vz.files else "UNSTAMPED"
            if vstamp != "full856k_clean":
                raise RuntimeError(
                    f"reservoir_val_cache {vpath.name} has oracle_id={vstamp!r}, expected "
                    "'full856k_clean'. Regenerate with --oracle_id_stamp full856k_clean."
                )
        vseqs = [str(s) for s in vz["sequences"]]
        vlabels = vz["oracle_labels"].astype(np.float32)
        vfin = np.isfinite(vlabels)
        if not vfin.all():
            vseqs = [s for s, ok in zip(vseqs, vfin) if ok]
            vlabels = vlabels[vfin]
        n_val = max(1, min(len(vseqs), int(val_frac * len(seqs))))
        rng_v = np.random.default_rng(seed + 2)
        vidx = rng_v.choice(len(vseqs), size=n_val, replace=False)
        val_seqs = [vseqs[i] for i in vidx]
        val_labels = vlabels[vidx]
        train_seqs = [str(s) for s in seqs]
        train_labels = labels.astype(np.float32)
        print(
            f"  Train={len(train_seqs):,}  Val(held-out transform-matched, "
            f"{vpath.name})={len(val_seqs):,}"
        )
    elif chr_val:
        # Real chr19+21+X val with oracle labels — production-style holdout.
        cv = np.load(CACHE / "chr_val_ref_only.npz", allow_pickle=True)
        val_seqs = [str(s) for s in cv["sequences"]]
        val_labels = cv["oracle_labels"].astype(np.float32)
        # Filter to finite labels (some real sequences have NaN oracle on bad inputs)
        finite = np.isfinite(val_labels)
        if not finite.all():
            val_seqs = [s for s, ok in zip(val_seqs, finite) if ok]
            val_labels = val_labels[finite]
        train_seqs = [str(s) for s in seqs]
        train_labels = labels.astype(np.float32)
        print(f"  Train={len(train_seqs):,}  Val(chr19+21+X)={len(val_seqs):,}")
    else:
        # 10% holdout for val (legacy / synthetic-strategy path)
        rng2 = np.random.default_rng(seed + 1)
        perm = rng2.permutation(len(seqs))
        n_val = int(val_frac * len(seqs))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        val_seqs = [str(s) for s in seqs[val_idx]]
        val_labels = labels[val_idx].astype(np.float32)
        train_seqs = [str(s) for s in seqs[train_idx]]
        train_labels = labels[train_idx].astype(np.float32)
        print(f"  Train={len(train_seqs):,}  Val(10% holdout)={len(val_seqs):,}")
    return train_seqs, train_labels, val_seqs, val_labels


def load_chr_test_genomic():
    """Chr-test genomic ref-only sequences with oracle (eval target) + real labels (secondary)."""
    z = np.load(REPO / "data/k562/test_sets_ag_s2_chrsplit/genomic_oracle.npz", allow_pickle=True)
    seqs = [str(s) for s in z["sequences"]]
    oracle = z["oracle_mean"].astype(np.float32)
    true_label = z["true_label"].astype(np.float32)
    print(f"  Chr-test genomic: n={len(seqs):,}")
    return seqs, oracle, true_label


# Comprehensive test-set battery (all under data/k562/test_sets_ag_s2_chrsplit/)
# Each *_oracle.npz has keys: sequences (object), oracle_mean (float32).
# Generated by scripts/build_comprehensive_test_sets.py (one-time).
TEST_SET_NAMES = [
    "genomic",  # 32k chr 7+13 (canonical in-dist; loaded above too)
    "ood",  # 22k designed sequences (out-of-distribution)
    "snv",  # SNV ref/alt pairs for SNV effect prediction
    "random_32k",  # 32k uniformly random
    "dinuc_shuffle",  # 32k dinuc-preserving shuffle of genomic
    "sub_low",
    "sub_med",
    "sub_high",
    "ins_low",
    "ins_med",
    "ins_high",
    "del_low",
    "del_med",
    "del_high",
    "translocation",
    "inversion",
]


def load_all_test_sets() -> dict:
    """Return {set_name: (sequences, oracle_labels)} for every test set that exists on disk.

    Missing or non-standard-schema sets are silently skipped so scaling_hp_search.py
    works even before the full battery is materialized (e.g. during incremental
    rollout). SNV has a paired ref/alt schema — handled separately as snv_ref and
    snv_alt sets if the file is present.
    """
    test_dir = REPO / "data/k562/test_sets_ag_s2_chrsplit"
    # Eager, loud SNV guard: a present-but-wrong SNV file must halt the run, not be
    # silently skipped by the tolerant per-set loop below (which only tolerates a
    # *missing* set during incremental rollout).
    snv_path = test_dir / "snv_oracle.npz"
    if snv_path.exists():
        assert_mono_snv(np.load(snv_path, allow_pickle=True), snv_path)
    out = {}
    for name in TEST_SET_NAMES:
        path = test_dir / f"{name}_oracle.npz"
        if not path.exists():
            # legacy "random_10k" fallback
            if name == "random_32k" and (test_dir / "random_10k_oracle.npz").exists():
                path = test_dir / "random_10k_oracle.npz"
            else:
                continue
        try:
            z = np.load(path, allow_pickle=True)
            if name == "snv":
                # paired ref/alt schema → split into 2 sets
                if "ref_sequences" not in z.files:
                    continue
                ref_seqs = [str(s) for s in z["ref_sequences"]]
                alt_seqs = [str(s) for s in z["alt_sequences"]]
                ref_lab = z["ref_mean"].astype(np.float32)
                alt_lab = z["alt_mean"].astype(np.float32)
                out["snv_ref"] = (ref_seqs, ref_lab)
                out["snv_alt"] = (alt_seqs, alt_lab)
                print(
                    f"  test set snv_ref         : n={len(ref_seqs):,}  μ={ref_lab.mean():+.3f} σ={ref_lab.std():.3f}"
                )
                print(
                    f"  test set snv_alt         : n={len(alt_seqs):,}  μ={alt_lab.mean():+.3f} σ={alt_lab.std():.3f}"
                )
                continue
            # Standard schema: sequences + oracle_mean (or oracle_labels)
            if "sequences" not in z.files:
                print(f"  [skip] {name}: no 'sequences' key (keys={z.files})")
                continue
            seqs = [str(s) for s in z["sequences"]]
            if "oracle_mean" in z.files:
                labels = z["oracle_mean"]
            elif "oracle_labels" in z.files:
                labels = z["oracle_labels"]
            else:
                print(f"  [skip] {name}: no oracle_mean/oracle_labels key")
                continue
            labels = labels.astype(np.float32)
            out[name] = (seqs, labels)
            print(
                f"  test set {name:<16s}: n={len(seqs):,}  μ={labels.mean():+.3f} σ={labels.std():.3f}"
            )
        except Exception as e:
            print(f"  [skip] {name}: error loading — {type(e).__name__}: {str(e)[:100]}")
    return out


# ── HP space ──────────────────────────────────────────────────────────────────


@dataclass
class HPConfig:
    lr: float
    batch_size: int
    conv_dropout: float
    dense_dropout: float
    n_layers: int
    width_base: int
    width_jitter: list  # per-layer multiplier in [0.5, 2.0]
    block_class: str  # {"eff", "ag", "plain"}
    ks: int  # kernel size
    pct_start: float  # OneCycleLR warmup fraction
    optimizer: str
    weight_decay: float
    use_shift_aug: bool
    shift_max: int
    use_evoaug: bool
    seed: int
    # LR-schedule axis. One of LR_SCHEDULE_CHOICES; default "plateau" keeps configs
    # that don't set it on the early-stop-correct schedule. The LLM may also put an
    # off-menu scheduler name here (or a torch class name) + an optional
    # extra["lr_schedule_kwargs"] dict — LegNetStudent builds it generically.
    lr_schedule: str = "plateau"
    # Free-form, off-menu axes proposed by the LLM AutoResearch strategy when
    # LLM_ALLOW_NOVEL_AXES=1. Empty for all core-axis configs. Recognized keys
    # (see EXPERIMENTAL_KNOBS) are applied to training/model; unrecognized keys
    # are recorded in the meta for human review but are inert.
    extra: dict = field(default_factory=dict)


# LR-schedule menu the fixed-axis optimizers sample from. LegNetStudent also accepts
# any other name (e.g. an off-menu torch scheduler class) and falls back to "plateau".
LR_SCHEDULE_CHOICES = [
    "plateau",  # ReduceLROnPlateau (default; correct under early stopping)
    "onecycle",  # OneCycleLR over the full epoch budget (per-batch)
    "cosine",  # CosineAnnealingLR over epochs
    "cosine_warm",  # CosineAnnealingWarmRestarts
    "step",  # StepLR
    "exponential",  # ExponentialLR
    "constant",  # no schedule
]


# ── Experimental / novel HP knobs ───────────────────────────────────────────
# Axes NOT in the core 15-D HPConfig that the LLM AutoResearch strategy may
# propose (inside an "extra" object) when LLM_ALLOW_NOVEL_AXES=1, to widen the
# diversity of trained models. Each recognized knob is validated + APPLIED to
# training/model. Any unrecognized key is RECORDED in the meta (hp.extra) for
# later human review / promotion into the formal space, but stays inert — the
# LLM cannot introduce executable code, only declarative values.
_ACTIVATIONS = {"silu", "relu", "gelu", "quickgelu", "mish", "elu"}
_LOSSES = {"mse", "huber", "smoothl1"}


def _clipf(v, lo, hi, default):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


# name -> (target_group, validator).  target_group in {"train", "model", "loss"}
EXPERIMENTAL_KNOBS = {
    "use_reverse_complement": ("train", lambda v: bool(v)),
    "evoaug_intensity": (
        "train",
        lambda v: str(v).lower() if str(v).lower() in {"light", "medium", "heavy"} else "medium",
    ),
    "evoaug_prob": ("train", lambda v: _clipf(v, 0.05, 1.0, 0.5)),
    "activation": ("model", lambda v: str(v).lower() if str(v).lower() in _ACTIVATIONS else "silu"),
    "se_reduction": ("model", lambda v: int(max(2, min(16, int(v))))),
    "loss": ("loss", lambda v: str(v).lower() if str(v).lower() in _LOSSES else "mse"),
    "huber_delta": ("loss", lambda v: _clipf(v, 0.1, 5.0, 1.0)),
}

# Injected into the LLM prompt only under novel-axes mode (keeps the default
# prompt byte-identical when the feature is off).
EXPERIMENTAL_KNOBS_DOC = """OPTIONAL EXPERIMENTAL AXES (beyond the 15 above) — put any of these in an "extra" object:
  use_reverse_complement : bool   — average fwd+reverse-complement loss & predictions
  evoaug_intensity       : "light"|"medium"|"heavy"  (only matters if use_evoaug=true)
  evoaug_prob            : float [0.05, 1.0]  — per-sample EvoAug apply probability
  activation             : "silu"|"relu"|"gelu"|"quickgelu"|"mish"|"elu"
  se_reduction           : int [2, 16]  — squeeze-excite bottleneck reduction factor
  loss                   : "mse"|"huber"|"smoothl1"
  huber_delta            : float [0.1, 5.0]  (only if loss is huber/smoothl1)
You MAY also invent entirely new keys inside "extra" if you have a strong, well-
reasoned idea — they will be recorded for review even though they are not yet
wired into training. Use "extra" to push model diversity further, NOT to restate
any of the 15 core axes."""


def apply_experimental_knobs(extra: dict):
    """Split a free-form `extra` dict into validated overrides.

    Returns (train_overrides, model_overrides, loss_overrides, applied, recorded_only)
    where `applied` is a list of "key=value" strings and `recorded_only` lists keys
    that were kept in the meta but not acted on (unknown to this code version)."""
    train_o, model_o, loss_o, applied, recorded = {}, {}, {}, [], []
    for k, v in (extra or {}).items():
        spec = EXPERIMENTAL_KNOBS.get(k)
        if spec is None:
            recorded.append(k)
            continue
        target, validate = spec
        val = validate(v)
        {"train": train_o, "model": model_o, "loss": loss_o}[target][k] = val
        applied.append(f"{k}={val}")
    return train_o, model_o, loss_o, applied, recorded


def _hp_to_dict(hp) -> dict:
    """asdict(hp), dropping an empty `extra` so core-axis metas stay byte-identical."""
    d = asdict(hp)
    if not d.get("extra"):
        d.pop("extra", None)
    return d


def batch_size_menu(D: int | None) -> list[int]:
    """D-aware batch_size menu (4× diversity window centered near ½·B_crit).

    Empirical anchors: B_crit=512 at D=30k (n=17,290), B_crit=1024 at D=300k
    (n=158); slope B_crit ∝ D^0.301. See ~/Downloads/hp_strategy_curves/.

    None / unknown D → full legacy menu (back-compat for ad-hoc runs)."""
    if D is None:
        return [32, 64, 128, 256, 512, 1024]
    if D <= 15_000:
        return [64, 128, 256, 512]
    if D <= 50_000:
        return [128, 256, 512, 1024]
    if D <= 500_000:
        return [256, 512, 1024]
    return [512, 1024, 2048]


def sample_random_hp(rng: np.random.Generator, seed: int, D: int | None = None) -> HPConfig:
    n_layers = int(rng.integers(2, 13))  # 2 to 12
    width_jitter = [float(2 ** rng.uniform(-1, 1)) for _ in range(n_layers)]
    return HPConfig(
        lr=float(10 ** rng.uniform(-5, -2)),
        batch_size=int(rng.choice(batch_size_menu(D))),
        conv_dropout=float(rng.uniform(0, 0.3)),
        dense_dropout=float(rng.uniform(0, 0.5)),
        n_layers=n_layers,
        width_base=int(rng.choice([16, 32, 64, 128, 256])),
        width_jitter=width_jitter,
        block_class=str(rng.choice(["eff", "ag", "plain"])),
        ks=int(rng.choice([3, 5, 7, 9, 11])),
        pct_start=float(rng.choice([0.1, 0.2, 0.3, 0.4])),
        optimizer=str(rng.choice(["adam", "adamw", "muon"])),
        weight_decay=float(10 ** rng.uniform(-6, -2)),
        use_shift_aug=bool(rng.random() < 0.5),
        shift_max=int(rng.choice([5, 10, 15, 20])),
        use_evoaug=bool(rng.random() < 0.3),
        lr_schedule=str(rng.choice(LR_SCHEDULE_CHOICES)),
        seed=seed,
    )


def build_block_sizes(
    n_layers: int, width_base: int, width_jitter: list | None = None
) -> list[int]:
    """Per-layer widths with jitter; caps max to 2*width_base to prevent blow-up."""
    if width_jitter is None:
        width_jitter = [1.0] * n_layers
    sizes = []
    for i in range(n_layers):
        # Pattern: layers 0-1 = width_base, 2-3 = /2, 4-5 = /4, 6+ = /8 (floor 8)
        tier = i // 2
        mult = max(0.125, 2.0 ** (-tier))  # 1, 1, 0.5, 0.5, 0.25, 0.25, ...
        jitter = width_jitter[i] if i < len(width_jitter) else 1.0
        sz = max(8, int(round(width_base * mult * jitter / 8)) * 8)
        sz = min(sz, 2 * width_base)  # cap at 2x base for jitter safety
        sizes.append(sz)
    return sizes


# ── Training ──────────────────────────────────────────────────────────────────


def train_one_model(
    hp: HPConfig,
    train_seqs,
    train_labels,
    val_seqs,
    val_labels,
    test_seqs,
    epochs: int = 30,
    device: str = "cuda",
    use_compile: bool = False,
    early_stopping_patience: int = 8,
    min_delta: float = 1e-3,
    extra_test_sets: dict | None = None,
):
    import sys

    sys.path.insert(0, str(REPO))
    import random as _random

    from models.legnet_student import LegNetStudent, TrainConfig

    # Seed every RNG the training path can touch so a (hp, seed) pair is
    # reproducible: CPU torch, all CUDA devices (shift-aug draws from CUDA RNG in
    # training.py), global numpy, and stdlib random.
    torch.manual_seed(hp.seed)
    torch.cuda.manual_seed_all(hp.seed)
    np.random.seed(hp.seed)
    _random.seed(hp.seed)
    width_jitter = hp.width_jitter if hp.width_jitter else [1.0] * hp.n_layers
    block_sizes = build_block_sizes(hp.n_layers, hp.width_base, width_jitter)
    # Off-menu novel axes (empty unless LLM_ALLOW_NOVEL_AXES was set when proposing).
    tr_over, md_over, ls_over, applied_knobs, recorded_knobs = apply_experimental_knobs(
        getattr(hp, "extra", {}) or {}
    )
    if not hp.use_evoaug:
        # Don't let an evoaug_intensity override silently enable EvoAug.
        tr_over.pop("evoaug_intensity", None)
    if applied_knobs or recorded_knobs:
        print(
            f"  [novel-axes] applied={applied_knobs or '—'} "
            f"recorded-only(inert)={recorded_knobs or '—'}",
            flush=True,
        )
    tcfg_kwargs = dict(
        lr=hp.lr,
        batch_size=hp.batch_size,
        weight_decay=hp.weight_decay,
        epochs=epochs,
        pct_start=hp.pct_start,
        optimizer=hp.optimizer,
        evoaug_intensity="medium" if hp.use_evoaug else None,
        shift_aug=hp.use_shift_aug,
        max_shift=hp.shift_max,
        num_workers=4,
        use_compile=use_compile,
        early_stopping_patience=early_stopping_patience,
        min_delta=min_delta,
    )
    tcfg_kwargs.update(tr_over)  # use_reverse_complement / evoaug_intensity / evoaug_prob
    tcfg_kwargs.update(ls_over)  # loss / huber_delta
    train_cfg = TrainConfig(**tcfg_kwargs)
    student = LegNetStudent(
        task_mode="k562",
        ensemble_size=1,
        block_sizes=block_sizes,
        ks=hp.ks,
        block_class=hp.block_class,
        device=device,
        train_config=train_cfg,
        in_channels=4,
        conv_dropout=hp.conv_dropout,
        dense_dropout=hp.dense_dropout,
        **md_over,  # activation / se_reduction
    )
    t0 = time.time()
    student.fit(train_seqs, train_labels, val_sequences=val_seqs, val_labels=val_labels)
    train_time = time.time() - t0

    val_pred = student.predict(val_seqs)
    test_pred = student.predict(test_seqs)
    val_r = float(pearsonr(val_pred, val_labels)[0])
    val_mse = float(((val_pred - val_labels) ** 2).mean())

    # Epoch diagnostics from the (single-member) training history
    epoch_diag = {}
    if student.histories:
        h = student.histories[0]
        vp = h.get("val_pearson_r", [])
        if vp:
            best_ep = int(np.argmax(vp))
            epoch_diag = {
                "best_epoch": best_ep,
                "epochs_trained": len(vp),
                "early_stopped": len(vp) < epochs,
                "best_val_pearson": float(vp[best_ep]),
                # Full per-epoch val trajectory so the stopping point can be re-examined
                # post-hoc (Pearson vs MSE criterion, plateau shape, recalibration).
                "val_trajectory": {
                    "val_pearson_r": [float(x) for x in vp],
                    "val_loss": [float(x) for x in h.get("val_loss", [])],
                    "val_spearman_r": [float(x) for x in h.get("val_spearman_r", [])],
                    "train_loss": [float(x) for x in h.get("train_loss", [])],
                },
            }

    result = {
        "val_pred": val_pred,
        "test_pred": test_pred,  # backward-compat: this is the genomic test set
        "val_pearson": val_r,
        "val_mse": val_mse,
        "train_time_sec": train_time,
        "hp": _hp_to_dict(hp),
        "block_sizes": block_sizes,
        **epoch_diag,
    }
    if applied_knobs or recorded_knobs:
        result["novel_axes"] = {"applied": applied_knobs, "recorded_only": recorded_knobs}

    # Predict on all extra test sets (comprehensive eval battery)
    if extra_test_sets:
        per_set_metrics = {}
        for set_name, (seqs, oracle_labels) in extra_test_sets.items():
            pred = student.predict(seqs)
            result[f"test_pred_{set_name}"] = pred
            # Pairwise metrics vs oracle labels (skip NaN-safe just in case)
            mask = np.isfinite(pred) & np.isfinite(oracle_labels)
            if mask.sum() >= 8:
                r = float(pearsonr(pred[mask], oracle_labels[mask])[0])
                mse = float(((pred[mask] - oracle_labels[mask]) ** 2).mean())
            else:
                r, mse = float("nan"), float("nan")
            per_set_metrics[set_name] = {"pearson": r, "mse": mse, "n": int(mask.sum())}
        result["per_set_metrics"] = per_set_metrics

    return result


# ── Multi-strategy driver ─────────────────────────────────────────────────────


def _preload_history(out_dir: Path, strategies: dict, expected_regime_key: str) -> int:
    """Seed each strategy's history from already-saved *_meta.json so a resumed
    run's suggest() calls see all prior results. Returns count of records loaded.

    Only metas whose stamped regime matches expected_regime_key are preloaded —
    a stray result from a different epoch/oracle regime must never seed history."""
    n = 0
    skipped_regime = 0
    per_strat: dict[str, tuple[list, list]] = {name: ([], []) for name in strategies}
    for meta_path in sorted(out_dir.glob("r*_meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if regime_key(meta.get("regime")) != expected_regime_key:
            skipped_regime += 1
            continue
        name = meta.get("strategy")
        if name not in per_strat or "val_pearson" not in meta:
            continue
        hp_fields = {
            k: v for k, v in meta.get("hp", {}).items() if k in HPConfig.__dataclass_fields__
        }
        per_strat[name][0].append(HPConfig(**hp_fields))
        per_strat[name][1].append(meta["val_pearson"])
        n += 1
    for name, (cs, vs) in per_strat.items():
        if cs:
            strategies[name].update(cs, vs)
            print(f"  [resume] preloaded {len(cs)} prior results into '{name}' history")
    if skipped_regime:
        print(
            f"  [resume] skipped {skipped_regime} prior result(s) from a DIFFERENT regime "
            f"(not preloaded — out_dir should hold a single regime)"
        )
    return n


def run_search(args):
    import sys

    sys.path.insert(0, str(REPO))
    from experiments.hp_strategies import get_strategy

    try:
        from experiments import llm_autoresearch  # registers llm_autoresearch strategy
    except ImportError:
        pass  # anthropic SDK not available
    try:
        from experiments.llm_autoresearch import RateLimitExceeded
    except Exception:

        class RateLimitExceeded(Exception):  # fallback: never raised w/o LLM strategy
            pass

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Regime stamp + no-mixing guard ──────────────────────────────────────
    # Resolve the effective patience/min_delta ONCE (same values used per-model
    # below) and stamp the whole run. If this out_dir already holds results from
    # a different regime, refuse to continue — mixing epoch/oracle regimes in one
    # dir silently corrupts rankings. Use a fresh out_dir per regime.
    esp = getattr(args, "early_stop_patience", None) or 15
    min_delta = getattr(args, "min_delta", 1e-3)
    battery_prov = load_battery_provenance()
    regime = build_regime(args, esp, min_delta, battery_prov)
    rk = regime_key(regime)
    regime_path = out_dir / "regime.json"
    if regime_path.exists():
        try:
            prior_regime = json.loads(regime_path.read_text())
        except Exception:
            prior_regime = None
        if regime_key(prior_regime) != rk:
            raise SystemExit(
                f"out_dir {out_dir} was created under a DIFFERENT regime:\n"
                f"  existing: {prior_regime}\n  current:  {regime}\n"
                f"Refusing to mix regimes in one dir — use a fresh --out_dir."
            )
    else:
        _atomic_write_text(regime_path, json.dumps(regime, indent=2))
    print(f"=== Regime: {regime} ===")

    print(
        f"=== Loading data (D={args.D}, ref_only={args.ref_only}, reservoir_cache={getattr(args, 'reservoir_cache', None)}, chr_val={getattr(args, 'chr_val', False)}) ==="
    )
    train_seqs, train_labels, val_seqs, val_labels = load_chr_train_pool(
        args.D,
        ref_only=args.ref_only,
        val_frac=0.1,
        seed=args.data_seed,
        reservoir_cache=getattr(args, "reservoir_cache", None),
        chr_val=getattr(args, "chr_val", False),
        reservoir_val_cache=getattr(args, "reservoir_val_cache", None),
    )
    test_seqs, test_oracle, test_true = load_chr_test_genomic()

    # Load comprehensive eval battery (may be partial; missing files are skipped).
    print("=== Loading comprehensive test-set battery ===")
    all_test_sets = load_all_test_sets()
    # The "genomic" entry in all_test_sets duplicates load_chr_test_genomic — that's ok;
    # the analysis pipeline expects per-set predictions for all sets, including genomic.

    # Save labels once: legacy keys + per-set oracle labels for downstream analysis
    label_dict = {
        "val_labels": val_labels,
        "test_oracle": test_oracle,
        "test_true": test_true,
    }
    for set_name, (_, oracle_labels) in all_test_sets.items():
        label_dict[f"oracle_{set_name}"] = oracle_labels
    # Stamp the labels archive with the regime so any consumer can verify which
    # oracle/test-set version produced these labels.
    label_dict["regime_json"] = np.array(json.dumps(regime))
    _atomic_savez(out_dir / "labels.npz", **label_dict)

    # Build strategies
    strategy_names = args.strategies.split(",")
    # D is threaded into strategies so batch_size_menu(D) is used at proposal time.
    strategies = {
        name: get_strategy(name, seed=args.hp_seed + i * 1000, D=args.D)
        for i, name in enumerate(strategy_names)
    }
    print(f"=== Strategies: {list(strategies)} ===")

    # Resume: seed strategy history from any results already on disk.
    n_preloaded = _preload_history(out_dir, strategies, rk)
    if n_preloaded:
        print(f"=== Resume: preloaded {n_preloaded} prior model results ===")

    total = 0
    for rd in range(args.rounds):
        print(f"\n=== Round {rd + 1}/{args.rounds} ===")
        # Proposal checkpoint: persist this round's configs so a resumed run reuses
        # the SAME configs (model_ids stay stable) instead of re-calling the LLM.
        proposals_path = out_dir / f"round_{rd:02d}_proposals.json"
        saved = None
        if proposals_path.exists():
            try:
                saved = json.loads(proposals_path.read_text())
            except Exception as e:
                # Corrupt/partial checkpoint (e.g. killed mid-write before atomic
                # writes landed). Don't crash-loop — discard and re-propose below.
                print(f"  [resume] proposals for round {rd} unreadable ({e}); re-proposing")
                saved = None
        if saved is not None:
            round_configs = [
                (
                    item["strategy"],
                    HPConfig(
                        **{
                            k: v
                            for k, v in item["hp"].items()
                            if k in HPConfig.__dataclass_fields__
                        }
                    ),
                )
                for item in saved
            ]
            print(f"  [resume] loaded {len(round_configs)} saved proposals for round {rd}")
        else:
            round_configs = []
            try:
                for name, strat in strategies.items():
                    n_per_strat = args.per_strategy_per_round
                    configs = strat.suggest(n_per_strat)
                    for c in configs:
                        round_configs.append((name, c))
                    print(f"  {name}: proposed {len(configs)} configs")
            except RateLimitExceeded as e:
                # FAIRNESS: never fall back to random. Checkpoint (completed models
                # are already on disk) and stop cleanly so this job can be resumed.
                print(f"\n=== RATE-LIMITED during proposal (round {rd}): {e} ===", flush=True)
                print(
                    f"=== Stopping cleanly. Re-launch the SAME command to resume "
                    f"(rounds 0..{rd - 1} done, history will be preloaded). ===",
                    flush=True,
                )
                sys.exit(42)
            _atomic_write_text(
                proposals_path,
                json.dumps(
                    [{"strategy": n_, "hp": _hp_to_dict(c)} for n_, c in round_configs], indent=2
                ),
            )

        # Train each config sequentially (single-GPU for now)
        # Track only models trained in THIS run; preloaded history already covers
        # results on disk, so re-adding them here would double-count.
        newly_trained: dict[str, tuple[list, list]] = {name: ([], []) for name in strategies}
        for i, (strat_name, hp) in enumerate(round_configs):
            model_id = f"r{rd:02d}_{strat_name}_{i:02d}"
            # Skip already-completed models (resume): meta file present = attempted.
            # But a meta from a DIFFERENT regime must not be counted as done — that
            # would leave a stale-regime result masquerading as current. The out-dir
            # guard above already blocks the common case; this is defense-in-depth.
            meta_path = out_dir / f"{model_id}_meta.json"
            if meta_path.exists():
                try:
                    prior_meta = json.loads(meta_path.read_text())
                except Exception:
                    prior_meta = {}
                if regime_key(prior_meta.get("regime")) != rk:
                    raise SystemExit(
                        f"{meta_path} exists under a different regime than the current run. "
                        f"Use a fresh --out_dir for this regime."
                    )
                # A meta counts as DONE only if it carries a real result (val_pearson).
                # An error-stub meta (written by the except branch on a transient
                # OOM/CUDA/preemption failure) must be RETRIED on resume, not treated
                # as complete — otherwise a single flake permanently drops that config.
                if "val_pearson" in prior_meta:
                    print(f"  [resume] skip {model_id} (already done)")
                    total += 1
                    continue
                print(
                    f"  [resume] retry {model_id} (prior attempt errored: "
                    f"{str(prior_meta.get('error', '?'))[:80]})"
                )
            print(
                f"\n  Training {model_id}: lr={hp.lr:.1e} bs={hp.batch_size} "
                f"layers={hp.n_layers} width={hp.width_base} opt={hp.optimizer}"
            )
            try:
                result = train_one_model(
                    hp,
                    train_seqs,
                    train_labels,
                    val_seqs,
                    val_labels,
                    test_seqs,
                    epochs=args.epochs,
                    use_compile=False,
                    early_stopping_patience=esp,
                    min_delta=min_delta,
                    extra_test_sets=all_test_sets,
                )
            except Exception as e:
                print(f"    ERROR: {e}")
                result = {
                    "hp": _hp_to_dict(hp),
                    "error": str(e),
                    "strategy": strat_name,
                    "round": rd,
                }
            result["model_id"] = model_id
            result["strategy"] = strat_name
            result["round"] = rd
            result["regime"] = regime
            _atomic_savez(
                out_dir / f"{model_id}.npz",
                **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
            )
            meta = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
            # meta written LAST + atomically: it is the resume gate (its presence
            # means "model done"), so it must never exist in a partial state and
            # must not appear before its companion .npz is fully on disk.
            _atomic_write_text(
                out_dir / f"{model_id}_meta.json", json.dumps(meta, indent=2, default=str)
            )
            # Log to global KB
            if result.get("val_mse") is not None:
                try:
                    from experiments.hp_knowledge_base import get_kb

                    get_kb().add(
                        hp=result.get("hp", {}),
                        val_metric=result.get("val_mse"),
                        context={
                            "D": args.D,
                            "ref_only": args.ref_only,
                            "epochs": args.epochs,
                            "strategy": strat_name,
                            "round": rd,
                        },
                    )
                except Exception as e:
                    print(f"    KB log failed: {e}", flush=True)
            if "val_pearson" in result:
                newly_trained[strat_name][0].append(hp)
                newly_trained[strat_name][1].append(result["val_pearson"])
            total += 1
            print(
                f"    val_pearson={result.get('val_pearson', 'ERR')}  time={result.get('train_time_sec', 0):.1f}s"
            )

        # Update each strategy with ONLY this run's newly-trained results
        # (preloaded history already accounts for results from prior runs).
        for name, strat in strategies.items():
            cs, vs = newly_trained[name]
            if cs:
                strat.update(cs, vs)

    # Final summary
    summary = {
        "strategies": list(strategies),
        "rounds": args.rounds,
        "per_strategy_per_round": args.per_strategy_per_round,
        "total_models": total,
        "D": args.D,
        "ref_only": args.ref_only,
        "epochs": args.epochs,
        "regime": regime,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== Done. {total} models trained. Run aggregate_ensemble.py on {out_dir} ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strategies",
        default="random",
        help="comma list, e.g. random,optuna_tpe,autoresearch_single",
    )
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--per_strategy_per_round", type=int, default=5)
    ap.add_argument("--D", type=int, default=10_000)
    ap.add_argument("--ref_only", action="store_true")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--hp_seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument(
        "--reservoir_cache",
        type=str,
        default=None,
        help="Path to a pre-generated reservoir-cache npz (sequences, oracle_labels) "
        "to use instead of chr_train_ref_only.npz. Used for non-genomic "
        "reservoir strategies (random, prm_*, evoaug_*, motif_planted, etc.). "
        "Generate with scripts/generate_reservoir_cache.py.",
    )
    ap.add_argument(
        "--chr_val",
        action="store_true",
        help="Use real chr19+21+X as the validation set (loads from chr_val_ref_only.npz) "
        "instead of a 10%% random holdout from train. Use this for genomic-based "
        "strategies so val matches the production protocol.",
    )
    ap.add_argument(
        "--reservoir_val_cache",
        type=str,
        default=None,
        help="Path to a held-out, transform-matched val npz (sequences, oracle_labels) "
        "for a non-genomic reservoir strategy — produced by applying the SAME transform "
        "to chr19/21/X backgrounds (scripts/generate_reservoir_cache.py --background_cache "
        "chr_val_ref_only.npz). Takes precedence over --chr_val / the 10%% holdout; "
        "subsampled to val_frac*D so val stays ~proportional to the train size.",
    )
    ap.add_argument(
        "--early_stop_patience",
        type=int,
        default=None,
        help="Override early stopping patience (default 15). "
        "Use lower (e.g. 5) for fair-budget fixed-cost scaling runs.",
    )
    ap.add_argument(
        "--min_delta",
        type=float,
        default=1e-3,
        help="Minimum val-metric improvement to reset the patience timer (default 1e-3). "
        "Filters noise-level wiggles so early stopping triggers on true plateaus.",
    )
    args = ap.parse_args()

    # Ray Tune scheduler engines (ray_asha/ray_bohb) own the whole trial loop, so
    # they cannot share a process with the round-based suggest/update strategies.
    # Require a Ray strategy to be the sole strategy for its out_dir.
    from experiments.ray_tune_search import RAY_STRATEGIES, run_ray_tune_search

    requested = [s.strip() for s in args.strategies.split(",") if s.strip()]
    ray_requested = [s for s in requested if s in RAY_STRATEGIES]
    if ray_requested:
        if len(requested) > 1:
            raise SystemExit(
                f"Ray scheduler strategies own the search loop and must be run alone "
                f"(one per out_dir). Got --strategies={args.strategies}."
            )
        run_ray_tune_search(args, ray_requested[0])
        return

    run_search(args)


if __name__ == "__main__":
    main()
