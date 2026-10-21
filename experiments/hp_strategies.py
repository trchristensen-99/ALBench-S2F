"""HP search strategies for the scaling-law experiment.

Each strategy implements:
  - suggest(n) -> List[HPConfig]   : propose n configs to try next
  - update(configs, val_pearsons)  : record results, refine internal state

Strategies:
  RandomStrategy         — uniform sampling (baseline)
  OptunaStrategy         — TPE Bayesian optimization (Optuna)
  OptunaCmaEs            — CMA-ES over continuous dims (Optuna sampler)
  OptunaGP               — Gaussian-process BO (Optuna BoTorch GPSampler)
  OptunaQMC              — scrambled-Sobol space-filling baseline
  AutoResearchSingle     — single-HP-change around current best
  AutoResearchBatch      — K parallel HP changes around current best
  AutoResearchExplore    — high temperature, lots of variation
  AutoResearchExploit    — low temperature, refine current best

All strategies share the same HP space defined in scaling_hp_search.HPConfig.
"""

from __future__ import annotations

import copy
import random
from dataclasses import asdict
from typing import Any

import numpy as np

from experiments.scaling_hp_search import (
    LR_SCHEDULE_CHOICES,
    HPConfig,
    batch_size_menu,
    sample_random_hp,
)


class Strategy:
    """Base class for HP search strategies."""

    name: str = "base"

    def __init__(self, seed: int = 0, D: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.D = D
        self.history: list[tuple[HPConfig, float]] = []

    def suggest(self, n: int) -> list[HPConfig]:
        raise NotImplementedError

    def update(self, configs: list[HPConfig], val_pearsons: list[float]) -> None:
        for c, v in zip(configs, val_pearsons):
            if v is not None:
                self.history.append((c, v))

    def best(self) -> HPConfig | None:
        if not self.history:
            return None
        return max(self.history, key=lambda x: x[1])[0]


class RandomStrategy(Strategy):
    name = "random"

    def suggest(self, n: int) -> list[HPConfig]:
        return [
            sample_random_hp(self.rng, seed=int(self.rng.integers(2**31)), D=self.D)
            for _ in range(n)
        ]


class _OptunaSamplerBase(Strategy):
    """Shared Optuna ask/tell driver — subclasses only pick the sampler.

    All Optuna-backed strategies map a trial onto the same HPConfig search space
    and use identical ask/tell bookkeeping; they differ ONLY in `make_sampler`.
    """

    name = "optuna_base"

    def make_sampler(self, seed: int):
        raise NotImplementedError

    def __init__(self, seed: int = 0, D: int | None = None):
        super().__init__(seed, D=D)
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(
            direction="maximize",
            sampler=self.make_sampler(seed),
        )
        # Pending trials waiting for update()
        self._pending: list[Any] = []

    def _sample_one(self) -> tuple[Any, HPConfig]:
        trial = self.study.ask()
        n_layers = trial.suggest_int("n_layers", 2, 12)
        # width_jitter is sampled per-layer up to MAX. We use a fixed-dim suggest then truncate.
        max_layers = 12
        width_jitter = [
            trial.suggest_float(f"width_jitter_{i}", 0.5, 2.0) for i in range(max_layers)
        ][:n_layers]
        hp = HPConfig(
            lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", batch_size_menu(self.D)),
            conv_dropout=trial.suggest_float("conv_dropout", 0.0, 0.3),
            dense_dropout=trial.suggest_float("dense_dropout", 0.0, 0.5),
            n_layers=n_layers,
            width_base=trial.suggest_categorical("width_base", [16, 32, 64, 128, 256]),
            width_jitter=width_jitter,
            block_class=trial.suggest_categorical("block_class", ["eff", "ag", "plain"]),
            ks=trial.suggest_categorical("ks", [3, 5, 7, 9, 11]),
            pct_start=trial.suggest_categorical("pct_start", [0.1, 0.2, 0.3, 0.4]),
            optimizer=trial.suggest_categorical("optimizer", ["adam", "adamw", "muon"]),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            use_shift_aug=trial.suggest_categorical("use_shift_aug", [False, True]),
            shift_max=trial.suggest_categorical("shift_max", [5, 10, 15, 20]),
            use_evoaug=trial.suggest_categorical("use_evoaug", [False, True]),
            lr_schedule=trial.suggest_categorical("lr_schedule", LR_SCHEDULE_CHOICES),
            seed=int(self.rng.integers(2**31)),
        )
        return trial, hp

    def suggest(self, n: int) -> list[HPConfig]:
        configs = []
        for _ in range(n):
            trial, hp = self._sample_one()
            self._pending.append((trial, hp))
            configs.append(hp)
        return configs

    def update(self, configs: list[HPConfig], val_pearsons: list[float]) -> None:
        super().update(configs, val_pearsons)
        # Match pending trials with results by config equality
        new_pending = []
        for trial, hp in self._pending:
            matched = False
            for c, v in zip(configs, val_pearsons):
                if asdict(c) == asdict(hp) and v is not None:
                    self.study.tell(trial, v)
                    matched = True
                    break
            if not matched:
                new_pending.append((trial, hp))
        self._pending = new_pending


class OptunaStrategy(_OptunaSamplerBase):
    name = "optuna_tpe"

    def make_sampler(self, seed: int):
        import optuna

        return optuna.samplers.TPESampler(seed=seed)


class AutoResearchBase(Strategy):
    """Base AutoResearch: propose configs by perturbing the current best.

    Subclasses override `n_changes` and `temperature`.
    """

    n_changes: int = 1
    temperature: float = 0.2  # 0 = pure exploit; 1 = pure explore

    def suggest(self, n: int) -> list[HPConfig]:
        if not self.history:
            # No best yet — start with random
            return [
                sample_random_hp(self.rng, seed=int(self.rng.integers(2**31)), D=self.D)
                for _ in range(n)
            ]
        # Sample around current top-3 best
        top = sorted(self.history, key=lambda x: x[1], reverse=True)[:3]
        configs = []
        for _ in range(n):
            base = top[self.rng.integers(0, len(top))][0]
            # Decide whether to explore (random restart) or exploit (perturb)
            if self.rng.random() < self.temperature:
                configs.append(
                    sample_random_hp(self.rng, seed=int(self.rng.integers(2**31)), D=self.D)
                )
            else:
                configs.append(self._perturb(base))
        return configs

    def _perturb(self, base: HPConfig) -> HPConfig:
        """Change `n_changes` HPs at random."""
        new = copy.deepcopy(base)
        # Map of HP name → perturbation function
        perturbations = {
            "lr": lambda v: float(np.clip(v * 10 ** self.rng.uniform(-0.5, 0.5), 1e-5, 1e-2)),
            "batch_size": lambda v: int(self.rng.choice(batch_size_menu(self.D))),
            "conv_dropout": lambda v: float(np.clip(v + self.rng.uniform(-0.05, 0.05), 0.0, 0.3)),
            "dense_dropout": lambda v: float(np.clip(v + self.rng.uniform(-0.1, 0.1), 0.0, 0.5)),
            "n_layers": lambda v: int(np.clip(v + self.rng.choice([-2, -1, 0, 1, 2]), 2, 12)),
            "width_base": lambda v: int(self.rng.choice([16, 32, 64, 128, 256])),
            "width_jitter": lambda v: [
                float(np.clip(x * 2 ** self.rng.uniform(-0.5, 0.5), 0.5, 2.0)) for x in v
            ],
            "block_class": lambda v: str(self.rng.choice(["eff", "ag", "plain"])),
            "ks": lambda v: int(self.rng.choice([3, 5, 7, 9, 11])),
            "pct_start": lambda v: float(self.rng.choice([0.1, 0.2, 0.3, 0.4])),
            "optimizer": lambda v: str(self.rng.choice(["adam", "adamw", "muon"])),
            "weight_decay": lambda v: float(
                np.clip(v * 10 ** self.rng.uniform(-0.5, 0.5), 1e-6, 1e-2)
            ),
            "use_shift_aug": lambda v: not v,
            "shift_max": lambda v: int(self.rng.choice([5, 10, 15, 20])),
            "use_evoaug": lambda v: not v,
            "lr_schedule": lambda v: str(self.rng.choice(LR_SCHEDULE_CHOICES)),
        }
        hp_names = list(perturbations.keys())
        change_set = self.rng.choice(
            hp_names, size=min(self.n_changes, len(hp_names)), replace=False
        )
        for name in change_set:
            setattr(new, name, perturbations[name](getattr(new, name)))
        # If n_layers changed, resize width_jitter
        if len(new.width_jitter) != new.n_layers:
            if len(new.width_jitter) < new.n_layers:
                # Pad with 1.0
                new.width_jitter = list(new.width_jitter) + [1.0] * (
                    new.n_layers - len(new.width_jitter)
                )
            else:
                new.width_jitter = list(new.width_jitter)[: new.n_layers]
        new.seed = int(self.rng.integers(2**31))
        return new


class AutoResearchSingle(AutoResearchBase):
    name = "evo_single"
    n_changes = 1
    temperature = 0.2


class AutoResearchBatch(AutoResearchBase):
    name = "evo_batch"
    n_changes = 3
    temperature = 0.2


class AutoResearchExplore(AutoResearchBase):
    name = "evo_explore"
    n_changes = 5
    temperature = 0.5


class AutoResearchExploit(AutoResearchBase):
    name = "evo_exploit"
    n_changes = 1
    temperature = 0.05


class AutoResearchMassive(AutoResearchBase):
    """Massively parallel AutoResearch with history-aware exploration/exploitation.

    Behavior:
      - n_changes=5 per perturbation (multiple HPs change at once)
      - 30% pure random (explore), 70% perturb from top-3 history (exploit)
      - Avoids HP value ranges that appear consistently in bottom-quartile history
    """

    name = "evo_massive"
    n_changes = 5
    temperature = 0.3

    def _bottom_quartile_summary(self) -> dict:
        """Identify HP value ranges associated with poor performance."""
        if len(self.history) < 8:
            return {}
        sorted_h = sorted(self.history, key=lambda x: x[1])  # ascending (lower val_r = worse)
        bot_n = max(1, len(sorted_h) // 4)
        bottom = [c for c, _ in sorted_h[:bot_n]]
        avoid = {}
        # Look for HP values that appear in bottom but not in top
        from dataclasses import asdict

        for hp_name in [
            "lr",
            "batch_size",
            "n_layers",
            "width_base",
            "ks",
            "optimizer",
            "pct_start",
        ]:
            vals = [getattr(c, hp_name) for c in bottom if hasattr(c, hp_name)]
            if vals:
                avoid[hp_name] = vals
        return avoid

    def suggest(self, n: int) -> list[HPConfig]:
        if not self.history:
            return [
                sample_random_hp(self.rng, seed=int(self.rng.integers(2**31)), D=self.D)
                for _ in range(n)
            ]
        configs = []
        avoid = self._bottom_quartile_summary()
        top = sorted(self.history, key=lambda x: x[1], reverse=True)[:3]
        n_explore = int(n * self.temperature)
        n_exploit = n - n_explore
        # Exploit: perturb from top
        for _ in range(n_exploit):
            base = top[self.rng.integers(0, len(top))][0]
            cand = self._perturb(base)
            # Reject if it matches a bottom-quartile pattern (for numeric HPs)
            tries = 0
            while tries < 5 and self._looks_like_bottom(cand, avoid):
                cand = self._perturb(base)
                tries += 1
            configs.append(cand)
        # Explore: pure random
        for _ in range(n_explore):
            configs.append(sample_random_hp(self.rng, seed=int(self.rng.integers(2**31)), D=self.D))
        return configs

    def _looks_like_bottom(self, cfg: "HPConfig", avoid: dict) -> bool:
        """Heuristic: avoid if too many HP values match bottom-quartile."""
        if not avoid:
            return False
        matches = 0
        for hp_name, bad_vals in avoid.items():
            if not bad_vals:
                continue
            cur = getattr(cfg, hp_name, None)
            if cur is None:
                continue
            if isinstance(cur, (int, str, bool)):
                if cur in bad_vals:
                    matches += 1
            elif isinstance(cur, float):
                # Numeric: check if within log-1e ratio of any bad value
                import math

                for bv in bad_vals:
                    if bv > 0 and cur > 0:
                        try:
                            if abs(math.log10(cur) - math.log10(bv)) < 0.3:
                                matches += 1
                                break
                        except Exception:
                            pass
        return matches >= 3  # avoid if 3+ HPs match bottom-quartile


class OptunaParallel(_OptunaSamplerBase):
    """Optuna TPE in parallel-suggest mode (emits N configs per call).

    All N share the same history. Unlike the base OptunaStrategy which is
    one-config-at-a-time, this asks Optuna for N configs in batch before
    seeing any results. This causes some redundancy but maximizes parallelism.
    Multivariate + constant_liar give better batch in-context learning.
    """

    name = "optuna_parallel"

    def make_sampler(self, seed: int):
        import optuna

        return optuna.samplers.TPESampler(
            seed=seed,
            multivariate=True,
            group=True,
            constant_liar=True,
        )


class OptunaCmaEs(_OptunaSamplerBase):
    """CMA-ES: covariance-matrix evolution over the CONTINUOUS HP dims.

    Strong on correlated continuous HPs (lr × weight_decay × dropouts), where the
    one-HP-at-a-time evo_* family is weakest. CMA-ES cannot model categoricals, so
    Optuna samples those (block_class, optimizer, schedules, …) with an independent
    sampler; `with_margin=True` handles the integer/stepped dims.
    """

    name = "optuna_cmaes"

    def make_sampler(self, seed: int):
        import optuna

        return optuna.samplers.CmaEsSampler(
            seed=seed, with_margin=True, warn_independent_sampling=False
        )


class OptunaGP(_OptunaSamplerBase):
    """Gaussian-process Bayesian optimization (Optuna's BoTorch-backed GPSampler).

    A true GP surrogate rather than TPE's density ratio — more sample-efficient at
    the small (200-model) budgets that matter here, at higher per-suggest cost.
    """

    name = "optuna_gp"

    def make_sampler(self, seed: int):
        import optuna

        return optuna.samplers.GPSampler(seed=seed)


class OptunaQMC(_OptunaSamplerBase):
    """Quasi-Monte-Carlo (scrambled Sobol) — a low-discrepancy space-filling baseline.

    Covers the space more evenly than i.i.d. `random`, so it is the fairer
    "no-model" floor for judging whether the adaptive methods actually learn.
    """

    name = "optuna_qmc"

    def make_sampler(self, seed: int):
        import optuna

        return optuna.samplers.QMCSampler(seed=seed, scramble=True)


class AutoResearchAdaptive(AutoResearchMassive):
    """Massive + stronger adaptive: also tracks per-HP-dim trends.

    On top of AutoResearchMassive, computes a per-dimension correlation
    between HP value and val performance. Biases sampling toward
    high-performing regions (e.g., 'lr above 1e-3 has been better — sample more there').
    """

    name = "evo_adaptive"
    n_changes = 4
    temperature = 0.25


class AutoResearchKnowledgeable(AutoResearchAdaptive):
    """AutoResearch with informed prior from the global HP knowledge base.

    On each suggest():
      1. Reads aggregated history from KB (filtered by current context)
      2. ~50% of proposals biased toward top-quartile HP regions
      3. ~30% perturbations from current session top
      4. ~20% pure random for exploration
    """

    name = "evo_knowledgeable"
    n_changes = 4

    def __init__(self, seed: int = 0, context_filter: dict | None = None):
        super().__init__(seed)
        from experiments.hp_knowledge_base import get_kb

        self.kb = get_kb()
        self.context_filter = context_filter or {}

    def suggest(self, n: int) -> list[HPConfig]:
        from dataclasses import asdict

        configs = []
        n_kb = int(n * 0.5)
        n_perturb = int(n * 0.3)
        n_random = n - n_kb - n_perturb

        # KB-biased samples (informed prior)
        for _ in range(n_kb):
            kb_dict = self.kb.informed_sample(
                self.rng, HPConfig, context_filter=self.context_filter, bias_strength=0.7
            )
            if kb_dict is None:
                # Not enough data, fall back to random
                configs.append(
                    sample_random_hp(self.rng, seed=int(self.rng.integers(2**31)), D=self.D)
                )
            else:
                # Construct HPConfig from dict, fix unsupported lists
                if not isinstance(kb_dict.get("width_jitter"), list):
                    n_l = kb_dict.get("n_layers", 6)
                    kb_dict["width_jitter"] = [1.0] * n_l
                try:
                    configs.append(HPConfig(**kb_dict))
                except Exception:
                    configs.append(
                        sample_random_hp(self.rng, seed=int(self.rng.integers(2**31)), D=self.D)
                    )

        # Perturbations from session top
        if self.history:
            top = sorted(self.history, key=lambda x: x[1], reverse=True)[:3]
            for _ in range(n_perturb):
                base = top[self.rng.integers(0, len(top))][0]
                configs.append(self._perturb(base))
        else:
            for _ in range(n_perturb):
                configs.append(
                    sample_random_hp(self.rng, seed=int(self.rng.integers(2**31)), D=self.D)
                )

        # Pure random
        for _ in range(n_random):
            configs.append(sample_random_hp(self.rng, seed=int(self.rng.integers(2**31)), D=self.D))
        return configs


# ---------------------------------------------------------------------------------------------
# Prior-matched classical baselines (Oct-20). The cadence "LLM vs classical" comparison could not say
# whether the LLM's edge is its PRIOR (what it tends to propose) or its SEARCH (adapting to feedback):
# the classical arm searched the full BROAD box with no prior, at 10 trials per sampler. These two arms
# take the prior away from the LLM and give it to classical search:
#   prior_marginal    — i.i.d. draws from the EMPIRICAL per-axis marginals of a reference set of LLM
#                       proposals (no feedback at all) => the prior alone
#   optuna_prior_tpe  — TPE restricted to the LLM-informed box (observed levels / observed ranges, the
#                       axes the LLM never varies fixed at their defaults) => prior + classical search
# Reference set: JSON list of hp dicts at $PRIOR_REF_JSON (see scratchpad/make_prior_ref.py).
# ---------------------------------------------------------------------------------------------
def _prior_ref() -> list[dict]:
    import json
    import os

    path = os.environ.get("PRIOR_REF_JSON", "")
    if not path or not os.path.exists(path):
        raise RuntimeError("PRIOR_REF_JSON must point at a JSON list of reference LLM-proposal hp dicts")
    ref = json.load(open(path))
    if len(ref) < 20:
        raise RuntimeError(f"prior reference too small ({len(ref)}); need >= 20 proposals")
    return ref


class PriorMarginalRandom(Strategy):
    """Random search whose per-axis marginals are the empirical marginals of LLM proposals."""

    name = "prior_marginal"

    def __init__(self, seed: int = 0, D: int | None = None):
        super().__init__(seed, D=D)
        self.ref = _prior_ref()

    def _pick(self, key, default=None):
        c = self.ref[int(self.rng.integers(len(self.ref)))]
        return c.get(key, default)

    def _one(self) -> HPConfig:
        r = self.rng
        n_layers = int(self._pick("n_layers"))
        donors = [c for c in self.ref if int(c["n_layers"]) == n_layers]
        wj = [float(x) for x in donors[int(r.integers(len(donors)))]["width_jitter"]]
        donor = self.ref[int(r.integers(len(self.ref)))]
        extra = {k: v for k, v in (donor.get("extra") or {}).items()}
        jitter = lambda v: float(v) * float(10 ** r.uniform(-0.15, 0.15))  # noqa: E731
        return HPConfig(
            lr=jitter(self._pick("lr")),
            batch_size=int(self._pick("batch_size")),
            conv_dropout=float(max(0.0, self._pick("conv_dropout") + r.normal(0, 0.02))),
            dense_dropout=float(max(0.0, self._pick("dense_dropout") + r.normal(0, 0.02))),
            n_layers=n_layers,
            width_base=int(self._pick("width_base")),
            width_jitter=wj,
            block_class=str(self._pick("block_class")),
            ks=int(self._pick("ks")),
            pct_start=float(self._pick("pct_start")),
            optimizer=str(self._pick("optimizer")),
            weight_decay=jitter(self._pick("weight_decay")),
            use_shift_aug=True,
            shift_max=int(self._pick("shift_max")),
            use_evoaug=False,
            lr_schedule=str(self._pick("lr_schedule", "plateau")),
            width_ratio=float(self._pick("width_ratio", 1.0)),
            pool_downsample=int(self._pick("pool_downsample", 0)),
            gradient_clip=float(self._pick("gradient_clip", 0.0)),
            seed=int(r.integers(2**31)),
            extra=extra,
        )

    def suggest(self, n: int) -> list[HPConfig]:
        return [self._one() for _ in range(n)]


class OptunaPriorTPE(_OptunaSamplerBase):
    """TPE inside the LLM-informed box; axes the LLM never varies are held at their defaults."""

    name = "optuna_prior_tpe"

    def make_sampler(self, seed: int):
        import os

        import optuna

        return optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=int(os.environ.get("PRIOR_TPE_STARTUP", "10"))
        )

    def __init__(self, seed: int = 0, D: int | None = None):
        self.ref = _prior_ref()
        super().__init__(seed, D=D)

    def _lv(self, key, default=None):
        """Levels the LLM actually uses (>= 5% of its proposals); a level it tried once is not its prior."""
        cnt: dict = {}
        for c in self.ref:
            v = c.get(key, default)
            cnt[v] = cnt.get(v, 0) + 1
        keep = [v for v, k in cnt.items() if k >= 0.05 * len(self.ref)]
        return sorted(keep or list(cnt), key=str)

    def _sample_one(self) -> tuple[Any, HPConfig]:
        trial = self.study.ask()
        ref = self.ref
        nl = [int(c["n_layers"]) for c in ref]
        n_layers = trial.suggest_int("n_layers", min(nl), max(nl))
        wj = [trial.suggest_float(f"width_jitter_{i}", 0.5, 2.0) for i in range(max(nl))][:n_layers]
        rng_ = lambda key: (  # noqa: E731
            float(np.quantile([c[key] for c in ref], 0.02)),
            float(np.quantile([c[key] for c in ref], 0.98)),
        )
        lo, hi = rng_("lr")
        wlo, whi = rng_("weight_decay")
        cd = [c["conv_dropout"] for c in ref]
        dd = [c["dense_dropout"] for c in ref]
        extra: dict = {}
        acts = [a for a in {(c.get("extra") or {}).get("activation") for c in ref} if a]
        if acts:
            extra["activation"] = trial.suggest_categorical("activation", sorted(acts) + ["silu"])
        return trial, HPConfig(
            lr=trial.suggest_float("lr", lo, hi, log=True),
            batch_size=trial.suggest_categorical("batch_size", self._lv("batch_size")),
            conv_dropout=trial.suggest_float("conv_dropout", float(min(cd)), float(max(cd))),
            dense_dropout=trial.suggest_float("dense_dropout", float(min(dd)), float(max(dd))),
            n_layers=n_layers,
            width_base=trial.suggest_categorical("width_base", self._lv("width_base")),
            width_jitter=wj,
            block_class=trial.suggest_categorical("block_class", self._lv("block_class")),
            ks=trial.suggest_categorical("ks", self._lv("ks")),
            pct_start=trial.suggest_categorical("pct_start", self._lv("pct_start")),
            optimizer=trial.suggest_categorical("optimizer", self._lv("optimizer")),
            weight_decay=trial.suggest_float("weight_decay", wlo, whi, log=True),
            use_shift_aug=True,
            shift_max=trial.suggest_categorical("shift_max", self._lv("shift_max")),
            use_evoaug=False,
            lr_schedule=trial.suggest_categorical("lr_schedule", self._lv("lr_schedule", "plateau")),
            seed=int(self.rng.integers(2**31)),
            extra=extra,
        )


STRATEGY_REGISTRY = {
    "random": RandomStrategy,
    "optuna_tpe": OptunaStrategy,
    "optuna_parallel": OptunaParallel,
    "optuna_cmaes": OptunaCmaEs,
    "optuna_gp": OptunaGP,
    "optuna_qmc": OptunaQMC,
    "prior_marginal": PriorMarginalRandom,
    "optuna_prior_tpe": OptunaPriorTPE,
    # Evolutionary / hill-climbing perturbation strategies. These were historically
    # (mis)named "autoresearch_*"; "AutoResearch" is reserved for the LLM-iterative
    # search in llm_autoresearch.py. Only the evo_* names are accepted now.
    "evo_single": AutoResearchSingle,
    "evo_batch": AutoResearchBatch,
    "evo_explore": AutoResearchExplore,
    "evo_exploit": AutoResearchExploit,
    "evo_massive": AutoResearchMassive,
    "evo_adaptive": AutoResearchAdaptive,
    "evo_knowledgeable": AutoResearchKnowledgeable,
}


def get_strategy(name: str, seed: int = 0, D: int | None = None) -> Strategy:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY)}")
    cls = STRATEGY_REGISTRY[name]
    try:
        return cls(seed=seed, D=D)
    except TypeError:
        # Back-compat: strategies that don't yet accept D in __init__ (e.g. LLM
        # AutoResearch variants). Set D post-hoc so code reading self.D works.
        inst = cls(seed=seed)
        inst.D = D
        return inst
