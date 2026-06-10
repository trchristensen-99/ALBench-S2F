"""HP search strategies for the scaling-law experiment.

Each strategy implements:
  - suggest(n) -> List[HPConfig]   : propose n configs to try next
  - update(configs, val_pearsons)  : record results, refine internal state

Strategies:
  RandomStrategy         — uniform sampling (baseline)
  OptunaStrategy         — TPE Bayesian optimization (Optuna)
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

from experiments.scaling_hp_search import LR_SCHEDULE_CHOICES, HPConfig, sample_random_hp


class Strategy:
    """Base class for HP search strategies."""

    name: str = "base"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
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
        return [sample_random_hp(self.rng, seed=int(self.rng.integers(2**31))) for _ in range(n)]


class OptunaStrategy(Strategy):
    name = "optuna_tpe"

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        # Pending trials waiting for update()
        self._pending: list[Any] = []

    def _sample_one(self) -> tuple[Any, HPConfig]:
        import optuna

        trial = self.study.ask()
        n_layers = trial.suggest_int("n_layers", 2, 12)
        # width_jitter is sampled per-layer up to MAX. We use a fixed-dim suggest then truncate.
        max_layers = 12
        width_jitter = [
            trial.suggest_float(f"width_jitter_{i}", 0.5, 2.0) for i in range(max_layers)
        ][:n_layers]
        hp = HPConfig(
            lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024]),
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
                sample_random_hp(self.rng, seed=int(self.rng.integers(2**31))) for _ in range(n)
            ]
        # Sample around current top-3 best
        top = sorted(self.history, key=lambda x: x[1], reverse=True)[:3]
        configs = []
        for _ in range(n):
            base = top[self.rng.integers(0, len(top))][0]
            # Decide whether to explore (random restart) or exploit (perturb)
            if self.rng.random() < self.temperature:
                configs.append(sample_random_hp(self.rng, seed=int(self.rng.integers(2**31))))
            else:
                configs.append(self._perturb(base))
        return configs

    def _perturb(self, base: HPConfig) -> HPConfig:
        """Change `n_changes` HPs at random."""
        new = copy.deepcopy(base)
        # Map of HP name → perturbation function
        perturbations = {
            "lr": lambda v: float(np.clip(v * 10 ** self.rng.uniform(-0.5, 0.5), 1e-5, 1e-2)),
            "batch_size": lambda v: int(self.rng.choice([32, 64, 128, 256, 512, 1024])),
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
                sample_random_hp(self.rng, seed=int(self.rng.integers(2**31))) for _ in range(n)
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
            configs.append(sample_random_hp(self.rng, seed=int(self.rng.integers(2**31))))
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


class OptunaParallel(Strategy):
    """Optuna TPE in parallel-suggest mode (emits N configs per call).

    All N share the same history. Unlike the base OptunaStrategy which is
    one-config-at-a-time, this asks Optuna for N configs in batch before
    seeing any results. This causes some redundancy but maximizes parallelism.
    """

    name = "optuna_parallel"

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # Use multivariate TPE for better in-context learning
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=seed,
                multivariate=True,
                group=True,
                constant_liar=True,
            ),
        )
        self._pending: list = []

    def _sample_one(self):
        trial = self.study.ask()
        n_layers = trial.suggest_int("n_layers", 2, 12)
        max_layers = 12
        width_jitter = [
            trial.suggest_float(f"width_jitter_{i}", 0.5, 2.0) for i in range(max_layers)
        ][:n_layers]
        hp = HPConfig(
            lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024]),
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

    def update(self, configs, val_pearsons):
        super().update(configs, val_pearsons)
        from dataclasses import asdict

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
                configs.append(sample_random_hp(self.rng, seed=int(self.rng.integers(2**31))))
            else:
                # Construct HPConfig from dict, fix unsupported lists
                if not isinstance(kb_dict.get("width_jitter"), list):
                    n_l = kb_dict.get("n_layers", 6)
                    kb_dict["width_jitter"] = [1.0] * n_l
                try:
                    configs.append(HPConfig(**kb_dict))
                except Exception:
                    configs.append(sample_random_hp(self.rng, seed=int(self.rng.integers(2**31))))

        # Perturbations from session top
        if self.history:
            top = sorted(self.history, key=lambda x: x[1], reverse=True)[:3]
            for _ in range(n_perturb):
                base = top[self.rng.integers(0, len(top))][0]
                configs.append(self._perturb(base))
        else:
            for _ in range(n_perturb):
                configs.append(sample_random_hp(self.rng, seed=int(self.rng.integers(2**31))))

        # Pure random
        for _ in range(n_random):
            configs.append(sample_random_hp(self.rng, seed=int(self.rng.integers(2**31))))
        return configs


STRATEGY_REGISTRY = {
    "random": RandomStrategy,
    "optuna_tpe": OptunaStrategy,
    "optuna_parallel": OptunaParallel,
    # Evolutionary / hill-climbing perturbation strategies. These were historically
    # (mis)named "autoresearch_*"; "AutoResearch" is reserved for the LLM-iterative
    # search in llm_autoresearch.py. Old names are kept as aliases below for
    # backward-compat with existing launchers and on-disk *_meta.json.
    "evo_single": AutoResearchSingle,
    "evo_batch": AutoResearchBatch,
    "evo_explore": AutoResearchExplore,
    "evo_exploit": AutoResearchExploit,
    "evo_massive": AutoResearchMassive,
    "evo_adaptive": AutoResearchAdaptive,
    "evo_knowledgeable": AutoResearchKnowledgeable,
}

# Deprecated -> canonical aliases (resolve old strings transparently).
STRATEGY_ALIASES = {
    "autoresearch_single": "evo_single",
    "autoresearch_batch": "evo_batch",
    "autoresearch_explore": "evo_explore",
    "autoresearch_exploit": "evo_exploit",
    "autoresearch_massive": "evo_massive",
    "autoresearch_adaptive": "evo_adaptive",
    "autoresearch_knowledgeable": "evo_knowledgeable",
}


def canonical_strategy(name: str) -> str:
    """Map deprecated strategy names to their canonical form (evo_*)."""
    return STRATEGY_ALIASES.get(name, name)


def get_strategy(name: str, seed: int = 0) -> Strategy:
    name = canonical_strategy(name)
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name](seed=seed)
