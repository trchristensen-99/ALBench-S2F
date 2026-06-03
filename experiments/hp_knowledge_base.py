"""HP Knowledge Base — accumulates HP→performance records across ALL experiments.

Every HP search run (any strategy, any D, any reservoir strategy) writes its
(hp_config, val_metric, context) tuples here. AutoResearchKnowledgeable reads it
to compute informed priors per HP dimension, filtered by context similarity.

Storage: outputs/hp_knowledge_base/records.jsonl (append-only)
Index:   outputs/hp_knowledge_base/index_summary.json (cached top-quartile summary)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

KB_PATH = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/hp_knowledge_base")
RECORDS_FILE = KB_PATH / "records.jsonl"


class HPKnowledgeBase:
    """Append-only HP record store + context-filtered summarizer."""

    def __init__(self, path: Path = KB_PATH):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.records_file = self.path / "records.jsonl"
        self._cache = None

    def add(self, hp: dict, val_metric: float, context: dict | None = None) -> None:
        """Record one HP→metric tuple. val_metric is val_MSE (lower=better)."""
        rec = {
            "hp": hp,
            "val_metric": float(val_metric),
            "context": context or {},
        }
        with open(self.records_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self._cache = None  # invalidate

    def load_all(self) -> list[dict]:
        """Load all records."""
        if self._cache is not None:
            return self._cache
        if not self.records_file.exists():
            return []
        records = []
        with open(self.records_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        self._cache = records
        return records

    def filter(self, **context_constraints) -> list[dict]:
        """Return records matching ALL listed context constraints.

        E.g., kb.filter(D=30000, strategy="autoresearch_batch")
        """
        out = []
        for r in self.load_all():
            match = True
            for k, v in context_constraints.items():
                if r.get("context", {}).get(k) != v:
                    match = False
                    break
            if match:
                out.append(r)
        return out

    def top_quartile(self, records: list[dict] | None = None, frac: float = 0.25) -> list[dict]:
        """Return top-`frac` of records by val_metric (lower better)."""
        if records is None:
            records = self.load_all()
        if not records:
            return []
        sorted_r = sorted(records, key=lambda r: r["val_metric"])
        n_top = max(1, int(len(sorted_r) * frac))
        return sorted_r[:n_top]

    def summary(self, records: list[dict] | None = None, frac: float = 0.25) -> dict:
        """Per-HP-dim characterization of the top quartile records.

        For numeric HPs: returns mean / median / IQR.
        For categorical HPs: returns frequency distribution.
        Used as informed prior for proposing new HPs.
        """
        top = self.top_quartile(records, frac)
        if not top:
            return {}

        # Inspect first hp dict to discover dims
        dims = set()
        for r in top:
            dims |= set(r.get("hp", {}).keys())

        out = {}
        for d in dims:
            vals = [r["hp"][d] for r in top if d in r["hp"]]
            if not vals:
                continue
            # Skip lists/dicts (e.g., width_jitter)
            if isinstance(vals[0], (list, dict)):
                continue
            if isinstance(vals[0], (int, float)) and not isinstance(vals[0], bool):
                arr = np.array(vals, dtype=float)
                out[d] = {
                    "type": "numeric",
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "q25": float(np.quantile(arr, 0.25)),
                    "q75": float(np.quantile(arr, 0.75)),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "n": len(vals),
                }
            else:
                # Categorical
                c = Counter(vals)
                out[d] = {
                    "type": "categorical",
                    "frequencies": {str(k): v for k, v in c.most_common()},
                    "n": len(vals),
                }
        return out

    def informed_sample(self, rng: np.random.Generator, hp_class,
                        context_filter: dict | None = None,
                        bias_strength: float = 0.7) -> dict:
        """Sample a new HP config biased toward top-quartile patterns.

        `bias_strength` ∈ [0,1]: 0 = pure random, 1 = pure top-quartile mode.
        Returns a dict with the same keys as HPConfig.
        """
        # Filter records by context, then summarize
        if context_filter:
            recs = self.filter(**context_filter)
        else:
            recs = self.load_all()
        if len(recs) < 5:
            # Not enough data — fall back to random
            return None
        summ = self.summary(recs)

        # Sample each dim
        from experiments.scaling_hp_search import sample_random_hp
        base = sample_random_hp(rng, seed=int(rng.integers(2**31)))
        new_hp = asdict(base)
        for d, info in summ.items():
            if rng.random() > bias_strength:
                continue  # use the random value
            if info["type"] == "numeric":
                # Sample within Q25-Q75 (interquartile)
                lo, hi = info["q25"], info["q75"]
                if hi > lo:
                    val = float(rng.uniform(lo, hi))
                    # Round for ints
                    if isinstance(new_hp[d], int):
                        val = int(round(val))
                    new_hp[d] = val
            else:
                # Categorical — weighted choice
                freqs = info["frequencies"]
                cats = list(freqs.keys())
                weights = np.array(list(freqs.values()), dtype=float)
                weights /= weights.sum()
                pick = rng.choice(len(cats), p=weights)
                v = cats[pick]
                # Cast back if int/bool
                if isinstance(new_hp[d], bool):
                    new_hp[d] = (v.lower() == "true")
                elif isinstance(new_hp[d], int):
                    try:
                        new_hp[d] = int(v)
                    except ValueError:
                        new_hp[d] = new_hp[d]  # keep base
                else:
                    new_hp[d] = v
        return new_hp


# Module-level singleton
_KB: HPKnowledgeBase | None = None


def get_kb() -> HPKnowledgeBase:
    global _KB
    if _KB is None:
        _KB = HPKnowledgeBase()
    return _KB
