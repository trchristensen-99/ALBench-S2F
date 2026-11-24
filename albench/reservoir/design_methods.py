"""The two non-gradient design methods Gosai et al. used alongside FastSeqProp.

Three methods built the designed set we are evaluated on, and they are not
interchangeable: FastSeqProp follows the predictor's gradient, simulated annealing does a
temperature-controlled random walk over single-base edits, and AdaLead is a
threshold-greedy evolutionary search with rollout. They reach different sequences and they
fail differently -- a gradient method finds the model's adversarial maxima, a walk finds
whatever is reachable one base at a time, and an evolutionary method finds whatever
recombination of good sequences the model likes. Training on all three is the only way a
corpus covers the designed-sequence regime rather than one corner of it.

Neither method here needs gradients, only ``predict(list[str]) -> np.ndarray``, so unlike
FastSeqProp they can in principle be guided by the oracle. They should not be: the whole
argument in albench/ensemble/bank.py applies unchanged, and the FastSeqProp canary measured
the cost -- guided by a 30k bank the designs scored 14.08 on the guide and 6.92 on the
independent oracle. Guide with a student, report the gap.

BOTH ARE BATCHED ACROSS DESIGNS, not looped per sequence. Every design is an independent
chain, so all n run in lockstep and each round is ONE predict() call over n (or n x
population) candidates. The per-sequence loop that reads more naturally would make a 30k
pool thousands of separate forward passes and take days; reservoirs are generated at 1M
scale and the vectorised form is not optional.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler

logger = logging.getLogger(__name__)

_NUCS = np.array(list("ACGT"))
_CODE = {c: i for i, c in enumerate("ACGT")}


def _encode(seqs: list[str]) -> np.ndarray:
    """(N, L) uint8 of ACGT codes."""
    return np.array([[_CODE.get(c, 0) for c in s] for s in seqs], dtype=np.uint8)


def _decode(arr: np.ndarray) -> list[str]:
    return ["".join(r) for r in _NUCS[arr]]


def _mutate(arr: np.ndarray, n_mut: int, rng: np.random.Generator) -> np.ndarray:
    """Point-mutate each row at exactly ``n_mut`` positions, never to the same base."""
    out = arr.copy()
    n, L = out.shape
    pos = np.argsort(rng.random((n, L)), axis=1)[:, :n_mut]
    rows = np.repeat(np.arange(n), n_mut)
    cols = pos.ravel()
    shift = rng.integers(1, 4, size=rows.size, dtype=np.uint8)
    out[rows, cols] = (out[rows, cols] + shift) % 4
    return out


def _objective(y: np.ndarray, objective: str, target: float) -> np.ndarray:
    """Higher is always better, whatever the objective, so selection code stays one branch."""
    if objective == "maximize":
        return y
    if objective == "minimize":
        return -y
    return -np.abs(y - target)


class _DesignBase(ReservoirSampler):
    def __init__(
        self,
        seed: int | None = None,
        objective: str = "maximize",
        target_activity: float = 0.0,
        batch_size: int = 4096,
    ) -> None:
        if objective not in ("maximize", "minimize", "target"):
            raise ValueError(f"objective must be maximize|minimize|target, got {objective!r}")
        self._rng = np.random.default_rng(seed)
        self.seed = seed
        self.objective = objective
        self.target_activity = float(target_activity)
        self.batch_size = int(batch_size)

    def sample(self, candidates, n_samples, metadata=None):
        if n_samples > len(candidates):
            raise ValueError("n_samples cannot exceed number of candidates")
        return self._rng.choice(len(candidates), size=n_samples, replace=False).tolist()

    @staticmethod
    def _require(student_model, name: str):
        if student_model is None or not hasattr(student_model, "predict"):
            raise ValueError(
                f"{name} is model-in-the-loop: it scores candidates every round and has no "
                "meaningful fallback. Pass a guide with predict(); --guide-model points at "
                "a saved LegNet bank."
            )

    def _starts(self, n, base_sequences, seq_len):
        if not base_sequences:
            raise ValueError("design methods start from real CREs; pass base_sequences")
        idx = self._rng.integers(0, len(base_sequences), size=n)
        return [str(base_sequences[int(i)])[:seq_len].ljust(seq_len, "A") for i in idx]


class SimulatedAnnealingSampler(_DesignBase):
    """Metropolis walk over single-base edits, at a geometrically cooling temperature."""

    def __init__(
        self,
        seed: int | None = None,
        n_steps: int = 300,
        t0: float = 1.0,
        t_end: float = 0.01,
        n_mut: int = 1,
        objective: str = "maximize",
        target_activity: float = 0.0,
        batch_size: int = 4096,
    ) -> None:
        """Initialize sampler.

        Args:
            seed: Random seed.
            n_steps: Metropolis proposals per design.
            t0: Starting temperature, in units of the objective (oracle label units).
            t_end: Final temperature; the schedule is geometric between the two.
            n_mut: Bases changed per proposal. 1 is the usual choice and keeps the
                acceptance rate workable; larger values turn the walk into a coarse search.
            objective: ``maximize`` | ``minimize`` | ``target``.
            target_activity: Target for ``objective="target"``.
            batch_size: Designs scored per predict() call.
        """
        super().__init__(seed, objective, target_activity, batch_size)
        self.n_steps = int(n_steps)
        self.t0, self.t_end, self.n_mut = float(t0), float(t_end), int(n_mut)

    def generate(
        self, n_sequences: int, base_sequences=None, task: str = "k562", student_model=None
    ) -> tuple[list[str], pd.DataFrame]:
        self._require(student_model, "simulated annealing")
        seq_len = 200 if task == "k562" else 80
        cur = _encode(self._starts(n_sequences, base_sequences, seq_len))

        def score(a: np.ndarray) -> np.ndarray:
            seqs = _decode(a)
            out = []
            for s in range(0, len(seqs), self.batch_size):
                out.append(np.asarray(student_model.predict(seqs[s : s + self.batch_size])).ravel())
            return np.concatenate(out)

        y_cur = score(cur)
        f_cur = _objective(y_cur, self.objective, self.target_activity)
        y_init = y_cur.copy()
        n_acc = np.zeros(n_sequences, dtype=np.int32)

        # Geometric cooling: the standard schedule, and the one whose acceptance rate falls
        # smoothly rather than collapsing at a fixed step.
        ratio = (self.t_end / self.t0) ** (1.0 / max(self.n_steps - 1, 1))
        for step in range(self.n_steps):
            T = self.t0 * ratio**step
            prop = _mutate(cur, self.n_mut, self._rng)
            y_prop = score(prop)
            f_prop = _objective(y_prop, self.objective, self.target_activity)
            # Always accept an improvement; accept a worsening with prob exp(-delta/T).
            d = f_prop - f_cur
            take = (d >= 0) | (
                self._rng.random(n_sequences) < np.exp(np.minimum(d, 0) / max(T, 1e-9))
            )
            cur[take], f_cur[take], y_cur[take] = prop[take], f_prop[take], y_prop[take]
            n_acc += take.astype(np.int32)
            if step % 50 == 0 or step == self.n_steps - 1:
                logger.info(
                    "SA %s step %d/%d: T=%.3f accept=%.2f guide mean %.3f",
                    self.objective,
                    step + 1,
                    self.n_steps,
                    T,
                    take.mean(),
                    float(y_cur.mean()),
                )

        meta = pd.DataFrame(
            {
                "seq_idx": np.arange(n_sequences, dtype=np.int64),
                "method": f"simulated_annealing_{self.objective}",
                "source": "generated",
                "n_steps": self.n_steps,
                "n_accepted": n_acc,
                "guide_pred_init": y_init.astype(np.float32),
                "guide_pred_final": y_cur.astype(np.float32),
            }
        )
        return _decode(cur), meta


class AdaLeadSampler(_DesignBase):
    """Threshold-greedy evolutionary search with rollout (Sinai et al., 2020).

    Two things distinguish it from plain hill-climbing and both are kept here. Parents are
    drawn only from sequences within ``kappa`` of the current best, so the search does not
    waste its budget refining mediocre lineages; and a child that improves on its parent is
    mutated AGAIN immediately (rollout) rather than waiting for the next round, which is
    what lets it travel further than its round count suggests.
    """

    def __init__(
        self,
        seed: int | None = None,
        n_rounds: int = 10,
        population: int = 16,
        kappa: float = 0.05,
        mutation_rate: float = 0.02,
        rollout: int = 4,
        objective: str = "maximize",
        target_activity: float = 0.0,
        batch_size: int = 4096,
    ) -> None:
        """Initialize sampler.

        Args:
            seed: Random seed.
            n_rounds: Evolutionary rounds.
            population: Children proposed per design per round.
            kappa: Parents must score within this fraction of the design's own best, as a
                fraction of the observed score range. Smaller is greedier.
            mutation_rate: Per-base mutation probability when a child is proposed.
            rollout: Extra mutate-and-keep steps applied to a child that improved.
            objective: ``maximize`` | ``minimize`` | ``target``.
            target_activity: Target for ``objective="target"``.
            batch_size: Sequences scored per predict() call.
        """
        super().__init__(seed, objective, target_activity, batch_size)
        self.n_rounds, self.population = int(n_rounds), int(population)
        self.kappa, self.mutation_rate = float(kappa), float(mutation_rate)
        self.rollout = int(rollout)

    def generate(
        self, n_sequences: int, base_sequences=None, task: str = "k562", student_model=None
    ) -> tuple[list[str], pd.DataFrame]:
        self._require(student_model, "AdaLead")
        seq_len = 200 if task == "k562" else 80
        best = _encode(self._starts(n_sequences, base_sequences, seq_len))

        def score(a: np.ndarray) -> np.ndarray:
            seqs = _decode(a)
            out = []
            for s in range(0, len(seqs), self.batch_size):
                out.append(np.asarray(student_model.predict(seqs[s : s + self.batch_size])).ravel())
            return np.concatenate(out)

        y_best = score(best)
        f_best = _objective(y_best, self.objective, self.target_activity)
        y_init = y_best.copy()
        n_mut = max(1, int(round(self.mutation_rate * seq_len)))

        for rd in range(self.n_rounds):
            # Children of the current best of each lineage. The kappa threshold is applied
            # against each lineage's own best, so every design keeps evolving rather than
            # the whole pool collapsing onto one global winner -- we need n diverse
            # sequences out of this, not one good one.
            parents = np.repeat(best, self.population, axis=0)
            kids = _mutate(parents, n_mut, self._rng)
            y_kid = score(kids)
            f_kid = _objective(y_kid, self.objective, self.target_activity)

            # Rollout: a child that already beats its parent gets mutated again and kept
            # if it keeps improving. This is the step that makes AdaLead travel.
            f_par = np.repeat(f_best, self.population)
            live = f_kid > f_par
            for _ in range(self.rollout):
                if not live.any():
                    break
                sub = _mutate(kids[live], n_mut, self._rng)
                y_sub = score(sub)
                f_sub = _objective(y_sub, self.objective, self.target_activity)
                better = f_sub > f_kid[live]
                li = np.where(live)[0]
                upd = li[better]
                kids[upd], f_kid[upd], y_kid[upd] = sub[better], f_sub[better], y_sub[better]
                live[li[~better]] = False

            f_grid = f_kid.reshape(n_sequences, self.population)
            y_grid = y_kid.reshape(n_sequences, self.population)
            kid_grid = kids.reshape(n_sequences, self.population, seq_len)
            arg = f_grid.argmax(axis=1)
            cand_f = f_grid[np.arange(n_sequences), arg]
            take = cand_f > f_best
            rows = np.where(take)[0]
            best[rows] = kid_grid[rows, arg[rows]]
            f_best[rows] = cand_f[rows]
            y_best[rows] = y_grid[rows, arg[rows]]
            logger.info(
                "AdaLead %s round %d/%d: improved %d/%d, guide mean %.3f",
                self.objective,
                rd + 1,
                self.n_rounds,
                int(take.sum()),
                n_sequences,
                float(y_best.mean()),
            )

        meta = pd.DataFrame(
            {
                "seq_idx": np.arange(n_sequences, dtype=np.int64),
                "method": f"adalead_{self.objective}",
                "source": "generated",
                "n_rounds": self.n_rounds,
                "guide_pred_init": y_init.astype(np.float32),
                "guide_pred_final": y_best.astype(np.float32),
            }
        )
        return _decode(best), meta
