"""FastSeqProp: design sequences by gradient ascent on the predictor, not by mutation.

WHY THIS AND NOT ANOTHER ise_ VARIANT. Gosai et al. designed their high-activity oligos
with three methods -- FastSeqProp, Simulated Annealing and AdaLead -- and the designed set
they produced is one of our evaluation corpora. Every model-guided reservoir we have is
evolutionary: mutate, score, select, repeat. That explores by sampling, so it only ever
visits sequences within a few mutations of something it has already scored. A gradient
method optimises the input directly and reaches a different part of the space, so
"model-guided design" and "in-silico evolution" are not interchangeable arms, and if we
are going to be evaluated on a designed set it is worth training on sequences built the
way that set was.

HOW IT DIFFERS FROM NAIVE GRADIENT DESIGN, which is the whole content of the paper:

  * The logits are INSTANCE-NORMALISED before the softmax at every step. Without it the
    softmax saturates within a few dozen steps, the gradient vanishes, and optimisation
    stalls at whatever it had reached -- the failure the paper exists to fix.
  * The forward pass sees a HARD one-hot, with gradients taken through the soft
    relaxation (straight-through). This matters more than it looks: optimise on soft
    inputs and the result is a sequence the model scores highly only as a probability
    distribution, which cannot be synthesised. We score what will actually be ordered.

GUIDED BY A STUDENT, NOT THE ORACLE, for the reason given in albench/ensemble/bank.py: a
pool selected at the labelling ensemble's own argmax is an upper bound no collaborator can
reproduce. Here the argument is stronger than for the evolutionary arms, because gradient
ascent is far better at finding a model's adversarial maxima than mutation is -- guided by
the oracle, this would produce the oracle's adversarial examples and then label them with
the oracle, which scores them as triumphs. That arm is worth running once as a diagnostic;
it is not the arm to put in a library.

WHAT TO WATCH. Gradient ascent from similar starts converges to similar sequences, so the
duplicate and diversity statistics are the first thing to read on the resulting pool, not
an afterthought -- ``build_pool.py`` already warns below 99% unique. ``init="pool"``
starts each design from a different real CRE, which is both more diverse and closer to
what a collaborator would do than starting from noise.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler

logger = logging.getLogger(__name__)

_NUCS = np.array(list("ACGT"))


class FastSeqPropSampler(ReservoirSampler):
    """Gradient-based sequence design against a differentiable student bank."""

    def __init__(
        self,
        seed: int | None = None,
        n_steps: int = 150,
        lr: float = 0.1,
        batch_size: int = 512,
        objective: str = "maximize",
        target_activity: float = 0.0,
        guide_members: int | None = 2,
        init: str = "pool",
        tau: float = 1.0,
        entropy_weight: float = 0.0,
    ) -> None:
        """Initialize sampler.

        Args:
            seed: Random seed.
            n_steps: Adam steps per batch. The paper reports convergence by ~100-200 on
                comparable models; the cost is linear in this and it is the first knob to
                cut if the arm is too slow.
            lr: Adam learning rate on the logits.
            batch_size: Sequences optimised simultaneously. Each is an independent
                optimisation -- there is no interaction between rows -- so this trades
                memory for wall-clock and changes nothing about the result.
            objective: ``maximize`` | ``minimize`` | ``target``. The same three the
                ise_ arms expose, so the two design families are comparable
                objective-for-objective rather than only at the maximum.
            target_activity: Target for ``objective="target"``, in oracle label units.
            guide_members: Bank members to backprop through, or None for all. Two is
                usually enough to optimise against and keeps the cost near a single
                model's; the resulting pool is still LABELLED by the full oracle.
            init: ``pool`` starts each design from a real CRE (more diverse, and closer
                to how a collaborator would use this), ``random`` from uniform logits.
            tau: Softmax temperature. Below 1 sharpens toward one-hot sooner.
            entropy_weight: Optional penalty on per-position entropy, added to keep the
                distribution from collapsing before the model has shaped it. 0 disables.
        """
        if objective not in ("maximize", "minimize", "target"):
            raise ValueError(f"objective must be maximize|minimize|target, got {objective!r}")
        if init not in ("pool", "random"):
            raise ValueError(f"init must be 'pool' or 'random', got {init!r}")
        self._rng = np.random.default_rng(seed)
        self.seed = seed
        self.n_steps = int(n_steps)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.objective = objective
        self.target_activity = float(target_activity)
        self.guide_members = guide_members
        self.init = init
        self.tau = float(tau)
        self.entropy_weight = float(entropy_weight)

    def sample(self, candidates, n_samples, metadata=None):
        if n_samples > len(candidates):
            raise ValueError("n_samples cannot exceed number of candidates")
        return self._rng.choice(len(candidates), size=n_samples, replace=False).tolist()

    # ------------------------------------------------------------------ guide

    @staticmethod
    def _torch_models(student_model, guide_members):
        """Pull differentiable modules out of the guide, or say why it cannot be used."""
        models = getattr(student_model, "models", None)
        if not models:
            raise ValueError(
                "FastSeqProp needs a DIFFERENTIABLE guide -- it takes gradients with "
                "respect to the input one-hot -- and the object it was handed exposes no "
                "`.models`. A LegNetBank works (albench.ensemble.bank); the AlphaGenome "
                "oracle does not, because it is a JAX ensemble behind a predict() call. "
                "Pass --guide-model <bank dir> to scripts/build_pool.py."
            )
        return list(models) if guide_members is None else list(models)[:guide_members]

    # --------------------------------------------------------------- generate

    def generate(
        self,
        n_sequences: int,
        base_sequences: list[str] | None = None,
        task: str = "k562",
        student_model=None,
    ) -> tuple[list[str], pd.DataFrame]:
        import torch

        from albench.arch_search import flank_context
        from albench.arch_search.harness_standard import FLANK_CTX, SHIFT_MAX

        if task != "k562":
            raise ValueError(
                f"FastSeqProp is wired to the K562 plasmid context (200-bp insert inside "
                f"its real flanks); task={task!r} needs its own context declared first."
            )
        if student_model is None:
            raise ValueError(
                "FastSeqProp is model-in-the-loop: it optimises the input against a "
                "predictor and there is no meaningful fallback. Pass a guide."
            )
        if self.init == "pool" and not base_sequences:
            raise ValueError("init='pool' needs base_sequences to start each design from")

        models = self._torch_models(student_model, self.guide_members)
        device = getattr(student_model, "device", None) or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        margin = FLANK_CTX + SHIFT_MAX
        L = 200

        # The flanks are REAL plasmid sequence and are not optimised -- only the insert is
        # synthesised. Build one wide one-hot and overwrite its centre each batch, so the
        # model reads designs in exactly the context it reads every other sequence in.
        for m in models:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

        seqs: list[str] = []
        pred_final: list[float] = []
        pred_init: list[float] = []
        n_done = 0
        while n_done < n_sequences:
            b = min(self.batch_size, n_sequences - n_done)

            if self.init == "pool":
                idx = self._rng.integers(0, len(base_sequences), size=b)
                starts = [str(base_sequences[int(i)])[:L].ljust(L, "A") for i in idx]
                wide = flank_context.wide_onehot(starts, margin)
                # Logits that reproduce the start sequence but are not saturated, so the
                # first gradient step can still move them.
                w0 = torch.as_tensor(
                    np.ascontiguousarray(wide[:, :, margin : margin + L]), dtype=torch.float32
                )
                logits = (w0 * 2.0 - 1.0).clone()
            else:
                wide = flank_context.wide_onehot(["A" * L] * b, margin)
                logits = torch.randn(b, 4, L) * 0.1

            wide_t = torch.as_tensor(np.ascontiguousarray(wide), dtype=torch.float32).to(device)
            logits = logits.to(device).requires_grad_(True)
            opt = torch.optim.Adam([logits], lr=self.lr)

            first = None
            for step in range(self.n_steps):
                opt.zero_grad(set_to_none=True)
                # Instance normalisation over POSITIONS, per sequence and per channel:
                # this is the paper's fix for the saturating softmax.
                z = logits - logits.mean(dim=2, keepdim=True)
                z = z / (logits.std(dim=2, keepdim=True) + 1e-6)
                p = torch.softmax(z / self.tau, dim=1)
                # Straight-through: hard one-hot forward, soft gradient backward.
                hard = torch.zeros_like(p).scatter_(1, p.argmax(dim=1, keepdim=True), 1.0)
                oh = hard + (p - p.detach())

                x = wide_t.clone()
                x[:, :, margin : margin + L] = oh
                # Both strands, as every student in this study is scored -- optimising a
                # forward-only score would optimise something we never report.
                preds = []
                for m in models:
                    preds.append(m(x).reshape(-1))
                    preds.append(m(x.flip(1).flip(2)).reshape(-1))
                yhat = torch.stack(preds).mean(dim=0)

                if first is None:
                    first = yhat.detach().float().cpu().numpy().copy()

                if self.objective == "maximize":
                    loss = -yhat.mean()
                elif self.objective == "minimize":
                    loss = yhat.mean()
                else:
                    loss = ((yhat - self.target_activity) ** 2).mean()
                if self.entropy_weight:
                    ent = -(p * torch.log(p + 1e-9)).sum(dim=1).mean()
                    loss = loss - self.entropy_weight * ent
                loss.backward()
                opt.step()

            with torch.no_grad():
                z = logits - logits.mean(dim=2, keepdim=True)
                z = z / (logits.std(dim=2, keepdim=True) + 1e-6)
                final = torch.softmax(z / self.tau, dim=1).argmax(dim=1).cpu().numpy()
                oh = torch.zeros(b, 4, L, device=device)
                oh.scatter_(1, torch.as_tensor(final, device=device)[:, None, :], 1.0)
                x = wide_t.clone()
                x[:, :, margin : margin + L] = oh
                preds = []
                for m in models:
                    preds.append(m(x).reshape(-1))
                    preds.append(m(x.flip(1).flip(2)).reshape(-1))
                yf = torch.stack(preds).mean(dim=0).float().cpu().numpy()

            seqs.extend("".join(r) for r in _NUCS[final])
            pred_final.extend(yf.tolist())
            pred_init.extend(first.tolist())
            n_done += b
            logger.info(
                "FastSeqProp %s: %d/%d designed, guide pred %.3f -> %.3f",
                self.objective,
                n_done,
                n_sequences,
                float(np.mean(first)),
                float(np.mean(yf)),
            )

        meta = pd.DataFrame(
            {
                "seq_idx": np.arange(len(seqs), dtype=np.int64),
                "method": f"fastseqprop_{self.objective}",
                "source": "generated",
                "n_steps": self.n_steps,
                "init": self.init,
                # The guide's own before/after score. NOT a result -- it is the guide
                # grading its own optimisation, and the gap between this and the oracle
                # label on the same sequence is the quantity worth reporting: it is how
                # far the design exploited the guide rather than improving the sequence.
                "guide_pred_init": np.asarray(pred_init, dtype=np.float32),
                "guide_pred_final": np.asarray(pred_final, dtype=np.float32),
            }
        )
        return seqs, meta
