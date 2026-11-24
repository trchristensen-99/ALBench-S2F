"""Reload a saved LegNet bank as a predictor, so a trained student can guide generation.

WHY THIS EXISTS. The model-guided reservoirs (``ise_*``) score every candidate with a
fitness model, and which model that is decides what the arm measures:

  * Guided by the ORACLE -- the same AG ensemble that then labels the pool -- the arm is
    an upper bound. The sequences are selected at the oracle's own argmax, which is
    exactly where the oracle is least constrained by data, and the student is then asked
    to reproduce its extrapolation. A collaborator cannot reproduce this, because they
    do not have an oracle; they have a model trained on their own data.
  * Guided by a STUDENT trained on data we already hold -- the 30k or 300k banks from
    Arm A -- the arm is the realistic one, and the question becomes the interesting one:
    how good does the guide have to be before guided design beats random mutagenesis?

It is also 40x cheaper. The AG oracle labels at ~300 sequences/s, so one 30k ise_ pool
at population 100 and five generations is 15M forward passes and ~14 GPU-hours. A LegNet
on an H100 runs ~10,957 sequences/s, which turns the same pool into ~25 minutes.

The banks this reads are written by ``scripts/program_oct22/bench33_legnet.py
--save-weights``: one bare ``Net`` state_dict per seed, as ``r{NN}_menu_000.pt``, beside
the ``*_meta.json`` that records which pool and recipe produced it. Nothing here rebuilds
a recipe -- the architecture comes from ``mpra_legnet_ref.SOURCE``, the same source the
trainer compiled, so a state_dict that does not fit is a loud error rather than a silently
reshaped model.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np

from albench.arch_search.harness_standard import FLANK_CTX, SHIFT_MAX


class LegNetBank:
    """A saved multi-seed LegNet bank, callable as ``predict(sequences) -> np.ndarray``.

    Predictions are the mean over seeds of the two-strand average, which is how every
    student in this study is scored; a guide that scored sequences differently from the
    way the resulting model is evaluated would be optimising a different objective.
    """

    def __init__(
        self,
        bank_dir: str | Path,
        *,
        device: str | None = None,
        flank_ctx: int = FLANK_CTX,
        shift_max: int = SHIFT_MAX,
        max_members: int | None = None,
    ) -> None:
        import torch

        from albench.arch_search import flank_context
        from albench.arch_search.mpra_legnet_ref import SOURCE

        self.bank_dir = Path(bank_dir)
        ckpts = sorted(glob.glob(str(self.bank_dir / "r*_menu_*.pt")))
        if not ckpts:
            raise FileNotFoundError(
                f"no LegNet checkpoints (r*_menu_*.pt) in {self.bank_dir}. A bank is only "
                f"written when the cell ran with --save-weights."
            )
        if max_members:
            ckpts = ckpts[:max_members]

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # The WIDE margin the crop module expects, not the final input length: it takes
        # its own ctx-wide window and needs shift_max bases of slack on each side to
        # shift within. Handing it a pre-cropped 240 leaves no margin and the
        # convolution silently receives a 10-wide tensor.
        self.margin = flank_ctx + shift_max
        ns: dict = {}
        exec(compile(SOURCE, "<mpra_legnet_ref>", "exec"), ns)  # noqa: S102

        self.models = []
        for c in ckpts:
            net = ns["Net"]()
            sd = torch.load(c, map_location="cpu")
            # bench33_legnet.py saves the INNER network, whose keys are bare
            # ("stem.block.0.weight"), while mpra_legnet_ref's Net holds it as a
            # `legnet` submodule and expects them prefixed. Load into whichever module
            # the keys actually describe rather than renaming them: a rename that
            # happened to fit a different architecture would load silently.
            target = net
            if not any(k.startswith("legnet.") for k in sd) and hasattr(net, "legnet"):
                target = net.legnet
            target.load_state_dict(sd, strict=True)
            # The crop wrapper is what makes this read sequences the way the bank was
            # trained to read them: the 240-bp standard with the centred window at eval.
            m = flank_context.wrap(net, ctx=flank_ctx, shift=shift_max)
            m.to(self.device).eval()
            self.models.append(m)

        # Provenance, so a result can say which bank guided it without re-deriving it
        # from the directory name -- the trainer's own naming has bitten us before.
        self.meta: list[dict] = []
        for c in ckpts:
            mj = c[:-3] + "_meta.json"
            if os.path.exists(mj):
                self.meta.append(json.load(open(mj)))
        self.pools = sorted({os.path.basename(m.get("pool", "") or "?") for m in self.meta})
        self.n_train = sorted({m.get("n_train") for m in self.meta})

    def __len__(self) -> int:
        return len(self.models)

    def describe(self) -> str:
        return (
            f"LegNetBank({self.bank_dir.name}): {len(self.models)} seeds, "
            f"trained on {self.pools} at n_train={self.n_train}"
        )

    def predict(self, sequences: list[str], batch_size: int = 2048) -> np.ndarray:
        import torch

        from albench.arch_search import flank_context

        out = np.zeros(len(sequences), np.float32)
        with torch.no_grad():
            for s in range(0, len(sequences), batch_size):
                chunk = list(sequences[s : s + batch_size])
                x = torch.as_tensor(
                    flank_context.wide_onehot(chunk, self.margin),
                    dtype=torch.float32,
                    device=self.device,
                )
                acc = torch.zeros(x.shape[0], device=self.device)
                for m in self.models:
                    p = m(x).reshape(-1) + m(x.flip(1).flip(2)).reshape(-1)
                    acc += p / 2
                out[s : s + len(chunk)] = (acc / len(self.models)).float().cpu().numpy()
        return out
