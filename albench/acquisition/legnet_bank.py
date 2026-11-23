"""A ``SequenceModel`` backed by a directory of trained tuned-LegNet checkpoints.

Acquisition needs a student that can be asked three different things -- a prediction, how
much the ensemble disagrees, and a last-layer embedding -- and the study's students are
written to disk by the canonical trainer as bare ``state_dict`` files. This wraps a
directory of them so the existing acquisition functions can drive the study's actual
student rather than a reimplementation of it.

Two things it is careful about, because both have bitten this codebase before.

*The input standard.* Every student sees the 200-bp insert inside 20 real plasmid bases per
side, and predictions are averaged over both strands. That is worth about +0.07 test r, so
a scorer that quietly used the bare insert would select against a different model than the
one being trained. The one-hot and flank construction here is imported from the trainer's
own helpers rather than rewritten, so the two cannot drift.

*Epistemic vs total uncertainty.* ``epistemic_uncertainty`` is the spread ACROSS ensemble
members, which is the part that shrinks as we label more and the part acquisition should
chase. A single-member bank cannot separate the two, so it reports
``separates_uncertainty = False`` instead of silently returning zeros -- a zero epistemic
score would make every candidate look equally informative and turn BADGE into k-means++ on
identical vectors, which is exactly the failure mode its controls exist to detect.
"""

from __future__ import annotations

import glob
import importlib.util
import os

import numpy as np

from albench.model import SequenceModel

ARCH = dict(
    in_ch=4,
    stem_ch=64,
    stem_ks=11,
    ef_ks=9,
    ef_block_sizes=[80, 96, 112, 128],
    pool_sizes=[2, 2, 2, 2],
    resize_factor=4,
)
FLANK_CTX = 20


def _bench_common():
    """The trainer's own input helpers, imported rather than reimplemented."""
    from albench.paths import REPO_ROOT

    root = str(REPO_ROOT)
    for p in (root, os.path.join(root, "scripts/program_oct22"), os.path.join(root, "experiments")):
        if p not in __import__("sys").path:
            __import__("sys").path.insert(0, p)
    import bench33_common as C  # type: ignore

    return C


def _legnet_module():
    from albench.paths import REPO_ROOT

    path = os.path.join(str(REPO_ROOT), "external/human_legnet/model.py")
    spec = importlib.util.spec_from_file_location("human_legnet_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LegNetBank(SequenceModel):
    """Ensemble of tuned-LegNet checkpoints from one output directory."""

    def __init__(self, bank_dir: str, *, device: str | None = None, batch: int = 2048):
        import torch

        self.paths = sorted(glob.glob(os.path.join(bank_dir, "*.pt")))
        if not self.paths:
            raise FileNotFoundError(
                f"no .pt checkpoints in {bank_dir}; the trainer needs --save-weights"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch = batch
        mod = _legnet_module()
        self.models = []
        for p in self.paths:
            m = mod.LegNet(**ARCH).to(self.device)
            m.load_state_dict(torch.load(p, map_location=self.device))
            m.eval()
            self.models.append(m)
        C = _bench_common()
        self._onehot = C.onehot_core200
        self._f5, self._f3 = C.mpra_flanks(FLANK_CTX)

    def __len__(self) -> int:
        return len(self.models)

    @property
    def separates_uncertainty(self) -> bool:
        return len(self.models) > 1

    def _to_x(self, seqs):
        import torch

        core = torch.from_numpy(self._onehot(list(seqs))).to(self.device)  # (N, 200, 4)
        n = len(core)
        core = torch.cat(
            [
                torch.from_numpy(self._f5).to(self.device).expand(n, FLANK_CTX, 4),
                core,
                torch.from_numpy(self._f3).to(self.device).expand(n, FLANK_CTX, 4),
            ],
            dim=1,
        )
        return core.permute(0, 2, 1).contiguous()

    def _member_predictions(self, sequences) -> np.ndarray:
        """(n_members, N) reverse-complement-averaged predictions."""
        import torch

        seqs = list(sequences)
        out = np.zeros((len(self.models), len(seqs)), np.float32)
        with torch.no_grad():
            for s in range(0, len(seqs), self.batch):
                x = self._to_x(seqs[s : s + self.batch])
                xr = x.flip(1).flip(2)
                for mi, m in enumerate(self.models):
                    p = (m(x).float() + m(xr).float()) / 2
                    out[mi, s : s + self.batch] = p.reshape(-1).cpu().numpy()
        return out

    def predict(self, sequences) -> np.ndarray:
        return self._member_predictions(sequences).mean(0)

    def uncertainty(self, sequences) -> np.ndarray:
        return self._member_predictions(sequences).std(0)

    def epistemic_uncertainty(self, sequences) -> np.ndarray:
        if not self.separates_uncertainty:
            raise RuntimeError(
                "a one-member bank has no ensemble spread; epistemic acquisition on it "
                "would score every candidate identically"
            )
        return self._member_predictions(sequences).std(0, ddof=1)

    def aleatoric_uncertainty(self, sequences) -> np.ndarray:
        """Oracle labels carry no per-sequence standard error, so this is genuinely flat.

        Returning ones is not a stand-in for a measurement: these students are trained on
        oracle PREDICTIONS, which have no replicate-derived error term. Anything dividing by
        it is therefore using epistemic alone, which is what we want here and what the
        caller should know it is getting.
        """
        return np.ones(len(list(sequences)), dtype=float)

    def embed(self, sequences) -> np.ndarray:
        """Penultimate features of the FIRST member, for BADGE's k-means++ half."""
        import torch

        m = self.models[0]
        last = None
        for mod_ in m.modules():
            if isinstance(mod_, torch.nn.Linear):
                last = mod_
        if last is None:
            raise RuntimeError("no Linear layer found; cannot take a penultimate embedding")

        feats: list[np.ndarray] = []
        hook = last.register_forward_pre_hook(
            lambda _m, inp: feats.append(inp[0].detach().float().cpu().numpy())
        )
        try:
            seqs = list(sequences)
            with torch.no_grad():
                for s in range(0, len(seqs), self.batch):
                    m(self._to_x(seqs[s : s + self.batch]))
        finally:
            hook.remove()
        return np.concatenate(feats, 0)
