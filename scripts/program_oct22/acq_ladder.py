"""One acquisition arm, end to end: train, select, grow, retrain -- in a single job.

THE QUESTION. A library has a fixed oligo budget. Having spent some of it on a pilot, is it better to buy the
rest in ONE batch, or to split it into rounds and let each round be chosen by a model that has seen the one
before? Splitting is strictly more work -- you retrain between rounds and the synthesis happens in stages --
so it has to pay for itself. Nothing in this study has measured whether it does.

  --schedule 75000           one-shot: select the whole remaining budget from the pilot model
  --schedule 25000,25000,25000   iterative: three rounds, retraining between them

Both arms end with the same number of labelled sequences, drawn from the same candidate pool, trained with
the same per-size recipe and scored on the same sets, so the only difference is when the choice was made.
`--acq random` is the control that makes the comparison readable: it runs the identical machinery while
choosing at random, so any gap between one-shot and iterative random is the machinery's own noise, not a
finding.

WHAT THE SELECTOR MAY SEE. The candidate pool is labelled -- cheaper and simpler than labelling on demand --
but the selector is handed sequences only. The labels enter in exactly two places: when a selected sequence
joins the training pool, and in the `oracle_top` arm, which deliberately selects ON the hidden label to put
an upper bound on what any activity-chasing selector could reach.

ACQUISITION ARMS, and why these three plus a control:
  random               the control. Also the honest baseline: it is what "just buy more of the reservoir" means.
  epistemic            top-k by disagreement across the five seeds. Uncertainty with no diversity term, which
                       is `badge_epistemic_only` in the library -- the control that says whether a BADGE gain
                       comes from the fusion or from the uncertainty half alone.
  badge                epistemic-weighted embeddings, k-means++ seeded. Uncertainty AND diversity.
  oracle_top           selects on the hidden oracle label. Not a method; a bound.

Note the ensemble is five seeds of ONE architecture, so it has no aleatoric estimate: BADGE's default
`mode="ratio"` would silently reduce to epistemic weighting (the library warns about exactly this). We
therefore pass `mode="epistemic"` explicitly, and the arm is named for what it is.

    python scripts/program_oct22/acq_ladder.py --seed_pool ... --candidates ... \
        --schedule 25000,25000,25000 --acq badge --n_seeds 5 --tag it3_badge_r1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import bench33_common as C  # noqa: E402

log = logging.getLogger("acq_ladder")

# The tuned recipe per training-set size (docs/TUNED_LEGNET.md), with the two mid sizes the Nov-4 run added.
# A size between rungs takes the recipe of the nearest measured rung below it, which is the conservative
# choice: weight decay has fallen monotonically with size, so erring low on size errs high on regularisation.
RECIPE = {
    10_000: "e800_wd4",
    30_000: "e500_wd2",
    55_000: "e500_wd15",
    80_000: "e500",  # the 100k rung's recipe; wd defaults to the published 0.1
    100_000: "e500",
    300_000: "e300_wd05",
    1_000_000: "e200_wd05",
}


def recipe_for(d: int) -> str:
    rungs = sorted(RECIPE)
    best = rungs[0]
    for r in rungs:
        if r <= d * 1.15:  # 105k counts as the 100k rung, 55k as its own
            best = r
    return RECIPE[best]


# ----------------------------------------------------------------------------- the trained ensemble
class LegNetEnsemble:
    """The five trained seeds, as the thing an acquisition function consumes.

    `embed` returns the FIRST seed's penultimate features rather than a concatenation or a mean of all five:
    the five seeds have unrelated coordinate systems, so averaging them is not an average of anything, and
    concatenating makes the k-means++ geometry depend on ensemble size. One model supplies the representation
    and all five supply the uncertainty, which is the division of labour BADGE assumes anyway.
    """

    separates_uncertainty = False  # point predictors: there is no aleatoric head to read

    def __init__(self, ckpts: list[Path], ctx: int, device: str = "cuda") -> None:
        import torch

        sys.path.insert(0, str(REPO / "scripts/program_oct22"))
        import bench33_legnet as B

        self.torch = torch
        self.ctx = ctx
        self.device = device
        self.models = []
        for p in ckpts:
            m = B.load_legnet().LegNet(**B.ARCH).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            m.eval()
            self.models.append(m)
        log.info("loaded %d checkpoints", len(self.models))
        self._emb_buf: list = []
        self.models[0].head.register_forward_pre_hook(self._capture)

    def _capture(self, _module, args):
        self._emb_buf.append(args[0].detach().float().cpu())

    def _to_x(self, seqs: list[str]):
        torch = self.torch
        core = torch.from_numpy(C.onehot_core200(seqs))
        if self.ctx:
            f5, f3 = C.mpra_flanks(self.ctx)
            n = len(core)
            core = torch.cat(
                [
                    torch.from_numpy(f5).expand(n, self.ctx, 4),
                    core,
                    torch.from_numpy(f3).expand(n, self.ctx, 4),
                ],
                dim=1,
            )
        return core.permute(0, 2, 1).contiguous()

    def _forward(self, seqs: list[str], want_emb: bool, bs: int = 4096):
        """Reverse-complement-averaged predictions from every model; embeddings from the first if asked."""
        torch = self.torch
        P = [[] for _ in self.models]
        E = []
        for i in range(0, len(seqs), bs):
            x = self._to_x(seqs[i : i + bs]).to(self.device)
            xr = x.flip(1).flip(2)
            for mi, m in enumerate(self.models):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    self._emb_buf.clear()
                    f = m(x).float()
                    e_fwd = self._emb_buf[-1] if (want_emb and mi == 0) else None
                    r = m(xr).float()
                P[mi].append(((f + r) / 2).cpu().numpy())
                if e_fwd is not None:
                    E.append(e_fwd.numpy())
            if i and (i // bs) % 25 == 0:
                log.info("  scored %d / %d", i, len(seqs))
        preds = np.stack([np.concatenate(p) for p in P])  # (M, N)
        emb = np.concatenate(E) if E else None
        return preds, emb

    def score(self, seqs: list[str], want_emb: bool = False):
        preds, emb = self._forward(seqs, want_emb)
        return preds.mean(0), preds.std(0), emb


# ----------------------------------------------------------------------------- selection
def kmeanspp_torch(points, k: int, seed: int, device: str = "cuda"):
    """k-means++ seeding, same algorithm as albench.acquisition.badge.kmeanspp, distances on the GPU.

    The library version is exact numpy and costs O(k.n.d) in float64: at n=1,000,000, d=256 and k=75,000 that
    is hours of CPU per selection, which would make the diversity arm unaffordable rather than merely slow.
    Only where the arithmetic runs changes; the sampling rule (probability proportional to squared distance
    from the nearest chosen centre) is identical. `verify_kmeanspp.py` measures the two against each other.
    """
    import torch

    g = torch.Generator(device=device).manual_seed(int(seed))
    X = torch.as_tensor(np.ascontiguousarray(points), dtype=torch.float32, device=device)
    n = X.shape[0]
    if k > n:
        raise ValueError(f"cannot select {k} from {n} candidates")
    first = int(torch.randint(n, (1,), generator=g, device=device).item())
    chosen = torch.empty(k, dtype=torch.long, device=device)
    chosen[0] = first
    min_d = ((X - X[first]) ** 2).sum(1)
    t0 = time.time()
    for i in range(1, k):
        probs = min_d.clamp_min(0).clone()
        probs[chosen[:i]] = 0.0
        tot = float(probs.sum())
        if tot <= 0:
            # Every remaining point coincides with a centre: the embedding has collapsed, which is the
            # degeneracy the BADGE module exists to avoid. Say so rather than returning a quiet random tail.
            log.warning(
                "k-means++ distances all zero after %d/%d picks: embedding collapsed, rest is random",
                i,
                k,
            )
            mask = torch.ones(n, dtype=torch.bool, device=device)
            mask[chosen[:i]] = False
            rest = torch.nonzero(mask).squeeze(1)
            perm = torch.randperm(rest.numel(), generator=g, device=device)[: k - i]
            chosen[i:] = rest[perm]
            break
        idx = int(torch.multinomial(probs, 1, generator=g).item())
        chosen[i] = idx
        min_d = torch.minimum(min_d, ((X - X[idx]) ** 2).sum(1))
        if i % 5000 == 0:
            log.info("  k-means++ %d / %d (%.0fs)", i, k, time.time() - t0)
    return chosen.cpu().numpy()


def select(method: str, ens, cand_seqs, cand_y, n_add: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if method == "random":
        return rng.choice(len(cand_seqs), size=n_add, replace=False)
    if method == "oracle_top":
        return np.argsort(-cand_y)[:n_add]  # the bound: selects on the label the others never see
    mean, sd, emb = ens.score(cand_seqs, want_emb=(method == "badge"))
    log.info(
        "candidate scores: mean %+.3f+-%.3f, epistemic sd %.4f+-%.4f",
        mean.mean(),
        mean.std(),
        sd.mean(),
        sd.std(),
    )
    if method == "epistemic":
        return np.argsort(-sd)[:n_add]
    if method == "badge":
        spread = float(np.linalg.norm(emb * sd[:, None], axis=1).std())
        if spread < 1e-12:
            raise RuntimeError("BADGE embeddings have no spread; selection would be random")
        log.info("badge embedding: dim %d, weighted-norm spread %.4f", emb.shape[1], spread)
        return kmeanspp_torch(emb * sd[:, None], n_add, seed)
    raise ValueError(f"unknown acquisition {method!r}")


# ----------------------------------------------------------------------------- the arm
def train(
    pool_path: Path,
    d: int,
    n_seeds: int,
    ctx: int,
    seeds: list[int],
    out_root: Path,
    smoke: bool = False,
) -> Path:
    cfg = recipe_for(d)
    spec = ",".join(f"{i}={s}" for i, s in enumerate(seeds[:n_seeds]))
    cmd = [
        sys.executable,
        str(HERE / "bench33_legnet.py"),
        "--D",
        str(d),
        "--cfg",
        cfg,
        "--pool",
        str(pool_path),
        "--seeds",
        spec,
        "--ctx",
        str(ctx),
        "--save-weights",
        "--out-root",
        str(out_root),
    ]
    if smoke:
        cmd.append("--smoke")
        out_root = REPO / "outputs/bench33/smoke"  # where bench33_legnet redirects itself
    log.info("train D=%d cfg=%s on %s", d, cfg, pool_path.name)
    subprocess.run(cmd, check=True, cwd=str(REPO))
    res = Path(pool_path).stem.split("_d")[0]
    return out_root / f"bench38_ln_{cfg}_{d // 1000}k_c{ctx}_{res}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_pool", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--schedule", required=True, help="comma list of how many to add per round")
    ap.add_argument("--acq", required=True, choices=("random", "epistemic", "badge", "oracle_top"))
    ap.add_argument("--tag", required=True, help="names the pools and output directories")
    ap.add_argument(
        "--seed_tag",
        default=None,
        help="name of the ROUND-0 pool. Every arm of one replicate shares it, so the pilot\nensemble is trained once and the arms differ only in what they select.",
    )
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--ctx", type=int, default=20)
    ap.add_argument(
        "--seed", type=int, default=0, help="selection RNG; also picks the training seeds"
    )
    ap.add_argument("--pool_dir", default="outputs/grid_pools/k562_acq52")
    ap.add_argument("--out_root", default="outputs/menu_retrain_v0")
    ap.add_argument("--done", default=None, help="touch this file when the whole arm finishes")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="two epochs per model: exercises every step of the arm in minutes, and the numbers it "
        "produces are worthless by construction",
    )
    a = ap.parse_args()

    adds = [int(x) for x in a.schedule.split(",")]
    seed_tag = a.seed_tag or (Path(a.seed_pool).stem.split("_d")[0] + f"_s{a.seed}")
    pool_dir = REPO / a.pool_dir
    pool_dir.mkdir(parents=True, exist_ok=True)
    out_root = REPO / a.out_root

    z = np.load(REPO / a.seed_pool, allow_pickle=True)
    pool_seqs = [str(s) for s in z["sequences"]]
    pool_y = z["oracle_labels"].astype(np.float32)
    oid = str(z["oracle_id"])
    cz = np.load(REPO / a.candidates, allow_pickle=True)
    cand_seqs = [str(s) for s in cz["sequences"]]
    cand_y = cz["oracle_labels"].astype(np.float32)
    cand_src = [str(s) for s in cz["source"]] if "source" in cz.files else None
    assert str(cz["oracle_id"]) == oid, (
        "seed pool and candidates were labelled by different oracles"
    )
    log.info(
        "seed pool %d, candidates %d, schedule %s, acq %s",
        len(pool_seqs),
        len(cand_seqs),
        adds,
        a.acq,
    )

    rng = np.random.default_rng(1000 + a.seed)
    train_seeds = [int(x) for x in rng.integers(1, 2**31 - 1, size=a.n_seeds)]
    available = np.ones(len(cand_seqs), dtype=bool)
    trace = []

    for r, n_add in enumerate(adds):
        d = len(pool_seqs)
        tag_r = seed_tag if r == 0 else a.tag
        pool_path = pool_dir / f"{tag_r}_d{d}.npz"
        if not pool_path.exists():
            C.atomic_savez(
                pool_path,
                sequences=np.asarray(pool_seqs, dtype=object),
                oracle_labels=pool_y,
                oracle_id=oid,
                strategy=tag_r,
                seed=np.int64(a.seed),
            )
        # An acquisition that never consults the model gains nothing from an intermediate retrain, so
        # skip it: an iterative random arm would otherwise pay three ensembles to reproduce, exactly,
        # what one random draw of the same size gives.
        needs_model = a.acq in ("epistemic", "badge")
        run_dir = (
            train(pool_path, d, a.n_seeds, a.ctx, train_seeds, out_root, a.smoke)
            if (needs_model or r == 0)
            else None
        )

        if n_add == 0:
            # `--schedule 0` trains the pilot ensemble and stops. Every arm of a replicate shares that
            # ensemble, so it is registered as its own cell and the arms wait on it instead of each
            # retraining it (and racing each other to write the same files).
            log.info("schedule 0: pilot ensemble only")
            break

        idx_avail = np.nonzero(available)[0]
        ens = None
        if needs_model:
            ckpts = sorted(run_dir.glob("r*_menu_000.pt"))
            if len(ckpts) < a.n_seeds:
                raise RuntimeError(f"{len(ckpts)} checkpoints in {run_dir}, expected {a.n_seeds}")
            ens = LegNetEnsemble(ckpts, a.ctx)
        t0 = time.time()
        pick_local = select(
            a.acq,
            ens,
            [cand_seqs[i] for i in idx_avail],
            cand_y[idx_avail],
            n_add,
            a.seed * 100 + r,
        )
        picks = idx_avail[pick_local]
        if ens is not None:  # the trainer runs as a child process and needs this GPU back
            import torch

            del ens
            torch.cuda.empty_cache()
        sel_y = cand_y[picks]
        comp = {}
        if cand_src is not None:
            for i in picks:
                comp[cand_src[i]] = comp.get(cand_src[i], 0) + 1
        log.info(
            "round %d: picked %d of %d in %.0fs | label %+.3f+-%.3f (pool so far %+.3f) | %s",
            r,
            len(picks),
            len(idx_avail),
            time.time() - t0,
            sel_y.mean(),
            sel_y.std(),
            pool_y.mean(),
            ", ".join(
                f"{k} {v / len(picks):.0%}" for k, v in sorted(comp.items(), key=lambda kv: -kv[1])
            ),
        )
        trace.append(
            dict(
                round=r,
                trained_at=d,
                run_dir=str(run_dir.relative_to(REPO)) if run_dir else None,
                n_add=int(n_add),
                sel_label_mean=float(sel_y.mean()),
                sel_label_sd=float(sel_y.std()),
                composition=comp,
                select_sec=round(time.time() - t0, 1),
            )
        )
        available[picks] = False
        pool_seqs += [cand_seqs[i] for i in picks]
        pool_y = np.concatenate([pool_y, sel_y])

    d = len(pool_seqs)
    pool_path = pool_dir / f"{a.tag}_d{d}.npz"
    if not pool_path.exists():
        C.atomic_savez(
            pool_path,
            sequences=np.asarray(pool_seqs, dtype=object),
            oracle_labels=pool_y,
            oracle_id=oid,
            strategy=a.tag,
            seed=np.int64(a.seed),
        )
    run_dir = train(pool_path, d, a.n_seeds, a.ctx, train_seeds, out_root, a.smoke)
    trace.append(
        dict(round=len(adds), trained_at=d, run_dir=str(run_dir.relative_to(REPO)), final=True)
    )

    (run_dir / "acq_trace.json").write_text(
        json.dumps(
            dict(
                tag=a.tag,
                acq=a.acq,
                schedule=adds,
                seed_pool=a.seed_pool,
                candidates=a.candidates,
                n_seeds=a.n_seeds,
                train_seeds=train_seeds,
                final_d=d,
                rounds=trace,
            ),
            indent=1,
        )
    )
    log.info("DONE %s -> %s", a.tag, run_dir)
    if a.done:
        p = REPO / a.done
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
