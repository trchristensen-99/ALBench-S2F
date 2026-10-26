"""Three-round acquisition dry run (30k genomic base + 3 x 23.3k selected = 100k), driven by cron so it runs unattended.

Compares with the one-round arms of acq_dryrun.py (70k selected at once from the base-size models; k8mean / k8std in the same candidate pool):
  round 1  select 23,333 with the 8-architecture ensemble trained on the 30k base (the scores already computed by the acq30k scoring run),
  round 2  retrain the 8 architectures on base + round-1 picks (53,333), score the remaining candidates, select 23,333,
  round 3  retrain on 76,666, score, select 23,334, then train the final 8-architecture menu (2 seeds) on all 100,000,
so the final models are trained and scored exactly like the one-round arms.  Variants: r3mean (highest ensemble mean) and r3std (highest across-architecture SD).

Idempotent: every call does whatever the files on disk allow (build the next selected pool, register its training cells) and then exits.

    python scripts/program_oct22/acq_rounds.py            # cron, every 10 minutes
"""

import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2_lib import GP, REPS, R, save_pool  # noqa: E402

ACQ = f"{GP}/k562_acq"
RD = f"{ACQ}/rounds"
MR = f"{R}/outputs/menu_retrain_v0"
MENU8 = f"{R}/outputs/acq_oct26/menu_k8.json"
CAND = f"{ACQ}/candidates_r1.npz"
N_ADD = (23333, 23333, 23334)
VARIANTS = {"r3mean": "mean", "r3std": "std"}
log = lambda *a: print(time.strftime("%m-%d %H:%M"), *a, flush=True)  # noqa: E731


def scores(P, crit):
    return P.mean(0) if crit == "mean" else P.std(0)


def poolpred(dirpath, want_idx=None):
    """(n_models, n_candidates) predictions of the 8-architecture menu models in a score_pool directory (first seed)."""
    P = []
    for f in sorted(glob.glob(f"{dirpath}/r00_menu_*_meta.json")):
        z = np.load(f.replace("_meta.json", ".npz"))
        if "poolpred" in z.files:
            P.append(z["poolpred"].astype(np.float32))
    return np.stack(P) if P else None


def round0_pred():
    """The base-size models' scores for the 8 best architectures (selection_info.json lists their menu indices)."""
    info = json.load(open(f"{ACQ}/selection_info.json"))
    P = []
    for i in info["k8"]:
        z = np.load(f"{MR}/acq30k_genomic_r1/r00_menu_{i:03d}.npz")
        P.append(z["poolpred"].astype(np.float32))
    return np.stack(P)


def build_pool(v, i, sel_all):
    """Pool after round i: base 30k + every selected candidate so far."""
    c = np.load(CAND, allow_pickle=True)
    base = np.load(f"{REPS}/genomic_r1_d30000.npz", allow_pickle=True)
    idx = np.concatenate(sel_all)
    seqs = list(base["sequences"][:30000]) + list(c["sequences"][idx])
    lab = np.concatenate([base["oracle_labels"][:30000], c["oracle_labels"][idx]])
    n = len(seqs)
    path = f"{RD}/{v}_P{i}_d{n}.npz"
    save_pool(path, seqs, lab, "oracle_v3", f"acq3r_{v}_{i}", 0)
    return path, n


def select(v, i, P):
    """Round-i picks: best unselected candidates by the variant's criterion."""
    done = [np.load(f"{RD}/{v}_S{j}.npy") for j in range(1, i)]
    s = scores(P, VARIANTS[v]).copy()
    if done:
        s[np.concatenate(done)] = -np.inf
    sel = np.sort(np.argsort(-s)[: N_ADD[i - 1]]).astype(np.int64)
    np.save(f"{RD}/{v}_S{i}.npy", sel)
    return sel


def cells(v, i, pool, n):
    final = i == 3
    out = f"outputs/menu_retrain_v0/acq3r_{v}_{'final' if final else i}"
    extra = "" if final else f" --score_pool {CAND}"
    if final:
        extra = " --seeds 2"
    res = []
    for k in range(4):
        res.append(
            dict(
                name=f"ind30r_{v}_{'f' if final else i}_s{k}",
                gpus=["h100", "v100"],
                ext=[],
                out_dir=out,
                env={},
                needs=[os.path.relpath(pool, R)],
                cmd=f"uv run --no-sync python scratchpad/menu_retrain.py --menu {MENU8} --reservoir_cache {pool} --D {n} --out_dir {R}/{out} --shard {k}/4 --base_cap_min {60 if n >= 70000 else 40}{extra}",
                done=dict(type="file", path=f"{out}/COMPLETE_{k}of4"),
            )
        )
    return res


def register(new):
    p = f"{R}/scratchpad/dispatch_registry.json"
    reg = json.load(open(p))
    have = {c["name"] for c in reg}
    new = [c for c in new if c["name"] not in have]
    if not new:
        return 0
    at = (
        max([i for i, c in enumerate(reg) if c["name"].startswith(("ind30q_", "mrr_nest_"))] + [0])
        + 1
    )  # ahead of the long backlog (nine seeds, Zoonomia)
    reg = reg[:at] + new + reg[at:]
    json.dump(reg, open(p + ".tmp", "w"), indent=1)
    os.replace(p + ".tmp", p)
    return len(new)


def complete(dirname):
    return all(os.path.exists(f"{R}/{dirname}/COMPLETE_{k}of4") for k in range(4))


def main():
    if not (os.path.exists(f"{ACQ}/selection_info.json") and os.path.exists(MENU8)):
        return
    os.makedirs(RD, exist_ok=True)
    for v in VARIANTS:
        sel_all = []
        for i in (1, 2, 3):
            sp = f"{RD}/{v}_S{i}.npy"
            if os.path.exists(sp):
                sel_all.append(np.load(sp))
                continue
            # round i needs the scores of the models trained after round i-1
            if i == 1:
                P = round0_pred()
            else:
                prev = f"outputs/menu_retrain_v0/acq3r_{v}_{i - 1}"
                if not complete(prev):
                    break
                P = poolpred(f"{R}/{prev}")
                if P is None or len(P) < 6:
                    log(
                        f"{v} round {i}: only {0 if P is None else len(P)} scored models in {prev}; waiting"
                    )
                    break
            sel = select(v, i, P)
            sel_all.append(sel)
            pool, n = build_pool(v, i, sel_all)
            added = register(cells(v, i if i < 3 else 3, pool, n))
            log(
                f"{v}: round {i} picked {len(sel):,} (mean label {np.load(CAND, allow_pickle=True)['oracle_labels'][sel].mean():.3f}), pool {n:,}; registered {added} cells"
            )
            break  # one round per call; the training cells must run before the next
        # a selected pool may exist without its cells (e.g. interrupted call): re-register idempotently
        for i, sel in enumerate(sel_all, 1):
            pools = glob.glob(f"{RD}/{v}_P{i}_d*.npz")
            if pools:
                n = int(pools[0].rsplit("_d", 1)[1].split(".")[0])
                register(cells(v, i, pools[0], n))


if __name__ == "__main__":
    main()
