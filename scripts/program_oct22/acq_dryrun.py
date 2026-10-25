"""Acquisition dry run: does a REDUCED selection ensemble (K of the 33 menu architectures) choose the same sequences, and train as good a model, as the full menu?

Setting: base = 30k genomic (pool r1), candidates = 300k labelled sequences (100k each of random, motif_ct_enriched, evoaug_paper2023; oracle_v3 labels are free, so
"labelling" a selected sequence is a lookup), add 70k, train at D=100k. The selection models are the menu retrained on the base (menu_retrain.py --score_pool).

    python scripts/program_oct22/acq_dryrun.py candidates        # build the candidate pool
    python scripts/program_oct22/acq_dryrun.py menus             # menu_k8.json: the 8 architectures with the best validation r at the base size
    python scripts/program_oct22/acq_dryrun.py select            # after scoring: build the selected 100k pools (needs the scoring cells' poolpred)
    python scripts/program_oct22/acq_dryrun.py overlap           # how much do the selections agree?

Selection criteria (all on the candidate pool, ensemble = validation-weighted-free plain mean/std over the chosen architectures):
  mean   the 70k with the highest ensemble-mean prediction (exploit)
  std    the 70k with the highest across-architecture standard deviation (explore)
  rand   70k at random (control)
Selection ensembles: full (all trained architectures), k8 (8 best by base validation r), k8r (8 random architectures, control for "any 8" vs "the 8 best").
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2_lib import GP, REPS, R, save_pool  # noqa: E402

ACQ = f"{GP}/k562_acq"
BASE_DIR = f"{R}/outputs/menu_retrain_v0/acq30k_genomic_r1"
MENU = f"{R}/outputs/menu_retrain_v0/menu_v0.json"
N_ADD = 70_000
VARIANTS = [
    ("rand", None, None),
    ("fullmean", "full", "mean"),
    ("fullstd", "full", "std"),
    ("k8mean", "k8", "mean"),
    ("k8std", "k8", "std"),
    ("k8rmean", "k8r", "mean"),
    ("k8rstd", "k8r", "std"),
]


def candidates():
    os.makedirs(ACQ, exist_ok=True)
    S, Y, SRC = [], [], []
    for i, st in enumerate(("random", "motif_ct_enriched", "evoaug_paper2023")):
        z = np.load(f"{REPS}/{st}_r1_d100000.npz", allow_pickle=True)
        assert str(z["oracle_id"]) == "oracle_v3"
        S += list(z["sequences"][:100000])
        Y.append(z["oracle_labels"][:100000].astype(np.float32))
        SRC += [i] * 100000
    S = np.asarray(S, dtype=object)
    print("candidates:", len(S), "unique:", len(set(S)))
    tmp = f"{ACQ}/candidates_r1.tmp.npz"
    np.savez(
        tmp,
        sequences=S,
        oracle_labels=np.concatenate(Y),
        source=np.asarray(SRC, np.int8),
        sources=np.asarray(["random", "motif_ct_enriched", "evoaug_paper2023"]),
    )
    os.replace(tmp, f"{ACQ}/candidates_r1.npz")


def base_meta():
    rows = []
    for f in glob.glob(f"{BASE_DIR}/r00_menu_*_meta.json"):
        m = json.load(open(f))
        rows.append((m["val_pearson"], m["menu_index"]))
    return sorted(rows, reverse=True)


def menus():
    rows = base_meta()
    assert len(rows) >= 30, len(rows)
    menu = json.load(open(MENU))
    top = [i for _, i in rows[:8]]
    os.makedirs(f"{R}/outputs/acq_oct26", exist_ok=True)
    json.dump(
        [menu[i] for i in sorted(top)], open(f"{R}/outputs/acq_oct26/menu_k8.json", "w"), indent=1
    )
    print("k8 =", sorted(top), "val", [round(v, 4) for v, _ in rows[:8]])


def load_scores():
    P, V, idx = [], [], []
    for f in sorted(glob.glob(f"{BASE_DIR}/r00_menu_*.npz")):
        if f.endswith("_meta.json"):
            continue
        z = np.load(f)
        if "poolpred" not in z.files:
            continue
        m = json.load(open(f.replace(".npz", "_meta.json")))
        P.append(z["poolpred"].astype(np.float32))
        V.append(m["val_pearson"])
        idx.append(m["menu_index"])
    return np.stack(P), np.asarray(V), np.asarray(idx)


def select_sets(P, V, rng_seed=0):
    n = P.shape[1]
    rng = np.random.default_rng(rng_seed)
    k8 = np.argsort(-V)[:8]
    k8r = rng.choice(len(V), 8, replace=False)
    ens = {"full": np.arange(len(V)), "k8": k8, "k8r": k8r}
    out = {"rand": rng.choice(n, N_ADD, replace=False)}
    for name, e, crit in VARIANTS[1:]:
        sub = P[ens[e]]
        score = sub.mean(0) if crit == "mean" else sub.std(0)
        out[name] = np.argsort(-score)[:N_ADD]
    return out, ens


def select():
    P, V, idx = load_scores()
    assert len(V) >= 30, f"only {len(V)} scored models"
    c = np.load(f"{ACQ}/candidates_r1.npz", allow_pickle=True)
    base = np.load(f"{REPS}/genomic_r1_d30000.npz", allow_pickle=True)
    sets, ens = select_sets(P, V)
    info = {
        "n_models": int(len(V)),
        "k8": [int(idx[i]) for i in ens["k8"]],
        "k8r": [int(idx[i]) for i in ens["k8r"]],
        "variants": {},
    }
    for name, sel in sets.items():
        sel = np.sort(sel)
        seqs = list(base["sequences"][:30000]) + list(c["sequences"][sel])
        lab = np.concatenate([base["oracle_labels"][:30000], c["oracle_labels"][sel]])
        save_pool(f"{ACQ}/sel_{name}_d100000.npz", seqs, lab, "oracle_v3", f"acq_{name}", 0)
        src = np.bincount(c["source"][sel], minlength=3) / len(sel)
        info["variants"][name] = {
            "label_mean": float(c["oracle_labels"][sel].mean()),
            "source_frac": src.tolist(),
            "n_unique": len(set(seqs)),
        }
        print(
            f"{name:9s} label mean {info['variants'][name]['label_mean']:.3f} source (random, motif, evoaug) {np.round(src, 3)} unique {len(set(seqs)):,}",
            flush=True,
        )
    json.dump({k: v for k, v in info.items()}, open(f"{ACQ}/selection_info.json", "w"), indent=1)
    np.savez(f"{ACQ}/selection_indices.npz", **{k: np.sort(v) for k, v in sets.items()})


def overlap():
    z = np.load(f"{ACQ}/selection_indices.npz")
    names = list(z.files)
    print("Jaccard overlap of the selected 70k (rows/cols: " + ", ".join(names) + ")")
    for a in names:
        print(
            f"{a:9s}",
            " ".join(
                f"{len(np.intersect1d(z[a], z[b])) / len(np.union1d(z[a], z[b])):.2f}"
                for b in names
            ),
        )


if __name__ == "__main__":
    {"candidates": candidates, "menus": menus, "select": select, "overlap": overlap}[sys.argv[1]]()
