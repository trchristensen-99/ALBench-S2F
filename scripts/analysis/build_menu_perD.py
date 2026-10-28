# ruff: noqa
#!/usr/bin/env python
"""Build a menu from the AutoResearch runs done AT a given dataset size (default-loop runs of outputs/archsearch_reps_oct22/<corpus>_<D>), in the format of menu_v1.json.

Same recipe as menu v1: the top configuration by search validation r of every run, then the best of the remaining candidates until `--total` configurations. Runs of the agent, explore and no-guidance
variants are left out (the menu came from default-loop runs). Candidates that failed or have no validation r are skipped. Corpora are used as given in --corpora; at most `--per-run-extra` extra
candidates per run so that no single run fills the tail.

    python scripts/analysis/build_menu_perD.py --D 10k --corpora genomic,random,motif_ct_enriched,evoaug_paper2023,evo2 --runs-per-corpus 3 --total 20 --out outputs/menus_oct28/M10.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def runs_of(corpus, D, k):
    ds = sorted(
        glob.glob(f"outputs/archsearch_reps_oct22/{corpus}_{D}/rep*"),
        key=lambda d: int(d.rsplit("rep", 1)[1]),
    )
    return ds[:k]


def candidates(rd):
    out = []
    for f in glob.glob(f"{rd}/r*_meta.json"):
        m = json.load(open(f))
        if m.get("val_pearson") is None or not m.get("source") or not m.get("train_config"):
            continue
        out.append(
            {
                "id": f"{rd.split('archsearch_reps_oct22/')[1].replace('/', '_')}_{os.path.basename(f).replace('_meta.json', '')}",
                "cell": rd.split("archsearch_reps_oct22/")[1],
                "idea": m.get("idea"),
                "orig_val": float(m["val_pearson"]),
                "n_params": m.get("n_params"),
                "source": m["source"],
                "train_config": m["train_config"],
            }
        )
    return sorted(out, key=lambda e: -e["orig_val"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", required=True, help="10k | 100k")
    ap.add_argument("--corpora", required=True)
    ap.add_argument("--runs-per-corpus", type=int, default=3)
    ap.add_argument("--total", type=int, default=20)
    ap.add_argument("--per-run-extra", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    per_run, top1 = {}, []
    for corp in a.corpora.split(","):
        for rd in runs_of(corp, a.D, a.runs_per_corpus):
            cs = candidates(rd)
            if cs:
                per_run[rd] = cs
                e = dict(cs[0])
                e["family"] = corp
                top1.append(e)
    chosen = sorted(top1, key=lambda e: -e["orig_val"])
    seen = {e["id"] for e in chosen}
    taken = {rd: 0 for rd in per_run}
    rest = sorted(
        ((e, rd) for rd, cs in per_run.items() for e in cs[1:]), key=lambda t: -t[0]["orig_val"]
    )
    for e, rd in rest:
        if len(chosen) >= a.total:
            break
        if e["id"] in seen or taken[rd] >= a.per_run_extra:
            continue
        e = dict(e)
        e["family"] = rd.split("archsearch_reps_oct22/")[1].split("/")[0].rsplit("_", 1)[0]
        chosen.append(e)
        seen.add(e["id"])
        taken[rd] += 1
    chosen = chosen[: a.total]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(chosen, open(a.out, "w"), indent=1)
    print(
        f"{len(per_run)} runs -> {len(chosen)} configurations; search val r {min(e['orig_val'] for e in chosen):.4f} to {max(e['orig_val'] for e in chosen):.4f}; recipes: "
        + str(
            sorted(
                {
                    (
                        e["train_config"].get("optimizer"),
                        e["train_config"].get("lr"),
                        e["train_config"].get("weight_decay"),
                        e["train_config"].get("batch_size"),
                    )
                    for e in chosen
                }
            )
        )[:300]
    )


if __name__ == "__main__":
    main()
