import glob
import json
import os

import pandas as pd


def parse_yeast_scaling():
    data = []
    for fp in glob.glob("outputs/exp0_yeast_scaling/**/result.json", recursive=True):
        path_parts = fp.split(os.sep)
        fraction_part = next((p for p in path_parts if "fraction_" in p), None)
        run = next((p for p in path_parts if "run_" in p or "rep_seed" in p), "unknown")
        if fraction_part:
            frac = float(fraction_part.replace("fraction_", ""))
            with open(fp, "r") as f:
                try:
                    d = json.load(f)
                    rnd = float(d["test_metrics"]["random"]["pearson_r"])
                    gen = float(d["test_metrics"]["genomic"]["pearson_r"])
                    snv = float(d["test_metrics"]["snv"]["pearson_r"])
                    data.append((frac, run, rnd, gen, snv))
                except (KeyError, ValueError, TypeError):
                    continue
    df = pd.DataFrame(
        data, columns=["fraction", "run", "test_random_pr", "test_genomic_pr", "test_snv_pr"]
    )
    df = df.sort_values(by=["fraction", "run"])
    os.makedirs("results/exp0_scaling/data", exist_ok=True)
    df.to_csv("results/exp0_scaling/data/yeast_baseline.csv", index=False)
    print("Compiled Yeast baseline results: results/exp0_scaling/data/yeast_baseline.csv")


def parse_k562_scaling():
    data = []
    for fp in glob.glob("outputs/exp0_k562_scaling/**/result.json", recursive=True):
        path_parts = fp.split(os.sep)
        fraction_part = next((p for p in path_parts if "fraction_" in p), None)
        run = next((p for p in path_parts if "run_" in p or "rep_seed" in p), "unknown")
        if fraction_part:
            frac = float(fraction_part.replace("fraction_", ""))
            with open(fp, "r") as f:
                try:
                    d = json.load(f)
                    id_r = float(d["test_metrics"]["in_distribution"]["pearson_r"])
                    snv_r = float(d["test_metrics"]["snv_abs"]["pearson_r"])
                    ood_r = float(d["test_metrics"]["ood"]["pearson_r"])
                    data.append((frac, run, id_r, snv_r, ood_r))
                except (KeyError, ValueError, TypeError):
                    continue
    df = pd.DataFrame(data, columns=["fraction", "run", "test_id_pr", "test_snv_pr", "test_ood_pr"])
    df = df.sort_values(by=["fraction", "run"])
    os.makedirs("results/exp0_scaling/data", exist_ok=True)
    df.to_csv("results/exp0_scaling/data/k562_baseline.csv", index=False)
    print("Compiled K562 baseline results: results/exp0_scaling/data/k562_baseline.csv")


def parse_oracle_benchmarks():
    data = []

    # 1. K562 DREAM-RNN Ensembles
    for fp in glob.glob("outputs/oracle_dream_rnn_k562_ensemble/**/result.json", recursive=True):
        run = fp.split(os.sep)[2]
        with open(fp, "r") as f:
            try:
                d = json.load(f)
                id_r = float(d["test_metrics"]["in_distribution"]["pearson_r"])
                snv_r = float(d["test_metrics"]["snv_abs"]["pearson_r"])
                ood_r = float(d["test_metrics"]["ood"]["pearson_r"])
                data.append(("K562", "DREAM-RNN", run, id_r, snv_r, ood_r))
            except (KeyError, ValueError, TypeError):
                continue

    # We will expand this script as AlphaGenome results sync down
    df = pd.DataFrame(
        data, columns=["dataset", "model", "run_seed", "test_id_pr", "test_snv_pr", "test_ood_pr"]
    )
    df = df.sort_values(by=["dataset", "model", "run_seed"])
    os.makedirs("results/oracle_benchmarks/data", exist_ok=True)
    df.to_csv("results/oracle_benchmarks/data/k562_oracles.csv", index=False)
    print("Compiled Oracle benchmarks: results/oracle_benchmarks/data/k562_oracles.csv")


if __name__ == "__main__":
    parse_yeast_scaling()
    parse_k562_scaling()
    parse_oracle_benchmarks()
