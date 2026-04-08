import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter


def format_func(value, tick_number):
    return f"{value:g}"


def generate_scaling_plots():
    os.makedirs("results/exp0_scaling/plots", exist_ok=True)

    yeast_df = pd.read_csv("results/exp0_scaling/data/yeast_baseline.csv")
    k562_df = pd.read_csv("results/exp0_scaling/data/k562_baseline.csv")

    y_agg = yeast_df.groupby("fraction")["test_random_pr"].agg(["mean", "std"]).reset_index()
    k_agg = k562_df.groupby("fraction")["test_id_pr"].agg(["mean", "std"]).reset_index()

    # 1. Plot Yeast
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        y_agg["fraction"],
        y_agg["mean"],
        yerr=y_agg["std"].fillna(0),
        label="Yeast (Test Random)",
        fmt="-o",
        capsize=5,
        color="blue",
    )
    plt.xscale("log")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
    plt.xlabel("Fraction of Training Data")
    plt.ylabel("Pearson R")
    plt.title("Yeast Baseline Performance Scaling (DREAM-RNN)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig("results/exp0_scaling/plots/yeast_scaling.png", dpi=300, bbox_inches="tight")
    plt.savefig(
        "/Users/christen/.gemini/antigravity/brain/f7b84075-24bc-4953-a4f5-52893c1b75da/yeast_scaling.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Plot K562
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        k_agg["fraction"],
        k_agg["mean"],
        yerr=k_agg["std"].fillna(0),
        label="K562 (Test ID)",
        fmt="-s",
        capsize=5,
        color="orange",
    )
    plt.xscale("log")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
    plt.xlabel("Fraction of Training Data")
    plt.ylabel("Pearson R")
    plt.title("K562 Baseline Performance Scaling (DREAM-RNN)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig("results/exp0_scaling/plots/k562_scaling.png", dpi=300, bbox_inches="tight")
    plt.savefig(
        "/Users/christen/.gemini/antigravity/brain/f7b84075-24bc-4953-a4f5-52893c1b75da/k562_scaling.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("Plots saved to results/exp0_scaling/plots/ and artifact directory.")


if __name__ == "__main__":
    generate_scaling_plots()
