"""Aggregate all HP-search trial results into one DataFrame for plotting.

Walks both:
  - results/preflight/hpsearch/{strategy}_{arch}_d{D}/         (Ray Tune)
  - results/preflight/hpsearch/autoresearch/{arch}_d{D}/...    (AutoResearch)

For each trial, reads:
  - run/result.json (run_single.py summary) OR result.json (legacy)
  - extracts: strategy, arch, d_train, lr, batch_size, weight_decay,
    dropout, width, depth, val_loss, test_loss, gpu_hrs, n_params

Outputs:
  - results/preflight/hpsearch/all_trials.csv
  - results/preflight/hpsearch/all_trials.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "results/preflight/hpsearch"


def _extract_abstract_hp(hp: dict) -> dict:
    """Reconstruct abstract HPs from run_single.py's resolved hp dict."""
    out = {
        "lr": hp.get("lr"),
        "batch_size": hp.get("batch_size"),
        "weight_decay": hp.get("weight_decay"),
    }
    # Per-arch translation
    if "block_sizes" in hp:  # LegNet
        bs = hp["block_sizes"]
        out["width"] = bs[-1] if bs else None
        out["depth"] = len(bs)
        out["dropout"] = hp.get("dropout")
        out["arch"] = "legnet"
    elif "hidden_dim" in hp:  # DREAM-RNN
        out["width"] = hp.get("hidden_dim")
        out["depth"] = hp.get("num_lstm_layers", 1)
        out["dropout"] = hp.get("dropout_cnn", hp.get("dropout_lstm", 0))
        out["arch"] = "dream_rnn"
    elif "embedding_dim" in hp:  # DREAM-ATTN
        out["width"] = hp.get("embedding_dim")
        out["depth"] = hp.get("num_blocks", 4)
        out["dropout"] = hp.get("first_block_dropout", 0)
        out["arch"] = "dream_attn"
    return out


def _load_trial_result(result_path: Path, strategy: str, source: str) -> dict | None:
    """Load one result.json into a flat row dict."""
    try:
        summary = json.loads(result_path.read_text())
    except Exception:
        return None
    hp = summary.get("hp", {})
    abstract = _extract_abstract_hp(hp)
    row = {
        "source": source,
        "strategy": strategy,
        "arch": abstract.get("arch") or summary.get("arch"),
        "d_train": summary.get("d_train"),
        "lr": abstract.get("lr"),
        "batch_size": abstract.get("batch_size"),
        "weight_decay": abstract.get("weight_decay"),
        "dropout": abstract.get("dropout"),
        "width": abstract.get("width"),
        "depth": abstract.get("depth"),
        "val_loss": summary.get("best_val_mse"),
        "test_loss": summary.get("test_mse_at_best_val"),
        "best_epoch": summary.get("best_epoch"),
        "n_params": summary.get("n_params"),
        "gpu_hrs": summary.get("gpu_hrs"),
        "augmentations": summary.get("augmentations", "rev_complement"),
        "trial_path": str(result_path.parent.relative_to(REPO)),
    }
    return row


def aggregate_raytune() -> list[dict]:
    """Walk Ray Tune output directories (one per (strategy, arch, D) cell)."""
    rows = []
    for cell_dir in (
        sorted(ROOT.glob("*_legnet_d*"))
        + sorted(ROOT.glob("*_dream_rnn_d*"))
        + sorted(ROOT.glob("*_dream_attn_d*"))
    ):
        if cell_dir.name.startswith("autoresearch"):
            continue
        cfg_path = cell_dir / "search_config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
            strategy = cfg.get("strategy", "?")
        except Exception:
            strategy = cell_dir.name.split("_")[0]
        # Ray Tune trials are under cell_dir/{run_name}/trainable_*/run/result.json
        for run_subdir in cell_dir.iterdir():
            if not run_subdir.is_dir():
                continue
            for trial_dir in run_subdir.glob("trainable_*"):
                result_path = trial_dir / "run" / "result.json"
                if not result_path.exists():
                    # Try direct path (smoke test legacy)
                    result_path = trial_dir / "result.json"
                if not result_path.exists():
                    continue
                row = _load_trial_result(result_path, strategy, source="raytune")
                if row:
                    rows.append(row)
    return rows


def aggregate_autoresearch() -> list[dict]:
    """Walk AutoResearch output directories."""
    rows = []
    ar_root = ROOT / "autoresearch"
    if not ar_root.exists():
        return rows
    for cell_dir in sorted(ar_root.iterdir()):
        if not cell_dir.is_dir():
            continue
        for round_dir in sorted(cell_dir.glob("round_*")):
            for agent_dir in sorted(round_dir.glob("agent_*")):
                role = agent_dir.name.split("_")[-1]
                strategy = f"autoresearch_{role}"
                for trial_dir in agent_dir.iterdir():
                    if not trial_dir.is_dir():
                        continue
                    result_path = trial_dir / "result.json"
                    if not result_path.exists():
                        continue
                    row = _load_trial_result(result_path, strategy, source="autoresearch")
                    if row:
                        row["round"] = round_dir.name
                        rows.append(row)
    return rows


def aggregate_agent_rounds() -> list[dict]:
    """Walk per-round agent_* dirs produced by `_convert_agent_proposals` +
    `parallel_gpu_runner` (subprocess-style; result.json sits directly in each
    trial dir).

    Dir layout: results/preflight/hpsearch/agent_{arch}_d{D}{_rN}/{label}/result.json
    """
    rows = []
    for cell_dir in sorted(ROOT.glob("agent_*")):
        if not cell_dir.is_dir():
            continue
        # Parse strategy label from dir name (e.g. agent_legnet_d20000_r2 → agent_r2)
        name = cell_dir.name
        round_tag = ""
        for suffix in ("_r4", "_r3", "_r2", "_r1"):
            if name.endswith(suffix):
                round_tag = suffix.lstrip("_")
                break
        strategy = f"agent_{round_tag}" if round_tag else "agent"
        for trial_dir in cell_dir.iterdir():
            if not trial_dir.is_dir():
                continue
            result_path = trial_dir / "result.json"
            if not result_path.exists():
                continue
            row = _load_trial_result(result_path, strategy, source="agent_round")
            if row:
                row["round"] = round_tag
                rows.append(row)
    return rows


def main():
    rows = aggregate_raytune() + aggregate_autoresearch() + aggregate_agent_rounds()
    if not rows:
        print("No completed trials found yet.")
        return
    df = pd.DataFrame(rows)
    out_csv = ROOT / "all_trials.csv"
    out_json = ROOT / "all_trials.json"
    df.to_csv(out_csv, index=False)
    df.to_json(out_json, orient="records", indent=2)

    print(f"Aggregated {len(df)} trials → {out_csv}")
    print("\nBy strategy × arch × D:")
    if "arch" in df.columns:
        summary = (
            df.groupby(["strategy", "arch", "d_train"])
            .agg(n=("val_loss", "count"), best_val=("val_loss", "min"))
            .reset_index()
            .sort_values(["arch", "d_train", "best_val"])
        )
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
