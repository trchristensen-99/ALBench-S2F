"""CAMEO ``study.json`` for an architecture-search run.

Why a module and not a one-off script: the entry format asks for things that cannot be
reconstructed later -- the model file as text, the training script as text, the resolved
settings, the seed, the parent run, wall-clock time and a measured inference speed. The loop
already writes most of them next to each model (``r*_meta.json``, ``r*.npz``, ``config.json``,
``agent_logs/plan_r*.json``); this module is the one place that knows how to read that layout
and emit the four parts the format wants, so a run can be deposited without hand editing.

Three subcommands::

    python -m albench.arch_search.cameo export RUN_DIR [RUN_DIR ...] --out study.json
    python -m albench.arch_search.cameo speed  RUN_DIR            # measured seq/s, needs a GPU
    python -m albench.arch_search.cameo check  study.json

``export`` is CPU-only and reads nothing but the run directories. ``speed`` rebuilds each distinct
architecture from its stored code and times a forward pass, writing ``cameo_speed.json`` into the
run directory; ``export`` picks that file up if it is there and writes ``"not_reported"`` if it is
not, because the format forbids inventing a value. ``check`` is the format's own checklist, so a
file can be verified here without the CAMEO kit.

One study may cover several run directories (the replicates of one arm, say). Ids are derived from
content -- a model id is the hash of its code -- so the same architecture found twice in two runs
is one ``models[]`` entry with two ``runs[]``, and re-exporting never renumbers anything.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from albench.arch_search.harness_standard import FLANK_CTX, INPUT_LEN, INSERT_LEN, SHIFT_MAX

FORMAT_VERSION = "0.3"
MODULE_DIR = Path(__file__).resolve().parent

# The loop's own modules that the training script imports; the format asks for them as text so a
# reader can run the recipe. Keep this list in step with the imports at the top of the launcher.
RECIPE_FILES = (
    "arch_autoresearch.py",
    "harness_standard.py",
    "flank_context.py",
    "train_generated.py",
    "validate.py",
)

FAMILY_PATTERNS = (
    ("legnet-variant", r"legnet|LegNet"),
    ("ssm", r"\bMamba\b|\bS4\b|selective_scan"),
    ("recurrent", r"\bnn\.(GRU|LSTM|RNN)\b"),
    ("attention", r"MultiheadAttention|scaled_dot_product_attention|class\s+\w*Attention"),
    ("dilated-cnn", r"dilation\s*=\s*[^01]"),
    ("se-cnn", r"squeeze|SELayer|class\s+SE\b|_se\b"),
    ("residual-cnn", r"class\s+\w*Res|x\s*\+\s*h\b|\+\s*residual"),
    ("cnn", r"nn\.Conv1d"),
)

# Keys of ``train_config`` that are genuinely per-run settings rather than part of the recipe.
SETTING_KEYS = ("optimizer", "lr", "weight_decay", "batch_size", "epochs", "lr_schedule")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def family_of(code: str) -> str:
    for name, pat in FAMILY_PATTERNS:
        if re.search(pat, code or "", re.I):
            return name
    return "unknown"


def _finite(x: Any) -> float | None:
    """A metric the format will accept: a real number, or None for one that diverged."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d else float("nan")


# --------------------------------------------------------------------------- reading a run dir


def _load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _lineage(run_dir: Path) -> dict[str, dict]:
    """model_id -> {parent, hypothesis, move, lineage} from the agent loop's round plans.

    The single-call loop writes no plans, so this is empty for those runs and the caller falls
    back to the round structure, which is what the loop actually used to pick a parent.
    """
    out: dict[str, dict] = {}
    for p in sorted((run_dir / "agent_logs").glob("plan_r*.json")):
        plan = _load_json(p)
        for idea in plan.get("ideas", []) or []:
            mid = idea.get("model_id") or idea.get("id")
            if not mid:
                continue
            out[str(mid)] = {
                "parent": idea.get("parent"),
                "hypothesis": (idea.get("why") or idea.get("hypothesis") or "")[:800],
                "move": idea.get("move"),
                "lineage": idea.get("lineage"),
            }
    return out


def read_run(run_dir: str | Path) -> dict:
    """Everything one loop output directory holds, in the shape ``build_study`` wants."""
    d = Path(run_dir)
    cfg = _load_json(d / "config.json")
    speeds = _load_json(d / "cameo_speed.json")
    lineage = _lineage(d)
    labels_p = d / "labels.npz"
    labels = np.load(labels_p) if labels_p.exists() else None
    metas = []
    for mp in sorted(d.glob("r*_meta.json")):
        meta = _load_json(mp)
        if not meta.get("model_id"):
            continue
        meta["_preds"] = d / f"{meta['model_id']}.npz"
        meta["_lineage"] = lineage.get(meta["model_id"], {})
        metas.append(meta)
    failed = []
    fp = d / "failed_runs.jsonl"
    if fp.exists():
        for line in fp.read_text().splitlines():
            try:
                failed.append(json.loads(line))
            except Exception:
                continue
    return {
        "dir": d,
        "config": cfg,
        "metas": metas,
        "failed": failed,
        "labels": labels,
        "speeds": speeds,
    }


def _argv_value(cfg: dict, flag: str, default: str | None = None) -> str | None:
    argv = cfg.get("argv") or []
    return (
        argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else default
    )


# ------------------------------------------------------------------------------------ metrics


def _metrics(meta: dict, labels) -> tuple[dict, dict, list]:
    """(valid, test, diverged) for one model, recomputed from its stored predictions.

    The meta file carries only the headline Pearson; the per-set predictions are next to it, and a
    deposit is worth more with every evaluation set the run already scored.
    """
    valid = {"pearson": _finite(meta.get("val_pearson")), "mse": _finite(meta.get("val_mse"))}
    test: dict[str, float | None] = {"pearson": _finite(meta.get("test_oracle_pearson"))}
    diverged = [f"valid.{k}" for k, v in valid.items() if v is None]
    p = meta.get("_preds")
    if labels is not None and p is not None and Path(p).exists():
        try:
            z = np.load(p)
        except Exception:
            z = None
        if z is not None:
            for key in z.files:
                if not key.startswith("test_pred_"):
                    continue
                name = key[len("test_pred_") :]
                lk = f"oracle_{name}"
                if lk not in labels.files:
                    continue
                r = _finite(_pearson(z[key], labels[lk]))
                test[f"pearson.{name}"] = r
                if r is None:
                    diverged.append(f"test.pearson.{name}")
            # the variant-effect endpoints: a difference of two aligned prediction vectors
            for name, (a, b) in (
                ("snv", ("snv_alt", "snv_ref")),
                ("inversion", ("inversion", "genomic")),
                ("translocation", ("translocation", "genomic")),
            ):
                ka, kb = f"test_pred_{a}", f"test_pred_{b}"
                la, lb = f"oracle_{a}", f"oracle_{b}"
                if all(k in z.files for k in (ka, kb)) and all(k in labels.files for k in (la, lb)):
                    r = _finite(_pearson(z[ka] - z[kb], labels[la] - labels[lb]))
                    test[f"delta_pearson.{name}"] = r
                    if r is None:
                        diverged.append(f"test.delta_pearson.{name}")
    diverged += [f"test.{k}" for k, v in test.items() if v is None]
    return valid, test, sorted(set(diverged))


# ------------------------------------------------------------------------------- the four parts


def _data_block(cfg: dict, runs: list[dict], n_valid: int | None) -> dict:
    pool = _argv_value(cfg, "--reservoir_cache", "") or ""
    D = _argv_value(cfg, "--D", "not_reported")
    split_code = (
        "# chromosome holdout, fixed for the whole study (albench AGENTS.md):\n"
        "TEST_CHROMS = ('chr7', 'chr13')\n"
        "VAL_CHROMS = ('chr19', 'chr21', 'chrX')\n"
        "# training pools exclude both; the validation set used for selection is the\n"
        "# oracle-labelled V0c asset unless --val_cache names another.\n"
        "train = [s for s in library if s.chrom not in TEST_CHROMS + VAL_CHROMS]\n"
    )
    return {
        "name": "Gosai MPRA (K562), oracle-relabelled",
        "accession": os.path.basename(pool) or "not_reported",
        "input": f"{INSERT_LEN} bp insert with {FLANK_CTX} bp of real plasmid flank each side, "
        f"one-hot (4, {INPUT_LEN})",
        "output": "one activity value (log2 fold change, oracle ensemble label)",
        "examples": {
            "train": int(D) if str(D).isdigit() else "not_reported",
            "valid": n_valid if n_valid is not None else "not_reported",
            "test": "not_reported",
        },
        "preprocessing": "one-hot A,C,G,T over the flanked window; labels are the "
        "oracle_v3 10-fold ensemble mean",
        "splits": {
            "how": "held-out chromosomes: chr7/chr13 test, chr19/chr21/chrX validation",
            "code": split_code,
        },
    }


def _environment(cfg: dict, metas: list[dict]) -> dict:
    env = cfg.get("env") or cfg.get("environment") or {}
    gpus = collections.Counter(str(m.get("gpu")) for m in metas if m.get("gpu"))
    device = gpus.most_common(1)[0][0] if gpus else "not_reported"
    harness = (metas[0].get("harness") if metas else {}) or {}
    return {
        "python": env.get("python", "not_reported"),
        "framework": env.get("framework", "not_reported"),
        "packages": env.get("packages", {}) or {"note": "not_reported"},
        "training_hardware": {
            "device": device,
            "count": 1,
            "precision": harness.get("AL_PRECISION", "fp32"),
        },
        "inference_hardware": {
            "device": device,
            "threads": "not_reported",
            "batch": 256,
            "precision": "fp32",
        },
    }


def _recipe(cfg: dict, metas: list[dict]) -> dict:
    """One recipe: the loop's harness. Per-run choices live in each run's ``settings``."""
    harness = (metas[0].get("harness") if metas else {}) or {}
    defaults = collections.Counter(
        json.dumps({k: m.get("train_config", {}).get(k) for k in SETTING_KEYS}, sort_keys=True)
        for m in metas
        if m.get("train_config")
    )
    settings = json.loads(defaults.most_common(1)[0][0]) if defaults else {}
    rc_mode = harness.get("AL_RC_MODE", "consistency")
    aug = [
        {
            "step": "random shift over real plasmid flank",
            "max_bases": int(harness.get("AL_SHIFT_MAX", SHIFT_MAX)),
            "per": "batch",
        },
        {
            "step": "reverse complement",
            "mode": "single-pass augmentation (half the samples flipped)"
            if rc_mode == "aug"
            else "two-pass consistency (loss averaged over both strands)",
        },
    ]
    patience = harness.get("AL_PATIENCE", "8")
    onecycle_full = str(harness.get("AL_ONECYCLE_FULL", "0")) == "1"
    early = (
        False
        if str(patience) in ("0", "9999")
        else {
            "metric": "mse",
            "split": "valid",
            "patience": int(patience),
            "note": "a one-cycle schedule is never stopped by patience" if onecycle_full else "",
        }
    )
    script = MODULE_DIR / "arch_autoresearch.py"
    cap = _argv_value(cfg, "--base_cap_min")
    return {
        "name": "albench architecture-search harness",
        "settings": {
            **settings,
            "per_model_wallclock_cap_min_h100_equivalent": float(cap) if cap else None,
            "ema_decay": float(harness.get("AL_EMA", 0) or 0),
        },
        "loss": {
            "kind": "mse",
            "terms": [{"output": "y", "weight": 1.0}],
            "reduction": "mean over examples, averaged over the two strands"
            if rc_mode != "aug"
            else "mean over examples",
        },
        "schedules": {
            "onecycle": {"max_lr": "settings.lr", "cycle_momentum": False},
            "cosine": {"T_max": "settings.epochs"},
            "plateau": {"factor": 0.5, "patience": 3},
        },
        "early_stopping": early,
        "checkpoint": "best_validation",
        "shuffle": True,
        "augmentation": aug,
        "init": "random",
        "definitions": {
            "AL_RC_MODE": "consistency = forward and reverse-complement losses are averaged "
            "(two passes); aug = half the samples are flipped (one pass)",
            "AL_ONECYCLE_FULL": "1 = patience never stops a one-cycle schedule",
            "per_model_wallclock_cap_min_h100_equivalent": "training is stopped at this many H100-minutes (x5 on a V100) and the best "
            "validation epoch so far is kept",
        },
        "code": {
            "entry_point": "main",
            "command": "python -m albench.arch_search.arch_autoresearch "
            + " ".join((cfg.get("argv") or [])[1:]),
            "script": script.read_text() if script.exists() else "not_reported",
            "files": {
                name: (MODULE_DIR / name).read_text()
                for name in RECIPE_FILES
                if name != "arch_autoresearch.py" and (MODULE_DIR / name).exists()
            },
        },
    }


def build_study(
    run_dirs: list[str | Path],
    *,
    experiment_id: str,
    title: str,
    summary: str,
    depositor: dict,
    code_url: str = "not_reported",
    publication: str = "not_reported",
    license_: str = "MIT",
    created: str | None = None,
    noise_floor: float | None = None,
) -> dict:
    runs_in = [read_run(d) for d in run_dirs]
    runs_in = [r for r in runs_in if r["metas"]]
    if not runs_in:
        raise SystemExit("no run directory held r*_meta.json files")
    cfg0 = runs_in[0]["config"]

    models: dict[str, dict] = {}
    runs: list[dict] = []
    n_valid = None
    for ri, run in enumerate(runs_in):
        lbl = run["labels"]
        if n_valid is None and lbl is not None and "val_labels" in lbl.files:
            n_valid = int(len(lbl["val_labels"]))
        tag = Path(run["dir"]).name
        for meta in run["metas"]:
            code = meta.get("source") or ""
            mid = f"m-{_sha(code)[:12]}" if code else f"m-{tag}-{meta['model_id']}"
            if mid not in models:
                models[mid] = {
                    "id": mid,
                    "name": (meta.get("idea") or meta["model_id"])[:80],
                    "family": family_of(code),
                    "parent": None,
                    "inputs": {"x": {"kind": "one-hot DNA", "shape": [4, INPUT_LEN]}},
                    "outputs": {"y": {"kind": "activity"}},
                    "code": code or "not_reported",
                    "inference_speed": {
                        "seq_per_s": run["speeds"].get(mid)
                        or run["speeds"].get(meta["model_id"], "not_reported")
                    },
                }
            lin = meta.get("_lineage") or {}
            idx = (
                int(re.search(r"arch_(\d+)", meta["model_id"]).group(1))
                if re.search(r"arch_(\d+)", meta["model_id"])
                else 0
            )
            valid, test, diverged = _metrics(meta, lbl)
            tc = meta.get("train_config") or {}
            runs.append(
                {
                    "id": f"{experiment_id}-{tag}-{meta['model_id']}",
                    "model": mid,
                    "training": "harness",
                    "seed": meta.get("seed", 1000 + idx),
                    "settings": {k: tc[k] for k in SETTING_KEYS if k in tc},
                    "history": {
                        "lineage": lin.get("lineage") or tag,
                        "parent": lin.get("parent"),
                        "move": lin.get("move")
                        or ("baseline" if meta.get("round") == 0 else "structural_change"),
                        "hypothesis": (meta.get("why") or lin.get("hypothesis") or "")[:800],
                        "predicted_effect": "not_reported",
                    },
                    "performance": {
                        "status": "done",
                        "score": valid["pearson"],
                        "valid": valid,
                        "test": test,
                        "diverged": diverged,
                        "parameters": meta.get("n_params"),
                        "training_minutes": (
                            float(meta["train_time_sec"]) / 60.0
                            if meta.get("train_time_sec")
                            else None
                        ),
                        "best_epoch": meta.get("best_epoch", "not_reported"),
                        "epochs_run": meta.get("epochs_trained"),
                    },
                }
            )
        for i, f in enumerate(run["failed"]):
            code = f.get("source") or ""
            mid = f"m-{_sha(code)[:12]}" if code else f"m-{tag}-failed-{i:03d}"
            models.setdefault(
                mid,
                {
                    "id": mid,
                    "name": (f.get("idea") or f"failed {i}")[:80],
                    "family": family_of(code),
                    "parent": None,
                    "inputs": {"x": {"kind": "one-hot DNA", "shape": [4, INPUT_LEN]}},
                    "outputs": {"y": {"kind": "activity"}},
                    "code": code or "not_reported",
                    "inference_speed": {"seq_per_s": "not_reported"},
                },
            )
            runs.append(
                {
                    "id": f"{experiment_id}-{tag}-failed-{i:03d}",
                    "model": mid,
                    "training": "harness",
                    "seed": f.get("seed", "not_reported"),
                    "settings": {
                        k: (f.get("train_config") or {})[k]
                        for k in SETTING_KEYS
                        if k in (f.get("train_config") or {})
                    },
                    "history": {
                        "lineage": tag,
                        "parent": None,
                        "move": "structural_change",
                        "hypothesis": (f.get("idea") or "")[:800],
                        "predicted_effect": "not_reported",
                    },
                    "performance": {
                        "status": f.get("status", "failed"),
                        "score": None,
                        "valid": {},
                        "test": {},
                        "diverged": [],
                        "reason": f.get("reason", "not_reported"),
                        "parameters": f.get("n_params", "not_reported"),
                        "training_minutes": f.get("train_minutes", "not_reported"),
                        "best_epoch": "not_reported",
                        "epochs_run": "not_reported",
                    },
                }
            )

    done = [
        r
        for r in runs
        if r["performance"]["status"] == "done" and r["performance"]["score"] is not None
    ]
    best = max(done, key=lambda r: r["performance"]["score"], default=None)
    all_metas = [m for r in runs_in for m in r["metas"]]
    experiment = {
        "id": experiment_id,
        "title": title,
        "summary": summary,
        "license": license_,
        "created": created or "not_reported",
        "code_url": code_url,
        "publication": publication,
        "depositor": depositor,
        "run_by": {
            "kind": "agent",
            "tool": "albench.arch_search.arch_autoresearch",
            "model": cfg0.get("LLM_MODEL", "not_reported"),
        },
        "objective": {
            "goal": "Find architectures that predict MPRA activity better than the tuned "
            "MPRA-LegNet at this training-set size, and that add to it in an ensemble.",
            "optimize": {"metric": "pearson", "split": "valid", "direction": "maximize"},
        },
        "data": _data_block(cfg0, runs, n_valid),
        "evaluation": {
            "primary_metric": "pearson",
            "direction": "maximize",
            "selection_split": "valid",
            "test_policy": "test sets are never shown to the proposer and never used for "
            "selection; they are scored once per trained model",
            "noise_floor": noise_floor if noise_floor is not None else "not_reported",
            "metrics": {
                "pearson": "Pearson r between prediction and oracle label over the set",
                "mse": "mean squared error on the validation set",
                "delta_pearson.snv": "Pearson r between predicted and oracle alt-minus-ref "
                "difference over the SNV pairs",
                "delta_pearson.inversion": "Pearson r between predicted and oracle "
                "perturbed-minus-parent difference",
            },
        },
        "environment": _environment(cfg0, all_metas),
        "search": {
            "method": "LLM proposal loop (propose / code / evaluate / reflect), "
            "validation-only feedback",
            "budget": {
                "rounds": _argv_value(cfg0, "--rounds"),
                "per_round": _argv_value(cfg0, "--per_round"),
                "budget_hours": _argv_value(cfg0, "--budget_hours"),
                "runs_recorded": len(runs),
            },
            "lineages": [
                {
                    "id": Path(r["dir"]).name,
                    "name": Path(r["dir"]).name,
                    "family": "architecture search",
                    "parents": [],
                }
                for r in runs_in
            ],
        },
        "conclusions": {
            "best_run": best["id"] if best else "not_reported",
            "findings": [],
            "open_questions": [],
        },
    }
    return {
        "format_version": FORMAT_VERSION,
        "experiment": experiment,
        "models": list(models.values()),
        "training": {"harness": _recipe(cfg0, all_metas)},
        "runs": runs,
    }


# ------------------------------------------------------------------------------------- checking


def check(study: dict) -> int:
    """The format's own checklist. Prints the two lines the spec asks a depositor to confirm."""
    problems: list[str] = []

    def need(cond: bool, msg: str) -> None:
        if not cond:
            problems.append(msg)

    need(
        study.get("format_version") == FORMAT_VERSION,
        f"format_version is {study.get('format_version')!r}, not {FORMAT_VERSION!r}",
    )
    for part in ("experiment", "models", "training", "runs"):
        need(part in study, f"missing part {part!r}")
    exp = study.get("experiment", {})
    env = exp.get("environment", {})
    need(bool(env.get("packages")), "environment.packages is empty")
    need(
        bool(env.get("training_hardware", {}).get("device")),
        "environment.training_hardware.device is missing",
    )
    need(
        bool(env.get("inference_hardware", {}).get("device")),
        "environment.inference_hardware.device is missing",
    )
    splits = exp.get("data", {}).get("splits", {})
    need(bool(splits.get("code") or splits.get("files")), "data.splits has neither code nor files")

    models = {m["id"]: m for m in study.get("models", [])}
    for m in models.values():
        need(bool(m.get("code")) and m["code"] != "not_reported", f"model {m['id']} has no code")
    recipes = study.get("training", {})
    for rid, rec in recipes.items():
        need(
            bool(rec.get("code", {}).get("script")) and rec["code"]["script"] != "not_reported",
            f"recipe {rid} has no script",
        )
        for field in ("settings", "loss", "checkpoint", "augmentation", "init"):
            need(field in rec, f"recipe {rid} is missing {field}")

    finished = 0
    with_speed = set()
    for r in study.get("runs", []):
        need(r.get("model") in models, f"run {r.get('id')} names an unknown model")
        need(r.get("training") in recipes, f"run {r.get('id')} names an unknown recipe")
        need(r.get("seed") is not None, f"run {r.get('id')} has no seed")
        p = r.get("performance", {})
        if p.get("status") == "done":
            finished += 1
            with_speed.add(r["model"])
            for field in ("score", "parameters", "training_minutes"):
                need(p.get(field) is not None, f"run {r['id']} has no {field}")
            need(bool(p.get("valid")) or bool(p.get("test")), f"run {r['id']} has no metrics")
            for name, v in list(p.get("valid", {}).items()) + list(p.get("test", {}).items()):
                if v is None:
                    need(
                        any(name in d for d in p.get("diverged", [])),
                        f"run {r['id']} metric {name} is null but not listed in diverged",
                    )
                else:
                    need(np.isfinite(v), f"run {r['id']} metric {name} is not a finite number")
    for mid in with_speed:
        sp = models[mid].get("inference_speed", {}).get("seq_per_s")
        need(
            isinstance(sp, (int, float)),
            f"model {mid} has a finished run but no measured inference_speed",
        )

    ok_exp = not [p for p in problems if p.startswith(("environment", "data", "format", "missing"))]
    ok_model = not [p for p in problems if p.startswith("model ")]
    ok_train = not [p for p in problems if p.startswith("recipe ")]
    ok_perf = not [p for p in problems if p.startswith("run ")]
    for p in problems:
        print(f"  gap: {p}", file=sys.stderr)
    print(
        f"Sufficient? experiment: {'yes' if ok_exp else 'no'} · "
        f"model: {'yes' if ok_model else 'no'} · training: {'yes' if ok_train else 'no'} · "
        f"performance: {'yes' if ok_perf else 'no'}"
    )
    if not problems:
        print("RESULT: ready to stage.")
        return 0
    print(f"RESULT: {len(problems)} gaps, {finished} finished runs.")
    return 1


# ---------------------------------------------------------------------------- inference speed


def measure_speed(run_dir: str | Path, *, batch: int = 256, repeats: int = 5) -> dict:
    """Rebuild every distinct architecture in a run and time a no-grad forward pass.

    The format wants examples per second at a fixed batch size after a warm-up, taking the fastest
    of several timed passes, so that is what this does -- on whatever device is available, which
    the caller records as ``inference_hardware``.
    """
    import time

    import torch

    from albench.arch_search.validate import build_and_check

    run = read_run(run_dir)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out: dict[str, float] = {}
    for meta in run["metas"]:
        code = meta.get("source") or ""
        if not code:
            continue
        mid = f"m-{_sha(code)[:12]}"
        if mid in out:
            continue
        model, verdict = build_and_check(code, seq_len=INPUT_LEN, max_params=10**9)
        if model is None:
            continue
        model = model.to(dev).eval()
        x = torch.zeros(batch, 4, INPUT_LEN, device=dev)
        with torch.no_grad():
            for _ in range(2):
                model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            best = float("inf")
            for _ in range(repeats):
                t0 = time.perf_counter()
                model(x)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                best = min(best, time.perf_counter() - t0)
        out[mid] = round(batch / best, 1)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    p = Path(run_dir) / "cameo_speed.json"
    p.write_text(json.dumps({"device": str(dev), "batch": batch, **out}, indent=1))
    print(f"wrote {p} ({len(out)} architectures, device {dev})")
    return out


# ------------------------------------------------------------------------------------- the CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("export", help="write study.json for one or more run directories")
    ex.add_argument("run_dirs", nargs="+")
    ex.add_argument("--out", required=True)
    ex.add_argument("--id", required=True, dest="experiment_id")
    ex.add_argument("--title", required=True)
    ex.add_argument("--summary", default="")
    ex.add_argument("--created", default=None, help="ISO date of the experiment")
    ex.add_argument("--depositor-name", default="not_reported")
    ex.add_argument("--depositor-orcid", default="not_reported")
    ex.add_argument("--depositor-lab", default="not_reported")
    ex.add_argument("--depositor-institution", default="not_reported")
    ex.add_argument("--depositor-contact", default="not_reported")
    ex.add_argument("--code-url", default="not_reported")
    ex.add_argument("--publication", default="not_reported")
    ex.add_argument("--noise-floor", type=float, default=None)

    sp = sub.add_parser("speed", help="measure inference speed into cameo_speed.json")
    sp.add_argument("run_dir")
    sp.add_argument("--batch", type=int, default=256)

    ck = sub.add_parser("check", help="run the format's checklist on a study.json")
    ck.add_argument("study")

    a = ap.parse_args(argv)
    if a.cmd == "speed":
        measure_speed(a.run_dir, batch=a.batch)
        return 0
    if a.cmd == "check":
        return check(json.loads(Path(a.study).read_text()))
    study = build_study(
        a.run_dirs,
        experiment_id=a.experiment_id,
        title=a.title,
        summary=a.summary,
        created=a.created,
        depositor={
            "name": a.depositor_name,
            "orcid": a.depositor_orcid,
            "lab": a.depositor_lab,
            "institution": a.depositor_institution,
            "contact": a.depositor_contact,
        },
        code_url=a.code_url,
        publication=a.publication,
        noise_floor=a.noise_floor,
    )
    Path(a.out).write_text(json.dumps(study, indent=1, default=str))
    n_done = sum(1 for r in study["runs"] if r["performance"]["status"] == "done")
    print(
        f"wrote {a.out}: {len(study['models'])} models, {len(study['runs'])} runs "
        f"({n_done} finished)"
    )
    return check(study)


if __name__ == "__main__":
    raise SystemExit(main())
