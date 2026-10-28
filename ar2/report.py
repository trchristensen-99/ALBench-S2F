"""Final report (spec 16.4) and menu export.

``write_report`` writes one markdown document from the ledger alone. ``export_menu`` writes, for each menu family, the model, recipe, card and tags to one directory for
downstream retraining, plus ``menu_ar2.json`` in the format of ``albench/ensemble/menus/menu_v1.json`` so the existing per-size retraining and ensembling can train the
archive's models unchanged (the recipe maps onto the menu entry's ``train_config``; the original ar2 recipe is kept under ``ar2_recipe``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ar2 import config as cfgmod
from ar2.memory import archive, final
from ar2.memory.db import DB


def _f(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _sd(xs):
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join("" if c is None else str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def menu(db: DB, cfg: dict) -> list[dict]:
    return archive.menu_families(db, cfg)


def write_report(ws: str | Path, out: str | Path | None = None) -> Path:
    ws = Path(ws)
    db = DB(ws / "ar2.db")
    cfg = cfgmod.load_frozen(ws)
    archive.rebuild(db, cfg)
    fams = {r["family"]: r for r in db.families()}
    runs = db.runs_for()
    calls = db.calls()
    sig_meas = archive.sigma_seed(db, cfg)
    sig = sig_meas if sig_meas is not None else cfg["controller"]["sigma_seed_prior"]
    se_diff = sig * math.sqrt(2 / 3)
    thr = max(cfg["controller"]["noise_floor"], 2 * se_diff)
    L: list[str] = []
    L.append(f"# ar2 search report: {ws.name}\n")
    # ---- run summary -------------------------------------------------------------------------------------------------------
    L.append("## Run summary\n")
    bud = cfg["controller"]["budget"]
    rows = []
    for ph, key in archive.PHASE_BUDGET_KEY.items():
        rows.append([ph, key, bud.get(key), _f(db.spend(phase=ph), 1)])
    L.append(_table(["phase", "name", "budget (run-equivalents)", "spent"], rows))
    models = sorted({c["model_id"] for c in calls if c.get("model_id")})
    by_mode: dict[str, int] = {}
    for c in calls:
        by_mode[c["mode"]] = by_mode.get(c["mode"], 0) + 1
    L.append(
        f"\nLLM calls: {len(calls)} ({', '.join(f'{k} {v}' for k, v in sorted(by_mode.items())) or 'none'}). Prompt version: `{db.meta_get('prompt_version')}`. "
        f"Proposer model ids seen: {', '.join(models) or 'none'}. Runs: {len(runs)} (D = {cfg['data']['D']:,}, pool `{db.meta_get('pool_id') or cfg['data'].get('pool_id')}`)."
    )
    if db.meta_get("proposer_changed"):
        L.append(
            f"\n**proposer_changed:** {db.meta_get('proposer_changed')}. Calls made under different model ids are not exchangeable; split the results by `calls.model_id`."
        )
    if db.meta_get("stop_requested") == "1":
        L.append("The controller's stop rule fired.")
    # ---- coverage ----------------------------------------------------------------------------------------------------------
    L.append("\n## Coverage\n")
    L.append(
        _table(
            ["family", "candidates", "state", "runs"],
            [[f, r["n_candidates"], r["state"], r["n_runs"]] for f, r in sorted(fams.items())],
        )
    )
    empty = [f for f in archive.TAXONOMY if f not in fams]
    if empty:
        L.append("\nNo candidate in: " + ", ".join(empty) + ".")
    # ---- archive -----------------------------------------------------------------------------------------------------------
    L.append("\n## Archive (confirmed families)\n")
    fin = final.read_final_scores(db.path)
    conf = [r for r in fams.values() if r["confirmed_mean"] is not None]
    conf.sort(key=lambda r: -r["confirmed_mean"])
    rows = []
    for r in conf:
        cand = db.candidate(r["best_code_hash"]) or {}
        vs = [x["val_select_r"] for x in fin if x["code_hash"] == r["best_code_hash"]]
        rec = (
            next(
                (
                    x["recipe"]
                    for x in runs
                    if x["code_hash"] == r["best_code_hash"]
                    and x["recipe_hash"] == r["best_recipe_hash"]
                ),
                {},
            )
            or {}
        )
        rows.append(
            [
                r["family"],
                _f(r["confirmed_mean"]),
                _f(r["confirmed_sd"]),
                r["n_confirm"],
                _f(sum(vs) / len(vs)) if vs else "n/a",
                cand.get("n_params"),
                _f(cand.get("throughput"), 0),
                f"{rec.get('optimizer', '?')} lr {rec.get('lr', '?')} wd {rec.get('weight_decay', '?')} bs {rec.get('batch_size', '?')} {rec.get('schedule', '?')}",
            ]
        )
    L.append(
        _table(
            [
                "family",
                "val_loop mean",
                "sd",
                "seeds",
                "val_select",
                "params",
                "seq/s",
                "best recipe",
            ],
            rows,
        )
        if rows
        else "No family has been confirmed yet."
    )
    # ---- variance split ----------------------------------------------------------------------------------------------------
    L.append("\n## Variance split\n")
    means = [r["confirmed_mean"] for r in conf]
    between = _sd(means)
    within_recipes = [r["recipe_spread"] for r in fams.values() if r["recipe_spread"] is not None]
    across_recipes = (
        (sum(x * x for x in within_recipes) / len(within_recipes)) ** 0.5
        if within_recipes
        else None
    )
    L.append(
        _table(
            ["source", "standard deviation"],
            [
                ["between families (confirmed means)", _f(between)],
                ["within families, across recipes (single seed)", _f(across_recipes)],
                [
                    "across seeds (pooled confirmation runs"
                    + ("" if sig_meas is not None else "; prior, fewer than 3 confirmed families")
                    + ")",
                    _f(sig),
                ],
            ],
        )
    )
    if between is not None and between < 2 * sig:
        L.append(
            "\nThe between-family spread is under twice the seed spread: **families are not distinguishable at this data size.**"
        )
    if across_recipes is not None and between is not None and across_recipes > between:
        L.append(
            "Recipe spread within families exceeds the spread between families: training settings matter more than architecture at this data size."
        )
    # ---- accuracy against cost ---------------------------------------------------------------------------------------------
    L.append("\n## Accuracy against cost\n")
    L.append(
        _table(
            ["family", "val_loop mean", "params", "seq/s"], [[r[0], r[1], r[5], r[6]] for r in rows]
        )
        if rows
        else "n/a"
    )
    # ---- procedure screen --------------------------------------------------------------------------------------------------
    L.append("\n## Procedure screen\n")
    adopt = archive.decisions_notes(db, "screen_adoption")
    if adopt:
        for n in adopt:
            L.append(
                f"Adopted: {', '.join(n.get('adopted') or []) or 'none'}. Effects: "
                + ", ".join(f"{k} {v:+.4f}" for k, v in (n.get("effects") or {}).items())
            )
    else:
        L.append(
            "The procedure screen was not run in this search (budget 0): losses and training procedures stay at the defaults (MSE, AdamW, no EMA)."
        )
    # ---- lessons -----------------------------------------------------------------------------------------------------------
    L.append("\n## Lessons\n")
    ls = db.lessons(status=None)
    L.extend([f"- ({x['status']}) {x['statement']}" for x in ls] or ["None recorded."])
    # ---- search health -----------------------------------------------------------------------------------------------------
    L.append("\n## Search health\n")
    explore = [
        c
        for c in calls
        if c["mode"] == "EXPLORE"
        and isinstance(c.get("assignment"), dict)
        and c["assignment"].get("outcome")
    ]
    if explore:
        nov = [
            1
            if (
                c["assignment"]["outcome"].get("landed_family")
                == c["assignment"]["outcome"].get("assigned_family")
            )
            else 0
            for c in explore
        ]
        L.append(
            f"Novelty rate (EXPLORE candidates landing in their assigned family): {sum(nov) / len(nov):.2f} over {len(nov)} calls; collapse events: "
            f"{sum(1 for c in explore if c['assignment']['outcome'].get('collapse'))}."
        )
    gate: dict[str, int] = {}
    for c in db.candidates():
        if c.get("gate_code"):
            gate[c["gate_code"]] = gate.get(c["gate_code"], 0) + 1
    ncand = len(db.candidates())
    L.append(
        f"Candidates: {ncand}; gate failures by code: {gate or 'none'}; repair calls: {by_mode.get('REPAIR', 0)}; runs censored by the time cap: {sum(1 for r in runs if r['status'] == 'time_cap')}; "
        f"diverged: {sum(1 for r in runs if r['status'] == 'diverged')}; exploded: {sum(1 for r in runs if r['status'] == 'exploded')}; infrastructure errors: "
        f"{sum(1 for r in runs if r['status'] == 'error')}."
    )
    cat = {c: db.spend(phase="M", category=c) for c in ("explore", "refine", "deep_tune")}
    tot = sum(cat.values()) or 1.0
    L.append(
        "Main-search budget by category: "
        + ", ".join(f"{k} {v:.1f} ({v / tot:.0%})" for k, v in cat.items())
        + "."
    )
    sets = db.decisions("SET")
    guards = [
        d
        for d in db.decisions("NOTE")
        if isinstance(d.get("payload"), dict) and d["payload"].get("kind") == "guard"
    ]
    if sets or guards:
        L.append("Balance changes and guard firings:")
        L.extend([f"- step {d['step']}: {d['reason']}" for d in sets + guards])
    # ---- selection check ---------------------------------------------------------------------------------------------------
    L.append("\n## Selection check (val_loop against val_select)\n")
    mf = menu(db, cfg)
    sel_rows, drops, order_loop, order_sel = [], [], [], []
    for r in mf:
        vs = [x["val_select_r"] for x in fin if x["code_hash"] == r["best_code_hash"]]
        if vs:
            m = sum(vs) / len(vs)
            drops.append(r["confirmed_mean"] - m)
            order_loop.append((r["confirmed_mean"], r["family"]))
            order_sel.append((m, r["family"]))
            sel_rows.append(
                [r["family"], _f(r["confirmed_mean"]), _f(m), f"{r['confirmed_mean'] - m:+.4f}"]
            )
    if sel_rows:
        L.append(_table(["family", "val_loop", "val_select", "drop"], sel_rows))
        mean_drop = sum(drops) / len(drops)
        changed = max(order_loop)[1] != max(order_sel)[1]
        if mean_drop > 0.005 or changed:
            L.append(
                f"\n**Selection overfitting:** mean drop {mean_drop:+.4f}{' and the top family changed' if changed else ''}."
            )
        else:
            L.append(f"\nMean drop {mean_drop:+.4f}; the top family is unchanged.")
    else:
        L.append("Phase D has not scored any family on val_select.")
    # ---- verdict -----------------------------------------------------------------------------------------------------------
    L.append("\n## Verdict\n")
    if mf:
        L.append(
            "Menu families: "
            + ", ".join(
                f"{r['family']} ({r['confirmed_mean']:.4f}{', flagged outside the margin' if r.get('flagged') else ''})"
                for r in mf
            )
            + "."
        )
    anchors = [r["confirmed_mean"] for r in conf if r["family"] in archive.ANCHORS]
    others = [r for r in conf if r["family"] not in archive.ANCHORS]
    if anchors and others:
        best = max(others, key=lambda r: r["confirmed_mean"])
        gain = best["confirmed_mean"] - max(anchors)
        if gain > thr:
            L.append(
                f"{best['family']} beats the best anchor by {gain:+.4f}, more than the noise threshold {thr:.4f}."
            )
        else:
            L.append(
                f"No new family beats the best anchor by more than the noise threshold ({thr:.4f}); best new family {best['family']} differs by {gain:+.4f}. **The space is flat at this data size.**"
            )
    elif not conf:
        L.append("Nothing is confirmed yet; no verdict.")
    path = Path(out) if out else ws / "report.md"
    path.write_text("\n".join(L) + "\n")
    db.close()
    return path


# --------------------------------------------------------------------------------------------------------------------------------- export


_WRAPPER = """

# ---- ar2 export adapter: the repository's menu retraining builds `Net()` and expects one value per sequence -----------------------------
import torch.nn as _ar2_nn

_AR2_HPARAMS = {hp}


class Net(_ar2_nn.Module):
    def __init__(self):
        super().__init__()
        self.model = build_model(1, **{{**HPARAMS, **_AR2_HPARAMS}})

    def forward(self, x):
        return self.model(x).reshape(x.shape[0], -1)[:, 0]
"""


def _rename_net(source: str) -> str:
    """A model file may itself define a class (or name) ``Net``; the adapter's ``Net`` would shadow it and break ``build_model``. Rename the original to ``Ar2Inner``
    (every ``Name`` and every class / function definition called ``Net``) before appending the adapter. Source without a ``Net`` is returned unchanged."""
    import ast

    tree = ast.parse(source)
    if not any(isinstance(n, ast.Name) and n.id == "Net" for n in ast.walk(tree)) and not any(
        isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == "Net" for n in ast.walk(tree)
    ):
        return source

    class R(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == "Net":
                node.id = "Ar2Inner"
            return node

        def visit_ClassDef(self, node):
            self.generic_visit(node)
            if node.name == "Net":
                node.name = "Ar2Inner"
            return node

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            if node.name == "Net":
                node.name = "Ar2Inner"
            return node

    return ast.unparse(ast.fix_missing_locations(R().visit(tree))) + "\n"


def to_train_config(recipe: dict) -> dict:
    return {
        "optimizer": "muon" if recipe.get("optimizer") == "muon" else "adamw",
        "lr": recipe.get("lr", 0.002),
        "weight_decay": recipe.get("weight_decay", 0.01),
        "batch_size": recipe.get("batch_size", 512),
        "epochs": recipe.get("max_epochs", 40),
        "lr_schedule": "onecycle" if recipe.get("schedule", "onecycle") == "onecycle" else "cosine",
    }


def export_menu(
    ws: str | Path, out_dir: str | Path, *, extra_families: list[str] | None = None
) -> Path:
    """Write each menu family's model.py, recipe.json, card.json, tags.json under ``out_dir/<family>/`` and a combined ``menu_ar2.json``."""
    ws, out_dir = Path(ws), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db = DB(ws / "ar2.db")
    cfg = cfgmod.load_frozen(ws)
    archive.rebuild(db, cfg)
    entries = []
    fams = menu(db, cfg)
    if extra_families:
        have = {r["family"] for r in fams}
        fams += [
            r
            for r in db.families()
            if r["family"] in extra_families and r["family"] not in have and r["best_code_hash"]
        ]
    runs = db.runs_for()
    for r in fams:
        ch = r["best_code_hash"]
        cand = db.candidate(ch)
        rec = (
            next(
                (
                    x["recipe"]
                    for x in runs
                    if x["code_hash"] == ch and x["recipe_hash"] == r["best_recipe_hash"]
                ),
                None,
            )
            or {}
        )
        fd = out_dir / r["family"]
        fd.mkdir(parents=True, exist_ok=True)
        (fd / "model.py").write_text(cand["source"])
        (fd / "recipe.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
        (fd / "card.json").write_text(json.dumps(cand.get("card") or {}, indent=1, sort_keys=True))
        (fd / "tags.json").write_text(json.dumps(cand.get("tags") or {}, indent=1, sort_keys=True))
        hp = (
            (cand.get("hparams") or {}).get("hparams", {})
            if isinstance(cand.get("hparams"), dict)
            else {}
        )
        hp = {**hp, **(rec.get("hparams") or {})}
        card = cand.get("card") if isinstance(cand.get("card"), dict) else {}
        entries.append(
            {
                "id": f"ar2_{r['family']}_{ch[:8]}",
                "cell": ws.name,
                "idea": card.get("summary", r["family"]),
                "orig_val": r["confirmed_mean"],
                "n_params": cand.get("n_params"),
                "source": _rename_net(cand["source"]) + _WRAPPER.format(hp=repr(hp)),
                "train_config": to_train_config(rec),
                "family": r["family"],
                "ar2_recipe": rec,
                "ar2_code_hash": ch,
                "ar2_confirmed_mean": r["confirmed_mean"],
                "ar2_flagged": bool(r.get("flagged")),
            }
        )
    p = out_dir / "menu_ar2.json"
    p.write_text(json.dumps(entries, indent=1))
    db.close()
    return p
