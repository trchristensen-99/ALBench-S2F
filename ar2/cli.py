"""``python -m ar2 <command>``: init | plan | simulate | step | run | status | set | report | export-menu | rebuild | import-legacy | worker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ar2 import config as cfgmod
from ar2.memory import archive
from ar2.memory.db import DB


def _overrides(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = cfgmod.parse_value(v)
    return out


def cmd_init(a) -> int:
    from ar2.orchestrator import Orchestrator

    cfg = cfgmod.load_config(a.config, _overrides(a.set), a.profile)
    if (
        cfg["proposer"]["backend"] == "claude_code_cli"
        and not cfg["proposer"].get("model")
        and not a.no_proposer_check
    ):
        raise SystemExit(
            "proposer.model is required (a full model id, not an alias): --set proposer.model=<id>"
        )
    for k in ("train_manifest", "val_loop_manifest"):
        if not cfg["data"].get(k) and not a.allow_missing_data:
            raise SystemExit(f"data.{k} is required (--set data.{k}=path.json)")
    o = Orchestrator.init_workspace(
        a.ws, cfg, check_proposer=not a.no_proposer_check, make_stress=not a.allow_missing_data
    )
    print(
        f"initialised {a.ws}: config hash {o.db.meta_get('config_hash')}, prompt {o.db.meta_get('prompt_version')}"
    )
    o.close()
    return 0


def cmd_plan(a) -> int:
    from ar2 import controller

    cfg = (
        cfgmod.load_frozen(Path(a.ws))
        if a.ws
        else cfgmod.load_config(a.config, _overrides(a.set), a.profile)
    )
    print(controller.plan(cfg))
    return 0


def cmd_simulate(a) -> int:
    """The controller under planted landscapes (no GPU, no LLM). ``--sweep balance.explore_fraction=0.2,0.4,0.6,0.8`` as in spec 5.7."""
    from ar2 import sim

    cfg = cfgmod.load_config(a.config, _overrides(a.set), a.profile)
    kinds = (
        {"broad": "broad", "deep": "never_new", "flat": "flat"}
        if a.kind == "all"
        else {a.kind: a.kind}
    )
    key, values = "balance.explore_fraction", [None]
    if a.sweep:
        key, vals = a.sweep.split("=", 1)
        values = [cfgmod.parse_value(v) for v in vals.split(",")]
    print(
        f"{'landscape':<10} {key if a.sweep else '(config)':<28} {'best confirmed':>14} {'menu families':>14} {'explore share':>14} {'budget ok':>10}"
    )
    for label, kind in kinds.items():
        for v in values:
            c = cfgmod.deep_merge(cfg, {})
            if a.sweep:
                cfgmod.set_dotted(c, key, v)
                if key.startswith("balance.explore_fraction"):
                    c["balance"]["mode"] = "fixed"
            res = sim.sweep(c, kind, range(a.seeds))
            best = [
                max(
                    (f.confirmed_mean for f in r.families.values() if f.confirmed_mean is not None),
                    default=float("nan"),
                )
                for r in res
            ]
            menu = [len(r.menu["menu"]) for r in res]
            print(
                f"{label:<10} {str(v) if a.sweep else '':<28} {sum(best) / len(best):>14.4f} {sum(menu) / len(menu):>14.1f} {sum(sim.explore_share(r) for r in res) / len(res):>14.2f} {sum(sim.budget_ok(r, c) for r in res):>5}/{len(res)}"
            )
    return 0


def cmd_step(a) -> int:
    from ar2.orchestrator import Orchestrator

    o = Orchestrator(a.ws, allow_config_change=a.allow_config_change, new_arm=a.new_arm)
    rep = o.step()
    print(json.dumps(rep.__dict__, default=str))
    o.close()
    return 0


def cmd_run(a) -> int:
    from ar2.orchestrator import Orchestrator

    o = Orchestrator(a.ws, allow_config_change=a.allow_config_change, new_arm=a.new_arm)
    try:
        rep = o.run(max_steps=a.max_steps, log=lambda m: print(m, flush=True))
    finally:
        o.close()
    print(json.dumps(rep.__dict__, default=str))
    return 0


def cmd_status(a) -> int:
    ws = Path(a.ws)
    db = DB(ws / "ar2.db")
    cfg, _ = cfgmod.effective_config(ws)
    archive.rebuild(db, cfg)
    bud = cfg["controller"]["budget"]
    print(
        f"workspace {ws}  step {db.meta_get('step')}  D {cfg['data']['D']}  prompt {db.meta_get('prompt_version')}  DONE={'yes' if (ws / 'DONE').exists() else 'no'}"
    )
    print(
        "spend by phase (run-equivalents):",
        {
            ph: f"{db.spend(phase=ph):.1f}/{bud.get(k)}"
            for ph, k in archive.PHASE_BUDGET_KEY.items()
        },
    )
    cat = {c: db.spend(phase="M", category=c) for c in ("explore", "refine", "deep_tune")}
    tot = sum(cat.values()) or 1.0
    print(
        "main search by category:",
        {k: f"{v:.1f} ({v / tot:.0%})" for k, v in cat.items()},
        "balance:",
        {
            k: cfg["balance"][k]
            for k in ("mode", "explore_fraction", "explore_start", "explore_end", "deep_tune_share")
        },
    )
    print("families:")
    for r in db.families():
        print(
            f"  {r['family']:<20} {r['state']:<11} cand {r['n_candidates']:<3} runs {r['n_runs']:<3} screen {r['screen_best'] if r['screen_best'] is None else round(r['screen_best'], 4)} confirmed {r['confirmed_mean'] if r['confirmed_mean'] is None else round(r['confirmed_mean'], 4)} (n={r['n_confirm']})"
        )
    print(
        "last outcomes: new families",
        archive._new_family_outcomes(db)[-4:],
        "| exploit",
        [("ok" if x["accepted"] else "rej") for x in archive._exploit_outcomes(db)[-4:]],
    )
    guards = [
        d
        for d in db.decisions("NOTE")
        if isinstance(d.get("payload"), dict) and d["payload"].get("kind") == "guard"
    ]
    print("guards fired:", [d["reason"] for d in guards[-4:]] or "none")
    print(
        "live balance changes:",
        [f"{r['key']}={r['value']}" for r in cfgmod.read_balance_overrides(ws)] or "none",
    )
    print(
        "runs:",
        {
            s: len(db.runs_for(status=s))
            for s in (
                "submitted",
                "running",
                "ok",
                "diverged",
                "exploded",
                "oom",
                "time_cap",
                "error",
                "early_stopped_by_tuner",
            )
        },
    )
    q = ws / "queue"
    if q.exists():
        print(
            "queue:", {s: len(list((q / s).glob("*.json"))) for s in ("pending", "running", "done")}
        )
    db.close()
    return 0


def cmd_set(a) -> int:
    try:
        rec = cfgmod.set_live(Path(a.ws), a.key, cfgmod.parse_value(a.value))
    except cfgmod.ConfigError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(
        f"recorded {rec['key']}: {rec['old']} -> {rec['value']} (applies at the next step; logged to decisions then)"
    )
    return 0


def cmd_report(a) -> int:
    from ar2 import report

    print(report.write_report(a.ws, a.out))
    return 0


def cmd_export(a) -> int:
    from ar2 import report

    print(report.export_menu(a.ws, a.out, extra_families=a.extra))
    return 0


def cmd_rebuild(a) -> int:
    ws = Path(a.ws)
    db = DB(ws / "ar2.db")
    cfg, _ = cfgmod.effective_config(ws)
    archive.rebuild(db, cfg)
    print(f"rebuilt: {len(db.families())} families, {len(db.lessons(status=None))} lessons")
    db.close()
    return 0


def cmd_import_legacy(a) -> int:
    print(
        "import-legacy is a documented stub in this version: tag existing model code with the tagger and insert it as candidates with source=legacy "
        "(coverage and do-not-repeat only; legacy scores are never shown). Not implemented.",
        file=sys.stderr,
    )
    return 2


def cmd_worker(a) -> int:
    from ar2.jobs import worker

    return worker.main(["--queue", str(Path(a.ws) / "queue"), "--idle-exit-s", str(a.idle_exit_s)])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ar2", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def ws_arg(sp, required=True):
        sp.add_argument("--ws", required=required, help="workspace directory")

    sp = sub.add_parser("init", help="create a workspace, freeze config and templates")
    ws_arg(sp)
    sp.add_argument("--config")
    sp.add_argument("--profile", choices=["deployment"])
    sp.add_argument("--set", action="append", metavar="KEY=VALUE")
    sp.add_argument("--no-proposer-check", action="store_true")
    sp.add_argument("--allow-missing-data", action="store_true")
    sp.set_defaults(fn=cmd_init)
    sp = sub.add_parser(
        "plan", help="forecast of new families / refinements / deep-tune recipes for the config"
    )
    ws_arg(sp, False)
    sp.add_argument("--config")
    sp.add_argument("--profile", choices=["deployment"])
    sp.add_argument("--set", action="append", metavar="KEY=VALUE")
    sp.set_defaults(fn=cmd_plan)
    sp = sub.add_parser(
        "simulate", help="run the controller on the planted landscapes (no GPU, no LLM)"
    )
    sp.add_argument(
        "--kind",
        default="all",
        choices=["all", "broad", "deep", "flat", "planted", "never_new", "never_refine"],
    )
    sp.add_argument("--seeds", type=int, default=5)
    sp.add_argument(
        "--sweep", metavar="KEY=V1,V2,...", help="e.g. balance.explore_fraction=0.2,0.4,0.6,0.8"
    )
    sp.add_argument("--config")
    sp.add_argument("--profile", choices=["deployment"])
    sp.add_argument("--set", action="append", metavar="KEY=VALUE")
    sp.set_defaults(fn=cmd_simulate)
    for name, fn in (("step", cmd_step), ("run", cmd_run)):
        sp = sub.add_parser(name)
        ws_arg(sp)
        sp.add_argument("--allow-config-change", action="store_true")
        sp.add_argument(
            "--new-arm",
            action="store_true",
            help="accept changed prompt templates as a new, separately reported arm",
        )
        if name == "run":
            sp.add_argument("--max-steps", type=int)
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("status")
    ws_arg(sp)
    sp.set_defaults(fn=cmd_status)
    sp = sub.add_parser("set", help="change a balance.* key during a run")
    ws_arg(sp)
    sp.add_argument("key")
    sp.add_argument("value")
    sp.set_defaults(fn=cmd_set)
    sp = sub.add_parser("report")
    ws_arg(sp)
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_report)
    sp = sub.add_parser("export-menu")
    ws_arg(sp)
    sp.add_argument("--out", required=True)
    sp.add_argument(
        "--extra",
        nargs="*",
        help="further families (by name) to export beside the menu-eligible ones",
    )
    sp.set_defaults(fn=cmd_export)
    sp = sub.add_parser("rebuild")
    ws_arg(sp)
    sp.set_defaults(fn=cmd_rebuild)
    sp = sub.add_parser("import-legacy")
    ws_arg(sp)
    sp.add_argument("dir")
    sp.set_defaults(fn=cmd_import_legacy)
    sp = sub.add_parser("worker", help="run a queue worker for this workspace")
    ws_arg(sp)
    sp.add_argument("--idle-exit-s", type=float, default=1800.0)
    sp.set_defaults(fn=cmd_worker)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
