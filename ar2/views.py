"""Prompt views and rendering (spec sections 11 and 12.1).

The orchestrator turns memory into a :class:`ViewContext` (plain data) and this module turns that into a prompt. Nothing here reads the database, so nothing here
can reach the held-back selection set or the final-results table, and a field that is not on the whitelists below cannot appear in a prompt however it got into the context
(single-seed screening scores, legacy scores and held-out results are filtered by construction, then scanned for once more before the text is returned).

Rules enforced in :func:`render`: reference material first and the assignment last; SKETCH and EXPLORE prompts carry no scores; REFINE prompts carry confirmed
scores only (a family view with fewer than 3 seeds is refused); at most 2 full code files per prompt (they live in the working directory, not the prompt);
at most ``ctx.max_prompt_tokens`` tokens (chars / 4), shrinking the lowest-priority views first and never the core, the assignment or the output contract;
no ``{{`` left over.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ar2.prompts import load_template, template_hash
from ar2.proposer.guard import scan as _scan
from ar2.tagger import TAXONOMY, taxonomy_text

MODES = ("SKETCH", "EXPLORE", "REFINE", "REPAIR", "TAGGER", "LESSON")
DIAG_KEYS = (
    "train_r",
    "val_r",
    "gap",
    "best_epoch",
    "max_epochs",
    "stop_reason",
    "pred_sd_ratio",
    "mse_by_activity_quintile",
    "mse_by_gc_tercile",
)
LESSON_RECORD_KEYS = (
    "kind",
    "scope",
    "subject",
    "change",
    "direction",
    "outcome",
    "accepted",
    "family",
)


@dataclass
class ViewContext:
    """Everything a prompt may show. B1 builds it from the database; every field has a default so a minimal context is valid."""

    D: int = 30000
    n_recipes: int = 8
    max_params: int = 40_000_000
    time_cap_min: float = 30.0
    noise_floor: float = 0.005
    taxonomy: Any = ""  # text, or a list of family names; empty -> the spec 4.4 table
    coverage: list[dict] = field(
        default_factory=list
    )  # family, candidates, state, note (no scores)
    near: list[dict] = field(
        default_factory=list
    )  # summary, family (do-not-repeat; B1 caps at 12 or 20)
    lessons_global: list[str] = field(default_factory=list)
    lessons_family: list[str] = field(default_factory=list)
    family: dict | None = None  # REFINE only
    failure_view: list[dict] | None = None  # code, description
    extra_constraints: list[str] = field(default_factory=list)
    forced_families: list[str] | None = None
    speed_note: str | None = None
    max_prompt_tokens: int = 12000


def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1


def prompt_version(cfg: dict | None = None) -> str:
    """``<label>+<hash of every template file>``; the label comes from ``proposer.prompt_version`` (default v1)."""
    label = "v1"
    if cfg:
        label = str((cfg.get("proposer", cfg) or {}).get("prompt_version") or "v1")
    return f"{label}+{template_hash()}"


# ----------------------------------------------------------------------------------------------- helpers


def _clean(s: Any) -> str:
    """Text inserted into a template: never allow it to contain a placeholder opener."""
    return str(s).replace("{{", "{ {").replace("}}", "} }")


def _f(x: Any, nd: int = 3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def _num_params(n: Any) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "n/a"
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def _taxonomy_text(ctx: ViewContext) -> str:
    t = ctx.taxonomy
    if not t:
        return taxonomy_text()
    if isinstance(t, (list, tuple)):
        return "FAMILY TAXONOMY\n" + "\n".join(f"- {x}" for x in t)
    return str(t)


def _bullets(items: list[str] | None) -> str:
    return "\n".join(f"- {_clean(x)}" for x in (items or []) if str(x).strip())


# ----------------------------------------------------------------------------------------------- views


def coverage_view(ctx: ViewContext, level: int = 0) -> str:
    """Families tried so far, without scores. Empty taxonomy families are listed so that the proposer can see what is untouched."""
    rows: dict[str, tuple[int, str, str]] = {}
    for r in ctx.coverage:
        fam = str(r.get("family", "")).strip()
        if fam:
            n = int(r.get("candidates", 0) or 0)
            state = str(r.get("state") or ("empty" if n == 0 else "tried"))
            rows[fam] = (n, state, str(r.get("note") or ""))
    for fam in TAXONOMY:
        rows.setdefault(fam, (0, "empty", ""))
    lines = ["COVERAGE (families tried so far; scores are withheld during exploration)"]
    if level >= 5:
        by_state: dict[str, int] = {}
        for v in rows.values():
            by_state[v[1]] = by_state.get(v[1], 0) + 1
        top = sorted(rows.items(), key=lambda kv: -kv[1][0])[:12]
        lines.append(
            f"{len(rows)} families: " + ", ".join(f"{n} {st}" for st, n in sorted(by_state.items()))
        )
        lines.append("largest: " + "; ".join(f"{k}:{v[0]}" for k, v in top))
        lines.append("empty: " + ", ".join(k for k, v in rows.items() if v[0] == 0)[:300])
        return _clean("\n".join(lines))
    if level >= 4:
        lines.append("; ".join(f"{k}:{v[0]}:{v[1]}" for k, v in rows.items()))
        return _clean("\n".join(lines))
    if level >= 3:
        shown = {k: v for k, v in rows.items() if v[0] > 0}
        empties = [k for k, v in rows.items() if v[0] == 0]
        lines.append("family               candidates  state")
        lines += [f"{k:<20} {v[0]:<11} {v[1]}" for k, v in shown.items()]
        if empties:
            lines.append("empty: " + ", ".join(empties))
        return _clean("\n".join(lines))
    lines.append("family               candidates  state      note")
    note_cap = {0: 120, 1: 40, 2: 0}.get(level, 0)
    for k, (n, state, note) in rows.items():
        note = note[:note_cap] if note_cap else ""
        lines.append(f"{k:<20} {n:<11} {state:<10} {note}".rstrip())
    return _clean("\n".join(lines))


def do_not_repeat_view(ctx: ViewContext, level: int = 0) -> str:
    if not ctx.near:
        return ""
    cap = {0: 20, 1: 8, 2: 4}.get(level, 0)
    shown = ctx.near[:cap]
    lines = [
        "DO-NOT-REPEAT (existing candidates nearest to this assignment; do not propose these again)"
    ]
    lines += [f"- [{r.get('family', '?')}] {str(r.get('summary', ''))[:200]}" for r in shown]
    if len(ctx.near) > len(shown):
        counts: dict[str, int] = {}
        for r in ctx.near[len(shown) :]:
            counts[str(r.get("family", "?"))] = counts.get(str(r.get("family", "?")), 0) + 1
        lines.append("and more in: " + ", ".join(f"{k} ({v})" for k, v in sorted(counts.items())))
    return _clean("\n".join(lines))


def lessons_view(
    lessons: list[str],
    level: int = 0,
    title: str = "LESSONS (confirmed findings, each scoped; wording only, numbers measured)",
) -> str:
    cap = {0: 15, 1: 10, 2: 5, 3: 3}.get(level, 0)
    items = [str(x).strip() for x in lessons if str(x).strip()][:cap]
    if not items:
        return ""
    return _clean("\n".join([title] + [f"- {x}" for x in items]))


def failure_view(ctx: ViewContext, level: int = 0) -> str:
    if not ctx.failure_view:
        return ""
    rows = ctx.failure_view[: (2 if level >= 4 else 3)]
    lines = ["RECENT AUTOMATIC-CHECK FAILURES (most frequent first; avoid them)"]
    lines += [f"- {r.get('code', '?')}: {str(r.get('description', ''))[:160]}" for r in rows]
    return _clean("\n".join(lines))


def family_view(ctx: ViewContext, level: int = 0) -> str:
    """REFINE only: confirmed scores. A family with fewer than 3 confirmation seeds is refused, so a single-seed score cannot be presented as a finding."""
    fam = ctx.family
    if not fam:
        raise ValueError("REFINE needs ctx.family (a confirmed family)")
    n_seeds = int(fam.get("n_seeds", 0) or 0)
    if n_seeds < 3:
        raise ValueError(f"family view needs a confirmed family (3 seeds); got n_seeds={n_seeds}")
    lines = [
        f"FAMILY {fam.get('family', '?')}: confirmed {_f(fam.get('confirmed_mean'))} +/- {_f(fam.get('confirmed_se'))} ({n_seeds} seeds). "
        f"Top family: {_f(fam.get('top_mean'))} +/- {_f(fam.get('top_se'))}.",
        f"BEST MODEL: parent_model.py ({_num_params(fam.get('params'))} parameters, {_num_params(fam.get('throughput'))} sequences per second)",
    ]
    d = {k: v for k, v in (fam.get("diagnostics") or {}).items() if k in DIAG_KEYS}
    if d:
        gap = d.get("gap")
        if gap is None and d.get("train_r") is not None and d.get("val_r") is not None:
            gap = float(d["train_r"]) - float(d["val_r"])
        parts = [
            f"train r {_f(d.get('train_r'), 2)}, validation r {_f(d.get('val_r'), 2)} (gap {_f(gap, 2)})"
        ]
        if d.get("best_epoch") is not None:
            parts.append(
                f"best epoch {d['best_epoch']} of {d.get('max_epochs', '?')}, {'stopped early' if d.get('stop_reason') == 'early_stop' else d.get('stop_reason', 'ended')}"
            )
        if d.get("pred_sd_ratio") is not None:
            parts.append(f"prediction spread {_f(d['pred_sd_ratio'], 2)} of label spread")
        if d.get("mse_by_activity_quintile"):
            parts.append(
                "MSE by activity quintile "
                + " ".join(_f(x, 2) for x in d["mse_by_activity_quintile"])
            )
        if d.get("mse_by_gc_tercile"):
            parts.append("MSE by GC tercile " + " ".join(_f(x, 2) for x in d["mse_by_gc_tercile"]))
        lines.append("DIAGNOSTICS: " + "; ".join(parts))
    rs = fam.get("recipe_sensitivity") or {}
    if rs and level < 3:
        lines.append(
            f"RECIPE SENSITIVITY (single-seed screening runs, noisy): {rs.get('n', '?')} recipes, validation r {_f(rs.get('lo'))} to {_f(rs.get('hi'))}"
            + (f"; best lr {rs['best_lr']}" if rs.get("best_lr") is not None else "")
            + (
                f", weight decay {rs['best_weight_decay']}"
                if rs.get("best_weight_decay") is not None
                else ""
            )
        )
    changes = list(fam.get("earlier_changes") or [])
    cap = {0: 20, 1: 6, 2: 3}.get(level, 0)
    if changes:
        lines.append("EARLIER CHANGES TO THIS FAMILY")
        for ch in changes[:cap]:
            seeds_ok = int(ch.get("n_seeds", 3) or 0) >= 3  # an unconfirmed change shows no number
            eff = (
                f"{float(ch['effect']):+.3f} +/- {_f(ch.get('se'))}, "
                if seeds_ok and ch.get("effect") is not None
                else ""
            )
            lines.append(
                f"- {str(ch.get('text', ''))[:200]}: {eff}{'accepted' if ch.get('accepted') else 'rejected'}"
            )
        if len(changes) > cap:
            lines.append(f"- ({len(changes) - cap} older changes not shown; do not repeat them)")
    return _clean("\n".join(lines))


# ----------------------------------------------------------------------------------------------- rendering


def _fill(template: str, values: dict[str, str]) -> str:
    """Substitute {{name}}; a line that is only a placeholder with an empty value disappears."""
    out: list[str] = []
    for line in template.split("\n"):
        stripped = line.strip()
        m = re.fullmatch(r"\{\{(\w+)\}\}", stripped)
        if m and m.group(1) in values and values[m.group(1)] == "":
            continue
        for k, v in values.items():
            if "{{" + k + "}}" in line:
                line = line.replace("{{" + k + "}}", v)
        out.append(line)
    return "\n".join(out)


def _render_core(ctx: ViewContext) -> str:
    return _fill(
        load_template("core.md"),
        {
            "D": f"{int(ctx.D):,}",
            "n_recipes": str(int(ctx.n_recipes)),
            "max_params": f"{int(ctx.max_params):,}",
            "time_cap_min": f"{float(ctx.time_cap_min):.0f}",
            "noise_floor": _f(ctx.noise_floor, 3),
        },
    )


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return text[:keep] + "\n... [truncated] ...\n" + text[-keep:]


def _build(mode: str, ctx: ViewContext, a: dict, level: int) -> str:
    core = _render_core(ctx)
    forced = ""
    if ctx.forced_families:
        forced = (
            "- Controller constraint for this round: the sketches are accepted only in these families: "
            + ", ".join(_clean(x) for x in ctx.forced_families)
            + "."
        )
    base = {
        "core": core,
        "D": f"{int(ctx.D):,}",
        "taxonomy": _taxonomy_text(ctx),
        "coverage_view": coverage_view(ctx, level),
        "do_not_repeat_view": do_not_repeat_view(ctx, level),
        "failure_view": failure_view(ctx, level),
        "forced_families": forced,
        "extra_constraints": _bullets(ctx.extra_constraints),
        "speed_note": f"- {_clean(ctx.speed_note)}" if ctx.speed_note else "",
        "lessons_view_global": lessons_view(ctx.lessons_global, level),
    }
    if mode == "SKETCH":
        return _fill(load_template("sketch.md"), {**base, "lessons_view_global": ""})
    if mode == "EXPLORE":
        sketch = a.get("sketch")
        if not isinstance(sketch, dict):
            raise ValueError("EXPLORE needs assignment['sketch'] (a dict)")
        return _fill(
            load_template("explore.md"),
            {**base, "sketch_json": _clean(json.dumps(sketch, indent=2))},
        )
    if mode == "REFINE":
        fam_lessons = [x for x in ctx.lessons_family] + [
            x for x in ctx.lessons_global if x not in ctx.lessons_family
        ]
        return _fill(
            load_template("refine.md"),
            {
                **base,
                "family_view": family_view(ctx, level),
                "lessons_view_family": lessons_view(fam_lessons, level),
                "angle": _clean(a.get("angle", "capacity (width and depth)")),
            },
        )
    if mode == "REPAIR":
        measured = a.get("measured", "")
        measured = (
            json.dumps(measured, sort_keys=True)
            if isinstance(measured, (dict, list))
            else str(measured)
        )
        tb = str(a.get("traceback", ""))
        tb = tb[-(2000 if level >= 1 else 4000) :]
        return _fill(
            load_template("repair.md"),
            {
                **base,
                "gate_code": _clean(a.get("gate_code", "")),
                "gate_description": _clean(a.get("gate_description", "")),
                "measured": _clean(measured),
                "traceback": _clean(tb),
            },
        )
    if mode == "TAGGER":
        tmpl = load_template("tagger.md")
        budget = max(2000, ctx.max_prompt_tokens * 4 - len(tmpl) - 2000)
        return _fill(
            tmpl,
            {
                "taxonomy": _taxonomy_text(ctx),
                "code": _clean(_truncate_middle(str(a.get("code", "")), budget)),
            },
        )
    if mode == "LESSON":
        ev = {k: v for k, v in (a.get("evidence") or {}).items() if k in LESSON_RECORD_KEYS}
        return _fill(
            load_template("lessons.md"), {"evidence_json": _clean(json.dumps(ev, sort_keys=True))}
        )
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")


def render_with_meta(
    mode: str, ctx: ViewContext, assignment: dict | None = None
) -> tuple[str, dict]:
    mode = mode.upper()
    a = assignment or {}
    last = ""
    for level in range(0, 6):
        text = _build(mode, ctx, a, level)
        if "{{" in text:
            raise ValueError(
                "unrendered placeholder left in the prompt: "
                + text[text.index("{{") : text.index("{{") + 40]
            )
        last = text
        if estimate_tokens(text) <= ctx.max_prompt_tokens:
            _scan(text)
            return text, {
                "mode": mode,
                "tokens": estimate_tokens(text),
                "truncation_level": level,
                "prompt_version": prompt_version(),
            }
    raise ValueError(
        f"{mode} prompt is {estimate_tokens(last)} tokens, over the limit of {ctx.max_prompt_tokens} even after truncation"
    )


def render(mode: str, ctx: ViewContext, assignment: dict | None = None) -> str:
    return render_with_meta(mode, ctx, assignment)[0]


# ----------------------------------------------------------------------------------------------- working directory

SELFCHECK_PY = '''"""Smoke test of model.py: builds it with out_dim 1 and 2, runs one forward pass on a random batch and prints shapes and parameter counts.

    python selfcheck.py
The real checks before training are broader (batch sizes, coverage of the input, speed); this only catches interface mistakes early.
"""
import importlib.util
import sys
import traceback


def main() -> int:
    try:
        import torch
    except ImportError:
        print("torch is not installed in this environment, so the model cannot be checked here")
        return 1
    try:
        spec = importlib.util.spec_from_file_location("candidate_model", "model.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for need in ("HPARAMS", "build_model"):
            if not hasattr(mod, need):
                print(f"model.py must define {need}")
                return 1
        for out_dim in (1, 2):
            torch.manual_seed(0)
            net = mod.build_model(out_dim, **dict(mod.HPARAMS))
            idx = torch.randint(0, 4, (2, 200))
            x = torch.nn.functional.one_hot(idx, 4).permute(0, 2, 1).float()
            net.eval()
            with torch.no_grad():
                y = net(x)
            n = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print(f"out_dim={out_dim}: output {tuple(y.shape)}, trainable parameters {n:,}")
            if tuple(y.shape) != (2, out_dim):
                print(f"ERROR: expected output shape (2, {out_dim})")
                return 1
            if not torch.isfinite(y).all():
                print("ERROR: non-finite output")
                return 1
        print("selfcheck passed")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
'''

REFERENCE_NAMES = ("cnn_residual", "rnn_hybrid", "transformer_hybrid")


def reference_model_code(counter: int = 0) -> str:
    """Source of one of the three reference models, rotated by ``counter`` so that the same example is not always shown."""
    name = REFERENCE_NAMES[counter % len(REFERENCE_NAMES)]
    return (
        Path(__file__).resolve().parent / "proposer" / "reference_models" / f"{name}.py"
    ).read_text()


def prepare_workdir(
    workdir: Path,
    mode: str,
    *,
    parent_code: str | None,
    reference_code: str,
    parent_card: dict | None = None,
) -> None:
    """Fill a call's scratch directory: CONTRACT.md, selfcheck.py, reference_model.py and, for REFINE and REPAIR, parent_model.py (at most 2 code files)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    mode = mode.upper()
    (workdir / "CONTRACT.md").write_text(load_template("model_contract.md"))
    (workdir / "selfcheck.py").write_text(SELFCHECK_PY)
    (workdir / "reference_model.py").write_text(reference_code)
    if mode in ("REFINE", "REPAIR"):
        if parent_code is None:
            raise ValueError(f"{mode} needs parent_code")
        (workdir / "parent_model.py").write_text(parent_code)
        if (
            parent_card is not None
        ):  # the repair / refine call edits this card instead of inventing one (a real CLI refused to write card.json from nothing)
            (workdir / "card.json").write_text(json.dumps(parent_card, indent=1))
    elif (workdir / "parent_model.py").exists():
        (workdir / "parent_model.py").unlink()
