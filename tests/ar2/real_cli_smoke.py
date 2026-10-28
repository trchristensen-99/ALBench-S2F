"""Standalone smoke test of the real Claude Code CLI proposer (run on the cluster: it needs the CLI, its credentials and, for the model check, torch).

    python tests/ar2/real_cli_smoke.py --model <full model id> [--tagger-model <id>] [--scratch-root /tmp/ar2_smoke] [--config ws/config.yaml]
                                       [--allowed-tools "Read(./**),Write(./**),Edit(./**)"]

It performs one real SKETCH, one EXPLORE, one REPAIR (a planted wrong-output-shape model) and one TAGGER call with the config's model and prints, for each,
whether the output is valid (JSON schema, model.py present, selfcheck.py passes when torch is installed). Exit code 0 only if all four are valid.
If calls fail, the raw stderr tail is printed; the first thing to try is ``--allowed-tools "Read,Write,Edit"`` (unscoped tool rules) when the scoped
rules are not accepted by the installed CLI version.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ar2 import tagger, views  # noqa: E402
from ar2.proposer import ProposerHalt, ProposerPaused, make_proposer  # noqa: E402
from ar2.proposer import mock_library as lib  # noqa: E402
from ar2.proposer.base import validate_card, validate_sketches  # noqa: E402


def selfcheck(workdir: Path) -> str:
    try:
        import torch  # noqa: F401
    except ImportError:
        return "skipped (torch not installed here)"
    r = subprocess.run(
        [sys.executable, "selfcheck.py"], cwd=workdir, capture_output=True, text=True, timeout=300
    )
    return "passed" if r.returncode == 0 else "FAILED: " + (r.stdout + r.stderr).strip()[-300:]


def report(name: str, ok: bool, res, extra: str = "") -> bool:
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name}: model={res.model_id} duration={res.duration_s:.0f}s tokens={res.usage.get('input_tokens')}/{res.usage.get('output_tokens')} "
        f"changed={res.proposer_changed} {extra}"
    )
    if not ok:
        print("   error:", res.error)
        print("   missing:", res.missing)
        print("   stderr tail:", (res.raw_stderr or "")[-400:].replace("\n", " | "))
        print("   stdout tail:", (res.raw_stdout or "")[-400:].replace("\n", " | "))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--tagger-model")
    ap.add_argument("--scratch-root", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--allowed-tools",
        default=None,
        help="comma separated tool rules overriding proposer.allowed_tools",
    )
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    cfg: dict = {}
    if a.config:
        import yaml

        cfg = dict(yaml.safe_load(Path(a.config).read_text()).get("proposer", {}))
    cfg["backend"] = "claude_code_cli"
    if a.model:
        cfg["model"] = a.model
    if a.tagger_model:
        cfg["tagger_model"] = a.tagger_model
    cfg["scratch_root"] = (
        a.scratch_root or cfg.get("scratch_root") or tempfile.mkdtemp(prefix="ar2_smoke_")
    )
    if a.allowed_tools:
        cfg["allowed_tools"] = [t for t in a.allowed_tools.split(",") if t]
    prop = make_proposer({"proposer": cfg})
    print("CLI:", prop.cli_version, "options:", prop.options)
    scratch = Path(cfg["scratch_root"])
    results = []
    ctx = views.ViewContext()
    try:
        # SKETCH
        wd = scratch / "smoke_sketch"
        views.prepare_workdir(
            wd, "SKETCH", parent_code=None, reference_code=views.reference_model_code(0)
        )
        res = prop.call(views.render("SKETCH", ctx, {}), wd, "sketches", a.timeout)
        errs = validate_sketches(res.parsed) if res.parsed is not None else ["no sketches"]
        fams = sorted({s.get("family") for s in res.parsed}) if res.ok else []
        results.append(report("SKETCH", res.ok and not errs, res, f"families={fams} errors={errs}"))
        sketch = (
            (res.parsed or [{}])[0]
            if res.ok
            else {
                "family": "unet",
                "name": "Encoder-decoder with skips",
                "mechanism": "skips",
                "why_for_mpra": "multi-resolution",
            }
        )
        # EXPLORE
        wd = scratch / "smoke_explore"
        views.prepare_workdir(
            wd, "EXPLORE", parent_code=None, reference_code=views.reference_model_code(1)
        )
        res = prop.call(
            views.render("EXPLORE", ctx, {"sketch": sketch}), wd, "candidate", a.timeout
        )
        ok = (
            res.ok
            and "def build_model" in res.files.get("model.py", "")
            and not validate_card(res.parsed or {})
        )
        sc = selfcheck(wd) if "model.py" in res.files else "no model.py"
        results.append(
            report(
                "EXPLORE",
                ok and not sc.startswith("FAILED"),
                res,
                f"family_claimed={(res.parsed or {}).get('family_claimed')} selfcheck={sc}",
            )
        )
        code = res.files.get("model.py") or lib.library_source("cnn_plain")
        # REPAIR (planted wrong output shape)
        wd = scratch / "smoke_repair"
        bad = lib.broken_source("wrong_shape", lib.library_source("cnn_plain"))
        views.prepare_workdir(
            wd,
            "REPAIR",
            parent_code=bad,
            reference_code=views.reference_model_code(2),
            parent_card={
                "family_claimed": "cnn_residual",
                "summary": "planted wrong-output-shape model",
                "hypothesis": "none",
                "what_changed": "",
                "parent_code_hash": "",
                "suggested_recipe": {},
                "recipe_rationale": "",
                "expected_failure": "wrong shape",
                "amp_safe": True,
                "length_agnostic": False,
            },
        )
        prompt = views.render(
            "REPAIR",
            ctx,
            {
                "gate_code": "shape",
                "gate_description": "Output must be (B, out_dim) and finite",
                "measured": {"batch": 2, "out_dim": 1, "output_shape": [2, 2]},
                "traceback": "AssertionError: output shape (2, 2) != (2, 1)",
            },
        )
        res = prop.call(prompt, wd, "candidate", a.timeout)
        sc = selfcheck(wd) if "model.py" in res.files else "no model.py"
        results.append(
            report(
                "REPAIR",
                res.ok and sc == "passed" or (res.ok and sc.startswith("skipped")),
                res,
                f"selfcheck={sc}",
            )
        )
        # TAGGER
        tags = tagger.tag(prop, code, tagger.TAXONOMY, {"proposer": cfg}, scratch / "smoke_tag")
        ok = tags["source"] in ("llm", "llm+rules_override")
        print(
            f"[{'PASS' if ok else 'FAIL'}] TAGGER: family={tags['family']} source={tags['source']} rule_family={tags['rule_family']} justification={tags['justification'][:80]!r}"
        )
        results.append(ok)
    except ProposerPaused as ex:
        print("PAUSED (rate or auth):", ex, "retry_after_s =", ex.retry_after_s)
        return 2
    except ProposerHalt as ex:
        print("HALT:", ex)
        return 3
    print(json.dumps({"all_valid": all(results)}))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
