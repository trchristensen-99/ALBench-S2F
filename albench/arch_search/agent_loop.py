"""Multi-agent proposal loop for the architecture AutoResearch (opt-in: ``--loop agent``).

Why this exists: the default loop makes ONE model call per round that has to invent two
architectures, write their code and choose their training recipe at once. This module splits
that call into roles so the question "does role separation plus a persistent lessons note beat
a single call?" can be answered with everything else held fixed:

    propose   -> ideas (IDEA / WHY / DESIGN / TRAIN recipe), no code
    code      -> one Net class per idea, checked by ``build_and_check``; a failure is shown back
                 to the coder for up to ``repairs`` repair attempts
    evaluate  -> the unchanged harness (``train_one``, validation-only selection, the same time
                 cap and exploded-prediction screens) plus a deterministic round report
    reflect   -> a short persistent LESSONS note, rewritten after each round and fed to the next
                 proposer call

What is held fixed against ``--loop single``: the trained-model target (rounds x per_round), the
per-model cap, the guidance block (KNOWN_GOOD / ensemble note, taken from the same ``_round_note``),
the training/screening code path, and the output layout, so ``load_cell`` and the analysis scripts
read the result unchanged. What differs is only who writes what, and the lessons note.

Nothing about the test sets reaches any agent: prompts are built from validation metrics only
(see ``ModelRecord`` and ``make_report``), and ``tests/test_agent_loop.py`` checks that the test
Pearson written to the meta files never appears in a prompt.

Stages are a list in the config (``--agent_cfg '{"stages": [...]}'``) looked up in ``STAGES``;
add one with ``@register_stage("name")``. Each stage is idempotent against the round's plan file
so a requeue resumes inside a round instead of re-paying for LLM calls already made.

Launch (same flags as the baseline arm, plus ``--loop agent``)::

    python scratchpad/arch_autoresearch.py --reservoir_cache POOL.npz --D 30000 --out_dir OUT \
        --rounds 10 --per_round 2 --max_params 40000000 --base_cap_min 25 --budget_hours 3 \
        --guidance known_good --ensemble_aware --loop agent --resume

``--agent_cfg`` takes inline JSON or a file path. Keys (all optional): ``stages`` (default
propose,code,evaluate,reflect; "reflect" may also come first), ``repairs`` (2), ``ideas_per_round``
(per_round), ``propose_retries`` (1), ``lessons_max_chars`` (1800), ``coder_workers`` (1),
``role_models`` ({"proposer"|"coder"|"reflector": model id}), ``reflector_noise_note`` (true).

Cost report for a finished run: ``python -m albench.arch_search.agent_loop summarize OUT [OUT ...]``.

Outputs beyond the single loop's layout: ``agent_logs/round_NN.jsonl`` (one line per LLM call:
role, prompt/response, chars, approximate tokens, seconds), ``agent_logs/plan_rNN.json`` (ideas,
coder repairs, per-idea status, the evaluator report), ``agent_logs/summary.json`` (calls and
tokens per role), ``lessons.md`` and ``agent_logs/lessons_history.jsonl``.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from albench.arch_search import propose as _propose
from albench.arch_search.propose import CONTRACT, SYSTEM, TASK, build_arch_prompt
from albench.arch_search.validate import build_and_check

DEFAULT_STAGES = ("propose", "code", "evaluate", "reflect")
# ``claude -p`` returns text only, so token counts are estimates (chars / 4). The CLI also bills a
# fixed prompt of its own on every call: measured 2026-10-23 with `claude -p ... --output-format
# json` on the cluster, a one-word prompt cost 31,149 input tokens (7,002 cache-creation +
# 24,145 cache-read). That constant dominates short calls, so cost is reported both ways.
CLI_OVERHEAD_TOKENS = int(os.environ.get("AGENT_CLI_OVERHEAD_TOKENS", "31150"))
# Plan-item states after which nothing more is done to the item in this round.
_RESOLVED = {"trained", "rejected", "failed", "skipped"}


class AgentConfigError(ValueError):
    pass


@dataclass
class AgentConfig:
    stages: tuple[str, ...] = DEFAULT_STAGES
    repairs: int = 2  # repair attempts per idea after the first coder call
    ideas_per_round: int | None = None  # None -> --per_round
    propose_retries: int = 1  # re-ask once when no idea could be parsed
    lessons_max_chars: int = 1800  # the lessons note is short by construction
    coder_workers: int = 1  # >1 codes ideas of a round in parallel (burstier on the usage limit)
    role_models: dict = field(default_factory=dict)  # {"proposer"|"coder"|"reflector": model id}
    reflector_noise_note: bool = True  # tell the reflector single-run val noise is ~0.008
    select_per_round: int | None = (
        None  # coordinator: how many ideas to keep per round; None -> --per_round
    )
    coordinator_mode: str = "llm"  # "llm": an agent chooses; "random" / "first": the k ideas are picked without an LLM call (control for the coordinator's judgment)
    proposer_style: str = "none"  # "explore" adds the bold/diverse paragraph of the single loop's --style explore to the proposer
    manager: str = "none"  # "schedule": the coordinator and proposer get an explore (early rounds) then exploit (later rounds) directive and a coverage table of architecture classes and recipes already trained;
    # "schedule_ens": additionally the coordinator sees the validation stack of everything trained so far (weight and correlation per model). Needs the coordinate stage.
    explore_frac: float = 0.4  # manager: the first round(explore_frac * rounds) rounds are explore rounds, the rest exploit
    # manager "quota": every round trains one EXPLOIT model and (select_per_round - 1) EXPLORE models; the coordinator sees a coverage table of architectural features (and which are still untried) and the validation stack.
    recipe_source: str = "proposal"  # "proposal": the proposer writes a TRAIN line per idea (the default of every earlier run); "harness": the proposer writes architectures only and the harness assigns each model a recipe from a safe set (AL_RECIPE_MODE=harness)
    cli_overhead_tokens: int = CLI_OVERHEAD_TOKENS

    @classmethod
    def from_arg(cls, s: str | None) -> AgentConfig:
        """Parse ``--agent_cfg``: '' (defaults), an inline JSON object, or a path to a JSON file.

        Unknown keys are an error, not ignored: a declared-and-never-read knob already cost this
        project one silent bug.
        """
        raw: dict = {}
        if s:
            s = s.strip()
            raw = json.loads(s) if s.startswith("{") else json.loads(Path(s).read_text())
        known = set(cls.__dataclass_fields__)
        extra = set(raw) - known
        if extra:
            raise AgentConfigError(
                f"unknown agent_cfg keys {sorted(extra)}; known: {sorted(known)}"
            )
        if isinstance(raw.get("stages"), str):
            raw["stages"] = [x.strip() for x in raw["stages"].split(",") if x.strip()]
        cfg = cls(**{**raw, **({"stages": tuple(raw["stages"])} if "stages" in raw else {})})
        cfg.validate()
        return cfg

    def validate(self) -> None:
        st = list(self.stages)
        bad = [x for x in st if x not in STAGES]
        if bad:
            raise AgentConfigError(f"unknown stages {bad}; registered: {sorted(STAGES)}")
        if len(set(st)) != len(st):
            raise AgentConfigError(f"duplicate stage in {st}")
        for need, before in (("code", "propose"), ("evaluate", "code"), ("coordinate", "propose")):
            if need in st and (before not in st or st.index(before) > st.index(need)):
                raise AgentConfigError(f"stage {need!r} needs {before!r} earlier in {st}")
        if "coordinate" in st and st.index("coordinate") > st.index("code"):
            raise AgentConfigError(f"stage 'coordinate' must come before 'code' in {st}")
        if self.proposer_style not in ("none", "explore"):
            raise AgentConfigError(
                f"proposer_style must be 'none' or 'explore', got {self.proposer_style!r}"
            )
        if self.manager not in ("none", "schedule", "schedule_ens", "quota"):
            raise AgentConfigError(
                f"manager must be 'none', 'schedule', 'schedule_ens' or 'quota', got {self.manager!r}"
            )
        if self.recipe_source not in ("proposal", "harness"):
            raise AgentConfigError(
                f"recipe_source must be 'proposal' or 'harness', got {self.recipe_source!r}"
            )
        if self.manager != "none" and "coordinate" not in st:
            raise AgentConfigError(
                "the manager directs the coordinator: add the 'coordinate' stage"
            )
        if not 0.0 <= self.explore_frac <= 1.0:
            raise AgentConfigError("explore_frac must be between 0 and 1")
        if self.coordinator_mode not in ("llm", "random", "first"):
            raise AgentConfigError(
                f"coordinator_mode must be 'llm', 'random' or 'first', got {self.coordinator_mode!r}"
            )
        if "evaluate" not in st:
            raise AgentConfigError(f"stages {st} never train a model; 'evaluate' is required")
        bad_roles = set(self.role_models) - {"proposer", "coder", "reflector", "coordinator"}
        if bad_roles:
            raise AgentConfigError(f"unknown role_models keys {sorted(bad_roles)}")
        if self.repairs < 0 or self.coder_workers < 1:
            raise AgentConfigError("repairs must be >= 0 and coder_workers >= 1")


# --------------------------------------------------------------------------- prompts

# The proposer keeps the single-call prompt's design-freedom paragraph verbatim so the A/B does
# not also change how adventurous the proposer is told to be.
_FREEDOM = SYSTEM.split("\n\n")[1]

PROPOSER_SYSTEM = (
    _FREEDOM
    + """

You are the PROPOSER in a team of agents. You decide WHAT to try next. A separate coding agent \
turns each idea into PyTorch from your DESIGN text, a fixed harness trains and scores it on a \
held-out validation set, and a reflector agent keeps a short LESSONS note that you will see in \
the next round. You do not write code and you never see test data.

Return exactly {n} proposals. For each, emit, in this order:

### IDEA: <one line naming the mechanism you are betting on>
### WHY: <one or two sentences -- what in the task or in the results so far makes this plausible>
### DESIGN: <enough detail that a programmer can implement it without guessing: stem, block type \
and count, channel widths, kernel sizes or dilations, pooling or downsampling, normalisation, \
head, dropout, rough parameter count. Prose or bullets, at most ~12 lines, no code>
### TRAIN: {"optimizer":"muon|adamw|adam","lr":2e-3,"weight_decay":1e-5,"batch_size":128,\
"epochs":80,"lr_schedule":"onecycle|plateau|cosine"}

Make the proposals differ from each other and from everything already tried. Use the lessons to \
avoid repeating what failed and to test what is still untested."""
)

PROPOSER_EXPLORE = (
    "EXPLORATION STYLE: be bold. Spend most of your proposals on structurally different ideas that the history does not yet contain "
    "(different block types, receptive-field layouts, pooling, attention placement, heads, widths and depths), not small variations of the current best. "
    "Each proposal should differ from the others and from earlier ones in more than one design choice, while still being a model you expect to train well."
)

COORDINATOR_SYSTEM = """\
You are the COORDINATOR in a team of agents that searches for neural-network architectures predicting \
MPRA activity from 200-bp DNA. A proposer has suggested {n} candidate ideas this round and only {k} can \
be trained. You choose which {k}. You never see test data; you see the validation results of everything \
trained so far, a lessons note and the candidates.

The final product is an ENSEMBLE of all trained models, so a good pick is (a) likely to reach a high \
validation Pearson r, (b) different from the models already trained and from your other picks (a \
different mechanism, receptive field, pooling or head, not a width tweak), and (c) unlikely to fail \
(shape errors, too slow for the time cap, unstable training, parameter limit). Drop duplicates of \
earlier failures. Exploit what the history shows works, but keep at least one bet that tests something \
untried when the history is thin.

Reply with exactly:
### SELECT: <{k} candidate numbers separated by commas>
### RATIONALE: <one short sentence per chosen candidate>"""


def parse_selection(text: str, n: int, k: int) -> list[int]:
    """0-based indices of the chosen candidates: 1-based numbers after '### SELECT:', duplicates and out-of-range numbers dropped, padded in proposal order."""
    m = re.search(r"###\s*SELECT:\s*([0-9][0-9,\s;and]*)", text or "")
    chosen: list[int] = []
    if m:
        for tok in re.findall(r"\d+", m.group(1)):
            i = int(tok) - 1
            if 0 <= i < n and i not in chosen:
                chosen.append(i)
    for i in range(n):  # fall back to proposal order for anything missing
        if len(chosen) >= k:
            break
        if i not in chosen:
            chosen.append(i)
    return chosen[:k]


PROPOSER_SETUP = (
    "SETUP: you choose WHAT to build and how to train it; a coding agent writes the Net class from "
    "your DESIGN, so be specific and stay within what can be written with the allowed imports. A "
    "fixed harness trains it: reverse-complement + random-shift augmentation are ALWAYS applied, "
    "optimizers available are muon (orthogonalized momentum, often best on conv stacks) / adamw / "
    "adam, with onecycle|plateau|cosine schedules. BUDGET: each model must train within "
    "~{max_min:.0f} min of GPU wall-clock (slow models are stopped early and keep their best "
    "epoch) and <= {max_params:,} params; design within that. Each tried architecture above "
    "reports its validation r AND its train seconds, so prefer ideas that are both strong and "
    "efficient.\n\n"
    "Give each proposal a ### TRAIN line (JSON). Fields you omit are sampled for you, so specify "
    "only what you care about."
)

CODER_SYSTEM = """\
You are the CODING agent in a team that searches for neural-network architectures on a genomics \
regression task. A proposer agent decides what to try; your only job is to turn ONE architecture \
idea into correct, trainable PyTorch code. You do not choose the idea, the training recipe or \
the hyperparameters of the optimiser.

Rules:
- Implement the DESIGN faithfully. Do not swap in a different, simpler network because it is \
easier to write. If part of the design cannot be met inside the contract, make the smallest \
change and say so in a one-line `# NOTE:` comment at the top of the code.
- Define `class Net(nn.Module)` with no constructor arguments; every size is hard-coded inside.
- Input x is (batch, 4, L) channels-first, one-hot ACGT. Conv1d takes it directly; transpose \
before Linear, attention or a batch_first GRU. Return shape (batch,): end with `.squeeze(-1)`.
- Track channel counts through every stride, pool and concatenation, and never hard-code a \
flattened length: use global pooling, or compute it from L.
- It is trained with batch norm in train mode at batch sizes from 2 to 512; avoid in-place ops \
that break autograd and Python branching on tensor values.
- Only torch, numpy, math, random, functools, itertools, typing, re may be imported (no einops, \
scipy, timm). No file, network or process access.
- Write no training code and no augmentation; the harness does both.

Reply with exactly one ```python block defining Net, and nothing after it."""

CODER_USER = """\
{task}
{contract}
BUDGET: <= {max_params:,} parameters; the model is trained for at most ~{max_min:.0f} minutes, \
so keep it efficient.

IDEA: {idea}
WHY: {why}
DESIGN:
{design}

(The harness will train it with: {recipe}.)

Write Net now."""

REPAIR_USER = """\
{task}
{contract}
BUDGET: <= {max_params:,} parameters; the model is trained for at most ~{max_min:.0f} minutes.

IDEA: {idea}
DESIGN:
{design}

Your previous implementation was REJECTED by the validator (repair attempt {attempt} of {total}).
VALIDATOR ERROR: {reason}
{earlier}
PREVIOUS CODE:
```python
{code}
```

Fix it with the smallest change that preserves the idea, and return the complete corrected \
source as exactly one ```python block."""

REFLECTOR_SYSTEM = """\
You are the REFLECTOR in a team that searches for neural-network architectures on a genomics \
regression task. After every round you rewrite the team's LESSONS note: a short, persistent \
list of evidence-backed statements that the proposer reads before choosing the next ideas.

Rules:
- Output ONLY the updated note as markdown bullets ("- ..."), at most {max_chars} characters in \
total and at most 12 bullets, each one sentence. No preamble, no code.
- Start each bullet with one tag: [WORKS], [HURTS], [NEUTRAL], [BUILD-FAIL], [HYPOTHESIS] or \
[TODO]. Cite the validation r and the idea it came from. Never invent a number.
- Rewrite, do not append: merge bullets about the same design choice, drop bullets that newer \
results contradict, keep the strongest evidence.
- Use the evaluator's diversity figures: say which designs give predictions that are \
decorrelated from the best model, and whether adding them raised the ensemble validation r, \
because the final deliverable is an ensemble.
- Record recurring build failures (what broke the code) so the proposer can avoid them.
{noise}You see validation results only."""

NOISE_NOTE = (
    "- Retraining the same design with another seed moves its validation r by roughly 0.005 to "
    "0.01 (measured noise floors in this project): do not call a gap below ~0.01 between two "
    "designs real unless it has repeated; label it [NEUTRAL] or [HYPOTHESIS].\n"
)

REFLECTOR_USER = """\
CURRENT LESSONS NOTE:
{note}

{report}

Write the updated LESSONS note now."""


def _fmt_recipe(spec: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in spec.items()) if spec else "sampled defaults"


# --------------------------------------------------------------------------- parsing

_HEADER = re.compile(r"^[ \t]*###[ \t]*(IDEA|WHY|DESIGN|TRAIN)[ \t]*:[ \t]*", re.M)
_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def parse_ideas(text: str, max_n: int | None = None) -> list[dict]:
    """Split a proposer response into ideas on ``### IDEA:`` and read the fields after it.

    An idea needs a non-empty DESIGN; the coder cannot implement a title alone, and a silently
    under-specified idea would make the coder invent the design, blurring the roles being tested.
    """
    heads = list(_HEADER.finditer(text))
    ideas: list[dict] = []
    cur: dict | None = None
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        field_, body = m.group(1), text[m.end() : end].strip()
        if field_ == "IDEA":
            cur = {
                "idea": body.splitlines()[0].strip() if body else "",
                "why": "",
                "design": "",
                "train_spec": {},
            }
            ideas.append(cur)
        elif cur is not None:
            if field_ == "WHY":
                cur["why"] = body
            elif field_ == "DESIGN":
                cur["design"] = _BLOCK.sub(lambda mm: mm.group(1), body).strip()
            elif field_ == "TRAIN":
                j = re.search(r"\{.*?\}", body, re.S)
                try:
                    spec = json.loads(j.group(0)) if j else {}
                    cur["train_spec"] = spec if isinstance(spec, dict) else {}
                except Exception:
                    cur["train_spec"] = {}  # same as the single loop: unparsable TRAIN -> sampled
    ideas = [x for x in ideas if x["idea"] and x["design"]]
    return ideas[:max_n] if max_n else ideas


def extract_code(text: str) -> str | None:
    """The coder is told to return one block; if it returns several, the last one defining Net wins."""
    blocks = [b.strip() for b in _BLOCK.findall(text)]
    if not blocks:
        return None
    with_net = [b for b in blocks if re.search(r"class\s+Net\b", b)]
    return (with_net or blocks)[-1]


def clean_lessons(text: str, max_chars: int) -> str:
    """Normalise a reflector reply to short markdown bullets; '' when nothing usable came back."""
    text = re.sub(r"```[a-z]*", "", text or "").strip()
    bullets: list[str] = []
    for ln in text.splitlines():
        ln = ln.strip()
        m = re.match(r"^(?:[-*•]|\d+[.)])\s+(.*)$", ln)
        if m and m.group(1).strip():
            bullets.append("- " + m.group(1).strip())
        elif bullets and ln and not ln.startswith("#"):
            bullets[-1] += " " + ln  # a wrapped bullet
    out: list[str] = []
    used = 0
    for b in bullets[:12]:
        if used + len(b) + 1 > max_chars:
            break
        out.append(b)
        used += len(b) + 1
    if not out and bullets:
        out = [bullets[0][: max_chars - 1]]
    return "\n".join(out)


# --------------------------------------------------------------------------- LLM wrapper + logs


def approx_tokens(s: str) -> int:
    return (len(s) + 3) // 4


class RoleLLM:
    """Every agent call goes through here: one JSONL line per call, cost recorded per role.

    The line keeps the full prompt and response so a round can be audited, and the character and
    token estimates so cost per trained model can be reported without parsing text.
    """

    def __init__(self, log_dir: Path, cfg: AgentConfig, llm: Callable | None = None):
        self.log_dir, self.cfg, self._llm = Path(log_dir), cfg, llm
        self._lock = threading.Lock()
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _call(self, system: str, user: str, model: str | None) -> str:
        if self._llm is not None:
            return self._llm(system, user, model)
        return _propose.call_model(system, user, model)  # looked up late so tests can patch it

    def call(self, role: str, rd: int, system: str, user: str, **extra) -> str:
        model = self.cfg.role_models.get(role)
        rec = {
            "event": "call",
            "round": rd,
            "role": role,
            "ts": time.time(),
            "model": model or os.environ.get("LLM_MODEL"),
            "system_chars": len(system),
            "user_chars": len(user),
            "approx_tokens_in": approx_tokens(system) + approx_tokens(user),
            **extra,
        }
        t0 = time.time()
        try:
            text = self._call(system, user, model) or ""
        except Exception as e:
            rec.update(
                ok=False,
                seconds=time.time() - t0,
                error=f"{type(e).__name__}: {str(e)[:300]}",
                response_chars=0,
                approx_tokens_out=0,
            )
            self._write(rd, rec)
            raise
        rec.update(
            ok=True,
            seconds=time.time() - t0,
            response_chars=len(text),
            approx_tokens_out=approx_tokens(text),
            prompt=system + "\n\n=====\n\n" + user,
            response=text,
        )
        self._write(rd, rec)
        return text

    def _write(self, rd: int, rec: dict) -> None:
        with self._lock, open(self.log_dir / f"round_{rd:02d}.jsonl", "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")


def summarize_usage(out_dir: str | Path, overhead: int = CLI_OVERHEAD_TOKENS) -> dict:
    """Aggregate ``agent_logs/round_*.jsonl`` and plan files into the fair-comparison numbers."""
    out_dir = Path(out_dir)
    roles: dict[str, dict] = {}
    for f in sorted((out_dir / "agent_logs").glob("round_*.jsonl")):
        for ln in f.read_text().splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("event") != "call":
                continue
            d = roles.setdefault(
                r["role"],
                dict(calls=0, failed_calls=0, tokens_in=0, tokens_out=0, seconds=0.0),
            )
            d["seconds"] += float(r.get("seconds") or 0)
            if r.get("ok"):
                d["calls"] += 1
                d["tokens_in"] += int(r["approx_tokens_in"])
                d["tokens_out"] += int(r["approx_tokens_out"])
            else:
                d["failed_calls"] += 1
    items = []
    for f in sorted((out_dir / "agent_logs").glob("plan_r*.json")):
        items += json.loads(f.read_text()).get("items", [])
    n_models = len(list(out_dir.glob("r*_arch_*_meta.json")))
    calls = sum(d["calls"] for d in roles.values())
    content = sum(d["tokens_in"] + d["tokens_out"] for d in roles.values())
    coded = [i for i in items if i.get("coder_calls")]
    ok_items = [i for i in coded if i.get("status") in ("coded", "trained", "failed", "skipped")]
    out = {
        "models_trained": n_models,
        "llm_calls": calls,
        "by_role": roles,
        "approx_content_tokens": content,
        "cli_overhead_tokens_per_call": overhead,
        "approx_tokens_incl_cli_overhead": content + calls * overhead,
        "calls_per_model": calls / n_models if n_models else None,
        "content_tokens_per_model": content / n_models if n_models else None,
        "tokens_incl_overhead_per_model": (content + calls * overhead) / n_models
        if n_models
        else None,
        "coder": {
            "ideas_coded": len(coded),
            "first_try_ok": sum(1 for i in ok_items if i.get("repairs", 0) == 0),
            "repaired_ok": sum(1 for i in ok_items if i.get("repairs", 0) > 0),
            "rejected_after_repairs": sum(1 for i in coded if i.get("status") == "rejected"),
            "repair_calls": sum(max(0, i.get("coder_calls", 0) - 1) for i in coded),
        },
    }
    return out


# --------------------------------------------------------------------------- plan files


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def plan_path(out_p: Path, rd: int) -> Path:
    return out_p / "agent_logs" / f"plan_r{rd:02d}.json"


def load_plan(out_p: Path, rd: int) -> dict | None:
    p = plan_path(out_p, rd)
    return json.loads(p.read_text()) if p.exists() else None


def save_plan(out_p: Path, plan: dict) -> None:
    _atomic_text(plan_path(out_p, plan["round"]), json.dumps(plan, indent=1, default=str))


def _new_item(slot: int, idea: dict) -> dict:
    return {
        "slot": slot,
        "idea": idea["idea"],
        "why": idea["why"],
        "design": idea["design"],
        "train_spec": idea["train_spec"],
        "status": "idea",
        "source": None,
        "n_params": None,
        "coder_calls": 0,
        "repairs": 0,
        "attempt_reasons": [],
        "reason": "",
        "model_id": None,
    }


# --------------------------------------------------------------------------- context


@dataclass
class ModelRecord:
    """The only per-model facts any agent may see: validation-side numbers, never test."""

    model_id: str
    round: int
    idea: str
    val: float
    n_params: int
    train_s: float
    epochs: int | None
    recipe: str


class AgentCtx:
    def __init__(self, args, h, cfg: AgentConfig, llm: RoleLLM, out_p: Path):
        self.args, self.h, self.cfg, self.llm, self.out_p = args, h, cfg, llm, out_p
        self.history: list[dict] = []  # same shape as the single loop's history
        self.records: list[ModelRecord] = []
        self._vp: dict[str, np.ndarray] = {}
        self.midx = 0
        self.lessons = ""
        self.reflected_through = -1
        self.target = args.rounds * args.per_round
        self.first_prompt_written = False
        self.lock = threading.Lock()

    def val_pred(self, mid: str) -> np.ndarray:
        if mid not in self._vp:
            self._vp[mid] = np.load(self.out_p / f"{mid}.npz")["val_pred"].astype(float)
        return self._vp[mid]

    def n_select(self) -> int:
        """Models to train this round (what the coordinator keeps)."""
        return max(
            1, min(self.cfg.select_per_round or self.args.per_round, self.target - self.midx)
        )

    def n_ideas(self) -> int:
        if (
            "coordinate" in self.cfg.stages
        ):  # over-generate; the coordinator cuts back to n_select()
            return max(self.n_select() + 1, self.cfg.ideas_per_round or 3 * self.args.per_round)
        n = self.cfg.ideas_per_round or self.args.per_round
        return max(1, min(n, self.target - self.midx))

    def say(self, msg: str) -> None:
        print(msg, flush=True)


# --------------------------------------------------------------------------- stages

STAGES: dict[str, Callable[[AgentCtx, dict], None]] = {}


def register_stage(name: str):
    def deco(fn):
        STAGES[name] = fn
        return fn

    return deco


def proposer_prompt(ctx: AgentCtx, n: int, rd: int) -> tuple[str, str]:
    args, h = ctx.args, ctx.h
    # Reuse the single loop's own history rendering (top-18 by score, discarded list with reasons)
    # so the proposer sees exactly what the baseline proposer sees about earlier models.
    _, user = build_arch_prompt(
        n,
        ctx.history,
        seq_len=args.seq_len,
        D=args.D,
        max_params=args.max_params,
        note=None,
    )
    final = f"\nPropose {n} architectures now."
    user = user.removesuffix(final)
    parts = [user]
    if ctx.lessons:
        parts.append(
            "\nLESSONS FROM EARLIER ROUNDS (kept by a reflector agent from validation results; "
            "evidence, not rules):\n" + ctx.lessons
        )
    note = h.guidance_note(rd)
    guidance = note[: -len(h.train_note)].rstrip() if note.endswith(h.train_note) else note
    harness_recipe = ctx.cfg.recipe_source == "harness"
    if harness_recipe:
        # with context the launcher appends its input-context paragraph AFTER the training note, so the suffix test above
        # misses it; the proposer must never be told it may choose a recipe
        guidance = re.sub(r"\n{3,}", "\n\n", guidance.replace(h.train_note, "")).strip()
    setup = (PROPOSER_SETUP_HARNESS if harness_recipe else PROPOSER_SETUP).format(
        max_min=h.cap_min, max_params=args.max_params
    )
    parts.append("\nNOTE: " + (guidance + "\n\n" if guidance else "") + setup)
    if ctx.cfg.manager == "quota":
        parts.append("\n" + coverage_text_quota(ctx))
    parts.append(
        f"\nPropose {n} architectures now (IDEA / WHY / DESIGN{'' if harness_recipe else ' / TRAIN'} for each)."
    )
    system = PROPOSER_SYSTEM.replace("{n}", str(n))
    if harness_recipe:
        system = _TRAIN_LINE.sub("", system)
    if ctx.cfg.proposer_style == "explore":
        system += "\n\n" + PROPOSER_EXPLORE
    mode = manager_mode(ctx, rd)
    if mode == "explore":
        system += "\n\n" + PROPOSER_EXPLORE + "\n\n" + PROPOSER_MANAGER_EXPLORE
    elif mode == "exploit":
        system += "\n\n" + PROPOSER_MANAGER_EXPLOIT
    elif mode == "quota":
        system += "\n\n" + PROPOSER_MANAGER_QUOTA
    if "coordinate" in ctx.cfg.stages:
        k = ctx.n_select()
        system += (
            f"\n\nA COORDINATOR agent will choose which {k} of your {n} proposals are trained, so give a spread of distinct "
            f"bets, each worth training, rather than {n} variants of one idea."
        )
    return system, "\n".join(parts)


@register_stage("propose")
def stage_propose(ctx: AgentCtx, plan: dict) -> None:
    if plan["items"]:
        return  # resumed inside this round: the ideas are already paid for
    rd, n = plan["round"], ctx.n_ideas()
    sys_p, user = proposer_prompt(ctx, n, rd)
    if not ctx.first_prompt_written:
        (ctx.out_p / "prompts").mkdir(exist_ok=True)
        _atomic_text(
            ctx.out_p / "prompts" / f"prompt_r{rd:02d}.txt",
            sys_p + "\n\n=====\n\n" + user,
        )
        ctx.first_prompt_written = True
    ideas: list[dict] = []
    for attempt in range(ctx.cfg.propose_retries + 1):
        extra = (
            ""
            if attempt == 0
            else (
                "\n\nYour previous reply could not be parsed. Reply again using exactly the "
                "### IDEA: / ### WHY: / ### DESIGN: / ### TRAIN: layout, with a non-empty DESIGN."
            )
        )
        text = ctx.llm.call("proposer", rd, sys_p, user + extra, attempt=attempt, n_requested=n)
        ideas = parse_ideas(text, max_n=n)
        if ideas:
            break
    plan["n_requested"], plan["n_parsed"] = n, len(ideas)
    plan["items"] = [_new_item(i, x) for i, x in enumerate(ideas)]
    save_plan(ctx.out_p, plan)
    ctx.say(f"  [round {rd}] proposer: {len(ideas)}/{n} ideas")


# --------------------------------------------------------------------------- manager (explore early, exploit late)
MANAGER_EXPLORE = (
    "MANAGER MODE: EXPLORE (round {rd1} of {total}; the first {n_explore} rounds are exploration). Pick the {k} candidates that most widen what "
    "the final ensemble covers: an architecture class that is under-represented in the coverage table below beats a better-looking duplicate of "
    "a class already well covered, and a training recipe (optimizer, learning rate, weight decay, batch size, schedule) that has not been tried "
    "yet is worth a bet. Accept a lower expected validation r for a new mechanism or recipe. Do not pick width or depth tweaks of models "
    "already trained."
)
MANAGER_EXPLOIT = (
    "MANAGER MODE: EXPLOIT (round {rd1} of {total}; exploration ended after round {n_explore}). Pick the {k} candidates most likely to raise the "
    "ensemble: refinements (depth, width, kernels, pooling, head, recipe) of the models that carry the most weight, or designs that fill a gap the "
    "stack shows. Avoid mechanisms that failed or scored clearly below the rest."
)
PROPOSER_MANAGER_EXPLORE = (
    "EXPLORATION ROUND: the search is early and the final product is an ensemble that should span many architectures and training recipes. "
    "Vary the TRAIN line as well as the network: at least half of your ideas should use a recipe that the tried list does not show yet "
    "(for example AdamW against Muon, a learning rate between 3e-4 and 4e-3, weight decay between 1e-5 and 5e-2, batch size 64 to 512, cosine "
    "or plateau schedule), and several should use a different architecture class from the best models so far."
)
PROPOSER_MANAGER_EXPLOIT = (
    "REFINEMENT ROUND: the exploration phase is over. Build on what the tried list shows works: improve the best architectures and their "
    "recipes (depth, width, kernel sizes, pooling, head, regularisation, learning rate), and keep one bet that tests something different."
)
MANAGER_QUOTA = (
    "MANAGER MODE: QUOTA. Every round trains one EXPLOIT model and {k1} EXPLORE model(s). EXPLOIT: the candidate most likely to raise the ensemble by refining a model that carries weight in the stack or by "
    "filling a gap the stack shows. EXPLORE: the candidate that adds the architectural features least represented in the coverage table (especially the ones listed as NOT YET TRIED), accepting a lower "
    "expected validation r; a width or depth tweak of a trained model does not count as exploration. Do not pick two candidates that rely on the same new feature."
)
PROPOSER_MANAGER_QUOTA = (
    "QUOTA ROUND: the final product is an ensemble, so each round needs both refinement and genuinely different bets. Make at least half of your ideas introduce an architectural feature that the coverage "
    "table below shows as rare or untried (a different block type, skip-connection scheme, normalisation, readout, receptive-field layout, attention placement, ...); make the others refinements of the "
    "models that carry weight."
)
PROPOSER_SETUP_HARNESS = (
    "SETUP: you choose WHAT to build; a coding agent writes the Net class from your DESIGN, so be specific and stay within what can be written with the allowed imports. A fixed harness trains it: "
    "the input includes real plasmid context, reverse-complement + random-shift augmentation are ALWAYS applied, and the harness assigns each model its own training recipe (optimizer, learning rate, "
    "weight decay, batch size, schedule), so do not propose one. BUDGET: each model must train within ~{max_min:.0f} min of GPU wall-clock (slow models are stopped early and keep their best epoch) and "
    "<= {max_params:,} params; design within that. Each tried architecture above reports its validation r AND its train seconds, so prefer ideas that are both strong and efficient."
)
_TRAIN_LINE = re.compile(r"### TRAIN:.*?\}\n", re.S)
_FEATURES = {
    "squeeze-excitation": r"class\s+\w*(SE|Squeeze)\w*|SELayer|Squeeze",
    "self-attention": r"MultiheadAttention|TransformerEncoder|nn\.Transformer|scaled_dot_product_attention",
    "recurrent": r"nn\.(LSTM|GRU|RNN)\b",
    "state-space": r"[Mm]amba|[Ss]tate.?[Ss]pace|selective.?scan",
    "U-Net-like (up/down path)": r"ConvTranspose|F\.interpolate|nn\.Upsample",
    "dilated convolutions": r"dilation\s*=\s*(?!1\b)\d+",
    "attention pooling": r"AttnPool|AttentionPool|[Aa]ttn[_A-Za-z]*\s*=",
    "dense / concatenation skips": r"torch\.(cat|concat)\(",
    "layer or group norm": r"nn\.(LayerNorm|GroupNorm|InstanceNorm1d|RMSNorm)",
    "depthwise convolutions": r"groups\s*=\s*(?!1\b)",
    "gating (GLU / sigmoid gates)": r"F\.glu|nn\.GLU|\*\s*torch\.sigmoid|\*\s*F\.sigmoid|\.sigmoid\(\)\s*\*",
    "top-k hit statistics": r"\.topk\(|torch\.topk",
    "position-resolved flatten readout": r"nn\.Flatten|\.flatten\(1\)|\.view\([^)]*-1\)|\.reshape\([^)]*-1\)",
    "global max pooling": r"AdaptiveMaxPool1d|amax\(|\.max\(dim|\.max\(-1\)|adaptive_max_pool1d",
    "windowed / multiple-instance": r"\.unfold\(|windows",
    "parallel multi-kernel branches": r"ModuleList",
    "second-order terms (einsum / bmm)": r"einsum|bmm|torch\.matmul",
}
_ARCH_CLASSES = {
    "self-attention": r"MultiheadAttention|TransformerEncoder|nn\.Transformer|scaled_dot_product_attention",
    "recurrent": r"nn\.(LSTM|GRU|RNN)\b",
    "state-space": r"[Mm]amba|[Ss]tate.?[Ss]pace|selective.?scan",
    "U-Net-like (up/down path)": r"ConvTranspose|F\.interpolate|nn\.Upsample",
    "dilated": r"dilation\s*=\s*(?!1\b)\d+",
    "attention pooling": r"AttnPool|AttentionPool|[Aa]ttn[_A-Za-z]*\s*=",
}


def manager_mode(ctx: AgentCtx, rd: int) -> str | None:
    """'explore' for the first round(explore_frac * rounds) rounds, then 'exploit'; None when the manager is off."""
    if ctx.cfg.manager == "none":
        return None
    if ctx.cfg.manager == "quota":
        return "quota"
    return "explore" if rd < round(ctx.args.rounds * ctx.cfg.explore_frac) else "exploit"


def coverage_text(ctx: AgentCtx) -> str:
    """Architecture classes (from the trained models' source) and training recipes of everything trained so far, validation-side facts only."""
    classes: dict[str, int] = {k: 0 for k in _ARCH_CLASSES}
    n = 0
    for m in ctx.records:
        try:
            src = (
                json.loads((ctx.out_p / f"{m.model_id}_meta.json").read_text()).get("source") or ""
            )
        except Exception:  # noqa: BLE001 - coverage is advice, never a reason to stop
            continue
        n += 1
        for k, pat in _ARCH_CLASSES.items():
            if re.search(pat, src):
                classes[k] += 1
    recipes: dict[str, int] = {}
    for m in ctx.records:
        recipes[m.recipe] = recipes.get(m.recipe, 0) + 1
    lines = [
        f"COVERAGE of the {len(ctx.records)} trained models (all are convolutional; counts below are models that also contain the feature):"
    ]
    lines.append(
        " - " + ", ".join(f"{k}: {v}" for k, v in classes.items())
        if n
        else " - (no source available yet)"
    )
    top = sorted(recipes.items(), key=lambda kv: -kv[1])[:8]
    lines.append(
        " - training recipes used (count): " + "; ".join(f"{r[:70]} ({c})" for r, c in top)
        if top
        else " - no recipes yet"
    )
    return "\n".join(lines)


def coverage_text_quota(ctx: AgentCtx) -> str:
    """Architectural features of everything trained so far (from the models' source) and the ones nobody has tried; recipes are left out when the harness assigns them."""
    counts = {k: 0 for k in _FEATURES}
    n = 0
    for m in ctx.records:
        try:
            src = (
                json.loads((ctx.out_p / f"{m.model_id}_meta.json").read_text()).get("source") or ""
            )
        except Exception:  # noqa: BLE001 - coverage is advice, never a reason to stop
            continue
        n += 1
        for k, pat in _FEATURES.items():
            if re.search(pat, src):
                counts[k] += 1
    lines = [
        f"COVERAGE of the {len(ctx.records)} trained models (counts are models whose source contains the feature):"
    ]
    if n:
        lines.append(" - " + ", ".join(f"{k}: {v}" for k, v in counts.items() if v))
        untried = [k for k, v in counts.items() if v == 0]
        lines.append(
            " - NOT YET TRIED: "
            + (
                ", ".join(untried)
                if untried
                else "(every listed feature appears in at least one model)"
            )
        )
    else:
        lines.append(" - (no source available yet)")
    if ctx.cfg.recipe_source == "proposal":
        recipes: dict[str, int] = {}
        for m in ctx.records:
            recipes[m.recipe] = recipes.get(m.recipe, 0) + 1
        top = sorted(recipes.items(), key=lambda kv: -kv[1])[:6]
        if top:
            lines.append(
                " - training recipes used (count): " + "; ".join(f"{r[:70]} ({c})" for r, c in top)
            )
    return "\n".join(lines)


def ensemble_status_text(ctx: AgentCtx) -> str:
    """The validation stack of all trained models (positive ElasticNetCV on z-scored predictions): which models carry weight."""
    if len(ctx.records) < 4:
        return ""
    try:
        from albench.ensemble.stack import fit_zscored_stack

        V = np.stack([ctx.val_pred(m.model_id) for m in ctx.records]).astype(np.float32)
        st = fit_zscored_stack(V, np.asarray(ctx.h.yva, float))
        coef = np.asarray(st.coef, float)
        pred = np.asarray(st.predict(V), float)
        r_ens = ctx.h.pear(pred, ctx.h.yva)
    except Exception as ex:  # noqa: BLE001
        return f"ENSEMBLE STATUS unavailable ({type(ex).__name__})"
    order = np.argsort(-coef)
    lines = [
        f"ENSEMBLE STATUS: stack of all {len(ctx.records)} trained models, fitted on validation (optimistic): val r {r_ens:.4f}; {int((coef > 1e-9).sum())} models carry weight, "
        "the rest add nothing beyond them."
    ]
    for i in order[:8]:
        if coef[i] <= 1e-9:
            break
        m = ctx.records[i]
        lines.append(
            f" - weight {coef[i] / max(coef.sum(), 1e-12):.2f}, own val r {m.val:.4f}, corr with stack {_corr(ctx.val_pred(m.model_id), pred):.2f}: {m.idea[:80]}"
        )
    return "\n".join(lines)


def coordinator_prompt(ctx: AgentCtx, plan: dict) -> tuple[str, str]:
    n, k = len(plan["items"]), ctx.n_select()
    done = sorted(ctx.records, key=lambda m: -m.val)[:15]
    parts = ["TRAINED SO FAR (validation r, train seconds; best first):"]
    parts += [
        f" - val r {m.val:.4f}, {m.train_s:.0f}s, {m.n_params:,} params: {m.idea[:90]}"
        for m in done
    ] or [" - nothing yet"]
    failed = [h_ for h_ in ctx.history if not h_.get("ok")][-8:]
    if failed:
        parts.append("\nRECENT FAILURES:")
        parts += [f" - {h_['idea'][:80]}: {str(h_.get('reason', ''))[:100]}" for h_ in failed]
    if ctx.lessons:
        parts.append(
            "\nLESSONS (kept by a reflector from validation results; evidence, not rules):\n"
            + ctx.lessons
        )
    mode = manager_mode(ctx, plan["round"])
    if mode == "quota":
        parts.append("\n" + coverage_text_quota(ctx))
        es = ensemble_status_text(ctx)
        if es:
            parts.append("\n" + es)
        parts.append("\n" + MANAGER_QUOTA.format(k1=max(k - 1, 0)))
    elif mode:
        total = ctx.args.rounds
        n_explore = round(total * ctx.cfg.explore_frac)
        parts.append("\n" + coverage_text(ctx))
        if ctx.cfg.manager == "schedule_ens":
            es = ensemble_status_text(ctx)
            if es:
                parts.append("\n" + es)
        directive = MANAGER_EXPLORE if mode == "explore" else MANAGER_EXPLOIT
        parts.append(
            "\n" + directive.format(rd1=plan["round"] + 1, total=total, n_explore=n_explore, k=k)
        )
    parts.append(f"\nCANDIDATES this round ({n}):")
    for i, it in enumerate(plan["items"], 1):
        recipe_line = (
            ""
            if ctx.cfg.recipe_source == "harness"
            else f"\n    TRAIN: {_fmt_recipe(it['train_spec'])}"
        )
        parts.append(
            f"[{i}] IDEA: {it['idea']}\n    WHY: {it['why']}\n    DESIGN: {it['design'][:900]}{recipe_line}"
        )
    parts.append(f"\nChoose {k} of the {n} candidates now.")
    return COORDINATOR_SYSTEM.replace("{n}", str(n)).replace("{k}", str(k)), "\n".join(parts)


@register_stage("coordinate")
def stage_coordinate(ctx: AgentCtx, plan: dict) -> None:
    if plan.get("coordinated"):
        return  # resumed inside this round
    rd, k = plan["round"], ctx.n_select()
    items = plan["items"]
    if len(items) <= k:
        plan["coordinated"] = True
        plan["coordinator"] = {
            "selected": list(range(len(items))),
            "text": "",
            "note": "no more candidates than slots",
        }
        save_plan(ctx.out_p, plan)
        return
    if ctx.cfg.coordinator_mode == "llm":
        sys_p, user = coordinator_prompt(ctx, plan)
        text = ctx.llm.call("coordinator", rd, sys_p, user, n_candidates=len(items), k=k)
        chosen = parse_selection(text, len(items), k)
    else:  # control arms: same over-generation, no judgment
        import random

        chosen = (
            sorted(random.Random(f"{ctx.out_p}|{rd}").sample(range(len(items)), k))
            if ctx.cfg.coordinator_mode == "random"
            else list(range(k))
        )
        text = f"coordinator_mode={ctx.cfg.coordinator_mode}: picked without an LLM call"
    for i, it in enumerate(items):
        if i not in chosen:
            it.update(
                status="skipped", reason="not selected by the coordinator", coordinator_skipped=True
            )
    plan["coordinated"] = True
    plan["coordinator"] = {
        "selected": chosen,
        "text": text[:2000],
        "manager_mode": manager_mode(ctx, rd),
    }
    save_plan(ctx.out_p, plan)
    ctx.say(f"  [round {rd}] coordinator: kept {[i + 1 for i in chosen]} of {len(items)}")


def coder_prompt(ctx: AgentCtx, item: dict) -> tuple[str, str]:
    args = ctx.args
    user = CODER_USER.format(
        task=TASK.format(D=args.D),
        contract=CONTRACT.format(
            seq_len=args.seq_len, max_params=args.max_params, max_fwd_ms=400.0
        ),
        max_params=args.max_params,
        max_min=ctx.h.cap_min,
        idea=item["idea"],
        why=item["why"] or "(none given)",
        design=item["design"],
        recipe="a recipe the harness assigns (do not assume one)"
        if ctx.cfg.recipe_source == "harness"
        else _fmt_recipe(item["train_spec"]),
    )
    return CODER_SYSTEM, user


def repair_prompt(
    ctx: AgentCtx, item: dict, code: str | None, reasons: list[str], attempt: int
) -> tuple[str, str]:
    args = ctx.args
    earlier = ""
    if len(reasons) > 1:
        earlier = (
            "EARLIER ATTEMPTS FAILED WITH: " + " | ".join(r[:120] for r in reasons[:-1]) + "\n"
        )
    user = REPAIR_USER.format(
        task=TASK.format(D=args.D),
        contract=CONTRACT.format(
            seq_len=args.seq_len, max_params=args.max_params, max_fwd_ms=400.0
        ),
        max_params=args.max_params,
        max_min=ctx.h.cap_min,
        idea=item["idea"],
        design=item["design"],
        attempt=attempt,
        total=ctx.cfg.repairs,
        reason=reasons[-1][:600],
        earlier=earlier,
        code=code if code else "(your reply contained no ```python block)",
    )
    return CODER_SYSTEM, user


def code_item(ctx: AgentCtx, rd: int, item: dict) -> None:
    """Coder call, validate, and up to ``repairs`` repair calls that show the error back."""
    args = ctx.args
    reasons: list[str] = list(item["attempt_reasons"])
    last_src: str | None = item["source"]
    for attempt in range(len(reasons), ctx.cfg.repairs + 1):
        if attempt == 0:
            sys_p, user = coder_prompt(ctx, item)
        else:
            sys_p, user = repair_prompt(ctx, item, last_src, reasons, attempt)
        text = ctx.llm.call("coder", rd, sys_p, user, slot=item["slot"], attempt=attempt)
        item["coder_calls"] += 1
        src = extract_code(text)
        if src is None:
            reason = "no python code block in the coder reply"
        else:
            last_src = src
            model, verdict = build_and_check(src, seq_len=args.seq_len, max_params=args.max_params)
            if model is not None:
                item.update(
                    status="coded",
                    source=src,
                    n_params=verdict.n_params,
                    repairs=attempt,
                )
                return
            reason = verdict.reason
        reasons.append(reason)
        item["attempt_reasons"] = reasons
        item["source"] = last_src
    item.update(status="rejected", reason=reasons[-1], repairs=ctx.cfg.repairs)


@register_stage("code")
def stage_code(ctx: AgentCtx, plan: dict) -> None:
    rd = plan["round"]
    todo = [it for it in plan["items"] if it["status"] == "idea"]

    def work(it: dict) -> None:
        code_item(ctx, rd, it)
        with ctx.lock:
            save_plan(ctx.out_p, plan)

    if ctx.cfg.coder_workers > 1 and len(todo) > 1:
        with ThreadPoolExecutor(max_workers=ctx.cfg.coder_workers) as ex:
            for f in [ex.submit(work, it) for it in todo]:
                f.result()
    else:
        for it in todo:
            work(it)
    for it in plan["items"]:
        if it["status"] == "rejected":
            if not any(h_.get("idea") == it["idea"] and not h_.get("ok") for h_ in ctx.history):
                ctx.history.append({"idea": it["idea"], "ok": False, "reason": it["reason"]})
    save_plan(ctx.out_p, plan)
    coded = [it for it in plan["items"] if not it.get("coordinator_skipped")]
    n_ok = sum(1 for it in coded if it["status"] != "rejected")
    n_rep = sum(1 for it in coded if it["status"] != "rejected" and it["repairs"] > 0)
    ctx.say(f"  [round {rd}] coder: {n_ok}/{len(coded)} built ({n_rep} needed repair)")


def _train_item(ctx: AgentCtx, plan: dict, item: dict) -> None:
    """Train + screen + save one coded idea.

    This mirrors the per-model block of ``arch_autoresearch.main`` (same seeding, time cap,
    exploded-prediction screens, meta keys) on purpose: the comparison is only fair if a model
    from either loop goes through the same training and screening.
    """
    import torch

    args, h, rd = ctx.args, ctx.h, plan["round"]

    def fail(reason: str) -> None:
        item.update(status="failed", reason=reason)
        ctx.history.append({"idea": item["idea"], "ok": False, "reason": reason})
        save_plan(ctx.out_p, plan)

    model, verdict = build_and_check(
        item["source"], seq_len=args.seq_len, max_params=args.max_params
    )
    if model is None:
        ctx.say(f"    discard: {verdict.reason[:75]}")
        return fail(f"rebuild failed: {verdict.reason}")
    if (
        getattr(h, "wrap_model", None) is not None
    ):  # flank-context students: WideCrop around the checked architecture
        model = h.wrap_model(model)
    tc = h.sample_train({} if ctx.cfg.recipe_source == "harness" else item["train_spec"])
    torch.manual_seed(1000 + ctx.midx)
    t0 = time.time()
    try:
        model, info, used = h.train_one(
            model, h.Xtr, h.ytr, h.Xva, h.yva, tc, max_sec=h.cap_min * 60
        )
    except Exception as e:
        ctx.say(f"    train-err: {type(e).__name__}: {str(e)[:60]}")
        return fail(f"train err {type(e).__name__}")
    if model is None:
        ctx.say(f"    train-fail: {info.get('error')}")
        return fail(info.get("error", "fail"))
    res = {
        "val_pred": h.predict_rc(model, h.Xva),
        "test_pred": h.predict_rc(model, h.Xte),
    }
    _vp = res["val_pred"]
    if (not np.isfinite(_vp).all()) or float(np.abs(_vp).max()) > 1e3:
        ctx.say("    train-fail: exploded predictions (|pred|>1e3)")
        return fail("exploded predictions (|pred|>1e3)")
    for k, Xk in h.Xsets.items():
        res[f"test_pred_{k}"] = h.predict_rc(model, Xk)
    _bad = [k for k, v in res.items() if (not np.isfinite(v).all()) or float(np.abs(v).max()) > 1e3]
    if _bad:
        ctx.say(f"    train-fail: exploded predictions on {_bad[0]} (|pred|>1e3)")
        return fail(f"exploded predictions on {_bad[0]} (|pred|>1e3)")
    r = h.pear(res["test_pred"], h.te_oracle)
    vr = h.pear(res["val_pred"], h.yva)
    mid = f"r{rd:02d}_arch_{ctx.midx:02d}"
    meta = {
        "model_id": mid,
        "strategy": "arch_search",
        "round": rd,
        "idea": item["idea"],
        "why": item["why"],
        "n_params": verdict.n_params,
        "train_config": tc,
        "optimizer_used": used,
        "val_pearson": vr,
        "best_val_pearson": vr,
        "test_oracle_pearson": r,
        "val_mse": info.get("best_val_mse"),
        "epochs_trained": info.get("epochs_trained"),
        "train_time_sec": time.time() - t0,
        "gpu": h.gpu,
        "source": item["source"],
        "harness": getattr(h, "harness", {}),
        # agent-loop extras (load_cell ignores unknown keys)
        "agent": {
            "design": item["design"],
            "coder_calls": item["coder_calls"],
            "repairs": item["repairs"],
            "train_spec": item["train_spec"],
            "recipe_source": ctx.cfg.recipe_source,
            "manager": ctx.cfg.manager,
        },
    }
    h.atomic_savez(
        ctx.out_p / f"{mid}.npz",
        **{k: np.asarray(x, np.float32) for k, x in res.items()},
    )
    h.atomic_write_text(ctx.out_p / f"{mid}_meta.json", json.dumps(meta, indent=2, default=str))
    ctx.history.append(
        {
            "idea": item["idea"],
            "ok": True,
            "score": vr,
            "n_params": verdict.n_params,
            "val": vr,
            "train_s": time.time() - t0,
        }
    )
    ctx.records.append(
        ModelRecord(
            mid,
            rd,
            item["idea"],
            vr,
            verdict.n_params,
            time.time() - t0,
            info.get("epochs_trained"),
            _fmt_recipe({**tc}),
        )
    )
    ctx._vp[mid] = np.asarray(res["val_pred"], np.float32).astype(float)
    item.update(status="trained", model_id=mid)
    ctx.midx += 1
    save_plan(ctx.out_p, plan)
    ctx.say(
        f"    {mid} r={r:.4f} val={vr:.4f} {verdict.n_params:,}p {used} "
        f"{info.get('epochs_trained')}ep {time.time() - t0:.0f}s  {item['idea'][:46]}"
    )


def _z(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / (a.std() + 1e-12)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def make_report(ctx: AgentCtx, plan: dict) -> str:
    """Deterministic evaluator summary of one round: validation numbers only.

    The diversity lines (correlation with the best earlier model, ensemble validation r before
    and after) are what the reflector needs to say which designs add something to an ensemble;
    the single-call proposer never sees them.
    """
    rd, yva = plan["round"], ctx.h.yva
    prior = [m for m in ctx.records if m.round < rd]
    now = [m for m in ctx.records if m.round <= rd]
    best_prior = max(prior, key=lambda m: m.val) if prior else None
    lines = [
        f"ROUND {rd} RESULTS (validation only; r = Pearson on the validation set, higher is better)"
    ]
    for it in plan["items"]:
        if it["status"] == "trained":
            m = next(x for x in ctx.records if x.model_id == it["model_id"])
            rank = 1 + sum(1 for x in now if x.val > m.val)
            bits = [f"val r {m.val:.4f}"]
            if best_prior:
                bits.append(f"{m.val - best_prior.val:+.4f} vs previous best {best_prior.val:.4f}")
            bits.append(f"rank {rank}/{len(now)}")
            line = f" - TRAINED: {', '.join(bits)}; {m.n_params:,} params; {m.recipe}; "
            line += f"{m.epochs} epochs, {m.train_s:.0f}s"
            if prior:
                vp = ctx.val_pred(m.model_id)
                cb = _corr(vp, ctx.val_pred(best_prior.model_id))
                cm = _corr(vp, np.mean([_z(ctx.val_pred(x.model_id)) for x in prior], axis=0))
                line += f"; prediction corr with best earlier model {cb:.2f}, with mean of earlier models {cm:.2f}"
            line += f"; idea: {m.idea[:80]}"
            if it["repairs"]:
                first = (it["attempt_reasons"] or [""])[0][:110]
                line += f" [coder needed {it['repairs']} repair(s); first validator error: {first}]"
            lines.append(line)
        elif it.get("coordinator_skipped"):
            continue
        else:
            why = it["reason"] or it["status"]
            lines.append(f" - NOT TRAINED ({it['status']}): {why[:160]}; idea: {it['idea'][:80]}")
    if len(now) >= 2 and now != prior:

        def ens(ms):
            return ctx.h.pear(np.mean([_z(ctx.val_pred(x.model_id)) for x in ms], axis=0), yva)

        after = ens(now)
        if prior:
            before = ens(prior)
            lines.append(
                f" ENSEMBLE (mean of z-scored predictions of all models) val r: "
                f"{before:.4f} -> {after:.4f} ({after - before:+.4f}) with {len(prior)} -> {len(now)} models"
            )
        else:
            lines.append(f" ENSEMBLE val r of this round's {len(now)} models: {after:.4f}")
    if now:
        best = max(now, key=lambda m: m.val)
        vals = [m.val for m in now]
        lines.append(
            f" RUNNING: {len(now)} models trained, best val r {best.val:.4f} ({best.idea[:60]}), "
            f"mean {np.mean(vals):.4f}"
        )
    return "\n".join(lines)


@register_stage("evaluate")
def stage_evaluate(ctx: AgentCtx, plan: dict) -> None:
    for it in plan["items"]:
        if it["status"] != "coded":
            continue
        if ctx.midx >= ctx.target:
            it.update(status="skipped", reason="target model count already reached")
            continue
        _train_item(ctx, plan, it)
    plan["report"] = make_report(ctx, plan)
    plan["evaluated"] = True
    save_plan(ctx.out_p, plan)
    ctx.say("  " + plan["report"].replace("\n", "\n  "))


def reflector_prompt(ctx: AgentCtx, plan: dict) -> tuple[str, str]:
    sys_p = REFLECTOR_SYSTEM.replace("{max_chars}", str(ctx.cfg.lessons_max_chars)).replace(
        "{noise}", NOISE_NOTE if ctx.cfg.reflector_noise_note else ""
    )
    user = REFLECTOR_USER.format(
        note=ctx.lessons or "(empty: this is the first round)", report=plan["report"]
    )
    return sys_p, user


@register_stage("reflect")
def stage_reflect(ctx: AgentCtx, plan: dict) -> None:
    """Reflect on every finished round not yet reflected on (so the stage can run first or last)."""
    done_rounds = sorted(
        int(p.stem.split("_r")[-1]) for p in (ctx.out_p / "agent_logs").glob("plan_r*.json")
    )
    for rd in done_rounds:
        if rd <= ctx.reflected_through:
            continue
        pl = plan if rd == plan["round"] else load_plan(ctx.out_p, rd)
        if not pl or not pl.get("report") or not pl["items"]:  # nothing evaluated, nothing to learn
            continue
        sys_p, user = reflector_prompt(ctx, pl)
        text = ctx.llm.call("reflector", rd, sys_p, user)
        note = clean_lessons(text, ctx.cfg.lessons_max_chars)
        if note:
            ctx.lessons = note
            _atomic_text(ctx.out_p / "lessons.md", note + "\n")
        with open(ctx.out_p / "agent_logs" / "lessons_history.jsonl", "a") as fh:
            fh.write(json.dumps({"round": rd, "accepted": bool(note), "note": note}) + "\n")
        ctx.reflected_through = rd
        ctx.say(
            f"  [round {rd}] reflector: lessons note {len(ctx.lessons)} chars"
            + ("" if note else " (reply unusable; kept the previous note)")
        )


# --------------------------------------------------------------------------- run


def _rebuild_state(ctx: AgentCtx) -> tuple[int, int | None]:
    """Rebuild history / lessons from disk; return (next fresh round, open round to finish or None)."""
    out_p = ctx.out_p
    rows = []
    for f in sorted(out_p.glob("r*_arch_*_meta.json")):
        try:
            m = json.loads(f.read_text())
        except Exception:
            continue
        if m.get("strategy") == "arch_search" and (out_p / (m["model_id"] + ".npz")).exists():
            rows.append(m)
    rows.sort(key=lambda m: (int(m["round"]), m["model_id"]))
    for m in rows:
        ctx.history.append(
            {
                "idea": m["idea"],
                "ok": True,
                "score": m.get("val_pearson"),
                "n_params": m.get("n_params"),
                "val": m.get("val_pearson"),
                "train_s": m.get("train_time_sec"),
            }
        )
        tc = m.get("train_config") or {}
        ctx.records.append(
            ModelRecord(
                m["model_id"],
                int(m["round"]),
                m["idea"],
                float(m["val_pearson"]),
                int(m.get("n_params") or 0),
                float(m.get("train_time_sec") or 0),
                m.get("epochs_trained"),
                _fmt_recipe(tc),
            )
        )
    if rows:
        ctx.midx = 1 + max(int(m["model_id"].split("_")[-1]) for m in rows)
    max_rd = max([int(m["round"]) for m in rows], default=-1)
    open_rd = None
    for f in sorted((out_p / "agent_logs").glob("plan_r*.json")):
        pl = json.loads(f.read_text())
        max_rd = max(max_rd, pl["round"])
        for it in pl["items"]:
            if it["status"] == "rejected":
                ctx.history.append({"idea": it["idea"], "ok": False, "reason": it["reason"]})
            elif it["status"] == "failed":
                ctx.history.append({"idea": it["idea"], "ok": False, "reason": it["reason"]})
        if any(it["status"] not in _RESOLVED for it in pl["items"]) or (
            pl["items"] and not pl.get("evaluated")
        ):
            open_rd = pl["round"]
    lp = out_p / "lessons.md"
    if lp.exists():
        ctx.lessons = lp.read_text().strip()
    lh = out_p / "agent_logs" / "lessons_history.jsonl"
    if lh.exists():
        ctx.reflected_through = max(
            [json.loads(x)["round"] for x in lh.read_text().splitlines() if x.strip()],
            default=-1,
        )
    return max_rd + 1, open_rd


def _write_role_prompts(out_p: Path, cfg: AgentConfig) -> None:
    txt = "\n\n#####\n\n".join(
        [
            "PROPOSER SYSTEM (n filled per round)\n" + PROPOSER_SYSTEM,
            "PROPOSER SETUP NOTE\n" + PROPOSER_SETUP,
            "CODER SYSTEM\n" + CODER_SYSTEM,
            "CODER USER TEMPLATE\n" + CODER_USER,
            "REPAIR USER TEMPLATE\n" + REPAIR_USER,
            "REFLECTOR SYSTEM\n" + REFLECTOR_SYSTEM,
            "REFLECTOR USER TEMPLATE\n" + REFLECTOR_USER,
        ]
    )
    _atomic_text(out_p / "prompts" / "agent_roles.txt", txt + "\n")


def run_agent_loop(args, h, llm: Callable | None = None) -> None:
    """Run the multi-agent loop to ``rounds x per_round`` trained models.

    ``h`` is the harness bundle built by ``arch_autoresearch.main`` (data arrays, ``train_one``,
    ``sample_train``, ``predict_rc``, ``pear``, atomic writers, ``guidance_note(rd)``,
    ``train_note``, ``cap_min``, ``gpu``). It is passed in rather than imported because that file is
    a script whose training code must stay the single source of truth.
    """
    if getattr(args, "full_control", False):
        raise AgentConfigError("--loop agent does not support --full_control")
    if getattr(args, "style", "none") not in ("none", None):
        raise AgentConfigError("--loop agent replaces --style; leave --style at 'none'")
    cfg = AgentConfig.from_arg(getattr(args, "agent_cfg", ""))
    if cfg.recipe_source == "harness":
        os.environ["AL_RECIPE_MODE"] = (
            "harness"  # the launcher's sample_train draws from the safe set
        )
    out_p = Path(args.out_dir)
    (out_p / "prompts").mkdir(parents=True, exist_ok=True)
    (out_p / "agent_logs").mkdir(exist_ok=True)
    ctx = AgentCtx(args, h, cfg, RoleLLM(out_p / "agent_logs", cfg, llm), out_p)
    start_rd, open_rd = (0, None)
    if args.resume:
        start_rd, open_rd = _rebuild_state(ctx)
        if ctx.midx:
            ctx.say(
                f"  [resume] {ctx.midx} models on disk; next round {start_rd}"
                + (f"; finishing open round {open_rd}" if open_rd is not None else "")
            )
    h.atomic_write_text(
        out_p / "config.json",
        json.dumps(
            {
                "args": vars(args),
                "agent": {**asdict(cfg), "stages": list(cfg.stages)},
                "LLM_MODEL": os.environ.get("LLM_MODEL"),
                "argv": sys.argv,
                "resumed_from_round": start_rd,
            },
            indent=2,
            default=str,
        ),
    )
    _write_role_prompts(out_p, cfg)

    t_start = time.time()
    max_rounds = args.rounds + max(6, args.rounds // 4)
    rd, budget_hit, last_err, streak = start_rd - 1, False, None, 0
    pending_open = open_rd
    while ctx.midx < ctx.target or pending_open is not None:
        if pending_open is not None:
            rd, pending_open = pending_open, None
        else:
            rd += 1
            if rd >= max_rounds:
                break
            if time.time() - t_start > args.budget_hours * 3600:
                ctx.say(f"  [budget {args.budget_hours}h reached at round {rd}] stopping")
                budget_hit = True
                break
        plan = load_plan(out_p, rd) or {"round": rd, "items": [], "evaluated": False}
        try:
            for name in cfg.stages:
                STAGES[name](ctx, plan)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            ctx.say(f"  [round {rd}] stage failed: {msg}")
            if type(e).__name__ == "RateLimitExceeded":
                ctx.say(
                    f"RATE_LIMITED at round {rd} ({ctx.midx} models so far): exiting 75 so the "
                    f"dispatcher backs off; no rounds burned"
                )
                raise SystemExit(75)
            for it in plan["items"]:
                if it["status"] not in _RESOLVED:
                    it.update(status="failed", reason=f"stage error: {msg[:120]}")
            plan["evaluated"] = True
            save_plan(out_p, plan)
            streak = streak + 1 if msg == last_err else 1
            last_err = msg
            if ctx.midx == 0 and streak >= 4:
                ctx.say(
                    f"FAILED_ARCH: identical stage failure x{streak} with nothing trained; aborting"
                )
                raise SystemExit(3)
            continue
        use = summarize_usage(out_p, cfg.cli_overhead_tokens)
        _atomic_text(out_p / "agent_logs" / "summary.json", json.dumps(use, indent=1))
        ctx.say(
            f"  [round {rd}] usage so far: {use['llm_calls']} calls, ~{use['approx_content_tokens']:,} "
            f"content tokens, {use['models_trained']} models"
        )
    ctx.say(f"DONE_ARCH trained={ctx.midx} elapsed={(time.time() - t_start) / 3600:.1f}h")
    use = summarize_usage(out_p, cfg.cli_overhead_tokens)
    _atomic_text(out_p / "agent_logs" / "summary.json", json.dumps(use, indent=1))
    ctx.say(
        "AGENT_USAGE "
        + json.dumps(
            {
                k: use[k]
                for k in (
                    "models_trained",
                    "llm_calls",
                    "approx_content_tokens",
                    "approx_tokens_incl_cli_overhead",
                    "calls_per_model",
                )
            }
        )
    )
    if not budget_hit and ctx.midx > 0:
        (out_p / "COMPLETE").write_text(
            f"rounds_run={rd + 1} models={ctx.midx} target={ctx.target}\n"
        )
    if ctx.midx == 0:
        ctx.say("FAILED_ARCH: 0 models trained")
        raise SystemExit(3)


def _main(argv: list[str]) -> None:
    """`python -m albench.arch_search.agent_loop summarize DIR...` -> cost table per run."""
    if len(argv) < 2 or argv[0] != "summarize":
        print(_main.__doc__)
        return
    tot = dict(models=0, calls=0, content=0, with_overhead=0)
    for d in argv[1:]:
        u = summarize_usage(d)
        roles = " ".join(f"{r}={v['calls']}" for r, v in sorted(u["by_role"].items()))
        print(
            f"{d}: models={u['models_trained']} calls={u['llm_calls']} ({roles}) "
            f"content_tokens={u['approx_content_tokens']:,} incl_cli_overhead="
            f"{u['approx_tokens_incl_cli_overhead']:,} coder={u['coder']}"
        )
        tot["models"] += u["models_trained"]
        tot["calls"] += u["llm_calls"]
        tot["content"] += u["approx_content_tokens"]
        tot["with_overhead"] += u["approx_tokens_incl_cli_overhead"]
    if tot["models"]:
        print(
            f"TOTAL models={tot['models']} calls/model={tot['calls'] / tot['models']:.2f} "
            f"content_tokens/model={tot['content'] / tot['models']:,.0f} "
            f"incl_overhead/model={tot['with_overhead'] / tot['models']:,.0f}"
        )


if __name__ == "__main__":
    _main(sys.argv[1:])
