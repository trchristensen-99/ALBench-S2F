"""LLM-based AutoResearch — Karpathy-style.

An LLM (Claude) acts as the HP optimizer, using:
  - The HP space specification (with reasonable ranges)
  - Recent history (HP configs + val_mse)
  - Global KB summary (cross-experiment patterns)
  - Configurable system prompt / persona

It proposes K new HP configs per round, using judgement and "taste" instead of
fixed perturbation rules.

Independent SLURM-submittable: this strategy plugs into the standard
scaling_hp_search.py pipeline. Requires ANTHROPIC_API_KEY env var.

Configurable knobs (via env vars at SLURM time):
  LLM_PROMPT_STYLE = "default" | "explore" | "exploit" | "critic" | "diverse" | "neutral"
  LLM_MODEL        = "claude-sonnet-4-6" (or claude-opus-4-7, claude-haiku-4-5)
  LLM_TEMPERATURE  = float (API path only; sample low/high pairs to probe explore/exploit)
  LLM_MAX_TOKENS   = 4000
  LLM_ALLOW_NOVEL_AXES = "0" (default) | "1" — let the LLM propose off-menu axes
                     in an "extra" object (recognized ones applied, others recorded)
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, str(REPO))

from experiments.hp_strategies import Strategy  # noqa: E402
from experiments.scaling_hp_search import EXPERIMENTAL_KNOBS_DOC, HPConfig  # noqa: E402

# The 15 core axes; anything else the LLM emits is gathered into HPConfig.extra
# (only when novel-axes mode is on).
_CORE_KEYS = {
    "lr",
    "batch_size",
    "conv_dropout",
    "dense_dropout",
    "n_layers",
    "width_base",
    "width_jitter",
    "block_class",
    "ks",
    "pct_start",
    "optimizer",
    "weight_decay",
    "use_shift_aug",
    "shift_max",
    "use_evoaug",
}


class RateLimitExceeded(Exception):
    """Backend is usage/rate-limited and a completion could not be obtained
    within the allotted wait budget.

    FAIRNESS CONTRACT: callers MUST NOT fall back to random sampling on this
    exception (that silently degrades a throttled LLM strategy into random
    search, which is unfair vs. un-throttled strategies). Instead, checkpoint
    and stop/resume — see scaling_hp_search.run_search.
    """

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


# Substrings (lowercased) that indicate a usage/rate/overload limit rather than
# a genuine error. Deliberately broad — we log the raw text so the exact CLI
# message format can be refined later ("find out down the road").
_RATE_LIMIT_MARKERS = (
    "usage limit",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "rate-limit",
    "session limit",
    "hit your session",
    "too many requests",
    "429",
    "overloaded",
    "529",
    "quota",
    "resets at",
    "· resets",
    "resets ",
    "will reset",
    "try again later",
    "usage limit reached",
)


def _looks_rate_limited(text: str | None) -> bool:
    t = (text or "").lower()
    return any(m in t for m in _RATE_LIMIT_MARKERS)


def _parse_reset_seconds(text: str | None) -> float | None:
    """Best-effort: extract a unix-epoch reset time embedded in a limit message
    and return seconds-from-now. Returns None if not found / implausible."""
    import time as _time

    m = re.search(r"\b(1[0-9]{9})\b", text or "")
    if m:
        delta = int(m.group(1)) - _time.time()
        if 0 < delta < 24 * 3600:
            return float(delta)
    return None


def _retry_after_from_exc(e: Exception) -> float | None:
    try:
        ra = e.response.headers.get("retry-after")  # type: ignore[attr-defined]
        return float(ra) if ra else None
    except Exception:
        return None


def _compute_wait(retry_after: float | None, attempt: int, base: float) -> float:
    """Server-provided retry_after wins; else capped exponential backoff + jitter."""
    import random

    if retry_after and retry_after > 0:
        return float(retry_after)
    wait = min(base * (2 ** (attempt - 1)), 1800.0)  # cap a single sleep at 30 min
    return wait + random.uniform(0, min(30.0, wait * 0.1))


HP_SPACE_SPEC = """HP SEARCH SPACE for LegNet sequence-to-function predictor:
  lr             : log-uniform in [1e-5, 1e-2]
  batch_size     : categorical {32, 64, 128, 256, 512, 1024}
  conv_dropout   : uniform [0.0, 0.3]   (Peter: conv layers need less dropout than dense)
  dense_dropout  : uniform [0.0, 0.5]
  n_layers       : int in [2, 12]       (LegNet inverted residual blocks)
  width_base     : categorical {16, 32, 64, 128, 256}  (channels in first layer)
  width_jitter   : list of length n_layers, each in [0.5, 2.0]
                   (PER-LAYER width multiplier — encourages non-uniform schedules; 1.0s = flat)
  block_class    : categorical {"eff", "ag", "plain"}
                   - "eff": EffBlock (inverted residual + SE), default LegNet
                   - "ag": AlphaGenome-style (RMSBatchNorm + StandardizedConv1D + QuickGELU);
                     Peter: stronger inductive bias for reg-genomics
                   - "plain": vanilla 2-layer conv (baseline)
  ks             : kernel size, categorical {3, 5, 7, 9, 11}
  pct_start      : OneCycleLR warmup fraction, categorical {0.1, 0.2, 0.3, 0.4}
  optimizer      : categorical {"adam", "adamw", "muon"}
                   (muon = Keller Jordan's; orthogonalized momentum, often outperforms AdamW on conv stacks)
  weight_decay   : log-uniform in [1e-6, 1e-2]
  use_shift_aug  : bool   (training-time random shift augmentation)
  shift_max      : categorical {5, 10, 15, 20}  (max shift if use_shift_aug)
  use_evoaug     : bool   (training-time evolutionary augmentation, intensity=medium)
"""


ADVANCED_GUIDANCE = """
ADVANCED HP EXPLORATION (use the full search space, not just lr/bs/dropout):

* Per-layer width with `width_jitter` — try NON-UNIFORM schedules. Examples:
    - wide-middle: [0.7, 1.5, 1.8, 1.8, 1.5, 0.7] (10-layer-style waist)
    - narrow-to-wide ramp: [0.5, 0.7, 1.0, 1.3, 1.6, 1.9]
    - wide-then-narrow: [2.0, 1.5, 1.0, 0.7, 0.5]
    All-1.0 is the lazy default — avoid it unless deliberately chosen.

* Conv vs dense dropout placement (the asymmetry IS the lever): try low-conv +
  high-dense (conv_dropout=0.05, dense_dropout=0.4) vs the inverse vs both-equal.

* Architecture × optimizer interactions to probe:
    - EffBlock + Muon → known best so far at D=30k. Refine its neighborhood.
    - AGBlock + AdamW → second-tier, but stronger inductive bias for reg-genomics
      and may dominate at larger D. Try wide widths (width_base=256) + 6-10 layers.
    - AGBlock + Muon → has had stability/NaN issues at moderate D. If proposing this combo,
      use lower lr (1e-4 to 5e-4) and lower weight_decay.
    - PlainBlock + Muon → competitive at D=300k. Worth testing especially for high D.
    - Avoid PlainBlock + AdamW (consistently weak).

* Depth × width trade: at fixed param count, deeper-narrower beats shallower-wider
  for sequence tasks. Suggest n_layers ∈ {8, 10, 12} with width_base ∈ {32, 64}
  more often than n_layers=2 with width_base=256.

* Aggressive HP combinations worth testing (orthogonal to the local optimum):
    - Very high lr (5e-3 to 1e-2) + Muon + cosine-equivalent (pct_start=0.3) — Muon's
      orthogonalization tolerates higher lr than AdamW.
    - Very low weight_decay (1e-6) + high dropout (>=0.3) for ensemble diversity.
    - Maximum n_layers (12) with small width_base (16) — under-explored regime.

Use these as IDEAS, not constraints — your judgement on what fits the current
trajectory matters most."""

PROMPT_STYLES = {
    "default": """You are an expert in deep learning hyperparameter optimization for genomic sequence-to-function prediction (LegNet on K562 MPRA oracle labels).

Your job: propose {n} new HP configurations that you believe will achieve LOW validation MSE (lower is better).

You have:
  - The full HP space below (use ALL dimensions, not just lr/bs/dropout)
  - Recent observations from this run + aggregated patterns from prior experiments
  - Advanced exploration guidance highlighting under-utilized dimensions

Apply your judgement and taste. Don't just sample uniformly — use the patterns. Be willing to try bold combinations that you think SHOULD work even if untested. Mix exploitation (refine current best) with smart exploration (probe under-explored regions of the space). Aim for a mix of GLOBAL optima search (probe far-from-best regions) and LOCAL refinement (small perturbations around the best).

OUTPUT FORMAT: A single JSON array with {n} HP config dicts. Each dict must have ALL the keys in HP_SPACE_SPEC. No commentary outside the JSON.""",
    "explore": """You are a creative HP search assistant. Propose {n} BOLD, DIVERSE new HP configurations for LegNet on K562. Many should explore regions the existing experiments have NOT tried. Don't just refine the current best — surprise me with combinations that might unlock new performance regimes.

EXPLOIT THE FULL SEARCH SPACE: use non-uniform width_jitter, try all 3 block_classes, try all 3 optimizers, vary conv vs dense dropout asymmetrically. Aggressive depths (10-12 layers) and unusual width_base values are under-explored. The advanced guidance below has specific suggestions.

OUTPUT: JSON array of {n} HP config dicts. No extra text.""",
    "exploit": """You are a precision HP tuner. Propose {n} new HP configurations that are SMALL VARIATIONS around the current best configs. Be surgical — change one or two HPs at a time, by small amounts, to find local optimums. Preserve the best (block_class, optimizer) combo; vary lr, weight_decay, conv_dropout, dense_dropout, width_jitter slightly.

OUTPUT: JSON array of {n} HP config dicts. No extra text.""",
    "critic": """You are a meta-learner analyzing HP search history. First, internally reason about why some configs underperformed (e.g. wrong optimizer-block combo, too-aggressive lr, dropout placement). Then propose {n} new configs that AVOID the failure modes you identified while building on what works.

Look especially for: combinations that diverged (val NaN or very high val_loss), HP regions consistently producing weak results, and structural issues (e.g. block_class+optimizer interactions).

OUTPUT: JSON array of {n} HP config dicts. No extra text.""",
    "diverse": """You are an ensemble-aware HP search assistant. The downstream user will ElasticNetCV ensemble all proposed configs. Propose {n} HP configurations that are likely INDIVIDUALLY DECENT but ALSO MAXIMALLY DIVERSE from each other — different architectures (eff/ag/plain), different optimizers (adam/adamw/muon), different regularization regimes (dropout asymmetry, weight_decay span), different depths and width schedules. The goal is a complementary ensemble that collectively covers the loss surface.

OUTPUT: JSON array of {n} HP config dicts. No extra text.""",
    # Deliberately minimal: states the task and objective only. No advice on which
    # HP regions to favor, no editorializing ("ag is best", "non-uniform is better"),
    # and (paired with neutral build_prompt) no cross-experiment KB priors. Used to
    # test whether the other styles' suggestions bias which HP regions get explored.
    "neutral": """You are a hyperparameter optimization assistant for LegNet on K562 MPRA oracle labels. Propose {n} new HP configurations to minimize validation MSE (lower is better). Each config must specify every key in the HP space below.

OUTPUT FORMAT: A single JSON array with {n} HP config dicts. No commentary outside the JSON.""",
}


def summarize_history(history: list[tuple], max_items: int = 30) -> str:
    """Format recent history compactly for the LLM prompt."""
    if not history:
        return "  (no observations yet)"
    # Sort by val_mse ascending if available, else by index
    # history is list[(HPConfig, val_pearson)] but we want val_mse — convert
    # For now using val_pearson (higher better); we'll switch if metric changes
    sorted_h = sorted(history, key=lambda x: -x[1])  # higher val_r first
    lines = []
    for i, (hp, metric) in enumerate(sorted_h[:max_items]):
        d = asdict(hp)
        # Compact: just key=value, drop width_jitter (long list)
        d.pop("width_jitter", None)
        d.pop("seed", None)
        kv = ", ".join(f"{k}={v}" for k, v in d.items())
        lines.append(f"  [val_r={metric:.4f}] {kv}")
    return "\n".join(lines)


def summarize_kb(kb_summary: dict) -> str:
    """Format KB top-quartile summary for the LLM prompt."""
    if not kb_summary:
        return "  (no KB data yet)"
    lines = []
    for hp_name, info in kb_summary.items():
        if info["type"] == "numeric":
            lines.append(
                f"  {hp_name:<18}: median={info['median']:.4g}, IQR=[{info['q25']:.4g}, {info['q75']:.4g}] (n={info['n']})"
            )
        else:
            top = list(info["frequencies"].items())[:3]
            top_str = ", ".join(f"{k}:{v}" for k, v in top)
            lines.append(f"  {hp_name:<18}: top: {top_str} (n={info['n']})")
    return "\n".join(lines)


def build_prompt(
    n: int,
    history: list[tuple],
    kb_summary: dict,
    style: str = "default",
    allow_novel: bool = False,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt).

    The ``neutral`` style strips every prior that could steer which HP regions
    the model explores: the ADVANCED_GUIDANCE block, the cross-experiment KB
    summary, and the editorial closing lines ("ag is theoretically best",
    "non-uniform schedules are often better"). It keeps the HP-space spec and
    the run's OWN observations (the search signal) only.

    When ``allow_novel`` is True, the prompt invites optional off-menu axes
    inside an "extra" object (default mode is byte-identical to before).
    """
    system = PROMPT_STYLES.get(style, PROMPT_STYLES["default"]).format(n=n)
    neutral = style == "neutral"

    parts = [HP_SPACE_SPEC]
    if not neutral:
        parts.append(ADVANCED_GUIDANCE)
    if allow_novel:
        parts.append(f"\n{EXPERIMENTAL_KNOBS_DOC}")
    parts.append(
        f"\nCURRENT SESSION TOP {min(30, len(history))} OBSERVATIONS "
        f"(sorted by val_pearson, descending):\n{summarize_history(history, max_items=30)}"
    )
    if not neutral and kb_summary:
        parts.append(
            f"\nGLOBAL KB SUMMARY (top-quartile across {kb_summary.get('_n_total', '?')} "
            f"prior records):\n"
            f"{summarize_kb({k: v for k, v in kb_summary.items() if not k.startswith('_')})}"
        )

    closing = (
        f"\nPropose {n} new HP configs. Output ONLY a valid JSON array, no extra text. "
        "Each dict must have these keys (in this order):\n"
        "  lr, batch_size, conv_dropout, dense_dropout, n_layers, width_base, width_jitter, "
        "block_class, ks, pct_start, optimizer, weight_decay, use_shift_aug, shift_max, use_evoaug\n"
        "`width_jitter` should be a list of length n_layers."
    )
    if not neutral:
        closing += (
            " Non-uniform schedules (e.g. wider middle, narrower end) are often better — "
            "don't default everything to 1.0.\n"
            "`block_class` is genuinely uncertain — try all three across your proposals. "
            '"ag" is theoretically best for regulatory genomics but rarely tested in this '
            'codebase. "eff" is the safe default. "plain" is the baseline.'
        )
    if allow_novel:
        closing += (
            '\nOptionally add an "extra" object per dict for the experimental axes described above.'
        )
    parts.append(closing)
    return system, "\n".join(parts)


def _call_claude_cli(system: str, user: str, model: str) -> str:
    """One CLI attempt. Raises RateLimitExceeded on usage-limit, RuntimeError else."""
    import subprocess

    prompt = f"<system>\n{system}\n</system>\n\n{user}"
    claude_bin = os.environ.get(
        "CLAUDE_BIN",
        "/grid/wsbs/home_norepl/christen/.conda/envs/claude_code_env/bin/claude",
    )
    # Strip date suffix from model id for CLI (CLI expects sonnet, opus, etc.)
    cli_model = model.split("-202")[0] if "-202" in model else model
    try:
        result = subprocess.run(
            [claude_bin, "-p", prompt, "--model", cli_model],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as e:
        # Transient; treat like a short throttle so the retry loop waits + retries
        # rather than aborting the whole strategy.
        raise RateLimitExceeded(f"claude CLI timed out after 300s: {e}", retry_after=30)
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode != 0:
        # Log raw text so the exact usage-limit message can be characterized later.
        print(
            f"  [cli rc={result.returncode}] stderr={result.stderr[:1000]!r} "
            f"stdout={result.stdout[:300]!r}",
            flush=True,
        )
        if _looks_rate_limited(combined):
            raise RateLimitExceeded(
                f"claude CLI usage/rate limit (rc={result.returncode}): {result.stderr[:300]}",
                retry_after=_parse_reset_seconds(combined),
            )
        raise RuntimeError(f"claude CLI failed rc={result.returncode}: {result.stderr[:500]}")
    # rc==0 but a usage-limit notice in stdout (no JSON array present) → treat as limit
    if "[" not in result.stdout and _looks_rate_limited(result.stdout):
        print(f"  [cli rc=0 looks rate-limited] {result.stdout[:500]!r}", flush=True)
        raise RateLimitExceeded(
            f"claude CLI returned usage-limit text: {result.stdout[:300]}",
            retry_after=_parse_reset_seconds(result.stdout),
        )
    # rc==0 with no JSON array and not a known limit message: log the raw text so
    # the downstream "No JSON recoverable" failure is diagnosable (e.g. opus arm).
    if "[" not in result.stdout:
        print(
            f"  [cli rc=0 no-JSON] model={model!r} "
            f"stdout={result.stdout[:800]!r} stderr={result.stderr[:300]!r}",
            flush=True,
        )
    return result.stdout


def _call_claude_api(
    system: str, user: str, model: str, max_tokens: int, temperature: float | None = None
) -> str:
    """One API attempt. Raises RateLimitExceeded on 429/529/conn, else propagates."""
    import anthropic

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            **kwargs,
        )
    except anthropic.RateLimitError as e:  # HTTP 429
        raise RateLimitExceeded(
            f"anthropic 429 rate limit: {e}", retry_after=_retry_after_from_exc(e)
        )
    except anthropic.APIStatusError as e:
        if getattr(e, "status_code", None) == 529 or "overloaded" in str(e).lower():
            raise RateLimitExceeded(
                f"anthropic 529 overloaded: {e}", retry_after=_retry_after_from_exc(e)
            )
        raise
    except anthropic.APIConnectionError as e:
        raise RateLimitExceeded(f"anthropic connection error: {e}", retry_after=30)
    return resp.content[0].text


def call_claude(
    system: str,
    user: str,
    model: str = None,
    max_tokens: int = 4000,
    temperature: float | None = None,
    max_wait_seconds: float | None = None,
    base_backoff: float = 60.0,
) -> str:
    """Call Claude with pause-and-wait on rate/usage limits.

    Uses Claude Code CLI (Max plan) when CLAUDE_CODE_OAUTH_TOKEN is set, else the
    anthropic API. On a usage/rate/overload limit, sleeps (server retry-after if
    available, else capped exponential backoff) and retries the SAME prompt until
    `max_wait_seconds` is exhausted, then raises RateLimitExceeded. NEVER falls
    back to random sampling.
    """
    import time

    model = model or os.environ.get("LLM_MODEL", "claude-sonnet-4-5-20250929")
    if max_wait_seconds is None:
        max_wait_seconds = float(os.environ.get("LLM_MAX_WAIT_SECONDS", 6 * 3600))
    use_cli = bool(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"))

    deadline = time.time() + max_wait_seconds
    attempt = 0
    while True:
        attempt += 1
        try:
            if use_cli:
                # The Claude CLI path does not expose a temperature knob; temperature
                # only affects the anthropic-API path. LLM_USE_CLI=0 to force the API.
                return _call_claude_cli(system, user, model)
            return _call_claude_api(system, user, model, max_tokens, temperature=temperature)
        except RateLimitExceeded as e:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise RateLimitExceeded(
                    f"Rate limited beyond max wait budget ({max_wait_seconds:.0f}s); "
                    f"stop and resume later. Last: {e}",
                    retry_after=e.retry_after,
                )
            wait = min(_compute_wait(e.retry_after, attempt, base_backoff), remaining + 1)
            print(
                f"  [rate-limit] attempt {attempt}: backend limited; sleeping "
                f"{wait:.0f}s then retrying SAME prompt "
                f"(retry_after={e.retry_after}, budget_left={remaining:.0f}s)",
                flush=True,
            )
            time.sleep(wait)


def _extract_json_objects(text: str) -> list[dict]:
    """Fallback parser: scan for sequential top-level JSON objects, skipping any
    leading/trailing prose or markup the array slice choked on. Uses raw_decode
    so it only ever accepts fully-parseable {...} objects — it never repairs or
    mutates values, so a recovered config is byte-identical to what the model
    emitted (and is scored/clipped downstream like any other candidate)."""
    decoder = json.JSONDecoder()
    objs: list[dict] = []
    i, ln = 0, len(text)
    while i < ln:
        if text[i] == "{":
            try:
                obj, end = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                i += 1
                continue
            if isinstance(obj, dict):
                objs.append(obj)
            i = end
        else:
            i += 1
    return objs


def parse_response(text: str, n: int) -> list[dict]:
    """Extract JSON array from LLM response. Tolerates ```json blocks etc."""
    # Strip code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    # Find first '[' and try to parse from there
    start = text.find("[")
    if start >= 0:
        # Find matching closing bracket
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end >= 0:
            slice_ = text[start:end]
            try:
                parsed = json.loads(slice_)
                if isinstance(parsed, list):
                    return parsed[:n]
            except json.JSONDecodeError:
                # Some models (e.g. opus-4-7) emit Python-literal booleans
                # (True/False/None) instead of JSON true/false/null. Parse the
                # array as a Python literal — faithful values, no execution.
                try:
                    parsed = ast.literal_eval(slice_)
                    if isinstance(parsed, list):
                        return parsed[:n]
                except (ValueError, SyntaxError):
                    pass  # fall through to object-level recovery
    # Fallback: array slice missing or unparseable — recover individual objects.
    objs = _extract_json_objects(text)
    if not objs:
        raise ValueError("No JSON objects recoverable from LLM response")
    return objs[:n]


def dict_to_hpconfig(d: dict, seed: int, allow_novel: bool = False) -> HPConfig:
    """Coerce LLM-proposed dict to a valid HPConfig. When allow_novel is True,
    any key outside the 15 core axes (plus an explicit "extra" object) is gathered
    into HPConfig.extra for downstream validation/recording."""
    # Defaults if missing
    n_layers = int(d.get("n_layers", 6))
    width_jitter = d.get("width_jitter", [1.0] * n_layers)
    if not isinstance(width_jitter, list) or len(width_jitter) != n_layers:
        width_jitter = [1.0] * n_layers
    # Clip widths to valid range
    width_jitter = [max(0.5, min(2.0, float(x))) for x in width_jitter]
    extra: dict = {}
    if allow_novel:
        nested = d.get("extra")
        if isinstance(nested, dict):
            extra.update(nested)
        # Also catch off-menu keys placed at the top level instead of under "extra".
        for k, v in d.items():
            if k not in _CORE_KEYS and k not in ("extra", "seed"):
                extra[k] = v
    return HPConfig(
        lr=max(1e-5, min(1e-2, float(d.get("lr", 1e-3)))),
        batch_size=int(d.get("batch_size", 256)),
        conv_dropout=max(0.0, min(0.3, float(d.get("conv_dropout", 0.1)))),
        dense_dropout=max(0.0, min(0.5, float(d.get("dense_dropout", 0.2)))),
        n_layers=max(2, min(12, n_layers)),
        width_base=int(d.get("width_base", 64)),
        width_jitter=width_jitter,
        block_class=(
            str(d.get("block_class", "eff")).lower()
            if str(d.get("block_class", "eff")).lower() in ("eff", "ag", "plain")
            else "eff"
        ),
        ks=int(d.get("ks", 5)),
        pct_start=float(d.get("pct_start", 0.3)),
        optimizer=(
            str(d.get("optimizer", "adamw")).lower()
            if str(d.get("optimizer", "adamw")).lower() in ("adam", "adamw", "muon")
            else "adamw"
        ),
        weight_decay=max(1e-6, min(1e-2, float(d.get("weight_decay", 1e-4)))),
        use_shift_aug=bool(d.get("use_shift_aug", False)),
        shift_max=int(d.get("shift_max", 15)),
        use_evoaug=bool(d.get("use_evoaug", False)),
        seed=seed,
        extra=extra,
    )


class LLMAutoResearch(Strategy):
    """LLM-based HP optimizer (Karpathy-style AutoResearch).

    Each suggest() builds a prompt from history + KB, calls Claude, parses
    response, returns HPConfigs. On a rate/usage limit it raises
    RateLimitExceeded (caller checkpoints + stops/resumes); on transient parse
    errors it retries the call. It NEVER falls back to random sampling — doing so
    would silently make a throttled LLM strategy behave like random search, which
    is unfair vs. un-throttled strategies.

    Configurable via env vars:
      LLM_PROMPT_STYLE (default | explore | exploit | critic | diverse | neutral)
      LLM_MODEL        (claude-opus-4-7, claude-sonnet-4-5, etc.)
      LLM_USE_KB       (1 = include KB summary in prompt, 0 = session-only)
      LLM_TEMPERATURE  (float; API path only — sample low/high pairs to probe
                        explore/exploit. Unset = backend default)

    The ``neutral`` style forces KB off regardless of LLM_USE_KB (it is the
    no-prior probe), so it never gets cross-experiment hints.
    """

    name = "llm_autoresearch"

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.style = os.environ.get("LLM_PROMPT_STYLE", "default")
        self.model = os.environ.get("LLM_MODEL", "claude-sonnet-4-5-20250929")
        # neutral is the no-prior probe: never feed it the cross-experiment KB.
        self.use_kb = os.environ.get("LLM_USE_KB", "1") == "1" and self.style != "neutral"
        _temp = os.environ.get("LLM_TEMPERATURE")
        self.temperature = float(_temp) if _temp not in (None, "") else None
        self.parse_retries = int(os.environ.get("LLM_PARSE_RETRIES", "3"))
        # Off-menu axes: only when explicitly enabled (default OFF keeps behavior
        # identical to the core-15-axis search).
        self.allow_novel = os.environ.get("LLM_ALLOW_NOVEL_AXES", "0") == "1"

    def suggest(self, n: int) -> list[HPConfig]:
        # Build context
        kb_summary = {}
        if self.use_kb:
            try:
                from experiments.hp_knowledge_base import get_kb

                kb = get_kb()
                recs = kb.load_all()
                kb_summary = kb.summary(recs) if recs else {}
                kb_summary["_n_total"] = len(recs)
            except Exception:
                pass

        configs: list[HPConfig] = []
        last_err: Exception | None = None
        for attempt in range(1, self.parse_retries + 1):
            need = n - len(configs)
            if need <= 0:
                break
            system, user = build_prompt(
                need, self.history, kb_summary, style=self.style, allow_novel=self.allow_novel
            )
            response = None
            try:
                response = call_claude(system, user, model=self.model, temperature=self.temperature)
                parsed = parse_response(response, need)
                configs.extend(
                    dict_to_hpconfig(
                        d, seed=int(self.rng.integers(2**31)), allow_novel=self.allow_novel
                    )
                    for d in parsed
                )
            except RateLimitExceeded:
                # Propagate — caller checkpoints + stops/resumes. NEVER random.
                raise
            except Exception as e:
                last_err = e
                print(
                    f"  LLMAutoResearch parse/call error "
                    f"(attempt {attempt}/{self.parse_retries}): {e}",
                    flush=True,
                )
                if response is not None:
                    print(
                        f"  [parse-fail raw response] model={self.model!r} "
                        f"len={len(response)} repr={response[:1500]!r}",
                        flush=True,
                    )

        if len(configs) < n:
            raise RuntimeError(
                f"LLMAutoResearch could not obtain {n} valid configs after "
                f"{self.parse_retries} attempts (got {len(configs)}). "
                f"Last error: {last_err}. Refusing to fall back to random — "
                f"stop and resume."
            )
        return configs[:n]


# Auto-register in hp_strategies.STRATEGY_REGISTRY when imported
def _register():
    from experiments import hp_strategies

    hp_strategies.STRATEGY_REGISTRY["llm_autoresearch"] = LLMAutoResearch


_register()
