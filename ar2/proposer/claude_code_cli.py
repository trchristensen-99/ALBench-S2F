"""Proposer backed by the headless Claude Code CLI (spec section 6).

One call is one fresh, isolated, single-shot session: a new scratch working directory, a throw-away configuration home, no resumed session, the prompt on
standard input, a pinned full model id, a tool allow-list and a turn limit. The adapter hard-codes only a small table of *expected* options; ``check()``
verifies each against ``claude --help`` and fails, naming the option, when one is absent, so a CLI update cannot silently change behaviour.

What is passed through to the CLI process is deliberately short: PATH, locale, proxy and certificate variables and the credential variables needed to
authenticate (``CLAUDE_CODE_OAUTH_TOKEN``, ``ANTHROPIC_API_KEY`` ...). ``HOME`` and ``CLAUDE_CONFIG_DIR`` point into a temporary directory, so user-level
memory files, settings and connected tools are not read. On a machine where login state lives in ``~/.claude`` (not in a token variable) set
``proposer.inherit_credentials: true`` to copy only the credentials file into the temporary configuration home.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from ar2.proposer.base import (
    VALIDATORS,
    ProposerCapabilityError,
    ProposerHalt,
    ProposerPaused,
    ProposerResult,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ALIASES = {"sonnet", "opus", "haiku", "default", "fast", "best", "opusplan"}
# capability -> acceptable spellings in `claude --help` (first match wins); the check fails when none is present
REQUIRED_OPTIONS: dict[str, tuple[str, ...]] = {
    "non-interactive print mode": ("--print", "-p"),
    "machine-readable output": ("--output-format",),
    "pinned model": ("--model",),
    "tool allow-list": ("--allowedTools", "--allowed-tools"),
    "turn limit": (
        "--max-turns",
        "--max-budget-usd",
    ),  # CLI 2.1.x has no --max-turns: a dollar budget per call bounds the session instead
}
TOOL_DISABLE_OPTIONS = ("--tools", "--disallowedTools", "--disallowed-tools")
ALL_TOOLS = (
    "Bash",
    "Edit",
    "MultiEdit",
    "Write",
    "Read",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "NotebookEdit",
    "Task",
    "TodoWrite",
)
PASSTHROUGH_ENV = (
    "PATH",
    "LANG",
    "LC_ALL",
    "TZ",
    "TMPDIR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
)
# copied from experiments/llm_autoresearch.py (not imported: ar2 is standalone). Applied only to calls that already failed.
RATE_LIMIT_MARKERS = (
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
    "will reset",
    "try again later",
    "usage limit reached",
)
AUTH_MARKERS = (
    "authentication",
    "unauthorized",
    "invalid api key",
    "invalid x-api-key",
    "oauth",
    "not logged in",
    "please run /login",
    "401",
    "403",
    "forbidden",
)
EXPECTED_FILES = {"sketches": ("sketches.json",), "candidate": ("model.py", "card.json")}
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def looks_rate_limited(text: str | None) -> bool:
    t = (text or "").lower()
    return any(m in t for m in RATE_LIMIT_MARKERS)


def looks_like_auth_error(text: str | None) -> bool:
    t = (text or "").lower()
    return any(m in t for m in AUTH_MARKERS)


def parse_reset_seconds(text: str | None) -> float | None:
    """Seconds until a unix-epoch reset time embedded in a limit message, or None."""
    m = re.search(r"\b(1[0-9]{9})\b", text or "")
    if m:
        delta = int(m.group(1)) - time.time()
        if 0 < delta < 24 * 3600:
            return float(delta)
    return None


def extract_json_object(text: str) -> Any:
    """First JSON object (or array) in a reply, fenced or bare; None when there is none."""
    dec = json.JSONDecoder()
    for fence in re.findall(r"```(?:json)?\s*\n(.*?)```", text or "", re.S):
        try:
            return json.loads(fence)
        except ValueError:
            continue
    for i, ch in enumerate(text or ""):
        if ch in "{[":
            try:
                return dec.raw_decode(text[i:])[0]
            except ValueError:
                continue
    return None


class ClaudeCodeCLIProposer:
    def __init__(self, cfg: dict[str, Any], sleep=time.sleep) -> None:
        self.cfg = dict(cfg)
        self.bin = str(os.environ.get("CLAUDE_BIN") or cfg.get("claude_bin") or "claude")
        self.model = cfg.get("model")
        self.tagger_model = cfg.get("tagger_model") or self.model
        self.max_turns = int(cfg.get("max_turns", 12))
        self.tagger_max_turns = int(cfg.get("tagger_max_turns", 2))
        self.max_budget_usd = float(cfg.get("max_budget_usd", 4.0))
        self.tagger_max_budget_usd = float(cfg.get("tagger_max_budget_usd", 1.0))
        self.retries = int(cfg.get("retries", 2))
        self.backoff_s = list(cfg.get("backoff_s", [60, 300]))
        self.halt_after = int(cfg.get("halt_after_failures", 5))
        self.allow_selfcheck = bool(cfg.get("allow_selfcheck", True))
        self.scratch_root = Path(cfg.get("scratch_root", "/tmp/ar2_scratch"))
        self.inherit_credentials = bool(cfg.get("inherit_credentials", False))
        self.extra_env = tuple(cfg.get("env_passthrough", ()) or ())
        self.allowed_tools = list(
            cfg.get("allowed_tools") or ["Read(./**)", "Write(./**)", "Edit(./**)"]
        )
        self._sleep = sleep
        self.cli_version: str | None = None
        self.options: dict[str, str] = {}
        self.consecutive_failures = 0
        self.model_ids_seen: dict[str, list[str]] = {"main": [], "tagger": []}
        self._help_text = ""
        self._validate_config()

    # ------------------------------------------------------------------------------------- setup

    def _validate_config(self) -> None:
        for key in ("model",):
            m = self.cfg.get(key)
            if not m:
                raise ValueError(
                    "proposer.model must be set to a full model id (spec 6.1: never an alias)"
                )
            if str(m).lower() in ALIASES:
                raise ValueError(
                    f"proposer.model {m!r} is an alias; use the full model id so that calls stay comparable"
                )
        if self.tagger_model and str(self.tagger_model).lower() in ALIASES:
            raise ValueError(
                f"proposer.tagger_model {self.tagger_model!r} is an alias; use a full model id"
            )
        root = self.scratch_root.resolve()
        if root == REPO_ROOT or REPO_ROOT in root.parents:
            raise ValueError(
                f"proposer.scratch_root {root} is inside the repository; it must be outside so the repository is unreachable"
            )

    def check(self) -> dict[str, Any]:
        """Run ``--version`` and ``--help`` and confirm every required option exists. Raises ProposerCapabilityError naming the first missing one."""
        try:
            v = subprocess.run([self.bin, "--version"], capture_output=True, text=True, timeout=60)
            h = subprocess.run([self.bin, "--help"], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as ex:
            raise ProposerCapabilityError(
                f"cannot run the Claude Code CLI ({self.bin!r}): {ex}"
            ) from ex
        if v.returncode != 0:
            raise ProposerCapabilityError(
                f"`{self.bin} --version` failed: {v.stderr.strip()[:200]}"
            )
        self.cli_version = (
            (v.stdout or v.stderr).strip().splitlines()[0]
            if (v.stdout or v.stderr).strip()
            else None
        )
        self._help_text = (h.stdout or "") + "\n" + (h.stderr or "")
        for capability, spellings in REQUIRED_OPTIONS.items():
            found = next(
                (
                    s
                    for s in spellings
                    if re.search(rf"(?<![\w-]){re.escape(s)}(?![\w-])", self._help_text)
                ),
                None,
            )
            if found is None:
                raise ProposerCapabilityError(
                    f"the Claude Code CLI does not offer the option for '{capability}' (expected one of: {', '.join(spellings)})"
                )
            self.options[capability] = found
        self.options["tool disable"] = next(
            (
                s
                for s in TOOL_DISABLE_OPTIONS
                if re.search(rf"(?<![\w-]){re.escape(s)}(?![\w-])", self._help_text)
            ),
            "",
        )
        if not self.options["tool disable"]:
            raise ProposerCapabilityError(
                f"the Claude Code CLI cannot disable tools for tagger and lesson calls (expected one of: {', '.join(TOOL_DISABLE_OPTIONS)})"
            )
        return {"cli_version": self.cli_version, "options": dict(self.options)}

    # ------------------------------------------------------------------------------------- call

    def call(self, prompt: str, workdir: Path, expect: str, timeout_s: int = 900) -> ProposerResult:
        if not self.options:
            self.check()
        workdir = self._check_workdir(Path(workdir))
        use_tools = expect in EXPECTED_FILES
        model = self.model if use_tools else self.tagger_model
        for name in EXPECTED_FILES.get(expect, ()):
            (workdir / name).unlink(
                missing_ok=True
            )  # a stale file from an earlier call must not count as this call's output
        attempts = 0
        last_err = ""
        started = time.time()
        res: ProposerResult | None = None
        while True:
            attempts += 1
            proc = self._run(prompt, workdir, model, use_tools, timeout_s)
            if proc["timeout"] or (
                proc["rc"] != 0 and not proc["rate_limited"] and not proc["auth"]
            ):
                last_err = (
                    "timed out after %ss" % timeout_s
                    if proc["timeout"]
                    else f"exit code {proc['rc']}: {(proc['stderr'] or proc['stdout'])[:300]}"
                )
                if attempts <= self.retries:
                    self._sleep(float(self.backoff_s[min(attempts - 1, len(self.backoff_s) - 1)]))
                    continue
                res = self._fail(proc, last_err, attempts - 1, started)
                break
            if proc["rate_limited"] or proc["auth"]:
                self._count_failure()
                reset = parse_reset_seconds(proc["stdout"] + proc["stderr"])
                kind = (
                    "authentication error"
                    if proc["auth"] and not proc["rate_limited"]
                    else "rate or usage limit"
                )
                raise ProposerPaused(
                    f"{kind} from the Claude Code CLI: {(proc['stderr'] or proc['stdout'])[:300]}",
                    reset if reset else float(self.backoff_s[-1]),
                )
            res = self._collect(proc, workdir, expect, attempts - 1, started)
            break
        if res.ok:
            self.consecutive_failures = 0
        else:
            self._count_failure()
        return res

    # ------------------------------------------------------------------------------------- internals

    def _count_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.halt_after:
            raise ProposerHalt(
                f"{self.consecutive_failures} consecutive failed proposer calls; halting (state is resumable)"
            )

    def _check_workdir(self, workdir: Path) -> Path:
        real = workdir.resolve()
        root = self.scratch_root.resolve()
        if root != real and root not in real.parents:
            raise ValueError(f"working directory {real} is not under scratch_root {root}")
        if REPO_ROOT == real or REPO_ROOT in real.parents:
            raise ValueError(f"working directory {real} is inside the repository")
        real.mkdir(parents=True, exist_ok=True)
        return real

    def _env(self, cfg_home: Path) -> dict[str, str]:
        env = {k: os.environ[k] for k in (*PASSTHROUGH_ENV, *self.extra_env) if k in os.environ}
        env["CLAUDE_CONFIG_DIR"] = str(cfg_home)
        env["HOME"] = str(cfg_home.parent)
        return env

    def _command(self, model: str, use_tools: bool) -> list[str]:
        cmd = [
            self.bin,
            self.options["non-interactive print mode"],
            "--output-format",
            "json",
            "--model",
            str(model),
        ]
        if self.options["turn limit"] == "--max-turns":
            cmd += ["--max-turns", str(self.max_turns if use_tools else self.tagger_max_turns)]
        else:  # no turn limit in this CLI version: bound the call by money instead
            cmd += [
                self.options["turn limit"],
                str(self.max_budget_usd if use_tools else self.tagger_max_budget_usd),
            ]
        if use_tools:
            tools = list(self.allowed_tools)
            if self.allow_selfcheck:
                tools.append("Bash(python selfcheck.py)")
            cmd += [self.options["tool allow-list"], *tools]
        else:
            opt = self.options["tool disable"]
            cmd += [opt, ""] if opt == "--tools" else [opt, *ALL_TOOLS]
        return cmd

    def _run(
        self, prompt: str, workdir: Path, model: str, use_tools: bool, timeout_s: int
    ) -> dict[str, Any]:
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        home = Path(tempfile.mkdtemp(prefix="claude_home_", dir=self.scratch_root.resolve()))
        cfg_home = home / ".claude"
        cfg_home.mkdir()
        if self.inherit_credentials:
            src = Path.home() / ".claude" / ".credentials.json"
            if src.exists():
                shutil.copy2(src, cfg_home / ".credentials.json")
        out = {
            "stdout": "",
            "stderr": "",
            "rc": -1,
            "timeout": False,
            "duration": 0.0,
            "rate_limited": False,
            "auth": False,
            "cmd": [],
        }
        t0 = time.time()
        try:
            cmd = self._command(model, use_tools)
            out["cmd"] = cmd
            p = subprocess.Popen(
                cmd,
                cwd=workdir,
                env=self._env(cfg_home),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                out["stdout"], out["stderr"] = p.communicate(prompt, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                out["stdout"], out["stderr"] = p.communicate()
                out["timeout"] = True
            out["rc"] = p.returncode if p.returncode is not None else -1
        except OSError as ex:
            out["stderr"] = f"could not start the CLI: {ex}"
        finally:
            shutil.rmtree(home, ignore_errors=True)
        out["duration"] = time.time() - t0
        data = self._json(out["stdout"])
        failed = out["rc"] != 0 or (isinstance(data, dict) and data.get("is_error"))
        if failed and not out["timeout"]:
            text = out["stdout"] + "\n" + out["stderr"]
            out["rate_limited"] = looks_rate_limited(text)
            out["auth"] = (not out["rate_limited"]) and looks_like_auth_error(text)
            if out["rc"] == 0:
                out["rc"] = 1
        return out

    @staticmethod
    def _json(stdout: str) -> Any:
        try:
            return json.loads(stdout)
        except ValueError:
            return None

    def _fail(self, proc: dict[str, Any], err: str, retries: int, started: float) -> ProposerResult:
        return ProposerResult(
            ok=False,
            raw_stdout=proc["stdout"],
            raw_stderr=proc["stderr"],
            cli_version=self.cli_version,
            duration_s=time.time() - started,
            exit_code=proc["rc"],
            error=err,
            retries=retries,
        )

    def _collect(
        self, proc: dict[str, Any], workdir: Path, expect: str, retries: int, started: float
    ) -> ProposerResult:
        data = self._json(proc["stdout"])
        reply = str(data.get("result", "")) if isinstance(data, dict) else proc["stdout"]
        usage: dict[str, Any] = {}
        model_id = None
        if isinstance(data, dict):
            usage = dict(data.get("usage") or {})
            for k in ("total_cost_usd", "num_turns", "duration_ms", "session_id"):
                if k in data:
                    usage[k] = data[k]
            mu = data.get("modelUsage")
            if isinstance(mu, dict) and mu:
                model_id = max(
                    mu,
                    key=lambda k: sum(
                        v for v in (mu[k] or {}).values() if isinstance(v, (int, float))
                    ),
                )
            model_id = model_id or data.get("model")
        role = "main" if expect in EXPECTED_FILES else "tagger"
        changed = self._note_model(
            model_id, self.model if role == "main" else self.tagger_model, role
        )
        res = ProposerResult(
            ok=False,
            raw_stdout=proc["stdout"],
            raw_stderr=proc["stderr"],
            model_id=model_id,
            cli_version=self.cli_version,
            usage=usage,
            duration_s=time.time() - started,
            exit_code=proc["rc"],
            retries=retries,
            proposer_changed=changed,
        )
        files: dict[str, str] = {}
        missing: list[str] = []
        parsed: Any = None
        if expect in EXPECTED_FILES:
            for name in EXPECTED_FILES[expect]:
                f = workdir / name
                if f.exists():
                    files[name] = f.read_text()
            if expect == "candidate" and "model.py" not in files:
                blocks = FENCE.findall(reply)
                if (
                    len(blocks) == 1
                ):  # spec 6.4: exactly one fenced Python block in the reply is accepted as model.py
                    files["model.py"] = blocks[0]
                    (workdir / "model.py").write_text(blocks[0])
            for name in EXPECTED_FILES[expect]:
                if name not in files:
                    missing.append(name)
            if expect == "sketches" and "sketches.json" in files:
                parsed = self._load_json(files["sketches.json"], "sketches.json", missing)
            if expect == "candidate" and "card.json" in files:
                parsed = self._load_json(files["card.json"], "card.json", missing)
        else:
            parsed = extract_json_object(reply)
            if parsed is None:
                missing.append("one JSON object in the reply")
        errs = [m if ("valid" in m or "reply" in m) else f"missing {m}" for m in missing]
        if parsed is not None:
            errs += VALIDATORS[expect](parsed)
        res.files, res.parsed = files, parsed
        res.missing = list(missing) + [
            e for e in errs if not e.startswith("missing ") and e not in missing
        ]
        if errs:
            res.error = "; ".join(errs)
            return res
        res.ok = True
        return res

    @staticmethod
    def _load_json(text: str, name: str, missing: list[str]) -> Any:
        try:
            return json.loads(text)
        except ValueError:
            missing.append(f"{name} (valid JSON)")
            return None

    def _note_model(self, reported: str | None, pinned: str | None, role: str) -> bool:
        """True when the reported model id differs from the pinned one or from the id seen earlier in this role (spec 6.5)."""
        if not reported:
            return False
        seen = self.model_ids_seen[role]
        changed = bool(pinned) and not (
            reported.startswith(str(pinned)) or str(pinned).startswith(reported)
        )
        if seen and reported not in seen:
            changed = True
        if reported not in seen:
            seen.append(reported)
        return changed
