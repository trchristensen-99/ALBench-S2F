"""Gatekeeper (spec section 7): no candidate reaches a GPU training run until it passes every check below, in order.

Static checks run in this process on the source text. Every dynamic check runs in a fresh sandboxed subprocess (python -I, stripped environment, CPU-time
limit, wall-clock timeout, sockets disabled before candidate code is imported); that subprocess is the only place candidate code is imported here.
Inputs and batch sizes are drawn fresh at every call, including sizes no prompt names, so a candidate cannot pass by special-casing the self-check.

Codes, in order: files, unsafe, duplicate, build, params, shape, constant, coverage, batch_leak, determinism, learn, amp, speed; ``near_duplicate`` (14) runs
after tagging, from the orchestrator, through :func:`near_duplicate`.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import resource
import secrets
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

ALLOWED_IMPORTS = {
    "torch",
    "math",
    "typing",
    "functools",
    "collections",
    "dataclasses",
    "__future__",
}
FORBIDDEN_CALLS = {
    "open",
    "eval",
    "exec",
    "__import__",
    "compile",
    "input",
    "breakpoint",
    "exit",
    "quit",
}
FORBIDDEN_NAMES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib", "builtins", "ctypes", "pickle",
    "urllib", "requests", "http", "tempfile",
}  # fmt: skip
FORBIDDEN_DUNDERS = {
    "__subclasses__", "__globals__", "__builtins__", "__import__", "__loader__", "__spec__", "__code__",
    "__closure__", "__mro__", "__bases__", "__getattribute__", "__reduce__", "__reduce_ex__",
}  # fmt: skip
FORBIDDEN_ATTR_CALLS = {
    "load_state_dict",
    "from_pretrained",
    "load_from_checkpoint",
    "load_url",
    "load_state_dict_from_url",
}
FORBIDDEN_TORCH_PATHS = (
    "torch.load",
    "torch.save",
    "torch.jit.load",
    "torch.jit.save",
    "torch.hub",
    "torch.serialization",
    "torch.utils.cpp_extension",
    "torch.utils.data",
)
CARD_REQUIRED = {"family_claimed": str, "summary": str, "suggested_recipe": dict}
CODES = (
    "files",
    "unsafe",
    "duplicate",
    "build",
    "params",
    "shape",
    "constant",
    "coverage",
    "batch_leak",
    "determinism",
    "learn",
    "amp",
    "speed",
    "near_duplicate",
)
REPAIRABLE = {
    "files",
    "build",
    "params",
    "shape",
    "constant",
    "coverage",
    "batch_leak",
    "determinism",
    "learn",
    "speed",
}


@dataclass
class GateResult:
    ok: bool
    code: str | None = None
    action: str = "pass"  # pass | repair | reject | amp_off
    detail: str = ""
    traceback: str = ""
    measured: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------------------------------------- source hashing


class _StripDocstrings(ast.NodeTransformer):
    def _strip(self, node):
        self.generic_visit(node)
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = visit_Module = _strip  # noqa: N815


def normalise_source(src: str) -> str:
    """The source with comments, docstrings and formatting removed (``ast.unparse`` of the parsed tree)."""
    return ast.unparse(_StripDocstrings().visit(ast.parse(src)))


def code_hash(src: str) -> str:
    """sha1[:16] of the normalised source; falls back to the raw text when it does not parse."""
    try:
        norm = normalise_source(src)
    except SyntaxError:
        norm = src
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------------------- static checks


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def static_check(source: str) -> tuple[bool, str | None, str]:
    """Syntax-tree scan (check 2). Returns (ok, code, detail); a syntax error is a repairable ``build`` failure, anything unsafe is ``unsafe`` (reject)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, "build", f"syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return (
                        False,
                        "unsafe",
                        f"import of {a.name!r} is not allowed (allow-list: {sorted(ALLOWED_IMPORTS - {'__future__'})})",
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_IMPORTS:
                return False, "unsafe", f"import from {node.module!r} is not allowed"
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in FORBIDDEN_CALLS:
                return False, "unsafe", f"call to {f.id}() is not allowed"
            if isinstance(f, ast.Attribute) and f.attr in FORBIDDEN_ATTR_CALLS:
                return False, "unsafe", f"call to .{f.attr}() (weight loading) is not allowed"
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_CALLS:
            return False, "unsafe", f"use of the name {node.id!r} is not allowed"
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_NAMES or node.attr in FORBIDDEN_DUNDERS:
                return False, "unsafe", f"access to attribute {node.attr!r} is not allowed"
            d = _dotted(node)
            if any(d == p or d.startswith(p + ".") for p in FORBIDDEN_TORCH_PATHS):
                return (
                    False,
                    "unsafe",
                    f"use of {d} is not allowed (no file access or weight loading)",
                )
    if not any(isinstance(n, ast.FunctionDef) and n.name == "build_model" for n in tree.body):
        return False, "build", "model.py defines no module-level build_model(out_dim, **hparams)"
    return True, None, ""


def check_card(card_path: Path) -> tuple[bool, str]:
    """Check 1 (second half): card.json is valid JSON and has the required fields with the right types."""
    try:
        card = json.loads(card_path.read_text())
    except Exception as e:  # noqa: BLE001
        return False, f"card.json is not valid JSON: {e}"
    if not isinstance(card, dict):
        return False, "card.json must be a JSON object"
    for k, t in CARD_REQUIRED.items():
        if k not in card or not isinstance(card[k], t):
            return False, f"card.json field {k!r} is missing or not a {t.__name__}"
    for k in ("amp_safe", "length_agnostic"):
        if k in card and not isinstance(card[k], bool):
            return False, f"card.json field {k!r} must be true or false"
    return True, ""


# ---------------------------------------------------------------------------------------------------------------------------- sandbox runner

RUNNER = r"""
import sys, json, time, random, traceback, importlib.util
args = json.loads(sys.stdin.read())
import torch, torch.nn as nn
import socket
def _deny(*a, **k):
    raise RuntimeError("network access is disabled in the gate sandbox")
class _NoNet(socket.socket):
    connect = connect_ex = bind = sendto = _deny
socket.socket = _NoNet; socket.create_connection = _deny; socket.getaddrinfo = _deny; socket.gethostbyname = _deny
rng = random.Random(args["rng_seed"])
cfg = args["cfg"]
SEQ = 200
OUT = {"ok": True, "code": None, "detail": "", "traceback": "", "measured": {}}

def fail(code, detail, tb=""):
    OUT.update(ok=False, code=code, detail=detail, traceback=tb[-3000:])
    print("AR2_GATE_RESULT " + json.dumps(OUT)); sys.exit(0)

def rand_x(b):
    idx = torch.randint(0, 4, (b, SEQ))
    return torch.nn.functional.one_hot(idx, 4).permute(0, 2, 1).float()

def sizes(base, extra=2, lo=3, hi=100):
    out = list(base)
    while len(out) < len(base) + extra:
        b = rng.randint(lo, hi)
        if b not in out:
            out.append(b)
    return out

try:
    spec = importlib.util.spec_from_file_location("candidate_model", args["model_path"])
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
except Exception:
    fail("build", "model.py could not be imported", traceback.format_exc())
hp = dict(getattr(mod, "HPARAMS", {}) or {}); hp.update(args.get("hparams") or {})
tun = getattr(mod, "TUNABLE", None)
if tun is not None:
    ok_t = isinstance(tun, dict) and len(tun) <= 4 and all(isinstance(v, (list, tuple)) and len(v) == 2 and v[0] < v[1] for v in tun.values()) and all(k in getattr(mod, "HPARAMS", {}) for k in tun)
    if not ok_t:
        fail("build", "TUNABLE must be a dict of at most 4 keys, each [low, high] with low < high and each key present in HPARAMS")

def build(out_dim, seed=0):
    torch.manual_seed(seed)
    m = mod.build_model(out_dim, **hp)
    if not isinstance(m, nn.Module):
        raise TypeError("build_model must return an nn.Module")
    return m

# 4 build
models = {}
for od in (1, 2, 32):
    try:
        models[od] = build(od)
    except Exception:
        fail("build", f"build_model failed for out_dim={od}", traceback.format_exc())
m1 = models[1]
# 5 params
n_params = sum(p.numel() for p in m1.parameters() if p.requires_grad)
OUT["measured"]["n_params"] = n_params
if n_params < cfg["min_params"] or n_params > cfg["max_params"]:
    fail("params", f"{n_params:,} trainable parameters, allowed {cfg['min_params']:,} to {cfg['max_params']:,}")
# 6 shape
try:
    for od, m in ((1, models[1]), (2, models[2])):
        m.train()
        for b in ([2, 64] if od == 1 else [2]) + ([rng.randint(3, 40)] if od == 1 else []):
            o = m(rand_x(b))
            if tuple(o.shape) != (b, od):
                fail("shape", f"train mode, out_dim={od}, batch {b}: output shape {tuple(o.shape)}, expected {(b, od)}")
            if not torch.isfinite(o).all():
                fail("shape", f"train mode, out_dim={od}, batch {b}: output is not finite")
        m.eval()
        with torch.no_grad():
            for b in sizes([1, 2, 64] if od == 1 else [1, 3], extra=2):
                o = m(rand_x(b))
                if tuple(o.shape) != (b, od):
                    fail("shape", f"eval mode, out_dim={od}, batch {b}: output shape {tuple(o.shape)}, expected {(b, od)}")
                if not torch.isfinite(o).all():
                    fail("shape", f"eval mode, out_dim={od}, batch {b}: output is not finite")
except SystemExit:
    raise
except Exception:
    fail("shape", "forward pass failed", traceback.format_exc())
m1.eval()
# 7 constant
with torch.no_grad():
    o = m1(rand_x(64))[:, 0]
if float(o.std()) <= 1e-6:
    fail("constant", f"outputs do not differ across 64 random inputs (std {float(o.std()):.2e})")
# 8 coverage
try:
    x = rand_x(16).requires_grad_(True)
    g, = torch.autograd.grad(m1(x)[:, 0].sum(), x)
    cov = float((g.abs().sum(1) > 1e-12).float().mean())
except Exception:
    fail("coverage", "gradient with respect to the input could not be computed", traceback.format_exc())
OUT["measured"]["coverage"] = cov
if cov < cfg["coverage_min"]:
    fail("coverage", f"the output depends on only {cov:.0%} of the 200 input positions (required {cfg['coverage_min']:.0%})")
# 9 batch leak
with torch.no_grad():
    b = rng.randint(5, 40)
    xa = rand_x(b); xb = xa.clone(); xb[1:] = rand_x(b - 1)
    oa, ob = m1(xa)[0], m1(xb)[0]
    diff = float((oa - ob).abs().max()); scale = max(1.0, float(oa.abs().max()))
if diff > cfg["batch_leak_tol"] * scale:
    fail("batch_leak", f"in eval mode the output for sequence 0 changes by {diff:.2e} when the rest of the batch is replaced")
# 10 determinism
with torch.no_grad():
    xs = rand_x(8); sd = rng.randint(0, 10000)
    oa = build(1, sd).eval()(xs); ob = build(1, sd).eval()(xs)
if not torch.allclose(oa, ob, atol=1e-6, rtol=1e-5):
    fail("determinism", f"two builds with the same seed differ by {float((oa - ob).abs().max()):.2e}")
# 11 learn
try:
    torch.manual_seed(rng.randint(0, 10000))
    m = build(1, 1).train(); xl = rand_x(64); yl = torch.randn(64)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    losses = []
    for _ in range(cfg["learn_steps"]):
        opt.zero_grad(); l = ((m(xl)[:, 0] - yl) ** 2).mean(); l.backward(); opt.step(); losses.append(float(l))
    first, last = sum(losses[:3]) / 3, min(sum(losses[-5:]) / 5, min(losses))
    if not (last <= (1 - cfg["learn_min_drop"]) * first):
        fail("learn", f"{cfg['learn_steps']} Adam steps reduced the MSE on 64 random-label examples from {first:.3f} to {last:.3f}; at least {cfg['learn_min_drop']:.0%} is required")
except SystemExit:
    raise
except Exception:
    fail("learn", "the learn check could not run", traceback.format_exc())
# 12 amp (not a failure)
amp_ok = True
try:
    m = build(1, 2).train(); xa = rand_x(16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        o = m(xa)
        l = o.float().pow(2).mean()
    l.backward()
    amp_ok = bool(torch.isfinite(o.float()).all()) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters())
except Exception:
    amp_ok = False
OUT["measured"]["amp_safe"] = amp_ok
# 13 speed
sp = args.get("speed")
if sp:
    dev = "cuda" if (sp.get("device") == "cuda" and torch.cuda.is_available()) else "cpu"
    try:
        m = build(1, 3).to(dev).train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        bs = int(sp.get("batch_size") or (512 if dev == "cuda" else 32))
        x = rand_x(bs).to(dev); y = torch.randn(bs, device=dev)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        def step():
            opt.zero_grad(); l = ((m(x)[:, 0] - y) ** 2).mean(); l.backward(); opt.step()
        for _ in range(3): step()
        if dev == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20): step()
        if dev == "cuda": torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        thr = 20 * bs / dt
        peak = (torch.cuda.max_memory_allocated() / 1e9) if dev == "cuda" else 0.0
    except Exception:
        fail("speed", "the throughput probe could not run", traceback.format_exc())
    OUT["measured"].update(throughput_seq_per_s=thr, throughput_h100_equiv=thr * float(sp.get("speed_factor", 1.0)), peak_mem_gb=peak, probe_device=dev)
    cap = sp.get("time_cap_s")
    if cap:
        proj = sp["D"] * sp["max_epochs"] / thr
        OUT["measured"]["projected_full_run_s"] = proj
        if proj > cap:
            fail("speed", f"projected full run {proj:.0f}s ({thr:.0f} sequences/s, {sp['max_epochs']} epochs of {sp['D']}) exceeds the time cap {cap:.0f}s")
    if peak > cfg["max_mem_gb"]:
        fail("speed", f"peak memory {peak:.1f} GB exceeds the limit {cfg['max_mem_gb']} GB")
print("AR2_GATE_RESULT " + json.dumps(OUT))
"""


def _limits(cpu_s: int, as_gb: float | None):
    def fn():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        if as_gb:
            resource.setrlimit(resource.RLIMIT_AS, (int(as_gb * 2**30), int(as_gb * 2**30)))

    return fn


def _sandbox_env(tmp: str, want_cuda: bool) -> dict:
    keep = ["PATH", "LD_LIBRARY_PATH", "LANG", "LC_ALL", "TMPDIR"] + (
        ["CUDA_VISIBLE_DEVICES", "CUDA_HOME"] if want_cuda else []
    )
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update(
        HOME=tmp,
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        PYTHONHASHSEED="0",
        NO_PROXY="*",
        no_proxy="*",
    )
    return env


def _gate_cfg(cfg: dict) -> dict:
    g = (cfg or {}).get("gate", {})
    return {
        "min_params": int(g.get("min_params", 10_000)),
        "max_params": int(g.get("max_params", 40_000_000)),
        "coverage_min": float(g.get("coverage_min", 0.90)),
        "batch_leak_tol": float(g.get("batch_leak_tol", 1.0e-5)),
        "learn_steps": int(g.get("learn_steps", 50)),
        "learn_min_drop": float(g.get("learn_min_drop", 0.2)),
        "max_mem_gb": float(g.get("max_mem_gb", 9.0)),
    }


def run_gate(
    candidate_dir: str | Path,
    cfg: dict,
    *,
    hparams: dict | None = None,
    existing_hashes: set[str] | None = None,
    device: str = "cpu",
    speed_probe: bool = False,
    speed_factor: float = 1.0,
    time_cap_s: float | None = None,
) -> GateResult:
    """Run checks 1 to 13 in order and return the first failure, or a pass.

    ``device`` is the device of the throughput probe only (all other dynamic checks run on the CPU of the sandbox). ``cfg`` is the resolved config (uses
    ``gate.*``, ``evaluator.max_epochs`` and ``D``). ``time_cap_s`` is the time cap of a full run on the reference GPU; the projection is compared with
    ``time_cap_s * speed_factor``, the cap for the device the probe ran on.
    """
    cdir = Path(candidate_dir)
    model_py, card = cdir / "model.py", cdir / "card.json"
    measured: dict = {}
    # 1 files
    if not model_py.is_file() or not model_py.read_text().strip():
        return GateResult(
            False, "files", "repair", "model.py is missing or empty", measured=measured
        )
    if not card.is_file():
        return GateResult(False, "files", "repair", "card.json is missing", measured=measured)
    ok, why = check_card(card)
    if not ok:
        return GateResult(False, "files", "repair", why, measured=measured)
    src = model_py.read_text()
    h = code_hash(src)
    measured["code_hash"] = h
    # 2 unsafe (a syntax error or missing build_model is reported as a repairable build failure)
    ok, code, detail = static_check(src)
    if not ok:
        return GateResult(
            False, code, "reject" if code == "unsafe" else "repair", detail, measured=measured
        )
    # 3 duplicate
    if existing_hashes and h in existing_hashes:
        return GateResult(
            False,
            "duplicate",
            "reject",
            f"the normalised code matches existing candidate {h}",
            measured=measured,
        )
    # 4 to 13 in a sandboxed subprocess
    gcfg = _gate_cfg(cfg)
    args = {
        "model_path": str(model_py.resolve()),
        "hparams": hparams or {},
        "rng_seed": secrets.randbelow(2**31),
        "cfg": gcfg,
        "speed": None,
    }
    if speed_probe:
        args["speed"] = {
            "device": device,
            "D": int((cfg or {}).get("D") or (cfg or {}).get("data", {}).get("D") or 100_000),
            "max_epochs": int((cfg or {}).get("evaluator", {}).get("max_epochs", 40)),
            "time_cap_s": (time_cap_s * speed_factor) if time_cap_s else None,
            "speed_factor": speed_factor,
            "batch_size": (cfg or {}).get("evaluator", {}).get("probe_batch_size"),
        }
    timeout = int((cfg or {}).get("gate", {}).get("sandbox_timeout_s", 300))
    with tempfile.TemporaryDirectory(prefix="ar2_gate_") as tmp:
        try:
            p = subprocess.run(
                [sys.executable, "-I", "-c", RUNNER],
                input=json.dumps(args),
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
                env=_sandbox_env(tmp, device == "cuda"),
                preexec_fn=_limits(timeout + 30, (cfg or {}).get("gate", {}).get("sandbox_as_gb")),
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            return GateResult(
                False,
                "speed",
                "repair",
                f"the dynamic checks did not finish within {timeout}s",
                measured=measured,
            )
    line = next(
        (ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("AR2_GATE_RESULT ")), None
    )
    if line is None:
        return GateResult(
            False,
            "build",
            "repair",
            f"the sandbox crashed (exit {p.returncode})",
            p.stderr[-3000:],
            measured,
        )
    res = json.loads(line[len("AR2_GATE_RESULT ") :])
    measured.update(res.get("measured", {}))
    if not res["ok"]:
        code = res["code"]
        return GateResult(
            False,
            code,
            "repair" if code in REPAIRABLE else "reject",
            res["detail"],
            res.get("traceback", ""),
            measured,
        )
    if (
        measured.get("amp_safe") is False
    ):  # check 12 is not a failure: the candidate trains with amp off
        return GateResult(
            True,
            "amp",
            "amp_off",
            "forward or backward under mixed precision is not finite; train with amp: false",
            "",
            measured,
        )
    return GateResult(True, None, "pass", "", "", measured)


def write_gate_json(candidate_dir: str | Path, result: GateResult) -> Path:
    """Write ``gate.json`` next to the candidate (atomic)."""
    path = Path(candidate_dir) / "gate.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(result), indent=1))
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------------------------------------------------------- check 14


def _close(a, b, tol: float = 0.10) -> bool:
    if (
        isinstance(a, bool)
        or isinstance(b, bool)
        or not isinstance(a, (int, float))
        or not isinstance(b, (int, float))
    ):
        return a == b
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-12)


def near_duplicate(tags: dict, hparams: dict, others: list[dict], tol: float = 0.10) -> str | None:
    """Check 14. ``others`` are dicts with ``code_hash``, ``tags`` and ``hparams``; returns the code_hash of an existing candidate in the same family whose tags are
    identical (n_params and the justification are ignored) and whose every HPARAMS value is within ``tol``, else None."""
    ignore = {"n_params", "justification", "family_claimed"}
    key = {k: v for k, v in (tags or {}).items() if k not in ignore}
    for o in others:
        ot = o.get("tags") or {}
        if ot.get("family") != (tags or {}).get("family"):
            continue
        if {k: v for k, v in ot.items() if k not in ignore} != key:
            continue
        oh = o.get("hparams") or {}
        if set(oh) == set(hparams or {}) and all(
            _close(oh[k], (hparams or {})[k], tol) for k in oh
        ):
            return o.get("code_hash")
    return None
