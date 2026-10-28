"""Configuration: YAML defaults, dotted overrides, a frozen hash, and the live `balance.*` overrides.

A workspace freezes its resolved config in ``ws/config.yaml`` at ``ar2 init`` and records the hash in the meta table. Only keys under ``balance.`` may change during a
run; each change is appended to ``ws/meta/balance_overrides.json`` (and written to the decisions table by the caller), so the frozen config and its hash never move.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent / "config" / "default.yaml"
LIVE_PREFIX = "balance."


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, new: dict) -> dict:
    """``new`` over ``base`` recursively (dicts merge, everything else replaces); returns a new dict."""
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get_dotted(cfg: dict, key: str, default: Any = ...) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            if default is ...:
                raise KeyError(key)
            return default
        cur = cur[part]
    return cur


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    cur = cfg
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
        if not isinstance(cur, dict):
            raise ConfigError(f"{key}: {part} is not a section")
    cur[parts[-1]] = value


def parse_value(text: str) -> Any:
    """CLI values: YAML scalars/lists (``0.5``, ``true``, ``[1,2]``, ``null``), falling back to the raw string."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    profile: str | None = None,
) -> dict:
    """Defaults, then an optional YAML file on top, then the ``deployment`` profile if asked for, then dotted overrides. The ``deployment`` section is dropped."""
    cfg = yaml.safe_load(DEFAULT_PATH.read_text())
    if path:
        cfg = deep_merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    deploy = cfg.pop("deployment", {}) or {}
    if profile:
        if profile != "deployment":
            raise ConfigError(f"unknown profile {profile!r} (only 'deployment')")
        cfg = deep_merge(cfg, deploy)
    for k, v in (overrides or {}).items():
        set_dotted(cfg, k, v)
    return cfg


def config_hash(cfg: dict) -> str:
    """Hash of the resolved config; the live `balance` section is part of it at freeze time (the overrides are kept elsewhere)."""
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def save_config(ws: Path, cfg: dict) -> str:
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    return config_hash(cfg)


def load_frozen(ws: Path) -> dict:
    return yaml.safe_load((ws / "config.yaml").read_text())


def _overrides_path(ws: Path) -> Path:
    return ws / "meta" / "balance_overrides.json"


def read_balance_overrides(ws: Path) -> list[dict]:
    p = _overrides_path(ws)
    return json.loads(p.read_text()) if p.exists() else []


def set_live(ws: Path, key: str, value: Any, step: int | None = None) -> dict:
    """Record a live change of a ``balance.*`` key. Anything else is refused (T16). Returns the change record."""
    if not key.startswith(LIVE_PREFIX):
        raise ConfigError(f"only keys under 'balance.' may change during a run, not {key!r}")
    frozen = load_frozen(ws)
    try:
        old = get_dotted(frozen, key)
    except KeyError as e:
        raise ConfigError(f"unknown balance key {key!r}") from e
    rec = {
        "key": key,
        "value": value,
        "old": old,
        "step": step,
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "applied": False,
    }
    p = _overrides_path(ws)
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = read_balance_overrides(ws)
    cur.append(rec)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, indent=1))
    tmp.replace(p)
    return rec


def effective_config(ws: Path) -> tuple[dict, list[dict]]:
    """Frozen config with every recorded balance override applied in order. Also returns the overrides not yet logged to decisions (``applied`` false)."""
    cfg = load_frozen(ws)
    pending = []
    for rec in read_balance_overrides(ws):
        set_dotted(cfg, rec["key"], rec["value"])
        if not rec.get("applied"):
            pending.append(rec)
    return cfg, pending


def mark_overrides_applied(ws: Path) -> None:
    p = _overrides_path(ws)
    if not p.exists():
        return
    recs = json.loads(p.read_text())
    for r in recs:
        r["applied"] = True
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(recs, indent=1))
    tmp.replace(p)
