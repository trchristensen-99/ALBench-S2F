"""The SQLite ledger: one database in write-ahead mode, written only by the orchestrator process (spec 10.1).

Plain methods, no ORM. JSON-valued columns are encoded on the way in and decoded on the way out (``JSON_COLS``). Ledger rows are never deleted; a row's status fields may
advance (``finish_run``, ``update_candidate``, ``update_call``). The derived tables (families, lessons) are replaced wholesale by ``replace_families`` / ``replace_lessons``.

Nothing in this module reads ``final_scores``: that table belongs to ``ar2.memory.final`` so that no code path from the prompt builders can reach it (spec 9.6, test T9).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = "1"
JSON_COLS = {
    "assignment",
    "recipe",
    "diagnostics",
    "tags",
    "card",
    "triage",
    "meta",
    "payload",
    "hparams",
    "detail",
    "body",
    "evidence_run_ids",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _enc(col: str, v: Any) -> Any:
    if col in JSON_COLS and v is not None and not isinstance(v, str):
        return json.dumps(v, sort_keys=True, default=str)
    return v


def _dec(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    for k in list(d):
        if k in JSON_COLS and isinstance(d[k], str):
            try:
                d[k] = json.loads(d[k])
            except ValueError:
                pass
    return d


def connect(path: str | Path) -> sqlite3.Connection:
    """Open the database in WAL mode with row access by name. Used by DB and by ar2.memory.final (which has its own connection helper on purpose)."""
    con = sqlite3.connect(str(path), timeout=60, check_same_thread=False, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


class DB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = connect(self.path)
        self._lock = threading.RLock()
        self.con.executescript(SCHEMA_PATH.read_text())
        if self.meta_get("schema_version") is None:
            self.meta_set("schema_version", SCHEMA_VERSION)

    def close(self) -> None:
        self.con.close()

    # ---- generic ---------------------------------------------------------------------------------------------------------------
    def _insert(self, table: str, row: dict, *, or_ignore: bool = False) -> bool:
        cols = list(row)
        sql = f"INSERT {'OR IGNORE ' if or_ignore else ''}INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
        with self._lock:
            cur = self.con.execute(sql, [_enc(c, row[c]) for c in cols])
            return cur.rowcount > 0

    def _update(self, table: str, key_col: str, key: Any, fields: dict) -> None:
        if not fields:
            return
        sql = f"UPDATE {table} SET {','.join(f'{c}=?' for c in fields)} WHERE {key_col}=?"
        with self._lock:
            self.con.execute(sql, [*(_enc(c, v) for c, v in fields.items()), key])

    def _all(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            return [_dec(r) for r in self.con.execute(sql, args).fetchall()]

    def _one(self, sql: str, args: tuple = ()) -> dict | None:
        with self._lock:
            return _dec(self.con.execute(sql, args).fetchone())

    # ---- meta ------------------------------------------------------------------------------------------------------------------
    def meta_get(self, key: str, default: str | None = None) -> str | None:
        r = self._one("SELECT value FROM meta WHERE key=?", (key,))
        return r["value"] if r else default

    def meta_set(self, key: str, value: Any) -> None:
        with self._lock:
            self.con.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value if isinstance(value, str) else json.dumps(value)),
            )

    # ---- calls -----------------------------------------------------------------------------------------------------------------
    def insert_call(self, row: dict) -> bool:
        row = {"created_at": now(), **row}
        return self._insert("calls", row, or_ignore=True)

    def update_call(self, call_id: str, **fields: Any) -> None:
        self._update("calls", "call_id", call_id, fields)

    def calls(self, mode: str | None = None) -> list[dict]:
        if mode:
            return self._all(
                "SELECT * FROM calls WHERE mode=? ORDER BY created_at, call_id", (mode,)
            )
        return self._all("SELECT * FROM calls ORDER BY created_at, call_id")

    # ---- candidates ------------------------------------------------------------------------------------------------------------
    def insert_candidate(self, row: dict) -> bool:
        """False when the code hash exists already (a duplicate; the caller reports it)."""
        row = {"created_at": now(), "repair_attempts": 0, **row}
        return self._insert("candidates", row, or_ignore=True)

    def update_candidate(self, code_hash: str, **fields: Any) -> None:
        self._update("candidates", "code_hash", code_hash, fields)

    def candidate(self, code_hash: str) -> dict | None:
        return self._one("SELECT * FROM candidates WHERE code_hash=?", (code_hash,))

    def candidates(self, family: str | None = None, gate_status: str | None = None) -> list[dict]:
        q, a = "SELECT * FROM candidates WHERE 1=1", []
        if family:
            q += " AND family=?"
            a.append(family)
        if gate_status:
            q += " AND gate_status=?"
            a.append(gate_status)
        return self._all(q + " ORDER BY created_at, code_hash", tuple(a))

    def all_code_hashes(self) -> set[str]:
        return {r["code_hash"] for r in self._all("SELECT code_hash FROM candidates")}

    # ---- runs ------------------------------------------------------------------------------------------------------------------
    def upsert_run(self, row: dict) -> bool:
        """Insert a run if its key is new; otherwise leave the existing row alone. Returns True when inserted. The UNIQUE constraint makes resubmission idempotent."""
        row = {"status": "submitted", "attempt": 0, "submitted_at": now(), **row}
        return self._insert("runs", row, or_ignore=True)

    def upsert_runs(self, rows: list[dict]) -> list[bool]:
        """Insert several runs in ONE transaction (all or nothing): an action that creates a family of runs (three confirmation seeds, a batch of recipes) can then never
        be half-recorded by a crash. Returns, per row, whether it was newly inserted."""
        out: list[bool] = []
        with self._lock:
            self.con.execute("BEGIN IMMEDIATE")
            try:
                for row in rows:
                    row = {"status": "submitted", "attempt": 0, "submitted_at": now(), **row}
                    cols = list(row)
                    cur = self.con.execute(
                        f"INSERT OR IGNORE INTO runs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                        [_enc(c, row[c]) for c in cols],
                    )
                    out.append(cur.rowcount > 0)
                self.con.execute("COMMIT")
            except Exception:
                self.con.execute("ROLLBACK")
                raise
        return out

    def finish_run(self, run_id: str, **fields: Any) -> None:
        fields.setdefault("finished_at", now())
        self._update("runs", "run_id", run_id, fields)

    def update_run(self, run_id: str, **fields: Any) -> None:
        self._update("runs", "run_id", run_id, fields)

    def run(self, run_id: str) -> dict | None:
        return self._one("SELECT * FROM runs WHERE run_id=?", (run_id,))

    def runs_for(
        self,
        code_hash: str | None = None,
        family: str | None = None,
        purpose: str | None = None,
        status: str | None = None,
        phase: str | None = None,
        pending: bool | None = None,
    ) -> list[dict]:
        q, a = "SELECT * FROM runs WHERE 1=1", []
        for col, v in (
            ("code_hash", code_hash),
            ("family", family),
            ("purpose", purpose),
            ("status", status),
            ("phase", phase),
        ):
            if v is not None:
                q += f" AND {col}=?"
                a.append(v)
        if pending is True:
            q += " AND status IN ('submitted','running')"
        elif pending is False:
            q += " AND status NOT IN ('submitted','running')"
        return self._all(q + " ORDER BY submitted_at, run_id", tuple(a))

    def spend(self, phase: str | None = None, category: str | None = None) -> float:
        """Budget spent in run-equivalents, from finished runs only (cost_equiv = epochs_run / max_epochs, 1 for a full run)."""
        q, a = (
            "SELECT COALESCE(SUM(cost_equiv),0) AS s FROM runs WHERE status NOT IN ('submitted','running')",
            [],
        )
        if phase:
            q += " AND phase=?"
            a.append(phase)
        if category:
            q += " AND category=?"
            a.append(category)
        return float(self._one(q, tuple(a))["s"])

    # ---- sketches --------------------------------------------------------------------------------------------------------------
    def insert_sketches(self, rows: list[dict]) -> int:
        return sum(
            self._insert("sketches", {"created_at": now(), "status": "unused", **r}, or_ignore=True)
            for r in rows
        )

    def sketches(self, status: str | None = None) -> list[dict]:
        if status:
            return self._all(
                "SELECT * FROM sketches WHERE status=? ORDER BY created_at, sketch_id", (status,)
            )
        return self._all("SELECT * FROM sketches ORDER BY created_at, sketch_id")

    def set_sketch_status(self, sketch_id: str, status: str) -> None:
        self._update("sketches", "sketch_id", sketch_id, {"status": status})

    # ---- decisions -------------------------------------------------------------------------------------------------------------
    def insert_decision(
        self,
        step: int,
        mode: str,
        target: Any,
        reason: str,
        archive_hash: str,
        payload: dict | None = None,
    ) -> None:
        tgt = target if isinstance(target, str) else json.dumps(target, sort_keys=True, default=str)
        with self._lock:
            self.con.execute(
                "INSERT INTO decisions(step,mode,target,reason,archive_hash,created_at,payload) VALUES (?,?,?,?,?,?,?)",
                (
                    step,
                    mode,
                    tgt,
                    reason,
                    archive_hash,
                    now(),
                    json.dumps(payload, sort_keys=True, default=str)
                    if payload is not None
                    else None,
                ),
            )

    def decisions(self, mode: str | None = None) -> list[dict]:
        if mode:
            return self._all(
                "SELECT rowid AS rowid_, * FROM decisions WHERE mode=? ORDER BY rowid", (mode,)
            )
        return self._all("SELECT rowid AS rowid_, * FROM decisions ORDER BY rowid")

    def last_step(self) -> int:
        r = self._one("SELECT COALESCE(MAX(step),0) AS s FROM decisions")
        return int(r["s"])

    # ---- derived tables --------------------------------------------------------------------------------------------------------
    def replace_families(self, rows: list[dict]) -> None:
        with self._lock:
            self.con.execute("BEGIN")
            try:
                self.con.execute("DELETE FROM families")
                for r in rows:
                    cols = list(r)
                    self.con.execute(
                        f"INSERT INTO families({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                        [_enc(c, r[c]) for c in cols],
                    )
                self.con.execute("COMMIT")
            except Exception:
                self.con.execute("ROLLBACK")
                raise

    def families(self) -> list[dict]:
        return self._all("SELECT * FROM families ORDER BY family")

    def replace_lessons(self, rows: list[dict]) -> None:
        with self._lock:
            self.con.execute("BEGIN")
            try:
                self.con.execute("DELETE FROM lessons")
                for r in rows:
                    cols = list(r)
                    self.con.execute(
                        f"INSERT INTO lessons({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                        [_enc(c, r[c]) for c in cols],
                    )
                self.con.execute("COMMIT")
            except Exception:
                self.con.execute("ROLLBACK")
                raise
        # created_at_cost orders lessons; nothing else to do

    def lessons(self, scope: str | None = None, status: str | None = "confirmed") -> list[dict]:
        q, a = "SELECT * FROM lessons WHERE 1=1", []
        if scope:
            q += " AND scope=?"
            a.append(scope)
        if status:
            q += " AND status=?"
            a.append(status)
        return self._all(q + " ORDER BY created_at_cost DESC, lesson_id", tuple(a))

    # ---- lesson wording ledger -------------------------------------------------------------------------------------------------
    def lesson_text(self, evidence_key: str) -> str | None:
        r = self._one("SELECT statement FROM lesson_text WHERE evidence_key=?", (evidence_key,))
        return r["statement"] if r else None

    def put_lesson_text(
        self, evidence_key: str, statement: str, call_id: str | None = None
    ) -> None:
        self._insert(
            "lesson_text",
            {
                "evidence_key": evidence_key,
                "statement": statement,
                "call_id": call_id,
                "created_at": now(),
            },
            or_ignore=True,
        )

    # ---- state hash ------------------------------------------------------------------------------------------------------------
    def archive_hash(self) -> str:
        """Hash of the archive state a decision was based on: the families table and the count and status of every ledger table."""
        parts = [json.dumps(self.families(), sort_keys=True, default=str)]
        for t, col in (("runs", "status"), ("candidates", "gate_status"), ("sketches", "status")):
            rows = self._all(
                f"SELECT {col} AS s, COUNT(*) AS n FROM {t} GROUP BY {col} ORDER BY {col}"
            )
            parts.append(json.dumps(rows, sort_keys=True, default=str))
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
