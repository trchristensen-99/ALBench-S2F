"""Phase D scores (val_select). Deliberately separate from ar2.memory.db: nothing the prompt builders import can reach this table (spec 9.6, test T9).

Only the Phase D ingest path writes here and only the report and the selection check read it.
"""

from __future__ import annotations

from pathlib import Path

from ar2.memory.db import connect


def write_final_score(
    db_path: str | Path,
    code_hash: str,
    recipe_hash: str,
    seed: int,
    val_select_r: float,
    val_loop_r: float,
) -> None:
    con = connect(db_path)
    try:
        con.execute(
            "INSERT INTO final_scores(code_hash,recipe_hash,seed,val_select_r,val_loop_r) "
            "SELECT ?,?,?,?,? WHERE NOT EXISTS (SELECT 1 FROM final_scores WHERE code_hash=? AND recipe_hash=? AND seed=?)",
            (code_hash, recipe_hash, seed, val_select_r, val_loop_r, code_hash, recipe_hash, seed),
        )
    finally:
        con.close()


def read_final_scores(db_path: str | Path) -> list[dict]:
    con = connect(db_path)
    try:
        return [
            dict(r)
            for r in con.execute(
                "SELECT * FROM final_scores ORDER BY code_hash, recipe_hash, seed"
            ).fetchall()
        ]
    finally:
        con.close()
