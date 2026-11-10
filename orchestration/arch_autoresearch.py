#!/usr/bin/env python3
"""Launcher for the AutoResearch loop at the path that registry cells invoke.

583 registered cells run ``python scratchpad/arch_autoresearch.py ...``; on the cluster that path is a
symlink to this file. The loop itself lives in ``albench/arch_search/arch_autoresearch.py``, because
``menu_retrain.py``, ``harness_ab.py`` and ``albench.ensemble.menu`` import ``train_one`` / ``predict_rc``
from there. Until 2026-11-10 the scratchpad file was a second, hand-patched copy that had drifted from the
library one (menu cells and loop cells were silently running different harnesses). Do not put code here:
a launcher that only imports keeps one module object, one copy, one harness.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from albench.arch_search.arch_autoresearch import main  # noqa: E402  (after the sys.path setup)

if __name__ == "__main__":
    main()
