"""Guard: chromosome-based splits must be the ONLY holdout regime.

Hashfrag was retired, but stale references lingered and two files still had *live fallbacks* that
silently substituted hashfrag-era test files when a chr-split file was missing -- mixing holdout
regimes with no error. This guard fails if hashfrag can re-enter the live pipeline.

Scope: the modules that actually run today (data/, albench/, and the current drivers). Legacy
launchers and archived analysis scripts under legacy_hashfrag_era/ and */archive/ are excluded --
they are kept for provenance and are already non-functional (they pass a `use_hashfrag` argument
that no longer exists in the dataset API).
"""

import pathlib
import sys

LIVE = [
    "data",
    "albench",
    "scripts/fm_scaling_driver.py",
    "scripts/build_delta_pairs.py",
    "experiments/exp1_1_scaling.py",
    "experiments/exp1_1_scaling_multitask.py",
]
SKIP = ("__pycache__", "/archive/", "legacy_hashfrag_era", ".venv")


def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    bad = []
    for target in LIVE:
        p = root / target
        files = [p] if p.is_file() else list(p.rglob("*.py")) if p.exists() else []
        for f in files:
            if any(s in str(f) for s in SKIP):
                continue
            try:
                txt = f.read_text(errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(txt.splitlines(), 1):
                low = line.lower()
                if "hashfrag" not in low:
                    continue
                if any(tok in line for tok in ALLOWED_ARTIFACT_TOKENS):
                    continue  # baked-in artifact name, not a split selector
                if low.lstrip().startswith("#") or '"""' in line:
                    continue  # explanatory comment about the removal
                bad.append(f"{f.relative_to(root)}:{i}: {line.strip()[:100]}")
    if bad:
        print("FAIL: hashfrag reference(s) in live pipeline code:")
        print(*bad, sep="\n  ")
        return 1
    print("OK: no hashfrag references in the live pipeline; chr-split is the only holdout regime.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
