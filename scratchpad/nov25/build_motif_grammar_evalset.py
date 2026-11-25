"""A motif eval set that varies the GRAMMAR, not just the vocabulary.

The stored `motif_planted` eval set is one configuration: 331-motif vocabulary, PWM-sampled
instances, 1-7 sites, positions uniform at random, always on the same background family. A
student can score well on it by learning "this background plus strong sites means active",
which is not the thing the 4M library has to teach. This set crosses the factors that a
designed corpus actually varies -- background, vocabulary, site count, arrangement, spacing,
orientation, instance quality -- so the endpoint asks whether a corpus taught the model
anything about motif SYNTAX rather than motif presence.

Every factor level is stored per sequence, so the endpoint can be broken down afterwards:
"which arrangement does a genomic-only corpus fail on" is then a groupby, not a new run.

Backgrounds come from the chr7/13 test battery, never from training sequence: the three
genomic-derived background families are the held-out genomic test sequences themselves and
their mono- and dinucleotide shuffles.

    python scratchpad/nov25/build_motif_grammar_evalset.py --n 24000 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("ALBENCH_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO))

BATTERY = REPO / "data/k562/test_sets_ag_s2_chrsplit"
ACGT = np.array(list("ACGT"))

BACKGROUNDS = ("random_gc", "mono_shuffle", "dinuc_shuffle", "genomic_test")
VOCABS = ("broad", "mid", "narrow")
N_SITES = (1, 2, 3, 5, 8, 12, 16)
ARRANGE = ("random", "periodic", "clustered", "dimer_pairs", "homotypic")
GAPS = (3, 6, 10, 16, 24)
ORIENT = ("forward", "revcomp", "mixed")
INSTANCE = ("consensus", "pwm_sample", "pwm_sample_mut10")
L = 200


def _dinuc_shuffle(s: str, rng: np.random.Generator) -> str:
    """Altschul-Erikson dinucleotide shuffle via an Eulerian walk on the 4-node graph.

    A mononucleotide shuffle destroys the dinucleotide composition, and CpG content alone
    moves predicted activity a long way in this assay -- so a shuffled background that did
    not preserve it would differ from its source in a way that has nothing to do with the
    motifs planted on top.
    """
    if len(s) < 3:
        return s
    edges: dict[str, list[str]] = {c: [] for c in "ACGTN"}
    for a, b in zip(s[:-1], s[1:], strict=True):
        edges.setdefault(a, []).append(b)
    last = s[-1]
    for _ in range(12):
        for k in edges:
            rng.shuffle(edges[k])
        # A valid Eulerian walk needs the last edge out of every node to lead to `last`.
        ok = True
        for k, v in edges.items():
            if not v or k == last:
                continue
            seen, cur = set(), k
            while cur != last and cur not in seen and edges.get(cur):
                seen.add(cur)
                cur = edges[cur][-1]
            if cur != last:
                ok = False
                break
        if ok:
            break
    else:
        return "".join(rng.permutation(list(s)))
    nxt = {k: list(v) for k, v in edges.items()}
    out, cur = [s[0]], s[0]
    for _ in range(len(s) - 1):
        if not nxt.get(cur):
            return "".join(rng.permutation(list(s)))
        c = nxt[cur].pop(0)
        out.append(c)
        cur = c
    return "".join(out)


def make_backgrounds(n: int, rng: np.random.Generator) -> tuple[list[str], list[str]]:
    gz = np.load(BATTERY / "genomic_oracle.npz", allow_pickle=True)
    gen = [str(x)[:L].upper().ljust(L, "A") for x in gz["sequences"]]
    print(f"background source: {len(gen):,} chr7/13 genomic test sequences", flush=True)

    per = n // len(BACKGROUNDS)
    seqs: list[str] = []
    kind: list[str] = []
    pick = rng.choice(len(gen), size=n, replace=len(gen) < n)

    for i in range(per):  # random_gc
        gc = rng.uniform(0.25, 0.75)
        p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
        seqs.append("".join(ACGT[rng.choice(4, size=L, p=p)]))
        kind.append("random_gc")
    for i in range(per):  # mononucleotide shuffle: GC kept, dinucleotides destroyed
        seqs.append("".join(rng.permutation(list(gen[pick[per + i]]))))
        kind.append("mono_shuffle")
    for i in range(per):  # dinucleotide shuffle: CpG kept too
        seqs.append(_dinuc_shuffle(gen[pick[2 * per + i]], rng))
        kind.append("dinuc_shuffle")
    while len(seqs) < n:  # real held-out genomic background
        seqs.append(gen[pick[len(seqs)]])
        kind.append("genomic_test")
    return seqs, kind


def build_vocabs(rng: np.random.Generator):
    from albench.motifs import vocabulary as V

    full = V.build(human_only=False, cluster_at=0.90, min_ic=6.0)
    full = [m for m in full if m.length <= 20]
    order = rng.permutation(len(full))
    out = {
        "broad": full,
        "mid": [full[i] for i in order[:80]],
        # Highest-information-content motifs: the easiest vocabulary, and the one a model
        # can get right from presence alone.
        "narrow": sorted(full, key=lambda m: -m.info_content)[:20],
    }
    for k, v in out.items():
        print(f"vocab {k:<7} {len(v):>4} motifs, mean length {np.mean([m.length for m in v]):.1f}")
    return out


def _positions(arrangement: str, k: int, widths: list[int], gap: int, rng) -> list[int] | None:
    """Start positions for k sites under one arrangement, or None if they do not fit."""
    total = sum(widths)
    if total + (k - 1) >= L:
        return None
    if arrangement == "periodic":
        step = (L - total) // k
        if step <= 0:
            return None
        pos, cur = [], rng.integers(0, max(1, step))
        for w in widths:
            pos.append(int(cur))
            cur += w + step
            if cur > L:
                return None
        return pos if pos[-1] + widths[-1] <= L else None
    if arrangement in ("clustered", "dimer_pairs", "homotypic"):
        # clustered/homotypic: one tight block; dimer_pairs: pairs separated by `gap`,
        # pairs themselves scattered.
        if arrangement == "dimer_pairs":
            pos: list[int] = []
            i = 0
            while i < k:
                pair = widths[i : i + 2]
                span = sum(pair) + gap * (len(pair) - 1)
                if span >= L:
                    return None
                start = int(rng.integers(0, L - span))
                if any(abs(start - p) < 30 for p in pos):
                    start = int(rng.integers(0, L - span))
                cur = start
                for w in pair:
                    pos.append(cur)
                    cur += w + gap
                i += 2
            return pos if max(p + w for p, w in zip(pos, widths, strict=True)) <= L else None
        span = total + gap * (k - 1)
        if span >= L:
            return None
        cur = int(rng.integers(0, L - span))
        pos = []
        for w in widths:
            pos.append(cur)
            cur += w + gap
        return pos
    # random, non-overlapping
    for _ in range(40):
        starts = sorted(int(x) for x in rng.integers(0, L - max(widths), size=k))
        ok = (
            all(starts[i] + widths[i] + 1 <= starts[i + 1] for i in range(k - 1))
            and starts[-1] + widths[-1] <= L
        )
        if ok:
            return starts
    return None


def _rc(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24_000)
    ap.add_argument("--out", default="data/k562/test_sets_ag_s2_chrsplit")
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument("--no-label", action="store_true")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    bg, bg_kind = make_backgrounds(a.n, rng)
    vocabs = build_vocabs(rng)

    seqs: list[str] = []
    rec: dict[str, list] = {
        k: []
        for k in (
            "background",
            "vocab",
            "n_sites",
            "arrangement",
            "gap",
            "orientation",
            "instance",
            "motifs",
            "positions",
        )
    }
    n_fail = 0
    for i in range(a.n):
        for _attempt in range(12):
            vk = VOCABS[rng.integers(len(VOCABS))]
            voc = vocabs[vk]
            k = int(N_SITES[rng.integers(len(N_SITES))])
            arr = ARRANGE[rng.integers(len(ARRANGE))]
            gap = int(GAPS[rng.integers(len(GAPS))])
            ori = ORIENT[rng.integers(len(ORIENT))]
            inst = INSTANCE[rng.integers(len(INSTANCE))]
            if arr == "homotypic":
                ms = [voc[int(rng.integers(len(voc)))]] * k
            else:
                ms = [voc[int(j)] for j in rng.integers(0, len(voc), size=k)]
            widths = [m.length for m in ms]
            pos = _positions(arr, k, widths, gap, rng)
            if pos is not None:
                break
        else:
            n_fail += 1
            seqs.append(bg[i])
            for key, v in (
                ("background", bg_kind[i]),
                ("vocab", "none"),
                ("n_sites", 0),
                ("arrangement", "none"),
                ("gap", 0),
                ("orientation", "none"),
                ("instance", "none"),
                ("motifs", ""),
                ("positions", ""),
            ):
                rec[key].append(v)
            continue

        s = list(bg[i])
        for m, p in zip(ms, pos, strict=True):
            if inst == "consensus":
                site = m.consensus
            else:
                site = m.sample(rng, 1)[0]
                if inst == "pwm_sample_mut10":
                    site = "".join(
                        (ACGT[rng.integers(4)] if rng.random() < 0.10 else c) for c in site
                    )
            flip = ori == "revcomp" or (ori == "mixed" and rng.random() < 0.5)
            if flip:
                site = _rc(site)
            s[p : p + len(site)] = list(site)
        seqs.append("".join(s[:L]).ljust(L, "A"))
        for key, v in (
            ("background", bg_kind[i]),
            ("vocab", vk),
            ("n_sites", k),
            ("arrangement", arr),
            ("gap", gap if arr in ("clustered", "dimer_pairs", "homotypic") else 0),
            ("orientation", ori),
            ("instance", inst),
            ("motifs", ",".join(m.name for m in ms)),
            ("positions", ",".join(str(p) for p in pos)),
        ):
            rec[key].append(v)

    assert all(len(x) == L for x in seqs), "a sequence is not 200 bp"
    print(f"built {len(seqs):,} sequences ({n_fail} fell back to a bare background)")
    for f in ("background", "vocab", "arrangement", "orientation", "instance"):
        u, c = np.unique(rec[f], return_counts=True)
        print(f"  {f:<12} " + "  ".join(f"{x}:{y}" for x, y in zip(u, c, strict=True)))
    u, c = np.unique(rec["n_sites"], return_counts=True)
    print("  n_sites      " + "  ".join(f"{x}:{y}" for x, y in zip(u, c, strict=True)))

    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    labels = np.zeros(len(seqs), np.float32)
    if not a.no_label:
        os.environ.setdefault("AG_S2_ORACLE_DIR", str(REPO / "outputs/oracle_v3/s2"))
        from experiments.exp1_1_scaling import _load_oracle
        from scripts.generate_labeled_pools import _label_sequences

        oracle = _load_oracle("k562", oracle_type="ag_s2")
        labels = np.asarray(_label_sequences(oracle, seqs), dtype=np.float32).ravel()
        assert np.isfinite(labels).all(), "oracle returned a non-finite label"
        print(f"labelled: mean {labels.mean():.3f} sd {labels.std():.3f}", flush=True)

    p = out / "motif_grammar_oracle.npz"
    tmp = str(p) + f".tmp{os.getpid()}"
    np.savez(
        tmp + ".npz",
        sequences=np.asarray(seqs, dtype=object),
        oracle_mean=labels,
        oracle_id=np.array("oracle_v3"),
        test_set_version=np.array("motif_grammar_v1"),
        **{f"f_{k}": np.asarray(v, dtype=object) for k, v in rec.items()},
    )
    os.replace(tmp + ".npz", p)
    (out / "motif_grammar_factors.json").write_text(
        json.dumps(
            {
                "n": len(seqs),
                "seed": a.seed,
                "backgrounds": list(BACKGROUNDS),
                "vocabs": {k: len(v) for k, v in vocabs.items()},
                "n_sites": list(N_SITES),
                "arrangements": list(ARRANGE),
                "gaps": list(GAPS),
                "orientations": list(ORIENT),
                "instances": list(INSTANCE),
                "background_source": "genomic_oracle.npz (chr7/13 held out)",
            },
            indent=2,
        )
    )
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
