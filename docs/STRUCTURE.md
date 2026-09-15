# Repo structure: the general harness vs our specific study

## The tension, stated precisely

This repo is two things at once, and they pull in opposite directions.

**A benchmark harness.** Given an initial dataset, a way of generating candidate
sequences, a way of selecting among them, an oracle that labels, and a student that
learns — measure which generation strategy produces the best student per sequence.
Nothing in that sentence mentions MPRA, AlphaGenome, K562, or LegNet.

**One specific study.** MPRA in K562/HepG2, an AlphaGenome-MPRA oracle ensemble,
MPRA-LegNet students, the Gosai pool, a particular HP-search protocol, hg38-derived
reservoirs, and an eval battery (SNV, OOD, structural, CRE) built for this assay.

Conflating them is what produced a repo where `albench/` was clean and portable while
`experiments/` and `scripts/` hardcoded one person's filesystem. The fix is not to
generalize everything — most of the specific parts *should* stay specific. The fix is
to put a **seam** in the right place and be honest about which side each file is on.

## Where the seam actually falls

This is an audit of what we have, not a wish list.

**Genuinely general** — works on any DNA sequence-function task:
- the AL loop, pool construction, nested subsetting, screen/sweep expansion
- asset resolution (`paths.py`)
- `random`, `dinuc_shuffle`, `mutagenesis`, `evoaug`, `motif_planted`
  (motif planting needs a motif database, but that is an *asset*, not a domain)
- every acquisition method: they consume embeddings and uncertainties, nothing more

**Irreducibly specific** — and should stay that way:
- both oracles (AlphaGenome-MPRA, DREAM-RNN) and both students
- `zoonomia` and `encode_accessibility` (mammalian alignments, human peak calls)
- `gc_matched` and anything else that matches against a reference pool
- every eval set: the SNV battery, the DREAM eval classes, OOD/structural splits
- the HP search space (LegNet block widths, kernel sizes)

**The thing we keep mis-filing**: HP search is neither a reservoir nor an acquisition.
It is a *student-fitting* strategy, orthogonal to both, and giving it its own home
stops it leaking into the reservoir abstraction.

## Proposed layout

```
albench/
  core/          loop, pools, registry, paths, run — knows nothing about any domain
  strategies/    domain-agnostic generators (random, dinuc, mutagenesis, evoaug, motif)
  acquisition/   domain-agnostic selectors (badge, batchbald, diversity, binned)
  hpsearch/      student-fitting strategies (random, evo_*, llm_autoresearch)
domains/
  k562_mpra/     oracle.py student.py reservoirs.py evalsets.py config.yaml
  yeast_dream/   oracle.py student.py reservoirs.py evalsets.py config.yaml
```

A new setting means one new `domains/<name>/` directory implementing five protocols.
Nothing in `albench/` changes. That is the extensibility test, and it is worth
stating as an acceptance criterion rather than a hope.

## The five interfaces

Keep them this small. Every method below already exists in our code in some form; the
work is naming them, not inventing them.

```python
class Oracle(Protocol):
    id: str                                        # stamped into every labelled pool
    def label(self, seqs: Sequence[str]) -> np.ndarray: ...

class Student(Protocol):
    def fit(self, seqs, labels, hp: dict) -> "Fitted": ...

class Fitted(Protocol):
    def predict(self, seqs) -> np.ndarray: ...
    def embed(self, seqs) -> np.ndarray | None:   # None => acquisition falls back
        ...

class Reservoir(Protocol):
    def generate(self, n: int, ctx: Context) -> tuple[list[str], pd.DataFrame]: ...

class EvalSuite(Protocol):
    def score(self, fitted: Fitted) -> dict[str, float]: ...
```

`Oracle.id` is not decoration. Label provenance has already bitten this project once:
pools labelled by a non-canonical oracle are indistinguishable from good ones without
a stamp, and `full856k_clean` became the canonical id only after an audit.

`Fitted.embed` returning `None` is deliberate. BADGE and BatchBALD need embeddings; a
student that cannot provide them should degrade to uncertainty-only selection rather
than crash, and the caller should be told which happened.

## What NOT to generalize

- **Do not abstract the eval sets.** They are the scientific content. A generic
  "EvalSuite returns a dict" is enough; anything more invents a schema nobody needs.
- **Do not make the oracle pluggable at runtime.** One study uses one oracle ensemble;
  swapping it mid-run is how label provenance gets lost.
- **Do not build a plugin-discovery mechanism.** Two domains do not justify entry
  points. An explicit import in `domains/__init__.py` is clearer and debuggable.

## Migration path

Incremental, because the screen and the yeast oracle are running against this code.

1. Write `albench/core/protocols.py` with the five Protocols. Nothing implements them
   yet — this is documentation that type-checks.
2. Move the existing modules into `albench/core/` and `albench/strategies/` with
   re-export shims at the old paths, so nothing breaks mid-experiment.
3. Create `domains/k562_mpra/` by *moving* `experiments/exp1_1_scaling.py`'s
   MPRA-specific halves into it. This is the only step with real risk; do it when no
   screen is mid-flight.
4. `domains/yeast_dream/` follows the same shape, which is the real test of whether
   the seam is in the right place.
5. Delete the shims.

Steps 1–2 are safe today. Step 3 should wait for the 105-cell screen to finish.
