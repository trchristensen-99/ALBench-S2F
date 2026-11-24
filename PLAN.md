# Plan log

Dated blocks, newest last. This is the running record of what was launched and why; the
measured results live in the memory notes and in `docs/`.

## 2026-11-24 — the parameter layer, the 300k start, and one dead end closed

**Context.** Arm A (start from 0) and Arm B (start from 30k) are complete, on genomic and
on both motif reservoirs. The immediate deliverable is the 4M human + 4M yeast library
the collaborators will synthesise, so the question moved from "which reservoir family" to
"which settings, at what budget, and in how many cell types".

**Launched.**

| arm | what | cost |
|---|---|---|
| `var30k` (4073064, 4073099) | 28 reservoir-PARAMETER pools at 30k: site density, vocabulary size, site strength, mutation rate, EvoAug operation decomposition, and five model-guided objectives | ~8 GPU-h |
| Arm A2 (registry) | 84 student cells, 28 variants x 3 seeds, gated on those pools | 34 GPU-h |
| `ab42` (4073098) | Arm B at the 300k start: 300k + 150k on both motif reservoirs, random vs epistemic vs badge_epistemic_only, 3 seeds | 65 GPU-h |
| `ye45` (4073094) | the DREAM test set scored by the AlphaGenome yeast ensemble, as a second opinion on the RNN oracle | 1 GPU-h |

~108 GPU-h, inside the standing 150 GPU-h grant.

**Decisions and their reasons.**

- *Four model-guided objectives registered.* `InSilicoEvolutionGenerativeSampler` has
  supported minimize, target and diverse_targets since it was written and the registry
  exposed only `maximize`. A library concentrated at the top of the dynamic range teaches
  a student nothing about the rest of it, so "which part of the range do we buy" is now
  four arms.
- *Model-guided arms are guided by Arm A's own banks, not by the oracle.* Guiding with
  the oracle selects at the argmax of the ensemble that then labels the pool: an upper
  bound no collaborator can reproduce, since they have a student trained on their own
  data. It is also ~14 GPU-h per pool against ~25 minutes. `albench/ensemble/bank.py`
  reloads a saved LegNet bank as a predictor and the guide is stamped into the pool
  identity, so a bank-guided and an oracle-guided pool cannot collide on one hash.
- *Arm B at 300k runs on the motif reservoirs and cannot run on genomic.* The real-CRE
  pool holds 314,981 distinct sequences, so a 300k seed set exhausts it. That is most of
  the argument for why a 4M library cannot be mostly real sequence.
- *Three acquisition rules, not six.* Arm B settled badge, badge_kmeanspp_only and
  batchbald with their own controls; re-running them at 300k would spend half the budget
  confirming a known result.

**Closed.**

- *The zoonomia duplicate confounder does not exist any more.* Measured, not assumed:
  every zoonomia and zoonomia_orthologs pool Arm A read is 0.00% duplicate and the 30k
  pool is an exact prefix of the 100k one. The 15.4% figure dates from Nov 6 and the Nov 8
  rebuild fixed it, so the arm's last place in Arm A is a real result.
- *There is no second yeast oracle.* The AlphaGenome yeast ensemble scores r = 0.6076 on
  the DREAM test set against the RNN's 0.9647, and r = −0.4652 on the 4.1% of sequences
  where the two disagree. Holding it out as an independent judge is not available. The
  replacement plan is to score on the 71,103 REAL DREAM measurements rather than on oracle
  labels, which removes the oracle-copying hazard from every reported endpoint.

**Two bugs the canary caught before they cost anything.** `bench33_legnet.py` saves the
inner LegNet with bare keys while `mpra_legnet_ref.Net` expects them prefixed; and it
derives its output directory as `stem.split("_d")[0]`, so `ise_diverse` and
`ise_diverse_g300k` would have shared one directory and overwritten each other's weights
while both reporting success. Submitting array tasks 0 and 21 ahead of the other 26 is
why this cost one job rather than eight GPU-hours.

**Still open, in priority order.** A yeast reservoir comparison (zero coverage, and half
the deliverable); expression-conditioned cell-type conditioning, which needs ENCODE pulls
for five cell lines; the ct45 LOCO fold, still diverging in 9 of 42 runs at lr 0.005 even
with clipping.

## 2026-11-24 (evening) — tiers, gradient design, and the pool-size axis nobody had moved

**Cancelled.** `ab42`, the 300k-start acquisition arm, six tasks in flight. The from-scratch
rung is what the library decision rests on and the start-from-300k arm needed 450k-sequence
runs to say anything. 65 GPU-h and six queue slots returned; the queue absorbed them inside
three minutes.

**Launched.**

| arm | what | cost |
|---|---|---|
| `sc46` (registry, 12 cells) | the missing `motif_ct_enriched` from-scratch curve at 400k/500k/600k/1M, seed-PAIRED with shared_core's own seeds at each size | 52 GPU-h |
| `ab43` (4074540) | candidate-pool SIZE via `--cap` (50k/500k/970k) and a UNION pool over both motif reservoirs, 5 seeds each | 24 GPU-h |
| `cd46` (4074575) | `onehot` and `embed` conditioning against `agnostic`/`text`/`text_shuffled` on SEEN cells, lr 0.002 | ~22 GPU-h |
| Arm A3 (4074595 + 30 cells) | six motif-TIER mixtures and four FastSeqProp designers | ~20 GPU-h |

**Three findings that changed what got built.**

- *Arm B never used its 1M pools.* `run_al` caps scored candidates at 200,000 and uniformly
  subsamples above that (`albench/acquisition/run_al.py:111`), so every published Arm B arm
  selected from 200k candidates whatever pool it was pointed at. The pool-size axis is
  therefore unmeasured, and `--cap` on one pool is a cleaner experiment than slicing pools:
  seed set, reservoir, recipe and oracle are all held byte-identical.
- *Native-motif protection was capping planting density, and the cap scales with vocabulary
  size.* Measured over 200 sequences: shared_core (71 entries) finds 3.8 native matches and
  plants 5.10 of the 3-7 asked; ct_enriched (152) finds 19.7 and plants 4.61; the
  unrestricted 628-entry vocabulary finds 179.4 and plants 2.21. Consensus strings run as
  short as 4 bp and match by chance. `native_min_len` fixes it, defaulting to 0 so no arm
  measured before today changes; only `motif_tiered` sets 8, which restored it to 4.96.
  **Consequence: the site-density variants now running must be read on MEASURED density.**
- *FastSeqProp optimises hard and partly optimises its guide.* Canary: guide 0.358 -> 14.08
  on its own designs, oracle 6.92 on the same sequences, 0.48 on the CREs they started from.
  The gain is real (the oracle is independent) but about half the claimed improvement is
  guide exploitation, and 6.92 is far outside the measured range (motif pools 0.78-1.21).
  Report the guide-oracle gap for every model-guided arm.

**Not launched, and why.** The AlphaGenome yeast oracle is S1 — a head trained on a FROZEN
human/mouse-pretrained encoder — which is the obvious explanation for r = 0.6076 against the
RNN's 0.9647: nothing makes a frozen human encoder linearly predict yeast promoter strength.
S2 (joint encoder + head fine-tuning) is already implemented as a student type with an HP
grid for `alphagenome_yeast_s2` and was never used for the oracle. But no training script for
the yeast AG oracle survives in the repo, so this is a new JAX/orbax trainer rather than a
config change, and it is not on the critical path: we have a 0.9647 oracle. Staged plan is
one fold first (~6 GPU-h) against the RNN's 0.9647 before committing to a 10-fold ensemble.

**Budget.** ~174 GPU-h committed across today, over the standing 150 GPU-h grant. The
instruction was to saturate the GPUs and the queue is in fact saturated (42 running, 13
pending, zero free, at the cap on every QOS). `cd46` is the arm to cut first if the grant
is meant to hold.
