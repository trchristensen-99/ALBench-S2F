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
