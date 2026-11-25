# Design: the attribution-driven explore/exploit reservoir

Status: **design only. Nothing here has been run.** Agreed on 2026-11-24 to settle the
design before spending compute. This document is the thing to argue with; when it is
agreed, the build is about three days.

## The idea, restated

Start from a dataset and a model. Use attributions to find which sequence features the
model has actually learned. Spend the next batch on *combinations* of those features until
combinations stop being informative, and spend the rest on exploring for features we do
not yet have. Shift the split between the two adaptively, based on how informative the
exploit half is still being.

This is the only proposal on the table where the reservoir is a function of what the model
has already learned. Every reservoir we have is fixed in advance.

## The loop

```
round r:
  1. ATTRIBUTE   gradient x input over the current training set, using the current bank
  2. DISCOVER    cluster high-attribution windows into a lexicon L_r of learned features
  3. BUDGET      split B into exploit/explore by the informativeness signal (below)
  4. EXPLOIT     generate combinations of L_r features: pairs, triples, spacings,
                 orientations, and dosage
  5. EXPLORE     generate from the fixed reservoirs (motif, EvoAug, design, random)
  6. LABEL       oracle-label both halves; retrain the bank; go to r+1
```

## What exists and what does not

| step | status |
|---|---|
| 1 attribute | **exists.** The differentiable path built for FastSeqProp (`albench/reservoir/fastseqprop.py`) gives gradient x input against a `LegNetBank` directly. ISM would be ~600x the cost for little gain at this stage. |
| 2 discover | **does not exist.** This is the build. |
| 3 budget | **does not exist**, and is the part that needs a decision (below). |
| 4 exploit | **mostly exists.** `MotifPlantedV2` already plants combinations at controlled spacing and orientation; it needs to accept a discovered PWM lexicon instead of CIS-BP. A vocabulary swap, not a new sampler. |
| 5 explore | **exists.** Every registered reservoir. |
| 6 label/retrain | **exists.** `build_pool.py` and the dispatcher. |

So the genuinely new code is step 2 plus the controller in step 3.

## Step 2: discovery

Take the top-scoring 8-20 bp windows by summed attribution, cluster them, and fit a PWM
per cluster. This is TF-MoDISco's job and TF-MoDISco is the obvious thing to use; the
reason to write a light version instead is that we need it to run every round on a 1M-scale
corpus, and we need the lexicon in `albench.motifs.vocabulary`'s `Motif` form so the
planting machinery takes it unchanged.

Decision to make: **do we need the discovered lexicon to be interpretable?** If the answer
is no, skip PWMs and let the exploit half plant the literal high-attribution k-mers. That
is simpler, faster and probably works; it is also much harder to present, because "the
model learned something and we planted more of it" with no name attached to the something
is a weak slide.

## Step 3: the informativeness signal, and the part I am least sure of

"Until those aren't informative anymore" has to become a number. Three candidates:

1. **Bank disagreement on the exploit candidates.** If the ensemble already agrees about
   the combinations, the combinations have stopped being informative. Cheap, needs no new
   data, and reuses the acquisition machinery. **But** it inherits exactly the confound
   found on Nov 24: ensemble spread grows with predicted activity because labels are
   log2FC, so this signal would drift toward "combinations are still informative" simply
   because combinations are active. It has to use the activity-normalised uncertainty
   (`activity_normalized`), not raw spread.
2. **Realised error on last round's exploit half.** After labelling, measure the bank's
   prediction error on the sequences the exploit half proposed. High error means the
   combinations were genuinely surprising. This is the honest signal and it is one round
   delayed, which is fine because the loop has rounds.
3. **Marginal held-out gain.** Retrain on exploit-only and explore-only increments and
   compare. This is the quantity we actually care about, and it costs a full extra
   training run per round, which is probably too expensive to use as a controller.

**Recommendation: (2) as the controller, (1) as a within-round tiebreak, (3) measured once
at the end to check the controller was tracking the right thing.**

## The confound that would invalidate the whole thing

The bank is trained on planted data. Its attributions will therefore rediscover the motifs
we planted, the exploit half will plant more of them, and the next bank will learn them
even better. The loop can report rising confidence while learning nothing new. This is not
a hypothetical: `motif_ct_enriched` corpora are ~5 planted sites per 200 bp from a
152-entry vocabulary, so the planted signal dominates the real genomic background.

Two controls, both required, not optional:

- **A genomic-seeded round 0.** The first lexicon comes from attributions over *real CREs
  only*. If the loop is working, L_0 should contain features we did not plant, and the
  overlap between L_r and the CIS-BP vocabulary we plant from is the number to report every
  round. Overlap climbing toward 1.0 means the loop is eating its own tail.
- **A frozen-reservoir control arm.** Same budget, same rounds, same oracle, but the
  reservoir never adapts. Without it, any gain is attributable to "more data" rather than
  to adaptation. This doubles the cost of the experiment and there is no version of this
  worth running without it.

## What would make us abandon it

Stated in advance so the decision is not made after seeing the numbers:

- L_r overlap with the planting vocabulary exceeds ~0.8 by round 2 (the tail-eating case);
- the adaptive arm does not beat the frozen-reservoir control by more than seed noise
  (~0.008) at equal budget;
- the controller never moves the split, i.e. the informativeness signal is flat, which
  would mean the exploit half was never distinguishable from the explore half.

## Cost

Per round at 100k acquired: attribution over the training set ~10 min on an H100; discovery
~20 min CPU; generation and oracle labelling ~25 min; bank retrain ~2 GPU-h at 3 seeds.
Call it **3 GPU-h per round**, so a 5-round arm is ~15 GPU-h and the mandatory frozen
control doubles it to **~30 GPU-h** for one complete comparison.

That is cheap enough that the reason to think before building is the design risk, not the
compute.

## Open questions for the discussion

1. Interpretable PWM lexicon, or literal k-mers? (Affects the build size and the slide.)
2. Is the frozen-reservoir control the right control, or should it be a fixed *mixture*
   matched to the adaptive arm's realised mixture? The latter is a harder test and
   separates "adaptation helped" from "the mixture it converged on was good" -- which is
   arguably the result we actually want, since a static mixture is what we would ship.
3. Does the loop run on K562 only, or does the yeast half need it too? Yeast has no
   reservoir comparison at all yet, so probably not.
