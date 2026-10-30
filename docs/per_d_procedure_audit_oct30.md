# Per-D procedure audit (Oct 30, written while the cluster was unreachable)

Question: for each choice in the procedure that turns "a training set of D oracle-labelled sequences" into "the student ensemble we score", is there an experiment behind it, how strong is it, and is anything obvious left on the table (the flanking context was the example)?
Sources: the Evidence Ledger, `slide_figures/benchmarks/*` (30k benchmark ladder, pk35, bench33), the study notes of Oct 12 to 29, `docs/MENU.md`. No cluster data were read after 04:25 on Oct 30, so results of the sweeps and arms still running are listed as pending.

Evidence grade: **A** = paired, several architectures or pools, effect clear of the noise (seed SE about 0.0013, run-to-run SD about 0.006); **B** = one pool, one size or one reservoir, or inside the noise; **C** = assumed, never tested, or tested only under the old (bare-insert) setup.

## 1. The procedure, step by step

| step | current choice | grade | evidence and where it stops |
|---|---|---|---|
| Labels | oracle_v3 labels for every sequence; experimental labels reported as a second column | A | oracle vs experiment r = 0.931 held out; students' ordering is the same against experiment. Students distil the oracle, so oracle-scored numbers flatter anything resembling the oracle backbone. |
| Pools | nested prefixes; independent replicate pools r1..r3 for every comparison | A | 3 independent pools beat 3 seeds on one pool in 12 of 12 comparisons. The Oct 13-14 scaling cells used a separate pool per D (not nested): the genomic curve is flat from 30k to 100k for that reason. |
| Input window | 200-bp insert + **20 real plasmid bases per side (240 bp)** | A at genomic 30k, B elsewhere | +0.049 (4 bases) ... +0.075 (20) ... +0.063 (200); all 18 sets plus SNV effect gain. Sweeps at 10k/100k, random and motif 30k (`sw38_`) not read yet; fairness of reservoir ranking with vs without context (`fr35_`, 10 reservoirs) not read (local copy: 2 of 10 started). |
| Shift augmentation | +-10 bases over the real flank, centred at test | A at 30k | none -0.012, +-30 -0.015, 200 bases of context -0.007 versus 20. The oracle-like variants (per-sample shift, shift-TTA, single-pass RC; `fx36_`, `sw38_`) not read. |
| RC | train with RC, average the two strands at test | A | RC test-time average +0.014 (Oct 20 A/B). |
| Architectures | menu v1c: 20 LLM-written configurations (5 hard-wired ones rewritten length-agnostic) | B | Frozen menu beats bespoke per-strategy searches in 20 of 20 cells (+0.0115 genomic, equal-count control +0.0093) **but a tuned MPRA-LegNet ensemble of 20 seeds ties the menu at 30k (0.782 vs 0.782)** and the menu leads only at 10k (+0.021). The menu was built before the flank result; menu v2 under the standard is planned. |
| Recipe | Muon, lr 2e-3, wd 1e-4, batch 256, one-cycle, patience 16, one seed per configuration | A at 30k, **C at 10k and 100k** | lr, batch and wd change a single model by <= 0.004; AdamW -0.023; patience 16 vs 8 +0.0044 on 20 of 21 pools. No lr/wd/epoch sweep at 10k or 100k (only a 600k check on 3 architectures). |
| Ensemble | positive ElasticNetCV on z-scored validation predictions of everything trained | A | ties greedy selection, beats the plain mean by 0.002 to 0.018 over 12 cells. |
| How many models | K = 20 at 10k/30k/100k | A at 30k/100k, B at 10k | 30k and 100k are within 0.002 of the full menu from 10 to 11 architectures; 10k still gains at 20. Extra seeds of the best three add +0.001 to +0.002 (one pool). |
| Validation | V0c: 29,449 real chr19/21/X sequences, oracle labels; drives early stopping and the stack | B | No variant of V0c gave a reliable win. It is genomic, so for random/motif training sets both stopping and weights are tuned on a different distribution (per-target validation was tested in August under the old setup only). At D = 10k it is three times larger than the training set. |
| Search settings (when AutoResearch is run) | Sonnet 5.5; guidance block; separate coder; 10 rounds x 2 models | A for the guidance block (+0.016 to +0.020), coder (+0.005), B for the rest | Opus +0.0055 (n.s.); rounds 10 to 15 +0.002; manager +0.006 (SE 0.003, 4 of 6). Steering arms (`st38_`) running; the collaborator's 200-model search not reproduced yet (`lng35_` running). |
| Evaluation | 18 sets plus SNV effect, Pearson r (error metrics being added) | A | Seed noise floor 0.0013 SE, run-to-run SD 0.006: no claim below ~0.01 without 3 pools. No noise ceiling (replicate-replicate r) has been computed; the PI asked for it. |

## 2. What could still be on the table (ranked by expected size of the gain times cheapness)

1. **Student class (decision of Oct 30: AlphaGenome-based students are baselines only).** The AlphaGenome-FT student trained on the same 30k sequences scores 0.917 against the experimental labels (10-model ensemble; one model 0.916) versus 0.742 for the menu ensemble and 0.742 for the tuned LegNet ensemble then (bare insert; with the 20-base context the menu reaches 0.816 and LegNet 0.827 against the experimental labels). It is reported as a baseline and a ceiling, individually and as an ensemble, and is **not** used as a member of the per-D ensembles or in any scaling comparison: its encoder carries a large amount of outside knowledge and is architecturally close to the oracle ensemble that labels every sequence, so it would not measure what a corpus adds. The question this item raises, whether the reservoir conclusions depend on the student, is answered with students that learn only from the training set (LegNet seeds, the menu, AutoResearch runs): the first AlphaGenome and LegNet pair on three reservoirs ranked them identically, which is a sanity check, not a design input.
2. **Compute-scaled selection versus a retrained frozen menu.** Keeping the best 20 *trained instances* out of all 1,297 models run on the 30k genomic pool gives 0.798 genomic / 0.297 designed / 0.743 random; the menu retrained from scratch gives 0.782 / 0.271 / 0.711. Retraining the 45 candidate configurations also gives only 0.780, so what carries the extra 0.016 is the selected instances (about 65 times the training), not the configurations. The frozen menu wins at **equal model count** (20 of 20 cells), but nobody has compared the two at **equal compute**. If compute is cheap relative to the labelling budget, "search per D, keep the best trained models" may be the stronger per-D procedure; menu v2 under the standard, with members chosen for ensemble contribution, plus one equal-compute comparison, would settle it.
3. **Recipe and regularisation at the sizes that were never swept (10k and 100k).** At 10k a small dataset plausibly wants more regularisation (weight decay, dropout, fewer epochs); at 100k, more width/depth. The hooks exist (`--lr_scale`, `--bs_scale`, `--epoch_scale`; width-scaled cells 11 of 33 done).
4. **Oracle-like training details beyond the flank** that the CNN students can copy: per-sample shift (half of samples shifted), single-pass RC augmentation, shift test-time augmentation, batch 128, dropout 0.1 (`fx36_`, `sw38_` arms, results unread). Excluded on purpose: using the oracle's disagreement as a loss weight.
5. **Auxiliary targets** (the oracle is trained on several cell types): HepG2 auxiliary target (`mt38_`, fixed a float64 bug on Oct 29, results unread).
6. **Loss function.** Everything is MSE although the score is Pearson r; Huber, correlation or rank-aware loss, and label transforms were never compared. Likely small, very cheap to test with a harness flag.
7. **Mutation-style augmentation inside the student** (EvoAug as a training augmentation, not as a reservoir): removed from the proposable surface in October because it was confounded with the reservoir of the same name; never tested cleanly on the standard input.
8. **Longer searches at 10k/30k** (collaborator: 200 models, best single 0.8165 on his files): we reach 0.80 to 0.83 singles with the flank, so the first-order explanation is the input, not the search length; `lng35_` will say how much is left.

## 3. Risks to the "unbiased across reservoirs" claim

* All reservoir rankings so far (Oct 13-14 scaling, Oct 28 menu scaling, Yash pools) used **bare-insert students**; nobody has re-ranked the reservoirs with the standard input. The fairness cells (`fr35_`, same architectures on ten reservoirs with and without context at 30k) answer exactly this and are the most important pending result.
* Validation is genomic for every reservoir (see Validation above).
* The Oct 13-14 scaling cells have a separate pool per D, so exponents from three points are noisy (differences below ~0.05 are noise) and the genomic reservoir shows a flat 30k to 100k.
* Reservoir ordering on the designed set is nearly flat in D (exponent 0.03 to 0.13) while everything else improves at 0.15 to 0.30: the designed-sequence generalisation is limited by what the pools contain, not by size; this is a corpus finding, not a student finding.

## 4. Proposed next experiments (once the cluster is back), in order

1. Read `fr35_` (ranking with vs without context) and `sw38_` (optimum context/shift per D and reservoir); if they hold, freeze `harness_standard.py`.
2. Student-robustness check of the reservoir ranking with students that start from scratch: the published MPRA-LegNet (many seeds, context input) and the menu on the five standard cells, compared with the AutoResearch runs (AlphaGenome students stay baseline-only).
3. Menu v2 under the standard, members picked by ensemble contribution from the `st38_` and `sw38_` runs; retrain at 10k/30k/100k.
4. Recipe sweep at 10k and 100k (lr, wd, epochs, batch) with the menu v2 architectures, 3 independent pools.
5. Loss-function arm (MSE vs Huber vs MSE + correlation) on the five standard cells.
6. Noise ceiling from replicate-replicate r of the experimental data.
