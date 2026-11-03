# The per-size search procedure and the evidence for every choice (Nov 3, 2026)

What the study currently does to turn "a training set of D oracle-labelled sequences" into "the student ensemble we score", why each step is the way it is, what was tried and left out, and what is still missing. Every number comes from a file in this
repository (sources in the last column or in the text); nothing here is new training except section 4.

**Reading guide.** Numbers are genomic-test Pearson r (chr7/13, never used for a choice) against oracle_v3 labels unless a column says otherwise; "mean of 18" is the mean over the 18 evaluation sets; OOD is the designed set. Selection and early stopping use V0c
(chr19/21/X, 29,449 sequences). Noise: a single model's seed SE is 0.0013; a loop *run* (search trajectory plus pool) has SD 0.006-0.008 on the stack (0.0015 for the H3 arms), so three independent pools resolve about 0.01 unpaired and
0.005-0.008 when paired by pool. **Grades:** A = paired, several pools or architectures, effect clear of the noise; B = one pool, size or reservoir, or inside the noise; C = assumed or never tested at that size.

## 1. The procedure

**Step 0, data.** Training sets are nested prefixes of canonical pools (`k562_nested`, seed 42) or independent replicate pools (`k562_reps` r1-r3, one per independent run). Labels are oracle_v3; experimental labels are a second column (oracle vs experiment r = 0.931).

**Step 1, the student input standard** (`harness_standard.py`, fixed): 200-bp insert in its real plasmid context, 20 real flank bases each side (240 bp), training window shifted by +-10 over the real flank (one shift per batch in the loop harness, per sample in the bank trainer), centred window and
two-strand average at evaluation, single-pass reverse-complement augmentation (half of the samples flipped; `AL_RC_MODE=aug`).

**Step 2, the reference bank (no LLM)** (`bench33_legnet.py`, `docs/TUNED_LEGNET.md`): the published MPRA-LegNet (1.32M parameters; AdamW, peak lr 0.01, batch 1024, one-cycle, fp16) with a per-size recipe found by sweep, no early stopping (the epoch with the best V0c Pearson is kept, always near the end of the schedule), 20 seeds:

| size | epochs | weight decay | single r | 20-seed stack (genomic / OOD / mean of 18) |
|---|---|---|---|---|
| 10k | 800 | 0.4 | 0.755 | 0.7801 / 0.288 / 0.731 |
| 30k | 500 | 0.2 | 0.8748 | 0.8907 / 0.422 / 0.853 |
| 100k | 500 | 0.1 | 0.9269 | 0.9381 / 0.584 / 0.917 |

To extend to a new size: sweep epochs (150-800) and weight decay (0.1-0.4) with lr and batch fixed (halving lr or doubling batch was worse at every size), 5 seeds per arm against the published recipe, select on V0c, then 20 seeds (`lr38_*`, 49 cells). 300k and up are not tuned.

**Step 3, the AutoResearch loop** (LLM architecture and recipe search; `albench/arch_search/agent_loop.py`, launcher `cluster_live/arch_autoresearch.py`). One run = 15 rounds x 2 trained models = 30 models, three independent pools per condition:

1. *Propose*: Sonnet 5.5 writes 6 ideas (IDEA / WHY / DESIGN / TRAIN). The prompt holds the guidance block (conv/SE/Muon/U-Net/attention facts from our data), the published LegNet source and the per-size bank bar (validation r of the tuned bank), the training facts (a one-cycle schedule always runs to the end) and the history of what was tried (top 18 by validation r, failures with reasons).
2. *Coordinate*: a second call picks 2 of the 6. Manager `schedule_ens`: rounds 0-5 (explore, `explore_frac` 0.4) show a coverage table of mechanism classes and recipes and ask for under-represented classes and new recipes; rounds 6-14 (exploit) show the validation stack (which models carry weight) and ask for refinements of the models that carry weight.
3. *Code*: a coder call writes the `Net` class from the design (up to 2 repairs; static and CPU forward checks; at most 40M parameters).
4. *Train and evaluate* in the harness **H3**: patience 16 on a one-cycle schedule never stops it (`AL_ONECYCLE_FULL`), fp16 (non-finite batches skipped), single-pass RC augmentation (`AL_RC_MODE=aug`), the standard input, a per-model wall cap of 40 H100-minutes at 30k, best epoch by V0c; the proposer and coordinator see validation numbers only.
5. *Transport*: the Claude CLI in an empty directory with `--tools ""`, the loop's own `--system-prompt`, `--effort medium` (`docs/LLM_LIMIT_NOV2.md`).

**Step 4, the ensemble.** Positive ElasticNetCV on the z-scored validation predictions of everything trained (the bank, the loop's models, the menu at 10k), fitted on V0c; members with weight zero are unused.

**Step 5, what to deploy per size (status at Nov 3).**

| size | best measured ensemble | status of the loop at this size under H3 |
|---|---|---|
| 10k | tuned LegNet bank + menu v2 (+ menu at 3x epochs): 0.7916 / 0.7941 (bank alone 0.7801; menu v2 alone 0.7846). Old-harness loop + bank: 0.787 | `rr40_10k`, `lv40_10k` running (3 pools each) |
| 30k | **loop + bank 0.8975 / 0.420 / 0.8615** against bank 0.8907 / 0.422 / 0.853 and loop alone 0.8865 / 0.385 / 0.846 (`rr40`); the menu adds nothing | done (3 pools); long and diversity arms running |
| 100k | tuned LegNet bank 0.9381; the old-harness loop (patience 8) 0.899 adds nothing to it | `rr40_100k`, `lv40_100k` waiting for H100 |

So there is **one procedure that is validated at all three sizes (the tuned-LegNet bank, no LLM)** and **one that adds to it at 30k (the H3 loop)**; the loop's value at 10k and 100k under H3 is not measured yet.

## 2. The decisions and their evidence

| # | decision | evidence | grade | not tested / limits |
|---|---|---|---|---|
| 1 | oracle_v3 labels for training, selection and scoring | oracle vs experiment r 0.931 held out; students' ordering is the same against experiment (`per_d_procedure_audit_oct30.md`) | A | students distil the oracle, which flatters anything resembling its backbone; AlphaGenome students are therefore baselines only |
| 2 | nested pools, independent replicate pools per comparison | 3 independent pools beat 3 seeds on one pool in 12 of 12 comparisons; Oct 13-14 non-nested pools made the genomic curve flat 30k-100k | A | - |
| 3 | 20 real flank bases per side, shift +-10, RC | context: +0.049 (4 bases), +0.075 (20), +0.073 (100), +0.063 (200); all 18 sets gain; shift none -0.012, +-30 -0.015; RC test-time average +0.014 (`STUDY_LOG`, `MENU.md`) | A at genomic 30k, B elsewhere | `sw38_` sweeps at 10k/100k and other reservoirs: 20 bases plateau at all D; fairness of reservoir ranking with context (`fr35_`): context lifts every reservoir +0.05..+0.12, rank corr 0.86-0.99 |
| 4 | the bank's recipe per size (long AdamW, no early stopping) | published recipe under-trained: tuned +0.066 (10k, SE 0.005), +0.024 (30k, SE 0.001), +0.007 (100k); e200 costs -0.006 at 30k; wd 0.4 too strong at 30k, wd 0.2 hurts at 100k; patience inside the schedule costs -0.065 / -0.049 / -0.18 (`TUNED_LEGNET.md`) | A | 10k gain still rising at 800 epochs; 300k+ untuned; selected from the sweep it is reported on (margins are 5-10 SE) |
| 5 | **harness fixes for the loop (H3)**: patience, one-cycle full, fp16, RC augmentation | the launcher stopped every model after 8 stagnant epochs: 91% stopped before 90% of their schedule (98% at 10k); published LegNet through the harness was 0.02-0.04 below the native trainer; the RC consistency loss (average of forward and RC losses, 2 passes) costs 0.032 genomic / 0.05 OOD, single-pass RC augmentation closes the gap to 0.001 (`pa38`, `pa39`); stack 0.8538 -> 0.8744 (patience, fp16) -> 0.8865 (RC aug) over three pools | A | cause of the remaining 10k/100k gaps unknown until the 10k/100k arms finish |
| 6 | reference LegNet source and bar shown from round 0 | open prompt without the hint -0.017 (SE 0.007), late reveal -0.010, stated target -0.015 (`op38_*`, 3 pools) | B | costs diversity (section 4) |
| 7 | guidance block in the prompt | +0.016 to +0.020 (Oct harness); the knowledge-base handout is weak (+0.008 over 5 cells, was +0.027 in the June harness) | A (guidance), B (KB) | minimal guidance (`st38` a4) is within noise of the standard |
| 8 | separate coder call | +0.005 | A | - |
| 9 | coordinator picks 2 of 6 | the coordinator's pick is +0.004 better than a random pick of 2 of 6 (+0.001 over no selection); +0.005/+0.019/+0.014 over the default in three pools but ~equal to the coder alone | B | - |
| 10 | manager `schedule_ens` (explore 40% then exploit) | +0.0062 (SE 0.0032, 4 of 6 pairs) over the coordinator alone; early rounds explore and late rounds exploit (picks with an untried feature 54% -> 4%; LegNet-derived 14% -> 55%); exploitation front-loaded: the last 4 rounds add 0.003 | B | the +0.006 is inside the noise in other comparisons (`STUDY_LOG`) |
| 11 | explore fraction 0.4 | 0.2 / 0.4 / 0.6 / 1.0: 0.8587 / 0.8744 / 0.8647 / 0.8700 genomic; paired by pool against 0.4: -0.016 (SE 0.006), -0.010 (0.007), -0.004 (0.005) | B | all three differences are in the same direction but none is significant with 3 pools (chance that the arm chosen earlier comes out best of four by luck: 1 in 4); not repeated under H3 |
| 12 | 15 rounds x 2 models, 6 ideas per round | rounds 10 -> 15 +0.002; 50-round runs (203 models) no better than 30-model runs; 75% of the gain by round 6-7 in the older harness; more than ~5 proposals per call adds nothing (n=8 +0.004, n=1 -0.029); 6 proposals x 10 rounds beat 1 x 60 (+0.016 at 60 models, random reservoir) and 2 x 30 was the best cadence (Oct 13) | A (saturation), B (exact values) | `rl40` (30 rounds) running under H3 |
| 13 | Sonnet 5.5 | Opus +0.0055 (n.s.) | B | Opus not tried under H3 |
| 14 | V0c for stopping, selection, proposer feedback and stack | validation r tracks test r across the models of a run (Spearman 0.94-0.98 on genomic; top 10: 0.54-0.83), no winner's curse (winners +0.0023 above the line, SE 0.0006); selection signal for OOD (0.49 among the top 10) and the GC sweep (0.24) is weak | A (genomic), C (OOD) | mixed validation: row in section 3 |
| 15 | positive ElasticNetCV stack over everything trained | ties greedy selection, beats the plain mean by 0.002-0.018 over 12 cells; greedy K=5 is 0.01 lower | A | - |
| 16 | bank + loop pooled in one stack | +0.007 genomic and +0.009 mean of 18 at 30k (`rr40`+bank 0.8975); at 10k +0.007 (old harness); 0 at 100k (old harness) | B | needs the 10k/100k H3 runs |
| 17 | the frozen menu is used only at 10k | menu beats bespoke per-strategy searches in 20 of 20 cells (+0.0115) but a 20-seed tuned LegNet ties it at 30k (0.782 vs 0.782 bare insert) and beats it at >=30k with the standard input (0.8907 vs 0.8674); menu + bank adds +0.012-0.014 at 10k, 0 at 30k and 100k; menu members are not under-trained (3x epochs +0.001 +- 0.004) | A | menu scaling exponent is the flattest (0.36 against 0.52 for tuned LegNet) |
| 18 | number of models: 20 seeds of the bank; 30 loop models; menu K = 20-30 at 10k | 30k and 100k within 0.002 of the full menu from 10-11 architectures; 10k still gains at 20; extra seeds of the best three +0.001 to +0.002 | A at 30k/100k, B at 10k | - |

## 3. Tried and not included (each row: result, source)

| tried | result | source |
|---|---|---|
| AlphaGenome-FT students (frozen or fine-tuned) as members | S2 stack 0.952 / 0.963 (10k / 30k) against oracle labels (0.907 / 0.917 against measured labels) vs 0.78 / 0.89 for the bank; reported as baseline and ceiling only: the encoder carries outside knowledge and is close to the oracle's backbone | `AGENTS.md`, `per_d_procedure_audit` |
| EvoAug as a student augmentation / in the proposable surface | removed (338 models lost to a regression; confounded with the reservoir of the same name); never tested cleanly | Oct 6 notes |
| reflector stage | +0.005 either way, nothing beyond the coordinator | `STUDY_LOG` |
| explore-style prompt, quota manager, harness-assigned recipes, minimal guidance | within noise (explore prompt 0 pooled, n=51) | `STUDY_LOG`, `st38` |
| fallback manager (compare with the reference) | -0.005 (SE 0.012) | `STUDY_LOG` |
| `diverse` proposer (propose complementary members, not the best) | +0.017 over three seeds in the June harness; **not in the current loop** (only the coordinator sees the stack) | Phase-0 notes |
| epochs x0.5 or x2 / x3 without early stopping (menu) | -0.005..+0.002 / +0.001 +- 0.004 | `STUDY_LOG`, `ep38` |
| patience 4 / 24 / 32 | worse (-0.02..-0.06 members) / members +0.0004..+0.002, ensemble -0.004..+0.003 for 1.3-1.6x time | `analyze_patience.py` |
| per-sample shift, the oracle's +-15 on half the samples, shift test-time averaging, contexts > 20 bases | -0.0002 +- 0.0022, -0.0046 +- 0.0019, +0.0003 +- 0.0020, 0 | `STUDY_LOG` |
| EMA / weight averaging with BN recalibration | single LegNet +0.013 at 100k, -0.016 at 10k; ~0 in ensembles | `pk35` |
| width x2 / x3 of the menu at 100k; auxiliary HepG2 target | -0.002 / -0.001; +0.001..+0.002 genomic but -0.008..-0.013 OOD | `scl100k_`, `mt38_` |
| single-pass RC augmentation on menu members | +0.004 (30k), +0.023 (100k), -0.014 (10k), OOD -0.013..-0.030 (for the LegNet inside the loop harness it removes a 0.032 loss) | `rc38_`, `pa39` |
| recipe sweeps of menu members (lr, batch, wd) at 10k and 100k; AdamW on menu members | flat within +-0.005; AdamW lr 1e-3 -0.023 | `hp38_`, `hp34_` |
| LegNet e100 recipe transplanted onto menu and the collaborator's architectures | members -0.02 to -0.07, ensembles hold at >= 30k | `tr38_` |
| the collaborator's open-world architectures re-implemented | 0.843 genomic single at 30k (stated 0.852-0.855), below the tuned LegNet at every size, best at 10k | `pi38_` |
| uncertainty-weighted and DPP-diverse ensemble selection | no gain over the stack | Oct 15 |
| dropping the bottom members | <= 0.0001 when chosen by retrained validation r (the positive stack already zero-weights them) | Oct 28 |
| mixed validation sets (V1 perturbed, V2 diverse, V3 thirds, balanced compositions) in the **menu** stack, post hoc | +0.0005..+0.0016 mean of 18; OOD +0.008 (V3, 6 of 6 pairs); all other sets inside noise | `LOOP_DIVERSITY_AND_VALIDATION.md` s.4, `LEGNET_VS_LOOP` s.4 |
| the same mixed set (B1, 16,800 sequences) **inside the loop** (`vc39`) | genomic -0.0167 (SE 0.0057, 3 of 3 pools negative), mean of 18 -0.0116 (SE 0.0052), OOD +0.007; validation-to-test rank agreement inside the runs is the same (0.95 vs 0.96) but the validation winner's regret is larger (0.008 vs 0.003) and exploit-phase models are 0.012 lower; the stack is fitted on B1 for this arm. Not significant with 3 pools (t about -2.2 to -2.9, 2 df); mechanism not separated | this document, `compare_legnet_vs_loop.py` |
| per-D proposal schedules; knee-based stopping | the gaps against a uniform schedule were trajectory luck, not a schedule effect (F24b/c); knee stopping is off by default (the full-rounds decision) | Oct 15, Oct 19 |

## 4. Diversity: what the loop does with architectures other than LegNet variants

Full numbers and mechanisms in [DIVERSITY_FUNNEL_NOV3.md](DIVERSITY_FUNNEL_NOV3.md) (21 standard 30k runs). In one paragraph: designs that do not name the LegNet trunk are 43% of explore proposals and are picked at the same rate as the others (35% vs 32%; the coordinator is neutral), but they are 9% of
exploit proposals, they were all trained with short recipes (median 58 epochs, against 150), they end 0.054 validation r below the best LegNet-named design, they hold 1% of the final stack weight, and the exploit phase never improves them (best-so-far unchanged from round 5 in 19 of 21 runs).
Diversity that reaches the stack is a BiGRU, attention or U-Net part on the LegNet trunk (16%, 9%, 3% of the weight). **The loop never gave the independent designs the long recipe**: 1 of 113 trained for 200 epochs or more (209 of 427 LegNet-named ones did), and in a partial recipe-matched bake-off the tuned recipe raises their validation r by +0.038 on average. Running (Nov 3): the bake-off (`bake_*`, no LLM), `ar41` (tuned recipe for every model), `ln41` (lanes: one LegNet-trunk and one independent-trunk model per round, both on the tuned recipe), `is40` (islands) and `qt40` (quota). `ar40`/`ia40` were retired after a bug trained them with the harness's 80-epoch Muon recipe (see the funnel note). Page with the figures: https://claude.ai/artifact/MAVM9LhXwRyr3NYk2mzyZb

## 5. What would still add evidence (ranked by value over cost)

1. **Does an independent family reach rough optimality when trained like LegNet?** (running: `bake_*`, `ar41`, `ln41`, `is40`; decision rules in the funnel note.)
2. **The loop at 10k and 100k under H3** (running; the 30k result is the only one measured under the fixed harness). Also whether the bank + loop union keeps adding at 100k.
3. **Explore fraction and mixed validation under H3 with more pools.** Both current results are 3 pools under H2; the fractions are paired-significant at 2 SE at best. Cheapest decisive version: rr40-style runs at 0.2 and 0.6 on three further replicate pools (`k562_reps` has r1-r3, so more pools need new labelled draws).
4. **Loss function.** Everything is MSE although the score is Pearson r; Huber, correlation and rank losses were never compared (one harness flag; likely small).
5. **Give the proposer the ensemble signal** (stack weights and correlation of the members; today only the coordinator sees them). The June `diverse` proposer gained +0.017 in the older harness; never rerun in the current one.
6. **Several different strong references in the prompt** (LegNet plus the collaborator's best non-LegNet designs once they are retrained to their own best), since a single shown reference is what the explore phase copies.
7. **Equal-compute comparison of "search per size and keep the best trained models" against a frozen menu** (the best 20 trained instances of 1,297 models score 0.798 against 0.782 for the retrained menu; nobody compared at equal compute).
8. **Recipe sweeps for non-LegNet families at each size** (menu v2 recipes are flat at 10k and 100k, but only for the menu's own recipe family).
9. **A noise ceiling** (replicate-replicate r of the experimental data; the PI asked) and the encoder fix for SNV-effect (indels left-aligned in loop/menu encoders, centre-padded in the bank: SNV columns are inflated by 0.02-0.03 and should be read on Hamming-1 pairs).
10. **300k and larger sizes** are untuned for both the bank recipe and the loop.

## 6. Where things are

`docs/TUNED_LEGNET.md` (bank, epochs, early stopping), `docs/LEGNET_VS_LOOP_AND_NEXT_ARMS.md` (harness gap, H3, arms), `docs/LOOP_DIVERSITY_AND_VALIDATION.md` (explore/exploit behaviour, validation sets), `docs/DIVERSITY_FUNNEL_NOV3.md` (this section's data), `docs/MENU.md`, `docs/STUDY_LOG_OCT29_NOV2.md` (what was run and what did not help),
`docs/per_d_procedure_audit_oct30.md` (grades, gaps), `docs/LLM_LIMIT_NOV2.md` (transport and the 5-hour limit), `docs/AUDIT_NOV2.md` (code audit). Registrations: `scripts/program_oct22/register_nov2b.py`, `register_nov2c.py`, `register_nov3.py`, `register_nov3b.py`, `register_nov3c.py` (bake-off), `register_nov3d.py` (islands).
