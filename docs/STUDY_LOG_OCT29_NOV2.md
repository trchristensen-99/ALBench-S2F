# What was run, Oct 29 to Nov 2, 2026 (and what it showed)

Everything below uses the study's input standard (200 bp insert + 20 real flank bases per side, shift +-10, RC augmentation and RC-averaged prediction), canonical nested pools, V0c validation, oracle-labelled
test sets that were never used for a choice. Cell counts are registry entries (`scratchpad/dispatch_registry.json`, 2840 cells in total, all finished). Details and numbers per experiment are in the dated
sections of `scratchpad/oct28/PLAN.md` (a local working log); the conclusions that matter for the recommendation are in [TUNED_LEGNET.md](TUNED_LEGNET.md).

## Input standard and the menu
* `sw38_` (102 cells), `fx35_` / `fx36_` / `fr35_` / `pk35_` (about 100): context length and shift sweeps. Result: 20 real flank bases per side with shift +-10 is the standard (worth about +0.07 test r over the bare insert); nothing beat it.
* Menu v1c (20 LLM-found configurations) and menu v2 (built from the standard-input 30k searches): `mv1c`, `v1c38` (replicate pools), `v2m38` (42 cells), `st38` steering cells (4, plus 4 at 100k).

## Per-size procedure audit follow-ups
* `hp38_` (88 cells): recipe sweeps of the menu at 10k and 100k (lr, batch, weight decay): flat within +-0.005.
* `rc38_` (34 cells): single-pass reverse-complement augmentation: +0.023 genomic at 100k (20 of 20 pairs better), -0.03 OOD, -0.014 at 10k.
* `scl100k_` / `scl30_` (width-scaled menu, 32 cells), `mt38_` (auxiliary-target, 12 cells), `lg38_` / `lg35_` (collaborator-style LegNet reference, 33 cells): capacity and auxiliary targets are not levers.
* Combined and pooled ensembles (analysis only): ensembles saturate; combining sets adds +0.003 to +0.006 at 10k and 30k and nothing at 100k.

## Seed banks and baselines
* `bench38_` (65 cells) and the 100k top-ups: the published MPRA-LegNet at 10k / 30k / 100k with three recipes (e100, b256, pub), 20 seeds per recipe; AlphaGenome-FT students are baselines only (AGENTS.md).

## AutoResearch (LLM loop) at 30k
* `mgr38_` (6) and `op38_` (9): reference shown from the start (best, 0.854 stack), fallback manager (no gain), open prompt without the LegNet hint (-0.017 against the reference-shown arm, SE 0.007), late reveal, target sentence. None reaches the
  tuned LegNet bank (0.891) or menu v2 (0.867); at 100k the AutoResearch stacks (0.904 to 0.908) are below the tuned LegNet.

## The collaborator's architectures and recipe transplants
* `pi38_` (55 cells): five of his open-world architectures re-implemented from his descriptions (widths fitted to his stated parameter counts), two recipes each, plus 30k sweeps. Single r at 30k within about 0.01 of his stated numbers; below the tuned LegNet.
* `tr38_` (29): menu v2 and the PI architectures under the LegNet e100 recipe: members lose 0.02 to 0.07, ensembles hold at >= 30k. `ep38_` (9): menu v2 with tripled epochs and no early stopping: no change (the menu is not under-trained).

## The LegNet recipe sweeps (the main result)
* `lr38_` (49 cells over four rounds plus the seed top-up): one-knob neighbours of the published recipe, then more epochs and more weight decay, up to 800 epochs and weight decay 0.4. Per-size recipe 10k e800_wd4, 30k e500_wd2, 100k e500; 20-seed banks at every size.
  Stacks (genomic, oracle labels): 0.7801 / 0.8907 / 0.9381 against 0.731 / 0.868 / 0.931 for the published recipe and 0.7846 / 0.8674 / 0.9205 for menu v2.

## Large-scale background cells
* 300k nested baselines (48), shared-core and held-out reservoir groups, `as_` acquisition cells (90), `ind30` collaborator-ensemble cells (71), `scl30_x2` at 300k, `un10_` / `un100_` (36), `sc31_` / `sc32_` scaling cells (66): finished by Nov 1.

## Infrastructure notes
* SSH key login to bamdev4 works only while the account home `/grid/koo/home/christen` is not group-writable (memory note); a 36-hour network outage on Oct 31 to Nov 1 cost nothing because every job ran under Slurm.
* bamdev4 became unreachable on Nov 1 around 23:07 (the dispatcher and its cron live there); bamdev1, bamdev2 and bamdev3 stayed reachable. `scratchpad/dispatch_registry.json` was found truncated to 0 bytes at 23:10 and restored from `dispatch_registry.json.bak_lr38s`
  plus the nine `lr38s_` cells (2840 cells; empty file kept as `dispatch_registry.json.empty_nov1_2310`). Cause not yet identified (see the crontab on bamdev4 when it is back).
