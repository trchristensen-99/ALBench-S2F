# ALBench-S2F Experiment Tracker

> **Last updated:** 2026-04-02
> **Purpose:** Track all experiments, hyperparameters, results, and gaps.

---

## Table of Contents
1. [Data Setup](#data-setup)
2. [Model Architectures](#model-architectures)
3. [Experiment 0: Scaling / Random Subsampling](#exp0)
4. [Bar/Scatter Plot: MPRA Model Comparison](#barplot)
5. [Technique Sweep Results](#techniques)
6. [Oracle Ensembles](#oracles)
7. [Known Issues & Bugs](#bugs)
8. [Gap List / TODO](#gaps)

---

## 1. Data Setup <a name="data-setup"></a>

### K562 MPRA Dataset
- **Source:** Gosai et al. 2024 Nature, Zenodo
- **File:** `data/k562/DATA-Table_S2__MPRA_dataset.txt`
- **Total oligos:** ~798K (ref + alt alleles), ~401K (ref-only)
- **Sequence length:** 200bp variable inserts, padded to 600bp with MPRA vector flanks
- **Cell types:** K562, HepG2, SK-N-SH (3 label columns)
- **Quality filter:** max SE < 1.0, ±6σ outlier removal

### Splits

| Split | Method | ref-only | ref+alt | Test chromosomes |
|---|---|---|---|---|
| **HashFrag** | Sequence homology (SW score < 60) | train=320K, val=41K, test=41K | Same 401K (cache is ref-only) | N/A (random) |
| **Chr-based** | Chromosome holdout | train=337K, val=31K, test=33K | train=659K, val=61K, test=66K | test=chr7+13, val=chr19+21+X |

### Yeast Dataset
- **Source:** Vaishnav et al. 2022
- **Total:** ~6M sequences, 80bp
- **Splits:** random 80/10/10

---

## 2. Model Architectures <a name="model-architectures"></a>

### From-Scratch Models

| Model | Params | Architecture | Input | Key HPs |
|---|---|---|---|---|
| **LegNet** | ~2.65M | 8 EffBlocks (SE + ResidualConcat), k=5, block_sizes=[256,256,128,128,64,64,32,32] | (B, C, L) C=5 K562, C=4 yeast | lr=0.005, bs=1024, wd=0.01, dropout=0.2 |
| **DREAM-CNN** | ~1.94M | BHI dual-kernel stem (k=9,15) + 5 inverted residual blocks with SE | (B, 4, L) | lr=0.005, bs=512, wd=0.01, dropout=0.2 |
| **DREAM-RNN** | ~1.2M | Bidirectional LSTM (hidden=256) + linear head | (B, 5, L) with RC flag | lr=0.005, bs=128, wd=0.01 |
| **Malinois** | ~4.8M | Basset-based: 3 conv + BN + maxpool, branched linear heads | (B, 4, 600) padded | lr=0.00327, bs=512, wd=3.44e-4 |

### Foundation Models (Encoder + Head)

| Model | Encoder Params | Head | Embed Dim | S1 (frozen) | S2 (fine-tuned) |
|---|---|---|---|---|---|
| **AlphaGenome** | ~550M | boda-flatten-512-512 MLP | 1536 | head-only, cached embeddings | unfreeze blocks 4+5 (K562) or all (yeast) |
| **Enformer** | ~251M | MLP (512 hidden, LayerNorm) | 3072 | head-only, cached embeddings | unfreeze all transformer layers |

---

## 3. Experiment 0: Scaling / Random Subsampling <a name="exp0"></a>

**Setup:** HashFrag splits, ref-only (~400K pool), random reservoir sampling at 7 sizes (K562) or 10 sizes (yeast), 3 seeds per HP config.

### K562 Exp0 Completeness

| Student | AG oracle (default) | DRNN oracle | Ground truth | Status |
|---|---|---|---|---|
| LegNet | 63 results (6/7 sizes) | 62 results (5/7) | 62 results (6/7) | **RUNNING** |
| DREAM-CNN | 31 results ✅ | 36 results ✅ | 26 results (4/7) | GT **RUNNING** |
| DREAM-RNN | 30 results ✅ | 36 results ✅ | 22 results (3/7) | GT **RUNNING** |
| AG S1 | 72 results ✅ | 72 results ✅ | — | Not needed (cached) |
| AG S2 cold | 21 results ✅ | — | — | Has dropout bug |
| AG S2 warm | 21 results ✅ | — | — | Fixed warm start |

**HP configs used:**
- LegNet: lr=[0.001, 0.005], bs=[512, 1024] — best: lr=0.005, bs=1024
- DREAM-CNN: lr=[0.005], bs=[512, 1024] — **only 1 LR tested, sweep running**
- DREAM-RNN: lr=[0.005], bs=[128, 512]
- AG S1: lr=[3e-4, 1e-3], bs=[128, 256]

**Result dirs:** `outputs/exp0_oracle_scaling_v4/k562/{model}/random/n{size}/hp{idx}/seed{seed}/result.json`

### Yeast Exp0 Completeness

| Student | DRNN oracle (default) | AG oracle | Ground truth | Status |
|---|---|---|---|---|
| LegNet | 48 results (4/10) | 44 results (3/10) | 57 results (7/10) | **RUNNING** |
| DREAM-CNN | 40 results ✅ | 40 results ✅ | 28 results (3/10) | GT **RUNNING** |
| DREAM-RNN | 41 results ✅ | 42 results ✅ | 25 results (2/10) | GT **RUNNING** |
| AG S1 | 84 results ✅ | 84 results ✅ | — | |
| AG S2 cold | 86 results (9/10) | 83 results ✅ | — | Missing n=6M |
| AG S2 warm | 0/10 | — | — | **RUNNING** (shape fix applied) |
| AG S2 hlr | 30 results ✅ | — | — | lr=1e-3 variant |

**Result dirs:** `outputs/exp0_oracle_scaling_v4/yeast/{model}/random/n{size}/hp{idx}/seed{seed}/result.json`

### Plots
- `results/exp0_scaling_plots/k562_scaling_2x2.png` — all models, oracle labels
- `results/exp0_scaling_plots/yeast_scaling_2x2.png` — all models, oracle labels
- `results/exp0_scaling_plots/{task}_cross_oracle.png` — cross-oracle comparison
- `results/exp0_scaling_plots/{task}_oracle_vs_real.png` — oracle vs real labels
- **Script:** `scripts/analysis/plot_exp0_scaling_curves.py`

---

## 4. Bar/Scatter Plot: MPRA Model Comparison <a name="barplot"></a>

**Setup:** Chr-based splits, ref+alt (~786K), weighted average across K562/HepG2/SknSh, OOD = K562-only (synthetic seqs only have K562 labels).

### Chr_split Results (v2 = ref+alt preferred)

| Model | K562 | HepG2 | SknSh | Predictions? |
|---|---|---|---|---|
| Malinois | 3 seeds (v2) ✅ | 3 seeds (v2) ✅ | 1 seed (v2) + 2 **RUNNING** | No |
| LegNet | 2 seeds **RUNNING** (fast) | 2 seeds **RUNNING** (slow) | 2 seeds **RUNNING** (slow) | No |
| DREAM-RNN | 1 seed (v2) + 1 **RUNNING** | 2 seeds **RUNNING** | 2 seeds **RUNNING** | No |
| Enf. S1 | 3 seeds (v1 ref-only) | 3 seeds (v1) | 3 seeds (v1) | No |
| Enf. S1 v2 | **REBUILDING** cache | **REBUILDING** | **REBUILDING** | No |
| Enf. S2 | **MISSING** (needs S1) | **MISSING** | **MISSING** | No |
| AG S1 | 1 seed (v2) ✅ | 1 seed (v2) ✅ | 1 seed (v2) ✅ | No |
| AG S2 warm | **RUNNING** (fast) | **PENDING** (default) | **PENDING** (default) | No |
| AG S2 cold | 1 seed (v1 ref-only) | 1 (v1) | 1 (v1) | No |

**Result dirs:** `outputs/chr_split_v2/{cell}/{model}/...`
**Fallback dirs:** `outputs/chr_split/{cell}/{model}/...` (old v1 ref-only)

### Key Bar Plot Numbers (current)

| Model | Reference | SNV Effect | Synthetic Seqs |
|---|---|---|---|
| Malinois | 0.830 | 0.337 | 0.500 |
| DREAM-RNN | 0.836 | 0.305 | 0.404 |
| Enf. (Probing) | 0.862 | 0.295 | 0.331 |
| AG (Probing) | 0.878 | 0.355 | 0.707 |
| AG (Fine-tuned) | 0.869 (cold!) | 0.343 | 0.641 |

### Colors (from PI meeting notes)
- Malinois/LegNet: `#E8DCCF` (baseline beige) / `#D4A017` (gold)
- DREAM-RNN: `#8B9DAF` (blue-gray)
- Enf. Probing: `#E7CDC2`, Fine-tuned: `#A65141`
- AG Probing: `#80A0C7`, Fine-tuned: `#394165`

### Plots
- `results/alan_style_plots/mpra_benchmark_chr_split.png`
- **Script:** `scripts/analysis/plot_alan_style_barplot.py`

---

## 5. Technique Sweep Results <a name="techniques"></a>

All on K562 chr_split, ref+alt.

| Technique | Malinois | DREAM-RNN | AG S1 |
|---|---|---|---|
| **Baseline (RC only)** | 0.835 | 0.822 | 0.881 |
| **+ Shift (±15bp)** | **0.839** (+0.4%) | **0.830** (+0.8%) | — |
| **+ High-activity dup** | 0.838 (+0.3%) | **0.852** (+3.0%) | 0.884 (+0.3%) |
| + Shift + dup | — | 0.817 (worse!) | — |
| + RC interleave | 0.836 (≈0) | — | — |
| + Cosine LR | 0.838 (≈0) | — | — |
| No augmentation | 0.827 (-0.8%) | — | — |

**Key finding:** Duplication is the biggest win for DREAM-RNN (+3%). Shift helps modestly. Combining shift+dup hurts DREAM-RNN (overfitting?).

**Result dirs:** `outputs/aug_sweep/`, `outputs/techniques_sweep/`, `outputs/multitask/`

---

## 6. Oracle Ensembles <a name="oracles"></a>

| Oracle | Location | Type | Seeds/Folds |
|---|---|---|---|
| AG S1 K562 (10-fold) | `outputs/ag_hashfrag_oracle_cached/oracle_{0-9}` | k-fold, out-of-fold predictions | 10 folds |
| DREAM-RNN K562 | `outputs/oracle_dream_rnn_k562_ensemble/` | 10 oracle folds + 5 ensemble runs | 10+5 |
| AG S2 K562 | `outputs/stage2_k562_full_train/run_{0-2}` | 3 fine-tuned runs | 3 |
| DREAM-RNN Yeast | `outputs/oracle_dream_rnn_yeast_ensemble/` | 5 ensemble runs | 5 |
| **LegNet K562** | — | **NEEDS K-FOLD IMPLEMENTATION** | — |

---

## 7. Known Issues & Bugs <a name="bugs"></a>

| Issue | Status | Impact | Details |
|---|---|---|---|
| AG S2 cold start | **FIXED** | S2 < S1 on real labels | Was reinitializing head. Now copies S1 head weights. |
| AG S2 dropout | **WORKAROUND** | Small (~0.1-0.5%) | alphagenome_ft doesn't pass is_training; editable install caches old code |
| OOD label mismatch | **FIXED** | HepG2/SknSh OOD was ~0 | OOD seqs only have K562 labels; now K562-only for OOD |
| Foundation S1 chr_split test | **FIXED** | Wrong test set | Was using hashfrag test, now uses chr7+13 |
| HashFrag include_alt_alleles | **KNOWN** | No effect | HashFrag cache was built ref-only; only chr-based supports ref+alt |
| orbax checkpoint save | **FIXED** | S1 checkpoints not saving | Needed absolute paths + parent-only mkdir |
| Yeast S2 warm shape mismatch | **FIXED** | Output layer 1 vs 18 | Now skips layers with mismatched shapes |

---

## 8. Gap List / TODO <a name="gaps"></a>

### Priority 1 (Currently Running)
- [ ] LegNet K562 Exp0 final 1-2 sizes
- [ ] LegNet Yeast Exp0 oracle runs (6/10 and 7/10 sizes remaining)
- [ ] Yeast S2 warm-start
- [ ] Ground truth scaling (6 jobs: CNN+RNN+LegNet × K562+Yeast)
- [ ] AG S2 warm chr_split K562/HepG2/SknSh (bar plot)
- [ ] DREAM-CNN LR sweep (lr=0.001, 0.01) K562+Yeast
- [ ] LegNet chr_split HepG2+SknSh (bar plot)
- [ ] DREAM-RNN chr_split HepG2+SknSh v2 (ref+alt)
- [ ] Malinois SknSh additional seeds
- [ ] HP probe (LegNet+DREAM-CNN at n=32K with new LRs)
- [ ] Enformer S1 v2 cache rebuild (3 cells)

### Priority 2 (Submit After Priority 1)
- [ ] Enformer S2 chr_split (needs S1 v2 cache)
- [ ] LegNet k-fold oracle ensemble (needs implementation)
- [ ] Prediction saving pass for scatter plots (all bar plot models)
- [ ] Ensemble comparison (ensemble_size=1 vs 3)

### Priority 3 (Nice to Have)
- [ ] AG S1/S2 ground truth Exp0 (needs custom pipeline)
- [ ] AG S2 DRNN oracle K562 Exp0
- [ ] Proper alphagenome_ft dropout fix (reinstall non-editable)
- [ ] AG multi-task debug (produces 0.005 Pearson)
- [ ] Cross-model prediction correlation analysis
- [ ] LegNet architecture sweep (4-block vs 8-block)
