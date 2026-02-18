# ALBench-S2F Architecture Guide

> **Purpose**: This document gives any human or AI agent everything needed to understand, modify, and run this codebase. Read it before touching any code.

## 1. What Is ALBench-S2F?

ALBench-S2F (Active Learning Benchmark — Sequence-to-Function) is a benchmarking platform for comparing **active learning strategies** on genomic sequence-to-function prediction tasks. It supports six registered-report experiments that evaluate how different reservoir sampling strategies (how sequences are *generated*) and acquisition functions (how sequences are *selected*) affect model performance when labeling budgets are limited.

### Scientific Context

- **Problem**: Training genomic models requires labeled data (e.g., MPRA assays) that is expensive to generate. Active learning asks: *which sequences should we label next to learn the most?*
- **Organisms**: Human (K562 lentiMPRA) and Yeast (random promoter MPRA)
- **Oracle**: AlphaGenome (frozen encoder + probing head) provides in-silico labels
- **Student**: DREAM-RNN (CNN + BiLSTM) learns from labeled data and estimates uncertainty
- **Goal**: Identify optimal (reservoir, acquisition) strategy combinations, then synthesize the winning sequences for real lentiMPRA experiments

### Experiment Overview

| Exp | Name | Purpose |
|-----|------|---------|
| 0 | Scaling Curves | How does performance scale with dataset size? Defines small/medium/large data regimes |
| 1 | Strategy Benchmark | Full grid of reservoir × acquisition combinations across regimes |
| 2 | Round Structure | Multiple small AL rounds vs. single large batch |
| 3 | Pool Size | How does acquisition function performance change with reservoir pool size? |
| 4 | Cost-Adjusted | Re-rank strategies by performance per unit computational cost |
| 5 | Best Student | Using best model + best strategies to select sequences for real synthesis |

**Build order** (from migration plan §11): Data pipeline → AlphaGenome oracle → Exp 0 → DREAM-RNN student → Random baseline AL loop → Uncertainty acquisition → Additional reservoir/acquisition strategies → Exp 2–5.

---

## 2. Repository Structure

```
ALBench-S2F/
├── albench/                    # Core library package
│   ├── model.py                # SequenceModel ABC (predict, uncertainty, embed, fit)
│   ├── task.py                 # TaskConfig dataclass
│   ├── loop.py                 # run_al_loop() — AL round driver with schedule dispatch
│   ├── evaluation.py           # evaluate_on_test_sets(), compute_scaling_curve()
│   ├── data/                   # Data loading and preprocessing
│   │   ├── base.py             # SequenceDataset ABC
│   │   ├── k562.py             # K562Dataset — human lentiMPRA (200bp, 5ch)
│   │   ├── yeast.py            # YeastDataset — yeast random promoter (150bp, 6ch)
│   │   ├── hashfrag_splits.py  # HashFragSplitter — homology-aware train/val/test splits
│   │   ├── pool.py             # Unlabeled pool management
│   │   └── utils.py            # one_hot_encode, reverse_complement, pad_sequence, etc.
│   ├── models/                 # Neural network architectures
│   │   ├── base.py             # SequenceModel nn.Module base
│   │   ├── dream_rnn.py        # DREAMRNN + create_dream_rnn() factory
│   │   ├── training.py         # train_model_optimized() — main training loop with W&B
│   │   ├── training_base.py    # create_optimizer_and_scheduler()
│   │   └── loss_utils.py       # YeastKLLoss (18-bin KL divergence)
│   ├── oracle/                 # Oracles provide ground-truth labels
│   │   ├── perfect_oracle.py   # PerfectOracle — lookup table from real labels
│   │   └── alphagenome.py      # AlphaGenomeOracle — stub (TODO: integrate via probing)
│   ├── student/                # Student models learn from labeled data
│   │   └── dream_rnn_student.py # DREAMRNNStudent — ensemble wrapper with MC dropout
│   ├── reservoir/              # Reservoir samplers generate candidate sequences
│   │   ├── base.py             # ReservoirSampler ABC
│   │   ├── random_sampler.py   # RandomSampler — uniform random selection
│   │   ├── genomic.py          # GenomicSampler — chromosome-restricted selection
│   │   ├── evoaug.py           # stub
│   │   ├── tf_motif_shuffle.py # stub
│   │   └── partial_mutagenesis.py # stub
│   └── acquisition/            # Acquisition functions select from candidate pool
│       ├── base.py             # AcquisitionFunction ABC
│       ├── random_acq.py       # RandomAcquisition
│       ├── uncertainty.py      # UncertaintyAcquisition — top-k by MC dropout variance
│       ├── diversity.py        # DiversityAcquisition — greedy farthest-point in embedding space
│       ├── combined.py         # CombinedAcquisition — weighted uncertainty + diversity
│       └── prior_knowledge.py  # stub
├── experiments/                # Hydra entry-point scripts
│   ├── exp0_scaling.py         # Experiment 0: scaling curves
│   ├── exp1_benchmark.py       # Experiment 1: strategy benchmark (--multirun)
│   └── exp2–5 stubs            # Experiments 2–5 (TODO)
├── configs/                    # Hydra YAML configs
│   ├── config.yaml             # Root config with defaults
│   ├── task/                   # k562.yaml, yeast.yaml
│   ├── student/                # dream_rnn.yaml
│   ├── acquisition/            # random.yaml, uncertainty.yaml
│   └── reservoir/              # random.yaml, genomic.yaml
├── scripts/
│   ├── download_data.py        # Dataset download utility
│   ├── setup_runtime.sh        # One-shot bootstrap (uv sync + torch auto-config)
│   ├── run_with_runtime.sh     # Wrapper to run without uv resync
│   ├── auto_configure_torch.py # Detects CUDA and installs correct torch wheel
│   └── slurm/
│       └── train_koo.sh        # SLURM batch template for Koo Lab HPC
├── tests/                      # pytest test suite
│   ├── test_data.py            # Data utility tests (one-hot, RC, padding)
│   ├── test_interfaces.py      # ABC contract tests (SequenceModel, Sampler, Acq)
│   ├── test_dataset_loaders.py # Dataset instantiation smoke tests
│   └── test_loop.py            # AL loop schedule dispatch and bookkeeping
├── pyproject.toml              # Package config, deps, ruff, pytest
├── .pre-commit-config.yaml     # ruff + ruff-format hooks
├── .gitignore                  # data/, checkpoints/, outputs/, wandb/, .env
└── README.md                   # Quick-start guide
```

---

## 3. Core Abstractions

### 3.1 SequenceModel ([model.py](file:///Users/christen/Downloads/ALBench-S2F/albench/model.py))

All oracles and students implement this interface:

| Method | Signature | Notes |
|--------|-----------|-------|
| `predict` | `(sequences: list[str]) → np.ndarray(N,)` | **Required**. Scalar activity predictions |
| `uncertainty` | `(sequences: list[str]) → np.ndarray(N,)` | MC dropout variance (N=30 passes) |
| `embed` | `(sequences: list[str]) → np.ndarray(N, D)` | 256-d pooled conv3 output before FC |
| `fit` | `(sequences: list[str], labels: np.ndarray) → None` | Retrain on labeled data |

### 3.2 ReservoirSampler ([reservoir/base.py](file:///Users/christen/Downloads/ALBench-S2F/albench/reservoir/base.py))

Generates or selects candidate sequences for the acquisition step. See migration plan §5.3.

### 3.3 AcquisitionFunction ([acquisition/base.py](file:///Users/christen/Downloads/ALBench-S2F/albench/acquisition/base.py))

Selects the most informative sequences from the candidate pool. See migration plan §5.4.

### 3.4 TaskConfig ([task.py](file:///Users/christen/Downloads/ALBench-S2F/albench/task.py))

Dataclass holding all task-specific configuration: organism, sequence length, data paths, test sets, flanking sequences, etc.

### 3.5 AL Loop ([loop.py](file:///Users/christen/Downloads/ALBench-S2F/albench/loop.py))

`run_al_loop()` implements the core active learning cycle:
1. Train student on initial labeled set
2. For each round:
   - Reservoir generates candidate pool
   - Acquisition selects top-k from pool
   - Oracle labels selected sequences
   - Student retrains on expanded labeled set
   - Evaluate on test sets, log to W&B

Schedule dispatch allows different reservoir/acquisition strategies per round.

---

## 4. Data Pipelines

### 4.1 K562 (Human lentiMPRA)

**Source**: Gosai et al., Nature 2023 ([Zenodo](https://zenodo.org/records/10698014))

| Property | Value |
|----------|-------|
| File | `DATA-Table_S2__MPRA_dataset.txt` |
| Filter | `allele_type == 'R'` OR `(ref == 'NA' AND alt == 'NA')` from IDs field `chr:pos:ref:alt:type:wc` |
| Length filter | ≥ 198bp |
| Standardization | Center-pad with Ns to exactly 200bp |
| Label column | `K562_log2FC` |
| Channels | 5: ACGT one-hot (4) + reverse complement flag (1, always 0 during training) |
| Splits | HashFrag homology-aware 80:10:10 → first 100K of 80% = train, remainder = pool |
| Split caching | `.npy` index files in `data/k562/hashfrag_splits/` |

### 4.2 Yeast (Random Promoter MPRA)

| Property | Value |
|----------|-------|
| Train file | Standard format |
| **Test file** | `filtered_test_data_with_MAUDE_expression.txt` (NOT `test.txt`) |
| Random region | 80bp |
| Flanking | `FLANK_5_PRIME = 'GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATCTCG'` |
| | `FLANK_3_PRIME = 'GGTTACGGCTGTT'` |
| Preprocessing | Strip partial 5' flank (last 17bp = `TGCATTTTTTTCACATC`) before re-adding full flanks |
| Total length | 150bp (padded) |
| Channels | 6: ACGT (4) + RC flag (1) + singleton flag (1, `expression % 1 == 0`) |
| Loss | 18-bin softmax → weighted average, KL divergence loss |
| Training | Use `model.get_logits()` for loss, `model()` (forward) for metrics |

### 4.3 HashFrag Splits

HashFrag prevents data leakage from homologous sequences:
1. **Two-pass**: First split (train+val) vs test, then split train vs val
2. **Requires**: BLAST+ and HashFrag executable on PATH
3. **Runtime**: Several hours for full K562 dataset (~367K sequences)
4. **Caching**: Results cached as `.npy` index files; once generated, splits load instantly
5. **Determinism**: Use a fixed seed for the train/pool random shuffle to ensure reproducibility across Citra and HPC

---

## 5. Model Architecture: DREAM-RNN

From [dream_rnn.py](file:///Users/christen/Downloads/ALBench-S2F/albench/models/dream_rnn.py):

```
Input: (B, C, L)  where C = input_channels, L = sequence_length
    ↓
Dual CNN Block 1:
    Conv1d(C, 160, kernel_size=9, padding='same')  ─┐
    Conv1d(C, 160, kernel_size=15, padding='same') ─┤→ cat → (B, 320, L) → ReLU → Dropout(0.1)
    ↓
BiLSTM: (B, L, 320) → hidden_size=320/direction → (B, L, 640)
    ↓
Dual CNN Block 2:
    Conv1d(640, 160, kernel_size=9)  ─┐
    Conv1d(640, 160, kernel_size=15) ─┤→ cat → (B, 320, L) → ReLU → Dropout(0.1)
    ↓
Conv1d(320, 256, kernel_size=1) → ReLU → Global Average Pool → (B, 256)
    ↓
Linear(256, output_dim)   # output_dim=1 for K562, 18 for yeast
```

**MC Dropout** for uncertainty: `dropout_cnn=0.1, dropout_lstm=0.1`, 30 forward passes with `model.train()`, variance across passes = uncertainty.

**Embedding**: 256-dim output of conv3 after global average pooling, **before** the FC layer.

### DREAMRNNStudent Ensemble

[dream_rnn_student.py](file:///Users/christen/Downloads/ALBench-S2F/albench/student/dream_rnn_student.py) wraps an ensemble (`ensemble_size=3` default):
- `predict()`: Mean across ensemble members (each in eval mode)
- `uncertainty()`: Mean of per-member MC dropout variance (30 passes each, train mode)
- `embed()`: Mean of per-member pooled conv3 features
- `fit()`: Retrain all members via `train_model_optimized()`

---

## 6. AlphaGenome Oracle (Probing Setup)

The AlphaGenome oracle will use a **frozen encoder + custom probing head**, not full fine-tuning (too expensive). Based on [`alphagenome_ft`](file:///Users/christen/Downloads/alphagenome_ft-main) package:

- **Architecture**: `EncoderOnlyHead` (CNN features only, for short sequences < 1kb)
- **Head configs to try**: Pool-flatten vs MLP variants (512→256, 512→512)
- **Framework**: JAX/Haiku (separate from PyTorch student models)
- **Status**: Stub at `albench/oracle/alphagenome.py` — needs integration

---

## 7. Compute Environments

> **Rule**: NEVER run training or data downloads on the local Mac. Local = CPU-only smoke tests.

### Local Mac (CPU-only)

```bash
cd ~/Downloads/ALBench-S2F
uv sync --extra dev
uv run pytest tests/ -v
uv run python -c "from albench.data.k562 import K562Dataset; print('OK')"
```

### Citra (Dev GPU Runs)

```bash
ssh trevorch@143.48.59.3
cd ~/ALBench-S2F
bash scripts/setup_runtime.sh
bash scripts/run_with_runtime.sh python experiments/exp0_scaling.py \
    +task=k562 +student=dream_rnn experiment.dry_run=true
```

### Koo Lab HPC (Full Experiments)

```bash
ssh trevorch@bamdev4.cshl.edu
cd /grid/koo/data/trevorch/ALBench-S2F
sbatch scripts/slurm/train_koo.sh
```

| Resource | Value |
|----------|-------|
| Partition | `kooq` |
| QoS | `koolab` |
| Node | `bamgpu101` |
| GPUs | 4× H100 NVL |
| Max wall time | 30 days |

### CSHL HPC (Alternative — `gpuq` partition)

```bash
ssh christen@bamdev4.cshl.edu
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
sbatch scripts/slurm/train_cshl.sh  # Uses gpuq partition
```

---

## 8. W&B Integration

- **Project**: `albench-s2f`
- **API key**: stored in `.env` (never committed); use `wandb login` on remote hosts
- **Metric axis**: `wandb.define_metric('test/pearson_r', step_metric='n_labeled')`
- **Per-epoch**: loss, pearson_r, spearman_r, lr (logged in `train_model_optimized()`)
- **Per-round**: n_labeled, test/{name}/pearson_r (logged in `run_al_loop()`)
- **Artifacts**: Model checkpoints saved as W&B Artifacts

---

## 9. Hydra Configuration

All experiments use `@hydra.main()`. Config composition:

```yaml
# configs/config.yaml (root)
defaults:
  - acquisition: random
  - reservoir: random
  - _self_

experiment:
  name: exp0_scaling
  dry_run: true
  n_rounds: 2
  batch_size: 8
  ...
seed: 42
wandb:
  project: albench-s2f
  mode: online
```

Override syntax:
```bash
uv run python experiments/exp0_scaling.py +task=k562 +student=dream_rnn experiment.dry_run=false
```

Task and student configs use `+` (append) because they're not in the root defaults — they're added per-experiment.

---

## 10. Engineering Standards Quick Reference

| Area | Rule |
|------|------|
| Package mgr | `uv` exclusively; `uv sync`, `uv run python ...` |
| Python | ≥ 3.11 |
| Commits | Conventional Commits format |
| Linting | ruff (E, F, I rules; see pyproject.toml) |
| Types | Full type annotations on all public functions |
| Docstrings | Google-style on all public classes/functions |
| Tests | `pytest tests/` — at least one test per interface class |
| Seeds | Every script accepts `--seed`; set all random seeds at start |
| Config | Hydra only (no argparse anywhere) |

---

## 11. Known TODOs and Current Status

### Implemented ✅
- Data pipeline (K562, Yeast, HashFrag)
- DREAM-RNN model + student ensemble wrapper
- PerfectOracle
- Random/Uncertainty/Diversity/Combined acquisition
- Random/Genomic reservoir sampling
- AL loop with schedule dispatch
- Evaluation + scaling curve utilities
- Exp 0 scaling script (dry-run mode works)
- Hydra configs, SLURM templates, W&B integration
- Pre-commit hooks (ruff)

### In Progress 🔧
- Correction pass for interface signatures (reservoir/acquisition)
- K562 100K train/pool split fix
- AL loop pool semantics alignment
- Exp 0 non-dry-run dataset wiring

### Not Yet Started ❌
- AlphaGenome oracle (probing head integration from alphagenome_ft)
- Dinucleotide-shuffled reservoir sampling
- EvoAug, TF motif shuffle, partial mutagenesis reservoir strategies
- BADGE, BatchBALD, prior knowledge acquisition strategies
- Experiments 1–5 full implementation
- Citra/HPC end-to-end validation
