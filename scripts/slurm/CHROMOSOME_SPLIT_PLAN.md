# Chromosome Split Experiment Plan

## Overview
Re-run all 7 models on the chromosome-based train/test split for K562 to enable
comparison between HashFrag (homology-aware) and chromosome holdout splits.

- **Test**: chr7 + chr13 (~62K sequences)
- **Val**: chr19 + chr21 + chrX (~33K sequences)
- **Train**: All other chromosomes (~306K sequences)

## Key Insight
The sequences are identical between HashFrag and chr splits — only the split
assignment changes. Foundation model embedding caches can be reused.

## Phase 1: Create chromosome split indices

```bash
# Create chr-split index files mapping into the full dataset
python -c "
from data.k562 import K562Dataset
import numpy as np

# Load chr-split dataset
ds_train = K562Dataset('data/k562', split='train', use_hashfrag=False, use_chromosome_fallback=True)
ds_val = K562Dataset('data/k562', split='val', use_hashfrag=False, use_chromosome_fallback=True)
ds_test = K562Dataset('data/k562', split='test', use_hashfrag=False, use_chromosome_fallback=True)

print(f'Chr train: {len(ds_train)} val: {len(ds_val)} test: {len(ds_test)}')
"
```

## Phase 2: From-scratch models (need retraining)

- DREAM-RNN K562 (3 seeds)
- DREAM-CNN K562 (3 seeds)
- Malinois K562 (3 seeds)

Use `exp1_1_scaling.py --oracle ground_truth --reservoir genomic` with chr-split data.

## Phase 3: Foundation S1 (reuse cached embeddings, retrain heads)

Embeddings are the same — just need to re-index the cache arrays by chr-split indices.

- Enformer S1 (3 seeds)
- Borzoi S1 (3 seeds)
- NTv3-post S1 (3 seeds)

## Phase 4: AlphaGenome S1 + S2

- AG fold-1 S1 (1 seed)
- AG all-folds S1 (1 seed)
- AG all-folds S2 with best K562 config (1 seed)

## Phase 5: Test evaluation

Each model needs to be evaluated on chr7+13 test set with:
- In-dist (reference) Pearson R
- SNV pairs (if available for chr7+13)
- SNV delta
- OOD (designed sequences — same for both splits)

## Output
Results in `outputs/chr_split/{model}/` with standard result.json format.
