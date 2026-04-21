#!/bin/bash
# Generate 3M additional evoaug_prior sequences and append to 2M pool.
# Result: 5M pool at outputs/labeled_pools_5m/k562/ag_s2/evoaug_prior/pool.npz
#
#SBATCH --job-name=gen_eap
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export ALPHAGENOME_WEIGHTS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"

OUT_DIR="outputs/labeled_pools_5m/k562/ag_s2/evoaug_prior"
POOL_FILE="${OUT_DIR}/pool.npz"
EXISTING_2M="outputs/labeled_pools_2m/k562/ag_s2/evoaug_prior/pool.npz"

[ -f "${POOL_FILE}" ] && echo "SKIP: 5M pool already exists" && exit 0

echo "=== Generating evoaug_prior 5M pool — $(date) ==="

# Strategy: generate 3M new sequences, label them, combine with existing 2M
uv run --no-sync python << PYEOF
import numpy as np, os, sys, time
from pathlib import Path
sys.path.insert(0, ".")

REPO = Path(".")
out_dir = Path("${OUT_DIR}")
out_dir.mkdir(parents=True, exist_ok=True)
existing = Path("${EXISTING_2M}")

# Load existing 2M
print("Loading existing 2M pool...")
data = np.load(existing, allow_pickle=True)
seqs_2m = list(data["sequences"])
labels_2m = data["labels"]
print(f"  Existing: {len(seqs_2m)} seqs")

# Generate 3M new sequences
print("Generating 3M new evoaug_prior sequences...")
from albench.reservoir.evoaug_structural import EvoAugStructuralSampler

sampler = EvoAugStructuralSampler(
    seed=12345,
    p_deletion=0.33, p_insertion=0.33, p_inversion=0.15,
    p_translocation=0.33, p_tandem_dup=0.10, p_point_mutation=0.30,
    max_indel_size=30, max_inversion_size=40, max_translocation_size=50,
    max_dup_size=20, point_mutation_rate=0.05,
    min_events=1, max_events=3,
)

# Generate from genomic base sequences
from data.k562 import K562Dataset
ds = K562Dataset(data_path="data/k562", split="train")
base_seqs = [ds[i][0] for i in range(min(len(ds), 50000))]
# Convert tensors to strings
base_seq_strs = []
mapping = {0: "A", 1: "C", 2: "G", 3: "T"}
for t in base_seqs:
    seq = ""
    for i in range(t.shape[1]):
        for j in range(4):
            if t[j, i] > 0.5:
                seq += mapping[j]
                break
        else:
            seq += "N"
    base_seq_strs.append(seq)

new_seqs = sampler.generate(3_000_000, pool_sequences=base_seq_strs)
print(f"  Generated: {len(new_seqs)} seqs")

# Label with AG S2 oracle
print("Labeling 3M sequences with AG S2 oracle...")
os.environ["ALPHAGENOME_WEIGHTS"] = "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"

from scripts.generate_labeled_pools import _label_sequences_batched

# Load oracle
from alphagenome_ft import create_model_with_heads
from models.alphagenome_heads import register_s2f_head
from models.embedding_cache import reinit_head_params

head_name = "alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4"
register_s2f_head(head_name=head_name, arch="boda-flatten-512-512", task_mode="human", num_tracks=1)

model = create_model_with_heads(
    "all_folds", heads=[head_name],
    checkpoint_path=os.environ["ALPHAGENOME_WEIGHTS"],
    use_encoder_output=True, detach_backbone=True,
)
reinit_head_params(model, head_name, num_tokens=5, dim=1536, rng=42)

# Load best oracle checkpoint
import orbax.checkpoint as ocp, jax
ckpt = REPO / "outputs" / "ag_hashfrag_oracle_cached" / "oracle_0" / "best_model" / "checkpoint"
if ckpt.exists():
    checkpointer = ocp.StandardCheckpointer()
    loaded_params, _ = checkpointer.restore(str(ckpt.resolve()))
    model._params = jax.device_put(loaded_params)
    print("  Loaded oracle weights")

t0 = time.time()
new_labels = _label_sequences_batched(model, new_seqs, batch_size=2048)
print(f"  Labeling done in {time.time()-t0:.0f}s")

# Combine
all_seqs = np.array(seqs_2m + new_seqs)
all_labels = np.concatenate([labels_2m, new_labels])
print(f"Combined: {len(all_seqs)} seqs, label_mean={all_labels.mean():.3f}")

# Save
np.savez_compressed(out_dir / "pool.npz", sequences=all_seqs, labels=all_labels)
print(f"Saved: {out_dir / 'pool.npz'}")
PYEOF

echo "=== DONE — $(date) ==="
