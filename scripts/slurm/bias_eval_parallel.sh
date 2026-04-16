#!/bin/bash
# Parallel bias evaluation: one job per model.
#
# Evaluates each model on random DNA, CpG-depleted, shuffled, intergenic,
# Gosai ctrl_neg with real labels.
#
# Array tasks map to model directories:
#   0-11:  debias_sweep configs
#   12-26: top old oracle_neg_sweep configs (OOD > 0.74, no bias_eval yet)
#
#SBATCH --job-name=biasevl
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export ALPHAGENOME_WEIGHTS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"

T=$SLURM_ARRAY_TASK_ID

# Build list of model directories
DEBIAS_DIRS=($(ls -d outputs/debias_sweep/*/test_metrics.json 2>/dev/null | sort | xargs -I{} dirname {}))
# Old neg_aug dirs with test_metrics but no bias_eval, sorted by OOD desc
OLD_DIRS=($(python3 -c "
import json, glob, os
results = []
for f in sorted(glob.glob('outputs/oracle_neg_sweep/*/test_metrics.json') + glob.glob('outputs/oracle_neg_sweep/*/fold_0/test_metrics.json')):
    d = os.path.dirname(f)
    if os.path.exists(os.path.join(d, 'bias_eval.json')):
        continue
    try:
        tm = json.load(open(f))
        ood = tm.get('test_metrics', {}).get('ood', {}).get('pearson_r', 0)
        if ood > 0.74:
            results.append((ood, d))
    except: pass
results.sort(key=lambda x: -x[0])
for _, d in results[:15]:
    print(d)
"))

ALL_DIRS=("${DEBIAS_DIRS[@]}" "${OLD_DIRS[@]}")
N_TOTAL=${#ALL_DIRS[@]}

if [ $T -ge $N_TOTAL ]; then
    echo "Task $T out of range ($N_TOTAL models)"
    exit 0
fi

MODEL_DIR="${ALL_DIRS[$T]}"
echo "=== Bias eval: ${MODEL_DIR} (task $T/$N_TOTAL) — $(date) ==="

[ -f "${MODEL_DIR}/bias_eval.json" ] && echo "SKIP: bias_eval.json exists" && exit 0

uv run --no-sync python << PYEOF
import json, os, sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from pathlib import Path
import jax, jax.numpy as jnp

REPO = Path(".")
model_dir = Path("${MODEL_DIR}")

# Load test_metrics to get head name
tm = json.load(open(model_dir / "test_metrics.json"))
head_name = tm.get("head_name", "alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4")

print(f"Loading model with head: {head_name}")
from alphagenome_ft import create_model_with_heads
model = create_model_with_heads(
    "all_folds", heads=[head_name],
    checkpoint_path=os.environ["ALPHAGENOME_WEIGHTS"],
    use_encoder_output=False,
)

# Load fine-tuned weights
best_path = model_dir / "best_model"
if best_path.exists():
    import orbax.checkpoint as ocp
    checkpointer = ocp.StandardCheckpointer()
    try:
        restored = checkpointer.restore(str(best_path))
        model._params = restored.get("params", model._params)
        print(f"Loaded fine-tuned weights from {best_path}")
    except Exception as e:
        print(f"WARNING: could not load checkpoint: {e}")
        print("Using base model weights")

@jax.jit
def predict_step(params, state, sequences):
    preds = model._predict(
        params, model._state, sequences,
        jnp.zeros(len(sequences), dtype=jnp.int32),
        negative_strand_mask=jnp.zeros(len(sequences), dtype=bool),
        strand_reindexing=None,
        requested_outputs=[head_name],
    )[head_name]
    return jnp.squeeze(preds, axis=-1) if preds.ndim > 1 else preds

_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
def _predict_seqs(seqs, batch_size=64):
    all_preds = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size]
        ohe = np.zeros((len(batch), 600, 4), dtype=np.float32)
        for j, seq in enumerate(batch):
            for k, c in enumerate(seq[:200].upper()):
                if c in _MAP:
                    ohe[j, 200+k, _MAP[c]] = 1.0
        preds = predict_step(model._params, model._state, jnp.array(ohe))
        all_preds.append(np.array(preds).flatten())
    return np.concatenate(all_preds)

bias_results = {}
rng = np.random.RandomState(42)

# 1. Random DNA (500 seqs)
print("Evaluating random DNA...")
random_seqs = ["".join("ACGT"[i] for i in rng.randint(0, 4, size=200)) for _ in range(500)]
rp = _predict_seqs(random_seqs)
bias_results["random_dna"] = {"mean": float(np.mean(rp)), "std": float(np.std(rp)),
    "pct_positive": float(np.mean(rp > 0)*100), "n": len(rp)}

# 2. CpG-depleted random
cpg_dep = [s.replace("CG", "TG") for s in random_seqs[:200]]
cd_p = _predict_seqs(cpg_dep)
bias_results["cpg_depleted_random"] = {"mean": float(np.mean(cd_p)), "std": float(np.std(cd_p)),
    "pct_positive": float(np.mean(cd_p > 0)*100), "n": len(cd_p)}

# 3. Dinuc-shuffled
def dinuc_shuf(seq, r):
    s = list(seq.upper())
    for i in range(len(s)-2, 0, -1):
        j = r.randint(0, i+1)
        s[i], s[j] = s[j], s[i]
    return "".join(s)
ds = [dinuc_shuf(s, np.random.RandomState(42+i)) for i, s in enumerate(random_seqs[:200])]
ds_p = _predict_seqs(ds)
bias_results["dinuc_shuffled"] = {"mean": float(np.mean(ds_p)), "std": float(np.std(ds_p)),
    "pct_positive": float(np.mean(ds_p > 0)*100), "n": len(ds_p)}

# 4. Agarwal controls
ctrl_path = REPO / "data" / "agarwal_2025" / "k562_all_controls_200bp.tsv"
if ctrl_path.exists():
    cdf = pd.read_csv(ctrl_path, sep="\t")
    for cat, label in [("shuffled_negative", "shuffled"), ("ernst_negative", "intergenic")]:
        sub = cdf[cdf["category"] == cat]
        if len(sub) > 0:
            sp = _predict_seqs(sub["sequence"].tolist())
            bias_results[label] = {"mean": float(np.mean(sp)), "std": float(np.std(sp)),
                "pct_positive": float(np.mean(sp > 0)*100), "n": len(sp)}

# 5. Gosai ctrl_neg with real labels
gosai = REPO / "data" / "k562" / "DATA-Table_S2__MPRA_dataset.txt"
if gosai.exists():
    gdf = pd.read_csv(gosai, sep="\t", low_memory=False)
    cn = pd.DataFrame()
    for col in ["group", "element_type", "category", "class", "data_project"]:
        if col in gdf.columns:
            mask = gdf[col].astype(str).str.contains("ctrl_neg|negative_control|scrambl", case=False, na=False)
            if mask.any():
                cn = gdf[mask].dropna(subset=["sequence", "K562_log2FC"])
                break
    if len(cn) > 0:
        cs = cn["sequence"].str[:200].tolist()
        cr = cn["K562_log2FC"].values.astype(np.float32)
        cp = _predict_seqs(cs)
        from scipy.stats import pearsonr, spearmanr
        pr, _ = pearsonr(cr, cp)
        sr, _ = spearmanr(cr, cp)
        bias_results["gosai_ctrl_neg"] = {"mean_pred": float(np.mean(cp)), "mean_real": float(np.mean(cr)),
            "pearson_r": float(pr), "spearman_r": float(sr), "mse": float(np.mean((cp-cr)**2)), "n": len(cp)}

# Save
out = model_dir / "bias_eval.json"
with open(out, "w") as f:
    json.dump(bias_results, f, indent=2)

# Print summary
for k, v in bias_results.items():
    val = v.get("mean", v.get("mean_pred", 0))
    print(f"  {k}: {val:+.3f}")
PYEOF

echo "=== DONE — $(date) ==="
