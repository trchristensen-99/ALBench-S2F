#!/bin/bash
# v20: Intergenic-focused debias strategies.
#
# OBSERVATION: All oracles predict ~+0.94 on intergenic eval sequences;
# TRUE measured intergenic K562_log2FC means ~-0.50. Residual = +1.4 (huge bias).
#
# NEW strategies (using REAL intergenic neg-aug):
#   c250: c91 + real_inter_all (mean=-0.5) at 3% — REPLACE dinuc with real intergenic
#   c251: c91 + real_inter_clip0 (mean=-0.6, clipped to ≤0) at 3%
#   c252: c91 + real_inter_negative_only (mean=-0.75) at 3% — most extreme negatives
#   c253: c91 + real_inter_all at 5% (more neg-aug)
#   c254: c91 + real_inter_plus_synth_dinuc (mixed real + synth) at 3%
#   c255: c170_03 winner recipe (blocks {0,5}, spectral λ=0.41) + real_inter at 3%
#   c256: c91 + mix_inter2_dinuc1 (intergenic 2x weight + dinuc 1x)
#   c257: c91 + real_inter_all at 3% + cpg_invariance λ=0.10 (stronger λ)

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/outputs/oracle_neg_sweep/debias_sweep_v20
mkdir -p $OUT
S1=$REPO/outputs/oracle_full_856k/s1/oracle_0
NEG_DIR=$REPO/data/synthetic_negatives

uv run --no-sync python <<PYEOF
import json
from pathlib import Path

def cfg(label, tsv, frac, mode, lam, blocks):
    return {
        "label": label,
        "fold_id": 0, "n_folds": 10,
        "stage1_dir": "$S1",
        "output_dir": f"outputs/oracle_neg_sweep/debias_sweep_v20/{label}/fold_0",
        "epochs": 80, "patience": 15,
        "extra_overrides": [
            f"++negatives_path={tsv}",
            f"++neg_fraction={frac}",
            f"++debias_mode={mode}",
            f"++debias_lambda={lam}",
            f"++unfreeze_encoder_blocks={blocks}",
        ],
    }

NEG = "$NEG_DIR"
configs = [
    cfg("c250_inter3_cpginv",       f"{NEG}/real_inter_all.tsv",                    0.03, "cpg_invariance", 0.05, "[0,1,2]"),
    cfg("c251_interclip3_cpginv",   f"{NEG}/real_inter_clip0.tsv",                  0.03, "cpg_invariance", 0.05, "[0,1,2]"),
    cfg("c252_interneg3_cpginv",    f"{NEG}/real_inter_negative_only.tsv",          0.03, "cpg_invariance", 0.05, "[0,1,2]"),
    cfg("c253_inter5_cpginv",       f"{NEG}/real_inter_all.tsv",                    0.05, "cpg_invariance", 0.05, "[0,1,2]"),
    cfg("c254_intermix3_cpginv",    f"{NEG}/real_inter_plus_synth_dinuc.tsv",       0.03, "cpg_invariance", 0.05, "[0,1,2]"),
    cfg("c255_inter3_blk05_spectral",f"{NEG}/real_inter_all.tsv",                   0.03, "spectral",       0.41, "[0,5]"),
    cfg("c256_intermix_dinuc1",     f"{NEG}/mix_inter2_dinuc1.tsv",                 0.03, "cpg_invariance", 0.05, "[0,1,2]"),
    cfg("c257_inter3_cpginv_lam10", f"{NEG}/real_inter_all.tsv",                    0.03, "cpg_invariance", 0.10, "[0,1,2]"),
]

CFG = "$OUT/configs.json"
Path(CFG).write_text(json.dumps(configs, indent=2))
print(f"  wrote {len(configs)} v20 configs (intergenic-focused)")
PYEOF

# Split into 3 batches of ~3 (single H100 jobs, k_parallel=3)
uv run --no-sync python <<PYEOF
import json
from pathlib import Path
configs = json.loads(Path("$OUT/configs.json").read_text())
n = len(configs)
b = (n + 2) // 3
batches = [configs[i:i+b] for i in range(0, n, b)]
for i, batch in enumerate(batches):
    Path(f"$OUT/configs_b{i}.json").write_text(json.dumps(batch, indent=2))
print(f"  split: {[len(b) for b in batches]}")
PYEOF

for tag in 0 1 2; do
    SCRIPT=$(mktemp)
    {
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=pf_debias_v20_b${tag}"
    echo "#SBATCH --output=$REPO/logs/%x-%j.out"
    echo "#SBATCH --error=$REPO/logs/%x-%j.err"
    echo "#SBATCH --partition=gpuq --qos=slow_nice --gres=gpu:h100:1 --cpus-per-task=14 --time=06:00:00 --mem=200G"
    echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
    echo "module load EB5; cd $REPO"
    echo "export PYTHONPATH=\"\$PWD\""
    echo "source scripts/slurm/setup_hpc_deps.sh"
    echo "uv run --no-sync python scripts/preflight/parallel_ag_s2_runner.py $OUT/configs_b${tag}.json 3"
    } > $SCRIPT
    JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT 2>&1)
    rm -f $SCRIPT
    echo "  v20_b${tag}: $JID"
done
