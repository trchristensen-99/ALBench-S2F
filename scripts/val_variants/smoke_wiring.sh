#!/bin/bash
# Wiring smoke test for --val_cache (menu_retrain + arch_autoresearch). Runs on CPU in a few minutes; submit with sbatch -p cpuq.
#SBATCH --job-name=valvar_smoke
#SBATCH --partition=cpuq --qos=fast
#SBATCH --cpus-per-task=8 --mem=64G --time=01:30:00
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/val_variants/logs/smoke_%j.log
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/experiments" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
T=outputs/val_variants/test
POOL=outputs/grid_pools/k562_reps/genomic_r1_d10000.npz
MENU_ORIG=${MENU_ORIG:-scratchpad/menu_retrain.py}
MENU_VC=${MENU_VC:-$T/menu_retrain_vc.py}
ARCH_ORIG=${ARCH_ORIG:-albench/arch_search/arch_autoresearch.py}
ARCH_VC=${ARCH_VC:-$T/arch_autoresearch_vc.py}
rm -rf $T/o_*; mkdir -p $T
# patched COPIES of the files as they are right now (the real files are patched with the same tool: scripts/val_variants/apply_val_cache_patch.py)
cp scratchpad/menu_retrain.py $T/menu_retrain_vc.py; cp albench/arch_search/arch_autoresearch.py $T/arch_autoresearch_vc.py
python scripts/val_variants/apply_val_cache_patch.py menu $T/menu_retrain_vc.py
python scripts/val_variants/apply_val_cache_patch.py arch $T/arch_autoresearch_vc.py
python - <<'PY'
import json, numpy as np
T = "outputs/val_variants/test"
z = np.load("outputs/chr_split_cache/chr_val_ref_only.npz", allow_pickle=True)
s, y = z["sequences"], z["oracle_labels"]
# the caches deliberately come from OTHER rows than the default validation set's first 3000 (which --smoke uses), so a cache that
# failed to replace the validation set would show up as identical val_labels / val_pearson
np.savez(f"{T}/val_test.npz", sequences=s[10000:14000], oracle_labels=y[10000:14000], oracle_id="oracle_v3", provenance=json.dumps({"variant": "TEST_A"}))
np.savez(f"{T}/val_test2.npz", sequences=s[14000:18000], oracle_labels=y[14000:18000], oracle_id="oracle_v3", provenance=json.dumps({"variant": "TEST_B"}))
np.savez(f"{T}/val_unstamped.npz", sequences=s[10000:14000], oracle_labels=y[10000:14000])
t = np.load("data/k562/test_sets_ag_s2_chrsplit/genomic_oracle.npz", allow_pickle=True)["sequences"]
np.savez(f"{T}/val_contaminated.npz", sequences=np.concatenate([s[10000:13999], t[:1]]), oracle_labels=y[10000:14000], oracle_id="oracle_v3")
m = json.load(open("outputs/menu_retrain_v0/menu_v0.json"))
json.dump(m[:1], open(f"{T}/menu1.json", "w"))
print("test inputs written; menu entry:", m[0]["id"])
PY
ARGS="--menu $T/menu1.json --reservoir_cache $POOL --D 10000 --smoke --base_cap_min 5"
echo "=== A. menu_retrain default path: original vs patched (must be identical) ==="
python scripts/val_variants/runseeded.py $MENU_ORIG $ARGS --out_dir $T/o_orig > $T/o_orig.log 2>&1; tail -2 $T/o_orig.log
python scripts/val_variants/runseeded.py $MENU_VC $ARGS --out_dir $T/o_patch > $T/o_patch.log 2>&1; tail -2 $T/o_patch.log
python - <<'PY'
import glob, numpy as np, os
T = "outputs/val_variants/test"
ok = True
for f in sorted(glob.glob(f"{T}/o_orig/*.npz")):
    g = f.replace("o_orig", "o_patch")
    a, b = np.load(f), np.load(g)
    assert a.files == b.files, (a.files, b.files)
    for k in a.files:
        if not np.array_equal(a[k], b[k]):
            d = float(np.nanmax(np.abs(a[k].astype(float) - b[k].astype(float))))
            print("DIFF", os.path.basename(f), k, "max abs", d); ok = False
print("A default-path outputs identical:", ok, "| files:", len(glob.glob(f"{T}/o_orig/*")), "vs", len(glob.glob(f"{T}/o_patch/*")))
PY
echo "=== B. menu_retrain --val_cache ==="
python scripts/val_variants/runseeded.py $MENU_VC $ARGS --out_dir $T/o_vc --val_cache $T/val_test.npz > $T/o_vc.log 2>&1; grep -E "val_cache=|train=|DONE_MENU|r00_menu" $T/o_vc.log
python - <<'PY'
import json, glob, numpy as np
T = "outputs/val_variants/test"
lab = np.load(f"{T}/o_vc/labels.npz"); c = np.load(f"{T}/val_test.npz", allow_pickle=True)
assert np.array_equal(lab["val_labels"], c["oracle_labels"][:3000]), "val_labels != cache labels"
lab0 = np.load(f"{T}/o_orig/labels.npz")["val_labels"]
assert not np.array_equal(lab["val_labels"], lab0), "cache did not change the validation set"
m = glob.glob(f"{T}/o_vc/r00_menu_*_meta.json")[0]; meta = json.load(open(m)); p = np.load(m.replace("_meta.json", ".npz"))
assert p["val_pred"].shape == (3000,), p["val_pred"].shape
print("B val_labels == cache labels[:3000]; val_pred", p["val_pred"].shape, "; meta.val_cache =", meta["val_cache"], "; val_pearson=%.4f" % meta["val_pearson"])
print("B marker:", open(f"{T}/o_vc/val_cache.json").read().replace("\n", " "))
PY
echo "=== B2. --extra_val cross-scoring (trained against cache A, also scored on A and B) ==="
python scripts/val_variants/runseeded.py $MENU_VC $ARGS --out_dir $T/o_ex --val_cache $T/val_test.npz --extra_val A=$T/val_test.npz,B=$T/val_test2.npz > $T/o_ex.log 2>&1; grep -E "extra_val|DONE_MENU" $T/o_ex.log
python - <<'PY'
import glob, numpy as np
T = "outputs/val_variants/test"
lab = np.load(f"{T}/o_ex/labels.npz"); p = np.load(glob.glob(f"{T}/o_ex/r00_menu_*.npz")[0])
c2 = np.load(f"{T}/val_test2.npz", allow_pickle=True)
assert np.array_equal(p["valpred_A"], p["val_pred"]), "valpred_A (same cache as the training validation set) != val_pred"
assert p["valpred_B"].shape == (3000,) and np.array_equal(lab["val_labels_B"], c2["oracle_labels"][:3000])
assert not np.array_equal(p["valpred_A"], p["valpred_B"])
print("B2 ok: valpred_A == val_pred; valpred_B", p["valpred_B"].shape, "; labels.npz has", [k for k in lab.files if k.startswith("val_labels")])
# the same model with the same seed must train identically with and without --extra_val (extras are read-only)
a, b = np.load(f"{T}/o_vc/{glob.glob(f'{T}/o_vc/r00_menu_*.npz')[0].split('/')[-1]}"), p
assert all(np.array_equal(a[k], b[k]) for k in a.files), "extra_val changed the trained model's outputs"
print("B2 ok: outputs of the original arrays unchanged by --extra_val")
PY
python scripts/val_variants/runseeded.py $MENU_VC $ARGS --out_dir $T/o_ex --val_cache $T/val_test.npz 2>&1 | tail -1
echo "=== C. guards (each must refuse) ==="
python scripts/val_variants/runseeded.py $MENU_VC $ARGS --out_dir $T/o_vc --val_cache $T/val_test2.npz 2>&1 | tail -1
python scripts/val_variants/runseeded.py $MENU_VC $ARGS --out_dir $T/o_vc 2>&1 | tail -1
python scripts/val_variants/runseeded.py $MENU_VC $ARGS --out_dir $T/o_orig --val_cache $T/val_test.npz 2>&1 | tail -1
python scripts/val_variants/runseeded.py $MENU_VC $ARGS --out_dir $T/o_new1 --val_cache $T/val_unstamped.npz 2>&1 | tail -1
python scripts/val_variants/runseeded.py $MENU_VC $ARGS --out_dir $T/o_new2 --val_cache $T/val_contaminated.npz 2>&1 | tail -1
echo "=== D. resume with the same cache (no retraining) ==="
python scripts/val_variants/runseeded.py $MENU_VC $ARGS --out_dir $T/o_vc --val_cache $T/val_test.npz 2>&1 | tail -1
echo "=== E. arch_autoresearch (stub proposer): default path original vs patched, then --val_cache ==="
AARGS="--reservoir_cache $POOL --D 2000 --rounds 1 --per_round 1 --base_cap_min 5"
python scripts/val_variants/arch_stub_run.py $ARCH_ORIG $AARGS --out_dir $T/o_aorig > $T/o_aorig.log 2>&1; tail -2 $T/o_aorig.log
python scripts/val_variants/arch_stub_run.py $ARCH_VC $AARGS --out_dir $T/o_apatch > $T/o_apatch.log 2>&1; tail -2 $T/o_apatch.log
python scripts/val_variants/arch_stub_run.py $ARCH_VC $AARGS --out_dir $T/o_avc --val_cache $T/val_test.npz > $T/o_avc.log 2>&1; grep -E "val_cache=|train=|DONE_ARCH|r00_arch" $T/o_avc.log
python - <<'PY'
import glob, json, numpy as np, os
T = "outputs/val_variants/test"
ok = True
for f in sorted(glob.glob(f"{T}/o_aorig/*.npz")):
    a, b = np.load(f), np.load(f.replace("o_aorig", "o_apatch"))
    for k in a.files:
        if not np.array_equal(a[k], b[k]):
            print("DIFF", os.path.basename(f), k, float(np.nanmax(np.abs(a[k].astype(float) - b[k].astype(float))))); ok = False
print("E default-path outputs identical:", ok)
ca, cb = json.load(open(f"{T}/o_aorig/config.json"))["args"], json.load(open(f"{T}/o_apatch/config.json"))["args"]
print("E config.json args differ only by:", {k: cb[k] for k in cb if k not in ca})
lab = np.load(f"{T}/o_avc/labels.npz"); c = np.load(f"{T}/val_test.npz", allow_pickle=True)
assert np.array_equal(lab["val_labels"], c["oracle_labels"]), "arch val_labels != cache labels"
assert not np.array_equal(lab["val_labels"][:3000], np.load(f"{T}/o_aorig/labels.npz")["val_labels"][:3000]), "arch: cache did not change the validation set"
m = glob.glob(f"{T}/o_avc/r00_arch_*_meta.json")[0]; meta = json.load(open(m)); p = np.load(m.replace("_meta.json", ".npz"))
print("E val_labels == cache labels (n=%d); val_pred %s; meta.val_cache=%s; val_pearson=%.4f" % (len(lab["val_labels"]), p["val_pred"].shape, meta["val_cache"], meta["val_pearson"]))
PY
echo "SMOKE DONE"
