"""Does LegNetBank reproduce the predictions the trainer itself wrote?

The bank re-implements nothing about the model, but it DOES rebuild the input: 200-bp
insert, 20 real plasmid bases each side, both strands averaged. If that construction is off
by anything -- wrong flank, wrong orientation, insert off-centre -- predictions still come
out finite and plausible, and every acquisition decision downstream is made by a different
model than the one being trained. The trainer stored its own predictions next to the
checkpoint, so this is checkable exactly rather than by eye.
"""

import glob
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scripts/program_oct22")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/experiments")

from albench.acquisition.legnet_bank import LegNetBank  # noqa: E402

d = (
    sys.argv[1]
    if len(sys.argv) > 1
    else sorted(glob.glob("outputs/wed41/bench38_ln_e500_wd2_30k_c20_*"))[0]
)
print("bank dir:", d)
stored = np.load(
    sorted(f for f in glob.glob(d + "/*.npz") if "labels" not in f)[0], allow_pickle=True
)
print("stored arrays:", [k for k in stored.files][:8])

import bench33_common as C  # noqa: E402

res = d.split("_c20_")[-1] if "_c20_" in d else "genomic"
data = C.load_data(30000, pool_path=f"outputs/grid_pools/k562_nested/{res}_d30000.npz")
one = sorted(glob.glob(d + "/*.pt"))[0]
tmp = tempfile.mkdtemp()
shutil.copy(one, tmp)
bank = LegNetBank(tmp)
print("reservoir:", res, "| single member:", os.path.basename(one))
print("members:", len(bank))

n = 5000
mine = bank.predict(data["val_seqs"][:n])
theirs = np.asarray(stored["val_pred"])[:n]
r = float(np.corrcoef(mine, theirs)[0, 1])
md = float(np.max(np.abs(mine - theirs)))
print(
    "\nval_pred: r = %.6f   max|diff| = %.5f   (mine mu=%.4f sd=%.4f / theirs mu=%.4f sd=%.4f)"
    % (r, md, mine.mean(), mine.std(), theirs.mean(), theirs.std())
)
if r < 0.999:
    raise SystemExit("FAIL: the bank does not reproduce the trainer's own predictions")
print("PASS: input construction and RC averaging match the trainer")

emb = bank.embed(data["val_seqs"][:256])
print("embed shape:", emb.shape, "finite:", bool(np.isfinite(emb).all()))
print("separates_uncertainty (1 member):", bank.separates_uncertainty)
