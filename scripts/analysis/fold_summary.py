import glob, json, re
import numpy as np
rows = []
for f in sorted(glob.glob("outputs/yeast_oracle_v2/fold_*/test_metrics.json")):
    m = json.load(open(f))
    tag = "B" if f.split("/")[-2].endswith("_B") else "A"
    rows.append((tag, m["fold_id"], m["val_metrics"]["pearson"], m["test_metrics"]["pearson"],
                 m.get("n_train", 0), m.get("eval_classes_added", {}).get("train", 0)))
for tag in ("A", "B"):
    r = [x for x in rows if x[0] == tag]
    if not r: continue
    print(f"variant {tag}: {len(r)} folds")
    print(f"  {'fold':>5}{'val r':>9}{'test r':>9}{'val-test':>10}{'n_train':>10}{'+eval':>8}")
    for t, fid, v, te, ntr, nex in sorted(r, key=lambda x: x[1]):
        print(f"  {fid:>5}{v:>9.4f}{te:>9.4f}{v - te:>+10.4f}{ntr:>10,}{nex:>8,}")
    vt = np.array([[x[2], x[3]] for x in r])
    print(f"  MEAN {vt[:,0].mean():>8.4f}{vt[:,1].mean():>9.4f}"
          f"{(vt[:,0]-vt[:,1]).mean():>+10.4f}")
    print()
