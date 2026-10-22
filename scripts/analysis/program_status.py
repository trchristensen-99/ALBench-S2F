"""Write outputs/justification/program_status.json: done / running / total per experiment group, from the dispatcher status file."""
import glob, json, os, re
R = "/grid/wsbs/home_norepl/christen/ALBench-S2F"; os.chdir(R)
rows = [l.split() for l in open("outputs/dispatch/status.txt") if l.startswith("  ") and len(l.split()) >= 3]
groups = [("genomic 30k search runs", "g30_"), ("random 30k search runs", "rnd30_"), ("unseen-reservoir search runs", "u_"), ("motif 30k search runs", "mot30_"), ("genomic 10k search runs", "g10_"),
          ("genomic 100k search runs", "g100x_"), ("mixture-corpus retrains", "mix_"), ("unseen-reservoir menu retrains", "un30_"), ("architecture-source retrains", "as_")]
out = {}
for lab, pre in groups:
    r = [x for x in rows if x[0].startswith(pre)]
    out[lab] = dict(total=len(r), done=sum(1 for x in r if x[1] == "DONE"), running=sum(1 for x in r if x[1].endswith("/R")))
models = {}
for d in sorted(glob.glob("outputs/archsearch_reps_oct22/*/rep*")):
    models[d.replace("outputs/archsearch_reps_oct22/", "")] = dict(models=len(glob.glob(d + "/r*_meta.json")), complete=os.path.exists(d + "/COMPLETE"))
json.dump(dict(groups=out, runs=models), open("outputs/justification/program_status.json", "w"), indent=1)
print({k: f"{v['done']}/{v['total']} done, {v['running']} running" for k, v in out.items()})
