import sys
sys.path.insert(0, "scripts/analysis")
import os
os.environ.setdefault("PI_OUT", "outputs/analysis/pi_deck")
from time_weighted_ensemble import analyze, RESERVOIRS_300K, OUT
analyze(300000, RESERVOIRS_300K, os.path.join(OUT, "fig15_time_weighted_ensemble_300k.png"))
print("DONE")
