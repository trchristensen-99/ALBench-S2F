"""Run a script with python/numpy RNGs seeded (the harness's shift augmentation uses the unseeded `random` module), so two
runs of identical code are comparable bit-for-bit on CPU.   usage: runseeded.py SCRIPT [args...]"""

import random
import runpy
import sys

import numpy as np

random.seed(12345)
np.random.seed(12345)
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(script)))
runpy.run_path(script, run_name="__main__")
