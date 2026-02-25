# albench

A lightweight, **model-agnostic** active-learning engine for sequence-to-function prediction.

`albench` provides the core loop, interfaces, and strategies you need to run active-learning experiments on genomic sequences — without tying you to any specific dataset, model architecture, or experimental framework.

## Install

```bash
pip install albench
# or for the scaling-curve helper (requires pandas):
pip install "albench[analysis]"
```

Or directly from the repo:
```bash
pip install ./albench
```

## Core concepts

| Abstraction | Location | Description |
|---|---|---|
| `SequenceModel` | `albench.model` | ABC for all models — oracle and student roles both use this |
| `TaskConfig` | `albench.task` | Specification of test sets, data paths, organism meta |
| `ALLoop` | `albench.loop` | Megaclass: initialize once, call `step()` per round or `run()` for all |
| `run_al_loop` | `albench.loop` | Functional alias for `ALLoop` (calls `ALLoop.run()` internally) |
| Reservoir strategies | `albench.reservoir` | Generate or select candidate sequences |
| Acquisition functions | `albench.acquisition` | Select the most informative sequences from candidates |
| Sequence utilities | `albench.utils` | `one_hot_encode`, `reverse_complement`, etc. |
| Evaluation | `albench.evaluation` | `evaluate_on_test_sets`, `compute_scaling_curve` |

## Quickstart

```python
from albench import ALLoop, SequenceModel, TaskConfig, RunConfig
from albench.reservoir.random_sampler import RandomSampler
from albench.acquisition.random_acq import RandomAcquisition
import numpy as np

# 1. Implement your oracle and student — both use SequenceModel (no separate classes)
class MyOracle(SequenceModel):
    def predict(self, sequences):
        return np.random.rand(len(sequences))

class MyStudent(SequenceModel):
    def predict(self, sequences):
        return np.random.rand(len(sequences))
    def fit(self, sequences, labels):
        pass  # train your model here

# 2. Define a task with test sets
task = TaskConfig(
    name="my_task",
    organism="human",
    sequence_length=200,
    data_root="data/",
    test_set={
        "held_out": {
            "sequences": ["ACGT" * 50],
            "labels": [0.5],
        }
    },
)

cfg = RunConfig(
    n_rounds=3,
    batch_size=5,
    reservoir_schedule={"default": RandomSampler()},
    acquisition_schedule={"default": RandomAcquisition()},
    output_dir="outputs/my_experiment",
)

# 3a. Class API — preferred when you need to inspect or resume state
loop = ALLoop(task=task, oracle=MyOracle(), student=MyStudent(),
              initial_labeled=["ACGT" * 50] * 10, run_config=cfg)
results = loop.run()          # run all rounds
# or: result = loop.step()   # run exactly one round

# 3b. Functional API — convenience wrapper around ALLoop
from albench import run_al_loop
results = run_al_loop(task=task, oracle=MyOracle(), student=MyStudent(),
                      initial_labeled=["ACGT" * 50] * 10, run_config=cfg)
print(f"Ran {len(results)} rounds.")
```

## Dependencies

| Dependency | Required | Notes |
|---|---|---|
| `numpy` | ✅ | Sequence encoding and array ops |
| `scipy` | ✅ | Pearson/Spearman correlation |
| `wandb` | optional | Experiment tracking; install with `albench[wandb]`. The loop skips W&B logging gracefully if not installed. |
| `pandas` | optional | `compute_scaling_curve`; install with `albench[analysis]` |

## Extending albench

Implement `SequenceModel` to plug in any oracle or student:

```python
from albench.model import SequenceModel

class MyAlphaGenomeOracle(SequenceModel):
    def predict(self, sequences):  # required
        ...
    def uncertainty(self, sequences):  # optional — for uncertainty acquisition
        ...
    def embed(self, sequences):  # optional — for diversity acquisition
        ...
    def fit(self, sequences, labels):  # optional — oracles usually skip this
        ...
```

Implement `ReservoirSampler` or `AcquisitionFunction` ABCs to add new strategies.

## License

MIT
