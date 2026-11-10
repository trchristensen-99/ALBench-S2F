"""Nov-10: seed from DREAM-RNN AND remove the LegNet architectural prior -- dr46's missing control.

dr46 swaps the reference implementation to DREAM-RNN but keeps `--guidance known_good`, which
tells the proposer in as many words that "LegNet-style inverted-residual blocks with
squeeze-excitation had the highest mean validation Pearson", together with recommended block
counts, widths and kernel sizes. Seeding from DREAM-RNN while leaving that in place changes the
example and keeps the recommendation, so a LegNet-shaped answer would prove nothing.

This arm uses `--guidance open`, the only guidance block with no architectural prior ("No
architecture family is known to be best on this task; unusual mechanisms have not been ruled
out"). `minimal` was rejected: it still names convolutional/squeeze-excitation models as the
strongest family.

Run with dr46, the pair attributes the convergence:
    ar41  LegNet reference + LegNet prior      -> 12/12 best-of-run LegNet-derived (measured)
    dr46  DREAM-RNN reference + LegNet prior   -> does the example alone matter?
    dr47  DREAM-RNN reference + no prior       -> does anything survive without the prior?

One bias channel remains in all three and is NOT removed here: with AL_RECIPE_MODE=tuned the
proposer is told to "favour designs that train stably with a high learning rate and a large batch
... no components that need warm-up or a small learning rate", which disfavours attention and
recurrent stacks on principle. Removing it would also change the recipe, which af42 already showed
costs non-LegNet member quality, so it is left in place and stated as a limitation.

Original description follows.

Nov-10: seed the architecture search from DREAM-RNN instead of MPRA-LegNet.

Every loop arm so far has been shown the published MPRA-LegNet as its reference and starting point
(`--reference_legnet`), and the result is uniform: 12 of 12 best-of-run models at 30k and 6 of 6 at
100k are LegNet-derived. The loop does not fail to find non-LegNet architectures so much as it is
never pointed at one. `lanes` attacks that by forcing one non-LegNet model per round, which is a
constraint on the OUTPUT. This attacks it at the input instead, which is what the PI did by hand
when they found an architecture that beat LegNet at 30k.

`arch_autoresearch.py` already supports it: `--baseline_file` inserts arbitrary text in place of
the LegNet reference when `--reference_legnet` is off (line 562). So the proposer is shown
DREAM-RNN's source and its measured bar, and refines from there.

30k, not 100k, for two reasons. DREAM-RNN's recipe is measured at 30k (e1600, wd 0.2, lr 0.01 ->
0.8511 genomic test, 0.8859 validation) and not yet above it -- `refrnn100k` is still running, and
seeding a search with an unmeasured recipe would confound architecture with recipe. And 30k is
where loop arms are cheap and where 15 rounds already run.

Matched to `ar41_30k_r1` on every budget flag; the only differences are the two that define the
experiment: `--reference_legnet` is dropped and `--baseline_file` is added.

What this is expected to show, either way:
  * if the best-of-run model is finally NOT LegNet-derived, the LegNet monoculture was an artefact
    of the prompt rather than a fact about the architecture space, and every loop arm should be
    re-read in that light;
  * if it still converges on LegNet-like designs from a DREAM-RNN start, that is strong evidence
    the convergence is driven by what wins on this data, which is a much more defensible sentence
    to write than "we only ever asked for LegNet".

    python scripts/program_oct22/register_nov10i.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov9m as m  # noqa: E402

sys.path.insert(0, m.R)
from albench.arch_search import dream_rnn_ref  # noqa: E402

R, OUT, DRY = m.R, m.OUT, m.DRY
BASE = "ar41_30k_r1"
TAG = "dr47_30k"
BASELINE_PATH = "outputs/menus_nov10/DREAM_RNN_BASELINE.txt"
# measured Nov 10 in refbank/refrnnlong: e1600 wd 0.2 lr 0.01, mean of 3 seeds
VAL_R, TEST_R = 0.8859, 0.8511


def baseline_text() -> str:
    return (
        "REFERENCE IMPLEMENTATION (DREAM-RNN, the Prix Fixe model built on the BHI team's core "
        "block, Rafi et al., Nature Biotechnology 2024). It is a bidirectional LSTM over the "
        "sequence rather than a convolutional tower, and it is the strongest published non-LegNet "
        f"architecture we have measured on this data: validation Pearson r = {VAL_R:.4f} "
        f"(held-out genomic test r = {TEST_R:.4f}) at 30,000 training sequences, trained with "
        "adamw, learning rate 0.01, weight decay 0.2, batch size 1024, 1600 epochs, one-cycle.\n\n"
        "Copy or modify it. Its recurrent core is a genuinely different inductive bias from a "
        "convolutional stack; prefer changes that keep that character over changes that turn it "
        "into a convolutional network.\n```python\n" + dream_rnn_ref.SOURCE + "```"
    )


def main():
    reg = json.load(open("scratchpad/dispatch_registry.json"))
    src = next((c for c in reg if c["name"] == BASE), None)
    assert src is not None, f"{BASE} is not in the registry"

    # the two changes that ARE the experiment
    assert "--reference_legnet" in src["cmd"], f"{BASE} does not show the LegNet reference"
    cmd = src["cmd"].replace(" --reference_legnet", "")
    # ...and drop the LegNet architectural prior, which dr46 keeps. `known_good` states outright
    # that "LegNet-style inverted-residual blocks with squeeze-excitation had the highest mean
    # validation Pearson", with recommended depths and widths, so seeding from DREAM-RNN while
    # leaving it in place swaps the reference implementation and keeps the recommendation.
    # `open` is the only guidance with no architectural prior: "No architecture family is known to
    # be best on this task". `minimal` still names convolutional/SE models as strongest.
    assert "--guidance known_good" in cmd, "the base arm's guidance moved"
    cmd = cmd.replace("--guidance known_good", "--guidance open")
    cmd += f" --baseline_file {R}/{BASELINE_PATH}"
    cmd = cmd.replace("genomic_30k_ar41/rep1", f"genomic_30k_{TAG}/rep1")
    assert "--reference_legnet" not in cmd and "--baseline_file" in cmd
    assert "known_good" not in cmd and "--guidance open" in cmd
    assert TAG in cmd and "ar41/rep1" not in cmd, "output directory still points at ar41"

    # ...and nothing else may move, or this stops being a seed comparison
    for flag in (
        "--rounds",
        "--per_round",
        "--budget_hours",
        "--base_cap_min",
        "--max_params",
        "--D",
    ):
        a = cmd.split(flag)[1].split()[0]
        b = src["cmd"].split(flag)[1].split()[0]
        assert a == b, f"{flag}: {a} against {BASE}'s {b}"
    assert '"manager": "schedule_ens"' in cmd, "must stay ar41's manager"

    env = dict(src.get("env", {}))
    env.pop("AL_NONFINITE_SKIP", None)  # read by no code
    assert env.get("AL_RECIPE_MODE") == "tuned", "keep the enforced recipe, as ar41 has"

    text = baseline_text()
    assert "class Net" in text and "nn.LSTM" in text, "the baseline must carry the architecture"
    assert f"{VAL_R:.4f}" in text, "the baseline must state the bar it sets"
    if not DRY:
        os.makedirs(os.path.join(R, os.path.dirname(BASELINE_PATH)), exist_ok=True)
        open(os.path.join(R, BASELINE_PATH), "w").write(text)

    odir = f"outputs/archsearch_reps_oct22/genomic_30k_{TAG}/rep1"
    cell = dict(
        name=f"{TAG}_r1",
        gpus=src.get("gpus", ["h100"]),
        ext=src.get("ext", []),
        env=env,
        out_dir=odir,
        needs=list(src.get("needs", [])) + [BASELINE_PATH],
        cmd=cmd,
        done=dict(type="file", path=f"{odir}/DONE"),
    )
    od = os.path.join(R, odir)
    assert not os.path.isdir(od) or not os.listdir(od), f"{odir} is not empty"
    print(f"guard: {TAG} seeded from DREAM-RNN (val r {VAL_R}), every budget flag matches {BASE}")

    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open("scratchpad/dispatch_registry.json"))
        if any(c["name"] == cell["name"] for c in reg):
            print(f"{cell['name']} already registered")
            return
        reg2 = reg + [cell]
        assert len({c["name"] for c in reg2}) == len(reg2)
        print(f"add 1 cell | registry {len(reg)} -> {len(reg2)}")
        if DRY:
            print(cell["cmd"][-260:])
            return
        shutil.copy(
            "scratchpad/dispatch_registry.json", "scratchpad/dispatch_registry.json.bak_nov10i"
        )
        json.dump(reg2, open("scratchpad/dispatch_registry.json.tmp", "w"), indent=1)
        os.replace("scratchpad/dispatch_registry.json.tmp", "scratchpad/dispatch_registry.json")
        print("registered")


if __name__ == "__main__":
    main()
