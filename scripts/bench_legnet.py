"""Measure MPRA-LegNet ensemble training throughput, then project to target scales.

Throughput depends on tensor shape and dtype, not on label values, so this times
the real model at the real sequence lengths on synthetic tensors. Each config is
timed twice and only the second run is reported, because cuDNN autotunes its
convolution algorithms on first use and a single pass attributes that one-time
cost to whichever config happened to run first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/models")

from models.legnet import LegNet  # noqa: E402

torch.backends.cudnn.benchmark = True

_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def build_block_sizes(n_layers: int, width_base: int) -> list[int]:
    grow = [width_base, width_base + 16, width_base + 32, width_base + 48]
    out: list[int] = []
    while len(out) < n_layers:
        out.extend(grow)
    return out[:n_layers]


def _loss(model, x, y_scalar, y_bins, task):
    if task == "yeast":
        # Real yeast training is an 18-bin soft classification with KL loss; the
        # forward pass collapses bins to a scalar, so use the logits path.
        logits = model.get_logits(x)
        return torch.nn.functional.kl_div(
            torch.log_softmax(logits.float(), dim=-1), y_bins, reduction="batchmean"
        )
    return torch.nn.functional.mse_loss(model(x).float().reshape(-1), y_scalar)


def time_train(model, x, ys, yb, task, amp, n_warmup=25, n_steps=50):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.1)
    dtype = _DTYPE[amp]

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype, enabled=(amp != "fp32")):
            loss = _loss(model, x, ys, yb, task)
        loss.backward()
        opt.step()

    for _ in range(n_warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_steps


def time_infer(model, x, amp, n_warmup=15, n_steps=40):
    dtype = _DTYPE[amp]
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            with torch.autocast("cuda", dtype=dtype, enabled=(amp != "fp32")):
                model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            with torch.autocast("cuda", dtype=dtype, enabled=(amp != "fp32")):
                model(x)
        torch.cuda.synchronize()
    model.train()
    return (time.perf_counter() - t0) / n_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/bench.json")
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--width_base", type=int, default=80)
    ap.add_argument(
        "--amp",
        default="bf16,fp32",
        help="Comma-separated precisions. Volta (V100) has no bf16 hardware support and "
        "falls back to something unusably slow, so use fp16,fp32 there.",
    )
    args = ap.parse_args()

    assert torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0)
    print(
        f"GPU: {gpu} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB), "
        f"torch {torch.__version__}"
    )
    bs_list = build_block_sizes(args.n_layers, args.width_base)
    print(f"block_sizes={bs_list}\n")

    TASKS = [("human", 200, "k562"), ("yeast", 110, "yeast")]
    results = []
    for task, seq_len, mode in TASKS:
        model = LegNet(in_channels=4, block_sizes=bs_list, ks=7, task_mode=mode).cuda()
        n_par = sum(p.numel() for p in model.parameters())
        print(f"=== {task}  L={seq_len}  params={n_par / 1e6:.2f}M ===")
        for bs in (256, 1024, 4096):
            x = torch.randn(bs, 4, seq_len, device="cuda")
            ys = torch.randn(bs, device="cuda")
            yb = torch.softmax(torch.randn(bs, 18, device="cuda"), dim=-1)
            for amp in [a.strip() for a in args.amp.split(",")]:
                try:
                    time_train(model, x, ys, yb, task, amp, n_warmup=10, n_steps=5)  # autotune
                    st = time_train(model, x, ys, yb, task, amp)
                    it = time_infer(model, x, amp)
                    results.append(
                        dict(
                            task=task,
                            seq_len=seq_len,
                            batch_size=bs,
                            amp=amp,
                            params=n_par,
                            sec_per_step=st,
                            train_seq_per_s=bs / st,
                            infer_seq_per_s=bs / it,
                        )
                    )
                    print(
                        f"  bs={bs:>4} {amp:>4}: {st * 1e3:7.2f} ms/step | "
                        f"train {bs / st:>9,.0f} seq/s | infer {bs / it:>10,.0f} seq/s"
                    )
                except torch.cuda.OutOfMemoryError:
                    print(f"  bs={bs:>4} {amp:>4}: OOM")
                    torch.cuda.empty_cache()
            del x, ys, yb
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
        print()

    # ---- projections -------------------------------------------------------
    print("=" * 78)
    print("PROJECTED ENSEMBLE TRAINING TIME (best throughput per task)")
    print("=" * 78)
    best = {}
    for r in results:
        k = r["task"]
        if k not in best or r["train_seq_per_s"] > best[k]["train_seq_per_s"]:
            best[k] = r
    PLANS = [
        ("human", [300_000, 600_000, 1_000_000], 60),
        ("yeast", [1_000_000, 6_000_000, 10_000_000], 30),
    ]
    proj = []
    for task, Ds, epochs in PLANS:
        if task not in best:
            continue
        b = best[task]
        sps = b["train_seq_per_s"]
        print(
            f"\n{task}: {sps:,.0f} seq/s (bs={b['batch_size']}, {b['amp']}), {epochs} epochs, 1 GPU"
        )
        print(f"  {'D':>12} {'1 model':>10} {'8 models':>11} {'8 mod, 4 GPU':>13}")
        for D in Ds:
            h1 = D * epochs / sps / 3600
            proj.append(dict(task=task, D=D, epochs=epochs, gpu_h_1model=h1, gpu_h_8model=h1 * 8))
            print(f"  {D:>12,} {h1:>9.2f}h {h1 * 8:>10.1f}h {h1 * 8 / 4:>12.1f}h")
    out = dict(
        gpu=gpu, torch=torch.__version__, block_sizes=bs_list, results=results, projections=proj
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
