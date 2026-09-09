"""Measure LegNet-ensemble training/inference throughput, then project to target scales.

Measures steady-state fwd+bwd step time for the real model at the real sequence
lengths and batch sizes, on whatever GPU the job lands on. Throughput is what we
need for a schedule, so this uses synthetic tensors of the correct shape rather
than real data -- step time depends on shape and dtype, not on label values.

Projections multiply measured step time by (D / batch_size) * epochs * n_models.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/models")

from models.legnet import LegNet  # noqa: E402


def build_block_sizes(n_layers: int, width_base: int) -> list[int]:
    """Canonical MPRA-LegNet growing widths, extended to n_layers."""
    grow = [width_base, width_base + 16, width_base + 32, width_base + 48]
    out = []
    while len(out) < n_layers:
        out.extend(grow)
    return out[:n_layers]


def time_train_steps(model, x, y, task, optimizer_name, amp, n_warmup=8, n_steps=30):
    dev = x.device
    if optimizer_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.1)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16"))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[amp]

    def one_step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype, enabled=(amp != "fp32")):
            out = model(x)
            if task == "yeast":
                # 18-bin soft classification, KL loss (DREAM convention)
                loss = torch.nn.functional.kl_div(
                    torch.log_softmax(out.float(), dim=-1), y, reduction="batchmean"
                )
            else:
                loss = torch.nn.functional.mse_loss(out.float().squeeze(-1), y)
        if amp == "fp16":
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

    for _ in range(n_warmup):
        one_step()
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        one_step()
    torch.cuda.synchronize(dev)
    return (time.perf_counter() - t0) / n_steps


def time_inference(model, x, amp, n_warmup=5, n_steps=20):
    dev = x.device
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[amp]
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            with torch.autocast("cuda", dtype=dtype, enabled=(amp != "fp32")):
                model(x)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            with torch.autocast("cuda", dtype=dtype, enabled=(amp != "fp32")):
                model(x)
        torch.cuda.synchronize(dev)
    model.train()
    return (time.perf_counter() - t0) / n_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/bench_legnet.json")
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--width_base", type=int, default=80)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no GPU visible"
    dev = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu}  ({total_mem:.0f} GB)")
    print(f"torch {torch.__version__}, block_sizes={build_block_sizes(args.n_layers, args.width_base)}")

    # (task, seq_len, in_channels, task_mode)
    TASKS = [
        ("human", 200, "k562"),
        ("yeast", 110, "yeast"),
    ]
    BATCH = [256, 1024]
    AMP = ["bf16", "fp32"]

    results = []
    for task, seq_len, task_mode in TASKS:
        bs_ = build_block_sizes(args.n_layers, args.width_base)
        model = LegNet(
            in_channels=4, block_sizes=bs_, ks=7, task_mode=task_mode
        ).to(dev)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"\n=== {task}  seq_len={seq_len}  task_mode={task_mode}  params={n_par/1e6:.2f}M ===")
        for bs in BATCH:
            x = torch.randn(bs, 4, seq_len, device=dev)
            if task_mode == "yeast":
                y = torch.softmax(torch.randn(bs, 18, device=dev), dim=-1)
            else:
                y = torch.randn(bs, device=dev)
            for amp in AMP:
                try:
                    st = time_train_steps(model, x, y, task, "adamw", amp)
                    inf = time_inference(model, x, amp)
                    tr_sps = bs / st
                    inf_sps = bs / inf
                    results.append(dict(
                        task=task, seq_len=seq_len, batch_size=bs, amp=amp,
                        params=n_par, sec_per_step=st,
                        train_seq_per_s=tr_sps, infer_seq_per_s=inf_sps,
                    ))
                    print(f"  bs={bs:>4} {amp:>4}: {st*1e3:7.2f} ms/step | "
                          f"train {tr_sps:>10,.0f} seq/s | infer {inf_sps:>10,.0f} seq/s")
                except torch.cuda.OutOfMemoryError:
                    print(f"  bs={bs:>4} {amp:>4}: OOM")
                    torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    out = dict(gpu=gpu, total_mem_gb=total_mem, torch=torch.__version__,
               n_layers=args.n_layers, width_base=args.width_base, results=results)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
