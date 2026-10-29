"""The published MPRA-LegNet (Agarwal et al.; external/human_legnet/model.py, K562 configuration) trained from scratch on a nested genomic pool labelled by
oracle_v3, selected on V0c, scored on the 18 sets of the menu retrains. One job trains several seeds of one configuration.

Architecture and optimiser are the published ones (stem 64 / ks 11, EffBlocks ks 9 with widths 80, 96, 112, 128, MaxPool 2 after each, resize factor 4, SiLU;
AdamW with weight decay 0.1, OneCycleLR max_lr 0.01 starting at max_lr/25 with pct_start 0.3, MSE, fp16 mixed precision, gradient clipping 1, the best epoch
by validation Pearson). The published model trains on ~600k sequences; at 10k to 30k the number of optimiser steps is tiny, so three configurations of
(batch size, epochs) are run and the one that is best on V0c is the "tuned" arm:

  pub    batch 1024, 25 epochs   the published K562 values (config.json of md_shift_reverse_noavg_noch)
  e100   batch 1024, 100 epochs  same recipe, longer schedule
  b256   batch  256, 60 epochs   smaller batch, more steps

Augmentation as published (shift, reverse complement), adapted to a 200 bp insert: the shift (uniform in -10..10, every sample) slides a 200 bp window over the
insert plus the real plasmid flank, the published code did the same with the vector primers of its 230 bp sequences; reverse complement with p 0.5.
Test-time predictions are reported reverse-complement averaged (as the menu students are); forward-only predictions are kept as fwd_*.

  python scripts/program_oct22/bench33_legnet.py --D 30000 --cfg pub --seeds 0=123456,1=654321 [--smoke]
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench33_common as C  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("bench33_legnet")

CFGS = {
    "pub": dict(batch=1024, epochs=25, max_lr=0.01),
    "e100": dict(batch=1024, epochs=100, max_lr=0.01),
    "b256": dict(batch=256, epochs=60, max_lr=0.01),
}
ARCH = dict(
    in_ch=4,
    stem_ch=64,
    stem_ks=11,
    ef_ks=9,
    ef_block_sizes=[80, 96, 112, 128],
    pool_sizes=[2, 2, 2, 2],
    resize_factor=4,
)
SHIFT = 10


def load_legnet():
    spec = importlib.util.spec_from_file_location(
        "human_legnet_model", C.R / "external/human_legnet/model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--cfg", choices=sorted(CFGS), required=True)
    ap.add_argument(
        "--seeds",
        required=True,
        help="comma list of index=seed (seeds drawn from os.urandom at registration)",
    )
    ap.add_argument("--pool", default=None)
    ap.add_argument("--out-root", default=str(C.OUT_ROOT))
    ap.add_argument(
        "--ctx",
        type=int,
        default=0,
        help="real plasmid flank bases given as input on each side (0 = the 200-bp insert alone, as in bench33)",
    )
    ap.add_argument(
        "--shift", type=int, default=SHIFT, help="+-shift of the training window over real flank"
    )
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.out_root = str(C.R / "outputs/bench33/smoke")
    cfg = dict(CFGS[a.cfg])
    if a.smoke:
        cfg["epochs"] = 2
    ctx_tag = f"_c{a.ctx}" if (a.ctx or a.shift != SHIFT) else ""
    prefix = "bench38" if ctx_tag else "bench33"
    pool_res = Path(a.pool).stem.split("_d")[0] if a.pool else "genomic"
    if pool_res != "genomic":  # students on another reservoir must not share an output directory
        ctx_tag += f"_{pool_res}"
        prefix = "bench38"
    out = Path(a.out_root) / f"{prefix}_ln_{a.cfg}_{a.D // 1000}k{ctx_tag}"
    todo = [(int(i), int(s)) for i, s in (kv.split("=") for kv in a.seeds.split(","))]

    import torch
    import torch.nn.functional as F

    dev = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    mod = load_legnet()
    d = C.load_data(a.D, a.pool, smoke=a.smoke)
    C.save_labels(out, d)
    n_tr = len(d["train_seqs"])

    S, FC = a.shift, a.ctx
    W = 200 + 2 * FC
    f5, f3 = C.mpra_flanks(FC + S)
    cores = torch.from_numpy(C.onehot_core200(d["train_seqs"])).to(dev)  # (N, 200, 4)
    wide = (
        torch.cat(
            [
                torch.from_numpy(f5).to(dev).expand(n_tr, FC + S, 4),
                cores,
                torch.from_numpy(f3).to(dev).expand(n_tr, FC + S, 4),
            ],
            dim=1,
        )
        .permute(0, 2, 1)
        .contiguous()
    )
    y_tr = torch.from_numpy(d["train_y"]).to(dev)
    arW = torch.arange(W, device=dev)
    fF5, fF3 = C.mpra_flanks(FC) if FC else (None, None)

    def to_x(seqs):
        core = torch.from_numpy(C.onehot_core200(seqs)).to(dev)  # (N, 200, 4)
        if FC:
            n = len(core)
            core = torch.cat(
                [
                    torch.from_numpy(fF5).to(dev).expand(n, FC, 4),
                    core,
                    torch.from_numpy(fF3).to(dev).expand(n, FC, 4),
                ],
                dim=1,
            )
        return core.permute(0, 2, 1).contiguous()  # (N, 4, 200 + 2F)

    _t = torch.arange(
        8, device=dev
    )  # the unshifted training window must equal the evaluation window
    assert torch.equal(
        torch.gather(wide[_t], 2, (S + arW)[None, None, :].expand(8, 4, -1)),
        to_x(d["train_seqs"][:8]),
    ), "window layout mismatch"

    va_x = to_x(d["val_seqs"])

    @torch.no_grad()
    def predict(model, x, rc: bool, bs: int = 4096):
        model.eval()
        out_ = []
        for i in range(0, len(x), bs):
            xb = x[i : i + bs]
            if rc:
                xb = xb.flip(1).flip(2)
            out_.append(model(xb).float().cpu())
        return torch.cat(out_).numpy()

    for idx, seed in todo:
        name = f"r{idx:02d}_menu_000"
        if (out / f"{name}_meta.json").exists():
            log.info("already done: %s", name)
            continue
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        torch.cuda.manual_seed_all(seed)
        model = mod.LegNet(**ARCH).to(dev)
        model.apply(mod.initialize_weights)
        n_params = sum(p.numel() for p in model.parameters())
        bs, epochs = cfg["batch"], cfg["epochs"]
        steps_pe = -(-n_tr // bs)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["max_lr"] / 25, weight_decay=0.1)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=cfg["max_lr"],
            three_phase=False,
            total_steps=epochs * steps_pe,
            pct_start=0.3,
            cycle_momentum=False,
        )
        scaler = torch.amp.GradScaler("cuda")
        best, best_state, best_ep, hist = -2.0, None, -1, []
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n_tr, device=dev)
            tl = []
            for b in range(steps_pe):
                ib = perm[b * bs : (b + 1) * bs]
                sh = torch.randint(-S, S + 1, (len(ib),), device=dev)
                pos = (S + sh)[:, None] + arW[None, :]
                xb = torch.gather(wide[ib], 2, pos[:, None, :].expand(-1, 4, -1))
                rcm = torch.rand(len(ib), device=dev) > 0.5
                xb = torch.where(rcm[:, None, None], xb.flip(1).flip(2), xb)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = F.mse_loss(model(xb).float(), y_tr[ib])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                tl.append(loss.item())
            r = C.pearson(predict(model, va_x, rc=False), d["val_y"])
            hist.append({"epoch": ep + 1, "train_loss": float(np.mean(tl)), "val_pearson": r})
            if np.isfinite(r) and r > best:
                best, best_ep = r, ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if best_state is None:
            raise RuntimeError(f"{name}: no finite validation Pearson")
        model.load_state_dict(best_state)
        preds = {}
        f, r_ = predict(model, va_x, rc=False), predict(model, va_x, rc=True)
        preds["val_pred"], preds["fwd_val_pred"] = (f + r_) / 2, f
        for k, (seqs, _y) in d["tests"].items():
            xk = to_x(seqs)
            f, r_ = predict(model, xk, rc=False), predict(model, xk, rc=True)
            preds[f"test_pred_{k}"], preds[f"fwd_test_pred_{k}"] = (f + r_) / 2, f
        preds["test_pred"], preds["fwd_test_pred"] = (
            preds["test_pred_genomic"],
            preds["fwd_test_pred_genomic"],
        )
        rep = C.eval_report(preds, d)
        meta = {
            "arm": "legnet",
            "cfg": a.cfg,
            "D": a.D,
            "seed": seed,
            "seed_index": idx,
            "n_train": n_tr,
            "n_params": n_params,
            "pool": d["pool"],
            "best_epoch": best_ep + 1,
            "val_pearson": rep["val_pearson"],
            "val_pearson_forward_only": C.pearson(preds["fwd_val_pred"], d["val_y"]),
            "report": rep,
            "history": hist,
            "train_time_sec": time.time() - t0,
            "gpu": gpu,
            "smoke": a.smoke,
            "hparams": {
                "architecture": ARCH,
                "optimizer": "AdamW",
                "weight_decay": 0.1,
                "max_lr": cfg["max_lr"],
                "start_lr": cfg["max_lr"] / 25,
                "schedule": "OneCycleLR pct_start 0.3",
                "batch_size": bs,
                "epochs": epochs,
                "loss": "mse",
                "precision": "fp16 autocast + GradScaler",
                "grad_clip": 1.0,
                "augmentation": f"shift uniform +-{S} on the real plasmid flank (every sample), reverse complement p .5",
                "flank_ctx": FC,
                "input_length": W,
                "selection": "best epoch by V0c Pearson (forward only)",
            },
            "pearson_note": "val/test_pred are reverse-complement averaged; fwd_* hold forward-only predictions",
        }
        C.save_model(out, name, preds, meta)
        log.info(
            "%s %s D=%d seed=%d: best epoch %d val %.4f genomic %.4f ood %.4f random_32k %.4f (%.0fs, %s)",
            a.cfg,
            name,
            a.D,
            seed,
            best_ep + 1,
            rep["val_pearson"],
            rep.get("test_genomic", float("nan")),
            rep.get("test_ood", float("nan")),
            rep.get("test_random_32k", float("nan")),
            time.time() - t0,
            gpu,
        )


if __name__ == "__main__":
    main()
