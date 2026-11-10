"""Retrain a MENU of candidate architectures on one dataset (reservoir pool at size D) with the arch loop.

Why: the deliverable is "a wide array of good architectures, trained at each dataset size D and ensembled", not one AutoResearch run
per reservoir/acquisition cell. This script is the retraining step: take candidate architectures (model-written source + the training
recipe they were found with) and train every one on a given dataset. It needs no LLM, is resumable, and can be sharded across jobs.

Outputs use the same layout as the AutoResearch cells (labels.npz + r{seed}_menu_{idx}_meta.json/.npz with RC-averaged predictions on
every eval set), so the existing scoring code (load_dir, enet_all_pos) works unchanged.

It answers: does the best architecture shift with D (rank stability across D)? how repeatable is a retrained menu (seeds)? does the
ensemble depend on which reservoir the menu was built from? (see scratchpad/menu_rank_stability.py)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def _gosai_se() -> np.ndarray:
    """Per-sequence K562 log2FC standard errors of the real assay (Gosai et al. Table S2): the noise the students would face on measured labels."""
    cache = Path(__file__).resolve().parents[1] / "outputs" / "gosai_k562_lfcse.npy"
    if cache.exists():
        return np.load(cache)
    with open(
        Path(__file__).resolve().parents[1] / "data" / "k562" / "DATA-Table_S2__MPRA_dataset.txt"
    ) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        j = hdr.index("K562_lfcSE")
        vals = []
        for line in fh:
            p = line.rstrip("\n").split("\t")
            try:
                vals.append(float(p[j]))
            except (ValueError, IndexError):
                pass
    se = np.asarray(vals, np.float32)
    se = se[np.isfinite(se) & (se > 0)]
    np.save(cache, se)
    return se


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--menu", required=True, help="menu.json from build_menu.py")
    ap.add_argument("--reservoir_cache", required=True)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--seeds", type=int, default=1, help="independent retrainings per entry (E1 consistency)"
    )
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument(
        "--shard", default="0/1", help="k/n: this job trains entries with index %% n == k"
    )
    ap.add_argument(
        "--base_cap_min",
        type=float,
        default=25.0,
        help="per-model wall cap, H100-equivalent minutes (x5 on V100)",
    )
    ap.add_argument("--max_params", type=int, default=40_000_000)
    ap.add_argument("--seq_len", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--val_cache",
        default="",
        help="npz (sequences, oracle_labels, oracle_id) replacing the chr19/21/X validation set for early stopping, val_pred, val_labels and proposer feedback; default empty = unchanged behaviour",
    )
    ap.add_argument(
        "--extra_val",
        default="",
        help="comma-separated NAME=PATH val caches whose predictions are ALSO saved (valpred_NAME in each model npz, val_labels_NAME in labels.npz) for cross-scoring a model on validation sets it was not trained against; default off = unchanged outputs",
    )
    ap.add_argument(
        "--label_noise_scale",
        type=float,
        default=0.0,
        help="assay-noise test: add N(0, (scale*se_i)^2) to the TRAINING and VALIDATION labels, se_i drawn from the empirical distribution of Gosai K562_lfcSE "
        "(scale 1 = the measured error of the real assay); test labels stay clean oracle values. 0 = off (unchanged behaviour)",
    )
    ap.add_argument(
        "--epoch_scale",
        type=float,
        default=1.0,
        help="multiply every entry's epoch budget (and so its LR schedule length); 1 = unchanged (training-recipe sensitivity study)",
    )
    ap.add_argument(
        "--lr_scale",
        type=float,
        default=1.0,
        help="multiply every entry's peak learning rate; 1 = unchanged",
    )
    ap.add_argument(
        "--bs_scale",
        type=float,
        default=1.0,
        help="multiply every entry's batch size; 1 = unchanged",
    )
    ap.add_argument(
        "--optimizer",
        default="",
        choices=["", "muon", "adamw", "adam"],
        help="replace every entry's optimizer; empty = keep the entry's own",
    )
    ap.add_argument(
        "--lr_abs",
        type=float,
        default=0.0,
        help="replace every entry's peak learning rate (applied before --lr_scale); 0 = keep",
    )
    ap.add_argument(
        "--wd_abs", type=float, default=0.0, help="replace every entry's weight decay; 0 = keep"
    )
    ap.add_argument(
        "--flank_ctx",
        type=int,
        default=-1,
        help="F real plasmid bases of input context on each side of the 200-bp insert (shift then runs over real bases); -1 = off, the zero-padded shift as before",
    )
    ap.add_argument(
        "--shift_max",
        type=int,
        default=-1,
        help="+-shift of the training window in the flank-context path (default 10 when --flank_ctx is given); -1 = off",
    )
    ap.add_argument(
        "--shift_per_sample",
        action="store_true",
        help="flank-context path: draw the training shift per sample (default: one shift per batch)",
    )
    ap.add_argument(
        "--shift_prob",
        type=float,
        default=1.0,
        help="flank-context path: fraction of samples that are shifted (the oracle uses 0.5)",
    )
    ap.add_argument(
        "--tta",
        type=int,
        default=0,
        help="flank-context path: average evaluation predictions over real-flank windows at offsets -K, -K/2, 0, K/2, K",
    )
    ap.add_argument(
        "--aux_labels",
        default="",
        help="npz (sequences, oracle_labels) of an auxiliary target for the same sequences (e.g. HepG2); trained as a second output, never used for evaluation",
    )
    ap.add_argument(
        "--aux_weight", type=float, default=0.5, help="weight of the auxiliary squared error"
    )
    ap.add_argument(
        "--no_context",
        action="store_true",
        help="train on the bare 200-bp insert (an experiment that deliberately differs from the standard: 20 real flank bases each side, shift +-10)",
    )
    ap.add_argument(
        "--save_models",
        action="store_true",
        help="also save each trained model's best-validation weights as {id}.pt (state_dict) so the ensemble can be reloaded for acquisition; default off",
    )
    ap.add_argument(
        "--score_pool",
        default="",
        help="npz with `sequences`: every trained model also saves its RC-averaged predictions on them as `poolpred` (acquisition dry runs); default off",
    )
    ap.add_argument(
        "--score_pool_n",
        type=int,
        default=0,
        help="only the first N sequences of --score_pool (0 = all)",
    )
    args = ap.parse_args()

    import torch
    from scaling_hp_search import (
        _atomic_savez,
        _atomic_write_text,
        load_all_test_sets,
        load_chr_test_genomic,
        load_chr_train_pool,
    )

    from albench.arch_search.arch_autoresearch import pear, predict_rc, train_one
    from albench.arch_search.flank_context import wide_onehot
    from albench.arch_search.flank_context import wrap as wrap_ctx
    from albench.arch_search.harness_standard import FLANK_CTX as STD_F
    from albench.arch_search.harness_standard import SHIFT_MAX as STD_S
    from albench.arch_search.train_generated import one_hot
    from albench.arch_search.validate import build_and_check

    if args.no_context:
        args.flank_ctx = args.shift_max = -1
    elif (
        args.flank_ctx < 0 and args.shift_max < 0
    ):  # the standard, unless this directory already holds models trained without context
        _od = Path(args.out_dir)
        _legacy = (
            any(_od.glob("r*_menu_*_meta.json"))
            or (_od / "train_override.json").exists()
            and "flank_ctx" not in (_od / "train_override.json").read_text()
        )
        if not _legacy:
            args.flank_ctx, args.shift_max = STD_F, STD_S
    wide = args.flank_ctx >= 0 or args.shift_max >= 0
    ctx_F = max(args.flank_ctx, 0)
    ctx_S = args.shift_max if args.shift_max >= 0 else 10
    if wide:
        if os.environ.get("AL_FLANK_SHIFT") == "1":
            raise SystemExit("--flank_ctx/--shift_max replace AL_FLANK_SHIFT; unset it")
        os.environ["AL_SHIFT_MAX"] = "0"  # the WideCrop wrapper does the shifting, on real bases

    def _enc(seqs):
        return wide_onehot(seqs, ctx_F + ctx_S) if wide else one_hot(seqs, args.seq_len)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    k, n = (int(x) for x in args.shard.split("/"))
    menu = json.load(open(args.menu))
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    cap_min = args.base_cap_min * (
        5.0 if "V100" in gpu else (8.0 if ("MIG" in gpu.upper() or "1g.12gb" in gpu) else 1.0)
    )  # a MIG slice is about 1/7 of an H100
    print(
        f"  GPU={gpu} per-model cap={cap_min:.0f} min | menu={len(menu)} entries | shard {k}/{n} | D={args.D}",
        flush=True,
    )

    tr_s, tr_y, va_s, va_y = load_chr_train_pool(
        args.D,
        ref_only=True,
        val_frac=0.1,
        seed=0,
        reservoir_cache=args.reservoir_cache,
        chr_val=True,
    )
    if args.val_cache:  # alternative validation set (albench/arch_search/val_cache.py); the default chr19/21/X path above is untouched
        from albench.arch_search.val_cache import guard_out_dir, load_val_cache

        va_s, va_y, _vc = load_val_cache(args.val_cache, train_seqs=tr_s)
        guard_out_dir(args.out_dir, _vc)
        print(
            f"  val_cache={_vc['path']} variant={_vc['variant']} n={_vc['n']:,} sha={_vc['sha1_16']} oracle={_vc['oracle_id']} "
            f"test_overlap={_vc['n_test_overlap']} train_overlap={_vc['n_train_overlap']}",
            flush=True,
        )
    elif os.path.exists(os.path.join(args.out_dir, "val_cache.json")):
        raise SystemExit(
            f"{args.out_dir} was started with --val_cache; pass the same --val_cache or use a fresh --out_dir"
        )
    te_s, te_oracle, te_true = load_chr_test_genomic()
    tests = load_all_test_sets()
    if args.smoke:
        tr_s, tr_y, va_s, va_y = tr_s[:2000], tr_y[:2000], va_s[:3000], va_y[:3000]
        te_s, te_oracle, te_true = te_s[:3000], te_oracle[:3000], te_true[:3000]
        tests = {kk: (v[0][:2000], v[1][:2000]) for kk, v in tests.items()}
    Xtr, ytr = _enc(tr_s), np.asarray(tr_y, np.float32)
    Xva, yva = _enc(va_s), np.asarray(va_y, np.float32)
    if args.aux_labels:
        _az = np.load(args.aux_labels, allow_pickle=True)
        _amap = dict(
            zip([str(x) for x in _az["sequences"]], np.asarray(_az["oracle_labels"], np.float32))
        )
        _miss = [x for x in tr_s if x not in _amap]
        if _miss:
            raise SystemExit(
                f"{len(_miss)} training sequences have no auxiliary label in {args.aux_labels}"
            )
        _ay = np.asarray([_amap[x] for x in tr_s], np.float32)
        ytr = np.stack([ytr, np.sqrt(args.aux_weight) * _ay], axis=1).astype(np.float32)
        print(
            f"  aux labels: {args.aux_labels} weight {args.aux_weight} corr(primary, aux) = {np.corrcoef(ytr[:, 0], _ay)[0, 1]:.3f}",
            flush=True,
        )
    yva_clean = yva
    if (
        args.label_noise_scale > 0
    ):  # assay-noise test; fixed seed so every architecture and every rerun sees the same noisy labels
        _nz = Path(args.out_dir) / "label_noise.json"
        _se = _gosai_se()
        _rg = np.random.default_rng(20261026)
        ytr = ytr + (
            args.label_noise_scale * _rg.choice(_se, len(ytr)) * _rg.standard_normal(len(ytr))
        ).astype(np.float32)
        yva = yva + (
            args.label_noise_scale * _rg.choice(_se, len(yva)) * _rg.standard_normal(len(yva))
        ).astype(np.float32)
        _info = {
            "scale": args.label_noise_scale,
            "se_median": float(np.median(_se)),
            "mean_se2": float(np.mean(_se**2)),
            "train_noise_sd": float(np.std(ytr - np.asarray(tr_y, np.float32))),
            "val_noise_sd": float(np.std(yva - yva_clean)),
        }
        if _nz.exists() and json.load(open(_nz))["scale"] != args.label_noise_scale:
            raise SystemExit(
                f"{args.out_dir} was started with --label_noise_scale {json.load(open(_nz))['scale']}"
            )
        out.mkdir(parents=True, exist_ok=True)
        _nz.write_text(json.dumps(_info))
        print(
            f"  label noise: scale={args.label_noise_scale} train noise SD={_info['train_noise_sd']:.3f} (labels SD {np.std(tr_y):.3f})",
            flush=True,
        )
    elif os.path.exists(os.path.join(args.out_dir, "label_noise.json")):
        raise SystemExit(
            f"{args.out_dir} was started with --label_noise_scale; pass the same value or use a fresh --out_dir"
        )
    Xte = _enc(te_s)
    Xsets = {kk: _enc(v[0]) for kk, v in tests.items()}
    extra = {}
    if args.extra_val:  # cross-scoring validation sets (albench/arch_search/val_cache.py); nothing here runs by default
        from albench.arch_search.val_cache import guard_extra, load_val_cache

        _infos = {}
        for _item in args.extra_val.split(","):
            _nm, _pth = _item.split("=", 1)
            _s, _y, _inf = load_val_cache(_pth, train_seqs=tr_s)
            if args.smoke:
                _s, _y = _s[:3000], _y[:3000]
            extra[_nm] = (_enc(_s), np.asarray(_y, np.float32))
            _infos[_nm] = {"path": _inf["path"], "sha1_16": _inf["sha1_16"], "n": len(_y)}
        guard_extra(args.out_dir, _infos)
        print(
            "  extra_val: "
            + ", ".join(f"{k}(n={v['n']:,} sha={v['sha1_16']})" for k, v in _infos.items()),
            flush=True,
        )
    elif os.path.exists(os.path.join(args.out_dir, "extra_val.json")):
        raise SystemExit(
            f"{args.out_dir} was started with --extra_val; pass the same --extra_val or use a fresh --out_dir"
        )
    labels = {"val_labels": yva, "test_oracle": te_oracle, "test_true": te_true}
    if args.label_noise_scale > 0:
        labels["val_labels_clean"] = yva_clean
    for kk, v in tests.items():
        labels[f"oracle_{kk}"] = v[1]
    labels.update({f"val_labels_{_k}": _v[1] for _k, _v in extra.items()})
    _atomic_savez(out / "labels.npz", **labels)
    print(
        f"  train={len(tr_s):,} val={len(va_s):,} test={len(te_s):,} evals={len(tests)}", flush=True
    )

    _ov = {
        "epoch_scale": args.epoch_scale,
        "lr_scale": args.lr_scale,
        "bs_scale": args.bs_scale,
        "al_patience": os.environ.get("AL_PATIENCE", "8"),
    }
    _new_hp = {"optimizer": args.optimizer, "lr_abs": args.lr_abs, "wd_abs": args.wd_abs}
    _new_harness = {
        k_.lower(): os.environ[k_] for k_ in ("AL_EMA", "AL_BN_RECALC") if os.environ.get(k_)
    }
    if wide:
        _new_harness.update({"flank_ctx": ctx_F, "shift_max": ctx_S})
        if args.shift_per_sample or args.shift_prob != 1.0 or args.tta:
            _new_harness.update(
                {
                    "shift_per_sample": bool(args.shift_per_sample),
                    "shift_prob": args.shift_prob,
                    "tta": args.tta,
                }
            )
    if args.aux_labels:
        _new_harness.update(
            {"aux_labels": os.path.basename(args.aux_labels), "aux_weight": args.aux_weight}
        )
    _ov.update(_new_harness)  # only when used, like the line below
    if any(_new_hp.values()):
        _ov.update(_new_hp)  # only when used, so override files of earlier runs still compare equal
    _ovp = out / "train_override.json"
    if _ovp.exists() and json.load(open(_ovp)) != _ov:
        raise SystemExit(
            f"{args.out_dir} was started with a different training override ({json.load(open(_ovp))}); use a fresh --out_dir"
        )
    if (
        (args.epoch_scale, args.lr_scale, args.bs_scale) != (1.0, 1.0, 1.0)
        or _ov["al_patience"] != "8"
        or any(_new_hp.values())
        or _new_harness
    ):
        _ovp.write_text(json.dumps(_ov))
    elif _ovp.exists():
        raise SystemExit(
            f"{args.out_dir} was started with a training override; pass the same flags or use a fresh --out_dir"
        )
    Xpool = None
    if args.score_pool:
        _pz = np.load(args.score_pool, allow_pickle=True)
        _ps = list(_pz["sequences"])[: (args.score_pool_n or None)]
        Xpool = _enc(_ps)
        print(f"  score_pool={args.score_pool} n={len(_ps):,}", flush=True)
    todo, done, failed = 0, 0, 0
    for s in range(args.seeds):
        for idx, e in enumerate(menu):
            if (
                idx % n != k or idx >= int(os.environ.get("MENU_MAX_N", "100000"))
            ):  # MENU_MAX_N: train only the first N (highest-priority) entries; a later run without it adds the rest
                continue
            todo += 1
            mid = f"r{s + args.seed_offset:02d}_menu_{idx:03d}"
            if (out / f"{mid}_meta.json").exists():
                done += 1
                continue
            torch.manual_seed(1000 + 100 * (s + args.seed_offset) + idx)
            np.random.seed(1000 + 100 * (s + args.seed_offset) + idx)
            net, verdict = build_and_check(
                e["source"],
                seq_len=args.seq_len + 2 * ctx_F,
                max_params=args.max_params,
                max_fwd_ms=float(os.environ.get("MENU_MAX_FWD_MS", "8000")),
            )
            if net is None:
                failed += 1
                print(f"    {mid} BUILD-FAIL {verdict.reason[:80]}", flush=True)
                with open(out / "failed.jsonl", "a") as fh:
                    fh.write(
                        json.dumps(
                            {"id": mid, "entry": e["id"], "why": f"build: {verdict.reason[:120]}"}
                        )
                        + "\n"
                    )
                continue
            if args.aux_labels:
                from albench.arch_search.multitask import wrap as mt_wrap

                try:
                    net = mt_wrap(net, 1, args.aux_weight)
                except ValueError as ex:
                    failed += 1
                    print(f"    {mid} AUX-FAIL {ex}", flush=True)
                    continue
            if wide:
                net = wrap_ctx(
                    net,
                    ctx_F,
                    ctx_S,
                    per_sample=args.shift_per_sample,
                    prob=args.shift_prob,
                    tta=args.tta,
                )
            tc = dict(e["train_config"])
            if args.optimizer:
                tc["optimizer"] = args.optimizer
            if args.lr_abs:
                tc["lr"] = args.lr_abs
            if args.wd_abs:
                tc["weight_decay"] = args.wd_abs
            if args.epoch_scale != 1.0:
                tc["epochs"] = max(10, int(round(tc["epochs"] * args.epoch_scale)))
            if args.lr_scale != 1.0:
                tc["lr"] = tc["lr"] * args.lr_scale
            if args.bs_scale != 1.0:
                tc["batch_size"] = max(16, int(round(tc["batch_size"] * args.bs_scale)))
            if args.smoke:
                tc["epochs"] = 3
            t0 = time.time()
            try:
                model, info, used = train_one(net, Xtr, ytr, Xva, yva, tc, max_sec=cap_min * 60)
            except Exception as ex:  # noqa: BLE001 - one bad entry must not stop the shard
                model, info, used = None, {"error": f"{type(ex).__name__}: {str(ex)[:100]}"}, None
            if model is None:
                failed += 1
                print(f"    {mid} TRAIN-FAIL {info.get('error')}", flush=True)
                with open(out / "failed.jsonl", "a") as fh:
                    fh.write(
                        json.dumps(
                            {"id": mid, "entry": e["id"], "why": f"train: {info.get('error')}"}
                        )
                        + "\n"
                    )
                continue
            res = {"val_pred": predict_rc(model, Xva), "test_pred": predict_rc(model, Xte)}
            if (not np.isfinite(res["val_pred"]).all()) or float(
                np.abs(res["val_pred"]).max()
            ) > 1e3:
                failed += 1
                print(f"    {mid} EXPLODED predictions (|pred|>1e3); not saved", flush=True)
                with open(out / "failed.jsonl", "a") as fh:
                    fh.write(
                        json.dumps({"id": mid, "entry": e["id"], "why": "exploded predictions"})
                        + "\n"
                    )
                continue
            for kk, Xk in Xsets.items():
                res[f"test_pred_{kk}"] = predict_rc(model, Xk)
            for _k, _v in extra.items():
                res[f"valpred_{_k}"] = predict_rc(model, _v[0])
            if Xpool is not None:
                res["poolpred"] = predict_rc(model, Xpool)
            _bad = [
                kk
                for kk, v in res.items()
                if (not np.isfinite(v).all()) or float(np.abs(v).max()) > 1e3
            ]
            if (
                _bad
            ):  # validates normally but explodes on a shifted eval set: unusable in an ensemble
                failed += 1
                print(
                    f"    {mid} EXPLODED predictions on {_bad[0]} (|pred|>1e3); not saved",
                    flush=True,
                )
                with open(out / "failed.jsonl", "a") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "id": mid,
                                "entry": e["id"],
                                "why": f"exploded predictions on {_bad[0]}",
                            }
                        )
                        + "\n"
                    )
                continue
            vr, tr_ = pear(res["val_pred"], yva), pear(res["test_pred"], te_oracle)
            meta = {
                "model_id": mid,
                "strategy": "menu_retrain",
                "round": s + args.seed_offset,
                "menu_entry": e["id"],
                "menu_index": idx,
                "idea": e.get("idea", ""),
                "origin_cell": e.get("cell"),
                "origin_val": e.get("orig_val"),
                "n_params": verdict.n_params,
                "train_config": tc,
                "optimizer_used": used,
                "val_pearson": vr,
                "best_val_pearson": vr,
                "test_oracle_pearson": tr_,
                "val_mse": info.get("best_val_mse"),
                "epochs_trained": info.get("epochs_trained"),
                "train_time_sec": time.time() - t0,
                "gpu": gpu,
                "D": args.D,
                "cap_hit": bool(
                    info.get("epochs_trained", 0) and (time.time() - t0) > cap_min * 60 * 0.98
                ),
            }
            meta["harness"] = {
                k_: os.environ.get(k_)
                for k_ in (
                    "AL_EMA",
                    "AL_BN_RECALC",
                    "AL_FLANK_SHIFT",
                    "AL_SHIFT_MAX",
                    "AL_PATIENCE",
                )
                if os.environ.get(k_) is not None
            }
            if wide:
                meta["harness"].update(
                    {
                        "flank_ctx": ctx_F,
                        "shift_max": ctx_S,
                        "shift_per_sample": bool(args.shift_per_sample),
                        "shift_prob": args.shift_prob,
                        "tta": args.tta,
                    }
                )
            if args.aux_labels:
                meta["harness"].update(
                    {"aux_labels": args.aux_labels, "aux_weight": args.aux_weight}
                )
            meta.update(
                {
                    k_: info[k_]
                    for k_ in ("ema_decay", "val_mse_before_bn_recalc", "bn_layers_recalculated")
                    if k_ in info
                }
            )
            if args.val_cache:
                meta["val_cache"] = {
                    "path": _vc["path"],
                    "sha1_16": _vc["sha1_16"],
                    "variant": _vc["variant"],
                    "n": _vc["n"],
                }
            if args.save_models:
                _tmp = out / f"{mid}.pt.tmp"
                torch.save({k_: v_.detach().cpu() for k_, v_ in model.state_dict().items()}, _tmp)
                os.replace(_tmp, out / f"{mid}.pt")
                meta["weights"] = f"{mid}.pt"
            _atomic_savez(
                out / f"{mid}.npz", **{kk: np.asarray(x, np.float32) for kk, x in res.items()}
            )
            _atomic_write_text(out / f"{mid}_meta.json", json.dumps(meta, indent=2, default=str))
            done += 1
            print(
                f"    {mid} val={vr:.4f} test={tr_:.4f} {verdict.n_params:,}p {used} {info.get('epochs_trained')}ep "
                f"{time.time() - t0:.0f}s  {e.get('idea', '')[:44]}",
                flush=True,
            )
    print(f"DONE_MENU shard {k}/{n}: {done}/{todo} trained, {failed} failed", flush=True)
    if done + failed >= todo:
        (out / f"COMPLETE_{k}of{n}").write_text(f"done={done} failed={failed}\n")


if __name__ == "__main__":
    main()
