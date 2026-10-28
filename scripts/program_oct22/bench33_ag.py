"""One AlphaGenome-FT-MPRA student (Stage 1 head on the frozen encoder, then Stage 2 full fine-tune that STARTS FROM the Stage 1 checkpoint), trained on a
nested genomic pool labelled by oracle_v3, selected on V0c, scored on the 18 sets of the menu retrains. One job = one seed.

It follows experiments/train_oracle_s2_v2.py, the script that trained the oracle members themselves (same head `boda-flatten-512-512`, 600 bp window =
200 bp insert + the real plasmid flanks, shift +-15 on the real flanks with p 0.5, reverse complement with p 0.5, batch 128, AdamW wd 1e-6, MSE), except that
the data come from a pool, validation is V0c, and both stages run in one process. Two recipes for Stage 2:

  oracle   encoder lr 1e-4, head lr 1e-3, constant, no warm-up (the oracle's own S2 recipe)
  winner   encoder lr 1.2e-4, head lr 5e-4, 3 head-only warm-up epochs, cosine decay (memory project_ag_s2_winner: best on the K562 chr-split full-data sweep)

Stage 2 loads the Stage 1 checkpoint from disk and the job asserts that the loaded model reproduces the Stage 1 validation Pearson before any Stage 2
update (a head that was re-initialised instead of loaded would not), and records that in the meta file.

  python scripts/program_oct22/bench33_ag.py --D 30000 --recipe oracle --seed 123456 --seed-index 0 [--smoke]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench33_common as C  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("bench33_ag")

HEAD_NAME = "bench33_k562_boda_flatten_512_512"
RECIPES = {
    "oracle": dict(s2_enc_lr=1e-4, s2_head_lr=1e-3, warmup=0, cosine=False, s2_patience=4),
    "winner": dict(s2_enc_lr=1.2e-4, s2_head_lr=5e-4, warmup=3, cosine=True, s2_patience=100),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    ap.add_argument(
        "--seed",
        type=int,
        required=True,
        help="drawn from os.urandom at registration; recorded in the meta file",
    )
    ap.add_argument("--seed-index", type=int, required=True)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--out-root", default=str(C.OUT_ROOT))
    ap.add_argument("--ckpt-root", default=str(C.R / "outputs/bench33/s1_ckpt"))
    ap.add_argument("--s1-epochs", type=int, default=60)
    ap.add_argument("--s1-patience", type=int, default=6)
    ap.add_argument("--s1-head-lr", type=float, default=1e-3)
    ap.add_argument("--s2-epochs", type=int, default=20)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-shift", type=int, default=15)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:  # tiny arrays must never land next to real results
        a.out_root, a.ckpt_root = (
            str(C.R / "outputs/bench33/smoke"),
            str(C.R / "outputs/bench33/smoke/s1_ckpt"),
        )
    rec = dict(RECIPES[a.recipe])
    tag = f"{a.recipe}_{a.D // 1000}k"
    out2 = Path(a.out_root) / f"bench33_agS2_{tag}"
    out1 = Path(a.out_root) / f"bench33_agS1_{tag}"
    name = f"r{a.seed_index:02d}_menu_000"
    if (out2 / f"{name}_meta.json").exists():
        log.info("already done: %s", out2 / name)
        return
    ckpt_dir = Path(a.ckpt_root) / f"{tag}_s{a.seed_index:02d}" / "best_model"
    t_start = time.time()

    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    from alphagenome_ft import create_model_with_heads

    from albench.paths import resolve
    from models.alphagenome_heads import register_s2f_head
    from models.embedding_cache import reinit_head_params

    d = C.load_data(a.D, a.pool, smoke=a.smoke)
    C.save_labels(out2, d)
    C.save_labels(out1, d)
    if a.smoke:
        a.s1_epochs, a.s2_epochs, rec["warmup"] = 1, 1, 0
    n_tr = len(d["train_seqs"])
    log.info(
        "D=%d recipe=%s seed=%d (index %d): train=%d val=%d tests=%d | %s",
        a.D,
        a.recipe,
        a.seed,
        a.seed_index,
        n_tr,
        len(d["val_seqs"]),
        len(d["tests"]),
        jax.devices(),
    )

    # ── data: 600 bp windows from 200 bp cores and the real plasmid flanks ───────────────────────────────────────────────────────────────────────────────
    CTX = 300
    f5_full, f3_full = C.mpra_flanks(CTX)
    f5, f3 = f5_full[-200:], f3_full[:200]
    X_tr, y_tr = C.onehot_core200(d["train_seqs"]), d["train_y"]
    rng = np.random.default_rng(a.seed)
    np.random.seed(a.seed % (2**32))

    def train_batch(idx):
        core = X_tr[idx]
        bsz = len(idx)
        wide = np.concatenate(
            [
                np.broadcast_to(f5_full, (bsz, CTX, 4)),
                core,
                np.broadcast_to(f3_full, (bsz, CTX, 4)),
            ],
            axis=1,
        )  # (B, 800, 4)
        x = np.empty((bsz, 600, 4), np.float32)
        for i in range(bsz):
            shift = (
                int(rng.integers(-a.max_shift, a.max_shift + 1))
                if (a.max_shift > 0 and rng.random() > 0.5)
                else 0
            )
            st = max(0, min(wide.shape[1] - 600, (wide.shape[1] - 600) // 2 + shift))
            w = wide[i, st : st + 600]
            x[i] = w[::-1, ::-1] if rng.random() > 0.5 else w
        return x, y_tr[idx]

    def eval_windows(cores):
        return np.concatenate(
            [
                np.broadcast_to(f5, (len(cores), 200, 4)),
                cores,
                np.broadcast_to(f3, (len(cores), 200, 4)),
            ],
            axis=1,
        )

    # ── model ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    register_s2f_head(
        head_name=HEAD_NAME,
        arch="boda-flatten-512-512",
        task_mode="human",
        num_tracks=1,
        dropout_rate=a.dropout,
    )
    weights = str(resolve("ag_weights"))
    model = create_model_with_heads(
        "all_folds",
        heads=[HEAD_NAME],
        checkpoint_path=weights,
        use_encoder_output=True,
        detach_backbone=False,
    )
    reinit_head_params(model, HEAD_NAME, num_tokens=5, dim=1536, rng=a.seed % (2**31))

    def path_str(path):
        return "/".join(str(p.key if hasattr(p, "key") else p) for p in path)

    def labels_for(stage):
        def fn(path, _leaf):
            s = path_str(path)
            if HEAD_NAME in s:
                return "head"
            return "encoder" if (stage == 2 and "sequence_encoder" in s) else "frozen"

        return jax.tree_util.tree_map_with_path(fn, model._params)

    def make_steps(optimizer):
        @jax.jit
        def train_step(params, opt_state, x, y):
            def loss_fn(p):
                pr = model._predict(
                    p,
                    model._state,
                    x,
                    jnp.zeros(x.shape[0], dtype=jnp.int32),
                    negative_strand_mask=jnp.zeros(x.shape[0], dtype=bool),
                    strand_reindexing=None,
                    requested_outputs=[HEAD_NAME],
                )[HEAD_NAME]
                pr = jnp.squeeze(pr, axis=-1) if pr.ndim > 1 else pr
                return jnp.mean((pr - y) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        return train_step

    def make_eval():
        @jax.jit
        def eval_step(params, x):
            pr = model._predict(
                params,
                model._state,
                x,
                jnp.zeros(x.shape[0], dtype=jnp.int32),
                negative_strand_mask=jnp.zeros(x.shape[0], dtype=bool),
                strand_reindexing=None,
                requested_outputs=[HEAD_NAME],
            )[HEAD_NAME]
            return jnp.squeeze(pr, axis=-1) if pr.ndim > 1 else pr

        return eval_step

    E = {"fn": make_eval()}

    def predict_cores(params, cores, rc: bool, bs: int = 256):
        """Fixed batch shape (the tail is padded and trimmed) so that XLA compiles the forward pass once."""
        out = []
        for i in range(0, len(cores), bs):
            c = cores[i : i + bs]
            n = len(c)
            if n < bs:
                c = np.concatenate([c, np.repeat(c[-1:], bs - n, axis=0)], axis=0)
            x = eval_windows(c)
            if rc:
                x = np.ascontiguousarray(x[:, ::-1, ::-1])
            out.append(np.asarray(E["fn"](params, jnp.asarray(x))).reshape(-1)[:n])
        return np.concatenate(out).astype(np.float32)

    def predict(params, seqs, rc: bool):
        return predict_cores(params, C.onehot_core200(seqs), rc)

    va_cores = C.onehot_core200(d["val_seqs"])
    n_batches = n_tr // a.batch_size

    def run_stage(stage, epochs, patience, build_opt, tag_):
        """Epoch loop with early stopping on V0c Pearson; returns (best host params, best val, best epoch, history)."""
        best, best_params, best_ep, bad, hist = -1.0, None, -1, 0, []
        cur_opt_key, train_step, opt_state = None, None, None
        for ep in range(epochs):
            key, opt = build_opt(ep)
            if key != cur_opt_key:
                cur_opt_key, train_step = key, make_steps(opt)
                opt_state = opt.init(model._params)
                log.info("%s epoch %d: optimizer %s", tag_, ep + 1, key)
            perm = rng.permutation(n_tr)
            losses, t0 = [], time.time()
            for b in range(n_batches):
                x, y = train_batch(perm[b * a.batch_size : (b + 1) * a.batch_size])
                model._params, opt_state, loss = train_step(
                    model._params, opt_state, jnp.asarray(x), jnp.asarray(y)
                )
                losses.append(float(loss))
            vp = predict_cores(model._params, va_cores, rc=False)
            r = C.pearson(vp, d["val_y"])
            hist.append(
                {
                    "epoch": ep + 1,
                    "train_loss": float(np.mean(losses)),
                    "val_pearson": r,
                    "seconds": time.time() - t0,
                }
            )
            log.info(
                "%s epoch %d: train_loss=%.4f val_pearson=%.4f (%.0fs)%s",
                tag_,
                ep + 1,
                hist[-1]["train_loss"],
                r,
                hist[-1]["seconds"],
                " *" if r > best else "",
            )
            if not np.isfinite(r):
                raise RuntimeError(f"{tag_}: non-finite validation Pearson at epoch {ep + 1}")
            if r > best:
                best, best_ep, bad = r, ep, 0
                best_params = jax.device_get(model._params)
            else:
                bad += 1
                if bad >= patience:
                    log.info(
                        "%s early stop at epoch %d (best epoch %d, val %.4f)",
                        tag_,
                        ep + 1,
                        best_ep + 1,
                        best,
                    )
                    break
        return best_params, best, best_ep, hist

    def all_preds(params, sets=None):
        names = list(d["tests"]) if sets is None else sets
        preds = {}
        f, r_ = predict_cores(params, va_cores, rc=False), predict_cores(params, va_cores, rc=True)
        preds["val_pred"], preds["fwd_val_pred"] = (f + r_) / 2, f
        for k in names:
            cores = C.onehot_core200(d["tests"][k][0])
            f, r_ = predict_cores(params, cores, rc=False), predict_cores(params, cores, rc=True)
            preds[f"test_pred_{k}"], preds[f"fwd_test_pred_{k}"] = (f + r_) / 2, f
        if "genomic" in names:
            preds["test_pred"], preds["fwd_test_pred"] = (
                preds["test_pred_genomic"],
                preds["fwd_test_pred_genomic"],
            )
        return preds

    # ── Stage 1: head only, encoder frozen ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
    labels1 = labels_for(1)
    opt1 = optax.multi_transform(
        {"head": optax.adamw(a.s1_head_lr, weight_decay=a.wd), "frozen": optax.set_to_zero()},
        labels1,
    )
    p1, v1, e1, h1 = run_stage(1, a.s1_epochs, a.s1_patience, lambda ep: ("s1", opt1), "S1")
    model._params = jax.device_put(p1)
    ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(str(ckpt_dir), save_full_model=True)
    s1_ckpt = (ckpt_dir / "checkpoint").resolve()
    assert s1_ckpt.exists(), f"Stage 1 checkpoint was not written: {s1_ckpt}"
    t_s1 = time.time() - t_start
    log.info(
        "Stage 1 done: best val %.4f at epoch %d (%.0fs); checkpoint %s", v1, e1 + 1, t_s1, s1_ckpt
    )
    main_sets = [k for k in ("genomic", "ood", "random_32k") if k in d["tests"]]
    p1_preds = all_preds(p1, sets=main_sets)
    s1_rep = (
        C.eval_report(p1_preds, {**d, "tests": {k: d["tests"][k] for k in main_sets}})
        if main_sets
        else {}
    )
    C.save_model(
        out1,
        name,
        p1_preds,
        {
            "stage": 1,
            "arm": "ag_s1",
            "recipe": a.recipe,
            "D": a.D,
            "seed": a.seed,
            "seed_index": a.seed_index,
            "n_train": n_tr,
            "best_epoch": e1 + 1,
            "val_pearson": v1,
            "history": h1,
            "report": s1_rep,
            "pool": d["pool"],
            "hparams": {
                "head_lr": a.s1_head_lr,
                "wd": a.wd,
                "batch_size": a.batch_size,
                "max_shift": a.max_shift,
                "dropout": a.dropout,
            },
        },
    )

    # ── Stage 2: LOAD the Stage 1 checkpoint, verify it, then fine-tune the whole encoder ───────────────────────────────────────────────────────────────────
    s1_params, s1_state = ocp.StandardCheckpointer().restore(str(s1_ckpt))
    model._params, model._state = jax.device_put(s1_params), jax.device_put(s1_state)
    E["fn"] = make_eval()  # retrace with the restored state
    log.info("Loaded Stage-1 checkpoint from %s", s1_ckpt)
    v_loaded = C.pearson(predict_cores(model._params, va_cores, rc=False), d["val_y"])
    if abs(v_loaded - v1) > 2e-3:
        raise RuntimeError(
            f"Stage 2 start is not the Stage 1 model: validation Pearson {v_loaded:.4f} after loading vs {v1:.4f} at the end of Stage 1"
        )
    log.info(
        "Stage 2 starts from Stage 1: validation Pearson %.4f after loading (Stage 1 best %.4f)",
        v_loaded,
        v1,
    )

    labels2 = labels_for(2)
    steps_pe = max(1, n_batches)
    warm = rec["warmup"]

    def build_opt2(ep):
        if ep < warm:  # head-only warm-up: encoder learning rate zero
            return "s2_warm", optax.multi_transform(
                {
                    "head": optax.adamw(rec["s2_head_lr"], weight_decay=a.wd),
                    "encoder": optax.set_to_zero(),
                    "frozen": optax.set_to_zero(),
                },
                labels2,
            )
        if rec["cosine"]:
            n_dec = max(1, a.s2_epochs - warm) * steps_pe
            enc = optax.cosine_decay_schedule(rec["s2_enc_lr"], n_dec, alpha=0.0)
            head = optax.cosine_decay_schedule(rec["s2_head_lr"], n_dec, alpha=0.0)
        else:
            enc, head = rec["s2_enc_lr"], rec["s2_head_lr"]
        return "s2_main", optax.multi_transform(
            {
                "head": optax.adamw(head, weight_decay=a.wd),
                "encoder": optax.adamw(enc, weight_decay=a.wd),
                "frozen": optax.set_to_zero(),
            },
            labels2,
        )

    p2, v2, e2, h2 = run_stage(2, a.s2_epochs, rec["s2_patience"], build_opt2, "S2")
    t_s2 = time.time() - t_start
    log.info(
        "Stage 2 done: best val %.4f at epoch %d (Stage 1 was %.4f); %.0fs in total",
        v2,
        e2 + 1,
        v1,
        t_s2,
    )

    # ── predictions with the validation-selected Stage 2 parameters ─────────────────────────────────────────────────────────────────────────────────────
    model._params = jax.device_put(p2)
    preds = all_preds(p2)
    rep = C.eval_report(preds, d)
    gpu = getattr(jax.devices()[0], "device_kind", str(jax.devices()[0]))
    meta = {
        "stage": 2,
        "arm": "ag_s2",
        "recipe": a.recipe,
        "D": a.D,
        "seed": a.seed,
        "seed_index": a.seed_index,
        "n_train": n_tr,
        "pool": d["pool"],
        "best_epoch": e2 + 1,
        "val_pearson": rep["val_pearson"],
        "val_pearson_forward_only": C.pearson(preds["fwd_val_pred"], d["val_y"]),
        "s1_checkpoint": str(s1_ckpt),
        "s1_loaded": True,
        "s1_best_val": v1,
        "s2_start_val_after_loading_s1": v_loaded,
        "s1_best_epoch": e1 + 1,
        "history_s1": h1,
        "history_s2": h2,
        "report": rep,
        "recipe_params": rec,
        "hparams": {
            "s1_head_lr": a.s1_head_lr,
            "s1_epochs": a.s1_epochs,
            "s1_patience": a.s1_patience,
            "s2_epochs": a.s2_epochs,
            "wd": a.wd,
            "batch_size": a.batch_size,
            "max_shift": a.max_shift,
            "dropout": a.dropout,
            "optimizer": "adamw",
            "loss": "mse",
            "window": "600 bp = 200 bp insert + 200 bp real plasmid flank each side",
            "augmentation": "shift +-15 on real flanks (p .5), reverse complement (p .5)",
            "unfrozen": "all encoder blocks (stage 2), head only (stage 1)",
            "head": "boda-flatten-512-512, 1 track",
        },
        "train_time_sec": t_s2,
        "stage1_time_sec": t_s1,
        "gpu": gpu,
        "smoke": a.smoke,
        "pearson_note": "val/test_pred are reverse-complement averaged; fwd_* hold forward-only predictions",
    }
    C.save_model(out2, name, preds, meta)
    log.info(
        "saved %s | val %.4f genomic %.4f ood %.4f random_32k %.4f",
        out2 / name,
        rep["val_pearson"],
        rep.get("test_genomic", float("nan")),
        rep.get("test_ood", float("nan")),
        rep.get("test_random_32k", float("nan")),
    )


if __name__ == "__main__":
    main()
