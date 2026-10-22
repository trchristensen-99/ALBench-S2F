"""Small WD ablation: PI's [1e-8, 1e-7] vs empirical [1e-5, 1e-3].
Uses random sampling so WD range is the only variable.
30 models per regime, single seed, D=30k genomic.

ENV: WD_MIN, WD_MAX, WD_REGIME_LABEL — used to set WD range.
"""
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.abspath('experiments'))
sys.path.insert(0, os.path.abspath('.'))
from scaling_hp_search import HPConfig, batch_size_menu, build_block_sizes, LR_SCHEDULE_CHOICES
from train_one_config import train_one_config

WD_MIN = float(os.environ['WD_MIN'])
WD_MAX = float(os.environ['WD_MAX'])
WD_REGIME = os.environ['WD_REGIME_LABEL']
N_MODELS = int(os.environ.get('N_MODELS', '30'))
D = int(os.environ.get('D', '30000'))
SEED = int(os.environ.get('SEED', '42'))
OUT_DIR = os.environ['OUT_DIR']
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)
print(f'=== WD ablation: regime={WD_REGIME} WD range=[{WD_MIN:.0e}, {WD_MAX:.0e}] D={D} N={N_MODELS} ===', flush=True)

t0 = time.time()
for i in range(N_MODELS):
    n_layers = int(rng.integers(2, 13))
    width_jitter = [float(2 ** rng.uniform(-1, 1)) for _ in range(n_layers)]
    hp = HPConfig(
        lr=float(10 ** rng.uniform(-5, -2)),
        batch_size=int(rng.choice(batch_size_menu(D))),
        conv_dropout=float(rng.uniform(0, 0.3)),
        dense_dropout=float(rng.uniform(0, 0.5)),
        n_layers=n_layers, width_base=int(rng.choice([16,32,64,128,256])),
        width_jitter=width_jitter, block_class=str(rng.choice(['eff','ag','plain'])),
        ks=int(rng.choice([3,5,7,9,11])),
        pct_start=float(rng.choice([0.1,0.2,0.3,0.4])),
        optimizer=str(rng.choice(['adam','adamw','muon'])),
        weight_decay=float(10 ** rng.uniform(np.log10(WD_MIN), np.log10(WD_MAX))),
        use_shift_aug=bool(rng.random() < 0.3),
        shift_max=int(rng.choice([5,10,15,20])),
        use_evoaug=bool(rng.random() < 0.3),
        lr_schedule=str(rng.choice(LR_SCHEDULE_CHOICES)),
        seed=int(rng.integers(2**31)),
    )
    model_id = f'r{i:02d}_random_00'
    out_path = os.path.join(OUT_DIR, f'{model_id}')
    print(f'[{i+1}/{N_MODELS}] WD={hp.weight_decay:.2e} lr={hp.lr:.2e} '
          f'n_layers={hp.n_layers} ks={hp.ks} width_base={hp.width_base}', flush=True)
    try:
        res = train_one_config(hp, D=D, data_seed=SEED, out_prefix=out_path,
                               reservoir='genomic', epochs=100, early_stop_patience=15,
                               min_delta=1e-3, round_idx=i, strategy_name='random',
                               regime=f'wd_ablation_{WD_REGIME}')
        print(f'  val_pearson={res.get("val_pearson", "?"):.4f}', flush=True)
    except Exception as e:
        print(f'  ERROR: {e}', flush=True)

print(f'\n=== done in {(time.time()-t0)/60:.1f}m ===')
