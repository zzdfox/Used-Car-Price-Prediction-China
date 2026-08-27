"""Experiment 2: Huber alpha tuning + target-encoding ablation + feature importance (lr=0.08 for faster convergence)"""
import sys, os, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error
import fe, trainer

tr_raw, te_raw = fe.load_raw()
df_all, n_train, y_price, y_log, cats = fe.build_static_features(tr_raw, te_raw)
df_train = df_all.iloc[:n_train].reset_index(drop=True)
rng = np.random.RandomState(20200313)
perm = rng.permutation(n_train); n_hold = int(n_train * 0.10)
hold_idx, dev_idx = perm[:n_hold], perm[n_hold:]
df_dev, y_dev = df_train.iloc[dev_idx].reset_index(drop=True), y_price[dev_idx]
df_hold, y_hold = df_train.iloc[hold_idx].reset_index(drop=True), y_price[hold_idx]

BASE = dict(boosting_type="gbdt", num_leaves=63, learning_rate=0.08, feature_fraction=0.7,
            bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=20, lambda_l2=2.0,
            num_threads=8, verbosity=-1, seed=42, metric="mae")

EXPS = [
    ("huber a=0.3 +TE", dict(BASE, objective="huber", alpha=0.3), True),
    ("huber a=0.6 +TE", dict(BASE, objective="huber", alpha=0.6), True),
    ("huber a=1.0 +TE", dict(BASE, objective="huber", alpha=1.0), True),
    ("huber a=0.6 -TE", dict(BASE, objective="huber", alpha=0.6), False),
    ("fair c=1.0  +TE", dict(BASE, objective="fair", fair_c=1.0), True),
]
rows = []
for name, params, use_te in EXPS:
    print(f"\n=== {name} ===", flush=True)
    oof, preds, info = trainer.run_cv(df_dev, y_dev, [("hold", df_hold)], "lgb", params,
                                      n_folds=3, seed=2020, use_te=use_te,
                                      num_round=12000, es=300)
    hm = mean_absolute_error(y_hold, trainer.to_price(preds["hold"]))
    rows.append((name, info["oof_mae"], hm, int(np.mean(info["best_iters"])), info["secs"]))
    print(f"  -> OOF {info['oof_mae']:.2f} | HOLDOUT {hm:.2f} | iters~{int(np.mean(info['best_iters']))} | {info['secs']:.0f}s", flush=True)

print("\n" + "="*74)
print(f"{'config':18s} {'OOF MAE':>10s} {'HOLDOUT MAE':>12s} {'iters':>8s} {'secs':>7s}")
for r in sorted(rows, key=lambda x: x[2]):
    print(f"{r[0]:18s} {r[1]:10.2f} {r[2]:12.2f} {r[3]:8d} {r[4]:7.0f}")
