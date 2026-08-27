"""Experiment 1: check whether target transform / loss function / target encoding help (quick 3-fold comparison)"""
import sys, os, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import fe, trainer

HOLDOUT_SEED, HOLDOUT_FRAC = 20200313, 0.10

tr_raw, te_raw = fe.load_raw()
df_all, n_train, y_price, y_log, cats = fe.build_static_features(tr_raw, te_raw)
df_train = df_all.iloc[:n_train].reset_index(drop=True)

rng = np.random.RandomState(HOLDOUT_SEED)
perm = rng.permutation(n_train)
n_hold = int(n_train * HOLDOUT_FRAC)
hold_idx, dev_idx = perm[:n_hold], perm[n_hold:]
df_dev, y_dev = df_train.iloc[dev_idx].reset_index(drop=True), y_price[dev_idx]
df_hold, y_hold = df_train.iloc[hold_idx].reset_index(drop=True), y_price[hold_idx]
print(f"dev={df_dev.shape}  holdout={df_hold.shape}  feats={df_dev.shape[1]}", flush=True)

BASE = dict(boosting_type="gbdt", num_leaves=63, learning_rate=0.05, feature_fraction=0.7,
            bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=20, lambda_l2=2.0,
            num_threads=8, verbosity=-1, seed=42)

EXPS = [
    ("log+L1 +TE",  dict(BASE, objective="regression_l1", metric="mae"), True,  "log"),
    ("log+L2 +TE",  dict(BASE, objective="regression",    metric="l2"),  True,  "log"),
    ("log+Huber+TE",dict(BASE, objective="huber", alpha=0.6, metric="mae"), True, "log"),
    ("log+L1 -TE",  dict(BASE, objective="regression_l1", metric="mae"), False, "log"),
    ("raw+L1 +TE",  dict(BASE, objective="regression_l1", metric="mae"), True,  "raw"),
]

rows = []
for name, params, use_te, target in EXPS:
    print(f"\n=== {name} ===", flush=True)
    oof, preds, info = trainer.run_cv(
        df_dev, y_dev, [("hold", df_hold)], "lgb", params,
        n_folds=3, seed=2020, use_te=use_te, num_round=8000, es=200, target=target)
    hp = (trainer.to_price(preds["hold"]) if target == "log"
          else np.clip(preds["hold"], 11, 99999))
    from sklearn.metrics import mean_absolute_error
    hm = mean_absolute_error(y_hold, hp)
    rows.append((name, info["oof_mae"], hm, int(np.mean(info["best_iters"])), info["secs"]))
    print(f"  -> OOF MAE {info['oof_mae']:.2f} | HOLDOUT MAE {hm:.2f} | {info['secs']:.0f}s", flush=True)

print("\n" + "="*72)
print(f"{'config':16s} {'OOF MAE':>10s} {'HOLDOUT MAE':>12s} {'iters':>8s} {'secs':>7s}")
for r in sorted(rows, key=lambda x: x[2]):
    print(f"{r[0]:16s} {r[1]:10.2f} {r[2]:12.2f} {r[3]:8d} {r[4]:7.0f}")
