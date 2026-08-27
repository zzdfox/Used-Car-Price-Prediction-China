"""Experiment 5: sample weight price^p -- align log-space training with price-space MAE"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.metrics import mean_absolute_error
import fe, trainer

tr_raw, te_raw = fe.load_raw()
d, n, y_price, _, _ = fe.build_static_features(tr_raw, te_raw); d = fe.add_group_stats(d)
dt = d.iloc[:n].reset_index(drop=True)
rng = np.random.RandomState(20200313); perm = rng.permutation(n)
nh = int(n*0.10); hold_idx, dev_idx = perm[:nh], perm[nh:]
df_dev, y_dev = dt.iloc[dev_idx].reset_index(drop=True), y_price[dev_idx]
df_hold, y_hold = dt.iloc[hold_idx].reset_index(drop=True), y_price[hold_idx]

P = dict(boosting_type="gbdt", objective="huber", alpha=0.6, num_leaves=63,
         min_data_in_leaf=20, learning_rate=0.05, feature_fraction=0.65,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
         num_threads=8, verbosity=-1, seed=42)

rows = []
for wp in [0.0, 0.25, 0.5, 0.75, 1.0]:
    print(f"\n=== weight = price^{wp} ===", flush=True)
    oof, preds, info = trainer.run_cv(df_dev, y_dev, [("hold", df_hold)], "lgb", P,
                                      n_folds=3, seed=2020, use_te=False,
                                      num_round=12000, es=300, weight_pow=wp)
    hm = mean_absolute_error(y_hold, trainer.to_price(preds["hold"]))
    rows.append((wp, info["oof_mae"], hm, int(np.mean(info["best_iters"])), info["secs"]))
    print(f"  -> OOF {info['oof_mae']:.2f} | HOLDOUT {hm:.2f} | iters~{int(np.mean(info['best_iters']))} | {info['secs']:.0f}s", flush=True)

print("\n" + "="*66)
print(f"{'weight_pow':>10s} {'OOF MAE':>10s} {'HOLDOUT':>10s} {'iters':>8s} {'secs':>7s}")
for r in sorted(rows, key=lambda x: x[1]):
    print(f"{r[0]:>10.2f} {r[1]:10.2f} {r[2]:10.2f} {r[3]:8d} {r[4]:7.0f}")
