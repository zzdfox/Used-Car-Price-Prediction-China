"""Experiment 4: do group-stat features help"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.metrics import mean_absolute_error
import fe, trainer

tr_raw, te_raw = fe.load_raw()
df_base, n_train, y_price, _, _ = fe.build_static_features(tr_raw, te_raw)
df_gs = fe.add_group_stats(df_base)
print(f"base={df_base.shape[1]} dims  +group_stats={df_gs.shape[1]} dims", flush=True)

rng = np.random.RandomState(20200313); perm = rng.permutation(n_train)
nh = int(n_train*0.10); hold_idx, dev_idx = perm[:nh], perm[nh:]

P = dict(boosting_type="gbdt", objective="huber", alpha=0.6, num_leaves=127,
         min_data_in_leaf=20, learning_rate=0.05, feature_fraction=0.65,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
         num_threads=8, verbosity=-1, seed=42)

rows=[]
for name, D in [("base", df_base), ("base+group_stats", df_gs)]:
    dt = D.iloc[:n_train].reset_index(drop=True)
    df_dev, y_dev = dt.iloc[dev_idx].reset_index(drop=True), y_price[dev_idx]
    df_hold, y_hold = dt.iloc[hold_idx].reset_index(drop=True), y_price[hold_idx]
    print(f"\n=== {name} ({df_dev.shape[1]} dims) ===", flush=True)
    oof, preds, info = trainer.run_cv(df_dev, y_dev, [("hold", df_hold)], "lgb", P,
                                      n_folds=3, seed=2020, use_te=False,
                                      num_round=15000, es=300)
    hm = mean_absolute_error(y_hold, trainer.to_price(preds["hold"]))
    rows.append((name, info["oof_mae"], hm, int(np.mean(info["best_iters"])), info["secs"]))
    print(f"  -> OOF {info['oof_mae']:.2f} | HOLDOUT {hm:.2f} | iters~{int(np.mean(info['best_iters']))} | {info['secs']:.0f}s", flush=True)

print("\n"+"="*74)
for r in rows: print(f"{r[0]:20s} OOF {r[1]:8.2f} | HOLDOUT {r[2]:8.2f} | iters {r[3]:6d} | {r[4]:.0f}s")
