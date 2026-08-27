"""Experiment 3: tree capacity (num_leaves / min_data_in_leaf). TE already ruled out, which speeds things up a lot."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.metrics import mean_absolute_error
import fe, trainer

tr_raw, te_raw = fe.load_raw()
df_all, n_train, y_price, _, _ = fe.build_static_features(tr_raw, te_raw)
df_train = df_all.iloc[:n_train].reset_index(drop=True)
rng = np.random.RandomState(20200313); perm = rng.permutation(n_train)
nh = int(n_train*0.10); hold_idx, dev_idx = perm[:nh], perm[nh:]
df_dev, y_dev = df_train.iloc[dev_idx].reset_index(drop=True), y_price[dev_idx]
df_hold, y_hold = df_train.iloc[hold_idx].reset_index(drop=True), y_price[hold_idx]

B = dict(boosting_type="gbdt", objective="huber", alpha=0.6, metric="mae",
         learning_rate=0.05, feature_fraction=0.65, bagging_fraction=0.8, bagging_freq=1,
         lambda_l2=2.0, num_threads=8, verbosity=-1, seed=42)

EXPS = [
    ("leaves=63  mdl=20",  dict(B, num_leaves=63,  min_data_in_leaf=20)),
    ("leaves=127 mdl=20",  dict(B, num_leaves=127, min_data_in_leaf=20)),
    ("leaves=255 mdl=40",  dict(B, num_leaves=255, min_data_in_leaf=40)),
    ("leaves=511 mdl=60",  dict(B, num_leaves=511, min_data_in_leaf=60)),
]
rows = []
for name, params in EXPS:
    print(f"\n=== {name} ===", flush=True)
    oof, preds, info = trainer.run_cv(df_dev, y_dev, [("hold", df_hold)], "lgb", params,
                                      n_folds=3, seed=2020, use_te=False,
                                      num_round=15000, es=300)
    hm = mean_absolute_error(y_hold, trainer.to_price(preds["hold"]))
    rows.append((name, info["oof_mae"], hm, int(np.mean(info["best_iters"])), info["secs"]))
    print(f"  -> OOF {info['oof_mae']:.2f} | HOLDOUT {hm:.2f} | iters~{int(np.mean(info['best_iters']))} | {info['secs']:.0f}s", flush=True)

print("\n"+"="*76)
print(f"{'config':20s} {'OOF MAE':>10s} {'HOLDOUT':>10s} {'iters':>8s} {'secs':>7s}")
for r in sorted(rows, key=lambda x: x[1]):
    print(f"{r[0]:20s} {r[1]:10.2f} {r[2]:10.2f} {r[3]:8d} {r[4]:7.0f}")
