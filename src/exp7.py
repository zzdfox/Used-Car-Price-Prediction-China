"""Experiment 7: feature pruning (FINDINGS TODO #5)

Of the 216 dims, 105 are the full pairwise products vx_i_j. Rank features by gain
importance, then measure OOF of top-K subsets under a quick config (lr=0.1, 3 folds)
to find a working set that keeps accuracy with fewer dims, used to speed up later
experiment iterations. Note: whether the final model switches to the subset must be
re-validated with the full config."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import lightgbm as lgb
import fe, trainer
from main import split_holdout, get_data, LGB_BASE, USER_DATA

df_train, y_price, df_test, test_ids, tr_raw, te_raw = get_data(fe.TEST_A, group_stats=True)
dev_idx, hold_idx = split_holdout(len(df_train))
df_fit = df_train.iloc[dev_idx].reset_index(drop=True); y_fit = y_price[dev_idx]

P = dict(LGB_BASE, seed=42, learning_rate=0.1)   # quick config: doubled lr
QUICK_ROUNDS, QUICK_ES, FOLDS = 8000, 200, 3

# --- 1. single 80/20 split to get gain importance --------------------------
rng = np.random.RandomState(7)
perm = rng.permutation(len(df_fit)); k = int(len(df_fit) * 0.8)
tr_i, va_i = perm[:k], perm[k:]
ylog = np.log1p(y_fit); shift = float(ylog[tr_i].mean())
dtr = lgb.Dataset(df_fit.iloc[tr_i].to_numpy(np.float32), ylog[tr_i] - shift)
dva = lgb.Dataset(df_fit.iloc[va_i].to_numpy(np.float32), ylog[va_i] - shift, reference=dtr)
m = lgb.train(P, dtr, QUICK_ROUNDS, valid_sets=[dva],
              callbacks=[lgb.early_stopping(QUICK_ES, verbose=False)])
imp = m.feature_importance("gain")
cols = df_fit.columns.to_numpy()
order = np.argsort(imp)[::-1]
pd.DataFrame({"feature": cols[order], "gain": imp[order]}).to_csv(
    USER_DATA + "feat_importance.csv", index=False)
print(f"importance saved to user_data/feat_importance.csv (best_iter={m.best_iteration})")
print("15 lowest-gain features:", ", ".join(cols[order[-15:]]), flush=True)

# --- 2. top-K subset 3-fold comparison -------------------------------------
print(f"\n{'K':>4s} {'OOF MAE':>10s} {'iters':>7s} {'secs':>6s}")
for K in [len(cols), 150, 120, 90, 60]:
    keep = list(cols[order[:K]])
    t0 = time.time()
    oof, _, info = trainer.run_cv(df_fit[keep], y_fit, [], "lgb", P,
                                  n_folds=FOLDS, seed=2020, use_te=False,
                                  num_round=QUICK_ROUNDS, es=QUICK_ES, verbose=False)
    print(f"{K:4d} {info['oof_mae']:10.2f} {np.mean(info['best_iters']):7.0f} "
          f"{time.time()-t0:6.0f}", flush=True)
