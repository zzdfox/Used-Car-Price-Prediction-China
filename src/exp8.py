"""Experiment 8: stacking -- replace linear blend weights with a level-2 meta model

Input: OOF predictions of the 5 base models from user_data/pred_eval_*.npz (135k, log space).
Evaluation discipline: the meta model gets its own 5-fold CV on the OOF to produce
meta-OOF (model selection looks at it only); holdout is reported once at the very end.
The lookup post-processing still applies after the meta model as usual.
"""
import sys, os, glob, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import fe, trainer, vmatch
from main import get_data, split_holdout, USER_DATA, N_THREADS

NAMES = ["lgb_a", "lgb_b", "cat", "xgb", "mlp"]
store = {nm: np.load(USER_DATA + f"pred_eval_{nm}.npz") for nm in NAMES}
d0 = store["lgb_a"]
y_fit, dev_idx, hold_idx = d0["y_fit"], d0["dev_idx"], d0["hold_idx"]
P_oof = np.column_stack([store[nm]["oof"] for nm in NAMES])     # log space
P_hold = np.column_stack([store[nm]["hold"] for nm in NAMES])

df_train, y_price, df_test, test_ids, tr_raw, te_raw = get_data(fe.TEST_A, group_stats=False)
y_hold = y_price[hold_idx]
assert np.allclose(y_fit, y_price[dev_idx])

# A few meta features: info on "which region the sample lies in" + model disagreement
EXTRA = ["cnt_name", "used_days", "power", "kilometer", "v_3", "v_12", "cnt_model"]
X_extra_dev = df_train.iloc[dev_idx][EXTRA].to_numpy(np.float32)
X_extra_hold = df_train.iloc[hold_idx][EXTRA].to_numpy(np.float32)

def derived(P):
    return np.column_stack([P.mean(1), P.std(1), P.max(1) - P.min(1)])

ylog = np.log1p(y_fit)
ylog_shift = ylog.mean()

LGB_META = dict(objective="huber", alpha=0.6, num_leaves=15, learning_rate=0.05,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                min_data_in_leaf=500, lambda_l2=2.0, num_threads=N_THREADS,
                verbosity=-1, seed=42)

def run_meta(kind, X, X_hold, tag):
    """Meta-level 5-fold CV (different seed from base level) -> meta-OOF; fit on all -> holdout."""
    folds = KFold(5, shuffle=True, random_state=2021)
    moof = np.zeros(len(X)); iters = []
    for tr_i, va_i in folds.split(X):
        if kind == "ridge":
            m = Ridge(alpha=1.0).fit(X[tr_i], ylog[tr_i] - ylog_shift)
            moof[va_i] = m.predict(X[va_i]) + ylog_shift
        else:
            dtr = lgb.Dataset(X[tr_i], ylog[tr_i] - ylog_shift)
            dva = lgb.Dataset(X[va_i], ylog[va_i] - ylog_shift, reference=dtr)
            m = lgb.train(LGB_META, dtr, 4000, valid_sets=[dva],
                          feval=trainer._make_price_mae(ylog_shift),
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            iters.append(m.best_iteration or 4000)
            moof[va_i] = m.predict(X[va_i], num_iteration=m.best_iteration) + ylog_shift
    oof_mae = mean_absolute_error(y_fit, trainer.to_price(moof))
    # holdout: refit on all data (lgb rounds = CV early-stop mean x1.1 to compensate for larger train set)
    if kind == "ridge":
        m = Ridge(alpha=1.0).fit(X, ylog - ylog_shift)
        hold_pred = m.predict(X_hold) + ylog_shift
    else:
        nr = int(np.mean(iters) * 1.1)
        dtr = lgb.Dataset(X, ylog - ylog_shift)
        m = lgb.train(LGB_META, dtr, nr)
        hold_pred = m.predict(X_hold) + ylog_shift
    hold_mae = mean_absolute_error(y_hold, trainer.to_price(hold_pred))
    print(f"{tag:28s} meta-OOF {oof_mae:8.2f} | HOLDOUT {hold_mae:8.2f}", flush=True)
    return oof_mae, hold_mae, hold_pred

blend_ref = json.load(open(USER_DATA + "blend_eval.json"))
print(f"{'ref: linear blend':26s} OOF {blend_ref['oof_mae']:8.2f} | HOLDOUT {blend_ref['holdout_mae']:8.2f} "
      f"(+lookup {blend_ref['holdout_mae_lookup']:.2f})\n", flush=True)

results = {}
results["ridge_preds"] = run_meta("ridge", P_oof, P_hold, "ridge (5 preds)")
results["lgb_preds"] = run_meta("lgb", np.hstack([P_oof, derived(P_oof)]),
                                np.hstack([P_hold, derived(P_hold)]), "lgb (preds+disagreement)")
results["lgb_extra"] = run_meta("lgb",
                                np.hstack([P_oof, derived(P_oof), X_extra_dev]),
                                np.hstack([P_hold, derived(P_hold), X_extra_hold]),
                                "lgb (preds+disagreement+meta feats)")

# Winner (selected by meta-OOF) plus lookup post-processing, compared against 410.32
best_tag = min(results, key=lambda k: results[k][0])
hold_pred = trainer.to_price(results[best_tag][2])
table = vmatch.build_table(tr_raw.iloc[dev_idx])
hold_pred2, n_hit = vmatch.apply_table(table, tr_raw.iloc[hold_idx], hold_pred)
print(f"\nSelected by meta-OOF: {best_tag}")
print(f"  +lookup HOLDOUT = {mean_absolute_error(y_hold, hold_pred2):.2f} "
      f"(hit {n_hit} rows)   [ref: linear blend+lookup {blend_ref['holdout_mae_lookup']:.2f}]")
