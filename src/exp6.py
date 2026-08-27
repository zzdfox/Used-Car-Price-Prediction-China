"""Experiment 6: reg_month_missing flag A/B test (FINDINGS section 4.3 / TODO #2)

A (with the column) reuses the baseline pred_eval_lgb_a.npz directly -- same seed
and same fold split as this script; B (without the column) is retrained here with
every other setting identical to main.py's lgb_a, forming a paired comparison.
Therefore this script must run after main.py --stage eval has completed."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.metrics import mean_absolute_error
import fe, trainer
from main import MODELS, split_holdout, get_data, USER_DATA

df_train, y_price, df_test, test_ids, tr_raw, te_raw = get_data(fe.TEST_A, group_stats=True)
dev_idx, hold_idx = split_holdout(len(df_train))
df_fit = df_train.iloc[dev_idx].reset_index(drop=True); y_fit = y_price[dev_idx]
df_hold = df_train.iloc[hold_idx].reset_index(drop=True); y_hold = y_price[hold_idx]

base = np.load(USER_DATA + "pred_eval_lgb_a.npz")
assert np.array_equal(base["dev_idx"], dev_idx), "baseline fold split differs from current one"
mae_A = mean_absolute_error(y_fit, trainer.to_price(base["oof"]))
hm_A = mean_absolute_error(y_hold, trainer.to_price(base["hold"]))
print(f"A(with reg_month_missing, reused baseline): OOF {mae_A:.2f} | HOLDOUT {hm_A:.2f}", flush=True)

drop = ["reg_month_missing"]
cfg = MODELS["lgb_a"]
oof, preds, info = trainer.run_cv(
    df_fit.drop(columns=drop), y_fit,
    [("hold", df_hold.drop(columns=drop))],
    cfg["kind"], cfg["params"], n_folds=5, seed=2020, use_te=False,
    num_round=cfg["num_round"], es=cfg["es"])
hm_B = mean_absolute_error(y_hold, trainer.to_price(preds["hold"]))
print(f"B(without reg_month_missing):               OOF {info['oof_mae']:.2f} | HOLDOUT {hm_B:.2f}")
print(f"Feature gain (B-A, positive = helpful): OOF {info['oof_mae'] - mae_A:+.2f} | HOLDOUT {hm_B - hm_A:+.2f}")
np.savez_compressed(USER_DATA + "exp6_no_regmm.npz", oof=oof, hold=preds["hold"],
                    fold_maes=np.array(info["fold_maes"]))
