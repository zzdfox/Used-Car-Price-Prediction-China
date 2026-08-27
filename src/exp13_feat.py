"""Experiment 13: expensive-car feature engineering A/B (FINDINGS Sec 7.2)

Background (Sec 6.3): 48% of the error concentrates in the top-2 price bands; the hard
small groups are near-new cars (km<3) and luxury brands (brand 24). Weighting/stacking/
calibration have all proven negative; feature engineering is the only untried path.

Variants (each stacked on top of base=216 dims; no target usage, so leak-free):
  NEW   near-new interactions (is_new=km<=3 x power/v_3/v_12/reg_year etc., 6 cols)
  BRAND brand-level group stats (gs_brand_*, 21 cols -- existing stats only use the model key)
  RANK  within-group percentiles (percentile rank of v_3/v_12/power within model/brand, 6 cols)
  ALL   all three combined

Protocol: quick lgb (lr=0.1, 3 folds, 8000 rounds); metrics = overall OOF +
expensive-segment (top 25% price) OOF.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error
import fe, trainer
from main import LGB_BASE, split_holdout

tr_raw, te_raw = fe.load_raw()
df_all, n_train, y_price, _, _ = fe.build_static_features(tr_raw, te_raw)
df_all = fe.add_group_stats(df_all)


def grp_new(df):
    km = df["kilometer"]
    new = (km <= 3).astype("float32")
    return pd.DataFrame({
        "is_new": new,
        "km_low": (3.0 - km).clip(lower=0).astype("float32"),
        "new_x_power": new * df["power"],
        "new_x_v3": new * df["v_3"],
        "new_x_v12": new * df["v_12"],
        "new_x_regyear": new * df["reg_year"],
    }, index=df.index)


def grp_brand(df):
    full = fe.add_group_stats(df, keys=["brand"], vars_=fe.GS_VARS)
    return full.iloc[:, df.shape[1]:]


def grp_rank(df):
    out = {}
    for k in ["model", "brand"]:
        g = df.groupby(k, dropna=False)
        for v in ["v_3", "v_12", "power"]:
            out[f"rk_{k}_{v}"] = g[v].rank(pct=True).astype("float32")
    return pd.DataFrame(out, index=df.index)


VARIANTS = {
    "base":  [],
    "NEW":   [grp_new],
    "BRAND": [grp_brand],
    "RANK":  [grp_rank],
    "ALL":   [grp_new, grp_brand, grp_rank],
}

P = dict(LGB_BASE, seed=42, learning_rate=0.1)
dev_idx, hold_idx = split_holdout(n_train)

rows = []
for name, fns in VARIANTS.items():
    df = df_all
    if fns:
        df = pd.concat([df_all] + [f(df_all) for f in fns], axis=1)
    df_fit = df.iloc[:n_train].iloc[dev_idx].reset_index(drop=True)
    y_fit = y_price[dev_idx]
    band = y_fit >= np.quantile(y_fit, 0.75)
    oof, _, info = trainer.run_cv(df_fit, y_fit, [], "lgb", P, n_folds=3,
                                  seed=2020, use_te=False, num_round=8000,
                                  es=200, verbose=False)
    hi = mean_absolute_error(y_fit[band], trainer.to_price(oof[band]))
    lo = mean_absolute_error(y_fit[~band], trainer.to_price(oof[~band]))
    rows.append((name, df_fit.shape[1], info["oof_mae"], hi, lo, info["secs"]))
    print(f"  {name:6s} dims={df_fit.shape[1]:3d}  OOF={info['oof_mae']:7.2f}  "
          f"expensive={hi:8.2f}  rest={lo:7.2f}  ({info['secs']:.0f}s)", flush=True)

print("\n" + "=" * 70)
print(f"{'variant':>6s} {'dims':>5s} {'OOF':>8s} {'expensive':>9s} {'rest':>8s}")
base = rows[0]
for r in rows:
    print(f"{r[0]:>6s} {r[1]:5d} {r[2]:8.2f} {r[3]:9.2f} {r[4]:8.2f}"
          + (f"   (d_overall {r[2]-base[2]:+.2f}, d_expensive {r[3]-base[3]:+.2f})" if r is not base else ""))
