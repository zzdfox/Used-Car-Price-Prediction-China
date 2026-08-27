"""Experiment 9: error decomposition -- break down blended OOF error by price decile/category + fold 4 diagnosis

Everything is based on existing pred_eval_*.npz and blend_eval.json; no model is trained.
The fold split can be reproduced with KFold(5, shuffle=True, random_state=2020) on the 135k set.
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
import fe, blend
from main import USER_DATA

NAMES = ["lgb_a", "lgb_b", "cat", "xgb", "mlp"]
store = {nm: np.load(USER_DATA + f"pred_eval_{nm}.npz") for nm in NAMES}
d0 = store["lgb_a"]
y_fit, dev_idx = d0["y_fit"], d0["dev_idx"]
P_oof = np.column_stack([store[nm]["oof"] for nm in NAMES])
cfg = json.load(open(USER_DATA + "blend_eval.json"))
w = np.array([cfg["weights"][nm] for nm in NAMES])
pred = blend.predict_blend(P_oof, w, space=cfg["space"])
err = np.abs(pred - y_fit)
n = len(y_fit)
print(f"blend OOF MAE = {err.mean():.2f}  (n={n})\n")

tr_raw, _ = fe.load_raw()
raw = tr_raw.iloc[dev_idx].reset_index(drop=True)

# --- 1. Price-decile decomposition -----------------------------------------
dec = pd.qcut(y_fit, 10, labels=False, duplicates="drop")
print("Price decile decomposition (share = this bin's error as fraction of total error):")
print(f"{'bin':>3s} {'price range':>16s} {'MAE':>9s} {'err share':>8s} {'bias(pred-y)':>12s}")
for d in range(dec.max() + 1):
    m = dec == d
    lo, hi = y_fit[m].min(), y_fit[m].max()
    bias = (pred[m] - y_fit[m]).mean()
    print(f"{d:3d} {f'{lo:.0f}-{hi:.0f}':>16s} {err[m].mean():9.2f} "
          f"{err[m].sum()/err.sum()*100:7.1f}% {bias:12.1f}")

# --- 2. Fold decomposition + does fold4 concentrate "hard samples" ---------
folds = list(KFold(5, shuffle=True, random_state=2020).split(np.arange(n)))
fold_of = np.zeros(n, int)
for k, (_, va) in enumerate(folds):
    fold_of[va] = k
print("\nfold decomposition:")
band = pd.qcut(y_fit, 4, labels=False, duplicates="drop")   # quartiles are more stable
print(f"{'fold':>4s} {'MAE':>9s} {'top25%prcMAE':>11s} {'restMAE':>9s} {'expensive%':>8s} {'med price':>9s}")
for k in range(5):
    m = fold_of == k
    hi = m & (band == 3); lo = m & (band < 3)
    print(f"{k+1:4d} {err[m].mean():9.2f} {err[hi].mean():11.2f} {err[lo].mean():9.2f} "
          f"{hi.sum()/m.sum()*100:7.1f}% {np.median(y_fit[m]):9.0f}")

# Where fold4's excess error comes from: all models jointly bad, or individual models bad
print("\nPer-base-model MAE by fold (log predictions converted back to price):")
from trainer import to_price
print(f"{'fold':>4s} " + " ".join(f"{nm:>8s}" for nm in NAMES))
for k in range(5):
    m = fold_of == k
    row = " ".join(f"{np.abs(to_price(store[nm]['oof'][m]) - y_fit[m]).mean():8.1f}" for nm in NAMES)
    print(f"{k+1:4d} {row}")

# --- 3. Category slices ----------------------------------------------------
def slice_report(title, mask):
    m = np.asarray(mask, bool)
    if m.sum() == 0: return
    print(f"  {title:34s} n={m.sum():6d} ({m.mean()*100:4.1f}%)  MAE={err[m].mean():8.2f}  "
          f"err share={err[m].sum()/err.sum()*100:5.1f}%")

print("\nSlice analysis:")
nrd = raw["notRepairedDamage"].astype(str)
slice_report("notRepairedDamage = '-'", (nrd == "-").to_numpy())
slice_report("notRepairedDamage = 1 (damaged)", (nrd == "1.0").to_numpy())
pw = raw["power"].to_numpy()
slice_report("power out of range (0 or >600)", (pw <= 0) | (pw > 600))
km = raw["kilometer"].to_numpy()
slice_report("kilometer = 15 (capped)", km == 15)
slice_report("kilometer < 3", km < 3)
regm = (raw["regDate"].astype("int64").astype(str).str.zfill(8).str[4:6] == "00").to_numpy()
slice_report("regDate month=00", regm)
cnt_name = raw["name"].map(raw["name"].value_counts()).to_numpy()
slice_report("name unique (cnt=1)", cnt_name == 1)

print("\nbrand error share top10:")
b = pd.DataFrame({"brand": raw["brand"].to_numpy(), "err": err, "y": y_fit})
g = b.groupby("brand").agg(n=("err", "size"), mae=("err", "mean"),
                           share=("err", "sum"), med_y=("y", "median"))
g["share"] = g["share"] / err.sum() * 100
print(g.sort_values("share", ascending=False).head(10).round(1).to_string())

# --- 4. Profile of highest-error rows --------------------------------------
top = np.argsort(err)[::-1][:1000]
print(f"\nProfile of top1000 error rows: median true price {np.median(y_fit[top]):.0f} "
      f"(overall {np.median(y_fit):.0f}), median prediction {np.median(pred[top]):.0f}, "
      f"underestimated fraction {(pred[top] < y_fit[top]).mean()*100:.0f}%, "
      f"expensive-car (top25%) share {(band[top] == 3).mean()*100:.0f}%, "
      f"median car age {((pd.to_datetime(raw['creatDate'].iloc[top], format='%Y%m%d', errors='coerce') - pd.to_datetime(raw['regDate'].iloc[top].astype(str).str.replace('0000$','0101',regex=True), format='%Y%m%d', errors='coerce')).dt.days / 365.25).median():.1f} years")
