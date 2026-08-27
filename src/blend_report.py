"""Collect per-model predictions -> blend -> report holdout test error -> generate testA submission file"""
import sys, os, json, glob, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error
import blend, trainer, fe, vmatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
USER_DATA, RESULT = ROOT + "user_data/", ROOT + "prediction_result/"
stage = sys.argv[1] if len(sys.argv) > 1 else "eval"

files = sorted(glob.glob(USER_DATA + f"pred_{stage}_*.npz"))
if not files:
    sys.exit(f"No pred_{stage}_*.npz found")
names = [os.path.basename(f)[len(f"pred_{stage}_"):-4] for f in files]
d0 = np.load(files[0])
y_fit, dev_idx, hold_idx, test_ids = d0["y_fit"], d0["dev_idx"], d0["hold_idx"], d0["test_ids"]

tr_raw, te_raw = fe.load_raw()
y_all = tr_raw["price"].to_numpy().astype(float)
y_hold = y_all[hold_idx]
# v-vector lookup (FINDINGS section 4.2): in the eval stage the table is built from dev only,
# in the final stage from all of train
table = vmatch.build_table(tr_raw if stage == "final" else tr_raw.iloc[dev_idx])

store = {nm: np.load(f) for nm, f in zip(names, files)}
print(f"Models in the blend: {names}\n")
print(f"{'Model':10s} {'OOF MAE':>12s} {'HOLDOUT MAE':>14s}")
rows = []
for nm in names:
    om = mean_absolute_error(y_fit, trainer.to_price(store[nm]["oof"]))
    hm = (mean_absolute_error(y_hold, trainer.to_price(store[nm]["hold"]))
          if "hold" in store[nm] else None)
    rows.append((nm, om, hm))
    print(f"{nm:10s} {om:12.2f} {('%.2f' % hm) if hm is not None else '-':>14s}")

P_oof = np.column_stack([store[nm]["oof"] for nm in names])
best = None
for space in ("log", "price"):
    w, v = blend.optimize_weights(P_oof, y_fit, space=space)
    print(f"\nBlend ({space} space) OOF MAE = {v:.2f}   weights: "
          + ", ".join(f"{n}={x:.3f}" for n, x in zip(names, w)))
    if best is None or v < best[1]:
        best = (w, v, space)
w, oof_mae, space = best

print("\n" + "=" * 72)
if stage == "eval" and "hold" in store[names[0]]:
    P_hold = np.column_stack([store[nm]["hold"] for nm in names])
    hold_price = blend.predict_blend(P_hold, w, space=space)
    hold_mae = mean_absolute_error(y_hold, hold_price)
    hold_price2, n_hit = vmatch.apply_table(table, tr_raw.iloc[hold_idx], hold_price)
    hold_mae2 = mean_absolute_error(y_hold, hold_price2)
    print(f"Blend model    OOF MAE = {oof_mae:.2f}    HOLDOUT MAE = {hold_mae:.2f}")
    print(f"Blend+lookup   HOLDOUT MAE = {hold_mae2:.2f}   <- final test error  (lookup matched {n_hit} rows)")
    json.dump(dict(weights=dict(zip(names, w.tolist())), space=space,
                   oof_mae=float(oof_mae), holdout_mae=float(hold_mae),
                   holdout_mae_lookup=float(hold_mae2), lookup_hits=n_hit,
                   per_model=[(n, o, h) for n, o, h in rows]),
              open(USER_DATA + "blend_eval.json", "w"), indent=2, ensure_ascii=False)
else:
    print(f"Blend model    OOF MAE = {oof_mae:.2f}")

P_test = np.column_stack([store[nm]["test"] for nm in names])
price = blend.predict_blend(P_test, w, space=space)
if np.array_equal(te_raw["SaleID"].to_numpy(), test_ids):
    price, n_hit = vmatch.apply_table(table, te_raw, price)
    print(f"Test-set v lookup matched {n_hit}/{len(price)} rows ({n_hit/len(price)*100:.2f}%)")
else:
    print("Warning: test set from fe.load_raw() does not match test_ids in the npz; skipping lookup post-processing")
os.makedirs(RESULT, exist_ok=True)
out = RESULT + ("predictions.csv" if stage == "final" else "predictions_eval.csv")
pd.DataFrame({"SaleID": test_ids, "price": price}).to_csv(out, index=False)
print(f"\ntestA inference results -> {out}   (min={price.min():.1f}, median={np.median(price):.1f}, max={price.max():.1f})")
