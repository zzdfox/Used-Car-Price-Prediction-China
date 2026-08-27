"""Experiment 16: CatBoost tuning -- cat is the weakest GBDT member (OOF 491.6 vs others ~467)

Stage search: quick 3-fold comparison of 6 configs; if the best beats base by >3, run the
    full 5-fold eval with that config and save pred_eval_cat_b.npz -- it joins the pool as
    a **new member** (does not replace cat; the old cat's diversity is left for the NM
    weights to arbitrate). Adoption decision saved to exp16_best.json.
Stage final: if adopted, retrain 5-fold on full data and save pred_final_cat_b.npz.

Usage: exp16_cat.py [search|final]
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.metrics import mean_absolute_error
import fe, trainer
from main import get_data, split_holdout, N_THREADS, USER_DATA

STAGE = sys.argv[1] if len(sys.argv) > 1 else "search"

BASE = dict(loss_function="MAE", eval_metric="MAE", learning_rate=0.08, depth=8,
            l2_leaf_reg=3.0, subsample=0.8, bootstrap_type="Bernoulli",
            random_seed=42, thread_count=N_THREADS, allow_writing_files=False)
VARIANTS = {
    "base":      {},
    "depth10":   dict(depth=10),
    "lossguide": dict(grow_policy="Lossguide", max_leaves=63, depth=16),
    "rsm65":     dict(rsm=0.65),
    "l2_1":      dict(l2_leaf_reg=1.0),
    "lg_rsm":    dict(grow_policy="Lossguide", max_leaves=63, depth=16, rsm=0.65),
}

df_train, y_price, df_test, test_ids, tr_raw, te_raw = get_data(
    fe.TEST_A if STAGE == "search" else fe.TEST_B)
dev_idx, hold_idx = split_holdout(len(df_train))

if STAGE == "search":
    df_fit = df_train.iloc[dev_idx].reset_index(drop=True); y_fit = y_price[dev_idx]
    rows = []
    for name, over in VARIANTS.items():
        try:
            oof, _, info = trainer.run_cv(df_fit, y_fit, [], "cat", dict(BASE, **over),
                                          n_folds=3, seed=2020, use_te=False,
                                          num_round=12000, es=300, verbose=False)
        except Exception as e:
            print(f"  {name:10s} failed: {e}", flush=True)
            continue
        rows.append((name, info["oof_mae"], int(np.mean(info["best_iters"])), info["secs"]))
        print(f"  {name:10s} OOF={info['oof_mae']:8.2f}  iters~{int(np.mean(info['best_iters'])):5d}"
              f"  ({info['secs']:.0f}s)", flush=True)
    rows.sort(key=lambda r: r[1])
    base_mae = [r for r in rows if r[0] == "base"][0][1]
    best = rows[0]
    adopt = best[0] != "base" and best[1] < base_mae - 3
    json.dump(dict(adopt=adopt, name=best[0], params=dict(BASE, **VARIANTS[best[0]])),
              open(USER_DATA + "exp16_best.json", "w"))
    print(f"\nBest {best[0]} ({best[1]:.2f}) vs base ({base_mae:.2f})"
          f" -> {'adopted, running full 5-fold' if adopt else 'insufficient gain, not adopted'}", flush=True)
    if adopt:
        df_hold = df_train.iloc[hold_idx].reset_index(drop=True); y_hold = y_price[hold_idx]
        oof, preds, info = trainer.run_cv(df_fit, y_fit,
                                          [("hold", df_hold), ("test", df_test)],
                                          "cat", dict(BASE, **VARIANTS[best[0]]),
                                          n_folds=5, seed=2020, use_te=False,
                                          num_round=12000, es=400)
        hm = mean_absolute_error(y_hold, trainer.to_price(preds["hold"]))
        print(f"  cat_b({best[0]}): OOF {info['oof_mae']:.2f} | HOLDOUT {hm:.2f}")
        np.savez_compressed(USER_DATA + "pred_eval_cat_b.npz", y_fit=y_fit,
                            dev_idx=dev_idx, hold_idx=hold_idx, test_ids=test_ids,
                            oof=oof, **preds)
        print("Saved pred_eval_cat_b.npz")
else:
    cfg = json.load(open(USER_DATA + "exp16_best.json"))
    if not cfg["adopt"]:
        print("cat_b not adopted, skipping")
        sys.exit(0)
    oof, preds, info = trainer.run_cv(df_train, y_price, [("test", df_test)],
                                      "cat", cfg["params"], n_folds=5, seed=2020,
                                      use_te=False, num_round=12000, es=400)
    print(f"  cat_b final: OOF {info['oof_mae']:.2f}")
    np.savez_compressed(USER_DATA + "pred_final_cat_b.npz", y_fit=y_price,
                        dev_idx=dev_idx, hold_idx=hold_idx, test_ids=test_ids,
                        oof=oof, **preds)
    print("Saved pred_final_cat_b.npz")
