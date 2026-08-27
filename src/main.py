"""
Used-car transaction price prediction -- main training/inference pipeline.

Evaluation protocol (important):
    The official testA data has no price column, so error cannot be computed directly.
    Therefore a fixed 10% (15000 rows) of the 150k training rows is split off as holdout;
    it never participates in any training, tuning, or blend-weight fitting, and is used
    to report the "test error".
    K-fold cross-validation on dev (135000 rows) yields OOF predictions, used for model
    selection and blend weights.

Usage:
    python3 src/main.py --stage eval    # train + evaluate on holdout (reports test error)
    python3 src/main.py --stage final   # retrain on all 150k rows, run inference on testA and write the submission file
"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error

import fe, trainer, blend, vmatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep

# --- Hardware-related parameters (measured on DGX Spark GB10, see user_data/bench_gb10.md) ---
# The machine interleaves big/little core numbering; performance cores are cpu 5-9,15-19.
# Always pin cores with taskset before running.
# LightGBM / CatBoost are fastest on CPU; XGBoost is the only backend clearly faster on GPU (2.7x).
N_THREADS = int(os.environ.get("UCP_THREADS", "10"))
XGB_DEVICE = os.environ.get("UCP_XGB_DEVICE", "cuda")
USER_DATA = ROOT + "user_data/"
RESULT = ROOT + "prediction_result/"
HOLDOUT_SEED, HOLDOUT_FRAC = 20200313, 0.10

# --------------------------------------------------------------------------
# Model configs (determined by the exp1/exp2/exp3 experiment results)
# --------------------------------------------------------------------------
LGB_BASE = dict(boosting_type="gbdt", objective="huber", alpha=0.6,
                num_leaves=63, learning_rate=0.05, feature_fraction=0.65,
                bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=20,
                lambda_l1=0.0, lambda_l2=2.0, num_threads=N_THREADS, verbosity=-1)

MODELS = {
    "lgb_a":  dict(kind="lgb", params=dict(LGB_BASE, seed=42), num_round=30000, es=500, use_te=False),
    # Seed-averaging members: differ from lgb_a / xgb only in the random seed;
    # decorrelated via the randomness of bagging / column sampling.
    "lgb_c":  dict(kind="lgb", params=dict(LGB_BASE, seed=2024), num_round=30000, es=500, use_te=False),
    "lgb_d":  dict(kind="lgb", params=dict(LGB_BASE, seed=555), num_round=30000, es=500, use_te=False),
    "lgb_b":  dict(kind="lgb", params=dict(LGB_BASE, seed=7, num_leaves=95,
                                           feature_fraction=0.5, min_data_in_leaf=40,
                                           alpha=0.4, lambda_l2=5.0),
                   num_round=30000, es=500, use_te=False, weight_pow=0.25),
    "cat":    dict(kind="cat", params=dict(loss_function="MAE", eval_metric="MAE",
                                           learning_rate=0.08, depth=8, l2_leaf_reg=3.0,
                                           subsample=0.8, bootstrap_type="Bernoulli",
                                           random_seed=42, thread_count=N_THREADS, allow_writing_files=False),
                   num_round=12000, es=400, use_te=False),
    # xgb/cat cannot use Huber: LightGBM's huber uses a constant hessian=1, while the other
    # two use the true second derivative; at the residual scale of log1p(price), delta=0.6
    # collapses the hessian to 4e-4, which gets crushed by lambda (in practice all
    # predictions stuck to the clip boundary). Use L1 instead -- it also happens to be the
    # theoretical optimum for the MAE metric.
    "xgb":    dict(kind="xgb", params=dict(objective="reg:absoluteerror",
                                           eval_metric="mae", max_depth=9, eta=0.05,
                                           subsample=0.8, colsample_bytree=0.65,
                                           min_child_weight=5, reg_lambda=2.0,
                                           nthread=N_THREADS, tree_method="hist",
                                           device=XGB_DEVICE, seed=42),
                   num_round=20000, es=400, use_te=False),
    "xgb_b":  dict(kind="xgb", params=dict(objective="reg:absoluteerror",
                                           eval_metric="mae", max_depth=9, eta=0.05,
                                           subsample=0.8, colsample_bytree=0.65,
                                           min_child_weight=5, reg_lambda=2.0,
                                           nthread=N_THREADS, tree_method="hist",
                                           device=XGB_DEVICE, seed=777),
                   num_round=20000, es=400, use_te=False),
    "mlp":    dict(kind="mlp", params=dict(hidden_layer_sizes=(256, 128, 64), activation="relu",
                                           solver="adam", alpha=1e-4, batch_size=512,
                                           learning_rate_init=1e-3, max_iter=120,
                                           early_stopping=True, n_iter_no_change=12,
                                           validation_fraction=0.1, random_state=42),
                   num_round=0, es=0, use_te=False),
}


def get_data(test_file=None, group_stats=True):
    tr_raw, te_raw = fe.load_raw(test_file=test_file)
    df_all, n_train, y_price, y_log, cats = fe.build_static_features(tr_raw, te_raw)
    if group_stats:
        df_all = fe.add_group_stats(df_all)
    df_train = df_all.iloc[:n_train].reset_index(drop=True)
    df_test = df_all.iloc[n_train:].reset_index(drop=True)
    test_ids = te_raw["SaleID"].to_numpy()
    return df_train, y_price, df_test, test_ids, tr_raw, te_raw


def split_holdout(n):
    rng = np.random.RandomState(HOLDOUT_SEED)
    perm = rng.permutation(n)
    k = int(n * HOLDOUT_FRAC)
    return perm[k:], perm[:k]          # dev_idx, hold_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="eval", choices=["eval", "final"])
    ap.add_argument("--models", default="lgb_a,lgb_b,cat,xgb,mlp")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument("--group-stats", type=int, default=1)
    ap.add_argument("--test", default="A", choices=["A", "B"],
                    help="Which test set to predict: A=testA (preliminary round)  B=testB (final leaderboard)")
    args = ap.parse_args()
    names = [m.strip() for m in args.models.split(",") if m.strip()]

    test_file = fe.TEST_A if args.test == "A" else fe.TEST_B
    df_train, y_price, df_test, test_ids, tr_raw, te_raw = get_data(
        test_file, group_stats=bool(args.group_stats))
    print(f"Test set: {test_file}", flush=True)
    n = len(df_train)
    dev_idx, hold_idx = split_holdout(n)
    print(f"feature_dim={df_train.shape[1]}  train={n}  testA={len(df_test)}", flush=True)

    if args.stage == "eval":
        df_fit = df_train.iloc[dev_idx].reset_index(drop=True); y_fit = y_price[dev_idx]
        df_hold = df_train.iloc[hold_idx].reset_index(drop=True); y_hold = y_price[hold_idx]
        eval_sets = [("hold", df_hold), ("test", df_test)]
        print(f"dev={len(df_fit)}  holdout={len(df_hold)} (holdout never used for training/tuning)\n", flush=True)
    else:
        df_fit, y_fit = df_train, y_price          # final submission: use all 150k rows
        eval_sets = [("test", df_test)]
        print(f"Final stage: retraining on all {len(df_fit)} rows\n", flush=True)

    store, results = {}, []
    for nm in names:
        cfg = MODELS[nm]
        print(f"===== {nm} ({cfg['kind']}) =====", flush=True)
        oof, preds, info = trainer.run_cv(
            df_fit, y_fit, eval_sets, cfg["kind"], cfg["params"],
            n_folds=args.folds, seed=args.seed, use_te=cfg["use_te"],
            num_round=cfg["num_round"], es=cfg["es"],
            weight_pow=cfg.get("weight_pow", 0.0))
        store[nm] = dict(oof=oof, **{k: v for k, v in preds.items()})
        line = f"  {nm}: OOF MAE = {info['oof_mae']:.2f}  ({info['secs']:.0f}s)"
        if "hold" in preds:
            hm = mean_absolute_error(y_hold, trainer.to_price(preds["hold"]))
            line += f" | HOLDOUT MAE = {hm:.2f}"
            results.append((nm, info["oof_mae"], hm))
        else:
            results.append((nm, info["oof_mae"], None))
        print(line + "\n", flush=True)

    tag = args.stage
    for nm, d in store.items():
        np.savez_compressed(USER_DATA + f"pred_{tag}_{nm}.npz",
                            y_fit=y_fit, dev_idx=dev_idx, hold_idx=hold_idx,
                            test_ids=test_ids, **d)
    if len(names) == 1:
        print(f"Saved predictions for {names[0]}; run src/blend_report.py to blend")
        return

    # ---- Blend ----
    P_oof = np.column_stack([store[nm]["oof"] for nm in names])
    best = None
    for space in ("log", "price"):
        w, v = blend.optimize_weights(P_oof, y_fit, space=space)
        print(f"Blend ({space} space) OOF MAE = {v:.2f}  weights = "
              + ", ".join(f"{nm}:{wi:.3f}" for nm, wi in zip(names, w)), flush=True)
        if best is None or v < best[1]:
            best = (w, v, space)
    w, oof_mae, space = best

    # v-vector lookup post-processing (FINDINGS Sec. 4.2):
    # in the eval stage the table is built from dev only (holdout treated as a real test
    # set); in the final stage it is built from all 150k rows.
    table = vmatch.build_table(tr_raw if args.stage == "final" else tr_raw.iloc[dev_idx])

    print("\n" + "=" * 76)
    print(f"{'Model':10s} {'OOF MAE':>12s} {'HOLDOUT MAE':>14s}")
    for nm, om, hm in results:
        print(f"{nm:10s} {om:12.2f} {('%.2f' % hm) if hm is not None else '-':>14s}")
    print("-" * 76)
    if args.stage == "eval":
        P_hold = np.column_stack([store[nm]["hold"] for nm in names])
        hold_price = blend.predict_blend(P_hold, w, space=space)
        hold_mae = mean_absolute_error(y_hold, hold_price)
        hold_price2, n_hit = vmatch.apply_table(table, tr_raw.iloc[hold_idx], hold_price)
        hold_mae2 = mean_absolute_error(y_hold, hold_price2)
        print(f"{'Blend':10s} {oof_mae:12.2f} {hold_mae:14.2f}")
        print(f"{'Blend+LUT':8s} {'-':>12s} {hold_mae2:14.2f}   <- final test error (MAE)  "
              f"(lookup hit {n_hit}/{len(hold_price)} rows)")
        json.dump(dict(weights=dict(zip(names, w.tolist())), space=space,
                       oof_mae=float(oof_mae), holdout_mae=float(hold_mae),
                       holdout_mae_lookup=float(hold_mae2), lookup_hits=n_hit),
                  open(USER_DATA + "blend_eval.json", "w"), indent=2)
    else:
        print(f"{'Blend':10s} {oof_mae:12.2f}")
        json.dump(dict(weights=dict(zip(names, w.tolist())), space=space,
                       oof_mae=float(oof_mae)),
                  open(USER_DATA + "blend_final.json", "w"), indent=2)

    # ---- testA inference ----
    P_test = np.column_stack([store[nm]["test"] for nm in names])
    price = blend.predict_blend(P_test, w, space=space)
    price, n_hit = vmatch.apply_table(table, te_raw, price)
    print(f"\nTest-set v lookup hit {n_hit}/{len(price)} rows ({n_hit/len(price)*100:.2f}%)")
    os.makedirs(RESULT, exist_ok=True)
    out = RESULT + (f"predictions_test{args.test}.csv" if args.stage == "final"
                    else f"predictions_eval_test{args.test}.csv")
    pd.DataFrame({"SaleID": test_ids, "price": price}).to_csv(out, index=False)
    print(f"\ntestA predictions written to: {out}")
    print(f"  predicted price min={price.min():.1f}  median={np.median(price):.1f}  max={price.max():.1f}")


if __name__ == "__main__":
    main()
