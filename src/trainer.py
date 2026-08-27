"""
Training core: in-fold target encoding + K-fold cross-validation for LGB / CatBoost / XGB.
The target variable is always log1p(price); MAE is always computed on the raw price scale.

On the choice of loss function:
    The evaluation metric is MAE on raw prices, whose optimal prediction is the
    conditional median. log1p is a monotonic transform, so
    median(log1p(y)) = log1p(median(y)).
    Therefore "log1p target + L1 loss + expm1 inverse" is theoretically MAE-optimal.
"""
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import te as te_mod


def mae_price(y_true_price, pred_log):
    """pred is in log space; invert it, then compute MAE on raw prices."""
    p = np.expm1(pred_log)
    p = np.clip(p, 11.0, 99999.0)   # actual value range of training-set prices
    return mean_absolute_error(y_true_price, p)


def to_price(pred_log):
    return np.clip(np.expm1(pred_log), 11.0, 99999.0)


# ----------------------------------------------------------------------------
# K-fold training of a single model
# ----------------------------------------------------------------------------
def run_cv(df_dev, y_dev, df_eval_list, model_name, params, n_folds=5, seed=2020,
           use_te=True, num_round=20000, es=300, log_every=0, verbose=True,
           target="log", center=True, weight_pow=0.0):
    """
    df_dev     : dev-set static features (DataFrame)
    y_dev      : raw prices (np.ndarray)
    df_eval_list: list of other sets to predict [(name, df), ...], e.g. holdout / testA
    Returns (oof_log, {name: pred_log}, info)
    """
    # target="log" -> train on log1p(price); target="raw" -> train on raw prices directly
    y_log = np.log1p(y_dev) if target == "log" else y_dev.astype(np.float64)
    n = len(df_dev)
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(n, dtype=np.float64)
    preds = {name: np.zeros(len(d), dtype=np.float64) for name, d in df_eval_list}
    best_iters, fold_maes = [], []
    t0 = time.time()

    for k, (tr_idx, va_idx) in enumerate(folds.split(df_dev)):
        df_tr, df_va = df_dev.iloc[tr_idx], df_dev.iloc[va_idx]
        y_tr_log, y_va_log = y_log[tr_idx], y_log[va_idx]
        # Target centering: CatBoost/XGB start their initial prediction near 0, while the
        # mean log price is about 7.6. Huber-style losses have bounded gradients, so
        # climbing from 0 to 7.6 burns many rounds (observed: xgb early-stopped after 17
        # rounds). After subtracting the training-fold mean, 0 becomes a sensible starting
        # point and all backends converge consistently.
        shift = float(np.mean(y_tr_log)) if center else 0.0
        y_tr_log, y_va_log = y_tr_log - shift, y_va_log - shift

        if use_te:
            # Training fold uses inner-KFold OOF encoding (prevents self-leakage);
            # validation/test use the encoding fit on the whole training fold.
            te_tr, enc, te_cols = te_mod.oof_target_encode(df_tr, y_tr_log, seed=seed + k)
            X_tr = np.hstack([df_tr.to_numpy(np.float32), te_tr])
            X_va = np.hstack([df_va.to_numpy(np.float32),
                              enc.transform(df_va)[te_cols].to_numpy(np.float32)])
            X_ev = {name: np.hstack([d.to_numpy(np.float32),
                                     enc.transform(d)[te_cols].to_numpy(np.float32)])
                    for name, d in df_eval_list}
        else:
            X_tr = df_tr.to_numpy(np.float32)
            X_va = df_va.to_numpy(np.float32)
            X_ev = {name: d.to_numpy(np.float32) for name, d in df_eval_list}

        # Sample weights: the competition metric is MAE in price space, while we train in
        # log space. Since d(price)/d(logprice)=price, weighting the log-space loss by
        # price^p pulls the optimization target toward price-space MAE (p=1 aligns exactly
        # in theory).
        if weight_pow > 0:
            w_tr = np.power(y_dev[tr_idx], weight_pow)
            w_tr = w_tr / w_tr.mean()
        else:
            w_tr = None

        if model_name == "lgb":
            va_pred, ev_pred, bi = _fit_lgb(X_tr, y_tr_log, X_va, y_va_log, X_ev,
                                            params, num_round, es, log_every,
                                            price_metric=(target == "log"), shift=shift,
                                            w_tr=w_tr)
        elif model_name == "cat":
            va_pred, ev_pred, bi = _fit_cat(X_tr, y_tr_log, X_va, y_va_log, X_ev,
                                            params, num_round, es, log_every)
        elif model_name == "mlp":
            va_pred, ev_pred, bi = _fit_mlp(X_tr, y_tr_log, X_va, y_va_log, X_ev,
                                            params, num_round, es, log_every)
        elif model_name == "xgb":
            va_pred, ev_pred, bi = _fit_xgb(X_tr, y_tr_log, X_va, y_va_log, X_ev,
                                            params, num_round, es, log_every)
        else:
            raise ValueError(model_name)

        va_pred = va_pred + shift          # undo centering first, then store OOF (order matters)
        oof[va_idx] = va_pred
        for name in preds:
            preds[name] += (ev_pred[name] + shift) / n_folds
        best_iters.append(bi)
        fm = (mae_price(y_dev[va_idx], va_pred) if target == "log"
              else mean_absolute_error(y_dev[va_idx], np.clip(va_pred, 11.0, 99999.0)))
        fold_maes.append(fm)
        if verbose:
            print(f"    fold {k+1}/{n_folds}  MAE={fm:8.2f}  best_iter={bi}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        del X_tr, X_va, X_ev, df_tr, df_va
        gc.collect()

    oof_mae = (mae_price(y_dev, oof) if target == "log"
               else mean_absolute_error(y_dev, np.clip(oof, 11.0, 99999.0)))
    info = {"best_iters": best_iters, "fold_maes": fold_maes,
            "oof_mae": oof_mae, "secs": time.time() - t0}
    return oof, preds, info


# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------
def _make_price_mae(shift=0.0):
    """Early stopping must track the real competition metric: MAE in raw price space, not in log space."""
    def _f(preds, ds):
        y = ds.get_label() + shift
        p = np.clip(np.expm1(preds + shift), 11.0, 99999.0)
        return "price_mae", mean_absolute_error(np.expm1(y), p), False
    return _f


def _fit_lgb(X_tr, y_tr, X_va, y_va, X_ev, params, num_round, es, log_every,
             price_metric=True, shift=0.0, w_tr=None):
    import lightgbm as lgb
    params = dict(params)
    feval = None
    if price_metric:
        params["metric"] = "None"          # early-stop only on the custom price-space MAE
        feval = _make_price_mae(shift)
    dtr = lgb.Dataset(X_tr, y_tr, weight=w_tr)
    dva = lgb.Dataset(X_va, y_va, reference=dtr)   # validation set unweighted; early stopping tracks the true metric
    cbs = [lgb.early_stopping(es, verbose=False)]
    if log_every:
        cbs.append(lgb.log_evaluation(log_every))
    m = lgb.train(params, dtr, num_round, valid_sets=[dva], callbacks=cbs, feval=feval)
    bi = m.best_iteration or num_round
    return (m.predict(X_va, num_iteration=bi),
            {k: m.predict(v, num_iteration=bi) for k, v in X_ev.items()}, bi)


def _fit_cat(X_tr, y_tr, X_va, y_va, X_ev, params, num_round, es, log_every):
    from catboost import CatBoostRegressor, Pool
    p = dict(params); p["iterations"] = num_round
    m = CatBoostRegressor(**p)
    m.fit(Pool(X_tr, y_tr), eval_set=Pool(X_va, y_va), use_best_model=True,
          early_stopping_rounds=es, verbose=log_every if log_every else False)
    bi = int(m.get_best_iteration() or num_round)
    return (m.predict(X_va), {k: m.predict(v) for k, v in X_ev.items()}, bi)


def _fit_xgb(X_tr, y_tr, X_va, y_va, X_ev, params, num_round, es, log_every):
    import xgboost as xgb
    dtr = xgb.DMatrix(X_tr, y_tr); dva = xgb.DMatrix(X_va, y_va)
    m = xgb.train(params, dtr, num_round, evals=[(dva, "va")],
                  early_stopping_rounds=es, verbose_eval=log_every if log_every else False)
    bi = int(m.best_iteration) + 1
    rng = (0, bi)
    return (m.predict(dva, iteration_range=rng),
            {k: m.predict(xgb.DMatrix(v), iteration_range=rng) for k, v in X_ev.items()}, bi)


def _fit_mlp(X_tr, y_tr, X_va, y_va, X_ev, params, num_round, es, log_every):
    """Neural-net component: mainly provides error structure decorrelated from the tree models, improving the blend."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.impute import SimpleImputer

    imp = SimpleImputer(strategy="median")
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             subsample=100000, random_state=0)
    A = qt.fit_transform(imp.fit_transform(X_tr))
    B = qt.transform(imp.transform(X_va))
    m = MLPRegressor(**params)
    m.fit(A, y_tr)
    return (m.predict(B),
            {k: m.predict(qt.transform(imp.transform(v))) for k, v in X_ev.items()},
            int(m.n_iter_))
