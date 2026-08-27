"""Blend weight search: non-negative weight optimization targeting MAE on raw prices directly."""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error


def _apply(P_log, w, space):
    if space == "log":                       # log-space weighting = geometric mean, preserves the median property
        return np.clip(np.expm1(P_log @ w), 11.0, 99999.0)
    return np.clip((np.expm1(P_log) @ w), 11.0, 99999.0)   # price-space weighting


def optimize_weights(P_log, y_price, space="log"):
    """P_log: (n_samples, n_models) each model's predictions in log space. Returns (w, mae)."""
    k = P_log.shape[1]

    def obj(w):
        w = np.abs(w); s = w.sum()
        if s <= 0:
            return 1e9
        return mean_absolute_error(y_price, _apply(P_log, w / s, space))

    best_w, best_v = None, np.inf
    starts = [np.ones(k) / k] + [np.eye(k)[i] * 0.7 + np.ones(k) * 0.3 / k for i in range(k)]
    for x0 in starts:
        r = minimize(obj, x0, method="Nelder-Mead",
                     options=dict(maxiter=3000, xatol=1e-4, fatol=1e-4))
        if r.fun < best_v:
            best_v, best_w = r.fun, np.abs(r.x) / np.abs(r.x).sum()
    return best_w, best_v


def eval_blend(P_log, w, y_price, space="log"):
    return mean_absolute_error(y_price, _apply(P_log, w, space))


def predict_blend(P_log, w, space="log"):
    return _apply(P_log, w, space)
