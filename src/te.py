"""
Target encoding -- must be fitted inside folds, otherwise it leaks badly.
All statistics are computed on log1p(price).
"""
import numpy as np
import pandas as pd

# (group keys, statistics) -- high-cardinality keys get only lightweight stats to avoid overfitting
TE_SPECS = [
    (("model",),                      ["mean", "median", "std", "min", "max", "count", "q25", "q75"]),
    (("brand",),                      ["mean", "median", "std", "min", "max", "count", "q25", "q75"]),
    (("brand_model",),                ["mean", "median", "std", "count"]),
    (("regionCode",),                 ["mean", "median", "std", "count"]),
    (("name",),                       ["mean", "median", "count"]),
    (("model", "gearbox"),            ["mean", "median", "count"]),
    (("model", "notRepairedDamage"),  ["mean", "median", "count"]),
    (("model", "bodyType"),           ["mean", "median", "count"]),
    (("model", "used_days_bin"),      ["mean", "median", "std", "count"]),
    (("brand", "used_days_bin"),      ["mean", "median", "std", "count"]),
    (("model", "power_bin"),          ["mean", "median", "count"]),
    (("model", "kilometer"),          ["mean", "median", "count"]),
    (("bodyType", "fuelType", "gearbox"), ["mean", "median", "count"]),
]

_AGG = {
    "mean": "mean", "median": "median", "std": "std",
    "min": "min", "max": "max", "count": "count",
    "q25": lambda x: x.quantile(0.25),
    "q75": lambda x: x.quantile(0.75),
}


class TargetEncoder:
    """Fit on the training fold; transform on train/validation/test."""

    def __init__(self, specs=TE_SPECS, smooth=20.0):
        self.specs = specs
        self.smooth = smooth

    def fit(self, df: pd.DataFrame, y_log: np.ndarray):
        self.prior_ = float(np.mean(y_log))
        self.tables_ = []
        # Take only the group-key columns, not a full-table copy (194 cols -> 12 cols,
        # saving hundreds of MB of memory shuffling per fold)
        key_cols = sorted({c for keys, _ in self.specs for c in keys})
        tmp = df[key_cols].copy()
        tmp["__y"] = y_log
        for keys, stats in self.specs:
            keys = list(keys)
            agg_map = {f"{'_'.join(keys)}__{s}": _AGG[s] for s in stats}
            g = tmp.groupby(keys, dropna=False)["__y"].agg(**agg_map)
            # Bayesian smoothing on the mean to suppress noise from small groups
            mean_c = f"{'_'.join(keys)}__mean"
            cnt_c = f"{'_'.join(keys)}__count"
            if mean_c in g.columns and cnt_c in g.columns:
                n = g[cnt_c].to_numpy()
                g[f"{'_'.join(keys)}__smean"] = (
                    g[mean_c].to_numpy() * n + self.prior_ * self.smooth
                ) / (n + self.smooth)
            self.tables_.append((keys, g.reset_index()))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = []
        idx = df.index
        for keys, table in self.tables_:
            merged = df[keys].merge(table, on=keys, how="left")
            merged.index = idx
            out.append(merged.drop(columns=keys))
        res = pd.concat(out, axis=1).astype("float32")
        # Unseen categories -> fill mean-like columns with the prior, leave the rest NaN for tree models
        for c in res.columns:
            if c.endswith("__mean") or c.endswith("__smean") or c.endswith("__median"):
                res[c] = res[c].fillna(self.prior_)
            elif c.endswith("__count"):
                res[c] = res[c].fillna(0.0)
        # Derived: relative difference (vehicle vs the price level of its group)
        return res

    def fit_transform(self, df, y_log):
        return self.fit(df, y_log).transform(df)


TE_KEY_COLS = sorted({c for keys, _ in TE_SPECS for c in keys})


def oof_target_encode(df_tr, y_tr_log, inner_folds=5, seed=777, specs=TE_SPECS, smooth=20.0):
    """
    Run another K-fold target encoding inside the fold to remove "own-label leakage".

    Fitting on the training fold and transforming it directly would make the encoded
    value of count==1 groups (name covers 58.9% of this data) equal to that row's own
    label; the model degenerates into "copying the answer" and collapses immediately
    on the validation set. Here the training fold is split again into inner_folds,
    and each fold's encoding is computed only from the other folds' labels.

    Returns (OOF encoding matrix for train, encoder fitted on the whole training fold, column names)
    """
    from sklearn.model_selection import KFold

    enc_full = TargetEncoder(specs=specs, smooth=smooth).fit(df_tr, y_tr_log)
    cols = list(enc_full.transform(df_tr.iloc[:2]).columns)

    out = np.empty((len(df_tr), len(cols)), dtype=np.float32)
    kf = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    for in_tr, in_va in kf.split(df_tr):
        e = TargetEncoder(specs=specs, smooth=smooth).fit(df_tr.iloc[in_tr], y_tr_log[in_tr])
        out[in_va] = e.transform(df_tr.iloc[in_va])[cols].to_numpy(np.float32)
    return out, enc_full, cols
