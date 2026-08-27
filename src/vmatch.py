"""
Exact-match v-vector lookup post-processing (see doc/FINDINGS.md section 4.2).

For 7.54% of testA rows, the 15-dim anonymous vector appears verbatim in train
(duplicate records of the same car); within train, 69.5% of these duplicate groups
have exactly identical prices. For matched rows, replace the model prediction with
the group's median price; measured net gain on holdout is about -2.75 MAE (the model
already predicts matched rows fairly well: 163.8 vs 118.9 for the lookup).
Approximate matching (v-distance threshold / name+regDate business key) was verified
to be net-negative, so only exact matching is done.
"""
import numpy as np

V_COLS = [f"v_{i}" for i in range(15)]


def build_table(tr_raw):
    """v vector -> in-group median price. tr_raw must contain the price column.
    In the eval stage only dev rows may be passed (holdout is treated as a real test set
    and must not enter the table)."""
    return tr_raw.groupby(V_COLS, sort=False)["price"].median().rename("v_price")


def apply_table(table, raw_rows, price_pred):
    """For rows in raw_rows whose v vector exactly matches the table, replace price_pred
    with the table value. Returns (new predictions, number of matched rows)."""
    m = raw_rows[V_COLS].merge(table.reset_index(), on=V_COLS, how="left")
    looked = m["v_price"].to_numpy()
    hit = ~np.isnan(looked)
    out = np.asarray(price_pred, dtype=float).copy()
    out[hit] = looked[hit]
    return out, int(hit.sum())
