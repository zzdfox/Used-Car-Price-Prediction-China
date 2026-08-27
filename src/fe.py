"""
Used-car transaction price prediction -- feature engineering module
- Static features (no price used): dates/car age, power correction, count encoding, anonymous-feature crosses
- Target encoding (uses price): must be computed inside CV folds, see TargetEncoder
"""
import os
import numpy as np
import pandas as pd

# Data directory: defaults to data/ under the project root; override with env var UCP_DATA_DIR
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "UCP_DATA_DIR", os.path.join(os.path.dirname(_HERE), "data") + os.sep
)
TRAIN_FILE = "used_car_train_20200313.csv"
# Test sets: testA is the preliminary-round evaluation set, testB is the final leaderboard set.
# Switch via env var UCP_TEST_FILE or load_raw(test_file=...).
TEST_A = "used_car_testA_20200313.csv"
TEST_B = "used_car_testB_20200421.csv"
TEST_FILE = os.environ.get("UCP_TEST_FILE", TEST_A)

# Anonymous features
V_COLS = [f"v_{i}" for i in range(15)]
# Anonymous features most correlated with log(price) (see EDA)
V_TOP = ["v_3", "v_8", "v_12", "v_0", "v_11", "v_10"]
# Raw categorical columns
CAT_RAW = ["model", "brand", "bodyType", "fuelType", "gearbox", "notRepairedDamage"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_raw(data_dir=DATA_DIR, test_file=None):
    tr = pd.read_csv(data_dir + TRAIN_FILE, sep=" ")
    te = pd.read_csv(data_dir + (test_file or TEST_FILE), sep=" ")
    return tr, te


def _parse_date(s: pd.Series) -> pd.Series:
    """Parse YYYYMMDD integers into dates. 11347 rows in the data have month 00 and need fixing."""
    s = s.astype("int64").astype(str).str.zfill(8)
    y = s.str[:4].astype(int)
    m = s.str[4:6].astype(int).clip(1, 12)   # month 00 -> 01
    d = s.str[6:8].astype(int).clip(1, 31)   # day 00 -> 01
    # Verified: no invalid combos like "Feb 31" in train/test; coerce falls back to NaT, which tree models handle
    return pd.to_datetime(dict(year=y, month=m, day=d), errors="coerce")


# ----------------------------------------------------------------------------
# Static features (no target involved, safe to compute on train+test jointly)
# ----------------------------------------------------------------------------
def build_static_features(tr: pd.DataFrame, te: pd.DataFrame):
    """Return (df_all, n_train, y_log, cat_cols). df_all contains feature columns only."""
    n_train = len(tr)
    y = tr["price"].astype("float64").to_numpy()
    y_log = np.log1p(y)

    df = pd.concat([tr.drop(columns=["price"]), te], axis=0, ignore_index=True)

    # --- 1. Cleaning ---------------------------------------------------------
    # '-' means unknown; convert to NaN (tree models handle missing values natively)
    df["notRepairedDamage"] = (
        df["notRepairedDamage"].replace("-", np.nan).astype("float32")
    )
    # seller / offerType are constant columns, drop them; SaleID is the primary key, train/test do not overlap, drop it
    df = df.drop(columns=["seller", "offerType"])

    # --- 2. Dates and car age ------------------------------------------------
    # Rows whose regDate month is 00 (train 11347 rows / 7.56%, testA 3736 rows):
    # _parse_date clips them to 01, mixing them with genuine January rows, so that
    # information is lost. The "dirtiness" itself carries an independent signal:
    # controlling for registration year, the median price is still 940 lower (100%
    # of years point the same way), and still 527 lower after also controlling for
    # model (90% the same way). Hence keep an explicit flag.
    # (Verified: day==00, abnormal creatDate, and invalid date combos are all 0 in train/test)
    df["reg_month_missing"] = (
        df["regDate"].astype("int64").astype(str).str.zfill(8)
        .str[4:6].astype(int) == 0
    ).astype("float32")

    reg = _parse_date(df["regDate"])
    crt = _parse_date(df["creatDate"])
    df["reg_year"] = reg.dt.year.astype("float32")
    df["reg_month"] = reg.dt.month.astype("float32")
    df["reg_day"] = reg.dt.day.astype("float32")
    df["reg_dow"] = reg.dt.dayofweek.astype("float32")
    df["crt_year"] = crt.dt.year.astype("float32")
    df["crt_month"] = crt.dt.month.astype("float32")
    df["crt_day"] = crt.dt.day.astype("float32")
    df["crt_dow"] = crt.dt.dayofweek.astype("float32")

    # Car age: the strongest hand-crafted feature in this task
    used_days = (crt - reg).dt.days.astype("float32")
    df["used_days"] = used_days
    df["used_years"] = used_days / 365.25
    df["used_months"] = (df["crt_year"] - df["reg_year"]) * 12 + (
        df["crt_month"] - df["reg_month"]
    )
    # Continuous registration year (year + month/12), finer-grained than the plain year
    df["reg_yearmonth"] = df["reg_year"] + (df["reg_month"] - 1) / 12.0
    df["crt_yearmonth"] = df["crt_year"] + (df["crt_month"] - 1) / 12.0
    df["used_days_bin"] = (used_days // 365).astype("float32")   # whole years of age

    # --- 3. Power correction -------------------------------------------------
    # Official field spec says power range is [0, 600]; train max is 19312, i.e. dirty data
    pw = df["power"].astype("float32")
    df["power_raw"] = pw
    df["power_is_zero"] = (pw == 0).astype("float32")
    df["power_is_out"] = ((pw > 600) | (pw <= 0)).astype("float32")
    pw_c = pw.clip(1, 600)
    df["power"] = pw_c
    df["power_log"] = np.log1p(pw_c)
    df["power_bin"] = (pw_c // 10).astype("float32")

    # --- 4. Mileage and derived ratios ---------------------------------------
    km = df["kilometer"].astype("float32")
    df["km_per_year"] = km / (df["used_years"].clip(lower=0.1))
    df["power_per_km"] = pw_c / (km + 1.0)
    df["power_x_years"] = pw_c * df["used_years"]
    df["km_x_years"] = km * df["used_years"]

    # --- 5. Combined categories ----------------------------------------------
    df["model"] = df["model"].fillna(-1).astype("int32")
    df["brand_model"] = (df["brand"].astype("int64") * 1000 + df["model"]).astype("int32")

    # --- 6. Count encoding (no target used; pooling train+test for counting is legitimate) ---
    for c in ["name", "model", "brand", "regionCode", "brand_model", "power_bin", "reg_yearmonth"]:
        df[f"cnt_{c}"] = df[c].map(df[c].value_counts()).astype("float32")
    for cols in [["name", "brand"], ["model", "regionCode"], ["model", "kilometer"]]:
        key = df[cols].astype(str).agg("_".join, axis=1)
        df["cnt_" + "_".join(cols)] = key.map(key.value_counts()).astype("float32")

    # --- 7. Anonymous-feature crosses ----------------------------------------
    V = df[V_COLS].astype("float32")
    df["v_sum"] = V.sum(axis=1)
    df["v_mean"] = V.mean(axis=1)
    df["v_std"] = V.std(axis=1)
    # All pairwise products (15*14/2 = 105); the anonymous features look like embedding
    # vectors, and the cross terms give a clear boost
    for i in range(15):
        for j in range(i + 1, 15):
            df[f"vx_{i}_{j}"] = V.iloc[:, i] * V.iloc[:, j]
    # anonymous features x car age
    for c in V_COLS:
        df[f"vd_{c}"] = V[c] * df["used_days"]
    # strongly correlated anonymous features x power / mileage
    for c in V_TOP:
        df[f"vp_{c}"] = V[c] * pw_c
        df[f"vk_{c}"] = V[c] * km

    df = df.drop(columns=["regDate", "creatDate", "SaleID"]).copy()  # defragment

    cat_cols = [c for c in CAT_RAW if c in df.columns]
    # Cast everything to float32 to save memory (8GB machine)
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
        elif df[c].dtype in ("int64", "int32"):
            df[c] = df[c].astype("float32")

    return df, n_train, y, y_log, cat_cols


# ----------------------------------------------------------------------------
# Group statistics features (no price used, so pooling train+test is legitimate and leak-free)
# Idea: within the same model/series, where do this car's anonymous features, power, mileage stand
# ----------------------------------------------------------------------------
# Keep only the model key: exp4 showed full group stats (63 cols) gain only ~2.4 MAE
# while straining memory and nearly tripling the runtime
GS_KEYS = ["model"]
GS_VARS = ["v_0", "v_3", "v_8", "v_12", "power", "kilometer", "used_days"]


def add_group_stats(df: pd.DataFrame, keys=GS_KEYS, vars_=GS_VARS) -> pd.DataFrame:
    new = {}
    for k in keys:
        g = df.groupby(k, dropna=False)
        for v in vars_:
            m = g[v].transform("mean")
            s = g[v].transform("std")
            new[f"gs_{k}_{v}_mean"] = m.astype("float32")
            new[f"gs_{k}_{v}_diff"] = (df[v] - m).astype("float32")      # deviation from the group mean
            new[f"gs_{k}_{v}_z"] = ((df[v] - m) / (s + 1e-6)).astype("float32")
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
