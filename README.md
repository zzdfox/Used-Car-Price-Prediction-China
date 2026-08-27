# Used-Car Price Prediction for the Chinese Market

Predict used-car transaction prices on the Chinese second-hand car market
(Tianchi competition 231784, 150k real transaction records). The official
metric is **MAE on raw prices**.

Current result (2026-08-26): **holdout MAE 397.75**, from a twelve-member
ensemble (LightGBM x4 + CatBoost x2 + XGBoost x2 + three neural
architectures: wide embedding MLP with 5-seed averaging, DCN-v2 with 5-seed
averaging, and a base MLP variant) plus duplicate-aware lookup
post-processing. Optimized from a 426.82 baseline in a single documented
campaign.

## Data

| File | Rows | Description |
|---|---|---|
| `data/used_car_train_20200313.csv` | 150,000 | Training set, 31 columns (incl. `price`), space-separated |
| `data/used_car_testA_20200313.csv` | 50,000 | Test set A, 30 columns (**no `price`**) |

`v_0`..`v_14` are anonymized features. `v_3` alone correlates with
`log(price)` at **-0.927** and is the strongest single feature. The raw CSV
files are not included in this repository; download them from the Tianchi
competition page into `data/`.

## Evaluation protocol

**The official test set has no `price` column, so errors cannot be computed
on it directly.** Therefore:

- A fixed **10% holdout (15,000 rows)** is cut from the 150,000 training
  rows (`seed=20200313`). It **never participates in training, tuning,
  feature selection, or blend-weight fitting** and is only used to report
  the final test error.
- The remaining 135,000 rows (dev) are evaluated with 5-fold
  cross-validation; the out-of-fold (OOF) predictions drive model selection
  and blend-weight fitting.

Select on OOF (135k rows, standard error about +/-1.9), not on the holdout
(15k rows, standard error about +/-5.7) -- holdout differences below about
15 are not statistically significant.

## Repository layout

```
src/fe.py            Feature engineering (static features + group statistics)
src/te.py            Target encoding (nested OOF encoding, leak-safe; unused in the final solution)
src/trainer.py       K-fold training core; lgb / cat / xgb / mlp backends
src/blend.py         Non-negative weight blending, optimized on price-scale MAE
src/main.py          Main pipeline (--stage eval / final, incl. blending and lookup)
src/blend_report.py  Aggregate member predictions, blend, produce the submission
src/vmatch.py        Exact-match lookup on duplicate vehicle records (train/test twins)
src/exp10_nn.py      Embedding-NN ensemble member (GPU)
src/exp1..exp16.py   Selection and ablation experiments
run_all.sh etc.      Serial run scripts (CPU-core pinning; run_round*.sh are later rounds)
user_data/           Intermediate artifacts (per-member .npz predictions, experiment logs; generated)
prediction_result/   Submission files (generated)
```

## Running

```bash
# Evaluation: train and report the test error on the holdout
for M in lgb_a lgb_b xgb cat mlp; do
  python3 src/main.py --stage eval --models $M --folds 5
done
python3 src/blend_report.py eval

# Final: retrain on all 150k rows and produce the test-B submission
for M in lgb_a lgb_b xgb cat mlp; do
  python3 src/main.py --stage final --test B --models $M --folds 5
done
python3 src/blend_report.py final
```

On the DGX Spark (GB10) server, pin the performance cores and run models
serially -- use `run_all.sh` / `run_final.sh` directly.

## Solution notes

### 1. Target transform: log1p + Huber

Prices are heavily right-skewed (mean 5,923, median 3,250, range
11-99,999). Train on `log1p(price)`, invert with `expm1` after prediction.
Measured comparison (3 folds, holdout MAE):

| Configuration | OOF | HOLDOUT |
|---|---|---|
| log + Huber | **479.65** | **432.22** |
| log + L1 | 494.54 | 450.55 |
| raw + L1 | 500.16 | 455.39 |
| log + L2 | 527.69 | 485.41 |

In theory the MAE-optimal prediction is the conditional median, and log1p
is monotone and median-preserving, so log+L1 should win; in practice L1
converged too slowly (still climbing at the round cap) and Huber performed
better on LightGBM.

### 2. Data cleaning (all findings measured in EDA)

- `regDate` has **11,347 rows with month `00`** -- naive `to_datetime`
  fails; clipped to 01
- `power` is documented as `[0, 600]` but reaches **19,312** -- clipped,
  with an out-of-range indicator kept
- `notRepairedDamage` value `'-'` (24,324 rows) -> NaN, handled natively by
  the tree models
- `seller` / `offerType` are constant columns -> dropped
- `SaleID` does not overlap between train and test (0-149,999 vs.
  200,000-249,999) -> dropped to avoid a spurious feature

### 3. Features (216 dimensions)

- **Vehicle age** `used_days = creatDate - regDate`, the strongest
  handcrafted feature in this task
- **Anonymous-feature crosses**: all pairwise products of `v_0..v_14`
  (105 columns) plus `v_i x age`
- **Frequency encoding**: counts of `name` / `model` / `regionCode` etc.
  (no target involved, so pooling train+test is legitimate)
- **Group statistics**: mean / deviation / z-score within the same `model`
  (no target, no leakage)

### 3.5 Duplicate lookup post-processing

About 7% of test rows share an **exact** 15-dimensional `v` vector with a
training row (repeated listings of the same physical car; 69.5% of
duplicate groups have identical prices). After blending, replace the
prediction of every matched row with the group's median price
(`src/vmatch.py`): measured -4.05 holdout MAE. Approximate matching
(distance thresholds / business keys) is measurably harmful; only exact
matching works.

### 4. Three key bugs (recorded so they are never repeated)

**Target-encoding self-leakage.** `name` has 99,662 unique values and
**58.9% of the groups contain a single record**. Fitting and transforming
on the same training fold makes `name__mean` equal to the row's own label
(measured correlation with the label: **+0.969**). The model degenerates
into copying answers, and validation MAE collapses from ~480 to **3,239**.
Fix: nested 5-fold OOF encoding inside each training fold
(`te.oof_target_encode`).

> Follow-up: after the fix, target encoding shows **almost no gain**
> (450.79 vs. 450.55) because the anonymous embedding already carries the
> same information. The final solution **does not use target encoding**.

**Uncentered targets silently break XGBoost.** The mean of the log price is
about 7.6 while XGBoost/CatBoost initialize near 0; with gradient-bounded
Huber-type losses, climbing from 0 to 7.6 costs thousands of rounds. In
practice XGBoost early-stopped after **11 rounds with MAE = 94,050** (pure
garbage). Fix: subtract the training-fold mean before fitting, add it back
at prediction time.

| Backend | Uncentered | Centered |
|---|---|---|
| lgb | 565.1 | 564.7 |
| cat | 6,841.4 | **858.2** |
| xgb | **94,050.0** | **616.2** |

**Huber hyperparameters do not transfer across libraries.** LightGBM's
Huber uses a constant unit Hessian, so `alpha=0.6` is harmless; XGBoost and
CatBoost use the true second derivative, and the same `delta=0.6` collapses
the Hessian to about 4e-4, which the L2 regularizer then crushes --
predictions pin to the clipping boundary (XGBoost MAE 5,923 / CatBoost
12,868). Fix: `xgb -> reg:absoluteerror`, `cat -> MAE`; after the fix,
holdout 431.80 / 437.70.

### 5. Early stopping on the real metric

The competition metric is price-scale MAE, but default early stopping
watches log-scale MAE. A custom feval computes MAE on `expm1`-restored
prices instead.

### 6. Blending

LightGBM x3 (different seeds / alpha / regularization) + XGBoost x2 +
CatBoost + embedding-NN + MLP, combined by a non-negative weight search
that minimizes price-scale MAE directly (Nelder-Mead; log-space and
price-space averaging both evaluated, the better kept).

Two lessons: **complementarity is worth more than single-model accuracy**
-- the embedding-NN trails the best booster by 23 points OOF yet takes the
largest weight (0.235+) because its error structure differs from the
trees'; and **cross-library / cross-seed decorrelation is the cheapest
gain** -- repairing the CatBoost/XGBoost losses moved the holdout from
426.8 to 414.4, and seed averaging plus the NN pushed it to 409.3. A
stacking meta-learner and isotonic calibration were both measured as
negative (linear MAE-aligned weights are the right blending layer here).

### 7. Directions that did not work (recorded to avoid re-investment)

| Direction | Outcome |
|---|---|
| Target encoding (13 price statistics groups) | Gain ~0 after leak repair (450.79 vs. 450.55); dropped |
| Larger trees, `num_leaves` 127/255 | 127 slightly worse (479.47 vs. 478.53) and 2x slower |
| Full group statistics (3 keys, 63 columns) | Only +2.4 MAE for ~3x runtime and memory pressure; trimmed to 21 single-key columns |
| Global multiplicative calibration | Optimal multiplier exactly 1.00; no systematic bias to exploit |
| Sample weights `price^p` to align with price-scale MAE | p=0/0.25/0.5 give 474.78/474.58/475.37; no real gain |
| Stacking meta-model (ridge / small LightGBM) | 447.5 / 510.2 vs. linear blend 442.25 -- a tree meta-model staircases continuous base predictions (exp8) |
| Isotonic piecewise calibration | Worse by 17 (price scale) / 6 (log scale); the negative bias conditional on the prediction is the natural signature of a median-optimal model, not miscalibration |
| Expensive-car feature engineering (near-new interactions / brand group stats / within-group ranks) | +1.8 to +4.1 overall and worse in the targeted segment -- the v embedding already encodes it (exp13) |
| NN name-hash embedding (32,768 buckets) | 505 vs. 487; overfits -- `cnt_name` already extracts the value (exp11) |
| FT-Transformer (top-96 tokens, d64 x 3 layers) | 2-seed average 525.8, below the pool threshold, and 15x slower than DCN (exp14) |

The sample-weight entry had a clear theoretical motivation (31% of the
error sits in the most expensive 10% of cars, and
`d(price)/d(log price) = price`, so weighting should align the log-scale
loss with price-scale MAE), but it fails in practice: the variance added by
up-weighting the sparse expensive region cancels the bias reduction.
