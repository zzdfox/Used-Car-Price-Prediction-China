"""Experiment 11: deeper dive on the NN direction (goal: push holdout below 400)

Stage search (eval protocol): under seed42 compare base / +name-hash / wide / name+wide,
    pick the best by OOF -> retrain the best config with seeds 7/2024 -> average the
    3 seeds in log space, overwrite pred_eval_nn.npz (back up the old file first);
    the best config is stored in exp11_best.json.
Stage final: read exp11_best.json, retrain 3 seeds on all data and average,
    overwrite pred_final_nn.npz.

Usage: exp11_nn.py [search|final]   (run with ~/limix-venv/bin/python)
"""
import sys, os, time, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
import fe

STAGE = sys.argv[1] if len(sys.argv) > 1 else "search"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
USER_DATA = ROOT + "user_data/"
HOLDOUT_SEED, HOLDOUT_FRAC = 20200313, 0.10
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NAME_BUCKETS = 32768
SEEDS = [42, 7, 2024]

CAT_COLS = ["model", "brand", "bodyType", "fuelType", "gearbox",
            "notRepairedDamage", "regionCode"]
CFGS = {
    "base":      dict(name_hash=False, widths=(512, 256, 128)),
    "name":      dict(name_hash=True,  widths=(512, 256, 128)),
    "wide":      dict(name_hash=False, widths=(1024, 512, 256)),
    "name_wide": dict(name_hash=True,  widths=(1024, 512, 256)),
}

# ---------------- Data (same protocol as exp10) ----------------
test_file = fe.TEST_A if STAGE == "search" else fe.TEST_B
tr_raw, te_raw = fe.load_raw(test_file=test_file)
df_all, n_train, y_price, _, _ = fe.build_static_features(tr_raw, te_raw)
df_all = fe.add_group_stats(df_all)
X_num_all = df_all.to_numpy(np.float32)
test_ids = te_raw["SaleID"].to_numpy()

raw_all = pd.concat([tr_raw[CAT_COLS], te_raw[CAT_COLS]], ignore_index=True)
C_base = np.zeros((len(raw_all), len(CAT_COLS)), dtype=np.int64)
cards_base = []
for j, c in enumerate(CAT_COLS):
    codes, uniq = pd.factorize(raw_all[c], use_na_sentinel=True)
    C_base[:, j] = codes + 1
    cards_base.append(len(uniq) + 1)
name_hash = (pd.concat([tr_raw["name"], te_raw["name"]], ignore_index=True)
             .astype("int64") % NAME_BUCKETS).to_numpy()
C_full = np.column_stack([C_base, name_hash])          # last column is the name hash

rng = np.random.RandomState(HOLDOUT_SEED)
perm = rng.permutation(n_train)
k = int(n_train * HOLDOUT_FRAC)
dev_idx, hold_idx = perm[k:], perm[:k]
X_num_train, X_num_test = X_num_all[:n_train], X_num_all[n_train:]
C_train, C_test = C_full[:n_train], C_full[n_train:]
if STAGE == "search":
    fit_idx = dev_idx
    eval_raw = {"hold": (X_num_train[hold_idx], C_train[hold_idx]),
                "test": (X_num_test, C_test)}
else:
    fit_idx = np.arange(n_train)
    eval_raw = {"test": (X_num_test, C_test)}
Xf, Cf, yf = X_num_train[fit_idx], C_train[fit_idx], y_price[fit_idx]
ylog = np.log1p(yf)

# Numeric preprocessing per fold is done once; reused across all configs/seeds
print("Preprocessing numeric features for each fold ...", flush=True)
folds = list(KFold(5, shuffle=True, random_state=2020).split(Xf))
folds_data = []
for tr_i, va_i in folds:
    imp = SimpleImputer(strategy="median")
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             subsample=100000, random_state=0)
    A = qt.fit_transform(imp.fit_transform(Xf[tr_i])).astype(np.float32)
    B = qt.transform(imp.transform(Xf[va_i])).astype(np.float32)
    ev = {nm: qt.transform(imp.transform(Xn)).astype(np.float32)
          for nm, (Xn, _) in eval_raw.items()}
    folds_data.append((tr_i, va_i, A, B, ev, float(ylog[tr_i].mean())))
print(f"Preprocessing done, setup: numeric {Xf.shape[1]} dims, categorical cards={cards_base}+name{NAME_BUCKETS}", flush=True)


class TabNN(nn.Module):
    def __init__(self, n_num, cards, widths):
        super().__init__()
        dims = [min(16, (c + 3) // 4) for c in cards]
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cards, dims)])
        d = n_num + sum(dims)
        layers = []
        for w in widths:
            layers += [nn.Linear(d, w), nn.BatchNorm1d(w), nn.SiLU(), nn.Dropout(0.15)]
            d = w
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, xn, xc):
        e = [emb(xc[:, j]) for j, emb in enumerate(self.embs)]
        return self.net(torch.cat([xn] + e, 1)).squeeze(1)


def to_price(p):
    return np.clip(np.expm1(p), 11.0, 99999.0)


def run_cv(cfg, seed, tag=""):
    """Train one config with 5-fold CV. Returns (oof_log, {name: pred_log}, oof_mae)"""
    torch.manual_seed(seed)
    ncat = len(CAT_COLS) + (1 if cfg["name_hash"] else 0)
    cards = cards_base + ([NAME_BUCKETS] if cfg["name_hash"] else [])
    t0 = time.time()
    oof = np.zeros(len(Xf))
    preds = {nm: np.zeros(len(v[0])) for nm, v in eval_raw.items()}
    for tr_i, va_i, A, B, ev, shift in folds_data:
        m = TabNN(A.shape[1], cards, cfg["widths"]).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
        lossf = nn.L1Loss()
        tXn = torch.tensor(A, device=DEV); tXc = torch.tensor(Cf[tr_i][:, :ncat], device=DEV)
        ty = torch.tensor(ylog[tr_i] - shift, dtype=torch.float32, device=DEV)
        vXn = torch.tensor(B, device=DEV); vXc = torch.tensor(Cf[va_i][:, :ncat], device=DEV)
        n = len(tXn); best, best_state, bad = np.inf, None, 0
        for ep in range(80):
            m.train()
            idx = torch.randperm(n, device=DEV)
            for i in range(0, n, 1024):
                b = idx[i:i + 1024]
                opt.zero_grad()
                lossf(m(tXn[b], tXc[b]), ty[b]).backward()
                opt.step()
            sched.step()
            m.eval()
            with torch.no_grad():
                pv = np.concatenate([m(vXn[i:i + 8192], vXc[i:i + 8192]).cpu().numpy()
                                     for i in range(0, len(vXn), 8192)])
            va = mean_absolute_error(yf[va_i], to_price(pv + shift))
            if va < best - 1e-4:
                best, bad = va, 0
                best_state = {k2: v.detach().clone() for k2, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= 10:
                    break
        m.load_state_dict(best_state); m.eval()
        with torch.no_grad():
            oof[va_i] = np.concatenate([m(vXn[i:i + 8192], vXc[i:i + 8192]).cpu().numpy()
                                        for i in range(0, len(vXn), 8192)]) + shift
            for nm in preds:
                tE = torch.tensor(ev[nm], device=DEV)
                tC = torch.tensor(eval_raw[nm][1][:, :ncat], device=DEV)
                preds[nm] += (np.concatenate(
                    [m(tE[i:i + 8192], tC[i:i + 8192]).cpu().numpy()
                     for i in range(0, len(tE), 8192)]) + shift) / 5
    oof_mae = mean_absolute_error(yf, to_price(oof))
    line = f"  {tag:16s} OOF {oof_mae:8.2f}"
    if "hold" in preds:
        line += f" | HOLDOUT {mean_absolute_error(y_price[hold_idx], to_price(preds['hold'])):8.2f}"
    print(line + f"  ({time.time()-t0:.0f}s)", flush=True)
    return oof, preds, oof_mae


if STAGE == "search":
    results = {}
    for key, cfg in CFGS.items():
        results[key] = run_cv(cfg, seed=42, tag=f"{key}/s42")
    best_key = min(results, key=lambda k2: results[k2][2])
    json.dump(dict(best=best_key, cfg=CFGS[best_key]), open(USER_DATA + "exp11_best.json", "w"))
    print(f"\nBest config: {best_key}, adding seed averaging:", flush=True)
    runs = [results[best_key]]
    for sd in SEEDS[1:]:
        runs.append(run_cv(CFGS[best_key], seed=sd, tag=f"{best_key}/s{sd}"))
    oof = np.mean([r[0] for r in runs], axis=0)
    preds = {nm: np.mean([r[1][nm] for r in runs], axis=0) for nm in runs[0][1]}
    oof_mae = mean_absolute_error(yf, to_price(oof))
    hm = mean_absolute_error(y_price[hold_idx], to_price(preds["hold"]))
    print(f"\n  3-seed average ({best_key}): OOF {oof_mae:.2f} | HOLDOUT {hm:.2f}")
    np.savez_compressed(USER_DATA + "pred_eval_nn.npz",
                        y_fit=yf, dev_idx=dev_idx, hold_idx=hold_idx,
                        test_ids=test_ids, oof=oof, **preds)
    print("overwrote pred_eval_nn.npz (3-seed average)")
else:
    best = json.load(open(USER_DATA + "exp11_best.json"))
    print(f"final: best config {best['best']}, retraining 3 seeds on all data", flush=True)
    runs = [run_cv(best["cfg"], seed=sd, tag=f"{best['best']}/s{sd}") for sd in SEEDS]
    oof = np.mean([r[0] for r in runs], axis=0)
    preds = {nm: np.mean([r[1][nm] for r in runs], axis=0) for nm in runs[0][1]}
    print(f"  3-seed average OOF {mean_absolute_error(yf, to_price(oof)):.2f}")
    np.savez_compressed(USER_DATA + "pred_final_nn.npz",
                        y_fit=yf, dev_idx=dev_idx, hold_idx=hold_idx,
                        test_ids=test_ids, oof=oof, **preds)
    print("overwrote pred_final_nn.npz (3-seed average)")
