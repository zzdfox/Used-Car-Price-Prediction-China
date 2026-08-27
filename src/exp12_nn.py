"""Experiment 12: NN deeper dive -- variant search (wider / lower dropout) + 5-seed averaging

exp11 conclusion: wide(1024,512,256) is best at 487.15@s42, name hash is a negative gain;
3-seed averaging took OOF 487->462, seed variance is the main noise source -> extend to 5 seeds.
Stage search: try wide_d10 / xwide (s42, same protocol as exp11's wide 487.15, so comparable)
    -> average best config over 5 seeds -> overwrite pred_eval_nn.npz, config saved to exp12_best.json
Stage final: read config, retrain on full data with 5 seeds and average -> overwrite pred_final_nn.npz

Usage: exp12_nn.py [search|final]   (~/limix-venv/bin/python)
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
SEEDS = [42, 7, 2024, 777, 314159]

CAT_COLS = ["model", "brand", "bodyType", "fuelType", "gearbox",
            "notRepairedDamage", "regionCode"]
WIDE_REF = 487.15          # OOF of exp11 wide/s42; same protocol and RNG stream, referenced directly
CFGS = {
    "wide":     dict(widths=(1024, 512, 256), dropout=0.15),
    "wide_d10": dict(widths=(1024, 512, 256), dropout=0.10),
    "xwide":    dict(widths=(2048, 1024, 512), dropout=0.15),
}

# ---------------- data (same protocol as exp10/11) ----------------
test_file = fe.TEST_A if STAGE == "search" else fe.TEST_B
tr_raw, te_raw = fe.load_raw(test_file=test_file)
df_all, n_train, y_price, _, _ = fe.build_static_features(tr_raw, te_raw)
df_all = fe.add_group_stats(df_all)
X_num_all = df_all.to_numpy(np.float32)
test_ids = te_raw["SaleID"].to_numpy()

raw_all = pd.concat([tr_raw[CAT_COLS], te_raw[CAT_COLS]], ignore_index=True)
C_all = np.zeros((len(raw_all), len(CAT_COLS)), dtype=np.int64)
cards = []
for j, c in enumerate(CAT_COLS):
    codes, uniq = pd.factorize(raw_all[c], use_na_sentinel=True)
    C_all[:, j] = codes + 1
    cards.append(len(uniq) + 1)

rng = np.random.RandomState(HOLDOUT_SEED)
perm = rng.permutation(n_train)
k = int(n_train * HOLDOUT_FRAC)
dev_idx, hold_idx = perm[k:], perm[:k]
X_num_train, X_num_test = X_num_all[:n_train], X_num_all[n_train:]
C_train, C_test = C_all[:n_train], C_all[n_train:]
if STAGE == "search":
    fit_idx = dev_idx
    eval_raw = {"hold": (X_num_train[hold_idx], C_train[hold_idx]),
                "test": (X_num_test, C_test)}
else:
    fit_idx = np.arange(n_train)
    eval_raw = {"test": (X_num_test, C_test)}
Xf, Cf, yf = X_num_train[fit_idx], C_train[fit_idx], y_price[fit_idx]
ylog = np.log1p(yf)

print("Preprocessing per-fold numeric features ...", flush=True)
folds_data = []
for tr_i, va_i in KFold(5, shuffle=True, random_state=2020).split(Xf):
    imp = SimpleImputer(strategy="median")
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             subsample=100000, random_state=0)
    A = qt.fit_transform(imp.fit_transform(Xf[tr_i])).astype(np.float32)
    B = qt.transform(imp.transform(Xf[va_i])).astype(np.float32)
    ev = {nm: qt.transform(imp.transform(Xn)).astype(np.float32)
          for nm, (Xn, _) in eval_raw.items()}
    folds_data.append((tr_i, va_i, A, B, ev, float(ylog[tr_i].mean())))
print("Preprocessing done", flush=True)


class TabNN(nn.Module):
    def __init__(self, n_num, cards_, widths, dropout):
        super().__init__()
        dims = [min(16, (c + 3) // 4) for c in cards_]
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cards_, dims)])
        d = n_num + sum(dims)
        layers = []
        for w in widths:
            layers += [nn.Linear(d, w), nn.BatchNorm1d(w), nn.SiLU(), nn.Dropout(dropout)]
            d = w
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, xn, xc):
        e = [emb(xc[:, j]) for j, emb in enumerate(self.embs)]
        return self.net(torch.cat([xn] + e, 1)).squeeze(1)


def to_price(p):
    return np.clip(np.expm1(p), 11.0, 99999.0)


def run_cv(cfg, seed, tag=""):
    torch.manual_seed(seed)
    t0 = time.time()
    oof = np.zeros(len(Xf))
    preds = {nm: np.zeros(len(v[0])) for nm, v in eval_raw.items()}
    for tr_i, va_i, A, B, ev, shift in folds_data:
        m = TabNN(A.shape[1], cards, cfg["widths"], cfg["dropout"]).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
        lossf = nn.L1Loss()
        tXn = torch.tensor(A, device=DEV); tXc = torch.tensor(Cf[tr_i], device=DEV)
        ty = torch.tensor(ylog[tr_i] - shift, dtype=torch.float32, device=DEV)
        vXn = torch.tensor(B, device=DEV); vXc = torch.tensor(Cf[va_i], device=DEV)
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
                tC = torch.tensor(eval_raw[nm][1], device=DEV)
                preds[nm] += (np.concatenate(
                    [m(tE[i:i + 8192], tC[i:i + 8192]).cpu().numpy()
                     for i in range(0, len(tE), 8192)]) + shift) / 5
    oof_mae = mean_absolute_error(yf, to_price(oof))
    line = f"  {tag:16s} OOF {oof_mae:8.2f}"
    if "hold" in preds:
        line += f" | HOLDOUT {mean_absolute_error(y_price[hold_idx], to_price(preds['hold'])):8.2f}"
    print(line + f"  ({time.time()-t0:.0f}s)", flush=True)
    return oof, preds, oof_mae


def average_and_save(runs, fname):
    oof = np.mean([r[0] for r in runs], axis=0)
    preds = {nm: np.mean([r[1][nm] for r in runs], axis=0) for nm in runs[0][1]}
    oof_mae = mean_absolute_error(yf, to_price(oof))
    line = f"\n  {len(runs)}-seed average: OOF {oof_mae:.2f}"
    if "hold" in preds:
        line += f" | HOLDOUT {mean_absolute_error(y_price[hold_idx], to_price(preds['hold'])):.2f}"
    print(line, flush=True)
    np.savez_compressed(USER_DATA + fname, y_fit=yf, dev_idx=dev_idx,
                        hold_idx=hold_idx, test_ids=test_ids, oof=oof, **preds)
    print(f"Overwrote {fname}")


if STAGE == "search":
    scores = {"wide": WIDE_REF}
    trial = {}
    for key in ["wide_d10", "xwide"]:
        trial[key] = run_cv(CFGS[key], seed=42, tag=f"{key}/s42")
        scores[key] = trial[key][2]
    best_key = min(scores, key=lambda k2: scores[k2])
    print(f"\nBest config: {best_key} (wide reference {WIDE_REF}), running 5-seed:", flush=True)
    json.dump(dict(best=best_key, cfg=CFGS[best_key]), open(USER_DATA + "exp12_best.json", "w"))
    runs = []
    for sd in SEEDS:
        if best_key in trial and sd == 42:
            runs.append(trial[best_key]); continue
        runs.append(run_cv(CFGS[best_key], seed=sd, tag=f"{best_key}/s{sd}"))
    average_and_save(runs, "pred_eval_nn.npz")
else:
    best = json.load(open(USER_DATA + "exp12_best.json"))
    print(f"final: config {best['best']}, 5-seed full-data retrain", flush=True)
    runs = [run_cv(best["cfg"], seed=sd, tag=f"{best['best']}/s{sd}") for sd in SEEDS]
    average_and_save(runs, "pred_final_nn.npz")
