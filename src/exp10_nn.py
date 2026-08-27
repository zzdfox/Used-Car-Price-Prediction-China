"""Experiment 10: PyTorch embedding-NN blend member (GPU, run with ~/limix-venv/bin/python)

Motivation: mlp is the only member with low correlation (0.79) to the tree models,
but its quality is poor (OOF 572). Replace it with categorical embeddings + a deeper
network + L1 loss to raise its quality and thus its blend contribution.

Protocol identical to main.py: same holdout split (seed=20200313), same KFold(5, seed=2020),
log1p target centered within each fold, early stopping monitors price-space MAE.
Outputs user_data/pred_{stage}_nn.npz (same format as main.py); blend_report picks it up automatically.

Usage: exp10_nn.py [eval|final]
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
import fe

STAGE = sys.argv[1] if len(sys.argv) > 1 else "eval"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
USER_DATA = ROOT + "user_data/"
HOLDOUT_SEED, HOLDOUT_FRAC = 20200313, 0.10
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42)

CAT_COLS = ["model", "brand", "bodyType", "fuelType", "gearbox",
            "notRepairedDamage", "regionCode"]

# ---------------- Data ----------------
test_file = fe.TEST_A if STAGE == "eval" else fe.TEST_B
tr_raw, te_raw = fe.load_raw(test_file=test_file)
df_all, n_train, y_price, _, _ = fe.build_static_features(tr_raw, te_raw)
df_all = fe.add_group_stats(df_all)
X_num_all = df_all.to_numpy(np.float32)
test_ids = te_raw["SaleID"].to_numpy()

# Categorical encoding: factorize over train+test combined; missing gets its own category
raw_all = pd.concat([tr_raw[CAT_COLS], te_raw[CAT_COLS]], ignore_index=True)
C_all = np.zeros((len(raw_all), len(CAT_COLS)), dtype=np.int64)
cards = []
for j, c in enumerate(CAT_COLS):
    codes, uniq = pd.factorize(raw_all[c], use_na_sentinel=True)
    C_all[:, j] = codes + 1                       # -1 (missing) -> 0
    cards.append(len(uniq) + 1)
print(f"numeric features {X_num_all.shape[1]} dims, categorical {dict(zip(CAT_COLS, cards))}", flush=True)

rng = np.random.RandomState(HOLDOUT_SEED)
perm = rng.permutation(n_train)
k = int(n_train * HOLDOUT_FRAC)
dev_idx, hold_idx = perm[k:], perm[:k]

X_num_train, X_num_test = X_num_all[:n_train], X_num_all[n_train:]
C_train, C_test = C_all[:n_train], C_all[n_train:]
if STAGE == "eval":
    fit_idx = dev_idx
    eval_sets = {"hold": (X_num_train[hold_idx], C_train[hold_idx]),
                 "test": (X_num_test, C_test)}
else:
    fit_idx = np.arange(n_train)
    eval_sets = {"test": (X_num_test, C_test)}
Xf, Cf, yf = X_num_train[fit_idx], C_train[fit_idx], y_price[fit_idx]
ylog = np.log1p(yf)

# ---------------- Model ----------------
class TabNN(nn.Module):
    def __init__(self, n_num, cards):
        super().__init__()
        dims = [min(16, (c + 3) // 4) for c in cards]
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cards, dims)])
        d_in = n_num + sum(dims)
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(),
            nn.Linear(128, 1))
    def forward(self, xn, xc):
        e = [emb(xc[:, j]) for j, emb in enumerate(self.embs)]
        return self.net(torch.cat([xn] + e, 1)).squeeze(1)

def to_price(p_log):
    return np.clip(np.expm1(p_log), 11.0, 99999.0)

def train_fold(Xtr, Ctr, ytr_log, Xva, Cva, yva_price, shift,
               epochs=80, bs=1024, patience=10):
    m = TabNN(Xtr.shape[1], cards).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.L1Loss()
    tXn = torch.tensor(Xtr, device=DEV); tXc = torch.tensor(Ctr, device=DEV)
    ty = torch.tensor(ytr_log - shift, dtype=torch.float32, device=DEV)
    vXn = torch.tensor(Xva, device=DEV); vXc = torch.tensor(Cva, device=DEV)
    n = len(tXn); best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        m.train()
        idx = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            opt.zero_grad()
            loss = lossf(m(tXn[b], tXc[b]), ty[b])
            loss.backward(); opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            pv = np.concatenate([m(vXn[i:i + 8192], vXc[i:i + 8192]).cpu().numpy()
                                 for i in range(0, len(vXn), 8192)])
        va_mae = mean_absolute_error(yva_price, to_price(pv + shift))
        if va_mae < best - 1e-4:
            best, bad = va_mae, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    return m, best, ep + 1

def predict(m, imp, qt, Xn, Xc, shift, bs=8192):
    A = qt.transform(imp.transform(Xn)).astype(np.float32)
    tXn = torch.tensor(A, device=DEV); tXc = torch.tensor(Xc, device=DEV)
    m.eval()
    with torch.no_grad():
        p = np.concatenate([m(tXn[i:i + bs], tXc[i:i + bs]).cpu().numpy()
                            for i in range(0, len(tXn), bs)])
    return p + shift

# ---------------- K-fold ----------------
t0 = time.time()
folds = KFold(5, shuffle=True, random_state=2020)
oof = np.zeros(len(Xf))
preds = {name: np.zeros(len(v[0])) for name, v in eval_sets.items()}
for kf, (tr_i, va_i) in enumerate(folds.split(Xf)):
    shift = float(ylog[tr_i].mean())
    imp = SimpleImputer(strategy="median")
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             subsample=100000, random_state=0)
    A = qt.fit_transform(imp.fit_transform(Xf[tr_i])).astype(np.float32)
    B = qt.transform(imp.transform(Xf[va_i])).astype(np.float32)
    m, va_mae, eps = train_fold(A, Cf[tr_i], ylog[tr_i], B, Cf[va_i],
                                yf[va_i], shift)
    with torch.no_grad():
        tB = torch.tensor(B, device=DEV); tC = torch.tensor(Cf[va_i], device=DEV)
        oof[va_i] = np.concatenate([m(tB[i:i + 8192], tC[i:i + 8192]).cpu().numpy()
                                    for i in range(0, len(tB), 8192)]) + shift
    for name, (Xn, Xc) in eval_sets.items():
        preds[name] += predict(m, imp, qt, Xn, Xc, shift) / 5
    print(f"    fold {kf+1}/5  MAE={va_mae:8.2f}  epochs={eps}  ({time.time()-t0:.0f}s)", flush=True)

oof_mae = mean_absolute_error(yf, to_price(oof))
line = f"  nn: OOF MAE = {oof_mae:.2f}  ({time.time()-t0:.0f}s)"
if STAGE == "eval":
    hm = mean_absolute_error(y_price[hold_idx], to_price(preds["hold"]))
    line += f" | HOLDOUT MAE = {hm:.2f}"
print(line, flush=True)

np.savez_compressed(USER_DATA + f"pred_{STAGE}_nn.npz",
                    y_fit=yf, dev_idx=dev_idx, hold_idx=hold_idx,
                    test_ids=test_ids, oof=oof, **preds)
print(f"saved pred_{STAGE}_nn.npz")
