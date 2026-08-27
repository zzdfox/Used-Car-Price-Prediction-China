"""Experiment 14: heterogeneous NN members -- DCN-v2 and FT-Transformer (FINDINGS Sec 7.4)

The same-architecture MLP parameter space is exhausted (exp11/12); switch structures
to find error shapes different from the existing nn member:
  dcn  DCN-v2: cross layers learn explicit feature interactions + parallel deep tower,
       full 216 numeric dims
  ftt  FT-Transformer: feature tokenization + attention; numeric features limited to the
       exp7 importance top-96 to control the quadratic token cost (1+96+7=104 tokens,
       d=64, 3 layers)

After multi-seed averaging each member joins the ensemble pool separately:
pred_eval_nn3.npz (dcn) / pred_eval_nn4.npz (ftt); members with average OOF > 500 do not
join the pool. Usage: exp14_nn.py [search|final] (~/limix-venv)
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
FTT_TOPK = 96
CAT_COLS = ["model", "brand", "bodyType", "fuelType", "gearbox",
            "notRepairedDamage", "regionCode"]

# ---------------- data (same protocol as exp10-12) ----------------
test_file = fe.TEST_A if STAGE == "search" else fe.TEST_B
tr_raw, te_raw = fe.load_raw(test_file=test_file)
df_all, n_train, y_price, _, _ = fe.build_static_features(tr_raw, te_raw)
df_all = fe.add_group_stats(df_all)
X_num_all = df_all.to_numpy(np.float32)
test_ids = te_raw["SaleID"].to_numpy()

imp_rank = pd.read_csv(USER_DATA + "feat_importance.csv")
top_cols = [c for c in imp_rank["feature"] if c in df_all.columns][:FTT_TOPK]
FTT_IDX = np.array([df_all.columns.get_loc(c) for c in top_cols])

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
print(f"Preprocessing done (FTT numeric tokens: top-{len(FTT_IDX)})", flush=True)


class DCN(nn.Module):
    def __init__(self, n_num, cards_, n_cross=3, deep=(512, 256), dropout=0.1):
        super().__init__()
        dims = [min(16, (c + 3) // 4) for c in cards_]
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cards_, dims)])
        d0 = n_num + sum(dims)
        self.cross = nn.ModuleList([nn.Linear(d0, d0) for _ in range(n_cross)])
        layers, d = [], d0
        for w in deep:
            layers += [nn.Linear(d, w), nn.BatchNorm1d(w), nn.SiLU(), nn.Dropout(dropout)]
            d = w
        self.deep = nn.Sequential(*layers)
        self.head = nn.Linear(d0 + d, 1)
    def forward(self, xn, xc):
        e = [emb(xc[:, j]) for j, emb in enumerate(self.embs)]
        x0 = torch.cat([xn] + e, 1)
        x = x0
        for lin in self.cross:
            x = x0 * lin(x) + x
        return self.head(torch.cat([x, self.deep(x0)], 1)).squeeze(1)


class FTT(nn.Module):
    def __init__(self, n_num, cards_, d=64, heads=8, nlayers=3, dropout=0.1):
        super().__init__()
        self.num_w = nn.Parameter(torch.randn(n_num, d) * 0.02)
        self.num_b = nn.Parameter(torch.zeros(n_num, d))
        self.cat_embs = nn.ModuleList([nn.Embedding(c, d) for c in cards_])
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads,
                                           dim_feedforward=d * 2, dropout=dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
    def forward(self, xn, xc):
        B = xn.shape[0]
        tn = xn.unsqueeze(-1) * self.num_w + self.num_b
        tc = torch.stack([e(xc[:, j]) for j, e in enumerate(self.cat_embs)], 1)
        t = torch.cat([self.cls.expand(B, -1, -1), tn, tc], 1)
        return self.head(self.enc(t)[:, 0]).squeeze(1)


ARCH = {
    "dcn": dict(cls=DCN, num_idx=None, lr=1e-3, bs=1024, epochs=80, patience=10,
                seeds=[42, 7, 2024, 777, 314159], fname="nn3"),
    "ftt": dict(cls=FTT, num_idx=FTT_IDX, lr=4e-4, bs=1024, epochs=60, patience=8,
                seeds=[42, 7], fname="nn4"),
}
ONLY = sys.argv[2] if len(sys.argv) > 2 else None   # run only the given architecture, e.g. "dcn"


def to_price(p):
    return np.clip(np.expm1(p), 11.0, 99999.0)


def run_cv(arch, seed, tag=""):
    a = ARCH[arch]
    torch.manual_seed(seed)
    t0 = time.time()
    oof = np.zeros(len(Xf))
    preds = {nm: np.zeros(len(v[0])) for nm, v in eval_raw.items()}
    for tr_i, va_i, A, B, ev, shift in folds_data:
        An = A if a["num_idx"] is None else A[:, a["num_idx"]]
        Bn = B if a["num_idx"] is None else B[:, a["num_idx"]]
        m = a["cls"](An.shape[1], cards).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=a["lr"], weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a["epochs"])
        lossf = nn.L1Loss()
        tXn = torch.tensor(An, device=DEV); tXc = torch.tensor(Cf[tr_i], device=DEV)
        ty = torch.tensor(ylog[tr_i] - shift, dtype=torch.float32, device=DEV)
        vXn = torch.tensor(Bn, device=DEV); vXc = torch.tensor(Cf[va_i], device=DEV)
        n = len(tXn); best, best_state, bad = np.inf, None, 0
        for ep in range(a["epochs"]):
            m.train()
            idx = torch.randperm(n, device=DEV)
            for i in range(0, n, a["bs"]):
                b = idx[i:i + a["bs"]]
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
                if bad >= a["patience"]:
                    break
        m.load_state_dict(best_state); m.eval()
        with torch.no_grad():
            oof[va_i] = np.concatenate([m(vXn[i:i + 8192], vXc[i:i + 8192]).cpu().numpy()
                                        for i in range(0, len(vXn), 8192)]) + shift
            for nm in preds:
                E = ev[nm] if a["num_idx"] is None else ev[nm][:, a["num_idx"]]
                tE = torch.tensor(E, device=DEV)
                tC = torch.tensor(eval_raw[nm][1], device=DEV)
                preds[nm] += (np.concatenate(
                    [m(tE[i:i + 8192], tC[i:i + 8192]).cpu().numpy()
                     for i in range(0, len(tE), 8192)]) + shift) / 5
    oof_mae = mean_absolute_error(yf, to_price(oof))
    line = f"  {tag:12s} OOF {oof_mae:8.2f}"
    if "hold" in preds:
        line += f" | HOLDOUT {mean_absolute_error(y_price[hold_idx], to_price(preds['hold'])):8.2f}"
    print(line + f"  ({time.time()-t0:.0f}s)", flush=True)
    return oof, preds, oof_mae


def average_and_save(runs, fname):
    oof = np.mean([r[0] for r in runs], axis=0)
    preds = {nm: np.mean([r[1][nm] for r in runs], axis=0) for nm in runs[0][1]}
    oof_mae = mean_absolute_error(yf, to_price(oof))
    line = f"  {len(runs)}-seed average: OOF {oof_mae:.2f}"
    if "hold" in preds:
        line += f" | HOLDOUT {mean_absolute_error(y_price[hold_idx], to_price(preds['hold'])):.2f}"
    print(line, flush=True)
    if oof_mae > 500:
        print(f"  OOF > 500, {fname} does not join the ensemble pool")
        return False
    np.savez_compressed(USER_DATA + f"pred_{'eval' if STAGE=='search' else 'final'}_{fname}.npz",
                        y_fit=yf, dev_idx=dev_idx, hold_idx=hold_idx,
                        test_ids=test_ids, oof=oof, **preds)
    print(f"  Saved pred_*_{fname}.npz")
    return True


if STAGE == "search":
    # error correlation with the existing nn member (price space), to gauge complementarity
    nn_ref = np.load(USER_DATA + "pred_eval_nn.npz")["oof"]
    kept = {}
    for arch in ["dcn", "ftt"]:
        if ONLY and arch != ONLY:
            continue
        runs = []
        for sd in ARCH[arch]["seeds"]:
            runs.append(run_cv(arch, sd, tag=f"{arch}/s{sd}"))
        err_a = to_price(np.mean([r[0] for r in runs], 0)) - yf
        err_n = to_price(nn_ref) - yf
        print(f"  {arch} error correlation with nn: {np.corrcoef(err_a, err_n)[0,1]:.3f}", flush=True)
        kept[arch] = average_and_save(runs, ARCH[arch]["fname"])
    json.dump(kept, open(USER_DATA + "exp14_kept.json", "w"))
else:
    kept = json.load(open(USER_DATA + "exp14_kept.json"))
    for arch in ["dcn", "ftt"]:
        if ONLY and arch != ONLY:
            continue
        if not kept.get(arch):
            print(f"{arch} not in pool, skipping")
            continue
        runs = [run_cv(arch, sd, tag=f"{arch}/s{sd}") for sd in ARCH[arch]["seeds"]]
        average_and_save(runs, ARCH[arch]["fname"])
