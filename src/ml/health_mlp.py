import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import numpy as np
import pandas as pd
from db_connection import get_engine
from health_multi import SETS, pooled
 
# ── PARAMETERS ───────────────────────────────────────────────────────────────
FEATURES      = SETS["P: professor (7 vars)"]   # same features as the professor
HIDDEN        = (32, 16)                        # neurons per hidden layer
L2_ALPHA      = 1e-3                            # weight penalty
LEARNING_RATE = 1e-3                            # Adam
BATCH_SIZE    = 200
MAX_EPOCHS    = 2000
PATIENCE      = 30                              # early stopping
VAL_FRACTION  = 0.15                            # share of training batteries used for validation
SEED          = 42
REG_ALPHAS    = [1e-3, 1e-1, 1.0, 10.0]         # penalties tested in regression
N_SEEDS       = 3                               # networks averaged for the final prediction
 
# Health classes (as defined by the professor)
CLASS_NAMES = ["Degraded (<=0.8)", "Moderate (0.8-0.9]", "Healthy (>0.9)"]
 
 
def health_class(h):
    return np.where(h <= 0.8, 0, np.where(h <= 0.9, 1, 2))
 
 
# ── MLP in NumPy ─────────────────────────────────────────────────────────────
class MLP:
    def __init__(self, n_in, n_out, task, seed, alpha=L2_ALPHA):
        self.task = task                        # "clf" or "reg"
        self.alpha = alpha                      # weight penalty
        rng = np.random.default_rng(seed)
        sizes = [n_in, *HIDDEN, n_out]
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.rng = rng
 
    def forward(self, X):
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            if i < len(self.W) - 1:
                z = np.maximum(z, 0.0)          # ReLU
            acts.append(z)
        return acts
 
    def output(self, X):
        z = self.forward(X)[-1]
        if self.task == "clf":
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=1, keepdims=True)
        return z[:, 0]
 
    def loss(self, X, y):
        out = self.output(X)
        if self.task == "clf":
            data = -np.mean(np.log(out[np.arange(len(y)), y] + 1e-12))
        else:
            data = 0.5 * np.mean((out - y) ** 2)
        return data + 0.5 * self.alpha * sum((W ** 2).sum() for W in self.W) / len(y)
 
    def gradients(self, X, y):
        acts = self.forward(X)
        n = len(X)
        if self.task == "clf":
            z = acts[-1] - acts[-1].max(axis=1, keepdims=True)
            p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
            p[np.arange(n), y] -= 1.0
            delta = p / n
        else:
            delta = (acts[-1][:, 0] - y)[:, None] / n
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ delta + self.alpha * self.W[i] / n
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (acts[i] > 0)
        return gW, gb
 
    def train(self, X, y, Xv, yv):
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        best, best_state, wait = np.inf, None, 0
 
        for epoch in range(MAX_EPOCHS):
            idx = self.rng.permutation(len(X))
            for s in range(0, len(X), BATCH_SIZE):
                bi = idx[s:s + BATCH_SIZE]
                gW, gb = self.gradients(X[bi], y[bi])
                grads = gW + gb
                t += 1
                for k, (p, g) in enumerate(zip(params, grads)):
                    m[k] = b1 * m[k] + (1 - b1) * g
                    v[k] = b2 * v[k] + (1 - b2) * g * g
                    mh = m[k] / (1 - b1 ** t)
                    vh = v[k] / (1 - b2 ** t)
                    p -= LEARNING_RATE * mh / (np.sqrt(vh) + eps)
 
            val = self.loss(Xv, yv)
            if val < best - 1e-6:
                best, wait = val, 0
                best_state = ([W.copy() for W in self.W], [b.copy() for b in self.b])
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
 
        self.W, self.b = best_state
        self.epochs = epoch + 1
        return self
 
 
# ── LOBO ─────────────────────────────────────────────────────────────────────
def run_lobo(df, task):
    X_all = df[FEATURES].values.astype(float)
    h_all = df["Health"].values.astype(float)
    y_all = health_class(h_all) if task == "clf" else h_all
    g_all = df["Battery_ID"].values
    preds = np.zeros(len(df)) if task == "reg" else np.zeros(len(df), dtype=int)
    base = np.zeros(len(df)) if task == "reg" else np.zeros(len(df), dtype=int)
    epochs, alphas = [], []
 
    for i, b in enumerate(sorted(np.unique(g_all))):
        te = g_all == b
        tr_batt = np.array(sorted(np.unique(g_all[~te])))
 
        # Validation: whole batteries from the training set (not random cycles)
        rng = np.random.default_rng(SEED + i)
        n_val = max(1, int(round(VAL_FRACTION * len(tr_batt))))
        val_batt = rng.choice(tr_batt, n_val, replace=False)
        va = np.isin(g_all, val_batt)
        tr = ~te & ~va
 
        mu = X_all[tr].mean(axis=0)
        sd = X_all[tr].std(axis=0)
        sd[sd == 0] = 1.0
        Z = (X_all - mu) / sd
 
        if task == "clf":
            net = MLP(len(FEATURES), 3, task, SEED + i).train(
                Z[tr], y_all[tr], Z[va], y_all[va])
            epochs.append(net.epochs)
            preds[te] = net.output(Z[te]).argmax(axis=1)
            base[te] = np.bincount(y_all[~te], minlength=3).argmax()
        else:
            # Standardize the target with the training mean and std
            y_mu, y_sd = y_all[tr].mean(), y_all[tr].std()
            y_z = (y_all - y_mu) / y_sd

            # Choose the penalty by the error on the validation batteries
            best_a, best_v = REG_ALPHAS[0], np.inf
            for a in REG_ALPHAS:
                net = MLP(len(FEATURES), 1, task, SEED + i, a).train(
                    Z[tr], y_z[tr], Z[va], y_z[va])
                v = np.mean(np.abs(net.output(Z[va]) - y_z[va]))
                if v < best_v:
                    best_a, best_v = a, v
            alphas.append(best_a)

            # Average of N_SEEDS networks with the chosen penalty
            outs = []
            for s in range(N_SEEDS):
                net = MLP(len(FEATURES), 1, task, SEED + i + 1000 * s, best_a).train(
                    Z[tr], y_z[tr], Z[va], y_z[va])
                epochs.append(net.epochs)
                outs.append(net.output(Z[te]))
            preds[te] = np.mean(outs, axis=0) * y_sd + y_mu
            base[te] = y_all[~te].mean()
 
    return y_all, preds, base, g_all, epochs, alphas
 
 
def f1_macro(y, p, k=3):
    f1s = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return np.mean(f1s)
 
 
def report_clf(y, p, base, g):
    print("\n── 1) MLP CLASSIFICATION, leave-one-battery-out ──")
    print("Class counts:", dict(zip(CLASS_NAMES, np.bincount(y, minlength=3))))
    acc_b = np.mean([np.mean(p[g == b] == y[g == b]) for b in np.unique(g)])
    acc_bb = np.mean([np.mean(base[g == b] == y[g == b]) for b in np.unique(g)])
    print(f"{'Model':>22} {'Accuracy':>9} {'F1-macro':>9} {'Acc_b':>7}")
    print(f"{'Baseline (most freq.)':>22} {np.mean(base == y):>9.4f} "
          f"{f1_macro(y, base):>9.4f} {acc_bb:>7.4f}")
    print(f"{'MLP 32-16':>22} {np.mean(p == y):>9.4f} "
          f"{f1_macro(y, p):>9.4f} {acc_b:>7.4f}")
    print("Acc_b: mean accuracy per battery")
 
    cm = np.zeros((3, 3), dtype=int)
    for t, q in zip(y, p):
        cm[t, q] += 1
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=["Degraded", "Moderate", "Healthy"])
          .to_string())
 
 
def report_reg(y, p, base, g):
    print("\n── 2) MLP REGRESSION, leave-one-battery-out ──")
    mae_m = np.array([np.mean(np.abs(y[g == b] - p[g == b])) for b in np.unique(g)])
    mae_b = np.array([np.mean(np.abs(y[g == b] - base[g == b])) for b in np.unique(g)])
    rmse, mae, r2 = pooled(y, p)
    b_rmse, b_mae, b_r2 = pooled(y, base)
    print(f"{'Model':>22} {'MAE_b':>7} {'Wins':>6} {'RMSE':>7} {'MAE':>7} {'R2':>7}")
    print(f"{'Baseline (mean)':>22} {mae_b.mean():>7.4f} {'':>6} "
          f"{b_rmse:>7.4f} {b_mae:>7.4f} {b_r2:>7.3f}")
    print(f"{'MLP 32-16':>22} {mae_m.mean():>7.4f} "
          f"{int((mae_m < mae_b).sum()):>3}/{len(mae_m):<2} "
          f"{rmse:>7.4f} {mae:>7.4f} {r2:>7.3f}")

    rows = []
    for b in np.unique(g):
        m = g == b
        rows.append({"Battery": b, "Cycles": int(m.sum()),
                     "MAE": np.mean(np.abs(y[m] - p[m])),
                     "Baseline": np.mean(np.abs(y[m] - base[m])),
                     "Actual_min": y[m].min(), "Actual_max": y[m].max(),
                     "Pred_min": p[m].min(), "Pred_max": p[m].max()})
    print("\nPer battery (MLP regression):")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
 
 
def main():
    engine = get_engine()
    if engine is None:
        return
    df = pd.read_sql("SELECT * FROM CYCLE_HEALTH", engine)
    df = df.dropna(subset=FEATURES + ["Health"]).sort_values(
        ["Battery_ID", "Cycle_Index"]).reset_index(drop=True)
    print(f"Cycles: {len(df)} | Batteries: {df['Battery_ID'].nunique()}")
    print(f"Features: {FEATURES}")
 
    y_c, p_c, base_c, g, ep_c, _ = run_lobo(df, "clf")
    report_clf(y_c, p_c, base_c, g)
    print(f"Epochs (median): {int(np.median(ep_c))}")
 
    y_r, p_r, base_r, g, ep_r, al_r = run_lobo(df, "reg")
    report_reg(y_r, p_r, base_r, g)
    print(f"Epochs (median): {int(np.median(ep_r))}")
    print("Chosen alpha per fold:", dict(zip(*np.unique(al_r, return_counts=True))))
 
    out = pd.DataFrame({
        "Battery_ID": df["Battery_ID"], "Cycle_Index": df["Cycle_Index"],
        "Health_actual": y_r, "Pred_MLP_reg": p_r, "Pred_baseline": base_r,
        "Class_true": y_c, "Class_MLP": p_c,
    })
    out.to_sql("HEALTH_PRED_MLP", con=engine, if_exists="replace",
               index=False, chunksize=200)
    print("\nSaved predictions to HEALTH_PRED_MLP.")
 
 
if __name__ == "__main__":
    main()
 