import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from db_connection import get_engine

# Υποσύνολα μεταβλητών (Temp_Max μόνο στο P: συσχέτιση 0.99 με Temp_Mean)
SETS = {
    "M1: Cycles":                    ["Cycle_Index"],
    "M2a: +Voltage_Mean":            ["Cycle_Index", "Voltage_Mean"],
    "M2b: +Voltage_Mean,Temp_Mean":  ["Cycle_Index", "Voltage_Mean", "Temp_Mean"],
    "M2c: all (no capacity)":        ["Cycle_Index", "Discharge_Time", "Temp_Mean",
                                      "Voltage_Min", "Voltage_Mean", "Current_Mean"],
    "M2d: all minus Discharge_Time": ["Cycle_Index", "Temp_Mean",
                                      "Voltage_Min", "Voltage_Mean", "Current_Mean"],
    "M2e: no Current_Mean":          ["Cycle_Index", "Discharge_Time", "Temp_Mean",
                                      "Voltage_Min", "Voltage_Mean"],
    "P: professor (7 vars)":         ["Cycle_Index", "Discharge_Time", "Voltage_Mean",
                                      "Voltage_Min", "Temp_Mean", "Temp_Max",
                                      "Current_Mean"],
    "B: all + Capacity (circular)":  ["Capacity_Ah", "Cycle_Index", "Discharge_Time",
                                      "Temp_Mean", "Voltage_Min", "Voltage_Mean",
                                      "Current_Mean"],
}

# Το μοντέλο που ελέγχεται μία-μία σε όλες τις μπαταρίες
CHECK_MODEL = "M2e: no Current_Mean"

# Τιμές alpha για το Ridge (ίδιες με τον καθηγητή)
ALPHAS = np.logspace(-3, 3, 25)


def fit(X, y):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    coef = np.linalg.lstsq(Z, y, rcond=None)[0]
    return coef, mu, sd


def predict(model, X):
    coef, mu, sd = model
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    return Z @ coef

def fit_ridge(X, y, alpha):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    P = alpha * np.eye(Z.shape[1])
    P[0, 0] = 0.0                      # ο σταθερός όρος δεν τιμωρείται
    coef = np.linalg.solve(Z.T @ Z + P, Z.T @ y)
    return coef, mu, sd


def choose_alpha(X, y, g):
    # Εσωτερικό leave-one-battery-out μόνο στις μπαταρίες εκπαίδευσης
    best_a, best_s = ALPHAS[0], np.inf
    for a in ALPHAS:
        maes = []
        for b in np.unique(g):
            m = g == b
            model = fit_ridge(X[~m], y[~m], a)
            maes.append(np.mean(np.abs(y[m] - predict(model, X[m]))))
        if np.mean(maes) < best_s:
            best_a, best_s = a, np.mean(maes)
    return best_a


def pooled(y, p):
    r = y - p
    rmse = np.sqrt(np.mean(r ** 2))
    mae = np.mean(np.abs(r))
    r2 = 1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)
    return rmse, mae, r2

def lobo(df, cols, method="ols"):
    X = df[cols].values.astype(float)
    y = df["Health"].values.astype(float)
    g = df["Battery_ID"].values
    maes_m, maes_b, y_all, p_all, b_all, alphas = [], [], [], [], [], []

    for b in sorted(np.unique(g)):
        te = g == b
        tr = ~te
        if method == "ridge":
            a = choose_alpha(X[tr], y[tr], g[tr])
            alphas.append(a)
            model = fit_ridge(X[tr], y[tr], a)
        else:
            model = fit(X[tr], y[tr])
        pred = predict(model, X[te])
        base = y[tr].mean()
        maes_m.append(np.mean(np.abs(y[te] - pred)))
        maes_b.append(np.mean(np.abs(y[te] - base)))
        y_all.append(y[te])
        p_all.append(pred)
        b_all.append(np.full(te.sum(), base))

    y_all, p_all, b_all = map(np.concatenate, (y_all, p_all, b_all))
    maes_m, maes_b = np.array(maes_m), np.array(maes_b)
    rmse, mae, r2 = pooled(y_all, p_all)
    b_rmse, b_mae, b_r2 = pooled(y_all, b_all)
    return {
        "MAE_b": maes_m.mean(), "Base_b": maes_b.mean(),
        "Wins": int((maes_m < maes_b).sum()), "N": len(maes_m),
        "RMSE": rmse, "MAE": mae, "R2": r2,
        "Base_RMSE": b_rmse, "Base_MAE": b_mae, "Base_R2": b_r2,
        "Alpha": np.median(alphas) if alphas else np.nan,
    }


def lobo_predictions(df):
    parts = []
    for b in sorted(df["Battery_ID"].unique()):
        te = df[df["Battery_ID"] == b].sort_values("Cycle_Index")
        tr = df[df["Battery_ID"] != b]
        out = pd.DataFrame({
            "Battery_ID": b,
            "Cycle_Index": te["Cycle_Index"].values,
            "Health_actual": te["Health"].values,
            "Pred_baseline": tr["Health"].mean(),
        })
        for name, cols in SETS.items():
            model = fit(tr[cols].values.astype(float), tr["Health"].values.astype(float))
            out["Pred_" + name.split(":")[0]] = predict(model, te[cols].values.astype(float))

        # Ridge για το σύνολο P του καθηγητή (alpha από εσωτερικό LOBO)
        cols = SETS["P: professor (7 vars)"]
        Xtr = tr[cols].values.astype(float)
        ytr = tr["Health"].values.astype(float)
        a = choose_alpha(Xtr, ytr, tr["Battery_ID"].values)
        model = fit_ridge(Xtr, ytr, a)
        out["Pred_P_ridge"] = predict(model, te[cols].values.astype(float))
        out["Alpha_P"] = a
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def group_of(i):
    if i < -3.5:
        return "A(-4A)"
    if i > -1.2:
        return "C(-1A)"
    return "B(-2A)"


def logo(df, cols):
    d = df.copy()
    avg_i = d.groupby("Battery_ID")["Current_Mean"].transform("mean")
    span = d.groupby("Battery_ID")["Current_Mean"].transform(lambda s: s.max() - s.min())
    d["Group"] = [group_of(a) if s < 1.0 else "D(mixed)" for a, s in zip(avg_i, span)]

    out = {}
    for g in sorted(d["Group"].unique()):
        te, tr = d[d["Group"] == g], d[d["Group"] != g]
        model = fit(tr[cols].values.astype(float), tr["Health"].values.astype(float))
        maes, bases = [], []
        for b in te["Battery_ID"].unique():
            t = te[te["Battery_ID"] == b]
            pred = predict(model, t[cols].values.astype(float))
            y = t["Health"].values.astype(float)
            maes.append(np.mean(np.abs(y - pred)))
            bases.append(np.mean(np.abs(y - tr["Health"].mean())))
        out[g] = (len(maes), np.mean(maes), np.mean(bases))
    return out


def summary_all(df, cols):
    rows = []
    for b in sorted(df["Battery_ID"].unique()):
        te = df[df["Battery_ID"] == b].sort_values("Cycle_Index")
        tr = df[df["Battery_ID"] != b]
        model = fit(tr[cols].values.astype(float), tr["Health"].values.astype(float))
        pred = predict(model, te[cols].values.astype(float))
        y = te["Health"].values.astype(float)
        err = pred - y
        k = min(5, len(te))
        rows.append({
            "Battery": b,
            "Cycles": len(te),
            "MAE": np.mean(np.abs(err)),
            "Bias": np.mean(err),
            "Actual_end": np.median(y[-k:]),
            "Pred_end": np.median(pred[-k:]),
            "Err_end": np.median(err[-k:]),
            "Baseline": np.mean(np.abs(y - tr["Health"].mean())),
        })
    res = pd.DataFrame(rows)
    res["Win"] = np.where(res["MAE"] < res["Baseline"], "yes", "no")
    print(f"\n── PER-BATTERY SUMMARY, {CHECK_MODEL}, leave-one-battery-out ──")
    print("Bias > 0: overestimates health | Err_end: error over last 5 cycles")
    print(res.round(3).to_string(index=False))


def main():
    engine = get_engine()
    if engine is None:
        return
    df = pd.read_sql("SELECT * FROM CYCLE_HEALTH", engine)
    df = df.dropna()

    print("── CORRELATION WITH HEALTH ──")
    for c in ["Cycle_Index", "Capacity_Ah", "Discharge_Time", "Temp_Mean",
              "Voltage_Min", "Voltage_Mean", "Current_Mean"]:
        print(f"{c:>15}: {df[c].corr(df['Health']):+.3f}")

    print("\n── LEAVE-ONE-BATTERY-OUT ──")
    print("MAE_b: mean MAE per battery | RMSE, MAE, R2: all cycles pooled")
    print(f"{'Model':>32} {'Method':>6} {'MAE_b':>7} {'Wins':>6} "
          f"{'RMSE':>7} {'MAE':>7} {'R2':>7} {'alpha':>8}")
    first = True
    for name, cols in SETS.items():
        for method in ["ols", "ridge"]:
            r = lobo(df, cols, method)
            if first:
                print(f"{'Baseline':>32} {'mean':>6} {r['Base_b']:>7.4f} {'':>6} "
                      f"{r['Base_RMSE']:>7.4f} {r['Base_MAE']:>7.4f} {r['Base_R2']:>7.3f}")
                first = False
            alpha = "" if np.isnan(r["Alpha"]) else f"{r['Alpha']:.3g}"
            print(f"{name:>32} {method:>6} {r['MAE_b']:>7.4f} "
                  f"{r['Wins']:>3}/{r['N']:<2} {r['RMSE']:>7.4f} {r['MAE']:>7.4f} "
                  f"{r['R2']:>7.3f} {alpha:>8}")

    preds = lobo_predictions(df)
    preds.to_sql("HEALTH_PRED_MULTI", con=engine, if_exists="replace",
                 index=False, chunksize=200)
    print("\nSaved predictions to HEALTH_PRED_MULTI.")

    cols = SETS["M2c: all (no capacity)"]
    model = fit(df[cols].values.astype(float), df["Health"].values.astype(float))
    print("\n── STANDARDIZED WEIGHTS, M2c, all batteries ──")
    print(f"{'intercept':>15}: {model[0][0]:+.4f}")
    for c, w in zip(cols, model[0][1:]):
        print(f"{c:>15}: {w:+.4f}")

    cols = SETS["P: professor (7 vars)"]
    Xp = df[cols].values.astype(float)
    yp = df["Health"].values.astype(float)
    a = choose_alpha(Xp, yp, df["Battery_ID"].values)
    coef, _, _ = fit_ridge(Xp, yp, a)
    print(f"\n── STANDARDIZED WEIGHTS, RIDGE, P (7 vars), alpha {a:.3g} ──")
    print(f"{'intercept':>15}: {coef[0]:+.4f}")
    for c, w in zip(cols, coef[1:]):
        print(f"{c:>15}: {w:+.4f}")

    print("\n── LEAVE-ONE-GROUP-OUT (by discharge current) ──")
    for name in ["M1: Cycles", "M2c: all (no capacity)", "M2e: no Current_Mean"]:
        print(f"\n{name}")
        for g, (n, mae, base) in logo(df, SETS[name]).items():
            print(f"  {g:>9} ({n:>2} batt)  MAE {mae:.4f}  baseline {base:.4f}")

    summary_all(df, SETS[CHECK_MODEL])


if __name__ == "__main__":
    main()