import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from db_connection import get_engine

# Υποσύνολα μεταβλητών (Temp_Max εκτός: συσχέτιση 0.99 με Temp_Mean)
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
    "B: all + Capacity (circular)":  ["Capacity_Ah", "Cycle_Index", "Discharge_Time",
                                      "Temp_Mean", "Voltage_Min", "Voltage_Mean",
                                      "Current_Mean"],
}

# Το μοντέλο που ελέγχεται μία-μία σε όλες τις μπαταρίες
CHECK_MODEL = "M2e: no Current_Mean"


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


def lobo(df, cols):
    maes_lin, maes_mean = [], []
    for b in sorted(df["Battery_ID"].unique()):
        te = df[df["Battery_ID"] == b]
        tr = df[df["Battery_ID"] != b]
        model = fit(tr[cols].values.astype(float), tr["Health"].values.astype(float))
        pred = predict(model, te[cols].values.astype(float))
        y = te["Health"].values.astype(float)
        maes_lin.append(np.mean(np.abs(y - pred)))
        maes_mean.append(np.mean(np.abs(y - tr["Health"].mean())))
    maes_lin = np.array(maes_lin)
    maes_mean = np.array(maes_mean)
    return maes_lin.mean(), maes_mean.mean(), int((maes_lin < maes_mean).sum()), len(maes_lin)


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

    print("\n── LEAVE-ONE-BATTERY-OUT (mean MAE per battery) ──")
    print(f"{'Model':>32} {'MAE':>8} {'Baseline':>9} {'Wins':>8}")
    for name, cols in SETS.items():
        mae, base, wins, n = lobo(df, cols)
        print(f"{name:>32} {mae:>8.4f} {base:>9.4f} {wins:>4}/{n}")

    cols = SETS["M2c: all (no capacity)"]
    model = fit(df[cols].values.astype(float), df["Health"].values.astype(float))
    print("\n── STANDARDIZED WEIGHTS, M2c, all batteries ──")
    print(f"{'intercept':>15}: {model[0][0]:+.4f}")
    for c, w in zip(cols, model[0][1:]):
        print(f"{c:>15}: {w:+.4f}")

    print("\n── LEAVE-ONE-GROUP-OUT (by discharge current) ──")
    for name in ["M1: Cycles", "M2c: all (no capacity)", "M2e: no Current_Mean"]:
        print(f"\n{name}")
        for g, (n, mae, base) in logo(df, SETS[name]).items():
            print(f"  {g:>9} ({n:>2} batt)  MAE {mae:.4f}  baseline {base:.4f}")

    summary_all(df, SETS[CHECK_MODEL])


if __name__ == "__main__":
    main()