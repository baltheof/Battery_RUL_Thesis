import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from db_connection import get_engine

FEATURES = ["Cycle_Index"]


def fit_predict(X_train, y_train, X_test):
    Xtr = np.column_stack([np.ones(len(X_train)), X_train])
    coeffs = np.linalg.lstsq(Xtr, y_train, rcond=None)[0]
    Xte = np.column_stack([np.ones(len(X_test)), X_test])
    return Xte @ coeffs, coeffs


def run(df, label):
    batteries = sorted(df["Battery_ID"].unique())
    rows = []
    coefs = []
    preds = []

    for b in batteries:
        test = df[df["Battery_ID"] == b]
        train = df[df["Battery_ID"] != b]

        X_tr = train[FEATURES].values.astype(float)
        y_tr = train["Health"].values.astype(float)
        X_te = test[FEATURES].values.astype(float)
        y_te = test["Health"].values.astype(float)

        pred_lin, c = fit_predict(X_tr, y_tr, X_te)
        pred_mean = np.full(len(y_te), y_tr.mean())

        rows.append({
            "Battery": b,
            "Rows": len(test),
            "MAE_linear": np.mean(np.abs(y_te - pred_lin)),
            "MAE_mean": np.mean(np.abs(y_te - pred_mean)),
        })
        coefs.append(c)
        preds.append(pd.DataFrame({
            "Battery_ID": b,
            "Cycle_Index": test["Cycle_Index"].values,
            "Health_actual": y_te,
            "Pred_linear": pred_lin,
            "Pred_baseline": pred_mean,
        }))

    res = pd.DataFrame(rows)
    coefs = np.array(coefs)

    print(f"\n── {label} ──")
    print(f"Batteries: {len(batteries)} | Rows: {len(df)}")
    print(f"Mean MAE (per battery)  linear: {res['MAE_linear'].mean():.4f} | "
          f"mean-baseline: {res['MAE_mean'].mean():.4f}")
    print(f"Linear better than baseline in "
          f"{(res['MAE_linear'] < res['MAE_mean']).sum()} / {len(res)} batteries")
    print(f"a0 across folds: mean {coefs[:,0].mean():.4f}, std {coefs[:,0].std():.5f}")
    print(f"a1 across folds: mean {coefs[:,1].mean():.6f}, std {coefs[:,1].std():.7f}")
    return res, pd.concat(preds, ignore_index=True)


def main():
    engine = get_engine()
    if engine is None:
        return

    df = pd.read_sql("SELECT * FROM CYCLE_HEALTH", engine)

    res_all, preds_all = run(df, "ALL batteries (B0038 as is)")

    mask_b38 = (df["Battery_ID"] == "B0038") & (df["Cycle_Index"] <= 12)
    res_no38, _ = run(df[~mask_b38], "B0038 cycles 2-12 removed")

    preds_all.to_sql("HEALTH_PRED_LINEAR", con=engine,
                     if_exists="replace", index=False, chunksize=200)
    print("\nSaved predictions to HEALTH_PRED_LINEAR.")

    print("\n── MAE PER BATTERY (all batteries version) ──")
    print(res_all.round(4).to_string(index=False))


if __name__ == "__main__":
    main()