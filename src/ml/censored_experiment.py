import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from db_connection import get_engine
from model_trainer import RandomForestRegressor

FEATURES = [
    'Capacity_Ah', 'Discharge_Time', 'Temp_Mean', 'Temp_Max',
    'Voltage_Min', 'Voltage_Mean', 'Current_Mean'
]
K = 4


def metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return mae, rmse, 1 - ss_res / ss_tot


def run_experiment():
    engine = get_engine()
    if engine is None:
        return

    df = pd.read_sql("SELECT * FROM CYCLE_FEATURES_CENSORED", engine)

    confirmed = sorted(df.loc[df["Is_Censored"] == 0, "Battery_ID"].unique())
    rng = np.random.default_rng(42)
    folds = np.array_split(rng.permutation(confirmed), K)

    results = {"A": [], "B": []}

    for i, test_batteries in enumerate(folds):
        in_test = df["Battery_ID"].isin(test_batteries)

        test_df = df[in_test & (df["Is_Censored"] == 0)]
        train_A = df[~in_test & (df["Is_Censored"] == 0)]
        train_B = df[~in_test]

        print(f"\nFold {i+1}: test = {list(test_batteries)}")
        print(f"   Test rows: {len(test_df)} | "
              f"Train A: {len(train_A)} | Train B: {len(train_B)}")

        y_test = test_df["RUL"].values.astype(float)

        for name, train_df in (("A", train_A), ("B", train_B)):
            model = RandomForestRegressor(
                n_estimators=50, max_depth=10,
                min_samples_split=5, random_state=42
            )
            model.fit(train_df[FEATURES], train_df["RUL"])
            y_pred = model.predict(test_df[FEATURES])
            results[name].append(metrics(y_test, y_pred))

    print("\n── RESULTS (mean over folds) ──")
    print(f"{'Model':>22} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    labels = {"A": "A: confirmed only", "B": "B: confirmed+censored"}
    for name in ("A", "B"):
        arr = np.array(results[name])
        print(f"{labels[name]:>22} {arr[:,0].mean():>8.2f} "
              f"{arr[:,1].mean():>8.2f} {arr[:,2].mean():>8.4f}")

    print("\n── MAE PER FOLD ──")
    for i in range(K):
        print(f"Fold {i+1}:  A = {results['A'][i][0]:.2f} | "
              f"B = {results['B'][i][0]:.2f}")


if __name__ == "__main__":
    run_experiment()