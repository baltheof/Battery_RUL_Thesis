import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from db_connection import get_engine

def train_test_split_manual(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_count = int(len(X) * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

def linear_regression_fit(X_train, y_train):
    # Προσθέτουμε στήλη 1 για το intercept
    X = np.column_stack([np.ones(len(X_train)), X_train])
    y = y_train.values
    # Κανονικές εξισώσεις: β = (XᵀX)⁻¹ Xᵀy
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return coeffs

def linear_regression_predict(X_test, coeffs):
    X = np.column_stack([np.ones(len(X_test)), X_test])
    return X @ coeffs

def train_baseline_model():
    engine = get_engine()
    if engine is None:
        return

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("Step 1/4: Loading CYCLE_FEATURES from database...")
    df = pd.read_sql("SELECT * FROM CYCLE_FEATURES", engine)
    print(f"   Loaded {len(df)} rows.")

    # ── Step 2: Define features (X) and target (y) ───────────────────────────
    print("Step 2/4: Preparing features and target...")
    feature_cols = [
        'Capacity_Ah', 'Discharge_Time', 'Temp_Mean', 'Temp_Max',
        'Voltage_Min', 'Voltage_Mean', 'Current_Mean'
    ]
    X = df[feature_cols]
    y = df['RUL']

    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_state=42)
    print(f"   Training set: {len(X_train)} rows")
    print(f"   Test set:     {len(X_test)} rows")

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    print("Step 3/4: Training Linear Regression baseline model...")
    coeffs = linear_regression_fit(X_train, y_train)
    print("   Training complete!")

    # ── Step 4: Evaluate ──────────────────────────────────────────────────────
    print("Step 4/4: Evaluating baseline model...")
    y_pred = linear_regression_predict(X_test, coeffs)

    mae  = np.mean(np.abs(y_test.values - y_pred))
    rmse = np.sqrt(np.mean((y_test.values - y_pred) ** 2))
    ss_res = np.sum((y_test.values - y_pred) ** 2)
    ss_tot = np.sum((y_test.values - np.mean(y_test.values)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print(f"\n   ── BASELINE MODEL RESULTS ──")
    print(f"   MAE  : {mae:.2f} cycles")
    print(f"   RMSE : {rmse:.2f} cycles")
    print(f"   R²   : {r2:.4f}")

    print(f"\n   ── FEATURE COEFFICIENTS ──")
    print(f"   {'Intercept':20s}: {coeffs[0]:.4f}")
    for feature, coef in zip(feature_cols, coeffs[1:]):
        print(f"   {feature:20s}: {coef:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.5, color='steelblue',
                    edgecolors='k', linewidths=0.3)
    axes[0].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 'r--', linewidth=1.5, label='Perfect prediction')
    axes[0].set_xlabel('Actual RUL (cycles)')
    axes[0].set_ylabel('Predicted RUL (cycles)')
    axes[0].set_title('BASELINE — ACTUAL vs PREDICTED RUL', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.4)

    residuals = y_test.values - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, color='coral',
                    edgecolors='k', linewidths=0.3)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('Predicted RUL (cycles)')
    axes[1].set_ylabel('Residuals (Actual - Predicted)')
    axes[1].set_title('BASELINE — RESIDUALS PLOT', fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig('baseline_results.png', dpi=150, bbox_inches='tight')
    print("\n   Saved: baseline_results.png")
    plt.show()

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

if __name__ == "__main__":
    train_baseline_model()