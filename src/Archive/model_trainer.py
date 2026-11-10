import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from db_connection import get_engine

# ── Manual Train/Test Split ───────────────────────────────────────────────────
def train_test_split_manual(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_count = int(len(X) * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

# ── Decision Tree (single tree) ───────────────────────────────────────────────
class DecisionTreeRegressor:
    def __init__(self, max_depth=10, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build(np.array(X), np.array(y), depth=0)

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return np.mean(y)
        
        best_feat, best_thresh, best_mse = None, None, float('inf')
        
        for feat in range(X.shape[1]):
            thresholds = np.percentile(X[:, feat], [25, 50, 75])
            for thresh in thresholds:
                left = y[X[:, feat] <= thresh]
                right = y[X[:, feat] > thresh]
                if len(left) == 0 or len(right) == 0:
                    continue
                mse = (len(left) * np.var(left) + len(right) * np.var(right)) / len(y)
                if mse < best_mse:
                    best_mse = mse
                    best_feat = feat
                    best_thresh = thresh

        if best_feat is None:
            return np.mean(y)

        left_mask = X[:, best_feat] <= best_thresh
        return {
            'feat': best_feat,
            'thresh': best_thresh,
            'left': self._build(X[left_mask], y[left_mask], depth + 1),
            'right': self._build(X[~left_mask], y[~left_mask], depth + 1)
        }

    def _predict_one(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node['feat']] <= node['thresh']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_one(row, self.tree) for row in np.array(X)])

# ── Random Forest ─────────────────────────────────────────────────────────────
class RandomForestRegressor:
    def __init__(self, n_estimators=50, max_depth=10, min_samples_split=5,
                 max_features=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X_arr, y_arr = np.array(X), np.array(y)
        n_samples, n_features = X_arr.shape
        max_feat = self.max_features or max(1, n_features // 3)

        self.trees = []
        self.feature_indices = []

        for i in range(self.n_estimators):
            # Bootstrap sample
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X_arr[idx], y_arr[idx]

            # Random feature subset
            feat_idx = np.random.choice(n_features, max_feat, replace=False)
            self.feature_indices.append(feat_idx)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_boot[:, feat_idx], y_boot)
            self.trees.append(tree)

            if (i + 1) % 10 == 0:
                print(f"   Trees trained: {i + 1}/{self.n_estimators}")

    def predict(self, X):
        X_arr = np.array(X)
        preds = np.array([
            tree.predict(X_arr[:, feat_idx])
            for tree, feat_idx in zip(self.trees, self.feature_indices)
        ])
        return np.mean(preds, axis=0)

    def feature_importances(self, X, y, feature_cols):
        baseline_mse = np.mean((self.predict(X) - np.array(y)) ** 2)
        importances = []
        X_arr = np.array(X)
        for i in range(X_arr.shape[1]):
            X_perm = X_arr.copy()
            np.random.shuffle(X_perm[:, i])
            perm_mse = np.mean((self.predict(X_perm) - np.array(y)) ** 2)
            importances.append(perm_mse - baseline_mse)
        total = sum(importances)
        return [imp / total for imp in importances]

def train_random_forest():
    engine = get_engine()
    if engine is None:
        return

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("Step 1/4: Loading CYCLE_FEATURES from database...")
    df = pd.read_sql("SELECT * FROM CYCLE_FEATURES", engine)
    print(f"   Loaded {len(df)} rows.")

    # ── Step 2: Prepare features and target ───────────────────────────────────
    print("Step 2/4: Preparing features and target...")
    feature_cols = [
        'Capacity_Ah', 'Discharge_Time', 'Temp_Mean', 'Temp_Max',
        'Voltage_Min', 'Voltage_Mean', 'Current_Mean'
    ]
    X = df[feature_cols]
    y = df['RUL']

    X_train, X_test, y_train, y_test = train_test_split_manual(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training set: {len(X_train)} rows")
    print(f"   Test set:     {len(X_test)} rows")

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    print("Step 3/4: Training Random Forest (50 trees)...")
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("   Training complete!")

    # ── Step 4: Evaluate ──────────────────────────────────────────────────────
    print("Step 4/4: Evaluating model...")
    y_pred = model.predict(X_test)

    mae  = np.mean(np.abs(y_test.values - y_pred))
    rmse = np.sqrt(np.mean((y_test.values - y_pred) ** 2))
    ss_res = np.sum((y_test.values - y_pred) ** 2)
    ss_tot = np.sum((y_test.values - np.mean(y_test.values)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print(f"\n   ── RANDOM FOREST RESULTS ──")
    print(f"   MAE  : {mae:.2f} cycles")
    print(f"   RMSE : {rmse:.2f} cycles")
    print(f"   R²   : {r2:.4f}")

    print(f"\n   ── ΣΥΓΚΡΙΣΗ ΜΕ BASELINE ──")
    print(f"   {'Μετρική':10s}  {'Baseline':>10s}  {'Random Forest':>15s}")
    print(f"   {'MAE':10s}  {'19.72':>10s}  {mae:>15.2f}")
    print(f"   {'RMSE':10s}  {'26.14':>10s}  {rmse:>15.2f}")
    print(f"   {'R²':10s}  {'0.5025':>10s}  {r2:>15.4f}")

    # ── Plot 1: Actual vs Predicted ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.5, color='steelblue',
                    edgecolors='k', linewidths=0.3)
    axes[0].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 'r--', linewidth=1.5, label='Perfect prediction')
    axes[0].set_xlabel('Actual RUL (cycles)')
    axes[0].set_ylabel('Predicted RUL (cycles)')
    axes[0].set_title('RANDOM FOREST — ACTUAL vs PREDICTED RUL', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.4)

    # ── Plot 2: Feature Importance ────────────────────────────────────────────
    print("\n   Calculating feature importances...")
    importances = model.feature_importances(X_test, y_test, feature_cols)
    imp_series = pd.Series(importances, index=feature_cols).sort_values()
    imp_series.plot(kind='barh', ax=axes[1], color='steelblue',
                    edgecolor='k', linewidth=0.5)
    axes[1].set_title('FEATURE IMPORTANCE', fontweight='bold')
    axes[1].set_xlabel('Importance Score')
    axes[1].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig('random_forest_results.png', dpi=150, bbox_inches='tight')
    print("   Saved: random_forest_results.png")
    plt.show()

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def plot_model_comparison():
    models = ['Baseline\n(Linear Regression)', 'Random Forest']
    mae_scores  = [19.72, 7.28]
    rmse_scores = [26.14, 10.19]
    r2_scores   = [0.5025, 0.9244]

    # Χρήση constrained_layout=True για τέλεια στοίχιση του κεντρικού τίτλου
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    colors = ['#d9534f', '#5b9bd5']

    # MAE
    axes[0].bar(models, mae_scores, color=colors, edgecolor='k', linewidth=1)
    axes[0].set_title('MAE (cycles)\nΧαμηλότερο = Καλύτερο', fontweight='bold', fontsize=13)
    axes[0].set_ylabel('Κύκλοι', fontsize=12)
    # Δίνουμε 15% παραπάνω χώρο στην κορυφή για να μην κόβονται τα νούμερα
    axes[0].set_ylim(0, max(mae_scores) * 1.15) 
    for i, v in enumerate(mae_scores):
        axes[0].text(i, v + 0.4, str(v), ha='center', fontweight='bold', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.5, axis='y')

    # RMSE
    axes[1].bar(models, rmse_scores, color=colors, edgecolor='k', linewidth=1)
    axes[1].set_title('RMSE (cycles)\nΧαμηλότερο = Καλύτερο', fontweight='bold', fontsize=13)
    axes[1].set_ylabel('Κύκλοι', fontsize=12)
    axes[1].set_ylim(0, max(rmse_scores) * 1.15)
    for i, v in enumerate(rmse_scores):
        axes[1].text(i, v + 0.5, str(v), ha='center', fontweight='bold', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.5, axis='y')

    # R²
    axes[2].bar(models, r2_scores, color=colors, edgecolor='k', linewidth=1)
    axes[2].set_title('R²\nΥψηλότερο = Καλύτερο', fontweight='bold', fontsize=13)
    axes[2].set_ylabel('R² Score', fontsize=12)
    axes[2].set_ylim(0, 1.15)
    for i, v in enumerate(r2_scores):
        axes[2].text(i, v + 0.02, str(v), ha='center', fontweight='bold', fontsize=12)
    axes[2].grid(True, linestyle='--', alpha=0.5, axis='y')

    fig.suptitle('ΣΥΓΚΡΙΣΗ ΜΟΝΤΕΛΩΝ — BASELINE vs RANDOM FOREST',
                 fontsize=16, fontweight='bold')
    
    # Υψηλή ανάλυση για το Word
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: model_comparison.png (High Resolution)")
    plt.show()

if __name__ == "__main__":
    train_random_forest()
    plot_model_comparison()