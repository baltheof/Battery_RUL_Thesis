import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from db_connection import get_engine

# ── Επαναχρησιμοποίηση κλάσεων από model_trainer.py ──────────────────────────
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
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X_arr[idx], y_arr[idx]
            feat_idx = np.random.choice(n_features, max_feat, replace=False)
            self.feature_indices.append(feat_idx)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_boot[:, feat_idx], y_boot)
            self.trees.append(tree)

    def predict(self, X):
        X_arr = np.array(X)
        preds = np.array([
            tree.predict(X_arr[:, feat_idx])
            for tree, feat_idx in zip(self.trees, self.feature_indices)
        ])
        return np.mean(preds, axis=0)


# ── K-Fold Cross Validation ───────────────────────────────────────────────────
def kfold_cross_validation():
    engine = get_engine()
    if engine is None:
        return

    print("Loading CYCLE_FEATURES from database...")
    df = pd.read_sql("SELECT * FROM CYCLE_FEATURES", engine)

    feature_cols = [
        'Capacity_Ah', 'Discharge_Time', 'Temp_Mean', 'Temp_Max',
        'Voltage_Min', 'Voltage_Mean', 'Current_Mean'
    ]
    X = df[feature_cols].values
    y = df['RUL'].values

    K = 5
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    folds = np.array_split(indices, K)

    mae_scores  = []
    rmse_scores = []
    r2_scores   = []

    print(f"\nStarting {K}-Fold Cross Validation...")
    print(f"{'Fold':>6} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print("-" * 40)

    for fold_idx in range(K):
        # Test set: τρέχον fold
        test_idx  = folds[fold_idx]
        # Train set: όλα τα υπόλοιπα folds
        train_idx = np.concatenate([folds[i] for i in range(K) if i != fold_idx])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Εκπαίδευση
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        model.fit(
            pd.DataFrame(X_train, columns=feature_cols),
            pd.Series(y_train)
        )

        # Αξιολόγηση
        y_pred = model.predict(X_test)

        mae  = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        mae_scores.append(mae)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

        print(f"   {fold_idx+1:>3} {mae:>10.2f} {rmse:>10.2f} {r2:>10.4f}")

    print("-" * 40)
    print(f"{'Mean':>6} {np.mean(mae_scores):>10.2f} {np.mean(rmse_scores):>10.2f} {np.mean(r2_scores):>10.4f}")
    print(f"{'Std':>6} {np.std(mae_scores):>10.2f} {np.std(rmse_scores):>10.2f} {np.std(r2_scores):>10.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fold_labels = [f'Fold {i+1}' for i in range(K)]
    colors = ['#5b9bd5'] * K

    for ax, scores, title, ylabel in zip(
        axes,
        [mae_scores, rmse_scores, r2_scores],
        ['MAE ανά Fold', 'RMSE ανά Fold', 'R² ανά Fold'],
        ['Κύκλοι', 'Κύκλοι', 'R² Score']
    ):
        bars = ax.bar(fold_labels, scores, color=colors, edgecolor='k', linewidth=0.5)
        ax.axhline(np.mean(scores), color='red', linestyle='--',
                   linewidth=1.5, label=f'Mean: {np.mean(scores):.4f}')
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4, axis='y')
        for bar, val in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

    plt.suptitle('5-FOLD CROSS VALIDATION — RANDOM FOREST',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('kfold_results.png', dpi=150, bbox_inches='tight')
    print("\nSaved: kfold_results.png")
    plt.show()

if __name__ == "__main__":
    kfold_cross_validation()