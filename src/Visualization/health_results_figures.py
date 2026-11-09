import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
CURVE_BATTERIES = ["B0005", "B0033"]          # ίδιες με τις σημειώσεις του καθηγητή
CLASS_NAMES = ["Degraded", "Moderate", "Healthy"]   # 0, 1, 2


def fig_scatter(df, path):
    # Μέρος Α: πρόβλεψη Ridge έναντι πραγματικής υγείας (όλοι οι κύκλοι, LOBO)
    y, p = df["Health_actual"], df["Pred_P_ridge"]
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, p, s=6, alpha=0.4)
    lim = [min(y.min(), p.min()) - 0.02, max(y.max(), p.max()) + 0.02]
    ax.plot(lim, lim, "k--", label="Ιδανική πρόβλεψη")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Πραγματική υγεία")
    ax.set_ylabel("Πρόβλεψη Ridge")
    ax.set_title(f"Ridge, 7 μεταβλητές, LOBO (R² = {r2:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_curves(df, batteries, path):
    # Μέρος Α: πραγματική υγεία, Ridge και baseline ανά κύκλο
    fig, axes = plt.subplots(1, len(batteries), figsize=(6 * len(batteries), 4.5),
                             sharey=True, squeeze=False)
    for ax, b in zip(axes[0], batteries):
        s = df[df["Battery_ID"] == b].sort_values("Cycle_Index")
        mae = np.mean(np.abs(s["Health_actual"] - s["Pred_P_ridge"]))
        ax.plot(s["Cycle_Index"], s["Health_actual"], "k", label="Πραγματική")
        ax.plot(s["Cycle_Index"], s["Pred_P_ridge"], label="Ridge")
        ax.plot(s["Cycle_Index"], s["Pred_baseline"], "--", color="gray", label="Baseline")
        ax.axhline(0.7, color="r", ls=":", lw=1)
        ax.set_title(f"{b} (MAE {mae:.3f})")
        ax.set_xlabel("Κύκλος")
    axes[0][0].set_ylabel("Υγεία")
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_confusion(df, path):
    # Μέρος Β: confusion matrix του MLP (όλοι οι κύκλοι, LOBO)
    yt, yp = df["Class_true"].astype(int), df["Class_MLP"].astype(int)
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(yt, yp):
        cm[t, p] += 1
    acc = np.trace(cm) / cm.sum()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Πρόβλεψη MLP")
    ax.set_ylabel("Πραγματική κλάση")
    ax.set_title(f"MLP 32-16, LOBO (Accuracy = {acc:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    from db_connection import get_engine
    engine = get_engine()
    if engine is None:
        return
    os.makedirs(OUT_DIR, exist_ok=True)

    multi = pd.read_sql("SELECT * FROM HEALTH_PRED_MULTI", engine)
    mlp = pd.read_sql("SELECT * FROM HEALTH_PRED_MLP", engine)

    fig_scatter(multi, os.path.join(OUT_DIR, "A_ridge_scatter.png"))
    fig_curves(multi, CURVE_BATTERIES, os.path.join(OUT_DIR, "A_ridge_curves.png"))
    fig_confusion(mlp, os.path.join(OUT_DIR, "B_mlp_confusion.png"))
    print(f"Saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()