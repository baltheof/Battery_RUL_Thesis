import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
CURVE_BATTERIES = ["B0005", "B0033"]                 # ίδιες με τις σημειώσεις του καθηγητή
CLASS_NAMES = ["Degraded", "Moderate", "Healthy"]    # 0, 1, 2
BLUE, GRAY = "#1f4e79", "#8c8c8c"


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_scatter(ax, df):
    y, p = df["Health_actual"], df["Pred_P_ridge"]
    lim = [min(y.min(), p.min()) - 0.02, max(y.max(), p.max()) + 0.02]
    ax.scatter(y, p, s=6, alpha=0.35, color=BLUE)
    ax.plot(lim, lim, color="black", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual health")
    ax.set_ylabel("Predicted health")
    ax.set_title("A. Ridge: predicted vs actual health")
    clean(ax)


def draw_curve(ax, df, battery):
    s = df[df["Battery_ID"] == battery].sort_values("Cycle_Index")
    ax.plot(s["Cycle_Index"], s["Health_actual"], color="black", lw=1.5, label="Actual")
    ax.plot(s["Cycle_Index"], s["Pred_P_ridge"], color=BLUE, lw=1.5, label="Ridge")
    ax.plot(s["Cycle_Index"], s["Pred_baseline"], color=GRAY, lw=1.2, ls="--", label="Baseline")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Health")
    ax.set_title(f"A. Health per cycle: {battery}")
    ax.legend(frameon=False)
    clean(ax)


def draw_confusion(ax, df):
    yt, yp = df["Class_true"].astype(int), df["Class_MLP"].astype(int)
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(yt, yp):
        cm[t, p] += 1
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(3))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("B. MLP: confusion matrix")


def show_overview(multi, mlp, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    draw_scatter(axes[0][0], multi)
    draw_curve(axes[0][1], multi, CURVE_BATTERIES[0])
    draw_curve(axes[1][0], multi, CURVE_BATTERIES[1])
    draw_confusion(axes[1][1], mlp)
    fig.tight_layout(h_pad=3, w_pad=3)
    fig.savefig(path, dpi=150)


def main():
    from db_connection import get_engine
    engine = get_engine()
    if engine is None:
        return
    os.makedirs(OUT_DIR, exist_ok=True)

    multi = pd.read_sql("SELECT * FROM HEALTH_PRED_MULTI", engine)
    mlp = pd.read_sql("SELECT * FROM HEALTH_PRED_MLP", engine)
    show_overview(multi, mlp, os.path.join(OUT_DIR, "results_overview.png"))
    print(f"Saved figure to {OUT_DIR}")
    plt.show()


if __name__ == "__main__":
    main()