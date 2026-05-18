import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from db_connection import get_engine

def plot_correlation_heatmap():
    engine = get_engine()
    if engine is None:
        return

    print("Loading CYCLE_FEATURES from database...")
    df = pd.read_sql("SELECT * FROM CYCLE_FEATURES", engine)

    # Keep only numeric feature columns + RUL
    feature_cols = [
        'Capacity_Ah', 'Discharge_Time', 'Temp_Mean', 'Temp_Max',
        'Voltage_Min', 'Voltage_Mean', 'Current_Mean', 'RUL'
    ]
    df = df[feature_cols]

    # Compute correlation matrix
    corr_matrix = df.corr()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title('CORRELATION HEATMAP — CYCLE FEATURES vs RUL', 
                 fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
    print("Saved: correlation_heatmap.png")
    plt.show()

if __name__ == "__main__":
    plot_correlation_heatmap()