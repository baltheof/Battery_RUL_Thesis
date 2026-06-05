import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors  

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_connection import get_engine

def plot_distributions():
    engine = get_engine()
    if engine is None:
        return

    print("Loading CYCLE_FEATURES from database...")
    df = pd.read_sql("SELECT * FROM CYCLE_FEATURES", engine)
    print(f"   Loaded {len(df)} rows.")

    feature_cols = [
        'RUL', 'Capacity_Ah', 'Discharge_Time',
        'Temp_Mean', 'Temp_Max',
        'Voltage_Min', 'Voltage_Mean', 'Current_Mean'
    ]

    # PLOT 1: STATIC HISTOGRAMS
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='#4C72B0', edgecolor='white', bins=25, alpha=0.7)
        axes[i].set_title(f'Distribution of {col}', fontweight='bold', fontsize=12)
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].set_ylabel('Count', fontsize=10)
        axes[i].grid(True, linestyle='--', alpha=0.5)

        mean_val = df[col].mean()
        axes[i].axvline(mean_val, color='#C44E52', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        axes[i].legend(fontsize=9, loc='upper right')

    fig.suptitle('DISTRIBUTION PLOTS — CYCLE FEATURES & RUL', fontsize=18, fontweight='bold')
    plt.savefig('distribution_plots.png', dpi=300, bbox_inches='tight')
    print("Saved static image: distribution_plots.png")
    
    plt.show()

  
    # PLOT 2: INTERACTIVE CAPACITY DEGRADATION
    fig2, ax2 = plt.subplots(figsize=(14, 7), constrained_layout=True)
    batteries = df['Battery_ID'].unique()
    colors = sns.color_palette("husl", len(batteries)) 
    
    lines = [] 

    for i, battery in enumerate(sorted(batteries)):
        battery_df = df[df['Battery_ID'] == battery].sort_values('Cycle_Index')
        line, = ax2.plot(
            battery_df['Cycle_Index'], battery_df['Capacity_Ah'],
            color=colors[i], linewidth=1.5, alpha=0.85, label=battery
        )
        lines.append(line)

    nominal = df['Capacity_Ah'].max()
    threshold = nominal * 0.70
    ax2.axhline(threshold, color='#C44E52', linestyle='--', linewidth=2, 
                label=f'Failure threshold (70% = {threshold:.3f} Ah)')

    ax2.set_title('CAPACITY DEGRADATION PER BATTERY (Click Legend to Hide/Show)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Cycle Index', fontsize=12)
    ax2.set_ylabel('Capacity (Ah)', fontsize=12)
    
    leg = ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, ncol=2)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # --- CLICK TO HIDE/SHOW ---
    map_legend_to_ax = {}
    for legline, legtext, origline in zip(leg.get_lines(), leg.get_texts(), lines):
        legline.set_picker(True)
        legline.set_pickradius(5)
        legtext.set_picker(True)
        map_legend_to_ax[legline] = origline
        map_legend_to_ax[legtext] = origline

    def on_pick(event):
        leg_artist = event.artist
        origline = map_legend_to_ax.get(leg_artist)
        if origline is None: return
        
        visible = not origline.get_visible()
        origline.set_visible(visible)
        leg_artist.set_alpha(1.0 if visible else 0.2)
        fig2.canvas.draw()

    fig2.canvas.mpl_connect('pick_event', on_pick)

    #  HOVER DYNAMIC COLOR 
    cursor = mplcursors.cursor(lines, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        battery_name = sel.artist.get_label()
        cycle = int(sel.target[0])
        cap = sel.target[1]
        line_color = sel.artist.get_color() # Εξάγουμε το χρώμα της συγκεκριμένης γραμμής
        
        sel.annotation.set_text(f"Battery: {battery_name}\nCycle: {cycle}\nCap: {cap:.3f} Ah")
        sel.annotation.get_bbox_patch().set_facecolor(line_color) # Αλλάζουμε το φόντο του κουτιού
        sel.annotation.get_bbox_patch().set_alpha(0.85) # Το κάνουμε ελαφρώς διαφανές
        sel.annotation.set_color("white") # Κάνουμε τα γράμματα λευκά για αντίθεση

    plt.savefig('capacity_degradation.png', dpi=300, bbox_inches='tight')
    print("Saved static image: capacity_degradation.png")
    
    plt.show()

if __name__ == "__main__":
    plot_distributions()