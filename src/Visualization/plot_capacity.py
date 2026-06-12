import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_connection import get_engine

class CapacityPlotter:
    def __init__(self):
        self.engine = get_engine()
        
        print("Loading all battery data from database. Please wait...")
        query = """
            SELECT Battery_ID, Cycle_Index, Capacity_Ah 
            FROM TEST_CYCLES 
            WHERE Operation_Type = 'discharge'
            ORDER BY Cycle_Index
        """
        try:
            self.df = pd.read_sql(query, self.engine)
        except Exception as e:
            print(f"Database connection error: {e}")
            return

        if self.df.empty:
            print("No data found in the database. Exiting.")
            return

        self.batteries = sorted(self.df['Battery_ID'].unique())
        
        self.fig, self.ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
        self.fig.patch.set_facecolor('#ffffff') 
        
        self.lines = []
        self.active_line = None  
        
        colors = sns.color_palette("husl", len(self.batteries))
        
        # 1. Σχεδίαση γραμμών (Αρχικά χωρίς dots)
        for i, battery in enumerate(self.batteries):
            b_df = self.df[self.df['Battery_ID'] == battery]
            line, = self.ax.plot(
                b_df['Cycle_Index'], b_df['Capacity_Ah'], 
                marker='', linestyle='-', linewidth=1.5, alpha=0.85, 
                color=colors[i], label=battery
            )
            # Αρχικό μεσαίο hitbox για όλες τις γραμμές (10 pixels)
            line.set_pickradius(10)
            self.lines.append(line)

        # 2. Μορφοποίηση Γραφήματος
        self.ax.set_title('CAPACITY DEGRADATION CURVES (Click Legend to Focus)', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Cycle Index', fontsize=12)
        self.ax.set_ylabel('Capacity [Ah]', fontsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        
        # 3. Το Υπόμνημα
        self.leg = self.ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, ncol=2)
        
        # 4. Σύνδεση των κλικ
        self.map_legend_to_line = {}
        for legline, legtext, origline in zip(self.leg.get_lines(), self.leg.get_texts(), self.lines):
            legline.set_picker(True)
            legline.set_pickradius(5)
            legtext.set_picker(True)
            self.map_legend_to_line[legline] = origline
            self.map_legend_to_line[legtext] = origline

        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # 5. Δυναμικό Hover
        cursor = mplcursors.cursor(self.lines, hover=True)
        @cursor.connect("add")
        def on_hover(sel):
            if self.active_line is not None and sel.artist != self.active_line:
                sel.annotation.set_visible(False)
                return
                
            battery_name = sel.artist.get_label()
            x, y = sel.target
            line_color = sel.artist.get_color()
            
            sel.annotation.set_text(f"Battery: {battery_name}\nCycle: {int(x)}\nCap: {y:.4f} Ah")
            sel.annotation.get_bbox_patch().set_facecolor(line_color)
            sel.annotation.get_bbox_patch().set_alpha(0.85)
            sel.annotation.set_color("white")

    def on_pick(self, event):
        leg_artist = event.artist
        selected_line = self.map_legend_to_line.get(leg_artist)
        if selected_line is None: 
            return

        # --- Ο ΑΠΟΛΥΤΟΣ ΚΑΘΑΡΙΣΜΟΣ TΩΝ PINS ---
        if self.active_line != selected_line:
            while self.ax.texts:
                self.ax.texts[-1].remove()

        self.active_line = selected_line

        for line, leg_text in zip(self.lines, self.leg.get_texts()):
            if line == selected_line:
                # FOCUS
                line.set_alpha(1.0)
                line.set_linewidth(3.0) 
                line.set_marker('o')  
                line.set_markersize(4)
                line.set_zorder(10)
                # ---> ΤΕΡΑΣΤΙΟ HITBOX ΓΙΑ ΕΥΚΟΛΟ HOVER (15 pixels) <---
                line.set_pickradius(15) 
                leg_text.set_fontweight('bold')
                leg_text.set_alpha(1.0)
            else:
                # FADE
                line.set_alpha(0.1) 
                line.set_linewidth(1.0)
                line.set_marker('')   
                line.set_zorder(1)
                # ---> ΑΠΕΝΕΡΓΟΠΟΙΗΣΗ HITBOX ΣΤΙΣ ΑΧΝΕΣ ΓΡΑΜΜΕΣ (2 pixels) <---
                line.set_pickradius(2) 
                leg_text.set_fontweight('normal')
                leg_text.set_alpha(0.3)

        self.fig.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        # ΑΡΙΣΤΕΡΟ ΚΛΙΚ (1): Καρφίτσωμα ταμπελιού
        if event.button == 1:
            lines_to_check = [self.active_line] if self.active_line else self.lines
            
            for line in lines_to_check:
                cont, ind = line.contains(event)
                if cont:
                    idx = ind['ind'][0]
                    x = line.get_xdata()[idx]
                    y = line.get_ydata()[idx]
                    color = line.get_color()

                    self.ax.annotate(
                        f"Cycle: {int(x)}\nCap: {y:.4f} Ah",
                        xy=(x, y), xytext=(15, 15),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="white", alpha=0.9),
                        color="white", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="black")
                    )
                    self.fig.canvas.draw()
                    break

        # ΔΕΞΙ ΚΛΙΚ (3): Διαγραφή ταμπελιού με δεξί κλικ πάνω του
        elif event.button == 3:
            for ann in list(self.ax.texts):
                cont, _ = ann.contains(event)
                if cont:
                    ann.remove()
                    self.fig.canvas.draw()
                    break

if __name__ == "__main__":
    plotter = CapacityPlotter()
    print("ALL SYSTEMS GO: Plotter is ready with Auto-Clear Pins and Expanded Hitbox.")
    plt.show()