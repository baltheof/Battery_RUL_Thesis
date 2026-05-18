import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
from matplotlib.widgets import TextBox
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_connection import get_engine

# Route configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from db_connection import get_engine

class CapacityPlotter:
    def __init__(self):
        self.engine = get_engine()
        self.fig, self.ax = plt.subplots(figsize=(11, 7))
        plt.subplots_adjust(bottom=0.2) 
        
        self.line = None
        self.df = None
        
        # CORRECT CONNECTION: Link 'button_press_event' to our 'on_click' function
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Initial Plot
        self.draw_plot('B0047')
        
        # Text Box setup
        ax_box = plt.axes([0.2, 0.05, 0.07, 0.04])
        self.text_box = TextBox(ax_box, 'Change Battery ID: ', initial='B0047')
        self.text_box.on_submit(self.submit)

    def draw_plot(self, battery_id):
        battery_id = battery_id.strip().upper()
        
        query = f"""
            SELECT Cycle_Index, Capacity_Ah 
            FROM TEST_CYCLES 
            WHERE Battery_ID = '{battery_id}' AND Operation_Type = 'discharge'
            ORDER BY Cycle_Index
        """
        self.df = pd.read_sql(query, self.engine)

        if self.df.empty:
            print(f"Warning: No data found for {battery_id}")
            return

        # Nuclear Clear: Wipe axis and all previous annotations/pins
        self.ax.cla()
        
        # Plot data
        self.line, = self.ax.plot(self.df['Cycle_Index'], self.df['Capacity_Ah'], 
                                 marker='o', linestyle='-', color="#39841e", 
                                 markersize=4, label='Capacity (Ah)')

        # HOVER ONLY: This handles temporary pop-ups
        cursor = mplcursors.cursor(self.line, hover=True)
        @cursor.connect("add")
        def on_hover(sel):
            x, y = sel.target
            sel.annotation.set_text(f"Cycle: {int(x)}\nCap: {y:.4f} Ah")
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.2, edgecolor="#39841e", boxstyle="round")

        # Formatting
        self.ax.set_title(f'CAPACITY DEGRADATION CURVE - BATTERY {battery_id}', fontsize=12, fontweight='bold')
        self.ax.set_xlabel('Cycle Index', fontsize=12)
        self.ax.set_ylabel('Capacity [Ah]', fontsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.legend()
        
        plt.draw()

    def on_click(self, event):
        # Ensure click is within the plot area
        if event.inaxes != self.ax:
            return

        # LEFT CLICK (1): PIN LABEL
        if event.button == 1:
            cont, ind = self.line.contains(event)
            if cont:
                idx = ind['ind'][0]
                x = self.df['Cycle_Index'].iloc[idx]
                y = self.df['Capacity_Ah'].iloc[idx]

                self.ax.annotate(
                    f"Cycle: {int(x)}\nCap: {y:.4f} Ah",
                    xy=(x, y), xytext=(15, 15),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#39841e", alpha=0.5),
                    arrowprops=dict(arrowstyle="->", color="black")
                )
                plt.draw()

        # RIGHT CLICK (3): DELETE LABEL
        elif event.button == 3:
            for ann in list(self.ax.texts):
                cont, _ = ann.contains(event)
                if cont:
                    ann.remove()
                    plt.draw()
                    break

    def submit(self, text):
        self.draw_plot(text)

if __name__ == "__main__":
    plotter = CapacityPlotter()
    print("ALL SYSTEMS GO: Plotter is ready.")
    plt.show()