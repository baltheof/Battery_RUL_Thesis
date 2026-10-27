import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_connection import get_engine


class HealthPlotter:
    VIEW_Y_MIN = 0.40
    VIEW_Y_MAX = 1.20
    SCROLL_STEP = 0.10

    def __init__(self):
        self.engine = get_engine()

        print("Loading CYCLE_HEALTH from database...")
        query = """
            SELECT Battery_ID, Cycle_Index, Capacity_Ah, Nominal, Health,
                   Discharge_Time, Temp_Mean, Voltage_Mean, Voltage_Min, Current_Mean
            FROM CYCLE_HEALTH
            ORDER BY Battery_ID, Cycle_Index
        """
        try:
            self.df = pd.read_sql(query, self.engine)
        except Exception as e:
            print(f"Database error: {e}")
            return

        if self.df.empty:
            print("No data found.")
            return

        self.batteries = sorted(self.df['Battery_ID'].unique())
        self.colors = sns.color_palette("husl", len(self.batteries))
        self.active_line = None

        self.view_height = self.VIEW_Y_MAX - self.VIEW_Y_MIN
        self.y_top_limit = max(self.VIEW_Y_MAX, self.df['Health'].max() + 0.05)
        self.y_bottom_limit = min(self.VIEW_Y_MIN, self.df['Health'].min() - 0.05)

        self._build_figure()

    def _build_figure(self):
        self.fig, self.ax = plt.subplots(
            figsize=(15, 8), constrained_layout=True
        )
        self.fig.patch.set_facecolor('#ffffff')

        self.lines = []
        for i, battery in enumerate(self.batteries):
            bdf = self.df[self.df['Battery_ID'] == battery].sort_values('Cycle_Index')
            line, = self.ax.plot(
                bdf['Cycle_Index'], bdf['Health'],
                linestyle='-', linewidth=1.5, alpha=0.75,
                color=self.colors[i], label=battery, marker='', picker=5
            )
            self.lines.append(line)

        # Failure threshold (0.70)
        self.ax.axhline(
            0.70, color='#C44E52', linestyle='--',
            linewidth=2.0, zorder=10,
            label='Failure threshold (Health = 0.70)'
        )

        self.ax.set_title(
            'HEALTH (Capacity / Nominal) PER BATTERY\n'
            'Click legend to focus | Mouse wheel = move view up/down | '
            'Left click = pin | Right click = delete',
            fontsize=13, fontweight='bold'
        )
        self.ax.set_xlabel('Cycle Index', fontsize=11)
        self.ax.set_ylabel('Health', fontsize=11)
        self.ax.grid(False)
        self.ax.fill_between(
            [0, self.df['Cycle_Index'].max()],
            0, 0.70, alpha=0.04, color='#C44E52'
        )
        self.ax.set_ylim(self.VIEW_Y_MIN, self.VIEW_Y_MAX)

        # Legend
        self.leg = self.ax.legend(
            bbox_to_anchor=(1.01, 1), loc='upper left',
            fontsize=8.5, ncol=2,
            framealpha=0.95, edgecolor='#CCCCCC',
            title='BATTERIES', title_fontsize=9
        )

        # Legend picker (μόνο για τις μπαταρίες, όχι για τη γραμμή του 0.70)
        self.legend_map = {}
        for legline, legtext, line in zip(
            self.leg.get_lines(), self.leg.get_texts(), self.lines
        ):
            legline.set_picker(True)
            legline.set_pickradius(8)
            legtext.set_picker(True)
            self.legend_map[legline] = line
            self.legend_map[legtext] = line

        # Hover
        cursor = mplcursors.cursor(self.lines, hover=True)

        @cursor.connect("add")
        def on_hover(sel):
            if self.active_line and sel.artist != self.active_line:
                sel.annotation.set_visible(False)
                return
            battery = sel.artist.get_label()
            x, y = sel.target
            color = sel.artist.get_color()
            sel.annotation.set_text(self._info_text(battery, x, y))
            sel.annotation.get_bbox_patch().set(
                fc=color, alpha=0.9, edgecolor='white', boxstyle='round'
            )
            sel.annotation.set_color('white')

        # Events
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def _info_text(self, battery, x, y):
        row = self.df[
            (self.df['Battery_ID'] == battery) &
            (self.df['Cycle_Index'] == int(round(x)))
        ]
        if row.empty:
            return f"Battery: {battery}\nCycle: {int(round(x))}\nHealth: {y:.3f}"
        r = row.iloc[0]
        return (
            f"Battery: {battery}\n"
            f"Cycle: {int(round(x))}\n"
            f"Health: {y:.3f}\n"
            f"Cap: {r['Capacity_Ah']:.3f} Ah\n"
            f"Nominal: {r['Nominal']:.3f} Ah\n"
            f"Disch. time: {r['Discharge_Time']:.0f} s\n"
            f"Temp mean: {r['Temp_Mean']:.1f}\n"
            f"V mean: {r['Voltage_Mean']:.3f}\n"
            f"V min: {r['Voltage_Min']:.3f}\n"
            f"I mean: {r['Current_Mean']:.3f}"
        )

    def on_pick(self, event):
        line = self.legend_map.get(event.artist)
        if line is None:
            return

        if self.active_line == line:
            self.active_line = None
            for l, lt in zip(self.lines, self.leg.get_texts()):
                l.set_alpha(0.75)
                l.set_linewidth(1.5)
                lt.set_alpha(1.0)
                lt.set_fontweight('normal')
        else:
            self.active_line = line
            for l, lt in zip(self.lines, self.leg.get_texts()):
                if l == line:
                    l.set_alpha(1.0)
                    l.set_linewidth(3.0)
                    lt.set_alpha(1.0)
                    lt.set_fontweight('bold')
                else:
                    l.set_alpha(0.05)
                    l.set_linewidth(0.8)
                    lt.set_alpha(0.3)
                    lt.set_fontweight('normal')

        self.fig.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Αριστερό κλικ: pin annotation
        if event.button == 1:
            for ann in list(self.ax.texts):
                cont, _ = ann.contains(event)
                if cont:
                    max_z = max(a.get_zorder() for a in self.ax.texts)
                    ann.set_zorder(max_z + 1)
                    self.fig.canvas.draw()
                    return

            lines_to_check = [self.active_line] if self.active_line else self.lines
            for line in lines_to_check:
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                if len(xdata) == 0:
                    continue
                distances = abs(xdata - event.xdata)
                idx = distances.argmin()
                if distances[idx] > 3:
                    continue
                x, y = xdata[idx], ydata[idx]
                color = line.get_color()
                battery = line.get_label()

                existing = [
                    (a.get_text().split("\n")[0], int(a.xy[0]))
                    for a in self.ax.texts if hasattr(a, 'xy')
                ]
                if (f"Battery: {battery}", int(x)) in existing:
                    return

                self.ax.annotate(
                    self._info_text(battery, x, y),
                    xy=(x, y), xytext=(15, 25),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", fc=color,
                              ec="white", alpha=0.9),
                    color="white", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black"),
                    zorder=999
                )
                self.fig.canvas.draw()
                break

        # Δεξί κλικ: διαγραφή annotation
        elif event.button == 3:
            overlapping = [
                ann for ann in list(self.ax.texts)
                if ann.contains(event)[0]
            ]
            if overlapping:
                max(overlapping, key=lambda a: a.get_zorder()).remove()
                self.fig.canvas.draw()

    def on_scroll(self, event):
        # Ο τροχός μετακινεί μόνο τον κάθετο άξονα
        if event.inaxes != self.ax:
            return

        lo, hi = self.ax.get_ylim()
        step = self.SCROLL_STEP if event.button == 'up' else -self.SCROLL_STEP

        new_lo = lo + step
        new_lo = max(self.y_bottom_limit, new_lo)
        new_lo = min(self.y_top_limit - self.view_height, new_lo)

        self.ax.set_ylim(new_lo, new_lo + self.view_height)
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    plotter = HealthPlotter()
    print("ALL SYSTEMS GO: Health Plotter is ready.")
    plt.show()