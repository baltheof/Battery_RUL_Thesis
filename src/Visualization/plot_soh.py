import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_connection import get_engine


class SoHPlotter:
    def __init__(self):
        self.engine = get_engine()

        print("Loading CYCLE_FEATURES_ALL from database...")
        query = """
            SELECT Battery_ID, Cycle_Index, Capacity_Ah, 
                   Capacity_MA, SoH, Flag, RUL
            FROM CYCLE_FEATURES_ALL
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
        self.colors    = sns.color_palette("husl", len(self.batteries))
        self.active_line = None

        self._build_figure()

    def _build_figure(self):
        # ── Layout: 2 subplots ────────────────────────────────────────────────
        self.fig, (self.ax_soh, self.ax_flag) = plt.subplots(
            2, 1, figsize=(15, 11),
            gridspec_kw={'height_ratios': [3, 1]},
            constrained_layout=True
        )
        self.fig.patch.set_facecolor('#ffffff')

        self.soh_lines  = []
        self.flag_lines = []

        for i, battery in enumerate(self.batteries):
            bdf   = self.df[self.df['Battery_ID'] == battery].sort_values('Cycle_Index')
            color = self.colors[i]

            # ── SoH line ─────────────────────────────────────────────────────
            line_soh, = self.ax_soh.plot(
                bdf['Cycle_Index'], bdf['SoH'],
                linestyle='-', linewidth=1.5, alpha=0.75,
                color=color, label=battery, marker='', picker=5
            )
            self.soh_lines.append(line_soh)

            # ── Flag line ─────────────────────────────────────────────────────
            line_flag, = self.ax_flag.plot(
                bdf['Cycle_Index'], bdf['Flag'],
                linestyle='-', linewidth=1.5, alpha=0.75,
                color=color, label=battery, marker='', picker=5
            )
            self.flag_lines.append(line_flag)

        # ── Failure threshold line (0.70) ─────────────────────────────────────
        self.ax_soh.axhline(
            0.70, color='#C44E52', linestyle='--',
            linewidth=2.0, zorder=10,
            label='Failure threshold (SoH = 0.70)'
        )

        # ── Flag threshold line (0.50) ────────────────────────────────────────
        self.ax_soh.axhline(
            0.50, color='#FF9800', linestyle=':',
            linewidth=1.5, zorder=10,
            label='Flag threshold (SoH = 0.50)'
        )

        # ── SoH formatting ────────────────────────────────────────────────────
        self.ax_soh.set_title(
            'STATE OF HEALTH (SoH) PER BATTERY\nClick legend to focus',
            fontsize=13, fontweight='bold'
        )
        self.ax_soh.set_ylabel('SoH (0.0 – 1.0)', fontsize=11)
        self.ax_soh.set_ylim(-0.05, 1.10)
        self.ax_soh.grid(False)
        self.ax_soh.fill_between(
            [0, self.df['Cycle_Index'].max()],
            0, 0.70,
            alpha=0.04, color='#C44E52'
        )

        # ── Flag formatting ───────────────────────────────────────────────────
        self.ax_flag.set_title(
            'FLAG PER CYCLE  (1 = Normal | 0 = Impedance)',
            fontsize=11, fontweight='bold'
        )
        self.ax_flag.set_xlabel('Cycle Index', fontsize=11)
        self.ax_flag.set_ylabel('Flag', fontsize=11)
        self.ax_flag.set_ylim(-0.2, 1.4)
        self.ax_flag.set_yticks([0, 1])
        self.ax_flag.set_yticklabels(['0 — Impedance', '1 — Normal'])
        self.ax_flag.grid(False)

        # ── Legend (SoH plot only) ────────────────────────────────────────────
        self.leg = self.ax_soh.legend(
            bbox_to_anchor=(1.01, 1), loc='upper left',
            fontsize=8.5, ncol=2,
            framealpha=0.95, edgecolor='#CCCCCC',
            title='BATTERIES', title_fontsize=9
        )

        # ── Legend picker ─────────────────────────────────────────────────────
        self.legend_map = {}
        for legline, legtext, soh_line, flag_line in zip(
            self.leg.get_lines(), self.leg.get_texts(),
            self.soh_lines, self.flag_lines
        ):
            legline.set_picker(True)
            legline.set_pickradius(8)
            legtext.set_picker(True)
            self.legend_map[legline] = (soh_line, flag_line)
            self.legend_map[legtext] = (soh_line, flag_line)

        # ── Hover ─────────────────────────────────────────────────────────────
        cursor = mplcursors.cursor(self.soh_lines, hover=True)

        @cursor.connect("add")
        def on_hover(sel):
            if self.active_line and sel.artist != self.active_line:
                sel.annotation.set_visible(False)
                return
            battery = sel.artist.get_label()
            x, y    = sel.target
            color   = sel.artist.get_color()

            # Βρες RUL για αυτόν τον κύκλο
            bdf = self.df[
                (self.df['Battery_ID'] == battery) &
                (self.df['Cycle_Index'] == int(round(x)))
            ]
            rul_val = bdf['RUL'].values[0] if not bdf.empty else 'N/A'
            flag_val = int(bdf['Flag'].values[0]) if not bdf.empty else 'N/A'

            sel.annotation.set_text(
                f"Battery: {battery}\n"
                f"Cycle: {int(x)}\n"
                f"SoH: {y:.3f}\n"
                f"Flag: {flag_val}\n"
                f"RUL: {rul_val}"
            )
            sel.annotation.get_bbox_patch().set(
                fc=color, alpha=0.9, edgecolor='white', boxstyle='round'
            )
            sel.annotation.set_color('white')

        # ── Events ───────────────────────────────────────────────────────────
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_pick(self, event):
        artist = event.artist
        pair   = self.legend_map.get(artist)
        if pair is None:
            return

        soh_line, flag_line = pair

        # Toggle focus
        if self.active_line == soh_line:
            # Reset
            self.active_line = None
            for sl, fl in zip(self.soh_lines, self.flag_lines):
                sl.set_alpha(0.75)
                sl.set_linewidth(1.5)
                fl.set_alpha(0.75)
                fl.set_linewidth(1.5)
            for lt in self.leg.get_texts():
                lt.set_alpha(1.0)
                lt.set_fontweight('normal')
        else:
            self.active_line = soh_line
            for sl, fl, lt in zip(
                self.soh_lines, self.flag_lines, self.leg.get_texts()
            ):
                if sl == soh_line:
                    sl.set_alpha(1.0)
                    sl.set_linewidth(3.0)
                    fl.set_alpha(1.0)
                    fl.set_linewidth(3.0)
                    lt.set_alpha(1.0)
                    lt.set_fontweight('bold')
                else:
                    sl.set_alpha(0.05)
                    sl.set_linewidth(0.8)
                    fl.set_alpha(0.05)
                    fl.set_linewidth(0.8)
                    lt.set_alpha(0.3)
                    lt.set_fontweight('normal')

        self.fig.canvas.draw()

    def on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return

        # ── ΑΡΙΣΤΕΡΟ ΚΛΙΚ — PIN annotation ───────────────────────────────────
        if event.button == 1:
            # Έλεγξε αν κλικάρισες πάνω σε υπάρχον annotation
            for ax in [self.ax_soh, self.ax_flag]:
                for ann in list(ax.texts):
                    cont, _ = ann.contains(event)
                    if cont:
                        max_z = max(a.get_zorder() for a in ax.texts)
                        ann.set_zorder(max_z + 1)
                        self.fig.canvas.draw()
                        return

            # Αν είμαστε στο SoH plot
            if event.inaxes == self.ax_soh:
                lines_to_check = [self.active_line] if self.active_line else self.soh_lines
                for line in lines_to_check:
                    xdata = line.get_xdata()
                    ydata = line.get_ydata()
                    if len(xdata) == 0:
                        continue
                    distances = abs(xdata - event.xdata)
                    idx = distances.argmin()
                    if distances[idx] > 3:
                        continue
                    x, y  = xdata[idx], ydata[idx]
                    color = line.get_color()
                    battery = line.get_label()

                    # Έλεγχος duplicate
                    existing = [int(a.xy[0]) for a in self.ax_soh.texts
                                if hasattr(a, 'xy')]
                    if int(x) in existing:
                        return

                    # Βρες RUL και Flag
                    bdf = self.df[
                        (self.df['Battery_ID'] == battery) &
                        (self.df['Cycle_Index'] == int(round(x)))
                    ]
                    rul_val  = bdf['RUL'].values[0] if not bdf.empty else 'N/A'
                    flag_val = int(bdf['Flag'].values[0]) if not bdf.empty else 'N/A'

                    self.ax_soh.annotate(
                        f"Battery: {battery}\nCycle: {int(x)}\n"
                        f"SoH: {y:.3f}\nFlag: {flag_val}\nRUL: {rul_val}",
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

        # ── ΔΕΞΙ ΚΛΙΚ — ΔΙΑΓΡΑΦΗ annotation ─────────────────────────────────
        elif event.button == 3:
            for ax in [self.ax_soh, self.ax_flag]:
                if event.inaxes == ax:
                    overlapping = [
                        ann for ann in list(ax.texts)
                        if ann.contains(event)[0]
                    ]
                    if overlapping:
                        max(overlapping, key=lambda a: a.get_zorder()).remove()
                        self.fig.canvas.draw()
                    return


if __name__ == "__main__":
    plotter = SoHPlotter()
    print("ALL SYSTEMS GO: SoH Plotter is ready.")
    plt.show()