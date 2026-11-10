import os
import sys
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from db_connection import get_engine

# ── PARAMETERS ───────────────────────────────────────────────────────────────-
MIN_VALID_CYCLES = 10   # minimum valid cycles for a battery to be included
MAX_HEALTH = 1.15   # cycles above this are treated as outliers
DROP_FIRST_CYCLE = True   # the first cycle of each battery is not representative
MIN_HEALTH = 0.5    # cycles below this are measurement dips, not degradation
REF_CYCLES = 5      # first cycles of each battery used as reference for relative features
RELATIVE_COLS = ["Voltage_Mean", "Voltage_Min", "Temp_Mean"]


def build_cycle_health():
    engine = get_engine()
    if engine is None:
        print("Database connection failed.")
        return

    # ── STEP 1: Valid cycles of all batteries ─────────────────────────────────
    print("Step 1/3: Loading valid cycles from CYCLE_FEATURES_ALL...")
    base_df = pd.read_sql("""
        SELECT Cycle_ID, Battery_ID, Cycle_Index, Capacity_Ah, Nominal
        FROM CYCLE_FEATURES_ALL
        WHERE Flag = 1
        ORDER BY Battery_ID, Cycle_Index
    """, engine)

    counts = base_df.groupby("Battery_ID")["Cycle_ID"].count()
    excluded = counts[counts < MIN_VALID_CYCLES]
    base_df = base_df[base_df["Battery_ID"].isin(counts[counts >= MIN_VALID_CYCLES].index)].copy()
    print(f"   {base_df['Battery_ID'].nunique()} batteries, {len(base_df)} cycles.")
    if len(excluded) > 0:
        print(f"   Excluded (< {MIN_VALID_CYCLES} valid cycles): {excluded.to_dict()}")

    # ── STEP 2: Target: raw capacity / Nominal ────────────────────────────────
    base_df["Health"] = base_df["Capacity_Ah"] / base_df["Nominal"]

    n_before = len(base_df)
    base_df = base_df[base_df["Health"] <= MAX_HEALTH].copy()
    print(f"   Removed {n_before - len(base_df)} cycles with Health > {MAX_HEALTH}.")

    n_before = len(base_df)
    base_df = base_df[base_df["Health"] >= MIN_HEALTH].copy()
    print(f"   Removed {n_before - len(base_df)} cycles with Health < {MIN_HEALTH}.")

    if DROP_FIRST_CYCLE:
        n_before = len(base_df)
        first_idx = base_df.groupby("Battery_ID")["Cycle_Index"].transform("min")
        base_df = base_df[base_df["Cycle_Index"] != first_idx].copy()
        print(f"   Removed {n_before - len(base_df)} first cycles.")

    # ── STEP 3: Features from MEASUREMENTS ────────────────────────────────────
    print("Step 2/3: Extracting features from MEASUREMENTS...")
    ids_str = ",".join(str(i) for i in base_df["Cycle_ID"].tolist())
    features_df = pd.read_sql(f"""
        SELECT
            Cycle_ID,
            MAX(Time_Seconds)        AS Discharge_Time,
            AVG(Temperature_Measure) AS Temp_Mean,
            MAX(Temperature_Measure) AS Temp_Max,
            MIN(Voltage_Measured)    AS Voltage_Min,
            AVG(Voltage_Measured)    AS Voltage_Mean,
            AVG(Current_Measured)    AS Current_Mean
        FROM MEASUREMENTS
        WHERE Cycle_ID IN ({ids_str})
          AND Current_Measured < -0.1
        GROUP BY Cycle_ID
    """, engine)

    out = base_df.merge(features_df, on="Cycle_ID", how="inner")
    print(f"   Rows after merge: {len(out)} (lost {len(base_df) - len(out)} without measurements).")

    out = out[[
        "Cycle_ID", "Battery_ID", "Cycle_Index",
        "Capacity_Ah", "Nominal", "Health",
        "Discharge_Time", "Temp_Mean", "Temp_Max",
        "Voltage_Min", "Voltage_Mean", "Current_Mean"
    ]].copy()

    # Relative features: change from the mean of each battery's first REF_CYCLES cycles
    out = out.sort_values(["Battery_ID", "Cycle_Index"]).reset_index(drop=True)
    for col in RELATIVE_COLS:
        ref = out.groupby("Battery_ID")[col].transform(lambda s: s.head(REF_CYCLES).mean())
        out["d" + col] = out[col] - ref

    # ── STEP 4: Save and summary ──────────────────────────────────────────────
    print("Step 3/3: Saving CYCLE_HEALTH...")
    out.to_sql("CYCLE_HEALTH", con=engine, if_exists="replace",
               index=False, chunksize=200)

    print("\n── HEALTH SUMMARY PER BATTERY ──")
    print(f"{'Battery':>8} {'Cycles':>7} {'Nominal':>8} "
          f"{'H_start':>8} {'H_end':>7} {'H_max':>7}")
    print("-" * 52)
    for bid, g in out.groupby("Battery_ID"):
        g = g.sort_values("Cycle_Index")
        print(f"{bid:>8} {len(g):>7} {g['Nominal'].iloc[0]:>8.3f} "
              f"{g['Health'].head(5).median():>8.3f} "
              f"{g['Health'].tail(5).median():>7.3f} "
              f"{g['Health'].max():>7.3f}")

    print("\nCYCLE_HEALTH is ready.")


if __name__ == "__main__":
    build_cycle_health()