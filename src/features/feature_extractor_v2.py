import os
import sys
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from db_connection import get_engine

# ── PARAMETERS ───────────────────────────────────────────────────────────────
MOVING_AVERAGE_WINDOW       = 5      # κύκλοι για εξομάλυνση
CENSORED_MAX_FINAL_SOH      = 0.80   # censored μόνο αν SoH τελευταίου κύκλου < 0.80
MIN_CENSORED_CYCLES         = 20     # ελάχιστοι valid κύκλοι για censored μπαταρία
FLAG_THRESHOLD              = 0.5    # κάτω από 50% του max → impedance cycle
FAILURE_THRESHOLD_SOH       = 0.70   # κάτω από 70% SoH → failure
RECOVERY_THRESHOLD_SOH      = 0.75   # πάνω από 75% → πραγματική ανάκαμψη
RECOVERY_CONSECUTIVE_CYCLES = 3      # συνεχόμενοι κύκλοι για επιβεβαίωση ανάκαμψης


def extract_features_v2():
    engine = get_engine()
    if engine is None:
        print("Database connection failed.")
        return

    # ── STEP 1: Load discharge cycles ─────────────────────────────────────────
    print("Step 1/4: Loading discharge cycles from TEST_CYCLES...")

    cycles_query = """
        SELECT
            tc.Cycle_ID,
            tc.Battery_ID,
            tc.Cycle_Index,
            tc.Capacity_Ah
        FROM TEST_CYCLES AS tc
        INNER JOIN BATTERIES AS b
            ON tc.Battery_ID = b.Battery_ID
        WHERE tc.Operation_Type = 'discharge'
        ORDER BY tc.Battery_ID, tc.Cycle_Index
    """
    cycles_df = pd.read_sql(cycles_query, engine)
    print(f"   Found {len(cycles_df)} discharge cycles across "
          f"{cycles_df['Battery_ID'].nunique()} batteries.")

    # ── STEP 2: Moving Average + SoH + Flag + RUL per battery ────────────────
    print("Step 2/4: Computing Moving Average, SoH, Flag and RUL...")

    results = []

    print(f"\n── BATTERY SUMMARY ──")
    print(f"{'Battery':>10} {'Cycles':>8} {'Max_MA':>8} "
          f"{'Threshold':>10} {'Valid':>7} {'Failure':>9} {'RUL_max':>9}")
    print("-" * 65)

    for battery_id, group in cycles_df.groupby("Battery_ID"):
        group = group.copy().sort_values("Cycle_Index")

        # STEP 1.1: Moving Average
        group["Capacity_MA"] = (
            group["Capacity_Ah"]
            .rolling(window=MOVING_AVERAGE_WINDOW, min_periods=1)
            .mean()
        )

        # STEP 1.2: Nominal με IQR φίλτρο
        group_normal = group[group["Capacity_Ah"] > 0.5].copy()
        first_10 = group_normal["Capacity_Ah"].head(10)
        q75 = first_10.quantile(0.75)
        q25 = first_10.quantile(0.25)
        iqr = q75 - q25
        filtered = first_10[first_10 <= q75 + 1.5 * iqr]
        nominal = filtered.max()
        group["Nominal"] = nominal

        # STEP 1.3: SoH με clip ώστε να μην ξεπερνά 1.0
        group["SoH"] = group["Capacity_MA"] / nominal

        # STEP 1.4: Flag
        group["Flag"] = (group["SoH"] >= FLAG_THRESHOLD).astype(int)

        # STEP 1.5: RUL (confirmed failure ή censored)
        group_valid = group[group["Flag"] == 1].copy()
        failure_threshold = nominal * FAILURE_THRESHOLD_SOH

        group["Is_Censored"] = 0
        group["RUL"] = np.nan
        failure_cycle = "N/A"
        rul_max = "N/A"

        healthy_cycles = group_valid[group_valid["SoH"] > RECOVERY_THRESHOLD_SOH]

        if not healthy_cycles.empty:
            first_healthy = healthy_cycles["Cycle_Index"].iloc[0]

            failed_after_healthy = group_valid[
                (group_valid["Cycle_Index"] > first_healthy) &
                (group_valid["Capacity_Ah"] < failure_threshold)
            ]

            if not failed_after_healthy.empty:
                # Confirmed failure
                failure_cycle = failed_after_healthy["Cycle_Index"].iloc[0]
                group["RUL"] = (failure_cycle - group["Cycle_Index"]).clip(lower=0)

            elif (len(group_valid) >= MIN_CENSORED_CYCLES and
                  group_valid["SoH"].iloc[-1] < CENSORED_MAX_FINAL_SOH):
                # Censored: RUL = κάτω όριο
                last_cycle = group_valid["Cycle_Index"].max()
                group["RUL"] = (last_cycle - group["Cycle_Index"]).clip(lower=0)
                group["Is_Censored"] = 1
                failure_cycle = "CENS"

            group.loc[group["Flag"] == 0, "RUL"] = np.nan

            if group["RUL"].notna().any():
                rul_max = int(group["RUL"].max())

        print(f"{battery_id:>10} {len(group_valid):>8} {nominal:>8.3f} "
              f"{nominal * FAILURE_THRESHOLD_SOH:>10.3f} "
              f"{len(group_valid):>7} {str(failure_cycle):>9} {str(rul_max):>9}")

        results.append(group)

    print("-" * 65)

    # ── ALL BATTERIES TOGETHER ──────────────────────────────────────────
    final_df = pd.concat(results, ignore_index=True)

    # Αποθηκεύουμε όλους τους κύκλους για έλεγχο -gia to 0
    final_df.to_sql(
        "CYCLE_FEATURES_ALL",
        con=engine,
        if_exists="replace",
        index=False
)

    # WE KEEP CYCLES ONLY WITH Flag=1 AND KNOWN RUL
    before = len(final_df)
    final_df = final_df[
        (final_df["Flag"] == 1) &
        (final_df["RUL"].notna())
    ].copy()

    print(f"\n   Kept {len(final_df)} valid cycles from "
          f"{final_df['Battery_ID'].nunique()} batteries "
          f"(dropped {before - len(final_df)}).")

    if final_df.empty:
        print("No valid cycles found.")
        return

    final_df["RUL"] = final_df["RUL"].astype(int)

    # ── STEP 3: Extract features from MEASUREMENTS ────────────────────────────
    print("\nStep 3/4: Extracting features from MEASUREMENTS...")

    cycle_ids = final_df["Cycle_ID"].tolist()
    ids_str   = ",".join(str(i) for i in cycle_ids)

    meas_query = f"""
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
    """
    features_df = pd.read_sql(meas_query, engine)
    print(f"   Features extracted for {len(features_df)} cycles.")

    # ── STEP 4: Merge and upload ───────────────────────────────────────────────
    print("\nStep 4/4: Merging and uploading to CYCLE_FEATURES...")

    output_df = final_df.merge(features_df, on="Cycle_ID", how="inner")

    output_df = output_df[[
        "Cycle_ID", "Battery_ID", "Cycle_Index",
        "Capacity_Ah", "Capacity_MA", "Nominal", "SoH", "Flag",
        "Discharge_Time", "Temp_Mean", "Temp_Max",
        "Voltage_Min", "Voltage_Mean", "Current_Mean",
        "RUL", "Is_Censored"
    ]].copy()

    confirmed_df = output_df[output_df["Is_Censored"] == 0].copy()

    print(f"   Confirmed: {len(confirmed_df)} rows, "
          f"{confirmed_df['Battery_ID'].nunique()} batteries")
    print(f"   Total (with censored): {len(output_df)} rows, "
          f"{output_df['Battery_ID'].nunique()} batteries")
    print(f"   RUL range: {output_df['RUL'].min()} – "
          f"{output_df['RUL'].max()} cycles")
    print(f"   Batteries: {sorted(output_df['Battery_ID'].unique())}")

    confirmed_df.to_sql(
        "CYCLE_FEATURES", con=engine,
        if_exists="replace", index=False, chunksize=200
    )
    output_df.to_sql(
        "CYCLE_FEATURES_CENSORED", con=engine,
        if_exists="replace", index=False, chunksize=200
    )

    print("\nFeature extraction v2 complete! CYCLE_FEATURES is ready.")


if __name__ == "__main__":
    extract_features_v2()



    # ερωτηση μπαταρια 41 τι να την κανουμε
    # να κρατσησουμε την πρωτη  πτωση ή λκαι μετα που ξαναανεβαινει πχ Β0033?