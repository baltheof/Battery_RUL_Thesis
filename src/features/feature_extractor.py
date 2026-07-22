import os
import sys


# Βρίσκουμε τον φάκελο src
SRC_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Προσθέτουμε τον src πριν από το import του db_connection
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


import numpy as np
import pandas as pd

from db_connection import get_engine


FAILURE_THRESHOLD_RATIO = 0.60


def extract_features():
    engine = get_engine()

    if engine is None:
        print("Database connection failed.")
        return


    # STEP 1: Load discharge cycles
    print("Step 1/4: Loading discharge cycles from TEST_CYCLES...")

    cycles_query = """
        SELECT
            tc.Cycle_ID,
            tc.Battery_ID,
            tc.Cycle_Index,
            tc.Capacity_Ah,
            b.Nominal_Capacity
        FROM TEST_CYCLES AS tc
        INNER JOIN BATTERIES AS b
            ON tc.Battery_ID = b.Battery_ID
        WHERE tc.Operation_Type = 'discharge'
        ORDER BY
            tc.Battery_ID,
            tc.Cycle_Index
    """

    cycles_df = pd.read_sql(
        cycles_query,
        engine
    )

    print(
        f"   Found {len(cycles_df)} discharge cycles across "
        f"{cycles_df['Battery_ID'].nunique()} batteries."
    )

    # STEP 2: Compute RUL per battery
    print("Step 2/4: Computing RUL for each cycle...")

    # ── DEBUG ──
    print("\n── BATTERY DEBUG ──")
    for battery_id, group in cycles_df.groupby("Battery_ID"):
        group_clean = group[group["Capacity_Ah"] > 0.5].copy()
        if group_clean.empty:
            print(f"{battery_id}: EMPTY after filter")
            continue

        nominal = group_clean["Capacity_Ah"].quantile(0.95)
        threshold = nominal * FAILURE_THRESHOLD_RATIO
        below = group_clean["Capacity_Ah"] < threshold
        failed = group_clean[group_clean["Capacity_Ah"] < threshold]

        status = "YES" if not failed.empty else "NO"

        print(f"{battery_id}: nominal={nominal:.3f}, threshold={threshold:.3f}, passed={status}, cycles={len(group_clean)}")
    print("──────────────────\n")

    rul_list = []

    for battery_id, group in cycles_df.groupby("Battery_ID"):
        group = group.copy()

        # Φίλτρο outliers
        group_clean = group[group["Capacity_Ah"] > 0.5].copy()

        if group_clean.empty:
            group["RUL"] = np.nan
            rul_list.append(group)
        else:
            # Dynamic threshold από 95th percentile
            nominal = group_clean["Capacity_Ah"].quantile(0.95)
            threshold = nominal * FAILURE_THRESHOLD_RATIO

            # Failure detection με 3 consecutive cycles
            below = group_clean["Capacity_Ah"] < threshold
            consecutive = below.rolling(3).sum() == 3
            failed = group_clean[consecutive]

            if failed.empty:
                group["RUL"] = np.nan
            else:
                failure_cycle = failed["Cycle_Index"].iloc[0]
                group["RUL"] = (
                    failure_cycle - group["Cycle_Index"]
                ).clip(lower=0)

            rul_list.append(group)


    # Ενώνουμε ξανά όλες τις επεξεργασμένες μπαταρίες
    cycles_df = pd.concat(
        rul_list,
        ignore_index=True
    )

    print(
        "Columns after RUL calculation:",
        cycles_df.columns.tolist()
    )

    # Αφαιρούμε τους κύκλους όπου το RUL είναι άγνωστο
    before = len(cycles_df)

    cycles_df = cycles_df.dropna(
        subset=["RUL"]
    ).copy()

    dropped = before - len(cycles_df)

    print(
        f"Dropped {dropped} cycles with undefined RUL."
    )

    print(
        f"RUL computed for {len(cycles_df)} cycles across "
        f"{cycles_df['Battery_ID'].nunique()} batteries."
    )

    # Αν δεν βρέθηκε καμία μπαταρία με γνωστό RUL, σταματάμε
    if cycles_df.empty:
        print("No cycles with valid RUL were found.")
        return

    # Το RUL μετριέται σε ακέραιους κύκλους
    cycles_df["RUL"] = cycles_df["RUL"].astype(int)

    # STEP 3: Extract features from measurements
    print("Step 3/4: Extracting features from MEASUREMENTS...")

    cycle_ids = cycles_df["Cycle_ID"].tolist()

    ids_str = ",".join(
        str(cycle_id)
        for cycle_id in cycle_ids
    )

    meas_query = f"""
        SELECT
            Cycle_ID,
            MAX(Time_Seconds) AS Discharge_Time,
            AVG(Temperature_Measure) AS Temp_Mean,
            MAX(Temperature_Measure) AS Temp_Max,
            MIN(Voltage_Measured) AS Voltage_Min,
            AVG(Voltage_Measured) AS Voltage_Mean,
            AVG(Current_Measured) AS Current_Mean
        FROM MEASUREMENTS
        WHERE
            Cycle_ID IN ({ids_str})
            AND Current_Measured < -0.1
        GROUP BY Cycle_ID
    """

    features_df = pd.read_sql(
        meas_query,
        engine
    )

    print(
        f"   Features extracted for "
        f"{len(features_df)} cycles."
    )

    # STEP 4: Merge and upload
    print(
        "Step 4/4: Merging and uploading "
        "to CYCLE_FEATURES..."
    )

    final_df = cycles_df.merge(
        features_df,
        on="Cycle_ID",
        how="inner"
    )

    final_df = final_df[
        [
            "Cycle_ID",
            "Battery_ID",
            "Cycle_Index",
            "Capacity_Ah",
            "Discharge_Time",
            "Temp_Mean",
            "Temp_Max",
            "Voltage_Min",
            "Voltage_Mean",
            "Current_Mean",
            "RUL",
        ]
    ].copy()

    final_df["RUL"] = final_df["RUL"].astype(int)

    print(
        f"   Final dataset: {len(final_df)} rows, "
        f"{final_df.shape[1]} columns."
    )

    print(
        f"   RUL range: "
        f"{final_df['RUL'].min()} – "
        f"{final_df['RUL'].max()} cycles"
    )

    final_df.to_sql(
        "CYCLE_FEATURES",
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=200
    )

    print(
        "\nFeature extraction complete! "
        "Table CYCLE_FEATURES is ready."
    )


if __name__ == "__main__":
    extract_features()