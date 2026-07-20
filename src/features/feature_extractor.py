"""
Create cycle-level battery features and a model-ready RUL dataset.

Outputs
-------
dbo.CYCLE_FEATURES_ALL
    Diagnostic table containing every discharge cycle, extracted features,
    the battery-specific failure threshold, the detected failure cycle,
    eligibility flags, and exclusion reasons.

dbo.CYCLE_FEATURES
    Model-ready table containing only cycles up to and including the detected
    failure cycle, with at most one RUL = 0 row per eligible battery.

Methodological note
-------------------
The failure cycle is defined as the beginning of the final continuous run in
which Capacity_Ah remains below 70% of the battery's Nominal_Capacity. Earlier
threshold crossings followed by recovery are flagged as temporary crossings
and excluded from the model-ready table instead of being treated as permanent
End of Life.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


# Make src importable when this file is run directly from src/features.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db_connection import get_engine  # pylint: disable=wrong-import-position


FAILURE_THRESHOLD_RATIO: Final[float] = 0.70
ACTIVE_CURRENT_THRESHOLD_A: Final[float] = -0.1
MIN_ACTIVE_MEASUREMENTS: Final[int] = 2
DATABASE_SCHEMA: Final[str] = "dbo"

MODEL_FEATURE_COLUMNS: Final[list[str]] = [
    "Capacity_Ah",
    "Discharge_Time",
    "Temp_Mean",
    "Temp_Max",
    "Voltage_Min",
    "Voltage_Mean",
    "Current_Mean",
    "RUL",
]

MODEL_OUTPUT_COLUMNS: Final[list[str]] = [
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


def load_discharge_cycles(engine: Engine) -> pd.DataFrame:
    """Load every discharge cycle and its battery nominal capacity."""
    query = text(
        """
        SELECT
            tc.Cycle_ID,
            tc.Battery_ID,
            tc.Cycle_Index,
            tc.Capacity_Ah,
            b.Nominal_Capacity
        FROM dbo.TEST_CYCLES AS tc
        INNER JOIN dbo.BATTERIES AS b
            ON tc.Battery_ID = b.Battery_ID
        WHERE tc.Operation_Type = 'discharge'
        ORDER BY tc.Battery_ID, tc.Cycle_Index
        """
    )

    cycles_df = pd.read_sql(query, engine)

    if cycles_df.empty:
        raise ValueError("No discharge cycles were found in dbo.TEST_CYCLES.")

    duplicate_cycle_ids = int(cycles_df["Cycle_ID"].duplicated().sum())
    if duplicate_cycle_ids:
        raise ValueError(
            f"Found {duplicate_cycle_ids} duplicate Cycle_ID values in TEST_CYCLES."
        )

    duplicate_battery_cycles = int(
        cycles_df.duplicated(subset=["Battery_ID", "Cycle_Index"]).sum()
    )
    if duplicate_battery_cycles:
        raise ValueError(
            "Found duplicate Battery_ID/Cycle_Index pairs in TEST_CYCLES: "
            f"{duplicate_battery_cycles}."
        )

    numeric_columns = ["Cycle_ID", "Cycle_Index", "Capacity_Ah", "Nominal_Capacity"]
    for column in numeric_columns:
        cycles_df[column] = pd.to_numeric(cycles_df[column], errors="coerce")

    invalid_identifiers = cycles_df[["Cycle_ID", "Cycle_Index"]].isna().any(axis=1)
    if invalid_identifiers.any():
        raise ValueError(
            "TEST_CYCLES contains rows with invalid Cycle_ID or Cycle_Index."
        )

    print(
        f"   Found {len(cycles_df)} discharge cycles across "
        f"{cycles_df['Battery_ID'].nunique()} batteries."
    )

    return cycles_df


def load_active_discharge_features(engine: Engine) -> pd.DataFrame:
    """Aggregate measurements only during active discharge."""
    query = text(
        """
        SELECT
            m.Cycle_ID,
            MAX(m.Time_Seconds) - MIN(m.Time_Seconds) AS Discharge_Time,
            AVG(m.Temperature_Measure) AS Temp_Mean,
            MAX(m.Temperature_Measure) AS Temp_Max,
            MIN(m.Voltage_Measured) AS Voltage_Min,
            AVG(m.Voltage_Measured) AS Voltage_Mean,
            AVG(m.Current_Measured) AS Current_Mean,
            COUNT(*) AS Active_Measurement_Count
        FROM dbo.MEASUREMENTS AS m
        INNER JOIN dbo.TEST_CYCLES AS tc
            ON m.Cycle_ID = tc.Cycle_ID
        WHERE
            tc.Operation_Type = 'discharge'
            AND m.Current_Measured < :active_current_threshold
        GROUP BY m.Cycle_ID
        HAVING COUNT(*) >= :minimum_measurements
        """
    )

    features_df = pd.read_sql(
        query,
        engine,
        params={
            "active_current_threshold": ACTIVE_CURRENT_THRESHOLD_A,
            "minimum_measurements": MIN_ACTIVE_MEASUREMENTS,
        },
    )

    numeric_columns = [
        "Cycle_ID",
        "Discharge_Time",
        "Temp_Mean",
        "Temp_Max",
        "Voltage_Min",
        "Voltage_Mean",
        "Current_Mean",
        "Active_Measurement_Count",
    ]
    for column in numeric_columns:
        features_df[column] = pd.to_numeric(features_df[column], errors="coerce")

    print(f"   Extracted active-discharge features for {len(features_df)} cycles.")
    return features_df


def assign_rul_to_battery(group: pd.DataFrame) -> pd.DataFrame:
    """
    Detect persistent End of Life and calculate RUL for one battery.

    Failure is the first cycle of the final continuous below-threshold run.
    Earlier below-threshold runs followed by recovery are flagged and excluded.
    """
    group = group.sort_values("Cycle_Index").reset_index(drop=True).copy()

    group["Failure_Threshold_Ah"] = np.nan
    group["Failure_Cycle"] = pd.NA
    group["RUL"] = np.nan
    group["Model_Eligible"] = False
    group["Exclusion_Reason"] = ""

    nominal_values = group["Nominal_Capacity"].dropna().unique()

    if len(nominal_values) != 1 or float(nominal_values[0]) <= 0:
        group["Exclusion_Reason"] = "INVALID_OR_INCONSISTENT_NOMINAL_CAPACITY"
        return group

    nominal_capacity = float(nominal_values[0])
    threshold = nominal_capacity * FAILURE_THRESHOLD_RATIO
    group["Failure_Threshold_Ah"] = threshold

    valid_capacity = group["Capacity_Ah"].notna()
    if not valid_capacity.any():
        group["Exclusion_Reason"] = "MISSING_CAPACITY"
        return group

    # A missing capacity interrupts continuity and cannot support EOL detection.
    below_threshold = valid_capacity & group["Capacity_Ah"].lt(threshold)

    valid_positions = np.flatnonzero(valid_capacity.to_numpy())
    last_valid_position = int(valid_positions[-1])

    if not bool(below_threshold.iloc[last_valid_position]):
        group["Exclusion_Reason"] = "NO_OBSERVED_PERSISTENT_FAILURE"
        return group

    failure_position = last_valid_position

    while failure_position > 0:
        previous_position = failure_position - 1

        if not bool(valid_capacity.iloc[previous_position]):
            break

        if not bool(below_threshold.iloc[previous_position]):
            break

        failure_position = previous_position

    failure_cycle = int(group.iloc[failure_position]["Cycle_Index"])
    group["Failure_Cycle"] = failure_cycle

    if failure_position == 0:
        group["Exclusion_Reason"] = "NO_PRE_FAILURE_HISTORY"
        return group

    group["RUL"] = failure_cycle - group["Cycle_Index"]

    temporary_crossing = below_threshold & group["Cycle_Index"].lt(failure_cycle)
    post_eol = group["Cycle_Index"].gt(failure_cycle)
    pre_eol_or_failure = group["Cycle_Index"].le(failure_cycle)

    group.loc[temporary_crossing, "Exclusion_Reason"] = (
        "TEMPORARY_THRESHOLD_CROSSING"
    )
    group.loc[post_eol, "Exclusion_Reason"] = "POST_EOL"

    # Keep pre-EOL cycles and the first EOL cycle. Temporary crossings are
    # excluded because they contradict a persistent degradation trajectory.
    group["Model_Eligible"] = pre_eol_or_failure & ~temporary_crossing

    # RUL must never be negative in the diagnostic table.
    group["RUL"] = group["RUL"].clip(lower=0)

    return group


def apply_rul_logic(all_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the battery-level RUL calculation to the full merged dataset."""
    processed_groups = [
        assign_rul_to_battery(group)
        for _, group in all_df.groupby("Battery_ID", sort=False)
    ]

    result = pd.concat(processed_groups, ignore_index=True)
    result["Failure_Cycle"] = result["Failure_Cycle"].astype("Int64")
    result["RUL"] = result["RUL"].round().astype("Int64")

    complete_model_features = result[MODEL_FEATURE_COLUMNS].notna().all(axis=1)

    missing_features_mask = (
        result["Model_Eligible"]
        & ~complete_model_features
        & result["Exclusion_Reason"].eq("")
    )

    result.loc[missing_features_mask, "Exclusion_Reason"] = (
        "MISSING_EXTRACTED_FEATURES"
    )
    result.loc[missing_features_mask, "Model_Eligible"] = False

    return result


def build_model_table(all_df: pd.DataFrame) -> pd.DataFrame:
    """Create the final model-ready table from eligible diagnostic rows."""
    model_df = all_df.loc[all_df["Model_Eligible"]].copy()

    if model_df.empty:
        raise ValueError(
            "No model-eligible rows remain. Review thresholds and source data."
        )

    model_df["RUL"] = model_df["RUL"].astype(int)
    model_df = model_df[MODEL_OUTPUT_COLUMNS].copy()

    duplicate_count = int(
        model_df.duplicated(subset=["Battery_ID", "Cycle_Index"]).sum()
    )
    if duplicate_count:
        raise ValueError(
            f"Model table contains {duplicate_count} duplicate battery-cycle rows."
        )

    zero_counts = model_df.loc[model_df["RUL"].eq(0)].groupby("Battery_ID").size()
    batteries_with_multiple_zeros = zero_counts[zero_counts.gt(1)]

    if not batteries_with_multiple_zeros.empty:
        raise ValueError(
            "More than one RUL = 0 row was found for these batteries: "
            f"{batteries_with_multiple_zeros.to_dict()}"
        )

    return model_df


def print_quality_summary(all_df: pd.DataFrame, model_df: pd.DataFrame) -> None:
    """Print diagnostics before writing the output tables."""
    print("\nQuality summary")
    print("---------------")
    print(f"Diagnostic rows: {len(all_df)}")
    print(f"Diagnostic batteries: {all_df['Battery_ID'].nunique()}")
    print(f"Model rows: {len(model_df)}")
    print(f"Model batteries: {model_df['Battery_ID'].nunique()}")
    print(f"Model RUL = 0 rows: {int(model_df['RUL'].eq(0).sum())}")
    print(f"Model RUL range: {model_df['RUL'].min()} - {model_df['RUL'].max()}")

    reason_counts = (
        all_df.loc[all_df["Exclusion_Reason"].ne(""), "Exclusion_Reason"]
        .value_counts()
        .sort_index()
    )

    print("\nExcluded/flagged rows by reason:")
    if reason_counts.empty:
        print("   None")
    else:
        for reason, count in reason_counts.items():
            print(f"   {reason}: {count}")

    battery_summary = (
        all_df.groupby("Battery_ID", as_index=False)
        .agg(
            Failure_Cycle=("Failure_Cycle", "first"),
            Model_Eligible_Rows=("Model_Eligible", "sum"),
            Total_Rows=("Cycle_ID", "count"),
        )
        .sort_values("Battery_ID")
    )

    print("\nBattery-level summary:")
    print(battery_summary.to_string(index=False))


def upload_tables(engine: Engine, all_df: pd.DataFrame, model_df: pd.DataFrame) -> None:
    """Replace both SQL output tables inside one database transaction."""
    with engine.begin() as connection:
        all_df.to_sql(
            "CYCLE_FEATURES_ALL",
            con=connection,
            schema=DATABASE_SCHEMA,
            if_exists="replace",
            index=False,
            chunksize=200,
        )

        model_df.to_sql(
            "CYCLE_FEATURES",
            con=connection,
            schema=DATABASE_SCHEMA,
            if_exists="replace",
            index=False,
            chunksize=200,
        )


def extract_features() -> None:
    """Run the complete feature-extraction and RUL-construction pipeline."""
    engine = get_engine()

    if engine is None:
        print("Database connection failed.")
        return

    try:
        print("Step 1/5: Loading discharge cycles from TEST_CYCLES...")
        cycles_df = load_discharge_cycles(engine)

        print("Step 2/5: Extracting active-discharge features...")
        features_df = load_active_discharge_features(engine)

        print("Step 3/5: Merging cycles and extracted features...")
        all_df = cycles_df.merge(
            features_df,
            on="Cycle_ID",
            how="left",
            validate="one_to_one",
        )

        missing_feature_rows = int(all_df["Discharge_Time"].isna().sum())
        if missing_feature_rows:
            print(
                f"   Warning: {missing_feature_rows} cycles have no valid "
                "active-discharge feature row."
            )

        print("Step 4/5: Detecting persistent failure and calculating RUL...")
        all_df = apply_rul_logic(all_df)
        model_df = build_model_table(all_df)
        print_quality_summary(all_df, model_df)

        print("\nStep 5/5: Writing SQL output tables...")
        upload_tables(engine, all_df, model_df)

        print("\nFeature extraction completed successfully.")
        print("Created/replaced:")
        print("   dbo.CYCLE_FEATURES_ALL")
        print("   dbo.CYCLE_FEATURES")

    except (ValueError, KeyError, pd.errors.DatabaseError) as error:
        print(f"\nFeature extraction failed: {error}")
        raise


if __name__ == "__main__":
    extract_features()