"""
Create cycle-level battery features and a model-ready RUL dataset.

Outputs
-------
dbo.CYCLE_FEATURES_ALL
    Diagnostic table containing every discharge cycle, extracted features,
    battery-specific threshold, detected failure cycle, eligibility flags,
    and exclusion reasons.

dbo.CYCLE_FEATURES
    Model-ready table containing only valid cycles up to and including the
    detected End-of-Life cycle, with at most one RUL = 0 row per battery.

RUL rule
--------
1. Capacity is smoothed with a centered rolling median.
2. A possible failure starts when capacity stays below 70% of nominal
   capacity for a minimum number of consecutive cycles.
3. If a sustained recovery above the threshold appears later, that crossing
   is treated as temporary and the search continues.
4. The accepted failure cycle is the first cycle of the first confirmed
   below-threshold run that is not followed by sustained recovery.
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


# ---------------------------------------------------------------------------
# Methodological parameters
# ---------------------------------------------------------------------------
FAILURE_THRESHOLD_RATIO: Final[float] = 0.70
ACTIVE_CURRENT_THRESHOLD_A: Final[float] = -0.1
MIN_ACTIVE_MEASUREMENTS: Final[int] = 2

# Rolling median used only for End-of-Life detection.
SMOOTHING_WINDOW_CYCLES: Final[int] = 3

# A failure candidate requires at least this many consecutive cycles below
# the threshold.
MIN_CONSECUTIVE_BELOW: Final[int] = 3

# A previous crossing is treated as temporary when at least this many
# consecutive cycles later return to or above the threshold.
MIN_CONSECUTIVE_RECOVERY_ABOVE: Final[int] = 5

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
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

    numeric_columns = [
        "Cycle_ID",
        "Cycle_Index",
        "Capacity_Ah",
        "Nominal_Capacity",
    ]
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
        features_df[column] = pd.to_numeric(
            features_df[column],
            errors="coerce",
        )

    print(f"   Extracted active-discharge features for {len(features_df)} cycles.")
    return features_df


# ---------------------------------------------------------------------------
# End-of-Life and RUL logic
# ---------------------------------------------------------------------------
def find_first_consecutive_run(
    mask: np.ndarray,
    run_length: int,
    start_position: int = 0,
) -> int | None:
    """Return the first position where a consecutive True run begins."""
    if run_length <= 0:
        raise ValueError("run_length must be positive.")

    consecutive_count = 0

    for position in range(start_position, len(mask)):
        if bool(mask[position]):
            consecutive_count += 1
        else:
            consecutive_count = 0

        if consecutive_count >= run_length:
            return position - run_length + 1

    return None


def detect_confirmed_failure_position(
    below_threshold: np.ndarray,
    above_threshold: np.ndarray,
) -> tuple[int | None, np.ndarray]:
    """
    Detect a confirmed failure position.

    A candidate requires MIN_CONSECUTIVE_BELOW consecutive cycles below the
    threshold. If a later run of MIN_CONSECUTIVE_RECOVERY_ABOVE consecutive
    cycles returns above the threshold, the candidate is temporary and the
    search resumes after that recovery.

    Returns
    -------
    tuple
        failure_position or None, and a boolean mask marking temporary
        below-threshold positions encountered before the accepted failure.
    """
    temporary_crossing_mask = np.zeros(len(below_threshold), dtype=bool)
    search_start = 0

    while search_start < len(below_threshold):
        candidate_start = find_first_consecutive_run(
            below_threshold,
            MIN_CONSECUTIVE_BELOW,
            start_position=search_start,
        )

        if candidate_start is None:
            return None, temporary_crossing_mask

        recovery_search_start = candidate_start + MIN_CONSECUTIVE_BELOW
        recovery_start = find_first_consecutive_run(
            above_threshold,
            MIN_CONSECUTIVE_RECOVERY_ABOVE,
            start_position=recovery_search_start,
        )

        if recovery_start is None:
            # No sustained recovery was observed after this candidate.
            return candidate_start, temporary_crossing_mask

        # Mark only below-threshold points before the sustained recovery as
        # temporary. The recovered region can still be used later.
        temporary_crossing_mask[candidate_start:recovery_start] |= (
            below_threshold[candidate_start:recovery_start]
        )

        search_start = recovery_start + MIN_CONSECUTIVE_RECOVERY_ABOVE

    return None, temporary_crossing_mask


def assign_rul_to_battery(group: pd.DataFrame) -> pd.DataFrame:
    """Detect confirmed End of Life and calculate RUL for one battery."""
    group = group.sort_values("Cycle_Index").reset_index(drop=True).copy()

    group["Capacity_Smoothed_Ah"] = np.nan
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

    # Centered rolling median reduces isolated capacity spikes. RUL labels are
    # constructed offline from the full degradation history, so using adjacent
    # cycles here is acceptable for target construction.
    smoothed_capacity = group["Capacity_Ah"].rolling(
        window=SMOOTHING_WINDOW_CYCLES,
        center=True,
        min_periods=1,
    ).median()

    # A missing raw capacity remains invalid even if neighbouring values would
    # allow the rolling median to be calculated.
    smoothed_capacity = smoothed_capacity.where(valid_capacity)
    group["Capacity_Smoothed_Ah"] = smoothed_capacity

    valid_smoothed = smoothed_capacity.notna()
    below_threshold = valid_smoothed & smoothed_capacity.lt(threshold)
    above_threshold = valid_smoothed & smoothed_capacity.ge(threshold)

    failure_position, temporary_mask_array = detect_confirmed_failure_position(
        below_threshold.to_numpy(dtype=bool),
        above_threshold.to_numpy(dtype=bool),
    )

    temporary_crossing = pd.Series(
        temporary_mask_array,
        index=group.index,
        dtype=bool,
    )

    if failure_position is None:
        group.loc[temporary_crossing, "Exclusion_Reason"] = (
            "TEMPORARY_THRESHOLD_CROSSING"
        )
        group.loc[
            group["Exclusion_Reason"].eq(""),
            "Exclusion_Reason",
        ] = "NO_OBSERVED_CONFIRMED_FAILURE"
        group.loc[~valid_capacity, "Exclusion_Reason"] = "MISSING_CAPACITY"
        return group

    failure_cycle = int(group.iloc[failure_position]["Cycle_Index"])
    group["Failure_Cycle"] = failure_cycle

    if failure_position == 0:
        group["Exclusion_Reason"] = "NO_PRE_FAILURE_HISTORY"
        return group

    group["RUL"] = failure_cycle - group["Cycle_Index"]
    group["RUL"] = group["RUL"].clip(lower=0)

    # Any below-threshold point before the accepted failure is treated as a
    # temporary crossing and excluded from model training.
    temporary_crossing = (
        below_threshold
        & group["Cycle_Index"].lt(failure_cycle)
    )

    post_eol = group["Cycle_Index"].gt(failure_cycle)
    pre_eol_or_failure = group["Cycle_Index"].le(failure_cycle)

    group.loc[temporary_crossing, "Exclusion_Reason"] = (
        "TEMPORARY_THRESHOLD_CROSSING"
    )
    group.loc[post_eol, "Exclusion_Reason"] = "POST_EOL"
    group.loc[~valid_capacity, "Exclusion_Reason"] = "MISSING_CAPACITY"

    group["Model_Eligible"] = (
        pre_eol_or_failure
        & ~temporary_crossing
        & valid_capacity
    )

    return group


def apply_rul_logic(all_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the battery-level RUL calculation to the merged dataset."""
    processed_groups = [
        assign_rul_to_battery(group)
        for _, group in all_df.groupby("Battery_ID", sort=False)
    ]

    result = pd.concat(processed_groups, ignore_index=True)
    result["Failure_Cycle"] = result["Failure_Cycle"].astype("Int64")
    result["RUL"] = result["RUL"].round().astype("Int64")

    complete_model_features = result[MODEL_FEATURE_COLUMNS].notna().all(axis=1)

    missing_features_mask = result["Model_Eligible"] & ~complete_model_features

    result.loc[
        missing_features_mask,
        "Exclusion_Reason",
    ] = "MISSING_EXTRACTED_FEATURES"

    result.loc[
        missing_features_mask,
        "Model_Eligible",
    ] = False

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


# ---------------------------------------------------------------------------
# Reporting and database output
# ---------------------------------------------------------------------------
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


def upload_tables(
    engine: Engine,
    all_df: pd.DataFrame,
    model_df: pd.DataFrame,
) -> None:
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

        print("Step 4/5: Detecting confirmed failure and calculating RUL...")
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