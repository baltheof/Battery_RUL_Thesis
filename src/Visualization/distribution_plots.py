"""
Exploratory distribution plots and battery capacity degradation plots.

The script creates three figures:
1. Diagnostic distributions from CYCLE_FEATURES_ALL.
2. Model-ready distributions from CYCLE_FEATURES.
3. Interactive State-of-Health degradation for all discharge batteries.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mplcursors
import pandas as pd
import seaborn as sns


# Add the src directory to the Python path.
SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db_connection import get_engine  # pylint: disable=wrong-import-position


FEATURE_COLUMNS = [
    "RUL",
    "Capacity_Ah",
    "Discharge_Time",
    "Temp_Mean",
    "Temp_Max",
    "Voltage_Min",
    "Voltage_Mean",
    "Current_Mean",
]

FEATURE_REQUIRED_COLUMNS = [
    "Battery_ID",
    "Cycle_Index",
    *FEATURE_COLUMNS,
]

CAPACITY_REQUIRED_COLUMNS = [
    "Battery_ID",
    "Cycle_Index",
    "Capacity_Ah",
    "Nominal_Capacity",
]

OUTPUT_DIR = Path(__file__).resolve().parent


def load_diagnostic_features(engine: Any) -> pd.DataFrame:
    """
    Load all diagnostic rows with a defined RUL from CYCLE_FEATURES_ALL.

    This table includes rows that may later be excluded from modeling,
    such as temporary threshold crossings and post-EOL cycles. It is used
    to show the data before the final modeling filter.
    """
    query = """
        SELECT
            Battery_ID,
            Cycle_Index,
            RUL,
            Capacity_Ah,
            Discharge_Time,
            Temp_Mean,
            Temp_Max,
            Voltage_Min,
            Voltage_Mean,
            Current_Mean
        FROM CYCLE_FEATURES_ALL
        WHERE RUL IS NOT NULL
        ORDER BY Battery_ID, Cycle_Index
    """

    print("Loading diagnostic features from CYCLE_FEATURES_ALL...")
    df = pd.read_sql(query, engine)

    if df.empty:
        raise ValueError(
            "CYCLE_FEATURES_ALL contains no rows with a defined RUL. "
            "Run feature_extractor.py first."
        )

    print(f"   Loaded {len(df)} diagnostic rows.")
    print(
        "   Batteries in diagnostic dataset: "
        f"{df['Battery_ID'].nunique()}"
    )

    return df


def load_model_features(engine: Any) -> pd.DataFrame:
    """Load the final model-ready rows from CYCLE_FEATURES."""
    query = """
        SELECT
            Battery_ID,
            Cycle_Index,
            RUL,
            Capacity_Ah,
            Discharge_Time,
            Temp_Mean,
            Temp_Max,
            Voltage_Min,
            Voltage_Mean,
            Current_Mean
        FROM CYCLE_FEATURES
        ORDER BY Battery_ID, Cycle_Index
    """

    print("\nLoading model features from CYCLE_FEATURES...")
    df = pd.read_sql(query, engine)

    if df.empty:
        raise ValueError(
            "CYCLE_FEATURES is empty. Run feature_extractor.py first."
        )

    print(f"   Loaded {len(df)} model rows.")
    print(
        "   Batteries in modeling dataset: "
        f"{df['Battery_ID'].nunique()}"
    )

    return df


def load_all_discharge_cycles(engine: Any) -> pd.DataFrame:
    """
    Load all discharge cycles for the State-of-Health graph.

    This query uses TEST_CYCLES rather than CYCLE_FEATURES, so batteries
    without an observed failure can still appear in the degradation plot.
    """
    query = """
        SELECT
            tc.Battery_ID,
            tc.Cycle_Index,
            tc.Capacity_Ah,
            b.Nominal_Capacity
        FROM TEST_CYCLES AS tc
        INNER JOIN BATTERIES AS b
            ON tc.Battery_ID = b.Battery_ID
        WHERE tc.Operation_Type = 'discharge'
        ORDER BY tc.Battery_ID, tc.Cycle_Index
    """

    print("\nLoading all discharge cycles from TEST_CYCLES...")
    df = pd.read_sql(query, engine)

    if df.empty:
        raise ValueError("No discharge cycles were found in TEST_CYCLES.")

    print(f"   Loaded {len(df)} discharge cycles.")
    print(
        "   Batteries with discharge data: "
        f"{df['Battery_ID'].nunique()}"
    )

    return df


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> pd.DataFrame:
    """Validate columns, numeric values, missing values, and duplicates."""
    missing_columns = sorted(set(required_columns).difference(df.columns))

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing_columns}"
        )

    clean_df = df.copy()

    numeric_columns = [
        column for column in required_columns if column != "Battery_ID"
    ]

    for column in numeric_columns:
        clean_df[column] = pd.to_numeric(
            clean_df[column],
            errors="coerce",
        )

    missing_values = clean_df[required_columns].isna().sum()

    print(f"\nMissing values in {dataset_name}:")

    if int(missing_values.sum()) == 0:
        print("   No missing values.")
    else:
        print(missing_values[missing_values > 0])

    duplicate_count = clean_df.duplicated(
        subset=["Battery_ID", "Cycle_Index"]
    ).sum()

    print(f"   Duplicate battery-cycle rows: {duplicate_count}")

    before = len(clean_df)
    clean_df = clean_df.dropna(subset=required_columns).copy()
    removed = before - len(clean_df)

    if removed > 0:
        print(
            f"   Removed {removed} rows with missing required values."
        )

    return clean_df


def print_rul_summary(df: pd.DataFrame, dataset_name: str) -> None:
    """Print a short RUL summary for a diagnostic or modeling dataset."""
    total_rows = len(df)
    zero_rows = int(df["RUL"].eq(0).sum())
    zero_percentage = (
        zero_rows / total_rows * 100 if total_rows > 0 else 0.0
    )

    print(f"\nRUL summary — {dataset_name}:")
    print(f"   Total rows: {total_rows}")
    print(f"   RUL = 0 rows: {zero_rows}")
    print(f"   RUL = 0 percentage: {zero_percentage:.2f}%")
    print(
        "   RUL range: "
        f"{df['RUL'].min():.0f} – {df['RUL'].max():.0f}"
    )


def plot_feature_distributions(
    df: pd.DataFrame,
    title: str,
    filename: str,
) -> None:
    """Plot distributions of RUL and the seven cycle features."""
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(20, 10),
        constrained_layout=True,
    )

    axes = axes.flatten()

    # KDE is avoided for variables with discrete or protocol-based values.
    columns_without_kde = {
        "RUL",
        "Current_Mean",
    }

    for index, column in enumerate(FEATURE_COLUMNS):
        use_kde = column not in columns_without_kde

        sns.histplot(
            data=df,
            x=column,
            kde=use_kde,
            bins=25,
            ax=axes[index],
            color="#4C72B0",
            edgecolor="white",
            alpha=0.7,
        )

        mean_value = df[column].mean()
        median_value = df[column].median()

        axes[index].axvline(
            mean_value,
            color="#C44E52",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_value:.2f}",
        )

        axes[index].axvline(
            median_value,
            color="#55A868",
            linestyle=":",
            linewidth=2,
            label=f"Median: {median_value:.2f}",
        )

        axes[index].set_title(
            f"Distribution of {column}",
            fontweight="bold",
            fontsize=12,
        )
        axes[index].set_xlabel(column, fontsize=10)
        axes[index].set_ylabel("Count", fontsize=10)
        axes[index].grid(True, linestyle="--", alpha=0.5)
        axes[index].legend(fontsize=8, loc="upper right")

    fig.suptitle(title, fontsize=18, fontweight="bold")

    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"\nSaved distribution plot: {output_path}")
    plt.show()


def calculate_state_of_health(
    capacity_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate State of Health using each battery's nominal capacity.

    SoH (%) = Capacity_Ah / Nominal_Capacity * 100
    """
    df = capacity_df.copy()

    valid_nominal = (
        df["Nominal_Capacity"].notna()
        & df["Nominal_Capacity"].gt(0)
    )

    invalid_count = int((~valid_nominal).sum())

    if invalid_count > 0:
        print(
            f"\nRemoved {invalid_count} rows with invalid nominal capacity."
        )

    df = df.loc[valid_nominal].copy()
    df["SoH_Percent"] = (
        df["Capacity_Ah"] / df["Nominal_Capacity"] * 100
    )

    return df


def plot_capacity_degradation_soh(
    capacity_df: pd.DataFrame,
) -> None:
    """Plot State of Health against cycle index for all batteries."""
    soh_df = calculate_state_of_health(capacity_df)

    if soh_df.empty:
        raise ValueError("No valid rows are available for the SoH plot.")

    fig, ax = plt.subplots(
        figsize=(14, 7),
        constrained_layout=True,
    )

    batteries = sorted(soh_df["Battery_ID"].unique())
    colors = sns.color_palette("husl", len(batteries))
    battery_lines = []

    for index, battery_id in enumerate(batteries):
        battery_df = (
            soh_df.loc[soh_df["Battery_ID"].eq(battery_id)]
            .sort_values("Cycle_Index")
        )

        line, = ax.plot(
            battery_df["Cycle_Index"],
            battery_df["SoH_Percent"],
            color=colors[index],
            linewidth=1.5,
            alpha=0.85,
            label=battery_id,
        )
        battery_lines.append(line)

    ax.axhline(
        70.0,
        color="#C44E52",
        linestyle="--",
        linewidth=2,
        label="Failure threshold (SoH = 70%)",
    )

    ax.set_title(
        "STATE OF HEALTH DEGRADATION PER BATTERY "
        "(Click Legend to Hide/Show)",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Cycle Index", fontsize=12)
    ax.set_ylabel("State of Health (%)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    legend = ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=9,
        ncol=2,
    )

    label_to_line = {
        line.get_label(): line for line in battery_lines
    }
    legend_artist_map = {}

    for legend_line, legend_text in zip(
        legend.get_lines(),
        legend.get_texts(),
    ):
        label = legend_text.get_text()
        original_line = label_to_line.get(label)

        # Ignore the threshold legend entry.
        if original_line is None:
            continue

        legend_line.set_picker(True)
        legend_line.set_pickradius(5)
        legend_text.set_picker(True)

        mapped_artists = (
            original_line,
            legend_line,
            legend_text,
        )
        legend_artist_map[legend_line] = mapped_artists
        legend_artist_map[legend_text] = mapped_artists

    def on_pick(event: Any) -> None:
        mapped_artists = legend_artist_map.get(event.artist)

        if mapped_artists is None:
            return

        original_line, legend_line, legend_text = mapped_artists
        visible = not original_line.get_visible()
        original_line.set_visible(visible)

        legend_alpha = 1.0 if visible else 0.2
        legend_line.set_alpha(legend_alpha)
        legend_text.set_alpha(legend_alpha)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)

    cursor = mplcursors.cursor(battery_lines, hover=True)

    @cursor.connect("add")
    def on_add(selection: Any) -> None:
        battery_name = selection.artist.get_label()
        cycle_index = int(round(selection.target[0]))
        soh_value = float(selection.target[1])
        line_color = selection.artist.get_color()

        selection.annotation.set_text(
            f"Battery: {battery_name}\n"
            f"Cycle: {cycle_index}\n"
            f"SoH: {soh_value:.1f}%"
        )

        annotation_box = selection.annotation.get_bbox_patch()
        annotation_box.set_facecolor(line_color)
        annotation_box.set_alpha(0.85)
        selection.annotation.set_color("white")

    output_path = OUTPUT_DIR / "capacity_degradation_soh.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"Saved SoH plot: {output_path}")
    plt.show()


def plot_distributions() -> None:
    """Create the three diagnostic, modeling, and SoH figures."""
    engine = get_engine()

    if engine is None:
        print("Database connection failed.")
        return

    try:
        # Figure 1: full diagnostic distributions.
        diagnostic_df = load_diagnostic_features(engine)
        diagnostic_df = validate_dataframe(
            diagnostic_df,
            FEATURE_REQUIRED_COLUMNS,
            "CYCLE_FEATURES_ALL",
        )
        print_rul_summary(diagnostic_df, "diagnostic dataset")
        plot_feature_distributions(
            diagnostic_df,
            title="DISTRIBUTION PLOTS — ALL DIAGNOSTIC CYCLES",
            filename="distribution_plots_all_cycles.png",
        )

        # Figure 2: final model-ready distributions.
        model_df = load_model_features(engine)
        model_df = validate_dataframe(
            model_df,
            FEATURE_REQUIRED_COLUMNS,
            "CYCLE_FEATURES",
        )
        print_rul_summary(model_df, "model-ready dataset")
        plot_feature_distributions(
            model_df,
            title="DISTRIBUTION PLOTS — MODEL-READY CYCLE FEATURES",
            filename="distribution_plots_model_ready.png",
        )

        # Figure 3: SoH degradation for all available discharge batteries.
        capacity_df = load_all_discharge_cycles(engine)
        capacity_df = validate_dataframe(
            capacity_df,
            CAPACITY_REQUIRED_COLUMNS,
            "TEST_CYCLES capacity data",
        )
        plot_capacity_degradation_soh(capacity_df)

    except (
        ValueError,
        KeyError,
        pd.errors.DatabaseError,
    ) as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    plot_distributions()