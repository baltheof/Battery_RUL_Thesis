from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mplcursors
import pandas as pd
import seaborn as sns

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from db_connection import get_engine


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

REQUIRED_COLUMNS = [
    "Battery_ID",
    "Cycle_Index",
    *FEATURE_COLUMNS,
]

OUTPUT_DIR = Path(__file__).resolve().parent


def load_cycle_features() -> pd.DataFrame:
    """
    Load the required cycle-level features from SQL Server.

    Returns
    -------
    pd.DataFrame
        Cycle-level battery features.

    Raises
    ------
    ConnectionError
        If the database connection cannot be created.
    ValueError
        If required columns are missing or no rows are returned.
    """
    engine = get_engine()

    if engine is None:
        raise ConnectionError(
            "Could not connect to the database."
        )

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
    """

    print("Loading CYCLE_FEATURES from database...")
    df = pd.read_sql(query, engine)

    if df.empty:
        raise ValueError(
            "CYCLE_FEATURES returned no rows."
        )

    missing_columns = sorted(
        set(REQUIRED_COLUMNS).difference(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    numeric_columns = [
        "Cycle_Index",
        *FEATURE_COLUMNS,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    print(f"Loaded rows: {len(df)}")
    print(
        "Distinct batteries:",
        df["Battery_ID"].nunique(),
    )

    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data-quality checks and remove rows that
    cannot be plotted safely.

    Parameters
    ----------
    df:
        Original cycle-level dataset.

    Returns
    -------
    pd.DataFrame
        Dataset without rows missing required values.
    """
    missing_values = df[REQUIRED_COLUMNS].isna().sum()

    print("\nMissing values:")
    print(missing_values[missing_values > 0])

    duplicate_count = df.duplicated(
        subset=["Battery_ID", "Cycle_Index"]
    ).sum()

    print(
        "\nDuplicate Battery_ID/Cycle_Index rows:",
        duplicate_count,
    )

    clean_df = df.dropna(
        subset=REQUIRED_COLUMNS
    ).copy()

    removed_rows = len(df) - len(clean_df)

    if removed_rows > 0:
        print(
            f"Removed {removed_rows} rows "
            "with missing required values."
        )

    return clean_df


def build_modeling_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep all cycles up to and including the first RUL = 0
    cycle of each battery.

    This removes repeated post-End-of-Life zero targets from
    the modeling view without deleting them from the database.

    Parameters
    ----------
    df:
        Full CYCLE_FEATURES dataset.

    Returns
    -------
    pd.DataFrame
        Pre-EOL dataset containing at most one zero-RUL cycle
        per battery.
    """
    first_zero_cycles = (
        df.loc[df["RUL"].eq(0)]
        .groupby("Battery_ID")["Cycle_Index"]
        .min()
        .rename("First_Zero_Cycle")
    )

    modeling_df = df.join(
        first_zero_cycles,
        on="Battery_ID",
    )

    keep_mask = (
        modeling_df["First_Zero_Cycle"].isna()
        | modeling_df["Cycle_Index"].le(
            modeling_df["First_Zero_Cycle"]
        )
    )

    modeling_df = (
        modeling_df.loc[keep_mask]
        .drop(columns="First_Zero_Cycle")
        .copy()
    )

    original_zero_count = int(
        df["RUL"].eq(0).sum()
    )

    modeling_zero_count = int(
        modeling_df["RUL"].eq(0).sum()
    )

    print("\nRUL filtering summary:")
    print(f"Original rows: {len(df)}")
    print(f"Original RUL = 0 rows: {original_zero_count}")
    print(f"Modeling rows: {len(modeling_df)}")
    print(f"Modeling RUL = 0 rows: {modeling_zero_count}")
    print(
        "Removed post-EOL rows:",
        len(df) - len(modeling_df),
    )

    return modeling_df


def add_state_of_health(
    df: pd.DataFrame,
    initial_window: int = 5,
) -> pd.DataFrame:
    """
    Calculate State of Health using the median capacity of the
    first valid cycles of each battery.

    SoH (%) = current capacity / initial capacity * 100

    Parameters
    ----------
    df:
        Cycle-level dataset.
    initial_window:
        Number of early cycles used to estimate the initial
        capacity.

    Returns
    -------
    pd.DataFrame
        Dataset with Initial_Capacity_Ah and SoH_Percent.
    """
    ordered_df = df.sort_values(
        ["Battery_ID", "Cycle_Index"]
    ).copy()

    positive_capacity = ordered_df.loc[
        ordered_df["Capacity_Ah"] > 0
    ]

    first_cycles = (
        positive_capacity
        .groupby(
            "Battery_ID",
            group_keys=False,
        )
        .head(initial_window)
    )

    initial_capacity = (
        first_cycles
        .groupby("Battery_ID")["Capacity_Ah"]
        .median()
        .rename("Initial_Capacity_Ah")
    )

    soh_df = ordered_df.join(
        initial_capacity,
        on="Battery_ID",
    )

    soh_df["SoH_Percent"] = (
        soh_df["Capacity_Ah"]
        / soh_df["Initial_Capacity_Ah"]
        * 100
    )

    missing_initial = (
        soh_df["Initial_Capacity_Ah"]
        .isna()
        .sum()
    )

    if missing_initial > 0:
        print(
            "Warning: rows without an initial capacity "
            f"estimate: {missing_initial}"
        )

    return soh_df


def plot_feature_distributions(
    df: pd.DataFrame,
    title: str,
    filename: str,
) -> None:
    """
    Plot histograms for RUL and the seven input features.
    """
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(20, 10),
        constrained_layout=True,
    )

    axes = axes.flatten()

    no_kde_columns = {
        "RUL",
        "Current_Mean",
    }

    for index, column in enumerate(FEATURE_COLUMNS):
        use_kde = column not in no_kde_columns

        sns.histplot(
            data=df,
            x=column,
            kde=use_kde,
            ax=axes[index],
            bins=25,
            edgecolor="white",
            alpha=0.7,
        )

        mean_value = df[column].mean()
        median_value = df[column].median()

        axes[index].axvline(
            mean_value,
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_value:.2f}",
        )

        axes[index].axvline(
            median_value,
            linestyle=":",
            linewidth=2,
            label=f"Median: {median_value:.2f}",
        )

        axes[index].set_title(
            f"Distribution of {column}",
            fontweight="bold",
            fontsize=12,
        )

        axes[index].set_xlabel(
            column,
            fontsize=10,
        )

        axes[index].set_ylabel(
            "Count",
            fontsize=10,
        )

        axes[index].grid(
            True,
            linestyle="--",
            alpha=0.5,
        )

        axes[index].legend(
            fontsize=8,
            loc="upper right",
        )

    fig.suptitle(
        title,
        fontsize=18,
        fontweight="bold",
    )

    output_path = OUTPUT_DIR / filename

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved: {output_path}")
    plt.show()


def plot_capacity_degradation_soh(
    df: pd.DataFrame,
) -> None:
    """
    Plot normalized capacity degradation as State of Health.

    A common 70% threshold is valid after normalizing each
    battery by its own estimated initial capacity.
    """
    soh_df = add_state_of_health(df)

    fig, ax = plt.subplots(
        figsize=(14, 7),
        constrained_layout=True,
    )

    batteries = sorted(
        soh_df["Battery_ID"].unique()
    )

    colors = sns.color_palette(
        "husl",
        len(batteries),
    )

    battery_lines = []

    for index, battery_id in enumerate(batteries):
        battery_df = (
            soh_df.loc[
                soh_df["Battery_ID"].eq(battery_id)
            ]
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
        linestyle="--",
        linewidth=2,
        label="Failure threshold (SoH = 70%)",
    )

    ax.set_title(
        "CAPACITY DEGRADATION PER BATTERY — SoH",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Cycle Index",
        fontsize=12,
    )

    ax.set_ylabel(
        "State of Health (%)",
        fontsize=12,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.5,
    )

    legend = ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=9,
        ncol=2,
    )

    label_to_line = {
        line.get_label(): line
        for line in battery_lines
    }

    legend_artist_map = {}

    for legend_line, legend_text in zip(
        legend.get_lines(),
        legend.get_texts(),
    ):
        label = legend_text.get_text()
        original_line = label_to_line.get(label)

        if original_line is None:
            continue

        legend_line.set_picker(True)
        legend_line.set_pickradius(5)
        legend_text.set_picker(True)

        legend_artist_map[legend_line] = (
            original_line,
            legend_line,
            legend_text,
        )

        legend_artist_map[legend_text] = (
            original_line,
            legend_line,
            legend_text,
        )

    def on_pick(event) -> None:
        mapped = legend_artist_map.get(
            event.artist
        )

        if mapped is None:
            return

        original_line, legend_line, legend_text = mapped
        visible = not original_line.get_visible()

        original_line.set_visible(visible)

        alpha = 1.0 if visible else 0.2
        legend_line.set_alpha(alpha)
        legend_text.set_alpha(alpha)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect(
        "pick_event",
        on_pick,
    )

    cursor = mplcursors.cursor(
        battery_lines,
        hover=True,
    )

    @cursor.connect("add")
    def on_add(selection) -> None:
        battery_name = selection.artist.get_label()
        cycle_index = int(round(selection.target[0]))
        soh_value = float(selection.target[1])
        line_color = selection.artist.get_color()

        selection.annotation.set_text(
            f"Battery: {battery_name}\n"
            f"Cycle: {cycle_index}\n"
            f"SoH: {soh_value:.1f}%"
        )

        annotation_box = (
            selection.annotation
            .get_bbox_patch()
        )

        annotation_box.set_facecolor(
            line_color
        )

        annotation_box.set_alpha(0.85)
        selection.annotation.set_color("white")

    output_path = (
        OUTPUT_DIR
        / "capacity_degradation_soh.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved: {output_path}")
    plt.show()


def plot_distributions() -> None:
    """
    Create diagnostic and modeling distribution plots and the
    normalized capacity-degradation plot.
    """
    try:
        full_df = load_cycle_features()
        full_df = validate_data(full_df)

        modeling_df = build_modeling_view(full_df)

        # Diagnostic view: includes every database row.
        plot_feature_distributions(
            full_df,
            title=(
                "DISTRIBUTION PLOTS — ALL AVAILABLE CYCLES"
            ),
            filename="distribution_plots_all_cycles.png",
        )

        # Modeling view: keeps only cycles up to first EOL.
        plot_feature_distributions(
            modeling_df,
            title=(
                "DISTRIBUTION PLOTS — PRE-EOL MODELING VIEW"
            ),
            filename="distribution_plots_model_view.png",
        )

        # Diagnostic SoH plot for every available cycle.
        plot_capacity_degradation_soh(full_df)

    except (
        ConnectionError,
        ValueError,
        pd.errors.DatabaseError,
    ) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    plot_distributions()