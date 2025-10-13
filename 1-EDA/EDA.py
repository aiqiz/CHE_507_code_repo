"""
Exploratory data analysis for the AMF_CA-TPD Fluxnet subset.

The script loads the dataset, derives core descriptive statistics, inspects
missing data patterns, summarises distributions, and performs a light-weight
time-series analysis on key flux variables.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_FILE = "AMF_CA-TPD_FLUXNET_SUBSET_HH_2012-2017_4-6.csv"

# Values flagged as missing in the Fluxnet product documentation.
DEFAULT_NA_VALUES = [-9999, -9999.0]

SELECTED_VARIABLES = ["TA_F", "VPD_F", "PA_F", "P_F", "RH", "TS_F_MDS_1", "PPFD_IN", "CO2_F_MDS", "GPP_DT_VUT_REF",]


def filter_valid_numeric(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    """Return numeric columns that contain at least one non-missing observation."""
    return [col for col in columns if df[col].notna().any()]


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Read the CSV file, parse timestamps, and harmonise missing values."""
    df = pd.read_csv(
        data_path,
        na_values=DEFAULT_NA_VALUES,
        low_memory=False,
    )

    for time_col in ("TIMESTAMP_START", "TIMESTAMP_END"):
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], format="%Y%m%d%H%M")

    return df


def classify_variable_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Classify columns into broad analytical types.

    Notes:
        * QC/flag columns are treated as categorical, even when numeric.
        * Integer-like columns with few unique values are also labelled
          categorical to ensure they receive level-count summaries.
    """
    type_map: Dict[str, str] = {}

    for column, dtype in df.dtypes.items():
        if pd.api.types.is_datetime64_any_dtype(dtype):
            type_map[column] = "datetime"
            continue

        if pd.api.types.is_bool_dtype(dtype):
            type_map[column] = "boolean"
            continue

        if pd.api.types.is_object_dtype(dtype):
            type_map[column] = "categorical"
            continue

        # For numeric columns, decide whether they behave like categorical flags.
        if isinstance(column, str) and (
            column.endswith("_QC")
            or "_QC_" in column
            or column.endswith("_FLAG")
            or "_FLAG" in column
            or column.endswith("_ID")
        ):
            type_map[column] = "categorical"
            continue

        unique_non_null = df[column].dropna().nunique()
        if unique_non_null and unique_non_null <= 12:
            type_map[column] = "categorical"
            continue

        type_map[column] = "numeric"

    return type_map


def basic_numeric_statistics(df: pd.DataFrame, numeric_columns: Iterable[str]) -> pd.DataFrame:
    """Compute mean, median, min, and max for each numeric variable."""
    numeric_columns = filter_valid_numeric(df, numeric_columns)
    if not numeric_columns:
        return pd.DataFrame()

    stats = df.loc[:, list(numeric_columns)].agg(["mean", "median", "min", "max"]).transpose()
    return stats


def distribution_summary(df: pd.DataFrame, numeric_columns: Iterable[str]) -> pd.DataFrame:
    """Summarise the distribution via key quantiles for each numeric variable."""
    numeric_columns = filter_valid_numeric(df, numeric_columns)
    if not numeric_columns:
        return pd.DataFrame()

    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    summary = df[numeric_columns].quantile(quantiles).transpose()
    summary.columns = [f"q{int(q * 100):02d}" for q in quantiles]
    return summary


def missing_values_report(df: pd.DataFrame, type_map: Dict[str, str]) -> pd.DataFrame:
    """Build a table with missing value counts, rates, and recommended handling."""
    total_rows = len(df)
    missing_counts = df.isna().sum()

    report = pd.DataFrame(
        {
            "missing_count": missing_counts,
            "missing_pct": (missing_counts / total_rows) * 100,
            "type": pd.Series(type_map),
        }
    )

    report["suggested_strategy"] = report.apply(
        lambda row: recommend_missing_strategy(row["type"], row["missing_pct"]),
        axis=1,
    )
    return report.sort_values(by="missing_pct", ascending=False)


def recommend_missing_strategy(variable_type: str, missing_pct: float) -> str:
    """Return a succinct strategy for handling missingness."""
    if missing_pct == 0:
        return "No action required"

    if variable_type == "numeric":
        if missing_pct <= 5:
            return "Interpolate or impute with median"
        if missing_pct <= 20:
            return "Evaluate temporal interpolation or model-based imputation"
        return "Consider feature exclusion or advanced imputation"

    if variable_type == "datetime":
        return "Investigate data gaps; align with neighbouring timestamps"

    if variable_type in {"categorical", "boolean"}:
        if missing_pct <= 5:
            return "Impute with mode or introduce 'Unknown'"
        return "Assess data source; consider 'Missing' category"

    return "Manual review recommended"


def categorical_levels(df: pd.DataFrame, categorical_columns: Iterable[str], top_n: int = 10) -> Dict[str, pd.Series]:
    """Compute value counts for each categorical column, limited to the top_n levels."""
    counts: Dict[str, pd.Series] = {}
    for column in categorical_columns:
        series = df[column]
        counts[column] = series.value_counts(dropna=False).head(top_n)
    return counts


def time_series_analysis(
    df: pd.DataFrame,
    numeric_columns: Iterable[str],
    timestamp_col: str = "TIMESTAMP_START",
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    """
    Deliver a basic time-series analysis:

    Returns:
        overview: Dict with time span, record counts, and inferred interval.
        daily_summary: Stats of daily mean profiles for selected metrics.
        monthly_trend: Monthly mean values for the same key metrics.
    """
    if timestamp_col not in df.columns:
        return {}, pd.DataFrame(), pd.DataFrame()

    ts_df = df.sort_values(timestamp_col).set_index(timestamp_col)
    ts_df = ts_df.loc[:, list(numeric_columns)]

    overview = {
        "start": ts_df.index.min(),
        "end": ts_df.index.max(),
        "observations": len(ts_df),
        "inferred_frequency": pd.infer_freq(ts_df.index[:100]) if len(ts_df) >= 3 else None,
    }

    key_metrics = [
        col
        for col in [
            "TA_F",  # air temperature
            "SW_IN_F",  # incoming shortwave radiation
            "VPD_F",  # vapour pressure deficit
            "NETRAD",  # net radiation
            "NEE_VUT_REF",  # net ecosystem exchange
            "GPP_NT_VUT_REF",  # gross primary production (night-time)
            "RECO_NT_VUT_REF",  # ecosystem respiration (night-time)
            "LE_F_MDS",  # latent heat flux
            "H_F_MDS",  # sensible heat flux
        ]
        if col in ts_df.columns
    ]

    if not key_metrics:
        # Fall back to the first five numeric columns if the canonical set is absent.
        key_metrics = list(ts_df.columns[:5])

    key_metrics = filter_valid_numeric(ts_df, key_metrics)
    if not key_metrics:
        return overview, pd.DataFrame(), pd.DataFrame()

    daily_means = ts_df[key_metrics].resample("D").mean()
    monthly_means = ts_df[key_metrics].resample("ME").mean()

    daily_summary = (
        daily_means.describe(percentiles=[0.25, 0.5, 0.75])
        .transpose()
        .rename(
            columns={
                "25%": "q25",
                "50%": "median",
                "75%": "q75",
            }
        )
    )
    monthly_trend = monthly_means

    return overview, daily_summary, monthly_trend


def describe_selected_variables(df: pd.DataFrame, variables: Iterable[str]) -> pd.DataFrame:
    """Produce descriptive statistics (count, mean, std, min, quartiles, max) for chosen variables."""
    available = [col for col in variables if col in df.columns]
    if not available:
        return pd.DataFrame()
    valid_columns = filter_valid_numeric(df, available)
    if not valid_columns:
        return pd.DataFrame()
    return df[valid_columns].describe().transpose()


def ensure_output_dir(path: Path) -> Path:
    """Create output directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_time_series_and_monthly_boxplots(
    df: pd.DataFrame,
    variables: Iterable[str],
    timestamp_col: str = "TIMESTAMP_START",
    output_dir: Path | None = None,
) -> List[Path]:
    """Generate combined time series and monthly box plots for target variables."""
    if timestamp_col not in df.columns:
        return []

    available_vars = [var for var in variables if var in df.columns]
    if not available_vars:
        return []

    if output_dir is None:
        output_dir = Path("figures")
    output_dir = ensure_output_dir(output_dir)

    ts_df = df[[timestamp_col] + available_vars].copy()
    ts_df = ts_df.dropna(how="all", subset=available_vars)
    ts_df = ts_df.sort_values(timestamp_col).set_index(timestamp_col)

    if ts_df.empty:
        return []

    record_start = ts_df.index.min()
    record_end = ts_df.index.max()

    daily_means = ts_df.resample("D").mean()
    monthly_means = ts_df.resample("ME").mean()

    saved_paths: List[Path] = []

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for var in available_vars:
        if var not in daily_means.columns:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"{var} – Full Record ({record_start.date()} to {record_end.date()})"
        )

        # Time series subplot
        axes[0].plot(daily_means.index, daily_means[var], color="tab:blue", linewidth=0.8)
        axes[0].set_title("Daily Mean Time Series")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel(var)
        axes[0].grid(alpha=0.3)

        # Monthly average box plot
        monthly_series = monthly_means[var].dropna()
        if not monthly_series.empty:
            boxplot_data = [
                monthly_series[monthly_series.index.month == month].values for month in range(1, 13)
            ]
            # Handle months with no data by filtering empty arrays to avoid matplotlib warnings.
            filtered_data = []
            filtered_labels: List[str] = []
            for data, label in zip(boxplot_data, month_labels):
                if len(data) > 0:
                    filtered_data.append(data)
                    filtered_labels.append(label)
            if filtered_data:
                axes[1].boxplot(filtered_data, tick_labels=filtered_labels, patch_artist=True)
                axes[1].set_title("Monthly Average Distribution")
                axes[1].set_xlabel("Month")
                axes[1].set_ylabel(var)
                axes[1].grid(alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, "No monthly data available", ha="center", va="center")
                axes[1].set_axis_off()
        else:
            axes[1].text(0.5, 0.5, "No monthly data available", ha="center", va="center")
            axes[1].set_axis_off()

        fig.autofmt_xdate()
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        file_path = output_dir / f"{var}_timeseries_boxplot.png"
        fig.savefig(file_path, dpi=150)
        plt.close(fig)

        saved_paths.append(file_path)

    return saved_paths


def print_section(title: str) -> None:
    """Utility for consistent section headers."""
    print("\n" + title)
    print("-" * len(title))


def main() -> None:
    data_path = Path(__file__).with_name(DATA_FILE)

    if not data_path.exists():
        raise FileNotFoundError(f"Expected dataset at {data_path}")

    df = load_dataset(data_path)
    type_map = classify_variable_types(df)

    numeric_columns = [col for col, typ in type_map.items() if typ == "numeric"]
    categorical_columns = [col for col, typ in type_map.items() if typ == "categorical"]
    numeric_columns = filter_valid_numeric(df, numeric_columns)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)

    print_section("Variable Types")
    for column, vartype in type_map.items():
        print(f"{column}: {vartype}")

    print_section("Basic Numeric Statistics (mean, median, min, max)")
    numeric_stats = basic_numeric_statistics(df, numeric_columns)
    if numeric_stats.empty:
        print("No numeric columns detected.")
    else:
        print(numeric_stats.round(3))

    print_section("Distribution Summary (quantiles)")
    dist_summary = distribution_summary(df, numeric_columns)
    if dist_summary.empty:
        print("No numeric columns detected.")
    else:
        print(dist_summary.round(3))

    print_section("Key Variable Statistics")
    selected_stats = describe_selected_variables(df, SELECTED_VARIABLES)
    if selected_stats.empty:
        print("Requested variables not found or contain only missing data.")
    else:
        print(selected_stats.round(3))

    print_section("Missing Values Report")
    missing_report = missing_values_report(df, type_map)
    print(missing_report.round({"missing_pct": 2}))

    print_section("Categorical Levels (top 10)")
    if not categorical_columns:
        print("No categorical columns detected.")
    else:
        cat_counts = categorical_levels(df, categorical_columns)
        for column, counts in cat_counts.items():
            print(f"\n{column}")
            print(counts)

    print_section("Time-Series Analysis")
    overview, daily_summary, monthly_trend = time_series_analysis(df, numeric_columns)
    if not overview:
        print("No timestamp column found; skipped time-series analysis.")
    else:
        print("Overview:")
        for key, value in overview.items():
            print(f"  {key}: {value}")

        print("\nDaily Mean Profile Summary")
        print(daily_summary.round(3))

        print("\nMonthly Mean Trend (first 12 rows)")
        print(monthly_trend.head(12).round(3))

    print_section("Generated Figures")
    figure_paths = plot_time_series_and_monthly_boxplots(df, SELECTED_VARIABLES)
    if not figure_paths:
        print("No figures created (missing variables or timestamps).")
    else:
        for path in figure_paths:
            print(path)


if __name__ == "__main__":
    main()
