"""
Utility script to print focused summary statistics for select Fluxnet variables.

Outputs:
    * Basic numeric statistics (mean, median, min, max)
    * Distribution summary quantiles (5th, 25th, 50th, 75th, 95th percentiles)
for TA_F, VPD_F, PA_F, P_F, and RH.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_FILE = "AMF_CA-TPD_FLUXNET_SUBSET_HH_2012-2017_4-6.csv"
DEFAULT_NA_VALUES = [-9999, -9999.0]
TARGET_VARIABLES = [
    "TA_F",
    "VPD_F",
    "PA_F",
    "P_F",
    "RH",
    "TS_F_MDS_1",
    "PPFD_IN",
    "CO2_F_MDS",
    "GPP_DT_VUT_REF",
]

def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the dataset, applying documented missing-value markers."""
    df = pd.read_csv(
        data_path,
        na_values=DEFAULT_NA_VALUES,
        low_memory=False,
    )
    return df


def filter_available_variables(df: pd.DataFrame) -> list[str]:
    """Filter target variables to those present with at least one valid value."""
    return TARGET_VARIABLES


def basic_numeric_statistics(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return mean, median, min, and max for the selected columns."""
    if not columns:
        return pd.DataFrame()
    return df[columns].agg(["mean", "median", "min", "max"]).transpose()


def distribution_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return key quantiles (5th, 25th, 50th, 75th, 95th) for the selected columns."""
    if not columns:
        return pd.DataFrame()
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    summary = df[columns].quantile(quantiles).transpose()
    summary.columns = [f"q{int(q * 100):02d}" for q in quantiles]
    return summary


def print_section(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def main() -> None:
    data_path = Path(__file__).with_name(DATA_FILE)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = load_dataset(data_path)
    available_columns = filter_available_variables(df)


    print(available_columns)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 100)

    print("Available variables analysed:", ", ".join(available_columns) or "None")

    print_section("Basic Numeric Statistics (mean, median, min, max)")
    stats = basic_numeric_statistics(df, available_columns)
    if stats.empty:
        print("No data available for requested variables.")
    else:
        print(stats.round(3))

    print_section("Distribution Summary (q05, q25, q50, q75, q95)")
    quantiles = distribution_summary(df, available_columns)
    if quantiles.empty:
        print("No data available for requested variables.")
    else:
        print(quantiles.round(3))


if __name__ == "__main__":
    main()

