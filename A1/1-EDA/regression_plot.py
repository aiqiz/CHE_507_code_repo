"""
Flexible regression plotting utility for Fluxnet variables.

Default behaviour plots air temperature (`TA_F`) against soil temperature
(`TS_F_MDS_1`), labels axes with descriptive names, and displays the plot with
the least-squares regression fit and its coefficient of determination (R²).

Custom variable pairs can be requested via CLI arguments.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_FILE = "AMF_CA-TPD_FLUXNET_SUBSET_HH_2012-2017_4-6.csv"
DEFAULT_NA_VALUES = [-9999, -9999.0]
DEFAULT_X = "TA_F"
DEFAULT_Y = "TS_F_MDS_1"

# Friendly labels for commonly used variables; fall back to the raw column name.
LABEL_MAP: Dict[str, str] = {
    "TA_F": "Air Temperature",
    "TS_F_MDS_1": "Soil Temperature",
    "GPP_NT_VUT_REF": "Gross Primary Production (NT)",
    "GPP_DT_VUT_REF": "Gross Primary Production (DT)",
    "CO2_F_MDS": "CO2 Concentration",
}


def load_data(data_path: Path, x_col: str, y_col: str) -> pd.DataFrame:
    """Load the CSV and return rows with valid values for the chosen variables."""
    df = pd.read_csv(
        data_path,
        na_values=DEFAULT_NA_VALUES,
        usecols=lambda col: col in {x_col, y_col},
    )
    return df.dropna(subset=[x_col, y_col])


def fit_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit y = slope * x + intercept and return slope, intercept, and R²."""
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return slope, intercept, r_squared


def make_regression_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str | None = None,
    y_label: str | None = None,
) -> None:
    """Render the regression plot directly."""
    x = df[x_col].values
    y = df[y_col].values

    slope, intercept, r_squared = fit_regression(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=6, alpha=0.3, color="tab:blue", edgecolors="none", label="Observations")
    ax.plot(
        x_line,
        y_line,
        color="tab:red",
        linewidth=2,
        label=f"Fit: {y_label or y_col} = {slope:.2f}·{x_label or x_col} + {intercept:.2f} (R² = {r_squared:.2f})",
    )

    ax.set_xlabel(x_label or LABEL_MAP.get(x_col, x_col))
    ax.set_ylabel(y_label or LABEL_MAP.get(y_col, y_col))
    ax.set_title(f"Regression: {LABEL_MAP.get(x_col, x_col)} vs {LABEL_MAP.get(y_col, y_col)}")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot regression between two Fluxnet variables.")
    parser.add_argument("--x", default=DEFAULT_X, help=f"Predictor column (default: {DEFAULT_X})")
    parser.add_argument("--y", default=DEFAULT_Y, help=f"Response column (default: {DEFAULT_Y})")
    parser.add_argument("--x-label", dest="x_label", help="Custom label for x-axis")
    parser.add_argument("--y-label", dest="y_label", help="Custom label for y-axis")
    parser.add_argument(
        "--data",
        default=DATA_FILE,
        help="Path to Fluxnet CSV (default: script directory / dataset name)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = script_dir.parent / data_path

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = load_data(data_path, args.x, args.y)
    if df.empty:
        raise ValueError(f"No overlapping records for {args.x} and {args.y}.")

    make_regression_plot(df, args.x, args.y, args.x_label, args.y_label)


if __name__ == "__main__":
    main()
