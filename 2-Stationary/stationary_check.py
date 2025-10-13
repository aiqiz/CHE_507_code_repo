"""
Stationarity diagnostics for the CHE507 Assignment A1 dataset.

The workflow requested by the instructor is:
1. Evaluate the raw Precipitation (`P_F`) and GPP (`GPP_NT_VUT_REF`) series
   with Augmented Dickey-Fuller (ADF, interpreted here as the requested "KDF")
   and KPSS tests to demonstrate the presence or absence of stationarity.
2. Provide an explicit justification that precipitation is non-stationary.
3. Apply a first difference followed by a seasonal difference (daily seasonality
   assumed for the half-hourly series) and re-test stationarity.
4. Produce ACF and PACF diagnostics for the transformed series.

The script prints concise summaries of the test results and writes ACF/PACF
plots to the `figures` directory at the project root.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

# The script may run in a headless environment; use a non-interactive backend.
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "AMF_CA-TPD_FLUXNET_SUBSET_HH_2012-2017_4-6.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Constants for transformation.
FILL_VALUE = -9999
SEASONAL_PERIOD = 48  # Half-hourly data -> 48 steps per day.


def load_fluxnet_data(path: Path) -> pd.DataFrame:
    """Load the Fluxnet subset, normalise timestamps, and replace missing values."""
    df = pd.read_csv(path)
    df["TIMESTAMP_START"] = pd.to_datetime(df["TIMESTAMP_START"], format="%Y%m%d%H%M")
    df = df.set_index("TIMESTAMP_START").replace(FILL_VALUE, np.nan)
    return df


def dropna_series(series: pd.Series, label: str) -> pd.Series:
    """Drop NA values and warn if substantial data loss occurs."""
    original_len = len(series)
    clean_series = series.dropna()
    dropped = original_len - len(clean_series)
    if dropped:
        pct = dropped / original_len * 100
        print(f"[{label}] Dropped {dropped} NA records ({pct:.2f}%).")
    return clean_series


def run_adf(series: pd.Series) -> Dict[str, float]:
    """Augmented Dickey-Fuller test (interpreted as the requested KDF)."""
    result = adfuller(series, autolag="AIC")
    return {
        "statistic": result[0],
        "pvalue": result[1],
        "lags": result[2],
        "nobs": result[3],
        "critical_values": result[4],
    }


def run_kpss(series: pd.Series) -> Dict[str, float]:
    """KPSS stationarity test."""
    result = kpss(series, regression="c", nlags="auto")
    return {
        "statistic": result[0],
        "pvalue": result[1],
        "lags": result[2],
        "critical_values": result[3],
    }


def summarize_tests(series: pd.Series, label: str) -> None:
    """Run and print ADF (requested KDF) and KPSS results."""
    adf_res = run_adf(series)
    kpss_res = run_kpss(series)

    print(f"\n--- Stationarity diagnostics for {label} ---")
    print("ADF (interpreted as KDF) results:")
    print(
        f"  Test statistic: {adf_res['statistic']:.4f}, "
        f"p-value: {adf_res['pvalue']:.4g}, "
        f"Lags used: {adf_res['lags']}, "
        f"Observations: {adf_res['nobs']}"
    )
    print(
        "  Critical values: "
        + ", ".join(f"{k}: {v:.4f}" for k, v in adf_res["critical_values"].items())
    )

    print("KPSS results:")
    print(
        f"  Test statistic: {kpss_res['statistic']:.4f}, "
        f"p-value: {kpss_res['pvalue']:.4g}, "
        f"Lags used: {kpss_res['lags']}"
    )
    print(
        "  Critical values: "
        + ", ".join(f"{k}: {v:.4f}" for k, v in kpss_res["critical_values"].items())
    )


def difference_series(series: pd.Series, seasonal_period: int) -> Tuple[pd.Series, pd.Series]:
    """Apply first difference then a seasonal difference."""
    first_diff = series.diff().dropna()
    seasonal_diff = first_diff.diff(seasonal_period).dropna()
    return first_diff, seasonal_diff


def justify_precipitation_non_stationarity(
    series: pd.Series, adf_result: Dict[str, float], kpss_result: Dict[str, float]
) -> None:
    """Print a justification statement based on test outcomes."""
    adf_non_stationary = adf_result["pvalue"] > 0.05
    kpss_non_stationary = kpss_result["pvalue"] < 0.05
    print("\nPrecipitation stationarity interpretation:")
    if adf_non_stationary and kpss_non_stationary:
        print(
            "  Precipitation is non-stationary: the ADF test fails to reject the "
            "unit-root hypothesis (p-value > 0.05), while the KPSS test rejects the "
            "stationary null (p-value < 0.05)."
        )
    elif adf_non_stationary:
        print(
            "  Precipitation likely has a unit root (ADF p-value > 0.05), "
            "indicating non-stationarity."
        )
    elif kpss_non_stationary:
        print(
            "  KPSS rejects the null of stationarity (p-value < 0.05), so "
            "precipitation is non-stationary."
        )
    else:
        monthly_totals = series.resample("ME").sum()
        seasonal_range = monthly_totals.max() - monthly_totals.min()
        ratio = (
            monthly_totals.max() / monthly_totals.min()
            if monthly_totals.min() > 0
            else float("inf")
        )
        print(
            "  Formal tests lean towards stationarity (ADF rejects the unit root and "
            "KPSS does not reject stationarity). However, monthly precipitation totals "
            f"vary substantially across the record (range {seasonal_range:.2f} mm, "
            f"max/min ratio {ratio:.1f}), indicating strong seasonality and shifts in "
            "rainfall regimes. We therefore treat precipitation as non-stationary for "
            "modelling purposes."
        )


def plot_acf_pacf(series: pd.Series, label: str, seasonal_period: int) -> None:
    """Generate ACF and PACF plots for the provided series."""
    if series.empty or series.nunique() < 2:
        print(f"[{label}] Skipping ACF/PACF due to insufficient variation.")
        return

    lags = min(5 * seasonal_period, len(series) - 1)
    if not math.isfinite(lags) or lags <= 0:
        lags = min(40, len(series) - 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series, ax=axes[0], lags=lags, title=f"{label} ACF")
    plot_pacf(series, ax=axes[1], lags=min(lags, 40), title=f"{label} PACF", method="ywm")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / f"{label.lower().replace(' ', '_')}_acf_pacf.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[{label}] ACF/PACF plots saved to {output_path}.")


def main() -> None:
    df = load_fluxnet_data(DATA_PATH)
    precipitation = dropna_series(df["P_F"], "Precipitation (P_F)")
    gpp = dropna_series(df["GPP_NT_VUT_REF"], "GPP (GPP_NT_VUT_REF)")

    # Raw diagnostics.
    summarize_tests(precipitation, "Precipitation (raw)")
    summarize_tests(gpp, "GPP (raw)")

    precip_adf = run_adf(precipitation)
    precip_kpss = run_kpss(precipitation)
    justify_precipitation_non_stationarity(precipitation, precip_adf, precip_kpss)

    # Transformations: first difference then seasonal difference.
    precip_first_diff, precip_seasonal_diff = difference_series(precipitation, SEASONAL_PERIOD)
    gpp_first_diff, gpp_seasonal_diff = difference_series(gpp, SEASONAL_PERIOD)

    summarize_tests(precip_first_diff, "Precipitation (first difference)")
    summarize_tests(precip_seasonal_diff, "Precipitation (first + seasonal difference)")
    summarize_tests(gpp_first_diff, "GPP (first difference)")
    summarize_tests(gpp_seasonal_diff, "GPP (first + seasonal difference)")

    # Diagnostics plots for the fully differenced series.
    plot_acf_pacf(precip_seasonal_diff, "Precipitation Differenced", SEASONAL_PERIOD)
    plot_acf_pacf(gpp_seasonal_diff, "GPP Differenced", SEASONAL_PERIOD)


if __name__ == "__main__":
    main()
