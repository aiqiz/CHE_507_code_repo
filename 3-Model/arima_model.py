"""
Daily ARIMAX (+Fourier) for GPP with full metrics, residual diagnostics,
robust NaN/column checks, and a TXT report.

Creates:
- out_basic/gpp_daily_forecast.png
- out_basic/residual_acf.png
- out_basic/sarimax_summary_daily.txt
- out_basic/model_report_daily.txt
- out_basic/forecast_daily.csv
"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf

# -----------------------
# User settings (edit me)
# -----------------------
FLUXNET_CSV = Path("AMF_CA-TPD_FLUXNET_SUBSET_HH_2012-2017_4-6.csv")
GCC_CSV     = Path("turkeypointdbf_DB_1000_roistats.csv")   # optional; set to non-existing path to skip
TARGET      = "GPP_DT_VUT_REF"
# keep drivers small & interpretable
EXOG_VARS   = ["TA_F", "VPD_F", "PA_F", "P_F", "RH", "TS_F_MDS_1", "PPFD_IN", "CO2_F_MDS"]
USE_GCC     = False

DAILY_FREQ  = "D"
TRAIN_FRAC  = 0.7
FOURIER_K   = 1                  # works; try 3–5 if you want smoother seasonality
ARIMA_ORDER = (1, 0, 1)          # light ARMA errors

OUT_DIR = Path("feature"); OUT_DIR.mkdir(exist_ok=True)
SUMMARY_PATH = OUT_DIR / "sarimax_summary_daily.txt"
REPORT_PATH  = OUT_DIR / "model_report_daily.txt"
PLOT_PATH    = OUT_DIR / "gpp_daily_forecast.png"
ACF_PATH     = OUT_DIR / "residual_acf.png"
FORECAST_CSV = OUT_DIR / "forecast_daily.csv"

MISSING_FLAG = -9999
warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------
# Helpers
# -----------------------
def load_fluxnet_daily(csv_path: Path, target: str, exog_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path, na_values=[MISSING_FLAG], low_memory=False)
    df["TIMESTAMP_START"] = pd.to_datetime(df["TIMESTAMP_START"], format="%Y%m%d%H%M")
    df = df.set_index("TIMESTAMP_START").sort_index()

    if {"PPFD_IN", "RH", "VPD_F"}.issubset(df.columns):
        df["LUE_proxy"] = df["PPFD_IN"] * df["RH"] / (df["VPD_F"] + 1e-3)

    keep = [c for c in [target, *exog_cols] if c in df.columns]
    if target not in keep:
        raise ValueError(f"Target {target} not found in {csv_path.name}. Available: {list(df.columns)[:10]}...")

    # Aggregate to daily means
    daily = df[keep].resample(DAILY_FREQ).mean()

    # Drop Feb-29 to keep a clean annual cycle for Fourier
    daily = daily[~((daily.index.month == 2) & (daily.index.day == 29))]

    # Coerce numeric (guard against object dtypes) and handle short gaps
    daily = daily.apply(pd.to_numeric, errors="coerce")
    daily = daily.ffill(limit=3).bfill(limit=3).dropna()

    return daily


def load_gcc_daily(csv_path: Path) -> pd.Series:
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))
    gcc = pd.read_csv(csv_path, comment="#", na_values=["NA"], usecols=["date", "gcc"])
    gcc["date"] = pd.to_datetime(gcc["date"], format="%Y-%m-%d")
    gcc = (gcc.groupby("date")["gcc"].mean().sort_index()
                 .asfreq(DAILY_FREQ)
                 .interpolate(limit=3).ffill().bfill())
    gcc = gcc[~((gcc.index.month == 2) & (gcc.index.day == 29))]
    return gcc.rename("gcc")


def fourier_terms(index: pd.DatetimeIndex, period_days: float = 365.25, K: int = 1) -> pd.DataFrame:
    t = np.arange(len(index), dtype=float)
    cols = {}
    for k in range(1, K + 1):
        cols[f"sin_{k}"] = np.sin(2.0 * np.pi * k * t / period_days)
        cols[f"cos_{k}"] = np.cos(2.0 * np.pi * k * t / period_days)
    Xf = pd.DataFrame(cols, index=index)
    return Xf.apply(pd.to_numeric, errors="coerce")


def split_train_test(df: pd.DataFrame, train_frac: float):
    n = len(df)
    cut = max(1, int(n * train_frac))
    train = df.iloc[:cut]
    test = df.iloc[cut:]
    if len(test) == 0:
        raise ValueError("Test set is empty; reduce TRAIN_FRAC.")
    return train, test


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mse = float(np.mean(err**2))
    mape = float(np.mean(np.abs(err) / np.clip(np.abs(y_true), 1e-9, None))) * 100.0
    return {"MAE": mae, "RMSE": rmse, "MSE": mse, "MAPE_%": mape}


def save_forecast_plot(y_train, y_test, y_pred, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_train.index, y_train.values, label="Train (actual)", linewidth=1.1)
    ax.plot(y_test.index,  y_test.values,  label="Test (actual)",  linewidth=1.1)
    ax.plot(y_pred.index,  y_pred.values,  label="Forecast",       linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel("GPP (daily mean)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_residual_acf_plot(residuals: pd.Series, path: Path):
    fig, ax = plt.subplots(figsize=(8, 3))
    plot_acf(residuals.dropna(), lags=30, ax=ax)
    ax.set_title("Residual ACF (train)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_txt_report(
    report_path: Path,
    *, dataset_info: dict,
    model_info: dict,
    train_metrics: dict,
    test_metrics: dict,
    lb_results: pd.DataFrame,
    jb_tuple: tuple,
    resid_mean: float,
    resid_var: float,
    top_params: list[tuple[str, float]]
):
    jb_stat, jb_p, skew, kurt = jb_tuple
    lines = []

    lines.append("TIME SERIES MODEL REPORT — ARIMAX + Fourier (Daily)\n")
    lines.append("Dataset\n-------")
    for k, v in dataset_info.items():
        lines.append(f"{k}: {v}")
    lines.append("")

    lines.append("Model\n-----")
    for k, v in model_info.items():
        lines.append(f"{k}: {v}")
    lines.append("")

    lines.append("Performance\n-----------")
    lines.append(f"Train metrics: " + ", ".join([f"{k}={v:.4f}" for k, v in train_metrics.items()]))
    lines.append(f"Test  metrics: " + ", ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()]))
    lines.append("")

    lines.append("Residual Diagnostics (train)\n----------------------------")
    lines.append(f"Zero-mean check: mean(resid)={resid_mean:.6f}")
    lines.append(f"Variance (resid): var={resid_var:.6f}")
    lines.append("Ljung–Box p-values:")
    for _, row in lb_results.iterrows():
        lines.append(f"  lag={int(row['lag'])}: p-value={row['lb_pvalue']:.4g}")
    lines.append(f"Jarque–Bera: stat={jb_stat:.3f}, p={jb_p:.4g}, skew={skew:.3f}, kurt={kurt:.3f}")
    lines.append("")

    if top_params:
        lines.append("Top parameter magnitudes (abs value)\n------------------------------------")
        for name, val in top_params:
            lines.append(f"{name}: {val:.6f}")
        lines.append("")

    lines.append("Artifacts\n---------")
    lines.append(f"Forecast plot: {PLOT_PATH}")
    lines.append(f"Residual ACF : {ACF_PATH}")
    lines.append(f"Model summary: {SUMMARY_PATH}")
    lines.append(f"Forecast CSV : {FORECAST_CSV}")
    lines.append("")

    report_path.write_text("\n".join(lines))


def assert_numeric_finite(df: pd.DataFrame, name: str):
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        # Find a few offenders to help debug quickly
        arr = df.to_numpy(dtype=float)
        bad = np.argwhere(~np.isfinite(arr))
        r, c = bad[0]
        raise ValueError(
            f"Non-finite values in {name}. Example at row {df.index[r]}, col {df.columns[c]}."
        )


# -----------------------
# Main
# -----------------------
def main():
    # 1) Load daily fluxnet (target + minimal exog)
    daily = load_fluxnet_daily(FLUXNET_CSV, TARGET, EXOG_VARS)

    # 2) Optional: attach daily GCC (disabled by default)
    if USE_GCC:
        try:
            gcc = load_gcc_daily(GCC_CSV)
            daily = daily.join(gcc, how="left")
        except FileNotFoundError:
            print("Warning: GCC file not found → proceeding without GCC.")

    # 3) Build exogenous matrix: only use columns that exist
    base_cols = [c for c in EXOG_VARS if c in daily.columns]
    X = daily[base_cols].copy()

    # Fourier seasonality
    fourier = fourier_terms(daily.index, period_days=365.25, K=FOURIER_K)
    X = pd.concat([X, fourier], axis=1)

    # Coerce numeric everywhere and drop any remaining NaNs
    y = daily[TARGET].rename("y").astype(float)
    X = X.apply(pd.to_numeric, errors="coerce")
    data = pd.concat([y, X], axis=1).dropna()

    # Freeze exact train/test feature columns
    feature_cols = [c for c in data.columns if c != "y"]

    # 4) Split
    train, test = split_train_test(data, TRAIN_FRAC)

    y_train = train["y"].astype(float)
    y_test  = test["y"].astype(float)
    X_train = train[feature_cols].astype(float)
    X_test  = test[feature_cols].astype(float)

    # Validate shapes/columns and finiteness
    if list(X_train.columns) != feature_cols or list(X_test.columns) != feature_cols:
        raise ValueError("Train/Test feature columns mismatch.")
    assert_numeric_finite(X_train, "X_train")
    assert_numeric_finite(X_test, "X_test")
    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("NaNs in target after cleaning.")

    # 5) Fit ARIMAX (no seasonal box; Fourier handles annual)
    model = SARIMAX(
        endog=y_train,
        exog=X_train,
        order=ARIMA_ORDER,
        seasonal_order=(0,0,0,0),
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    res = model.fit(disp=False, maxiter=500)

    # Save statsmodels textual summary
    SUMMARY_PATH.write_text(res.summary().as_text())

    # 6) Train fit & metrics
    fit_train = pd.Series(res.fittedvalues, index=y_train.index, name="fitted").astype(float)
    train_m = metrics(y_train, fit_train)

    # 7) Forecast test horizon
    fc = res.get_forecast(steps=len(y_test), exog=X_test)
    y_pred = pd.Series(fc.predicted_mean, index=y_test.index, name="forecast").astype(float)

    # If NaNs appear, drop or interpolate them
    if y_pred.isna().any():
        nan_idx = y_pred.index[y_pred.isna()]
        print(f"⚠️  Forecast produced NaNs at {len(nan_idx)} timestamps — filling via interpolation.")
        # Option 1: interpolate (keeps continuity)
        y_pred = y_pred.interpolate(limit_direction="both")
        # Option 2 (if you prefer): y_pred = y_pred.dropna(); y_test = y_test.loc[y_pred.index]

    # Continue as normal
    test_m = metrics(y_test, y_pred)

    # Save forecast CSV
    pd.DataFrame({"actual": y_test, "forecast": y_pred}).to_csv(FORECAST_CSV, index=True)

    # 8) Residual diagnostics (train)
    resid_train = pd.Series(res.resid, index=y_train.index).dropna()
    lb = acorr_ljungbox(resid_train, lags=[7, 14, 21], return_df=True)
    lb = lb.reset_index().rename(columns={"index": "lag"})
    jb_tuple = jarque_bera(resid_train)
    resid_mean = float(resid_train.mean())
    resid_var  = float(resid_train.var(ddof=1))

    # 9) Plots
    save_forecast_plot(y_train, y_test, y_pred, PLOT_PATH,
                       title=f"Daily {TARGET} Forecast (ARIMAX + Fourier K={FOURIER_K})")
    save_residual_acf_plot(resid_train, ACF_PATH)

    # 10) Compose TXT report
    dataset_info = {
        "Target": TARGET,
        "Frequency": "Daily",
        "Total obs": len(data),
        "Train obs": len(train),
        "Test obs": len(test),
        "Train window": f"{y_train.index.min().date()} → {y_train.index.max().date()}",
        "Test window":  f"{y_test.index.min().date()} → {y_test.index.max().date()}",
    }
    params = res.params.sort_values(key=lambda s: s.abs(), ascending=False)
    top = [(name, val) for name, val in params.head(10).items()]

    model_info = {
        "Model": "ARIMAX (no seasonal box) + Fourier",
        "Order (p,d,q)": ARIMA_ORDER,
        "Fourier K": FOURIER_K,
        "Exogenous": ", ".join([c for c in base_cols] + [f"sin_{k}" for k in range(1, FOURIER_K+1)] + [f"cos_{k}" for k in range(1, FOURIER_K+1)]),
        "AIC": f"{res.aic:.2f}",
        "BIC": f"{res.bic:.2f}",
        "HQIC": f"{res.hqic:.2f}",
    }

    write_txt_report(
        REPORT_PATH,
        dataset_info=dataset_info,
        model_info=model_info,
        train_metrics=train_m,
        test_metrics=test_m,
        lb_results=lb,
        jb_tuple=jb_tuple,
        resid_mean=resid_mean,
        resid_var=resid_var,
        top_params=top,
    )

    print("\n=== Done ===")
    print(f"Summary → {SUMMARY_PATH}")
    print(f"Report  → {REPORT_PATH}")
    print(f"Plot    → {PLOT_PATH}")
    print(f"ACF     → {ACF_PATH}")
    print(f"Forecast CSV → {FORECAST_CSV}")


if __name__ == "__main__":
    main()
