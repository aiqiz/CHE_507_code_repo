"""
Daily feed-forward ANN regression for GPP with cross-validation,
training/validation error curves, residual diagnostics, and a TXT report.

Outputs:
- full/ann_gpp_forecast.png          (actual vs. predicted time series)
- full/ann_training_curve.png        (train/val RMSE per epoch)
- full/ann_residual_acf.png          (ACF of hold-out residuals)
- full/ann_model_report_daily.txt    (comprehensive text report)
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera

# -----------------------
# User settings (edit me)
# -----------------------
FLUXNET_CSV = Path("AMF_CA-TPD_FLUXNET_SUBSET_HH_2012-2017_4-6.csv")
GCC_CSV = Path("turkeypointdbf_DB_1000_roistats.csv")  # optional GCC vegetation index
TARGET = "GPP_DT_VUT_REF"
EXOG_VARS = [
    "TA_F",
    "VPD_F",
    "PA_F",
    "P_F",
    "RH",
    "PPFD_IN",
    "LUE_proxy",
]
USE_GCC = False  # flip to True if GCC input is available and desired

DAILY_FREQ = "D"
TRAIN_FRAC = 0.7
USE_FOURIER = False
FOURIER_K = 1

# ANN hyperparameters
HIDDEN_LAYERS = (32, 16)
LEARNING_RATE = 1e-3
L2_ALPHA = 1e-4
MAX_EPOCHS = 400
PATIENCE = 25
VAL_FRAC_WITHIN_TRAIN = 0.2  # fraction of the training window reserved for validation curves
RANDOM_STATE = 123

OUT_DIR = Path("ANN-feature")
OUT_DIR.mkdir(exist_ok=True)
REPORT_PATH = OUT_DIR / "ann_model_report_daily.txt"
FORECAST_PLOT_PATH = OUT_DIR / "ann_gpp_forecast.png"
TRAIN_CURVE_PATH = OUT_DIR / "ann_training_curve.png"
RESIDUAL_ACF_PATH = OUT_DIR / "ann_residual_acf.png"

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
        raise ValueError(
            f"Target {target} not found in {csv_path.name}. "
            f"Available columns include: {list(df.columns)[:10]}"
        )

    daily = df[keep].resample(DAILY_FREQ).mean()
    daily = daily[~((daily.index.month == 2) & (daily.index.day == 29))]
    daily = daily.apply(pd.to_numeric, errors="coerce")
    daily = daily.ffill(limit=3).bfill(limit=3).dropna()
    return daily


def load_gcc_daily(csv_path: Path) -> pd.Series:
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))
    gcc = pd.read_csv(csv_path, comment="#", na_values=["NA"], usecols=["date", "gcc"])
    gcc["date"] = pd.to_datetime(gcc["date"], format="%Y-%m-%d")
    gcc = (
        gcc.groupby("date")["gcc"]
        .mean()
        .sort_index()
        .asfreq(DAILY_FREQ)
        .interpolate(limit=3)
        .ffill()
        .bfill()
    )
    gcc = gcc[~((gcc.index.month == 2) & (gcc.index.day == 29))]
    return gcc.rename("gcc")


def fourier_terms(index: pd.DatetimeIndex, period_days: float = 365.25, K: int = 1) -> pd.DataFrame:
    t = np.arange(len(index), dtype=float)
    cols: dict[str, np.ndarray] = {}
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


def metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mse = float(np.mean(err**2))
    mape = float(np.mean(np.abs(err) / np.clip(np.abs(y_true), 1e-9, None))) * 100.0
    return {"MAE": mae, "RMSE": rmse, "MSE": mse, "MAPE_%": mape}


def perform_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    hidden_layers: tuple[int, ...],
    learning_rate: float,
    alpha: float,
    n_splits: int = 5,
) -> list[dict]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            learning_rate_init=learning_rate,
            alpha=alpha,
            max_iter=400,
            random_state=RANDOM_STATE,
            shuffle=False,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
            tol=1e-4,
        )
        mlp.fit(X_train_scaled, y_train)

        train_pred = mlp.predict(X_train_scaled)
        val_pred = mlp.predict(X_val_scaled)

        results.append(
            {
                "fold": fold,
                "train_metrics": metrics(y_train, train_pred),
                "val_metrics": metrics(y_val, val_pred),
                "n_iter": mlp.n_iter_,
                "train_r2": float(mlp.score(X_train_scaled, y_train)),
                "val_r2": float(mlp.score(X_val_scaled, y_val)),
            }
        )

    return results


def aggregate_cv_metrics(cv_results: list[dict]) -> dict[str, dict[str, float]]:
    """Aggregate CV metrics across folds (mean/std)."""
    agg: dict[str, dict[str, float]] = {"train": {}, "val": {}}
    for split in ["train", "val"]:
        for metric_name in cv_results[0][f"{split}_metrics"]:
            values = [fold[f"{split}_metrics"][metric_name] for fold in cv_results]
            agg[split][metric_name] = float(np.mean(values))
            agg[split][f"{metric_name}_std"] = float(np.std(values, ddof=1))
    return agg


def train_with_curves(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    hidden_layers: tuple[int, ...],
    learning_rate: float,
    alpha: float,
    max_epochs: int,
    patience: int,
) -> tuple[MLPRegressor, StandardScaler, list[float], list[float], int]:
    """Train ANN while tracking RMSE on train/validation splits."""
    if not 0 < VAL_FRAC_WITHIN_TRAIN < 1:
        raise ValueError("VAL_FRAC_WITHIN_TRAIN must lie in (0, 1).")

    val_size = max(1, int(len(X_train) * VAL_FRAC_WITHIN_TRAIN))
    X_subtrain = X_train.iloc[:-val_size]
    X_val = X_train.iloc[-val_size:]
    y_subtrain = y_train.iloc[:-val_size]
    y_val = y_train.iloc[-val_size:]

    if len(X_subtrain) == 0:
        raise ValueError(
            "Training window too small after applying VAL_FRAC_WITHIN_TRAIN; "
            "reduce the validation fraction or enlarge the training sample."
        )

    scaler = StandardScaler()
    X_subtrain_scaled = scaler.fit_transform(X_subtrain)
    X_val_scaled = scaler.transform(X_val)

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        learning_rate_init=learning_rate,
        alpha=alpha,
        max_iter=1,
        warm_start=True,
        random_state=RANDOM_STATE,
        shuffle=False,
        tol=0.0,
    )

    train_curve: list[float] = []
    val_curve: list[float] = []
    best_model: MLPRegressor | None = None
    best_val_rmse = float("inf")
    best_epoch = 0
    epochs_without_improve = 0

    for epoch in range(1, max_epochs + 1):
        mlp.fit(X_subtrain_scaled, y_subtrain)

        train_pred = mlp.predict(X_subtrain_scaled)
        val_pred = mlp.predict(X_val_scaled)
        train_rmse = float(np.sqrt(np.mean((y_subtrain - train_pred) ** 2)))
        val_rmse = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))

        train_curve.append(train_rmse)
        val_curve.append(val_rmse)

        if val_rmse + 1e-6 < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = deepcopy(mlp)
            best_epoch = epoch
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= patience:
            break

    if best_model is None:
        best_model = mlp
        best_epoch = len(train_curve)

    return best_model, scaler, train_curve, val_curve, best_epoch


def refit_final_ann(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    hidden_layers: tuple[int, ...],
    learning_rate: float,
    alpha: float,
    epochs: int,
) -> tuple[MLPRegressor, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        learning_rate_init=learning_rate,
        alpha=alpha,
        max_iter=1,
        warm_start=True,
        random_state=RANDOM_STATE,
        shuffle=False,
        tol=0.0,
    )

    for _ in range(max(1, epochs)):
        mlp.fit(X_train_scaled, y_train)

    return mlp, scaler


def save_training_curve(train_curve: list[float], val_curve: list[float], path: Path):
    epochs = np.arange(1, len(train_curve) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_curve, label="Train RMSE", marker="o", linewidth=1.2)
    ax.plot(epochs, val_curve, label="Validation RMSE", marker="s", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("ANN Training / Validation RMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_forecast_plot(
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: pd.Series,
    path: Path,
    title: str,
):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_train.index, y_train.values, label="Train (actual)", linewidth=1.1)
    ax.plot(y_test.index, y_test.values, label="Test (actual)", linewidth=1.1)
    ax.plot(y_pred.index, y_pred.values, label="Test (predicted)", linewidth=1.4)
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
    ax.set_title("Residual ACF (ANN hold-out)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(
    report_path: Path,
    *,
    dataset_info: dict,
    ann_config: dict,
    cv_results: list[dict],
    cv_summary: dict,
    train_metrics: dict,
    test_metrics: dict,
    residual_info: dict,
):
    lines: list[str] = []

    lines.append("TIME SERIES MODEL REPORT — Feed-forward ANN (Daily)\n")
    lines.append("Dataset\n-------")
    for k, v in dataset_info.items():
        lines.append(f"{k}: {v}")
    lines.append("")

    lines.append("Model Configuration\n-------------------")
    for k, v in ann_config.items():
        lines.append(f"{k}: {v}")
    lines.append("")

    lines.append("Cross-validation (TimeSeriesSplit)\n----------------------------------")
    for fold in cv_results:
        cv_line = (
            f"Fold {fold['fold']:>2} | "
            f"Train MAE={fold['train_metrics']['MAE']:.4f}, RMSE={fold['train_metrics']['RMSE']:.4f} "
            f"| Val MAE={fold['val_metrics']['MAE']:.4f}, RMSE={fold['val_metrics']['RMSE']:.4f} "
            f"| Train R²={fold['train_r2']:.3f}, Val R²={fold['val_r2']:.3f}, Iter={fold['n_iter']}"
        )
        lines.append(cv_line)
    lines.append("")
    lines.append("Cross-validation summary (mean ± std)")
    for split in ["train", "val"]:
        lines.append(f"{split.title()} metrics:")
        for metric_name, value in cv_summary[split].items():
            if metric_name.endswith("_std"):
                continue
            std = cv_summary[split][f"{metric_name}_std"]
            lines.append(f"  {metric_name}: {value:.4f} ± {std:.4f}")
    lines.append("")

    lines.append("Hold-out performance\n--------------------")
    lines.append(
        "Train metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
    )
    lines.append(
        "Test metrics : " + ", ".join(f"{k}={v:.4f}" for k, v in test_metrics.items())
    )
    lines.append("")

    lines.append("Residual diagnostics (test residuals)\n-----------------------------------")
    lines.append(f"Mean: {residual_info['mean']:.6f}")
    lines.append(f"Variance: {residual_info['variance']:.6f}")
    lines.append("Ljung–Box p-values:")
    for lag, pval in residual_info["ljung_box_pvalues"]:
        lines.append(f"  lag={lag:>2}: p-value={pval:.4g}")
    lines.append(
        f"Heteroskedasticity (ARCH LM) p-value: {residual_info['arch_pvalue']:.4g}"
    )
    lines.append(
        f"Jarque–Bera: stat={residual_info['jb_stat']:.3f}, "
        f"p={residual_info['jb_pvalue']:.4g}, skew={residual_info['jb_skew']:.3f}, "
        f"kurt={residual_info['jb_kurt']:.3f}"
    )
    lines.append("")

    lines.append("Artifacts\n---------")
    lines.append(f"Forecast plot : {FORECAST_PLOT_PATH}")
    lines.append(f"Training curve: {TRAIN_CURVE_PATH}")
    lines.append(f"Residual ACF  : {RESIDUAL_ACF_PATH}")
    lines.append("")

    report_path.write_text("\n".join(lines))


# -----------------------
# Main
# -----------------------
def main():
    np.random.seed(RANDOM_STATE)

    daily = load_fluxnet_daily(FLUXNET_CSV, TARGET, EXOG_VARS)

    if USE_GCC:
        try:
            gcc = load_gcc_daily(GCC_CSV)
            daily = daily.join(gcc, how="left")
        except FileNotFoundError:
            print("Warning: GCC file not found → proceeding without GCC.")

    base_cols = [c for c in EXOG_VARS if c in daily.columns]
    X_parts = [daily[base_cols].copy()]
    if USE_FOURIER and FOURIER_K > 0:
        fourier = fourier_terms(daily.index, period_days=365.25, K=FOURIER_K)
        X_parts.append(fourier)
    X = pd.concat(X_parts, axis=1)

    y = daily[TARGET].rename("y").astype(float)
    X = X.apply(pd.to_numeric, errors="coerce")
    data = pd.concat([y, X], axis=1).dropna()

    feature_cols = [c for c in data.columns if c != "y"]
    X_all = data[feature_cols].astype(float)
    y_all = data["y"].astype(float)

    # Cross-validation over the full historical window.
    cv_results = perform_cross_validation(
        X_all,
        y_all,
        hidden_layers=HIDDEN_LAYERS,
        learning_rate=LEARNING_RATE,
        alpha=L2_ALPHA,
    )
    cv_summary = aggregate_cv_metrics(cv_results) if cv_results else {}

    train_df, test_df = split_train_test(data, TRAIN_FRAC)

    X_train = train_df[feature_cols].astype(float)
    y_train = train_df["y"].astype(float)
    X_test = test_df[feature_cols].astype(float)
    y_test = test_df["y"].astype(float)

    best_model, train_scaler, train_curve, val_curve, best_epoch = train_with_curves(
        X_train,
        y_train,
        hidden_layers=HIDDEN_LAYERS,
        learning_rate=LEARNING_RATE,
        alpha=L2_ALPHA,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
    )
    save_training_curve(train_curve, val_curve, TRAIN_CURVE_PATH)

    final_model, final_scaler = refit_final_ann(
        X_train,
        y_train,
        hidden_layers=HIDDEN_LAYERS,
        learning_rate=LEARNING_RATE,
        alpha=L2_ALPHA,
        epochs=best_epoch,
    )

    X_train_scaled_final = final_scaler.transform(X_train)
    X_test_scaled = final_scaler.transform(X_test)

    train_pred = pd.Series(
        final_model.predict(X_train_scaled_final),
        index=y_train.index,
        name="ann_train_pred",
    )
    test_pred = pd.Series(
        final_model.predict(X_test_scaled),
        index=y_test.index,
        name="ann_test_pred",
    )

    train_metrics = metrics(y_train, train_pred)
    test_metrics = metrics(y_test, test_pred)

    residuals = (y_test - test_pred).rename("residual")
    lb = acorr_ljungbox(residuals.dropna(), lags=[7, 14, 21], return_df=True)
    if "lag" in lb.columns:
        lags = lb["lag"].astype(int).to_numpy()
    else:
        lags = lb.index.astype(int)
    pvals = lb["lb_pvalue"].astype(float).to_numpy()
    ljung_box_stats = list(zip(lags, pvals))

    arch_stat, arch_pvalue, _, arch_fpvalue = het_arch(residuals.dropna(), maxlag=7)
    jb_stat, jb_pvalue, skew, kurt = jarque_bera(residuals.dropna())

    residual_info = {
        "mean": float(residuals.mean()),
        "variance": float(residuals.var(ddof=1)),
        "ljung_box_pvalues": ljung_box_stats,
        "arch_stat": float(arch_stat),
        "arch_pvalue": float(arch_pvalue),
        "arch_fpvalue": float(arch_fpvalue),
        "jb_stat": float(jb_stat),
        "jb_pvalue": float(jb_pvalue),
        "jb_skew": float(skew),
        "jb_kurt": float(kurt),
    }

    save_forecast_plot(
        y_train,
        y_test,
        test_pred,
        FORECAST_PLOT_PATH,
        title=f"Daily {TARGET} Forecast (Feed-forward ANN)",
    )
    save_residual_acf_plot(residuals, RESIDUAL_ACF_PATH)

    dataset_info = {
        "Target": TARGET,
        "Frequency": "Daily",
        "Total obs": len(data),
        "Train obs": len(train_df),
        "Test obs": len(test_df),
        "Train window": f"{y_train.index.min().date()} → {y_train.index.max().date()}",
        "Test window": f"{y_test.index.min().date()} → {y_test.index.max().date()}",
    }
    ann_config = {
        "Hidden layers": HIDDEN_LAYERS,
        "Learning rate": LEARNING_RATE,
        "L2 alpha": L2_ALPHA,
        "Fourier harmonics": FOURIER_K if USE_FOURIER else 0,
        "GCC enabled": USE_GCC,
        "Feature set": ", ".join(feature_cols),
        "Best epoch (val RMSE)": best_epoch,
    }

    write_report(
        REPORT_PATH,
        dataset_info=dataset_info,
        ann_config=ann_config,
        cv_results=cv_results,
        cv_summary=cv_summary,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        residual_info=residual_info,
    )

    print("\n=== ANN pipeline complete ===")
    print(f"Report          → {REPORT_PATH}")
    print(f"Forecast plot   → {FORECAST_PLOT_PATH}")
    print(f"Training curve  → {TRAIN_CURVE_PATH}")
    print(f"Residual ACF    → {RESIDUAL_ACF_PATH}")


if __name__ == "__main__":
    main()
