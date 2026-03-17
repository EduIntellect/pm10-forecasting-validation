#!/usr/bin/env python3
"""
Rolling-origin H* computation for PM10 using XGBoost.

Protocol:
- Initial train: 2017-01-01 to 2019-12-31
- Rolling-origin evaluation: 2020-01-01 to 2023-12-31
- Monthly retraining frequency (expanding window)
- Horizons: h=1..7
- Baseline: lag-1 persistence (y_t for all horizons)
- Skill(h) = 1 - RMSE_model(h) / RMSE_persistence(h)
- H* = max(h : mean skill(h) > 0)

Outputs under results_rolling/:
- rolling_hstar_results_xgb.csv
- rolling_hstar_summary_xgb.csv
- rolling_hstar_summary_xgb.txt
- rolling_hstar_xgb.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: XGBoost not installed. Install with: pip install xgboost")
    sys.exit(1)


INPUT_PATH = Path("pm10_clean.csv")
OUTPUT_DIR = Path("results_rolling")

LAGS = list(range(1, 8))
HORIZONS = list(range(1, 8))
METEO_VARS = ["temp", "hr", "ws", "wd"]

TRAIN_START = pd.Timestamp("2017-01-01")
INITIAL_TRAIN_END = pd.Timestamp("2019-12-31")
EVAL_START = pd.Timestamp("2020-01-01")
EVAL_END = pd.Timestamp("2023-12-31")

MIN_TEST_OBS_PER_FOLD = 15

XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["date", "pm10"]).copy()
    return df


def add_features_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for lag in LAGS:
        out[f"pm10_lag{lag}"] = out["pm10"].shift(lag)

    for h in HORIZONS:
        out[f"pm10_h{h}"] = out["pm10"].shift(-h)

    return out


def monthly_folds(eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    starts = pd.date_range(eval_start, eval_end, freq="MS")
    folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start in starts:
        end = start + pd.offsets.MonthEnd(0)
        if end > eval_end:
            end = eval_end
        folds.append((start, end))
    return folds


def evaluate_rolling_xgb(df: pd.DataFrame) -> pd.DataFrame:
    lag_cols = [f"pm10_lag{lag}" for lag in LAGS]
    feature_cols = lag_cols + [v for v in METEO_VARS if v in df.columns]

    rows: list[dict] = []

    for fold_start, fold_end in monthly_folds(EVAL_START, EVAL_END):
        train_end = fold_start - pd.Timedelta(days=1)

        # Enforce initial training minimum window.
        if train_end < INITIAL_TRAIN_END:
            continue

        train_mask = (df["date"] >= TRAIN_START) & (df["date"] <= train_end)
        base_test_mask = (df["date"] >= fold_start) & (df["date"] <= fold_end)

        df_train_base = df.loc[train_mask].copy()
        df_test_base = df.loc[base_test_mask].copy()

        if df_train_base.empty or df_test_base.empty:
            continue

        # Fit scaler once per fold using only fold-train data with complete features.
        train_base_for_scaler = df_train_base[feature_cols].dropna()
        if train_base_for_scaler.empty:
            continue
        scaler = MinMaxScaler()
        scaler.fit(train_base_for_scaler)

        # Enforce minimum test size once per fold on base fold test data.
        test_base_valid = df_test_base.dropna(subset=feature_cols + ["pm10"])
        if len(test_base_valid) < MIN_TEST_OBS_PER_FOLD:
            continue

        for h in HORIZONS:
            target_col = f"pm10_h{h}"

            # Horizon-specific train set (drop rows missing any required field).
            tr_cols = feature_cols + [target_col]
            train_h = df_train_base.dropna(subset=tr_cols)

            if train_h.empty:
                continue

            X_train = train_h[feature_cols]
            y_train = train_h[target_col].values

            X_train_scaled = scaler.transform(X_train)

            model = xgb.XGBRegressor(**XGB_PARAMS)
            model.fit(X_train_scaled, y_train)

            # Horizon-specific test set; persistence baseline is y_t (current-row pm10).
            te_cols = feature_cols + [target_col, "pm10"]
            test_h = df_test_base.dropna(subset=te_cols)

            n_test = len(test_h)

            X_test = test_h[feature_cols]
            y_true = test_h[target_col].values
            y_pers = test_h["pm10"].values

            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)

            rmse_model = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            rmse_pers = float(np.sqrt(mean_squared_error(y_true, y_pers)))
            skill = float(1.0 - (rmse_model / rmse_pers))

            rows.append(
                {
                    "fold_month": fold_start.strftime("%Y-%m"),
                    "fold_start": fold_start.date().isoformat(),
                    "fold_end": fold_end.date().isoformat(),
                    "horizon": h,
                    "n_test": n_test,
                    "rmse_xgb": rmse_model,
                    "rmse_persistence": rmse_pers,
                    "skill": skill,
                }
            )

    return pd.DataFrame(rows)


def summarize_by_horizon(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results_df.groupby("horizon", as_index=False)
        .agg(
            n_folds=("skill", "size"),
            mean_rmse_xgb=("rmse_xgb", "mean"),
            std_rmse_xgb=("rmse_xgb", "std"),
            mean_rmse_persistence=("rmse_persistence", "mean"),
            std_rmse_persistence=("rmse_persistence", "std"),
            mean_skill=("skill", "mean"),
            std_skill=("skill", "std"),
            min_skill=("skill", "min"),
            max_skill=("skill", "max"),
        )
        .sort_values("horizon")
    )
    return summary


def compute_h_star(summary_df: pd.DataFrame) -> int:
    positive = summary_df.loc[summary_df["mean_skill"] > 0, "horizon"].tolist()
    return int(max(positive)) if positive else 0


def save_text_summary(results_df: pd.DataFrame, summary_df: pd.DataFrame, h_star: int, out_path: Path) -> None:
    valid_fold_months = results_df["fold_month"].nunique()

    global_min_skill = float(results_df["skill"].min())
    global_max_skill = float(results_df["skill"].max())

    lines = [
        "=" * 72,
        "ROLLING-ORIGIN H* SUMMARY (XGBoost vs lag-1 persistence)",
        "=" * 72,
        "",
        f"Protocol: monthly rolling-origin, expanding train, evaluation {EVAL_START.date()} to {EVAL_END.date()}",
        f"Initial train: {TRAIN_START.date()} to {INITIAL_TRAIN_END.date()}",
        f"Horizons: {HORIZONS}",
        f"Minimum valid test observations per fold-horizon: {MIN_TEST_OBS_PER_FOLD}",
        "",
        f"Valid fold months (with at least one valid horizon): {valid_fold_months}",
        f"H*: {h_star}",
        f"Global skill range across valid fold-horizon pairs: [{global_min_skill:.3f}, {global_max_skill:.3f}]",
        "",
        "Mean skill by horizon:",
    ]

    for _, r in summary_df.iterrows():
        lines.append(
            f"  h={int(r['horizon'])}: mean={r['mean_skill']:.3f}, std={r['std_skill']:.3f}, n_folds={int(r['n_folds'])}"
        )

    lines.extend(["", "=" * 72])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_skill(summary_df: pd.DataFrame, h_star: int, out_path: Path) -> None:
    x = summary_df["horizon"].values
    y = summary_df["mean_skill"].values
    yerr = summary_df["std_skill"].fillna(0).values

    plt.figure(figsize=(8, 5))
    plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=4, linewidth=2, label="Mean skill ± 1 std")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Forecast horizon h (days)")
    plt.ylabel("Skill = 1 - RMSE_model / RMSE_persistence")
    plt.title(f"Rolling-origin PM10 skill (XGBoost vs lag-1 persistence), H*={h_star}")
    plt.xticks(HORIZONS)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"ERROR: Missing input file: {INPUT_PATH}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(INPUT_PATH)
    df = add_features_targets(df)

    results_df = evaluate_rolling_xgb(df)

    if results_df.empty:
        print("ERROR: No valid fold-horizon pairs after filtering.")
        return 1

    summary_df = summarize_by_horizon(results_df)
    h_star = compute_h_star(summary_df)

    results_csv = OUTPUT_DIR / "rolling_hstar_results_xgb.csv"
    summary_csv = OUTPUT_DIR / "rolling_hstar_summary_xgb.csv"
    summary_txt = OUTPUT_DIR / "rolling_hstar_summary_xgb.txt"
    fig_path = OUTPUT_DIR / "rolling_hstar_xgb.png"

    results_df.to_csv(results_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    save_text_summary(results_df, summary_df, h_star, summary_txt)
    plot_skill(summary_df, h_star, fig_path)

    print("Rolling-origin run completed.")
    print(f"Valid fold-horizon rows: {len(results_df)}")
    print(f"Valid fold months: {results_df['fold_month'].nunique()}")
    print(f"H*: {h_star}")
    print(f"Saved: {results_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_txt}")
    print(f"Saved: {fig_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
