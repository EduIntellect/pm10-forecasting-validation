#!/usr/bin/env python3
"""
Rolling-origin H* computation for PM10 using SARIMA.

Protocol:
- Initial train: 2017-01-01 to 2019-12-31
- Rolling-origin evaluation: 2020-01-01 to 2023-12-31
- Monthly folds; evaluation over all valid daily origins within each fold
- Horizons: h=1..7
- Baseline: persistence y_t for all horizons
- Skill(h) = 1 - RMSE_model(h) / RMSE_persistence(h)
- H* = max(h : mean skill(h) > 0)

Outputs under results_rolling/:
- rolling_hstar_results_sarima.csv
- rolling_hstar_summary_sarima.csv
- rolling_hstar_summary_sarima.txt
- rolling_hstar_sarima.png
"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


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
MIN_TRAIN_OBS_SARIMA = 60
MAX_FORECAST_ABS = 500.0
SEASONAL_PERIOD = 7

# Update strategy note required by protocol.
# We use full refit per origin for methodological clarity and stability.
UPDATE_STRATEGY = "fit_once_per_fold_append_refit_false"


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


def identify_sarima_order(initial_series: pd.Series) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    """
    Identify (order, seasonal_order) once using only the initial train window.
    """
    p = [0, 1, 2]
    d = [0, 1]
    q = [0, 1, 2]
    P = [0, 1]
    D = [0, 1]
    Q = [0, 1]

    best_aic = np.inf
    best_cfg: tuple[tuple[int, int, int], tuple[int, int, int, int]] | None = None

    for order in itertools.product(p, d, q):
        for s_order_3 in itertools.product(P, D, Q):
            seasonal_order = (s_order_3[0], s_order_3[1], s_order_3[2], SEASONAL_PERIOD)

            # Avoid fully null model.
            if order == (0, 0, 0) and seasonal_order[:3] == (0, 0, 0):
                continue

            try:
                model = SARIMAX(
                    initial_series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit = model.fit(disp=False, maxiter=80)
                if np.isfinite(fit.aic) and fit.aic < best_aic:
                    best_aic = fit.aic
                    best_cfg = (order, seasonal_order)
            except Exception:
                continue

    if best_cfg is None:
        # Defensive fallback: weekly seasonal baseline.
        best_cfg = ((1, 0, 1), (1, 0, 1, SEASONAL_PERIOD))

    return best_cfg


def evaluate_rolling_sarima_daily_origins(
    df: pd.DataFrame,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> tuple[pd.DataFrame, list[str], set[str]]:
    """
    Evaluate SARIMA on monthly folds using all valid daily origins per month.

    Returns:
    - results dataframe (one row per fold_month, horizon)
    - log lines (invalid/skip fold or origin reasons)
    - set of fold months with at least one valid result
    """
    lag_cols = [f"pm10_lag{lag}" for lag in LAGS]
    feature_cols = lag_cols + [v for v in METEO_VARS if v in df.columns]

    rows: list[dict] = []
    logs: list[str] = []
    valid_fold_months: set[str] = set()

    logs.append(
        "GLOBAL | INFO | update_policy=fit_once_per_fold_then_append_refit_false (fallback: fold-level static forecast)"
    )

    for fold_start, fold_end in monthly_folds(EVAL_START, EVAL_END):
        fold_month = fold_start.strftime("%Y-%m")
        train_end = fold_start - pd.Timedelta(days=1)

        fold_mask = (df["date"] >= fold_start) & (df["date"] <= fold_end)
        df_fold = df.loc[fold_mask].copy()

        if df_fold.empty:
            logs.append(f"{fold_month} | SKIP_FOLD | empty fold window")
            continue

        # Keep fold-validity criterion aligned with rolling_origin_hstar.py.
        test_base_valid = df_fold.dropna(subset=feature_cols + ["pm10"])
        if len(test_base_valid) < MIN_TEST_OBS_PER_FOLD:
            logs.append(
                f"{fold_month} | SKIP_FOLD | base test valid obs={len(test_base_valid)} < {MIN_TEST_OBS_PER_FOLD}"
            )
            continue

        # Fit SARIMA ONCE per fold with history up to fold_start - 1 day.
        train_mask = (df["date"] >= TRAIN_START) & (df["date"] <= train_end)
        train_series = df.loc[train_mask, "pm10"].dropna()
        if len(train_series) < MIN_TRAIN_OBS_SARIMA:
            logs.append(
                f"{fold_month} | SKIP_FOLD | train valid obs={len(train_series)} < {MIN_TRAIN_OBS_SARIMA}"
            )
            continue

        try:
            base_model = SARIMAX(
                train_series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            base_fit = base_model.fit(disp=False, maxiter=80)
        except Exception as e:
            logs.append(f"{fold_month} | SKIP_FOLD | fold fit failed: {type(e).__name__}: {e}")
            continue

        # Accumulate squared errors per horizon across all valid origins in the fold.
        errors_model: dict[int, list[float]] = {h: [] for h in HORIZONS}
        errors_pers: dict[int, list[float]] = {h: [] for h in HORIZONS}

        # Primary path: update state within fold with append(refit=False).
        # Fallback path: if append/update is not available/stable, keep fold-level static forecast
        # from base_fit (still no daily parameter re-estimation).
        current_fit = base_fit
        last_appended_date = train_end
        use_static_fallback = False

        for origin_date in sorted(df_fold["date"].unique()):
            origin_date = pd.Timestamp(origin_date)
            origin_row = df.loc[df["date"] == origin_date]
            if origin_row.empty or pd.isna(origin_row["pm10"].iloc[0]):
                logs.append(f"{fold_month} | {origin_date.date()} | SKIP_ORIGIN | missing y_t")
                continue

            y_t = float(origin_row["pm10"].iloc[0])

            if not use_static_fallback:
                # Reveal observations from (last_appended_date, origin_date] and update state without refit.
                new_obs = df.loc[
                    (df["date"] > last_appended_date) & (df["date"] <= origin_date), "pm10"
                ].dropna()

                if len(new_obs) == 0:
                    logs.append(
                        f"{fold_month} | {origin_date.date()} | SKIP_ORIGIN | no revealed observations to update state"
                    )
                    continue

                try:
                    if hasattr(current_fit, "append"):
                        current_fit = current_fit.append(new_obs.values, refit=False)
                        last_appended_date = origin_date
                    else:
                        use_static_fallback = True
                        logs.append(
                            f"{fold_month} | INFO | append(refit=False) unavailable; using fold-level static forecast fallback"
                        )
                except Exception as e:
                    use_static_fallback = True
                    logs.append(
                        f"{fold_month} | INFO | append(refit=False) failed ({type(e).__name__}: {e}); using fold-level static forecast fallback"
                    )

            try:
                fit_for_forecast = base_fit if use_static_fallback else current_fit
                forecast = np.asarray(fit_for_forecast.forecast(steps=max(HORIZONS)), dtype=float)
            except Exception as e:
                logs.append(
                    f"{fold_month} | {origin_date.date()} | SKIP_ORIGIN | "
                    f"forecast failed: {type(e).__name__}: {e}"
                )
                continue

            if (not np.all(np.isfinite(forecast))) or np.any(np.abs(forecast) > MAX_FORECAST_ABS):
                logs.append(
                    f"{fold_month} | {origin_date.date()} | SKIP_ORIGIN | "
                    f"invalid forecast values (NaN/inf or abs>{MAX_FORECAST_ABS})"
                )
                continue

            for h in HORIZONS:
                target_date = origin_date + pd.Timedelta(days=h)
                target_row = df.loc[df["date"] == target_date]

                if target_row.empty or pd.isna(target_row["pm10"].iloc[0]):
                    logs.append(
                        f"{fold_month} | {origin_date.date()} | h={h} | SKIP_H | "
                        f"missing y_(t+h) on {target_date.date()}"
                    )
                    continue

                y_true = float(target_row["pm10"].iloc[0])
                y_pred = float(forecast[h - 1])
                y_pers = y_t  # Persistence baseline is y_t for all horizons.

                errors_model[h].append((y_pred - y_true) ** 2)
                errors_pers[h].append((y_pers - y_true) ** 2)

        # Aggregate fold results per horizon (RMSE over all valid origins in the fold).
        fold_has_valid = False
        for h in HORIZONS:
            n_origins_valid = len(errors_model[h])
            if n_origins_valid < 5:
                logs.append(
                    f"{fold_month} | h={h} | SKIP_FOLD_H | n_origins_valid={n_origins_valid} < 5"
                )
                continue

            rmse_model = float(np.sqrt(np.mean(errors_model[h])))
            rmse_pers = float(np.sqrt(np.mean(errors_pers[h])))

            if rmse_pers == 0:
                logs.append(f"{fold_month} | h={h} | SKIP_FOLD_H | rmse_persistence=0")
                continue

            skill = float(1.0 - (rmse_model / rmse_pers))

            rows.append(
                {
                    "fold_month": fold_month,
                    "fold_start": fold_start.date().isoformat(),
                    "fold_end": fold_end.date().isoformat(),
                    "horizon": h,
                    "n_origins_valid": n_origins_valid,
                    "rmse_sarima": rmse_model,
                    "rmse_persistence": rmse_pers,
                    "skill": skill,
                }
            )
            fold_has_valid = True

        if fold_has_valid:
            valid_fold_months.add(fold_month)

    results_df = pd.DataFrame(rows)
    return results_df, logs, valid_fold_months


def summarize_by_horizon(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results_df.groupby("horizon", as_index=False)
        .agg(
            n_folds=("skill", "size"),
            mean_rmse_sarima=("rmse_sarima", "mean"),
            std_rmse_sarima=("rmse_sarima", "std"),
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


def save_text_summary(
    summary_df: pd.DataFrame,
    results_df: pd.DataFrame,
    logs: list[str],
    h_star: int,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    valid_fold_months: set[str],
    out_path: Path,
) -> None:
    global_min_skill = float(results_df["skill"].min())
    global_max_skill = float(results_df["skill"].max())

    lines = [
        "=" * 78,
        "ROLLING-ORIGIN H* SUMMARY (SARIMA vs y_t persistence)",
        "=" * 78,
        "",
        f"Protocol: monthly folds with all valid daily origins, evaluation {EVAL_START.date()} to {EVAL_END.date()}",
        f"Initial train: {TRAIN_START.date()} to {INITIAL_TRAIN_END.date()}",
        f"Horizons: {HORIZONS}",
        f"Fold-validity threshold: base test valid obs >= {MIN_TEST_OBS_PER_FOLD}",
        f"Minimum train per origin: {MIN_TRAIN_OBS_SARIMA}",
        f"Forecast validation: all finite and abs(value) <= {MAX_FORECAST_ABS}",
        f"Update strategy: {UPDATE_STRATEGY}",
        "",
        f"Selected SARIMA order: order={order}, seasonal_order={seasonal_order}",
        f"Valid fold months (>=1 valid row): {len(valid_fold_months)}",
        f"Valid rows (fold_month, horizon): {len(results_df)}",
        f"H*: {h_star}",
        f"Global skill range across valid rows: [{global_min_skill:.3f}, {global_max_skill:.3f}]",
        "",
        "Mean skill by horizon:",
    ]

    for _, r in summary_df.iterrows():
        lines.append(
            f"  h={int(r['horizon'])}: mean={r['mean_skill']:.3f}, std={r['std_skill']:.3f}, n_rows={int(r['n_folds'])}"
        )

    lines.extend(["", "Log (skipped/invalid folds-origins):"])
    if logs:
        lines.extend([f"  - {x}" for x in logs])
    else:
        lines.append("  - none")

    lines.extend(["", "=" * 78])
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
    plt.title(f"Rolling-origin PM10 skill (SARIMA vs y_t persistence), H*={h_star}")
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

    initial_series = df.loc[
        (df["date"] >= TRAIN_START) & (df["date"] <= INITIAL_TRAIN_END), "pm10"
    ].dropna()

    if len(initial_series) < MIN_TRAIN_OBS_SARIMA:
        print(
            f"ERROR: initial train has {len(initial_series)} valid observations; "
            f"requires >= {MIN_TRAIN_OBS_SARIMA}."
        )
        return 1

    print("Identifying SARIMA order on initial window (2017-2019)...")
    order, seasonal_order = identify_sarima_order(initial_series)
    print(f"Selected order={order}, seasonal_order={seasonal_order}")

    results_df, logs, valid_fold_months = evaluate_rolling_sarima_daily_origins(
        df, order, seasonal_order
    )

    if results_df.empty:
        print("ERROR: No valid rows after fold/origin filtering and robustness checks.")
        return 1

    summary_df = summarize_by_horizon(results_df)
    h_star = compute_h_star(summary_df)

    results_csv = OUTPUT_DIR / "rolling_hstar_results_sarima.csv"
    summary_csv = OUTPUT_DIR / "rolling_hstar_summary_sarima.csv"
    summary_txt = OUTPUT_DIR / "rolling_hstar_summary_sarima.txt"
    fig_path = OUTPUT_DIR / "rolling_hstar_sarima.png"

    results_df.to_csv(results_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    save_text_summary(summary_df, results_df, logs, h_star, order, seasonal_order, valid_fold_months, summary_txt)
    plot_skill(summary_df, h_star, fig_path)

    print("Rolling-origin SARIMA run completed.")
    print(f"Valid rows (fold, origin, horizon): {len(results_df)}")
    print(f"Valid fold months: {len(valid_fold_months)}")
    print(f"H*: {h_star}")
    print(f"Saved: {results_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_txt}")
    print(f"Saved: {fig_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
