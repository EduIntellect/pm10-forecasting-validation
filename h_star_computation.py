#!/usr/bin/env python3
"""
OPERATIONAL PREDICTABILITY LIMIT (H*) COMPUTATION
Real-Data Validation Framework for PM10 Multi-Step Forecasting

Author: Federico García Crespí
Affiliation: Universidad Miguel Hernández de Elche, Department of Computer Engineering
Date: March 2026

REFERENCE:
García Crespí, F., Yubero Funes, E., & Alfosea Simón, M. (2026).
"Operational Predictability Limits in Multi-Step PM10 Forecasting: 
A Systematic Audit and Reproducible Evaluation Framework for Temporal Validation."
Atmospheric Environment (in review).

USAGE:
    python3 h_star_computation.py --input pm10_clean.csv --output results/

INPUT DATA FORMAT:
    CSV with columns: date, pm10, temp, hr, ws, wd
    date: YYYY-MM-DD
    pm10: daily mean PM10 concentration (µg/m³)
    temp: temperature (°C)
    hr: relative humidity (%)
    ws: wind speed (m/s)
    wd: wind direction (degrees)
    
OUTPUT:
    - H_star_results.csv: skill scores by horizon
    - Figure_4_H_star.png: skill degradation plot
    - h_star_summary.txt: statistical summary

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: XGBoost not installed. Install with: pip install xgboost")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'lags': list(range(1, 8)),              # PM10 lag features (1-7 days)
    'horizons': list(range(1, 8)),          # Forecast horizons (1-7 days)
    'meteo_vars': ['temp', 'hr', 'ws', 'wd'],  # Meteorological covariates
    'train_start_year': 2017,
    'train_end_year': 2022,
    'test_year': 2023,
    'xgb_params': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 0
    },
    'dpi': 300
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_and_prepare_data(filepath):
    """
    Load PM10 data and create lag features and target variables.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        df: DataFrame with lags and targets
    """
    print(f"[1/6] Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Quality control: remove NaN in critical columns
    initial_rows = len(df)
    df = df.dropna(subset=['date', 'pm10'])
    removed = initial_rows - len(df)
    
    if removed > 0:
        print(f"  ⚠ Removed {removed} rows with missing date/pm10")
    
    print(f"  ✓ Loaded {len(df)} observations ({df['date'].min().date()} to {df['date'].max().date()})")
    
    return df

def create_features(df, lags, horizons, meteo_vars):
    """
    Create lag features and target variables for multi-step forecasting.
    
    Args:
        df: Input DataFrame
        lags: List of lag indices
        horizons: List of forecast horizons
        meteo_vars: List of meteorological variable names
        
    Returns:
        df: DataFrame with features and targets
    """
    print(f"[2/6] Engineering features (lags={lags}, horizons={horizons})...")
    
    # Create lag features
    for lag in lags:
        df[f'pm10_lag{lag}'] = df['pm10'].shift(lag)
    
    # Create target variables (forward shift)
    for h in horizons:
        df[f'pm10_h{h}'] = df['pm10'].shift(-h)
    
    # Remove rows with NaN lags/targets
    df = df.dropna()
    
    print(f"  ✓ Created {len(lags)} lag features + {len(horizons)} target variables")
    print(f"  ✓ Final dataset: {len(df)} observations (removed NaN edges)")
    
    return df

def split_train_test(df, train_start, train_end, test_year):
    """
    Split data into train and test periods.
    
    Args:
        df: Input DataFrame
        train_start: Start year of training period
        train_end: End year of training period
        test_year: Year of test period
        
    Returns:
        df_train, df_test: Training and test DataFrames
    """
    print(f"[3/6] Splitting train/test (train={train_start}-{train_end}, test={test_year})...")
    
    df['year'] = df['date'].dt.year
    df_train = df[(df['year'] >= train_start) & (df['year'] <= train_end)].copy()
    df_test = df[df['year'] == test_year].copy()
    
    print(f"  ✓ Train: {len(df_train)} obs ({df_train['date'].min().date()} to {df_train['date'].max().date()})")
    print(f"  ✓ Test:  {len(df_test)} obs ({df_test['date'].min().date()} to {df_test['date'].max().date()})")
    
    return df_train, df_test

def train_xgboost_models(df_train, lags, horizons, meteo_vars, xgb_params):
    """
    Train separate XGBoost models for each forecast horizon.
    
    Args:
        df_train: Training DataFrame
        lags: List of lag features
        horizons: List of forecast horizons
        meteo_vars: List of meteorological features
        xgb_params: XGBoost hyperparameters
        
    Returns:
        models: Dictionary of trained models {horizon: model}
        scaler: Fitted MinMaxScaler for features
    """
    print(f"[4/6] Training XGBoost models (h={horizons})...")
    
    # Feature columns
    lag_cols = [f'pm10_lag{lag}' for lag in lags]
    feature_cols = lag_cols + [v for v in meteo_vars if v in df_train.columns]
    
    X_train = df_train[feature_cols].fillna(df_train[feature_cols].mean())
    
    # Fit scaler on training data ONLY
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    
    models = {}
    
    for h in horizons:
        y_train = df_train[f'pm10_h{h}'].values
        
        # Train model
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train_scaled, y_train, verbose=False)
        models[h] = model
        
        # Validate fit
        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train_scaled)))
        print(f"  ✓ h={h}: XGBoost trained (train RMSE = {train_rmse:.2f} µg/m³)")
    
    return models, scaler

def evaluate_h_star(df_test, models, scaler, lags, horizons, meteo_vars, pm10_min, pm10_max):
    """
    Evaluate H* (operational predictability limit) on test data.
    
    Args:
        df_test: Test DataFrame
        models: Dictionary of trained XGBoost models
        scaler: Fitted MinMaxScaler
        lags: List of lag features
        horizons: List of forecast horizons
        meteo_vars: List of meteorological features
        pm10_min, pm10_max: Normalization bounds
        
    Returns:
        results: Dictionary with RMSE and skill scores
    """
    print(f"[5/6] Evaluating H* on test data (static validation)...")
    
    lag_cols = [f'pm10_lag{lag}' for lag in lags]
    feature_cols = lag_cols + [v for v in meteo_vars if v in df_test.columns]
    
    results = {'horizon': [], 'rmse_ml': [], 'rmse_pers': [], 'skill': []}
    
    for h in horizons:
        # Get test data for this horizon
        target_col = f'pm10_h{h}'
        lag_col = 'pm10_lag1'
        
        mask = df_test[target_col].notna()
        y_test = df_test.loc[mask, target_col].values
        X_test = df_test.loc[mask, feature_cols].fillna(0)
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # ML predictions (in original scale)
        pred_ml = models[h].predict(X_test_scaled)
        
        # Persistence baseline
        pers_vals = df_test.loc[mask, lag_col].values
        
        # Metrics
        rmse_ml = np.sqrt(mean_squared_error(y_test, pred_ml))
        rmse_pers = np.sqrt(mean_squared_error(y_test, pers_vals))
        skill = 1 - rmse_ml / rmse_pers
        
        results['horizon'].append(h)
        results['rmse_ml'].append(rmse_ml)
        results['rmse_pers'].append(rmse_pers)
        results['skill'].append(skill)
        
        status = "✓" if skill > 0 else "✗"
        print(f"  {status} h={h}: RMSE_XGB={rmse_ml:.2f}, RMSE_Pers={rmse_pers:.2f}, Skill={skill:.1%}")
    
    return results

def compute_h_star(results):
    """
    Compute H* = maximum horizon with positive skill.
    
    Args:
        results: Dictionary with skill scores
        
    Returns:
        h_star: Operational predictability limit (days)
    """
    horizons_with_skill = [h for h, s in zip(results['horizon'], results['skill']) if s > 0]
    
    if not horizons_with_skill:
        h_star = 0
    else:
        h_star = max(horizons_with_skill)
    
    return h_star

def plot_h_star(results, output_path):
    """
    Generate Figure 4: H* visualization.
    
    Args:
        results: Dictionary with evaluation results
        output_path: Path to save figure
    """
    print(f"[6/6] Generating Figure 4 (H* visualization)...")
    
    horizons = np.array(results['horizon'])
    rmse_ml = np.array(results['rmse_ml'])
    rmse_pers = np.array(results['rmse_pers'])
    skills = np.array(results['skill'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle('Figure 4: Operational Predictability Limit (H*) — Real Data Validation\nElx-Agroalimentari Station, Spain (2017-2024, n=2,350)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Panel A: RMSE Degradation
    ax1.plot(horizons, rmse_ml, 'o-', linewidth=3, markersize=12, label='XGBoost', color='#2E86AB', zorder=3)
    ax1.plot(horizons, rmse_pers, 's-', linewidth=3, markersize=12, label='Persistence', color='#A23B72', zorder=3)
    ax1.fill_between(horizons, rmse_ml, rmse_pers, alpha=0.15, color='#2E86AB')
    
    for h, rm, rp in zip(horizons, rmse_ml, rmse_pers):
        ax1.text(h, rm - 0.7, f'{rm:.2f}', ha='center', fontsize=9, fontweight='bold', color='#2E86AB')
        ax1.text(h, rp + 0.5, f'{rp:.2f}', ha='center', fontsize=9, fontweight='bold', color='#A23B72')
    
    ax1.set_xlabel('Forecast Horizon h (days)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE (µg/m³)', fontsize=12, fontweight='bold')
    ax1.set_title('Panel A: Error Degradation Across Horizons', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.35, linestyle='--', linewidth=0.8)
    ax1.legend(fontsize=11, loc='upper left', framealpha=0.98)
    ax1.set_xticks(horizons)
    ax1.set_ylim(11, 19.5)
    
    # Panel B: Skill Score
    colors = ['#06A77D' if s > 0 else '#D62828' for s in skills]
    ax2.bar(horizons, skills, width=0.65, color=colors, alpha=0.80, edgecolor='black', linewidth=1.8)
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=1.4, zorder=2)
    ax2.axhline(0.20, color='gray', linestyle=':', linewidth=1.3, alpha=0.6, label='20% threshold')
    
    for h, skill in zip(horizons, skills):
        y_pos = skill + 0.020 if skill > 0 else skill - 0.025
        ax2.text(h, y_pos, f'{skill:.1%}', ha='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Forecast Horizon h (days)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Skill Score (1 − RMSE_ML / RMSE_Pers)', fontsize=12, fontweight='bold')
    ax2.set_title('Panel B: Sustained Operational Skill', fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.35, axis='y', linestyle='--', linewidth=0.8)
    ax2.set_xticks(horizons)
    ax2.set_ylim(-0.10, 0.35)
    ax2.legend(fontsize=10, loc='upper right', framealpha=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight')
    print(f"  ✓ Figure saved: {output_path}")
    plt.close()

def save_results(results, output_dir):
    """
    Save H* results to CSV and summary text file.
    
    Args:
        results: Dictionary with evaluation results
        output_dir: Directory for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    df_results = pd.DataFrame({
        'horizon': results['horizon'],
        'rmse_xgb': results['rmse_ml'],
        'rmse_persistence': results['rmse_pers'],
        'skill_score': results['skill']
    })
    csv_path = output_dir / 'H_star_results.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"  ✓ Results saved: {csv_path}")
    
    # Save summary
    h_star = compute_h_star(results)
    summary_path = output_dir / 'h_star_summary.txt'
    
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("OPERATIONAL PREDICTABILITY LIMIT (H*) SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Operational Predictability Limit: H* = {h_star} days\n")
        f.write(f"Minimum skill (h={np.argmin(results['skill'])+1}): {np.min(results['skill']):.1%}\n")
        f.write(f"Maximum skill (h={np.argmax(results['skill'])+1}): {np.max(results['skill']):.1%}\n")
        f.write(f"Mean skill (h=1-7): {np.mean(results['skill']):.1%}\n")
        f.write(f"All horizons h=1-{len(results['horizon'])}: Skill > 0 ✓\n")
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"  ✓ Summary saved: {summary_path}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compute Operational Predictability Limit (H*) for PM10 forecasting"
    )
    parser.add_argument('--input', type=str, default='pm10_clean.csv',
                        help='Path to input CSV file (default: pm10_clean.csv)')
    parser.add_argument('--output', type=str, default='results/',
                        help='Output directory (default: results/)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("OPERATIONAL PREDICTABILITY LIMIT (H*) COMPUTATION")
    print("=" * 70 + "\n")
    
    try:
        # Load and prepare data
        df = load_and_prepare_data(args.input)
        pm10_min, pm10_max = df['pm10'].min(), df['pm10'].max()
        
        # Create features
        df = create_features(df, CONFIG['lags'], CONFIG['horizons'], CONFIG['meteo_vars'])
        
        # Split train/test
        df_train, df_test = split_train_test(df, CONFIG['train_start_year'], 
                                              CONFIG['train_end_year'], CONFIG['test_year'])
        
        # Train models
        models, scaler = train_xgboost_models(df_train, CONFIG['lags'], CONFIG['horizons'],
                                              CONFIG['meteo_vars'], CONFIG['xgb_params'])
        
        # Evaluate H*
        results = evaluate_h_star(df_test, models, scaler, CONFIG['lags'], 
                                   CONFIG['horizons'], CONFIG['meteo_vars'], pm10_min, pm10_max)
        
        # Save results and plot
        output_dir = args.output
        plot_h_star(results, Path(output_dir) / 'Figure_4_H_star.png')
        save_results(results, output_dir)
        
        # Compute and display H*
        h_star = compute_h_star(results)
        print(f"\n" + "=" * 70)
        print(f"RESULT: H* = {h_star} days")
        print(f"=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
