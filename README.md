# PM10 Forecasting Validation: Static and Rolling-Origin H* Evaluation

## Repository purpose
This repository contains reproducible experiments for a PM10 forecasting study focused on temporal validation protocols.
It includes static split H* computation, rolling-origin XGBoost evaluation, and rolling-origin SARIMA evaluation.
The objective is to quantify forecast skill by horizon against a persistence baseline under leakage-aware setups.
All scripts are organized to generate traceable numeric outputs for direct methodological comparison.

## Main scripts
- `h_star_computation.py`: Static chronological split experiment and baseline H* computation.
- `rolling_origin_hstar.py`: Rolling-origin monthly evaluation for XGBoost with horizon-wise skill aggregation.
- `rolling_origin_sarima.py`: Rolling-origin monthly evaluation for SARIMA with fixed initial identification and fold-based skill summaries.

## Data
The experiment uses `pm10_clean.csv` as the consolidated input dataset.

## Reproducibility workflow
- `python h_star_computation.py`
- `python rolling_origin_hstar.py`
- `python rolling_origin_sarima.py`

## Main outputs
- `results/`: static-split outputs, including H* summary artifacts and associated figure(s).
- `results_rolling/`: rolling-origin outputs for XGBoost and SARIMA (per-fold results, horizon summaries, text summaries, and plots).

## Frozen local commits
- `c6eb55f` : corrected static H* computation
- `a453e04` : rolling-origin XGBoost and SARIMA evaluation

## Notes
H* must be interpreted together with the adopted temporal validation protocol.
