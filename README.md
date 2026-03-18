# pm10-forecasting-validation: Multi-Step PM10 Forecasting with Rolling-Origin H*

## Repository purpose
This repository supports deployment-realistic evaluation of multi-step PM10 forecasting.
It provides reproducible experiments for static and rolling-origin temporal validation using PM10 time-series data.
The evaluation pipeline uses train-only preprocessing, persistence-relative skill, and horizon-wise H* (Operational Predictability Limit).
The focus is methodological consistency and reproducible forecast-skill estimation across horizons.

## Repository scope
- Multi-step PM10 forecasting experiments (`h=1..7`) with persistence baseline comparison.
- Rolling-origin evaluation for machine-learning (XGBoost) and classical time-series (SARIMA) models.
- Skill definition relative to persistence and horizon-based H* estimation.
- Reproducible script outputs under `results/` and `results_rolling/`.

## Main scripts
- `h_star_computation.py`: static chronological split H* computation.
- `rolling_origin_hstar.py`: rolling-origin monthly evaluation for XGBoost.
- `rolling_origin_sarima.py`: rolling-origin monthly evaluation for SARIMA.

## Data
The consolidated input dataset is `pm10_clean.csv`.

## Reproducibility
Run in this order:
- `python h_star_computation.py`
- `python rolling_origin_hstar.py`
- `python rolling_origin_sarima.py`

Main generated artifacts:
- `results/`: static-split H* tables/summary and figure.
- `results_rolling/`: rolling-origin per-fold results, horizon summaries, textual summaries, and plots for XGBoost and SARIMA.

## Associated manuscript
This repository accompanies an IJF-style manuscript on operational predictability limits in multi-step PM10 forecasting.
TODO: replace with final bibliographic citation (authors, title, venue, year, DOI) when available.

## Frozen local commits
- `c6eb55f` : corrected static H* computation
- `a453e04` : rolling-origin XGBoost and SARIMA evaluation

## Notes
Interpret H* together with the adopted temporal validation protocol and baseline definition.
