# FMN Project 1 Forecasting Stage

## Deliverables

* `notebooks/03_lightgbm_forecasting.ipynb` — executed model development and evaluation notebook.
* `src/forecast.py` — reusable forecasting functions.
* `artifacts/evaluation/forecast_model_comparison.csv` — baseline versus LightGBM.
* `artifacts/evaluation/forecast_by_lead_time.csv` — performance by lead-time group.
* `artifacts/model/lightgbm_inventory_forecast.joblib` — final pooled LightGBM artifact.
* `artifacts/model/lightgbm_metadata.json` — model configuration and evaluation metadata.

## Model decision

Pooled LightGBM is the prototype champion with 6.52% lead-time WAPE on the untouched test period.

The weekday-adjusted 28-day moving average is retained as the transparent fallback.

## Important limitation

The assessment dataset is only six months long and is controlled/simulated in important respects. These results support the assessment prototype and should not be represented as production performance on live FMN data.
