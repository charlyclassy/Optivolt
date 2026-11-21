# Optivolt Reports, System Alerts and Demand Prediction alignment

## What this patch changes

The **Reports** page now reads the **same latest saved depot-specific LSTM forecast** created in Demand Prediction and used by System Alerts. It does not run a separate legacy forecast or reuse legacy XGBoost metrics.

- `app.py` — routes Reports to `forecast_reports.render`, retaining the previous System Alerts synchronisation route.
- `forecast_reports.py` — new Reports page with forecast summary, capacity status, time-of-warning/breach, and consistent CSV/PDF exports and interval-level CSV.
- `forecast_alerts.py` — retained from the previous alert synchronisation patch, and used to calculate capacity risk consistently on both pages.

**No model weights, experiment code, or datasets have changed.** Historical training statistics, synthetic-data provenance, and performance metrics are **not available** from a saved future forecast snapshot and are not represented as if they were derived from it. Those should be reported separately using correctly associated model evaluation records.

## Install with GitHub Desktop

1. Extract this package.
2. Select your Optivolt repository in GitHub Desktop and choose **Repository > Show in Explorer**.
3. Copy `app.py`, `forecast_alerts.py`, and `forecast_reports.py` into the repository root, beside its existing `app.py`. Replace the existing `app.py` and `forecast_alerts.py` when prompted; `forecast_reports.py` is new.
4. In GitHub Desktop, verify these three file changes and commit with `Sync reports with LSTM demand predictions and alerts`, then **Push origin**.
5. Once Streamlit has deployed, select a depot in Demand Prediction and generate a fresh LSTM forecast. Visit System Alerts and Reports, selecting the **same depot**. The forecast peak, grid capacity, warning threshold and headroom must agree on all three screens.
6. Download the operational **CSV and PDF** reports and verify that they show the same values.

### Important behaviour

- If no saved LSTM forecast is available, Reports prompts the user to run Demand Prediction first.
- Changing capacity settings or source data does not retroactively change the last saved forecast. Generate a new prediction before reviewing updated reports or alerts.
- The forecast is a scenario-dependent estimate, not measured future load or a verified contracted connection limit.
- The original `dashboard/pages/reports.py` remains untouched because the integrated app routes its operational Reports navigation through the new module.

## Verification

Syntax checks passed for all three Python files. The report calculations and PDF rendering were checked with representative saved LSTM snapshots, including a no-alert case and an above-capacity case. The complete app has not been deployed or exercised against a live Streamlit Cloud session here.
