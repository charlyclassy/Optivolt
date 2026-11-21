# Optivolt — EV Depot Demand Forecasting and Grid Capacity Planning

Streamlit application with depot LSTM forecasting plus a separate, reproducible Python forecasting experiment.

## Launch the Streamlit app

```bash
pip install -r requirements.txt
streamlit run app.py
```

The default entrypoint for Streamlit Community Cloud is `app.py`. The web app uses the saved LSTM models and their matching scalers in `models/`. The `models/global_model/` XGBoost files are retained solely because legacy dashboard utilities still reference them; they are **not** the active LSTM forecast models. Some depots do not have a complete short-term LSTM pair; the app indicates availability.

Do not enable public registration with real customer data before adding production-grade account security, storage and access controls. Review `config.py` for deployment settings.

## Reproduce the forecasting experiment

The experiment script is `optivolt_experiment.py`. See `EXPERIMENT_README.md` for the methodology, scope and interpretation caveats. Its dependencies are listed separately in `experiment_requirements.txt` so they do not alter the web deployment environment.

```bash
pip install -r experiment_requirements.txt
python optivolt_experiment.py --archive "30days(1).zip" --output results --epochs-short 25 --epochs-long 20 --accuracy-tolerance-kw 1.0
```

The compact `30days(1).zip` contains the nine **original source CSVs**, using the `30days/` paths expected by the script. It excludes archived Git history and extraneous files. The same depot data also appears as root-level CSVs for the Streamlit app (normalised line endings).

The experiment creates its synthetic training samples from training-only data at execution time. Six reference 365-day synthetic CSVs are retained under `synthetic_training_data/` because they were supplied with the app repository; the experiment script does not load these reference CSVs.

## Repository structure

- `app.py`, `dashboard/`, application `.py` modules and `dashboard/assets/` — Streamlit user interface and functionality.
- `models/` — model weights, metadata and paired scalers required by the application's forecasting and legacy dashboard components.
- `data/`, nine `UKPN_OptimisePrime_*.csv` files and `synthetic_training_data/` — app data and supporting reference datasets.
- `heldout_predictions_all_models.csv`, `model_comparison_metrics.csv`, `lstm_24h_recursive_forecast.csv`, `21_lstm_training_validation_loss.png` — existing figures/data referenced by app screens.
- `optivolt_experiment.py`, `EXPERIMENT_README.md`, `experiment_requirements.txt`, `30days(1).zip` — independently reproducible experiment.

Generated results, intermediate reports, previous experimental variants, test snapshots and personal local state are intentionally excluded.
