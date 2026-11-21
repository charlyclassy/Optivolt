# Optivolt — shared LSTM baseline for capacity planning

## Update

Replace only `optivolt_lstm.py` in the root of the GitHub repository with the file in this ZIP. Keep the existing `app.py`, `forecast_results_store.py`, data files, models and other scripts.

## Expected workflow

1. Open Demand Prediction, select the depot, forecast start date and forecast horizon (for example 7 days), and click Generate forecast.
2. Open Fleet Growth & Capacity Planning. The saved forecast selection now initially uses the latest stored Demand Prediction start date and exact duration, including 7-day/14-day/30-day periods.
3. The LSTM baseline is the exact array of predicted demand values saved by Demand Prediction; it is not independently recalculated. Its peak must therefore match the saved forecast peak.
4. Change fleet growth, added site load, peak coincidence, managed charging, flexible charging, time shifts, surges or load caps.
5. Click **Generate planning-based demand forecast**. The adjusted output is computed from the saved baseline, without changing that baseline.
6. If you change a planning condition, click the button again to update the results. If no saved matching forecast exists, generate one in Demand Prediction; the Planning page no longer substitutes an older or independent baseline.

The capacity and 80% warning threshold for the comparison are taken from the matching saved forecast.

## Changes

- Removed the independently calculated / cached LSTM baseline fallback in Planning.
- Reads depot, period, array and capacity from the active forecast stored by Demand Prediction.
- Automatically selects the new saved forecast's start date and horizon when the saved forecast changes.
- Retained the explicit Step 3 generation button and all planning transformation controls and calculations.
- No model weights, data, forecast generation, system alerts or reports were changed.

## Deployment

In GitHub Desktop, copy `optivolt_lstm.py` into the cloned Optivolt repository, commit, and push. Wait for Streamlit to deploy, generate a new Demand Prediction forecast, then open Planning for the same depot and exact period. Verify the baseline peak matches Demand Prediction before changing a planning control.

## Verification

Python syntax compilation passed. Live Streamlit Cloud behaviour has not been verified in this environment.
