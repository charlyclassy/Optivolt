# Optivolt — Fleet Growth & Capacity Planning correction

This patch updates only `optivolt_lstm.py`. The datasets, trained model files, experiment script and other application pages do not change.

## Changes

- The planning page rejects invalid, missing, non-finite, negative, or entirely zero LSTM baseline arrays. It does not misrepresent missing forecasts as zero demand.
- If a saved Demand Prediction forecast matches the depot and the exact selected dates, it can be used as the planning baseline, subject to validity checks.
- Newly generated LSTM baselines are checked before they enter the planning cache.
- Planning inputs now update the scenario results automatically when the controls change; the additional planning-generation click is no longer needed.
- The planning page displays a clear instruction if no valid LSTM baseline is available for the selected depot and date range.

## Installation in GitHub Desktop

1. Extract this ZIP.
2. In GitHub Desktop, choose your Optivolt repository and click **Repository > Show in Explorer**.
3. Copy the included `optivolt_lstm.py` into the repository root, replacing the existing file.
4. Commit and push the change, then reboot or refresh your Streamlit app.
5. In **Demand Prediction**, run a forecast for a selected depot and 72-hour period.
6. In **Fleet Growth & Capacity Planning**, select **the same depot and dates**. Verify that the LSTM baseline is non-zero.
7. Try **+20% fleet growth** and **New fixed site load** presets. The planning peak and comparison lines should now update without an extra button click.

If a valid LSTM baseline still cannot be generated, the app will display an error rather than displaying a false zero-demand result. A correct saved depot-specific model and scaler are still needed for independent planning forecasts.

## Verification

The changed Python file passed a syntax compilation check. End-to-end Streamlit deployment and model inference have not been tested in this environment.
