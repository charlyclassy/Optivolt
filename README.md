# Optivolt — Visible Forecast Consistency Update v1.1

This update makes the previously subtle forecast consistency changes visible in the Demand Prediction workspace, and prevents an unchanged forecast from receiving a different reference on Streamlit reruns.

## Files to replace (repository root)

- `demand_workspace.py` — NEW, prominent 'Saved LSTM Forecast · Consistency Check' area immediately below depot selection. It is visible even before generating a forecast. It shows the update marker `Forecast consistency upgrade v1.1`, a saved forecast reference, its peak, capacity and readings, or explains that no saved result exists.
- `forecast_results_store.py` — avoids overwriting an identical saved prediction when Streamlit reruns, retaining the same generated timestamp and reference; continues to validate records.
- `forecast_alerts.py` — shows the matching reference with operational capacity alerts.
- `forecast_reports.py` — shows the reference on the report screen and in report exports.
- `optivolt_lstm.py` — displays the matching saved reference on Fleet Growth & Capacity Planning; preserves the earlier 'Steps 2 and 3 visible' and Admin contrast work from the earlier patch.

The last three files are included again so the package can be installed even if the earlier Stage 1 ZIP was missed. **No `app.py` replacement is required.**

## Install

1. Back up the five existing files, especially if you have modified them since the last Optivolt Planning fix.
2. Copy all five updated `.py` files into the root of your local Optivolt GitHub repository. Do not copy this README unless you want these installation notes to replace your existing project README.
3. In GitHub Desktop, confirm **five changed Python files** are listed; commit (`Show visible forecast consistency status`) and **Push origin**.
4. In Streamlit Community Cloud, confirm your app points to the **same GitHub repository and branch**, then check the latest deployment finished successfully.
5. Open the deployed app and hard refresh (Ctrl+F5). On Demand Prediction, select Mount Pleasant. Immediately below the Depot selector, look for **Saved LSTM Forecast · Consistency Check** and **Forecast consistency upgrade v1.1**.
6. If neither heading appears, the deployed app is still using an older copy of `demand_workspace.py` — verify the deployment/repository rather than applying more CSS changes.
7. Generate a new forecast, note its reference, then compare Planning, System Alerts and Reports for the same depot.

## Verification and limitations

- `python -m py_compile` passed for all five scripts.
- An automated storage test verified that rerunning with identical forecast data keeps the same reference and changing a capacity override gives a new reference.
- A live Streamlit Cloud deployment and UI rendering have **not** been tested here.
- Existing storage uses Streamlit's local filesystem; deployments/restarts may erase records. That requires a separate persistent-storage improvement.
- `demand_workspace.py` was taken from the previously provided Optivolt repository archive and modified with the visible panel. If your deployed copy has newer unrelated functionality, merge rather than overwrite it.
