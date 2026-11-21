# Optivolt — Fleet Growth & Capacity Planning restoration

This update replaces only `optivolt_lstm.py`. It is based on the version containing the successful Admin Settings contrast correction.

## Behaviour

- Planning automatically uses the latest saved LSTM forecast for the currently selected depot from Demand Prediction.
- The baseline demand series, start time, end time, horizon, and temporary capacity are read from that saved forecast. No independently calculated or older baseline is substituted.
- To work with a different forecast date or duration, first generate the matching forecast in Demand Prediction; Planning will then use that new forecast automatically.
- All planning presets and sliders remain available. Press **Generate planning-based demand forecast** to apply the selected conditions and compare the adjusted demand with the unchanged baseline.
- If no valid saved LSTM forecast is available, the app displays guidance rather than a zero-demand result.
- The Admin Settings contrast styling that was working in the previous version is retained.

## Installation

1. Extract this ZIP.
2. Replace `optivolt_lstm.py` in the root of your GitHub repository.
3. Commit and push using GitHub Desktop.
4. After deployment, generate a fresh forecast in Demand Prediction, then open Fleet Growth & Capacity Planning for the same depot.
5. Confirm the baseline peak matches Demand Prediction to two decimal places. Change a planning condition, click the Generate button, and check the adjusted peak and chart.

No trained models, datasets, forecasts, or other app files are modified by this patch. This update passed Python syntax compilation but has not been tested inside your live Streamlit Cloud deployment.
