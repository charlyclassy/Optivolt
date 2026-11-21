# Optivolt — Fleet Growth & Capacity Planning display fix

## What changed

The previous Planning page called `st.stop()` immediately after Step 1 whenever no compatible saved Demand Prediction forecast could be loaded. That prevented the remaining step headings from appearing.

This revision leaves Step 2 and Step 3 visible with a clear explanation and disabled Generate button when there is no usable saved baseline. When a valid saved LSTM forecast exists, all the original planning sliders, Generate button, transformations and comparison charts remain available. The forecast data and computation logic have not been changed.

## Installation

Replace the root-level `optivolt_lstm.py` in your Streamlit GitHub repository, commit and push. Generate a forecast in Demand Prediction for the same depot, then return to Planning.

If the workflow still reports no saved baseline, check that Demand Prediction has persisted a latest-forecast record for the same organisation and depot; this update cannot invent a missing forecast.
