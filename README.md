# Optivolt Planning — NameError fix

This patch fixes the undefined `_planning_model_source` variable in the status message around line 6492 of `optivolt_lstm.py`.

The status message now displays the known model source (`_borrowed_from` when a transfer model is explicitly selected, or `_planning_depot` otherwise).

**Only this one line was changed.** Demand Prediction baseline reuse, all slider settings, scenario calculations, model weights, and other application files are unchanged.

## Install

1. Extract this ZIP.
2. Copy `optivolt_lstm.py` into the root of your cloned GitHub repository, replacing the current version.
3. Commit and push with GitHub Desktop.
4. Wait for Streamlit Cloud to deploy, then refresh and test Fleet Growth & Capacity Planning.

The file has passed Python syntax compilation. A full live Streamlit deployment has not been tested.
