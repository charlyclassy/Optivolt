# Optivolt Admin Settings Button Contrast

This update changes only the Streamlit Admin Settings action button appearance. Enabled administrative buttons use a dark navy background with white text. Disabled buttons use dark text on a light grey background. It also scopes styles to the seven Admin Settings action keys to avoid relying on general styles from `app.py`.

## Install

1. Copy `optivolt_lstm.py` into the root of your GitHub Optivolt repository, replacing the existing file.
2. Commit and push with GitHub Desktop.
3. Confirm the live Streamlit app is deployed from that repository/branch, then refresh.

**Note:** The file is based on the most recent `optivolt_lstm.py` supplied in the earlier two-decimal display package. If you've edited this script separately after that package, merge these changes instead of overwriting your newer work.

No modelling, data, training or forecasting calculations were changed. Python syntax was checked; visual behaviour still needs confirmation in the deployed app.
