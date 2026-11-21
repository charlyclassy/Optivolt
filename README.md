# Optivolt dashboard contrast update

Replace only `app.py` in the root of your existing Optivolt GitHub repository with the `app.py` in this package.

The change improves readability of disabled buttons, disabled form controls, control labels, pale alert panels, and captions in the authenticated dashboard, including Admin Settings. Disabled actions remain disabled. No forecasting models, data, thresholds, application calculations, or page navigation were changed.

Commit and push using GitHub Desktop. After Streamlit redeploys, refresh Admin Settings and check the disabled buttons and notice panels.

Validation: Python syntax compilation and static placement checks passed. A full Streamlit Cloud/browser test was not performed.
