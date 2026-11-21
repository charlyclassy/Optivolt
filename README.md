# Optivolt contrast update

Replace only `app.py` in your GitHub repository with the file in this package.

This version injects the final accessible disabled-button styling **after** the selected page renders. This fixes a CSS cascade problem where page-specific rules could override the previous patch. It styles disabled labels as dark text on grey with a visible border. It does not enable disabled buttons or change the app's forecasting logic.

Commit and push through GitHub Desktop, allow Streamlit to redeploy, and hard-refresh the browser (Ctrl+F5). Confirm that the app is deploying the new repository/branch. Buttons that are disabled because there are no unsaved changes will remain disabled, but their labels should be readable.

The patch passed Python syntax validation. Actual rendering still requires verification in the deployed app.
