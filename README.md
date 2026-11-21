# Optivolt — Admin Settings contrast correction

Replace only `app.py` in the root of the deployed Optivolt GitHub repository with this file. Do not replace any model, dataset, experiment script, or other application module.

This change corrects the cause of the unreadable disabled button text: the primary-button stylesheet previously applied white text to **all** primary buttons, including disabled buttons. White text is now applied only to enabled primary buttons. Disabled buttons retain their disabled behaviour but use dark slate text (#1e293b) on a slate-grey background (#cbd5e1), with a visible border.

It also removes a duplicated late-stage style block to reduce conflicting rules. No forecasting, model, data, settings-save, or authentication logic was changed.

## Deploy

1. Copy `app.py` to the root of your cloned GitHub repository, replacing its existing copy.
2. Commit and push in GitHub Desktop.
3. Verify Streamlit Cloud is deploying that repository and branch, then reboot the app and refresh your browser using Ctrl+F5.
4. In Admin Settings, confirm the disabled Apply buttons have dark, readable labels. Edit a setting and check the enabled Apply action still has white text on the usual blue background.

Validation performed: `python -m py_compile app.py` and stylesheet source checks. Visual appearance in the deployed app has not been verified.
