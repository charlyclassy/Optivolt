# Optivolt — complete electricity-demand forecasting experiment

`optivolt_experiment.py` is **one executable Python script**, arranged into 20 numbered, commented steps. It reads **only the original depot CSV files** inside `30days(1).zip`. It does **not** read external prediction, performance-metric, plot, or trained-model files. It fits models and calculates every output during execution.

## Setup

Use Python 3.10+ and install:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost keras torch
```

Keras uses the PyTorch backend, so TensorFlow is not required.

Place the original archive `30days(1).zip` alongside the script or pass its full path. To run the complete nine-depot experiment:

```bash
python optivolt_experiment.py --archive "30days(1).zip" --output results --epochs-short 25 --epochs-long 20
```

A much faster **functional check**, not a meaningful final model evaluation:

```bash
python optivolt_experiment.py --archive "30days(1).zip" --output quick_check --epochs-short 1 --epochs-long 1 --long-depots Bexleyheath --max-short-lstm-windows 256 --max-long-lstm-windows 32
```

The default complete experiment takes substantially longer, depending on hardware. `--long-depots` lets you specify particular depots, e.g. `--long-depots Camden Dartford`. To compare long-term **real-only** training, run separately with `--no-augment-long` and a different output directory. Do not compare training and test results generated under different settings without documenting those differences.

## What the script does

- Loads all nine real depot datasets and produces a data-quality audit.
- Uses time-ordered training (70%), validation (15%), and testing (15%) boundaries.
- Generates 365 days of synthetic examples for the six historically augmented depots, sampling weekday/month-matched **training-period** days with scale and jitter; never uses validation/test days to generate synthetic samples.
- Combines **training examples** (not timestamps or real data records) with `np.concatenate()` for the LSTM direct 30-day task. Real validation and test windows stay separate. Synthetic 2030 timestamps are identifiers, not actual forecasts for 2030.
- For short-term demand, calculates the previous-day same-time baseline and trains Random Forest, XGBoost, and LSTM on Bexleyheath. The baseline uses the demand from the same time 24 hours earlier (`lag_96`), while the trained models make one-step, 15-minute-ahead forecasts; it also creates a 24-hour *recursive* LSTM forecast.
- For long-term demand, independently trains a 60-day-input/30-day-output LSTM for each of nine depots, computes held-out error and compares with repeating the prior 30 observed days (persistence).
- Computes result tables and creates 25 numbered short-term charts (the original 22 plus three additional evaluation charts), along with 30-day comparison, augmentation and depot-specific charts under `results/figures`.

## Interpretation and limitations

Numerical metrics depend on training settings, random seed, software versions, time-series handling and computing hardware. Use sufficient training epochs for a final analysis; the one-epoch command is a functionality test only.

**Source-data flags:** Some source CSV rows contain numerical `demand_kw` values with `observed=0`. This script retains those values to follow the supplied long-horizon training method. They should not automatically be described as directly observed measurements; the origin/quality of those values should be confirmed before scientific reporting. This choice may affect validation conclusions.

**Temporal windows:** The long-horizon windows overlap in time, as in the supplied 60-day/30-day training code. Split membership is determined by the *end* of the target window; some held-out windows can therefore contain earlier target days. This is a property of the original method, not a claim of fully non-overlapping evaluation periods.

**Capacity alerts:** By default the alert threshold is 17 kW, which can be changed using `--capacity-kw`. The short-term comparison concerns Bexleyheath; it must not be described as a nine-depot short-term model evaluation.

To see exactly what data enters training, read steps 05–09 for short-term and steps 16–18 for the 30-day models. Both concatenate synthetic samples **only within their own training arrays**.

## Nine evaluation criteria — calculated on every new short-term test run

The single script computes the following directly from the test-set predictions, without loading existing metrics: MAE (lower), RMSE (lower), R² (higher), F1-score (higher), balanced accuracy (higher), training time (lower), capacity-breach recall (higher), false alarm ratio (lower), and peak-demand MAE (lower).

- Capacity-breach recall = TP / (TP + FN), using the configured capacity threshold (default 17 kW).
- False alarm ratio = FP / (TP + FP), the proportion of predicted breach alerts that were incorrect. If a model predicts no alerts, the ratio is undefined (not assigned an artificial win).
- Peak-demand MAE = mean absolute error **only for test intervals where actual demand > 17 kW** (or the threshold chosen with `--capacity-kw`). If there are no such test intervals, the criterion is undefined.
- Training-time comparisons include the previous-day baseline (no fitting). For machine-learning-only time comparisons, consult the training time plot and exclude the baseline.

Additional output images: `23_capacity_breach_recall.png`, `24_false_alarm_ratio.png`, and `25_peak_demand_mae.png`. The existing `12_metric_winner_scorecard.png` now compares nine criteria. New CSVs: `short_term_nine_criteria_winners.csv` and `short_term_nine_criteria_win_counts.csv`. The primary `short_term_model_comparison.csv` also includes `PeakDemandMAE_kW`, `ActualPeakIntervals` and `PredictedAlerts`.

To run in VS Code after extracting the package, with `30days(1).zip` alongside the script:

```powershell
python optivolt_experiment.py --archive "30days(1).zip" --output results_9criteria --epochs-short 25 --epochs-long 20
```

The `results_9criteria` output directory avoids overwriting older figures and tables. The ranking and winner counts are determined from the results of this run; they are not hard-coded.

## Historical forecast accuracy — trace each prediction back to its actual test reading

**Step 13A** compares the newly calculated short-term predictions with the historical demand at each matching timestamp in the chronological **test** period. By default, a point is labelled *within tolerance* when its absolute forecast error is **at most 1.0 kW**. This is an explicit analysis threshold, not a claim that every acceptable forecast is exact. It is separate from the 17 kW threshold used for capacity-breach detection.

New figures in `results/figures`:

- `27_lstm_historical_accuracy_timeline.png` — first seven test days; actual readings, LSTM predictions, green within-tolerance markers and red outside-tolerance markers.
- `28_historical_accuracy_model_comparison.png` — percentage of test predictions within the same stated tolerance for all four approaches.
- `29_lstm_daily_historical_accuracy.png` — the LSTM's within-tolerance rate for each historical test day.

New tables in `results/tables`:

- `short_term_historical_accuracy_by_timestamp.csv` — actual value, forecast, signed and absolute error, and accuracy flag for each model at each timestamp.
- `short_term_historical_accuracy_by_model.csv` — accurate/outside-tolerance counts, proportions and MAE for each model.
- `short_term_historical_accuracy_by_day.csv` — each model's daily rate and prediction counts.
- `30day_<depot>_historical_accuracy_by_day.csv` — daily historical error and tolerance pass rate for the first complete held-out 30-day forecast window at each trained depot.

To change the tolerance, add `--accuracy-tolerance-kw 1.5` (for an absolute tolerance of 1.5 kW). For example:

```powershell
python optivolt_experiment.py --archive "30days(1).zip" --output results_accuracy --epochs-short 25 --epochs-long 20 --accuracy-tolerance-kw 1.0
```

**Important interpretation:** These are retrospective checks on held-out historical observations and are not evidence that a forecast for a future date has already been verified. All percentages are computed on this run's model predictions. A future 24-hour recursive forecast without future measurements cannot receive an accuracy flag until actual measurements become available. Some `demand_kw` values in source files have `observed=0` and should not be represented as independently measured observations without confirming the origin of those values.


## Training data counts

The experiment prints and saves historical 15-minute row counts and the actual numbers of real and synthetic model-training examples. Check `tables/short_term_training_data_counts.csv` and `tables/30day_training_data_counts.csv`. The script distinguishes raw rows from 96-reading short-term LSTM windows and 60-day input/30-day output long-term windows. Quick-test sampling is included in the actual-fitting counts.

## Final-report figure styles

The single experiment script now saves these presentation-friendly graphics using the same output filenames as before:

- `03_error_comparison.png`: MAE versus RMSE dumbbell chart (units: kW).
- `05_r2_comparison.png`: R² lollipop chart.
- `10_actual_vs_all_models_7day.png`: multi-line actual-versus-predicted comparison.
- `11_residual_boxplot.png`: model residual distributions.
- `12_metric_winner_scorecard.png`: nine-criterion winner heatmap; one model marked per metric.
- `18_lstm_actual_vs_predicted_scatter.png`: LSTM predicted-versus-actual scatter.
- `27_lstm_historical_accuracy_timeline.png`: LSTM historical accuracy time series.
- `28_historical_accuracy_model_comparison.png`: four-model horizontal dot plot; within ±tolerance accuracy.
- `29_lstm_daily_historical_accuracy.png`: LSTM daily within-tolerance accuracy line plot.

All numerical values, predictions and winner counts are calculated when the experiment runs. The program does not read saved prediction or metrics tables. Other diagnostic figures retain their existing designs. The nine-criterion winner heatmap deliberately uses binary winning status rather than colouring incomparable raw units.

Run in VS Code from the project directory (keeping `30days(1).zip` alongside `optivolt_experiment.py`):

```powershell
python optivolt_experiment.py --archive "30days(1).zip" --output results_visual_variety --epochs-short 25 --epochs-long 20 --accuracy-tolerance-kw 1.0
```

Keep in mind that the previous-day baseline does not require training, so its training time is structurally zero. It can be useful to report model-training time separately from predictive performance when interpreting scorecard wins.

## 24-hour forecast error comparison

**Step 13B** creates `30_24hour_forecast_absolute_error_comparison.png`. It selects the complete held-out historical test day with the highest **actual** demand peak, and plots aligned actual and model-predicted demand above the absolute errors for the previous-day same-time baseline, Random Forest, XGBoost, and LSTM. No previous prediction or metric files are loaded. The comparison is based on 96 consecutive 15-minute test readings.

Two CSV files are also created:

- `short_term_24hour_absolute_errors.csv`: timestamp-level actual demand, predictions and four absolute errors.
- `short_term_24hour_error_summary.csv`: 24-hour MAE, RMSE and number of forecasts within ±1 kW, for each approach.

Use the error panel to see when a particular model performs better or worse. Do not conclude that LSTM wins every 15-minute interval merely because it has the lowest error over the entire test. The graph's day is selected by the actual historical peak, not by which day makes LSTM look strongest.

Run using the same command as above; a new output directory avoids confusion with older files:

```powershell
python optivolt_experiment.py --archive "30days(1).zip" --output results_24hour_comparison --epochs-short 25 --epochs-long 20 --accuracy-tolerance-kw 1.0
```

## Enhanced model-comparison visuals for the final report

The single Python script now generates an improved seven-day comparison, together with two complementary 24-hour comparisons. All three are produced from the **predictions created in the current training run**; the script does not load saved predictions, model metrics, or plotted figures.

- `10_actual_vs_all_models_7day.png` — black, thicker **actual demand** line; red **LSTM** solid line; green dashed **Random Forest**; purple dash-dot **XGBoost**; muted orange dotted **previous-day same-time baseline**. The legend sits above the plot, with capacity threshold lines showing the default **13.6 kW warning** and **17.0 kW planning capacity** (both adjust when `--capacity-kw` changes).
- `31_four_model_peak_demand_zoom.png` — a 24-hour section of those seven test days, centred as closely as possible on the **highest actual demand peak**, with the same colours, patterns, and thresholds. The period is selected by the actual demand rather than by a model's accuracy.
- `30_24hour_forecast_absolute_error_comparison.png` — two aligned panels: actual and predicted demand above, and four models' **absolute prediction errors** below. The day is the complete held-out day with the largest observed peak, so it may differ from the day chosen within Figure 10's seven-day excerpt. Small error is better. Table outputs include `short_term_24hour_absolute_errors.csv` and `short_term_24hour_error_summary.csv`.

The script still generates the original short-term model comparisons, the full nine-criterion evaluation, the training sample counts, synthetic-data analyses and the 30-day forecasts across the selected depots. No result is preassigned to LSTM; the plots show where other models perform better too.

For VS Code, **extract the downloaded ZIP first**, then copy the included `optivolt_experiment.py` over the copy inside your `Optivolt_Project` folder. Keep `30days(1).zip` in that folder. Avoid opening the Python file directly from within the ZIP. Use a **new output folder** so old results are not mistaken for this run:

```powershell
python optivolt_experiment.py --archive "30days(1).zip" --output results_enhanced_plots --epochs-short 25 --epochs-long 20 --accuracy-tolerance-kw 1.0
```

For a quick formatting check, use `--epochs-short 1 --epochs-long 1 --long-depots Bexleyheath --max-short-lstm-windows 256 --max-long-lstm-windows 32`, keeping in mind that a one-epoch training loss does not provide a meaningful learning curve or final forecasting metrics.


## Final formatting refinements

This version makes three presentation-only refinements requested for the report:

- **Confusion matrices:** numbers on darker blue cells are now drawn in **white** for better contrast.
- **Forecast comparison legends:** the legends for the seven-day comparison, the 24-hour peak-detail comparison, and the 24-hour absolute-error comparison are moved outside the main plotting area so they do not obscure the data.
- **Chronological split figure:** `02_data_split_timeline.png` now includes the **number of rows** in the training, validation, and held-out test segments.

These are layout/clarity changes only; no modelling logic or evaluation calculations were changed.

## Depot labels in figure titles

Short-term forecasting is performed with Bexleyheath demand data. The short-term plots now show **Bexleyheath** explicitly in their titles, including the chronological split, evaluation comparisons, LSTM diagnostics, forecast accuracy, and capacity-breach confusion matrices. The nine-depot 30-day workflow and its existing depot-specific titles are unchanged. No forecasting algorithms, samples, metric formulas, or other visual styling were modified in this update.
