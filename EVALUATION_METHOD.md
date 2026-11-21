# Evaluation methodology and evidence provenance

## Source
Original Optivolt saved `.keras` weights, scaler files and `UKPN_OptimisePrime_*.csv` files from the user-provided clean repository. Evaluations use the last 15% chronological test segment of contiguous complete-day readings. Results have been verified against saved MAE values. No augmented Bexleyheath four-model experiment scores were transferred to these models.

## LSTM 1, recursive 72 hours
Ninety-six observed 15-minute intervals are the history window. Generate 288 future 15-minute predictions recursively, feeding the latest predicted demand into the next input window. Select non-overlapping 72-hour held-out windows beginning after the 85% chronological cutoff. This matches the previously saved LSTM 1 evaluation protocol.

## LSTM 2, direct 30 days
Use 60 days of history to predict 30 days of 15-minute demand directly. Select saved-model held-out windows that end beyond the 85% chronological cutoff according to the existing saved evaluation protocol. Predictions are aligned with observed target windows, including overlapping forecast windows; classification counts are **forecast-window interval outcomes** and not unique real-world breach events.

## Classification definitions
At the depot's default planning capacity C (kW):
- Actual positive when observed demand >= C; predicted positive when forecast demand >= C.
- TP = observed and predicted positive; FP = predicted positive but observed negative; FN = observed positive but prediction negative; TN = both negative.
- Precision = TP/(TP+FP), recall = TP/(TP+FN), specificity = TN/(TN+FP).
- F1 = 2TP/(2TP+FP+FN).
- Balanced accuracy = (recall+specificity)/2.
- False alarm ratio = FP/(TP+FP). **Undefined if no alerts were predicted.**
- Peak-demand MAE = the saved, model-specific mean absolute error in maximum demand per evaluation forecast window, not a daily peak error.
- Training duration = original training time; unmeasured because original logs do not record it.

## Evidence diagnostics
See `lstm_capacity_evaluations.json` for source paths, per-depot thresholds, matched test-window counts, TP/FP/FN/TN and calculated ratios, and `capacity_evaluation_audit.csv` for a flat evidence table. These results are for historical model assessment only, not evidence of verified physical grid connection limits.

## Reproduction validation
Long-term MAE and short-term MAE obtained from held-out model inference agree within 0.02 kW of each saved model's evaluation metadata. Keras 3 with a PyTorch backend was used for offline inference. For legacy LSTM 1 model loading, compatibility-only Keras initializer configuration keys were removed in a **temporary copy of the serialized model**; the original project weights and deployed model files were not modified. Do not interpret these evaluation runtimes as original training time.
