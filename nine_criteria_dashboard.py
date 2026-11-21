"""Read-only nine-criteria model-evaluation views for Optivolt.

Only display evidence from the selected evaluation dataset. Never substitute a
score from another depot, horizon, or training run.
"""
from __future__ import annotations

import html
import io

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

CRITERIA = (
    ('Forecast accuracy', (
        ('MAE', 'MAE (kW)', 'kW', 'lower'),
        ('RMSE', 'RMSE (kW)', 'kW', 'lower'),
        ('R²', 'R²', '', 'higher'),
    )),
    ('Capacity-risk detection', (
        ('F1-score', 'F1-score (%)', '%', 'higher'),
        ('Balanced accuracy', 'Balanced accuracy (%)', '%', 'higher'),
        ('Capacity-breach recall', 'Breach recall (%)', '%', 'higher'),
        ('False alarm ratio', 'False alarm ratio (%)', '%', 'lower'),
    )),
    ('Operational performance', (
        ('Peak-demand MAE', 'Peak-demand MAE (kW)', 'kW', 'lower'),
        ('Training duration', 'Training duration (s)', 's', 'lower'),
    )),
)


def daily_peak_mae(actual, predicted, timestamps):
    """MAE of daily maximum loads on matched days, not interval MAE."""
    try:
        frame = pd.DataFrame({'timestamp': pd.to_datetime(timestamps, errors='coerce'),
                              'actual': np.asarray(actual, dtype=float),
                              'predicted': np.asarray(predicted, dtype=float)})
        frame = frame.dropna(subset=['timestamp', 'actual', 'predicted'])
        if frame.empty:
            return None
        frame['day'] = frame['timestamp'].dt.normalize()
        peaks = frame.groupby('day')[['actual', 'predicted']].max()
        return float((peaks['actual'] - peaks['predicted']).abs().mean()) if not peaks.empty else None
    except (ValueError, TypeError, AttributeError):
        return None


def depot_rows(actual, lstm_pred, baseline_pred, timestamps, capacity_kw, training_seconds=None):
    """Nine criterion records from the same held-out observations and threshold."""
    a = np.asarray(actual, dtype=float)
    l = np.asarray(lstm_pred, dtype=float)
    b = np.asarray(baseline_pred, dtype=float)
    ts = pd.to_datetime(timestamps, errors='coerce')
    valid = np.isfinite(a) & np.isfinite(l) & np.isfinite(b) & ~pd.isna(ts)
    a, l, b, ts = a[valid], l[valid], b[valid], ts[valid]
    if len(a) < 2:
        return []
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    rows = []
    for name, pred in [('Saved LSTM', l), ('Previous-day baseline', b)]:
        real_breach, predicted_breach = a >= capacity_kw, pred >= capacity_kw
        tp = int((real_breach & predicted_breach).sum())
        fp = int((~real_breach & predicted_breach).sum())
        fn = int((real_breach & ~predicted_breach).sum())
        tn = int((~real_breach & ~predicted_breach).sum())
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        specificity = tn / (tn + fp) if tn + fp else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
        rows.append({
            'Model': name,
            'MAE (kW)': float(mean_absolute_error(a, pred)),
            'RMSE (kW)': float(np.sqrt(mean_squared_error(a, pred))),
            'R²': float(r2_score(a, pred)),
            'F1-score (%)': None if f1 is None else 100 * f1,
            'Balanced accuracy (%)': None if recall is None or specificity is None else 50 * (recall + specificity),
            'Breach recall (%)': None if recall is None else 100 * recall,
            'False alarm ratio (%)': None if tp + fp == 0 else 100 * fp / (tp + fp),
            'Peak-demand MAE (kW)': daily_peak_mae(a, pred, ts),
            'Training duration (s)': training_seconds if name == 'Saved LSTM' else None,
        })
    return rows


def archived_rows(metrics_csv, predictions_csv=None):
    """Load Bexleyheath four-model experiment, calculating peak errors if possible."""
    metrics = pd.read_csv(metrics_csv)
    predictions = None
    if predictions_csv is not None and predictions_csv.exists():
        try:
            predictions = pd.read_csv(predictions_csv)
        except (OSError, ValueError, pd.errors.ParserError):
            pass
    prediction_fields = {
        'Previous-day same-time': 'baseline_kw',
        'Random Forest': 'random_forest_kw',
        'XGBoost': 'xgboost_kw',
        'LSTM': 'lstm_kw',
    }
    rows = []
    for _, item in metrics.iterrows():
        name = str(item['Model'])
        peak = None
        column = prediction_fields.get(name)
        if predictions is not None and column and {'timestamp', 'actual_kw', column}.issubset(predictions.columns):
            peak = daily_peak_mae(predictions['actual_kw'], predictions[column], predictions['timestamp'])
        def val(field, factor=1):
            number = pd.to_numeric(item.get(field), errors='coerce')
            return float(number) * factor if pd.notna(number) else None
        rows.append({
            'Model': name,
            'MAE (kW)': val('MAE_kW'),
            'RMSE (kW)': val('RMSE_kW'),
            'R²': val('R2'),
            'F1-score (%)': val('F1_Score', 100),
            'Balanced accuracy (%)': val('Balanced_Accuracy', 100),
            'Breach recall (%)': val('Breach_Recall', 100),
            'False alarm ratio (%)': val('False_Alarm_Ratio', 100),
            'Peak-demand MAE (kW)': peak,
            'Training duration (s)': val('Training_Time_Seconds') if name != 'Previous-day same-time' else None,
        })
    return rows


def _format(value, unit):
    try:
        n = float(value)
        return f'{n:,.3f}{(" " + unit) if unit else ""}' if np.isfinite(n) else 'Not evaluated'
    except (TypeError, ValueError):
        return 'Not evaluated'


def render_dashboard(rows, *, title, context, key):
    """Grouped blue cards with readable text plus downloadable real-data evidence."""
    st.markdown(f'### {title}')
    st.caption(context)
    if not rows:
        st.info('No evaluated results are available for this dataset.')
        return
    names = [row['Model'] for row in rows]
    model = st.selectbox('Evaluation model', names, key=f'{key}_model')
    selected = next(row for row in rows if row['Model'] == model)
    for group_title, definitions in CRITERIA:
        st.markdown(f'#### {group_title}')
        columns = st.columns(3 if len(definitions) == 3 else (2 if len(definitions) == 4 else 2))
        for idx, (label, metric_key, unit, direction) in enumerate(definitions):
            value = _format(selected.get(metric_key), unit)
            card = (
                '<div style="background:#173C70;border:1px solid #416FA9;border-radius:13px;'
                'padding:15px;min-height:112px;box-sizing:border-box;">'
                '<div style="font-size:13px;font-weight:650;color:#C6DAFA!important;'
                '-webkit-text-fill-color:#C6DAFA!important;">' + html.escape(label) + '</div>'
                '<div style="font-size:22px;font-weight:800;color:#FFFFFF!important;'
                '-webkit-text-fill-color:#FFFFFF!important;margin-top:7px;">' + html.escape(value) + '</div>'
                '<div style="font-size:11px;color:#C6DAFA!important;'
                '-webkit-text-fill-color:#C6DAFA!important;margin-top:5px;">' + html.escape(direction.capitalize()) + ' is better</div>'
                '</div>'
            )
            with columns[idx % len(columns)]:
                components.html('<html><head><meta name="viewport" content="width=device-width,initial-scale=1"><style>html,body{margin:0;padding:0;background:transparent;font-family:Inter,Arial,sans-serif}*{box-sizing:border-box}</style></head><body>'+card+'</body></html>', height=125, scrolling=False)
    st.caption('Not evaluated means evidence is unavailable or the denominator is undefined; it does not mean zero.')
    st.caption('Peak-demand MAE = mean absolute difference between actual and predicted daily maxima on matched held-out dates. Training time requires the original run record; the previous-day baseline requires no model training.')
    with st.expander('Compare all models across the nine criteria', expanded=False):
        columns = ['Model'] + [m[1] for _, group in CRITERIA for m in group]
        table = pd.DataFrame(rows).reindex(columns=columns)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button('Download nine-criteria evaluation (CSV)',
                           data=table.to_csv(index=False).encode('utf-8'),
                           file_name=f'optivolt_{key}_nine_criteria.csv', mime='text/csv',
                           key=f'{key}_download')


def render_archived_four_model_comparison(metrics_csv, predictions_csv):
    """Always-visible Bexleyheath archive, independent of the selected depot."""
    st.markdown('### Bexleyheath · Archived Four-Model Comparison')
    st.caption('Historical held-out Bexleyheath experiment, **not** the currently selected depot or a live forecast. Compares previous-day baseline, Random Forest, XGBoost and LSTM on the same test records.')
    try:
        records = archived_rows(metrics_csv, predictions_csv)
        if len(records) != 4:
            st.warning('The archived experiment is incomplete: four model rows are required.')
            return
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as e:
        st.warning(f'Archived Bexleyheath comparison could not be loaded: {e}')
        return
    cols = st.columns(4)
    for col, item in zip(cols, records):
        name = str(item['Model'])
        cells = [('MAE', _format(item.get('MAE (kW)'), 'kW')),
                 ('RMSE', _format(item.get('RMSE (kW)'), 'kW')),
                 ('R²', _format(item.get('R²'), ''))]
        card = ('<div style="background:#173C70;border:1px solid #3d79cf;border-radius:12px;'
                'padding:14px;min-height:172px;">'
                '<div style="font-weight:800;color:#FFFFFF!important;-webkit-text-fill-color:#FFFFFF!important;font-size:16px;margin-bottom:14px">'
                + html.escape(name) + '</div>')
        for label, val in cells:
            card += ('<div style="color:#D5E7FF!important;-webkit-text-fill-color:#D5E7FF!important;font-size:12px;margin:5px 0">'
                     + html.escape(label) + ': <strong style="color:#FFFFFF!important;-webkit-text-fill-color:#FFFFFF!important">'
                     + html.escape(val) + '</strong></div>')
        with col:
            components.html('<html><head><meta name="viewport" content="width=device-width,initial-scale=1"><style>html,body{margin:0;padding:0;background:transparent;font-family:Inter,Arial,sans-serif}*{box-sizing:border-box}</style></head><body>'+card+'</div></body></html>', height=182, scrolling=False)
    st.markdown('#### Full nine-criteria comparison · all four models')
    keys = ['Model'] + [metric[1] for _, metrics in CRITERIA for metric in metrics]
    frame = pd.DataFrame(records).reindex(columns=keys)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button('Download Bexleyheath four-model evaluation (CSV)',
                       frame.to_csv(index=False).encode('utf-8'),
                       file_name='bexleyheath_four_model_nine_criteria.csv', mime='text/csv',
                       key='bexleyheath_four_model_visible_download')
    st.caption('Peak-demand MAE uses matched daily maxima from held-out predictions. The previous-day baseline has no trainable model, so its training duration is Not evaluated.')
