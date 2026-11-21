"""Evidence-grounded LSTM 1/LSTM 2 held-out evaluation cards.

Scores come from trained, saved per-depot models and their historical holdouts.
Classification threshold is the app's documented default planning threshold,
not an assumed contractual grid limit. No duration is invented for original
training runs without training logs.
"""
from __future__ import annotations
import csv
import html
import io
import json
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent


def load_evidence(depot: str):
    slug = depot.lower().replace(' ', '_').replace('-', '_')
    paths = [ROOT / 'models' / 'short_term_lstm_evaluation' / f'{slug}.json',
             ROOT / 'models' / 'long_term_lstm' / slug / 'metadata.json']
    try:
        combined = json.loads((ROOT / 'lstm_capacity_evaluations.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        combined = {}
    result = []
    for kind, path in zip(('LSTM 1 · Short-term', 'LSTM 2 · Long-term'), paths):
        try:
            base = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            base = {}
        measured = combined.get(depot, {}).get('short' if kind.startswith('LSTM 1') else 'long', {})
        # Reject stale/accidental data if recalculated error does not agree
        # with the stored evidence for the exact saved model.
        try:
            valid = abs(float(measured['evaluation_mae_kw']) - float(base['test_mae_kw'])) < .02
        except (KeyError, ValueError, TypeError):
            valid = False
        result.append((kind, base, measured if valid else {}))
    return result


def _number(meta, key, suffix='', scale=1.):
    try:
        n = float(meta.get(key)) * scale
        return f'{n:,.3f}{suffix}' if math.isfinite(n) else 'Record unavailable'
    except (TypeError, ValueError):
        return 'Record unavailable'


def metric_rows(base, measured):
    # These are the nine agreed measures, in the agreed order.
    issued = measured.get('risk_tp', 0) + measured.get('risk_fp', 0)
    return [
        ('MAE', _number(base, 'test_mae_kw', ' kW'), 'Average interval error'),
        ('RMSE', _number(base, 'test_rmse_kw', ' kW'), 'Penalises large errors'),
        ('R²', _number(base, 'test_r2'), 'Variance explained'),
        ('F1-score', _number(measured, 'test_f1_score', '%', 100), 'Breach-detection F1'),
        ('Balanced accuracy', _number(measured, 'test_balanced_accuracy', '%', 100), 'Recall and specificity'),
        ('Capacity-breach recall', _number(measured, 'test_breach_recall', '%', 100), 'Detected actual breaches'),
        ('False alarm ratio', 'No alerts issued' if measured and not issued else _number(measured, 'test_false_alarm_ratio', '%', 100), 'False / predicted alarms'),
        ('Peak-demand MAE', _number(base, 'test_peak_error_kw', ' kW'), 'Absolute peak error per forecast window'),
        ('Training duration', 'Not recorded', 'Original model training log unavailable'),
    ]


def render(depot: str, current_capacity_kw=None):
    import streamlit as st
    import streamlit.components.v1 as components
    records = load_evidence(depot)
    if not any(item[1] for item in records):
        st.info('Saved LSTM evaluation records are unavailable for this depot.')
        return
    st.markdown('### LSTM 1 and LSTM 2 · Nine-criteria evaluation')
    st.caption(f'{depot} · Archived held-out actual demand and predictions from each saved model; not a future forecast. The two horizons have different evaluation windows and are not a matched head-to-head test.')
    cols = st.columns(2, gap='medium')
    exports = []
    for col, (name, base, measured) in zip(cols, records):
        horizon = '72-hour recursive forecast' if 'LSTM 1' in name else '30-day direct forecast'
        risk_cap = measured.get('risk_capacity_kw')
        subtitle = (f'Capacity-risk scores evaluated at {float(risk_cap):g} kW planning threshold'
                    if risk_cap is not None else 'Capacity-risk scores require matched held-out predictions')
        values = metric_rows(base, measured)
        cells = ''.join(
            '<div class="metric"><div class="name">'+html.escape(label)+'</div>'
            '<div class="value">'+html.escape(value)+'</div>'
            '<div class="detail">'+html.escape(note)+'</div></div>'
            for label, value, note in values
        )
        footer = ''
        if measured:
            footer = ('<div class="foot">Held-out intervals: '+format(measured['risk_evaluated_points'], ',')
                      +' · Actual breaches: '+str(measured['risk_tp']+measured['risk_fn'])
                      +' · Detected: '+str(measured['risk_tp'])
                      +' · False alarms: '+str(measured['risk_fp'])+'</div>')
        markup = ('<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><style>'
                  '*{box-sizing:border-box}html,body{margin:0;padding:0;background:transparent;font-family:Inter,Arial,sans-serif}'
                  '.panel{background:#17365d;border:1px solid #3478f6;border-top:5px solid #3478f6;border-radius:16px;padding:18px;color:white}'
                  '.title{color:#fff;font-size:20px;font-weight:800}.sub{color:#d2e5fa;font-size:13px;margin-top:7px}'
                  '.scope{color:#bfdaff;font-size:12px;margin-top:9px;margin-bottom:16px}'
                  '.metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}'
                  '.metric{background:#234870;border:1px solid #486c97;padding:13px;border-radius:11px;min-height:103px}'
                  '.name{font-size:12px;font-weight:700;color:#c9dcfa}.value{font-size:19px;font-weight:800;color:#fff;margin-top:7px;overflow-wrap:anywhere}'
                  '.detail{font-size:10px;color:#c9dcfa;margin-top:5px}.foot{color:#d3e5fa;font-size:12px;margin-top:14px}'
                  '@media(max-width:440px){.metrics{grid-template-columns:1fr}.metric{min-height:80px}.value{font-size:18px}}'
                  '</style></head><body><div class="panel"><div class="title">'+html.escape(name)+'</div>'
                  '<div class="sub">'+html.escape(horizon)+'</div><div class="scope">'+html.escape(subtitle)+'</div>'
                  '<div class="metrics">'+cells+'</div>'+footer+'</div></body></html>')
        with col:
            components.html(markup, height=790, scrolling=False)
        for label, value, note in values:
            exports.append({'Depot':depot,'Model':name,'Evaluation_horizon':horizon,'Criterion':label,
                            'Result':value,'Notes':note, 'Risk_threshold_kw':risk_cap or '',
                            'Evidence':'Saved model held-out evaluation' if label in ['MAE','RMSE','R²','Peak-demand MAE'] else ('Recomputed saved-model holdout' if measured and label != 'Training duration' else 'Original training log not supplied')})
    st.caption('Risk metrics use the specified historical planning threshold, not a verified contractual limit. “No alerts issued” means the false-alarm ratio denominator is zero; it is not a measured 0% false-alarm rate. Original training durations were not present in the supplied records.')
    with st.expander('Evaluation evidence and threshold details',expanded=False):
        for name, base, measured in records:
            st.markdown(f'**{name}**')
            if measured:
                st.write(f"Capacity threshold {measured['risk_capacity_kw']:g} kW · TP {measured['risk_tp']} · FP {measured['risk_fp']} · FN {measured['risk_fn']} · TN {measured['risk_tn']} · {measured['risk_evaluated_points']:,} intervals")
            else:
                st.write('Missing or unverified matched hold-out predictions.')
        if current_capacity_kw is not None:
            for _, _, measured in records:
                if measured and abs(float(current_capacity_kw) - float(measured['risk_capacity_kw'])) > .001:
                    st.warning('Your current depot capacity differs from the threshold used for these archived risk scores. Generate a fresh risk evaluation for the changed threshold before interpreting them as current.')
                    break
    stream = io.StringIO();out = csv.DictWriter(stream,fieldnames=list(exports[0]));out.writeheader();out.writerows(exports)
    st.download_button('Download LSTM 1 and LSTM 2 nine-criteria evaluation (CSV)',stream.getvalue().encode(),
                       file_name='optivolt_saved_lstm_nine_criteria.csv',mime='text/csv',key='saved_lstm_nine_criteria_export')
