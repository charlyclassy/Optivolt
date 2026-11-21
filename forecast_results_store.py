"""Per-organisation, per-depot snapshot of the latest generated demand forecast.

Results are written only after successful inference. No inference happens when reading.
"""
import json
import hashlib
import math
from html import escape
import os
from pathlib import Path
import pandas as pd


def forecast_reference(payload):
    """Deterministic short reference for the *contents* of a forecast snapshot.

    Works with existing saved files; does not depend on sessions or file paths.
    A different threshold, horizon, timestamp or predicted value changes the ID.
    """
    fields = ("depot", "model", "model_version", "capacity_kw",
              "generated_at_utc", "timestamps", "demand_kw")
    canonical = json.dumps({key: payload.get(key) for key in fields},
                           sort_keys=True, ensure_ascii=True,
                           separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12].upper()


def validate_snapshot(payload, depot=None):
    """Validate a saved 15-minute demand forecast before any page uses it."""
    if not isinstance(payload, dict):
        raise ValueError("Forecast record is not an object.")
    if not payload.get("depot") or (depot is not None and payload["depot"] != depot):
        raise ValueError("Forecast depot does not match the selected depot.")
    if not payload.get("model"):
        raise ValueError("Forecast model is missing.")
    timestamps = payload.get("timestamps")
    values = payload.get("demand_kw")
    if not isinstance(timestamps, list) or not isinstance(values, list) or not timestamps:
        raise ValueError("Forecast timestamps or demand readings are missing.")
    if len(timestamps) != len(values):
        raise ValueError("Forecast timestamps and demand readings have different lengths.")
    times = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="raise"))
    if times.hasnans or not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError("Forecast timestamps are invalid or out of order.")
    if len(times) > 1 and not (times[1:] - times[:-1] == pd.Timedelta(minutes=15)).all():
        raise ValueError("Forecast must contain consecutive 15-minute readings.")
    try:
        demand = [float(v) for v in values]
        capacity = float(payload["capacity_kw"])
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("Forecast demand or capacity is not numeric.") from exc
    if not all(math.isfinite(v) and v >= 0 for v in demand):
        raise ValueError("Forecast contains negative or non-finite demand.")
    if not math.isfinite(capacity) or capacity <= 0:
        raise ValueError("Forecast grid capacity is invalid.")
    return payload


def _session_snapshots():
    """Shared across Streamlit pages within the same logged-in browser session."""
    import streamlit as st
    return st.session_state.setdefault('optivolt_saved_forecast_snapshots', {})


def _session_key(depot):
    import depot_updates as dep
    return f'{dep._scope()}::{depot.casefold().strip()}'


def _path(depot):
    import depot_updates as dep
    return dep.depot_dir(depot) / 'latest_forecast.json'


def save(depot, prediction, capacity, model_label, model_version=None):
    """Persist timestamped predictions; never share results between depots."""
    series = pd.Series(prediction, dtype=float).sort_index()
    if series.empty or series.isna().any():
        raise ValueError('Cannot save empty or invalid forecast.')
    payload = {
        'depot': depot,
        'model': model_label,
        'model_version': str(model_version or 'active'),
        'capacity_kw': float(capacity),
        'generated_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'timestamps': [pd.Timestamp(t).isoformat() for t in series.index],
        'demand_kw': [float(v) for v in series.to_numpy()],
    }
    validate_snapshot(payload, depot)
    # Streamlit reruns the Demand Prediction page on every widget interaction.
    # If a rerun displays the exact same forecast, do not overwrite its record
    # or generate a different reference. This is NOT a new prediction.
    existing = load(depot)
    if existing is not None:
        comparable = ("depot", "model", "model_version", "capacity_kw", "timestamps", "demand_kw")
        if all(existing.get(field) == payload.get(field) for field in comparable):
            return existing
    # Make the forecast immediately accessible to the other pages in this
    # browser session, even if Streamlit Cloud's local disk is not writable.
    _session_snapshots()[_session_key(depot)] = payload
    try:
        path = _path(depot)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload), encoding='utf-8')
        os.replace(tmp, path)
        _save_status()['storage'] = 'session and local file'
        _save_status()['detail'] = ''
    except OSError as exc:
        _save_status()['storage'] = 'browser session only'
        _save_status()['detail'] = str(exc)
    return payload


def _save_status():
    import streamlit as st
    return st.session_state.setdefault('optivolt_forecast_save_status', {})


def last_save_status():
    """Diagnostic status for the currently logged-in Streamlit session."""
    return dict(_save_status())


def load(depot):
    # The currently generated result always wins over an older file snapshot.
    # Session storage is scoped to both the organisation and the depot.
    try:
        session_payload = _session_snapshots().get(_session_key(depot))
        if session_payload is not None:
            return validate_snapshot(session_payload, depot)
    except (ValueError, KeyError, TypeError, OSError):
        pass
    path = _path(depot)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        validated = validate_snapshot(payload, depot)
        _session_snapshots()[_session_key(depot)] = validated
        return validated
    except (OSError, ValueError, KeyError, TypeError):
        return None


def render(depot, heading='Latest forecast results'):
    import streamlit as st
    import plotly.graph_objects as go
    payload = load(depot)
    st.subheader(heading)
    if not payload:
        st.info(f'No future demand forecast is saved for {depot}. The historical LSTM model and its evaluation results may still be available.')
        return False
    times = pd.to_datetime(payload['timestamps'])
    series = pd.Series(payload['demand_kw'], index=times)
    capacity = float(payload['capacity_kw'])
    exceed = int((series > capacity).sum())
    above_warning = int((series >= capacity * .8).sum())
    st.caption(f"{depot} · {payload['model']} · version {payload.get('model_version', 'active')} · generated {payload.get('generated_at_utc', 'unknown')[:16].replace('T',' ')} UTC · Forecast ref: {forecast_reference(payload)}")
    peak_at = series.idxmax()
    # Local, responsive styling: do not change the Overview page's other widgets.
    forecast_metrics = [
        ('Highest expected demand', f'{series.max():.2f} kW'),
        ('Grid capacity', f'{capacity:.2f} kW'),
        ('Maximum predicted exceedance', f'{max(0., float(series.max())-capacity):.2f} kW'),
        ('Time above capacity', f'{exceed*.25:g} hours'),
    ]
    cards = ''.join(
        '<div class="optivolt-forecast-metric">'
        f'<div class="optivolt-forecast-label">{escape(label)}</div>'
        f'<div class="optivolt-forecast-value">{escape(value)}</div>'
        '</div>'
        for label, value in forecast_metrics
    )
    st.markdown(
        '<style>'
        '.optivolt-forecast-metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin:12px 0 18px}'
        '.optivolt-forecast-metric{box-sizing:border-box;min-width:0;min-height:152px;padding:24px 22px;'
        'border-radius:18px;background:linear-gradient(145deg,#11264A,#234A78);'
        'border:1px solid #355D8C;box-shadow:0 10px 25px rgba(17,38,74,.13)}'
        '.optivolt-forecast-label{color:#C5D9F5;font-size:16px;line-height:1.4;overflow-wrap:anywhere}'
        '.optivolt-forecast-value{color:#FFFFFF;font-size:clamp(25px,2.6vw,43px);font-weight:750;'
        'line-height:1.25;margin-top:16px;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}'
        '@media(max-width:1100px){.optivolt-forecast-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}}'
        '@media(max-width:550px){.optivolt-forecast-metrics{grid-template-columns:1fr}}'
        '</style><div class="optivolt-forecast-metrics">' + cards + '</div>',
        unsafe_allow_html=True,
    )
    st.caption(f'Forecast peak expected: {peak_at:%d %b %Y at %H:%M} · Forecast window: {times.min():%d %b %Y %H:%M} to {times.max():%d %b %Y %H:%M}')
    if exceed:
        st.warning(f'Predicted demand could exceed the depot grid limit for {exceed * .25:g} hours. '
                   f'At its busiest, the depot may need {max(0., float(series.max()) - capacity):.2f} kW '
                   'more than the configured capacity. Consider staggering vehicle charging, '
                   'reducing peak charging, and checking the configured grid limit. '
                   'This is a forecast, not a measured overload.')
    elif above_warning:
        st.info(f'Predicted demand stays within grid capacity, but reaches at least 80% of '
                f'the limit for {above_warning * .25:g} hours. Consider monitoring busy charging periods.')
    else:
        st.success('Predicted demand stays below 80% of the configured grid capacity during this forecast.')
    st.caption(f'Total predicted electricity use: {series.sum() * .25:.1f} kWh. '
               'These results are saved predictions, not live meter readings.')
    fig = go.Figure()
    fig.add_scatter(x=series.index, y=series.values, mode='lines', name='Predicted demand', line=dict(color='#087F8C', width=3))
    fig.add_hline(y=capacity, line_dash='dash', line_color='#DC2626', annotation_text='Grid capacity')
    fig.add_hline(y=capacity * .8, line_dash='dot', line_color='#D97706', annotation_text='80% warning')
    fig.update_layout(height=330, font=dict(size=15,color='#14243B'), paper_bgcolor='#FFFFFF', plot_bgcolor='#F7FAFF',
                      margin=dict(l=50,r=30,t=30,b=50), xaxis_title='Date and time', yaxis_title='Demand (kW)',
                      xaxis=dict(gridcolor='#E2E8F0'), yaxis=dict(gridcolor='#E2E8F0'), hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    table = pd.DataFrame({'Date and time':times, 'Predicted demand (kW)':series.values.round(2),
                          'Grid capacity status':['Exceeded' if v>capacity else 'Within limit' for v in series.values]})
    with st.expander('Forecast values and grid capacity', expanded=False):
        st.dataframe(table, hide_index=True, use_container_width=True, height=370,
                     column_config={'Date and time':st.column_config.DatetimeColumn('Date and time',format='DD MMM YYYY, HH:mm'),
                                    'Predicted demand (kW)':st.column_config.NumberColumn('Predicted demand (kW)',format='%.2f'),
                                    'Grid capacity status':st.column_config.TextColumn('Grid status')})
    st.download_button('Download forecast CSV', table.to_csv(index=False), file_name=f'{depot.replace(" ","_")}_forecast.csv', mime='text/csv')
    return True
