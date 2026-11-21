"""Capacity alerts calculated from the saved LSTM Demand Prediction result.

This page deliberately does not train or independently infer another forecast.
It reads the same per-organisation, per-depot snapshot created by
`depot_updates.render_fast_forecast` through `forecast_results_store.save`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import depot_updates as dep
import forecast_results_store


def evaluate_snapshot(payload):
    """Build capacity metrics from the exact forecast values used on Demand Prediction."""
    if not payload or not payload.get('timestamps') or not payload.get('demand_kw'):
        raise ValueError('No saved forecast is available.')
    demand = np.asarray(payload['demand_kw'], dtype=float)
    times = pd.DatetimeIndex(pd.to_datetime(payload['timestamps']))
    if demand.size != len(times) or not np.isfinite(demand).all():
        raise ValueError('Saved forecast contains missing or invalid demand values.')
    if not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError('Saved forecast timestamps are not chronological and unique.')
    capacity = float(payload['capacity_kw'])
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError('Saved grid capacity is invalid.')
    # The Demand Prediction workspace explicitly uses a warning at 80% capacity.
    warning = capacity * 0.8
    series = pd.Series(demand, index=times, name='LSTM predicted demand (kW)')
    breach_mask = demand >= capacity
    warning_mask = demand >= warning
    peak_position = int(np.argmax(demand))
    peak = float(demand[peak_position])
    interval_h = float(pd.Series(times).diff().dropna().median() / pd.Timedelta(hours=1)) if len(times) > 1 else .25
    if not np.isfinite(interval_h) or interval_h <= 0:
        interval_h = .25
    return {
        'series': series, 'capacity': capacity, 'warning': warning,
        'peak': peak, 'peak_time': times[peak_position],
        'headroom': capacity - peak, 'utilisation': 100 * peak / capacity,
        'breaches': int(breach_mask.sum()), 'warnings': int(warning_mask.sum()),
        'breach_hours': float(breach_mask.sum() * interval_h),
        'warning_hours': float(warning_mask.sum() * interval_h),
        'first_warning': times[int(np.flatnonzero(warning_mask)[0])] if warning_mask.any() else None,
        'first_breach': times[int(np.flatnonzero(breach_mask)[0])] if breach_mask.any() else None,
    }


def render(selected=None):
    st.title('🚨 System Alerts')
    st.caption('Capacity-risk alerts from the latest saved LSTM Demand Prediction forecast — no separate forecast is calculated.')
    names = dep.visible_depots()
    if not names:
        st.info('No depot data is available. Add a depot and generate a forecast first.')
        return

    # Demand Prediction's selection is preferred so the user sees exactly the
    # same depot on arrival, even if the portfolio sidebar was set to All 9 Depots.
    last = st.session_state.get('dw_forecast_depot')
    preferred = last if last in names else selected if selected in names else names[0]
    # Explicit selector avoids accidentally showing another depot's saved result.
    depot = st.selectbox('Depot for capacity alerts', names,
                         index=names.index(preferred), key='ov_alerts_depot')
    payload = forecast_results_store.load(depot)
    if not payload:
        st.info(f'No saved LSTM prediction is available for {depot}. Open Demand Prediction, select this depot, and generate a forecast.')
        return
    if 'lstm' not in str(payload.get('model', '')).lower():
        st.warning('The latest saved prediction is not labelled as an LSTM forecast. Run Demand Prediction with the saved LSTM model before reviewing operational alerts.')
        return
    try:
        risk = evaluate_snapshot(payload)
    except (TypeError, ValueError, OverflowError) as exc:
        st.error(f'The saved forecast cannot be used for capacity alerts: {exc}')
        return

    series = risk['series']
    generated = pd.to_datetime(payload.get('generated_at_utc'), utc=True, errors='coerce')
    generated_text = generated.strftime('%d %b %Y %H:%M UTC') if pd.notna(generated) else 'Unknown'
    st.caption(f"**Depot:** {depot} · **Active model:** {payload['model']} · "
               f"**Forecast generated:** {generated_text} · "
               f"**Forecast period:** {series.index[0]:%d %b %Y %H:%M} – {series.index[-1]:%d %b %Y %H:%M}")

    if risk['breaches']:
        st.error('Capacity breach predicted — the saved LSTM forecast reaches or exceeds the configured grid limit.')
        recommendation = ('Consider staggering charging, reducing simultaneous vehicle charging, and verifying the depot grid-capacity setting. '
                          'Review the forecast peak and timing before taking action.')
    elif risk['warnings']:
        st.warning('Capacity warning predicted — the saved LSTM forecast reaches 80% of the configured grid limit.')
        recommendation = 'Review anticipated busy periods, stagger flexible charging where possible, and continue monitoring.'
    else:
        st.success('No capacity warning predicted — the saved LSTM forecast stays below 80% of the configured grid capacity.')
        recommendation = 'No capacity intervention is indicated by this forecast. Continue routine monitoring.'

    st.subheader('Capacity Risk Summary')
    a,b,c,d = st.columns(4)
    a.metric('LSTM forecast peak demand', f"{risk['peak']:.2f} kW")
    b.metric('Available grid headroom', f"{risk['headroom']:.2f} kW")
    c.metric('Warning threshold', f"{risk['warning']:.2f} kW")
    d.metric('Grid capacity threshold', f"{risk['capacity']:.2f} kW")
    st.caption(f"Peak utilisation: {risk['utilisation']:.1f}% of grid capacity · "
               f"Busiest predicted time: {risk['peak_time']:%d %b %H:%M}")

    st.subheader('Forecast Demand and Capacity Thresholds')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name='LSTM future prediction',
                             mode='lines', line=dict(color='#087F8C', width=3)))
    fig.add_hline(y=risk['warning'], line_color='#D97706', line_dash='dot',
                  annotation_text=f"80% warning ({risk['warning']:.1f} kW)")
    fig.add_hline(y=risk['capacity'], line_color='#B91C1C', line_dash='dash',
                  annotation_text=f"Grid capacity ({risk['capacity']:.1f} kW)")
    fig.update_layout(height=430, margin=dict(l=40,r=35,t=45,b=45),
                      xaxis_title='Forecast timestamp', yaxis_title='Electricity demand (kW)',
                      plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF', hovermode='x unified',
                      legend=dict(orientation='h', y=1.12, x=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Warning and Breach Timing')
    x,y,z = st.columns(3)
    x.metric('First warning', risk['first_warning'].strftime('%d %b %H:%M') if risk['first_warning'] is not None else 'Not projected')
    y.metric('First capacity breach', risk['first_breach'].strftime('%d %b %H:%M') if risk['first_breach'] is not None else 'Not projected')
    z.metric('Total capacity breach duration', f"{risk['breach_hours']:g} hours")
    st.subheader('Operational Recommendation')
    st.write(recommendation)
    st.caption('These are predictions from the saved Demand Prediction run, not live electrical measurements. '
               'Run a new demand prediction after changing depot inputs or capacity settings to update this assessment.')
