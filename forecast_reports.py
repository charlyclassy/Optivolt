"""Operational LSTM reports from the saved Demand Prediction forecast.

No prediction inference is performed here. The report, CSV and PDF are based
on the same persisted snapshot as System Alerts, not legacy model statistics.
"""
from __future__ import annotations

from io import BytesIO
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

import depot_updates as dep
import forecast_results_store
from forecast_alerts import evaluate_snapshot


def build_report(payload):
    """Return report table, interval detail, and risk from one saved LSTM run."""
    if 'lstm' not in str(payload.get('model', '')).lower():
        raise ValueError('The saved forecast is not labelled as an LSTM prediction.')
    risk = evaluate_snapshot(payload)
    data = risk['series']
    generated = pd.to_datetime(payload.get('generated_at_utc'), utc=True, errors='coerce')
    generated_text = (generated.strftime('%d %b %Y %H:%M UTC')
                      if pd.notna(generated) else 'Not recorded')
    if risk['breaches']:
        status = 'Capacity breach predicted'
        recommendation = ('Review the predicted breach intervals; consider staggering charging and '
                          'verify the configured grid-capacity limit before taking action.')
    elif risk['warnings']:
        status = 'Capacity warning predicted'
        recommendation = ('Monitor the predicted high-demand intervals and consider staggering '
                          'flexible charging.')
    else:
        status = 'No capacity warning predicted'
        recommendation = ('No capacity intervention is indicated by the current forecast. '
                          'Continue routine monitoring.')
    rows = [
        ('Depot Name', str(payload.get('depot', 'Unknown'))),
        ('Forecast Reference', forecast_results_store.forecast_reference(payload)),
        ('Active Model', str(payload['model'])),
        ('Model Version', str(payload.get('model_version', 'active'))),
        ('Forecast Generated', generated_text),
        ('Forecast Start', data.index[0].strftime('%d %b %Y %H:%M')),
        ('Forecast End', data.index[-1].strftime('%d %b %Y %H:%M')),
        ('Forecast Intervals', str(len(data))),
        ('Forecast Horizon', f"{len(data) * (float(pd.Series(data.index).diff().dropna().median() / pd.Timedelta(hours=1)) if len(data)>1 else .25):g} hours"),
        ('Predicted Peak', f"{risk['peak']:.2f} kW"),
        ('Peak Forecast Time', risk['peak_time'].strftime('%d %b %Y %H:%M')),
        ('Grid Capacity Threshold', f"{risk['capacity']:.2f} kW"),
        ('Warning Threshold (80%)', f"{risk['warning']:.2f} kW"),
        ('Available Grid Headroom', f"{risk['headroom']:.2f} kW"),
        ('Peak Grid Utilisation', f"{risk['utilisation']:.2f}%"),
        ('Total Forecast Energy', f"{data.sum() * (float(pd.Series(data.index).diff().dropna().median() / pd.Timedelta(hours=1)) if len(data)>1 else .25):.2f} kWh"),
        ('Capacity Risk Status', status),
        ('First Warning', risk['first_warning'].strftime('%d %b %Y %H:%M') if risk['first_warning'] is not None else 'Not projected'),
        ('First Capacity Breach', risk['first_breach'].strftime('%d %b %Y %H:%M') if risk['first_breach'] is not None else 'Not projected'),
        ('Forecast Warning Duration', f"{risk['warning_hours']:g} hours"),
        ('Forecast Breach Duration', f"{risk['breach_hours']:g} hours"),
        ('Operational Recommendation', recommendation),
        ('Evaluation Metrics', 'Historical test metrics are not part of this saved future-forecast snapshot.'),
        ('Synthetic Data', 'Training-data provenance is not recorded in this saved forecast.'),
        ('Capacity Disclaimer', 'Application planning capacity, not a verified contracted grid connection limit.'),
    ]
    report = pd.DataFrame(rows, columns=['Report Field', 'Value'])
    forecasts = pd.DataFrame({
        'Timestamp': data.index,
        'LSTM Forecast (kW)': data.to_numpy(),
        'Warning Threshold (kW)': risk['warning'],
        'Grid Capacity (kW)': risk['capacity'],
    })
    forecasts['Capacity Status'] = np.select(
        [forecasts['LSTM Forecast (kW)'] >= risk['capacity'],
         forecasts['LSTM Forecast (kW)'] >= risk['warning']],
        ['Breach', 'Warning'], default='Within warning threshold'
    )
    return report, forecasts, risk, status


def build_pdf(report, depot):
    """PDF uses the exact report table exported to CSV and shown in the UI."""
    buff = BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=A4, leftMargin=17*mm, rightMargin=17*mm,
                            topMargin=16*mm, bottomMargin=15*mm,
                            title=f'{depot} LSTM Operational Forecast Report')
    styles = getSampleStyleSheet()
    heading = styles['Title']
    cell = ParagraphStyle('OptivoltCell', parent=styles['Normal'], fontSize=8,
                          leading=11, spaceAfter=0)
    story = [Paragraph('Optivolt — LSTM Operational Forecast Report', heading),
             Paragraph(escape(str(depot)), styles['Heading2']), Spacer(1, 9)]
    table_data = [[Paragraph('<b>Report Field</b>', cell), Paragraph('<b>Value</b>', cell)]]
    for label, value in report.itertuples(index=False, name=None):
        table_data.append([Paragraph(escape(str(label)), cell),
                           Paragraph(escape(str(value)), cell)])
    table = Table(table_data, colWidths=[60*mm, 116*mm], repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dbeafe')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f7fc')]),
        ('LINEBELOW', (0, 0), (-1, -1), .3, colors.HexColor('#d6e0eb')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    doc.build(story)
    return buff.getvalue()


def render(selected=None):
    st.title('📋 Reports')
    st.caption('Operational report from the latest saved LSTM Demand Prediction forecast. '
               'The same saved values are used by System Alerts.')
    names = dep.visible_depots()
    if not names:
        st.info('No depot is available. Add a depot and generate an LSTM prediction first.')
        return
    last = st.session_state.get('dw_forecast_depot')
    preferred = last if last in names else selected if selected in names else names[0]
    depot = st.selectbox('Depot for operational report', names,
                         index=names.index(preferred), key='ov_report_depot')
    payload = forecast_results_store.load(depot)
    if payload is None:
        st.info(f'No saved LSTM forecast is available for {depot}. '
                'Open Demand Prediction, generate a forecast, and return to Reports.')
        return
    try:
        report, forecasts, risk, status = build_report(payload)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        st.error(f'The saved forecast cannot be reported: {exc}')
        return
    st.caption(f"**Depot:** {depot} · **Active model:** {payload.get('model', 'LSTM')} · "
               f"**Forecast period:** {risk['series'].index[0]:%d %b %Y %H:%M} – "
               f"{risk['series'].index[-1]:%d %b %Y %H:%M} · "
                f"**Forecast ref:** {forecast_results_store.forecast_reference(payload)}")
    a,b,c,d = st.columns(4)
    a.metric('LSTM forecast peak', f"{risk['peak']:.2f} kW")
    b.metric('Grid capacity', f"{risk['capacity']:.2f} kW")
    c.metric('Available headroom', f"{risk['headroom']:.2f} kW")
    d.metric('Warning threshold (80%)', f"{risk['warning']:.2f} kW")
    if risk['breaches']:
        st.error(status)
    elif risk['warnings']:
        st.warning(status)
    else:
        st.success(status)
    st.subheader('Operational Asset Report Preview')
    st.dataframe(report, use_container_width=True, hide_index=True, height=560)
    st.subheader('Forecast Interval Detail')
    st.dataframe(forecasts, use_container_width=True, hide_index=True, height=330)
    stem = str(depot).strip().replace(' ', '_') + '_LSTM_Operational_Forecast_Report'
    c1,c2,c3=st.columns(3)
    c1.download_button('⬇️ Download Operational Report (CSV)',
                       data=report.to_csv(index=False).encode('utf-8'),
                       file_name=stem+'.csv', mime='text/csv', use_container_width=True)
    c2.download_button('⬇️ Download Operational Report (PDF)',
                       data=build_pdf(report, depot), file_name=stem+'.pdf',
                       mime='application/pdf', use_container_width=True)
    c3.download_button('⬇️ Download Forecast Intervals (CSV)',
                       data=forecasts.to_csv(index=False).encode('utf-8'),
                       file_name=stem+'_intervals.csv', mime='text/csv', use_container_width=True)
    st.info('Grid capacity is an application planning value, not a verified contracted grid connection limit. '
            'Results reflect the last saved forecast for this depot. Generate a new LSTM prediction '
            'after changing data or planning capacity settings.')
