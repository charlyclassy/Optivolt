"""Unified Optivolt demand workspace; delegates modelling to existing modules."""
import html
import streamlit as st
import depot_updates as dep

SOURCES = ('Use existing UKPN depot data', 'Upload new depot demand data', 'Update existing depot demand data')


def render(selected=None):
    st.markdown('''<div style="background:linear-gradient(110deg,#173b77,#087e89);padding:19px 24px;border-radius:17px;margin-bottom:12px;box-shadow:0 7px 20px #12345a22">
      <div style="color:white;font-size:29px;font-weight:800;line-height:1.2">⚡ Demand Prediction Workspace</div>
      <div style="color:#e4f7ff;font-size:15px;margin-top:5px">Select data → Prepare → Choose method → Predict</div></div>''',unsafe_allow_html=True)
    st.markdown('### 1 · Select data source')
    source=st.radio('Data source', SOURCES, key='dw_data_source', label_visibility='collapsed')
    names=dep.visible_depots()
    if source != SOURCES[0]:
        # Reuse the existing, tested upload / update and version-control functions.
        # The upload form owns validation, timestamp merging, and explicit model activation.
        st.info('Upload and save the readings below. Uploading does not automatically retrain or replace a saved model.')
        dep.render_upload(default_mode='Add new depot' if source == SOURCES[1] else 'Update existing depot', embedded=True)
        st.divider()
        st.markdown('### 2 · Select a depot for prediction')
    else:
        st.markdown('### 2 · Choose depot and prediction method')
    default = selected if selected in names else (names[0] if names else None)
    if not names:
        st.warning('No depot data is available. Upload a depot dataset first.')
        return
    depot=st.selectbox('Depot',names,index=names.index(default),key='dw_forecast_depot')
    # This summary is visible BEFORE the forecast form, even without a saved result.
    # It confirms that the current deployment contains the consistency update.
    st.markdown('#### Saved LSTM Forecast · Consistency Check')
    st.caption('Forecast consistency upgrade v1.2 · Shared baseline for Planning, Alerts and Reports')
    try:
        import forecast_results_store as _ov_forecast_store
        _snapshot = _ov_forecast_store.load(depot)
        if _snapshot is None:
            st.info('No validated saved forecast for this depot. Generate a new LSTM forecast below; it will become the shared reference for Planning, Alerts and Reports.')
            _save_diagnostics = _ov_forecast_store.last_save_status()
            if _save_diagnostics.get('detail'):
                st.caption('Last save diagnostic: ' + str(_save_diagnostics['detail']))
        else:
            _series = _snapshot['demand_kw']
            _ref = _ov_forecast_store.forecast_reference(_snapshot)
            _col1, _col2, _col3 = st.columns(3)
            _col1.metric('Saved LSTM peak', f'{max(_series):.2f} kW')
            _col2.metric('Forecast grid limit', f"{float(_snapshot['capacity_kw']):.2f} kW")
            _col3.metric('Forecast readings', f'{len(_series):,}')
            st.success(f'Shared forecast reference: {_ref} · {depot} · {_snapshot["model"]}. Match this reference in Planning, Alerts and Reports.')
            _storage = _ov_forecast_store.last_save_status().get('storage')
            if _storage:
                st.caption(f'Saved in: {_storage}. Local Streamlit storage is not guaranteed to survive an app restart.')
    except (ImportError, OSError, KeyError, TypeError, ValueError) as _forecast_exc:
        st.warning(f'Saved forecast status could not be checked: {_forecast_exc}')
    if source != SOURCES[0]:
        st.caption('After saving a new depot, select it here. Newly uploaded depots need a compatible trained model before forecasting.')
    st.markdown('### 3 · Demand prediction')
    try:
        model, scaler=dep.active_paths(depot)
        if not model.exists() or not scaler.exists():
            st.warning('No compatible saved LSTM is available for this depot. Train and activate a model before forecasting.')
            return
        dep.render_fast_forecast(depot)
    except Exception as exc:
        st.error(f'Prediction unavailable: {exc}')
