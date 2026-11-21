#!/usr/bin/env python3
"""OPTIVOLT — single-script electricity-demand forecasting experiment.

All reported forecasts, metrics, plots and tables are computed on execution.


Example:
 python optivolt_experiment.py --archive '30days(1).zip' --output results --epochs-short 25 --epochs-long 20

For a quick end-to-end check:
 python optivolt_experiment.py --archive '30days(1).zip' --output smoke --epochs-short 1 --epochs-long 1 --long-depots Bexleyheath
"""
# =============================================================================
# STEP 01 — IMPORT DEPENDENCIES, SET RANDOM SEED AND CONFIGURE THE EXPERIMENT
# =============================================================================
import argparse
import io
import os
import random
import time
import zipfile
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'torch')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             confusion_matrix, precision_score, recall_score,
                             f1_score, balanced_accuracy_score)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DEPOTS = ['Bexleyheath', 'Camden', 'Dartford', 'Islington', 'Mount Pleasant',
          'Orpington', 'Premier Park', 'Victoria', 'Whitechapel']
SYNTHETIC_DEPOTS = DEPOTS[:6]
INTERVALS_PER_DAY = 96
INPUT_DAYS, OUTPUT_DAYS = 60, 30
SEED = 42
FEATURES = ['lag_1', 'lag_4', 'lag_96', 'lag_672', 'rolling_4', 'rolling_96',
            'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'weekend']
MODEL_COLS = {'Previous-day same-time': 'baseline_kw', 'Random Forest': 'random_forest_kw',
              'XGBoost': 'xgboost_kw', 'LSTM': 'lstm_kw'}


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=155, bbox_inches='tight')
    plt.close(fig)


def keras_library():
    import keras
    keras.utils.set_random_seed(SEED)
    return keras


# =============================================================================
# STEP 02 — READ ALL NINE REAL DEPOT DATASETS FROM THE SOURCE ZIP ARCHIVE
# =============================================================================
def read_real(z, depot):
    name = f'30days/UKPN_OptimisePrime_{depot.replace(" ", "_")}.csv'
    if name not in z.namelist():
        raise FileNotFoundError(f'Missing raw input: {name}')
    df = pd.read_csv(io.BytesIO(z.read(name)))
    if not {'timestamp', 'demand_kw'}.issubset(df.columns):
        raise ValueError(f'{depot}: raw CSV missing timestamp/demand_kw')
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['demand_kw'] = pd.to_numeric(df['demand_kw'], errors='coerce')
    # Preserve the source demand values exactly, including rows flagged observed=0.
    # These rows contain numerical readings in this archive and the original
    # long-term training method retained them. Report their status in documentation.
    df = df.dropna(subset=['timestamp']).drop_duplicates('timestamp', keep='last')
    series = df.set_index('timestamp').sort_index()['demand_kw'].resample('15min').mean()
    series.name = 'demand_kw'
    return series


# =============================================================================
# STEP 03 — AUDIT DATA QUALITY AND IDENTIFY COMPLETE OBSERVED DAYS
# =============================================================================
def longest_complete_block(series, minimum_days=1):
    counts = series.resample('D').count()
    dates = pd.DatetimeIndex(counts[counts == INTERVALS_PER_DAY].index)
    if not len(dates):
        raise ValueError('There are no complete observed days')
    breaks = np.flatnonzero(np.diff(dates.values).astype('timedelta64[D]').astype(int) != 1) + 1
    groups = np.split(dates, breaks)
    block = max(groups, key=len)
    if len(block) < minimum_days:
        raise ValueError(f'Only {len(block)} consecutive complete days (need {minimum_days})')
    values = series.loc[block[0]: block[-1] + pd.Timedelta(hours=23, minutes=45)]
    return values.to_numpy(dtype=np.float32).reshape(-1, INTERVALS_PER_DAY), block


def audit_datasets(real, tables):
    report = []
    for depot, s in real.items():
        days, dates = longest_complete_block(s)
        report.append({'depot':depot,'first_timestamp':str(s.index.min()),
                       'last_timestamp':str(s.index.max()), 'total_15min_slots':len(s),
                       'observed_slots':int(s.notna().sum()),
                       'missing_slots':int(s.isna().sum()),
                       'longest_complete_days':len(dates),
                       'observed_mean_kw':round(float(s.mean()),4),
                       'observed_peak_kw':round(float(s.max()),4)})
    pd.DataFrame(report).to_csv(tables/'01_data_quality_all_nine_depots.csv',index=False)


# =============================================================================
# STEP 04 — SPLIT ORIGINAL OBSERVATIONS CHRONOLOGICALLY (70% / 15% / 15%)
# =============================================================================
def chronological_boundaries(series):
    valid = series.dropna()
    tr = valid.index[int(.70*len(valid))]
    va = valid.index[int(.85*len(valid))]
    return tr,va


# =============================================================================
# STEP 05 — GENERATE SYNTHETIC DAILY PROFILES FROM TRAINING-ONLY REAL DATA
# =============================================================================
def make_synthetic_days(train_days, train_dates, n_days=365, seed=SEED):
    rng = np.random.default_rng(seed)
    future_dates = pd.date_range('2030-01-01',periods=n_days,freq='D')
    profiles=[]
    for d in future_dates:
        weekday = np.flatnonzero(train_dates.dayofweek == d.dayofweek)
        month = weekday[train_dates[weekday].month == d.month]
        pool = month if len(month) else weekday if len(weekday) else np.arange(len(train_dates))
        base = train_days[rng.choice(pool)]
        scale = rng.uniform(.90,1.10)
        jitter = np.maximum(0,rng.normal(1,.04,INTERVALS_PER_DAY))
        profiles.append(np.maximum(0,base*scale*jitter))
    return np.asarray(profiles,dtype=np.float32),future_dates


# =============================================================================
# STEP 06 — CHECK SYNTHETIC DATA COMPATIBILITY WITH REAL TRAINING OBSERVATIONS
# =============================================================================
def synthetic_compatibility(real_days, synthetic_days, depot, tables, records):
    records.append({'depot':depot,'real_training_days':len(real_days),
                    'synthetic_training_days':len(synthetic_days),
                    'matching_15min_intervals':real_days.shape[1]==synthetic_days.shape[1]==96,
                    'real_mean_kw':float(real_days.mean()),
                    'synthetic_mean_kw':float(synthetic_days.mean()),
                    'real_p95_kw':float(np.quantile(real_days,.95)),
                    'synthetic_p95_kw':float(np.quantile(synthetic_days,.95)),
                    'synthetic_negative_readings':int((synthetic_days<0).sum())})


# =============================================================================
# STEP 07 — ENGINEER CAUSAL FEATURES FOR 15-MINUTE SHORT-TERM PREDICTIONS
# =============================================================================
def engineer_features(series):
    idx = series.index
    df = pd.DataFrame(index=idx)
    for n in [1,4,96,672]: df[f'lag_{n}'] = series.shift(n)
    for n in [4,96]: df[f'rolling_{n}'] = series.shift(1).rolling(n,min_periods=n).mean()
    t=idx.hour + idx.minute/60
    weekday=idx.dayofweek
    df['hour_sin']=np.sin(2*np.pi*t/24)
    df['hour_cos']=np.cos(2*np.pi*t/24)
    df['weekday_sin']=np.sin(2*np.pi*weekday/7)
    df['weekday_cos']=np.cos(2*np.pi*weekday/7)
    df['weekend']=(weekday>=5).astype(float)
    df['target']=series
    return df.dropna()


# =============================================================================
# STEP 08 — MAKE REAL TRAINING/VALIDATION/TEST WINDOWS WITHOUT DATA LEAKAGE
# =============================================================================
def short_lstm_windows(series, lookback=96):
    vals=series.to_numpy(dtype=np.float32)
    indices=[]; xx=[]; yy=[]
    # Do not bridge missing observations; accept only fully observed windows.
    for end in range(lookback,len(vals)):
        window=vals[end-lookback:end+1]
        if np.isfinite(window).all():
            xx.append(window[:-1]); yy.append(window[-1]); indices.append(series.index[end])
    return np.asarray(xx,dtype=np.float32),np.asarray(yy,dtype=np.float32),pd.DatetimeIndex(indices)


def short_synthetic_features(profiles, dates):
    # Synthetic data is independent: never stitch 2030 samples onto real chronology.
    vals = profiles.reshape(-1)
    times = pd.date_range(dates[0], periods=len(vals),freq='15min')
    return engineer_features(pd.Series(vals,index=times,name='demand_kw'))


# =============================================================================
# STEP 09 — CALCULATE PREVIOUS-DAY BASELINE AND TRAIN RF, XGBOOST AND LSTM
# =============================================================================
def train_short_term(series, synthetic_profiles, synthetic_dates, epochs, tables, figdir, capacity, max_windows=0):
    keras=keras_library()
    train_end,val_end=chronological_boundaries(series)
    feat=engineer_features(series)
    tr=feat.index<train_end
    va=(feat.index>=train_end)&(feat.index<val_end)
    te=feat.index>=val_end
    synfeat=short_synthetic_features(synthetic_profiles,synthetic_dates)
    train_features=pd.concat([feat.loc[tr,FEATURES],synfeat[FEATURES]],ignore_index=True)
    train_target=np.r_[feat.loc[tr,'target'].values,synfeat['target'].values]
    real_rows_training = int(series.loc[series.index < train_end].notna().sum())
    real_rows_validation = int(series.loc[(series.index >= train_end) & (series.index < val_end)].notna().sum())
    real_rows_testing = int(series.loc[series.index >= val_end].notna().sum())
    test=feat.loc[te].copy()
    # Previous-day same-time baseline: lag_96 is exactly 24 hours earlier
    # because the series has 96 demand observations per day (15-minute intervals).
    # This baseline is a direct reference forecast and is not model-trained.
    result=pd.DataFrame({'timestamp':test.index,'actual_kw':test.target.to_numpy(),
                         'baseline_kw':test.lag_96.to_numpy()}).reset_index(drop=True)
    fit_speeds={}
    importance={}
    for name,model,col in [
        ('Random Forest',RandomForestRegressor(n_estimators=120,min_samples_leaf=2,n_jobs=-1,random_state=SEED),'random_forest_kw'),
        ('XGBoost',XGBRegressor(n_estimators=250,max_depth=5,learning_rate=.055,
                                subsample=.9,colsample_bytree=.9,n_jobs=4,
                                objective='reg:squarederror',random_state=SEED),'xgboost_kw')]:
        tic=time.monotonic()
        model.fit(train_features,train_target)
        fit_speeds[name]=time.monotonic()-tic
        result[col]=np.maximum(0,model.predict(test[FEATURES]))
        importance[name]=pd.DataFrame({'feature':FEATURES,'importance':model.feature_importances_}).sort_values('importance')
        importance[name].to_csv(tables/f'{name.lower().replace(" ","_")}_feature_importance.csv',index=False)

    # LSTM takes the previous 96 readings to predict the next reading (15 minutes).
    x,y,times=short_lstm_windows(series)
    sm=(times<train_end); vm=(times>=train_end)&(times<val_end)
    tm=times>=val_end
    mu=float(series.loc[series.index<train_end].mean())
    sd=max(float(series.loc[series.index<train_end].std()),1e-6)
    # Synthetic LSTM sequences are created within the synthetic sequence itself.
    synth_seq=pd.Series(synthetic_profiles.reshape(-1),index=pd.date_range(synthetic_dates[0],periods=synthetic_profiles.size,freq='15min'))
    sx,sy,_=short_lstm_windows(synth_seq)
    xtr=np.concatenate([x[sm],sx],axis=0)
    ytr=np.concatenate([y[sm],sy],axis=0)
    if max_windows>0 and len(xtr)>max_windows:
        choices=np.linspace(0,len(xtr)-1,max_windows,dtype=int)
        xtr,ytr=xtr[choices],ytr[choices]
    # Report historical rows separately from engineered samples and LSTM windows.
    row_counts = [
        ('Original historical demand', 'All nonmissing records', int(series.notna().sum()), '15-minute rows'),
        ('Chronological split', 'Real training', real_rows_training, '15-minute rows'),
        ('Chronological split', 'Real validation', real_rows_validation, '15-minute rows'),
        ('Chronological split', 'Real test', real_rows_testing, '15-minute rows'),
        ('Random Forest / XGBoost', 'Real training', int(tr.sum()), 'feature rows'),
        ('Random Forest / XGBoost', 'Synthetic training', int(len(synfeat)), 'feature rows'),
        ('Random Forest / XGBoost', 'Total fitted training', int(len(train_features)), 'feature rows'),
        ('Short-term LSTM', 'Real training', int(sm.sum()), '96-observation windows'),
        ('Short-term LSTM', 'Synthetic training', int(len(sx)), '96-observation windows'),
        ('Short-term LSTM', 'Total before optional sampling', int(sm.sum()+len(sx)), '96-observation windows'),
        ('Short-term LSTM', 'Actually fitted (after sampling)', int(len(xtr)), '96-observation windows'),
        ('Short-term LSTM', 'Real validation', int(vm.sum()), '96-observation windows'),
        ('Short-term LSTM', 'Real test', int(tm.sum()), '96-observation windows')]
    report = pd.DataFrame(row_counts,columns=['Stage','Dataset','Count','Unit'])
    report.to_csv(tables/'short_term_training_data_counts.csv',index=False)
    print('\nSHORT-TERM DATA COUNTS\n'+report.to_string(index=False)+'\n',flush=True)
    model=keras.Sequential([keras.layers.Input((96,1)),keras.layers.LSTM(48),
                            keras.layers.Dense(24,activation='relu'),keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.Adam(0.001),loss='mse')
    tic=time.monotonic()
    history=model.fit(((xtr-mu)/sd)[...,None],(ytr-mu)/sd,
        validation_data=(((x[vm]-mu)/sd)[...,None],(y[vm]-mu)/sd),
        epochs=epochs,batch_size=256,shuffle=False,verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=6,restore_best_weights=True)])
    fit_speeds['LSTM']=time.monotonic()-tic
    lstm_pred=np.maximum(0,model.predict(((x[tm]-mu)/sd)[...,None],batch_size=512,verbose=0).ravel()*sd+mu)
    lp=pd.Series(lstm_pred,index=times[tm]); result['lstm_kw']=result.timestamp.map(lp)
    result=result.dropna().reset_index(drop=True)
    losses=pd.DataFrame(history.history)
    losses.insert(0,'epoch',np.arange(1,len(losses)+1))
    losses.to_csv(tables/'short_term_lstm_epoch_losses.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4));ax.plot(losses.epoch,losses.loss,marker='o',label='Train loss')
    ax.plot(losses.epoch,losses.val_loss,marker='o',label='Validation loss');ax.legend();ax.set(title='Bexleyheath — LSTM Training and Validation Loss',xlabel='Epoch',ylabel='MSE (normalised)')
    save_figure(fig,figdir/'21_lstm_training_validation_loss.png')
    pd.DataFrame([{'dataset':'real training','examples':int(sm.sum())},
                  {'dataset':'synthetic training','examples':len(sx)},
                  {'dataset':'real validation','examples':int(vm.sum())},
                  {'dataset':'real test','examples':int(tm.sum())}]).to_csv(tables/'short_term_training_integration.csv',index=False)
    return result,importance,fit_speeds,model,mu,sd,train_end,val_end


# =============================================================================
# STEP 10 — CALCULATE NINE EVALUATION CRITERIA FROM THE TEST PREDICTIONS
# =============================================================================
def calculate_metrics(pred, training_times, cap, tables):
    # A breach occurs when actual or predicted demand exceeds the planning
    # capacity threshold. The same threshold is used for all four approaches.
    actual = pred['actual_kw'].to_numpy(dtype=float)
    actual_breach = actual > cap
    peak_count = int(actual_breach.sum())
    if peak_count == 0:
        print(f'No actual test readings exceed {cap:g} kW: Peak-Demand MAE is undefined.')

    metrics = []
    for name, col in MODEL_COLS.items():
        predicted = pred[col].to_numpy(dtype=float)
        predicted_breach = predicted > cap
        tn, fp, fn, tp = confusion_matrix(
            actual_breach, predicted_breach, labels=[False, True]
        ).ravel()
        # False Alarm Ratio (false discovery rate) = FP / (TP + FP).
        # Where there are no predicted alerts, the ratio is undefined.
        alert_count = int(tp + fp)
        false_alarm_ratio = float(fp / alert_count) if alert_count else np.nan
        # Peak MAE uses the ORIGINAL actual-demand > threshold intervals only;
        # never select intervals using the model's predicted values.
        peak_mae = float(np.mean(np.abs(actual[actual_breach] - predicted[actual_breach]))) if peak_count else np.nan
        metrics.append({
            'Model': name,
            'MAE_kW': mean_absolute_error(actual, predicted),
            'RMSE_kW': np.sqrt(mean_squared_error(actual, predicted)),
            'R2': r2_score(actual, predicted),
            'Precision': precision_score(actual_breach, predicted_breach, zero_division=0),
            'Recall': recall_score(actual_breach, predicted_breach, zero_division=0),
            'F1': f1_score(actual_breach, predicted_breach, zero_division=0),
            'BalancedAccuracy': balanced_accuracy_score(actual_breach, predicted_breach),
            'FalseAlarmRatio': false_alarm_ratio,
            'PeakDemandMAE_kW': peak_mae,
            'ActualPeakIntervals': peak_count,
            'PredictedAlerts': alert_count,
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
            'TrainingSeconds': float(training_times.get(name, 0.0)),
        })
    frame = pd.DataFrame(metrics)
    frame.to_csv(tables / 'short_term_model_comparison.csv', index=False)
    pred.to_csv(tables / 'short_term_test_predictions.csv', index=False)
    return frame


# =============================================================================
# STEP 11 — DRAW HISTORICAL DEMAND AND REAL TRAIN / VALIDATION / TEST SPLIT
# =============================================================================
def plot_history(series,train_end,val_end,figdir):
    sub=series.dropna().iloc[:96*25]
    fig,ax=plt.subplots(figsize=(11,3.8));ax.plot(sub.index,sub.values,lw=.8)
    ax.set(title='Bexleyheath — Historical Electricity Demand',xlabel='Timestamp',ylabel='Demand (kW)')
    save_figure(fig,figdir/'01_historical_demand_profile.png')

    valid = series.dropna().sort_index()
    train_rows = int((valid.index < train_end).sum())
    validation_rows = int(((valid.index >= train_end) & (valid.index < val_end)).sum())
    test_rows = int((valid.index >= val_end).sum())

    start_time = valid.index.min()
    end_time = valid.index.max()

    fig,ax=plt.subplots(figsize=(13.5,4.8))
    ax.axvspan(start_time, train_end, color='tab:blue', alpha=.18, label='Training')
    ax.axvspan(train_end, val_end, color='tab:orange', alpha=.25, label='Validation')
    ax.axvspan(val_end, end_time, color='tab:green', alpha=.22, label='Test')
    ax.axvline(train_end, color='tab:blue', linestyle='--', linewidth=1.1)
    ax.axvline(val_end, color='tab:blue', linestyle='--', linewidth=1.1)

    def mid(a, b):
        return a + (b - a) / 2

    ymin, ymax = 0, 1
    ax.set_ylim(ymin, ymax)
    box = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#333', alpha=.92)
    ax.text(mid(start_time, train_end), .53, f'Training\n{train_rows:,} rows',
            ha='center', va='center', fontsize=12, fontweight='bold', bbox=box)
    ax.text(mid(train_end, val_end), .53, f'Validation\n{validation_rows:,} rows',
            ha='center', va='center', fontsize=12, fontweight='bold', bbox=box)
    ax.text(mid(val_end, end_time), .53, f'Held-out test\n{test_rows:,} rows',
            ha='center', va='center', fontsize=12, fontweight='bold', bbox=box)

    ax.set(title='Bexleyheath — Chronological Data Split: Development and Final Test', xlabel='Date', yticks=[])
    ax.spines[['left','right']].set_visible(False)
    ax.legend(loc='upper right')
    fig.text(0.01, 0.01, 'Training and validation are used for model development; the held-out test is not used for fitting.', fontsize=9)
    save_figure(fig,figdir/'02_data_split_timeline.png')


# =============================================================================
# STEP 12 — DRAW ALL NINE CRITERIA AND SELECT THE WINNER FOR EACH METRIC
# =============================================================================
def plot_comparison(metrics, figdir, tables):
    names = metrics['Model'].to_list()
    ix = np.arange(len(names))

    def bars(fname, cols, title, ylabel, percentage=False):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        width = .75 / len(cols)
        for j, col in enumerate(cols):
            values = metrics[col].to_numpy(dtype=float)
            if percentage:
                values = values * 100
            positions = ix - .375 + width * (j + .5)
            ax.bar(positions, values, width, label=col)
        ax.set_xticks(ix, names, rotation=12)
        ax.set(title=title, ylabel=ylabel)
        if len(cols) > 1:
            ax.legend(fontsize=8)
        save_figure(fig, figdir / fname)

    # Figure 03 — Dumbbell comparison: the MAE/RMSE scales are both kW.
    fig, ax = plt.subplots(figsize=(10, 5.1))
    y = np.arange(len(names))
    mae = metrics['MAE_kW'].to_numpy(dtype=float)
    rmse = metrics['RMSE_kW'].to_numpy(dtype=float)
    for i, (a, b) in enumerate(zip(mae, rmse)):
        ax.plot([a, b], [i, i], color='#a8b6c8', linewidth=2.5, zorder=1)
    ax.scatter(mae, y, color='#007f86', s=95, label='MAE', zorder=3)
    ax.scatter(rmse, y, color='#d17b27', s=95, marker='D', label='RMSE', zorder=3)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set(xlabel='Forecast error (kW; lower is better)',
           title='Bexleyheath — Forecast Error Comparison: MAE and RMSE')
    ax.legend(loc='best')
    ax.grid(axis='x', linestyle=':', alpha=.35)
    save_figure(fig, figdir / '03_error_comparison.png')
    bars('04_operational_comparison.png', ['Precision', 'Recall'], 'Bexleyheath — Capacity Alert Precision and Recall', 'Fraction')
    # Figure 05 — Lollipop chart rather than a second bar chart.
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    r2 = metrics['R2'].to_numpy(dtype=float)
    y = np.arange(len(names))
    start = min(0.0, float(np.nanmin(r2)))
    for i, value in enumerate(r2):
        ax.plot([start, value], [i, i], color='#a5b8c7', linewidth=2.5)
    ax.scatter(r2, y, s=110, color='#236d9f', zorder=3)
    spread = max(float(np.nanmax(r2) - start), .1)
    for i, value in enumerate(r2):
        ax.annotate(f'{value:.3f}', (value, i), xytext=(6, 0),
                    textcoords='offset points', va='center', fontsize=9)
    ax.set_xlim(start - spread*.05, float(np.nanmax(r2)) + spread*.17)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set(xlabel='R² (higher is better)', title='Bexleyheath — Explained Demand Variability: R²')
    ax.grid(axis='x', linestyle=':', alpha=.35)
    save_figure(fig, figdir / '05_r2_comparison.png')
    bars('06_breach_counts.png', ['TP', 'FP', 'FN'], 'Bexleyheath — Capacity Breach Classification Counts', 'Number of test samples')
    bars('07_f1_score.png', ['F1'], 'Bexleyheath — Capacity Breach F1 Score', 'F1 score')
    bars('08_balanced_accuracy.png', ['BalancedAccuracy'], 'Bexleyheath — Capacity Breach Balanced Accuracy', 'Balanced accuracy')
    bars('09_training_time.png', ['TrainingSeconds'], 'Bexleyheath — Measured Training Time', 'Seconds')

    # Three additional, independently labelled charts.
    bars('23_capacity_breach_recall.png', ['Recall'], 'Bexleyheath — Capacity-Breach Recall', 'Recall (%) — higher is better', percentage=True)
    bars('24_false_alarm_ratio.png', ['FalseAlarmRatio'], 'Bexleyheath — False Alarm Ratio (FP / All Predicted Alerts)', 'False alarms (%) — lower is better', percentage=True)
    bars('25_peak_demand_mae.png', ['PeakDemandMAE_kW'], 'Bexleyheath — Peak-Demand Forecast Error (Actual > Capacity)', 'Peak-demand MAE (kW) — lower is better')

    # Nine evaluation criteria, direction of improvement, and display label.
    criteria = [
        ('MAE_kW', True, 'Lowest MAE'),
        ('RMSE_kW', True, 'Lowest RMSE'),
        ('R2', False, 'Highest R²'),
        ('F1', False, 'Highest F1-score'),
        ('BalancedAccuracy', False, 'Highest balanced accuracy'),
        ('TrainingSeconds', True, 'Fastest training'),
        ('Recall', False, 'Highest breach recall'),
        ('FalseAlarmRatio', True, 'Lowest false alarm ratio'),
        ('PeakDemandMAE_kW', True, 'Lowest peak-demand MAE'),
    ]
    winners = []
    for metric, lower_is_better, label in criteria:
        available = metrics[['Model', metric]].dropna()
        if available.empty:
            winning_model, winning_value = 'Not applicable', np.nan
        else:
            best = available.sort_values(
                [metric, 'Model'], ascending=[lower_is_better, True]
            ).iloc[0]
            winning_model = str(best['Model'])
            winning_value = float(best[metric])
        winners.append({'Criterion': label, 'MetricColumn': metric,
                        'Direction': 'Lower' if lower_is_better else 'Higher',
                        'WinningModel': winning_model, 'WinningValue': winning_value})
    winner_table = pd.DataFrame(winners)
    winner_table.to_csv(tables / 'short_term_nine_criteria_winners.csv', index=False)
    counts = (winner_table.loc[winner_table.WinningModel != 'Not applicable', 'WinningModel']
              .value_counts().reindex(names, fill_value=0)
              .rename_axis('Model').reset_index(name='CriteriaWon'))
    counts['TotalCriteria'] = len(criteria)
    counts.to_csv(tables / 'short_term_nine_criteria_win_counts.csv', index=False)

    # Figure 12 — 9 × 4 winner heatmap. This shows winning status only;
    # raw metric values have different units and should not share a colour scale.
    winner_grid = np.zeros((len(winners), len(names)), dtype=float)
    for row_index, winner in enumerate(winners):
        if winner['WinningModel'] in names:
            winner_grid[row_index, names.index(winner['WinningModel'])] = 1.0
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    from matplotlib.colors import ListedColormap
    ax.imshow(winner_grid, aspect='auto', interpolation='nearest',
              cmap=ListedColormap(['#edf1f4', '#176a89']), vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(names)), names, rotation=18, ha='right')
    ax.set_yticks(np.arange(len(winners)), [w['Criterion'] for w in winners])
    for row_index in range(winner_grid.shape[0]):
        for col_index in range(winner_grid.shape[1]):
            if winner_grid[row_index, col_index] == 1:
                ax.text(col_index, row_index, 'WIN', ha='center', va='center',
                        color='white', fontsize=9, fontweight='bold')
    ax.set_xticks(np.arange(-.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(winners), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_title('Bexleyheath — Nine-Criterion Model Winner Scorecard')
    save_figure(fig, figdir / '12_metric_winner_scorecard.png')
    print('Nine-criterion winner counts:')
    print(counts.to_string(index=False))


# =============================================================================
# STEP 13 — PLOT SEVEN-DAY PREDICTIONS, COMPLETE TEST SERIES AND RESIDUALS
# =============================================================================
# Model colours and line patterns remain identical in all comparison figures.
# Black = actual demand; red = LSTM; green = Random Forest;
# purple = XGBoost; muted orange = previous-day same-time baseline.
FORECAST_STYLES = {
    'Previous-day same-time': dict(color='#bc8b52', linestyle=':', linewidth=1.3, alpha=.74),
    'Random Forest': dict(color='#2ca02c', linestyle='--', linewidth=1.55, alpha=.87),
    'XGBoost': dict(color='#9467bd', linestyle='-.', linewidth=1.55, alpha=.87),
    'LSTM': dict(color='#d62728', linestyle='-', linewidth=2.05, alpha=.95),
}


def add_capacity_reference(ax, capacity_kw):
    """Show the warning and planning thresholds, not evaluation metrics."""
    warning_kw = .8 * capacity_kw
    ax.axhline(warning_kw, color='#2c6f9c', linestyle=(0, (4, 3)),
               linewidth=1, alpha=.7, zorder=0,
               label=f'Warning threshold ({warning_kw:g} kW)')
    ax.axhline(capacity_kw, color='#2c6f9c', linestyle=(0, (7, 3)),
               linewidth=1.2, alpha=.9, zorder=0,
               label=f'Planning capacity ({capacity_kw:g} kW)')


def draw_model_lines(ax, chunk, capacity_kw, thresholds=True):
    """Draw the same test timestamps for the observed series and every forecast."""
    ax.plot(chunk.timestamp, chunk.actual_kw, color='#161616',
            linewidth=2.65, alpha=.96, zorder=6, label='Actual demand')
    # Controlled draw order keeps the actual series visible despite overlaps.
    for name, col in MODEL_COLS.items():
        ax.plot(chunk.timestamp, chunk[col], label=name,
                zorder=4 if name == 'LSTM' else (2 if name == 'Previous-day same-time' else 3),
                **FORECAST_STYLES[name])
    if thresholds:
        add_capacity_reference(ax, capacity_kw)
    ax.set_ylabel('Electricity demand (kW)')
    ax.grid(axis='both', alpha=.19, linewidth=.7)
    ax.spines[['top', 'right']].set_visible(False)


def plot_prediction_diagnostics(pred, figdir, capacity_kw=17.0):
    # Figure 10: seven consecutive days from the chronological test set.
    seven = pred.iloc[:7 * INTERVALS_PER_DAY].copy()
    fig, ax = plt.subplots(figsize=(14, 5.7))
    draw_model_lines(ax, seven, capacity_kw)
    ax.set(title='Bexleyheath — Actual Demand and Four Forecasting Approaches: Seven Test Days',
           xlabel='Historical test timestamp')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, 1.12),
              ncol=4, fontsize=8.3, frameon=False)
    fig.subplots_adjust(top=.74)
    fig.savefig(figdir / '10_actual_vs_all_models_7day.png', dpi=155, bbox_inches='tight')
    plt.close(fig)

    # Figure 31: detailed 24-hour view around the largest observed demand peak.
    if len(seven) >= INTERVALS_PER_DAY:
        peak_pos = int(np.nanargmax(seven.actual_kw.to_numpy(dtype=float)))
        first = max(0, min(peak_pos - INTERVALS_PER_DAY // 2,
                           len(seven) - INTERVALS_PER_DAY))
        zoom = seven.iloc[first:first + INTERVALS_PER_DAY]
        fig, ax = plt.subplots(figsize=(13.2, 5.4))
        draw_model_lines(ax, zoom, capacity_kw)
        peak_time = seven.iloc[peak_pos].timestamp
        ax.axvline(peak_time, color='#555', lw=1, alpha=.45, linestyle=':')
        ax.set(title='Bexleyheath — Forecast Comparison: 24-Hour Peak Detail',
               xlabel='Historical test timestamp')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, 1.13),
                  ncol=4, fontsize=8.2, frameon=False)
        fig.subplots_adjust(top=.72)
        fig.savefig(figdir / '31_four_model_peak_demand_zoom.png', dpi=155, bbox_inches='tight')
        plt.close(fig)

    residuals=[pred.actual_kw-pred[col] for col in MODEL_COLS.values()]
    fig,ax=plt.subplots(figsize=(9,4));ax.boxplot(residuals,labels=list(MODEL_COLS))
    ax.set(title='Bexleyheath — Held-out Test Residuals',ylabel='Actual − Predicted (kW)')
    save_figure(fig,figdir/'11_residual_boxplot.png')
    fig,ax=plt.subplots(figsize=(6,5));ax.scatter(pred.actual_kw,pred.lstm_kw,s=6,alpha=.3)
    top=max(pred.actual_kw.max(),pred.lstm_kw.max());ax.plot([0,top],[0,top],'k--',lw=1)
    ax.set(title='Bexleyheath — LSTM: Actual vs Predicted',xlabel='Actual (kW)',ylabel='Predicted (kW)')
    save_figure(fig,figdir/'18_lstm_actual_vs_predicted_scatter.png')
    resid=(pred.actual_kw-pred.lstm_kw).to_numpy()
    lo,hi=np.quantile(resid,[.01,.99]);resid=resid[(resid>=lo)&(resid<=hi)]
    fig,ax=plt.subplots(figsize=(8,4));ax.hist(resid,bins=55)
    ax.set(title='Bexleyheath — LSTM Residual Distribution (Central 98%)',xlabel='Actual − Predicted (kW)',ylabel='Count')
    save_figure(fig,figdir/'19_lstm_residual_distribution_central_98.png')
    fig,ax=plt.subplots(figsize=(12,4));ax.plot(pred.timestamp,pred.actual_kw,label='Actual',lw=.8)
    ax.plot(pred.timestamp,pred.lstm_kw,label='LSTM',lw=.75)
    ax.legend();ax.set(title='Bexleyheath — LSTM Predictions Across the Held-out Test',ylabel='Demand (kW)')
    save_figure(fig,figdir/'20_lstm_heldout_actual_vs_predicted_full.png')


# =============================================================================
# STEP 13B — COMPARE MODEL FORECAST ERRORS OVER ONE HISTORICAL 24-HOUR PERIOD
# =============================================================================
def plot_24hour_absolute_error(pred, figdir, tables):
    """Compare aligned 15-minute prediction errors on a complete held-out day.

    Select the complete test day with the highest *actual* demand peak.
    The selection depends only on historical target readings, not on a model's
    prediction accuracy, to avoid choosing a day that favours any model.
    """
    frame = pred[['timestamp', 'actual_kw', *MODEL_COLS.values()]].copy()
    frame['timestamp'] = pd.to_datetime(frame['timestamp'])
    frame = frame.sort_values('timestamp').drop_duplicates('timestamp')
    frame['date'] = frame['timestamp'].dt.floor('D')

    complete_days = []
    expected_interval = pd.Timedelta(minutes=15)
    for date, day in frame.groupby('date'):
        if len(day) != INTERVALS_PER_DAY:
            continue
        if not day['timestamp'].diff().iloc[1:].eq(expected_interval).all():
            continue
        if not day[['actual_kw', *MODEL_COLS.values()]].notna().all().all():
            continue
        complete_days.append((float(day['actual_kw'].max()), date, day.copy()))
    if not complete_days:
        print('No complete 24-hour day with aligned predictions: skipping 24-hour error comparison.')
        return

    _, chosen_date, day = max(complete_days, key=lambda item: item[0])
    absolute_errors = {}
    for name, column in MODEL_COLS.items():
        absolute_errors[name] = np.abs(day['actual_kw'].to_numpy() - day[column].to_numpy())
        day[f'{column}_absolute_error_kw'] = absolute_errors[name]

    # Both tables use only the held-out 15-minute actual values and forecasts.
    detail_cols = ['timestamp', 'actual_kw', *MODEL_COLS.values(),
                   *(f'{col}_absolute_error_kw' for col in MODEL_COLS.values())]
    day[detail_cols].to_csv(tables / 'short_term_24hour_absolute_errors.csv', index=False)
    ranked = pd.DataFrame([{
        'Model': name,
        'Date': chosen_date.date().isoformat(),
        'Intervals': len(day),
        'MAE_kW': float(values.mean()),
        'RMSE_kW': float(np.sqrt(np.mean(values**2))),
        'Intervals_Within_1kW': int((values <= 1.0).sum()),
    } for name, values in absolute_errors.items()]).sort_values('MAE_kW')
    ranked.to_csv(tables / 'short_term_24hour_error_summary.csv', index=False)

    # Top panel: demand context. Bottom panel: absolute error at matching times.
    fig, (ax_demand, ax_error) = plt.subplots(
        2, 1, figsize=(13, 7.5), sharex=True,
        gridspec_kw={'height_ratios': [1, 1.35]}
    )
    ax_demand.plot(day['timestamp'], day['actual_kw'], color='black', lw=2.2,
                   label='Actual demand', zorder=5)
    for name, col in MODEL_COLS.items():
        ax_demand.plot(day['timestamp'], day[col], lw=1.25,
                       color=FORECAST_STYLES[name]['color'], linestyle=FORECAST_STYLES[name]['linestyle'],
                       alpha=0.9, label=name)
        ax_error.plot(day['timestamp'], absolute_errors[name],
                      lw=1.5 if name != 'LSTM' else 2.0,
                      color=FORECAST_STYLES[name]['color'], linestyle=FORECAST_STYLES[name]['linestyle'],
                      label=f'{name} (MAE {absolute_errors[name].mean():.2f} kW)',
                      alpha=.9)
    ax_demand.set_ylabel('Demand (kW)')
    ax_demand.set_title(f'Bexleyheath — Actual vs Predicted Demand: 24 Hours ({chosen_date.date()})')
    demand_handles, demand_labels = ax_demand.get_legend_handles_labels()
    ax_demand.legend(demand_handles, demand_labels, ncol=3, loc='lower center',
                     bbox_to_anchor=(.5, 1.08), fontsize=8, frameon=False)
    ax_error.set_ylabel('Absolute error (kW) — lower is better')
    ax_error.set_xlabel('Historical test timestamp')
    error_handles, error_labels = ax_error.get_legend_handles_labels()
    ax_error.legend(error_handles, error_labels, ncol=1, fontsize=8, frameon=False,
                    loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
    ax_demand.grid(alpha=.2)
    ax_error.grid(alpha=.2)
    fig.subplots_adjust(top=.82, right=.78, hspace=.28)
    fig.autofmt_xdate(rotation=20)
    fig.savefig(figdir / '30_24hour_forecast_absolute_error_comparison.png', dpi=155, bbox_inches='tight')
    plt.close(fig)
    print(f'24-hour forecast-error comparison: {chosen_date.date()}\n'
          + ranked[['Model', 'MAE_kW', 'RMSE_kW']].to_string(index=False), flush=True)


# =============================================================================
# STEP 13A — VERIFY FORECAST ACCURACY AGAINST ACTUAL HISTORICAL TEST READINGS
# =============================================================================
def plot_historical_accuracy(pred, figdir, tables, tolerance_kw=1.0):
    """Check freshly trained models on the same unseen historical test readings.

    An individual forecast is 'within tolerance' when
    abs(forecast_kw - actual_kw) <= tolerance_kw.  The tolerance is a user-
    configured reporting criterion, not a universal definition of accuracy.
    None of these labels are used to select training examples or train models.
    """
    if tolerance_kw <= 0:
        raise ValueError('Accuracy tolerance must be strictly positive (in kW).')
    needed = {'timestamp', 'actual_kw', *MODEL_COLS.values()}
    missing = needed.difference(pred.columns)
    if missing:
        raise ValueError(f'Missing test prediction columns: {sorted(missing)}')
    test = pred.copy()
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    test = test.sort_values('timestamp').reset_index(drop=True)
    if test.empty:
        raise ValueError('Historical forecast accuracy cannot be computed on an empty test set.')

    detail = []
    summary = []
    daily_frames = []
    for name, col in MODEL_COLS.items():
        error = test[col].to_numpy(dtype=float) - test.actual_kw.to_numpy(dtype=float)
        absolute_error = np.abs(error)
        within = absolute_error <= tolerance_kw
        detail.append(pd.DataFrame({
            'timestamp': test.timestamp,
            'model': name,
            'actual_kw': test.actual_kw.to_numpy(),
            'forecast_kw': test[col].to_numpy(),
            'signed_error_kw': error,
            'absolute_error_kw': absolute_error,
            'within_tolerance': within,
        }))
        summary.append({
            'Model': name,
            'Tolerance_kW': tolerance_kw,
            'Evaluated_Intervals': len(within),
            'Accurate_Intervals': int(within.sum()),
            'Outside_Tolerance_Intervals': int((~within).sum()),
            'Within_Tolerance_Pct': float(100 * within.mean()),
            'MAE_kW': float(np.mean(absolute_error)),
        })
        grouped = pd.DataFrame({'date': test.timestamp.dt.date, 'accurate': within})
        by_day = grouped.groupby('date', sort=True)['accurate'].agg(['sum', 'count']).reset_index()
        by_day['model'] = name
        by_day['within_tolerance_pct'] = 100 * by_day['sum'] / by_day['count']
        daily_frames.append(by_day.rename(columns={'sum': 'accurate_intervals', 'count': 'total_intervals'}))

    pd.concat(detail, ignore_index=True).to_csv(tables / 'short_term_historical_accuracy_by_timestamp.csv', index=False)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(tables / 'short_term_historical_accuracy_by_model.csv', index=False)
    daily = pd.concat(daily_frames, ignore_index=True)
    daily.to_csv(tables / 'short_term_historical_accuracy_by_day.csv', index=False)

    # Figure 27: seven-day LSTM timeline, with accurately and inaccurately
    # predicted intervals highlighted against real held-out observations.
    week = test.iloc[:7 * INTERVALS_PER_DAY].copy()
    week_error = np.abs(week.lstm_kw.to_numpy() - week.actual_kw.to_numpy())
    accurate_week = week_error <= tolerance_kw
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(week.timestamp, week.actual_kw, color='black', linewidth=1.0, label='Actual historical demand')
    ax.plot(week.timestamp, week.lstm_kw, color='royalblue', linewidth=.7, alpha=.65, label='LSTM prediction')
    ax.scatter(week.timestamp[accurate_week], week.lstm_kw[accurate_week],
               s=9, color='forestgreen', alpha=.7, label=f'Within ±{tolerance_kw:g} kW')
    ax.scatter(week.timestamp[~accurate_week], week.lstm_kw[~accurate_week],
               s=11, color='crimson', alpha=.8, label='Outside tolerance')
    ax.set(title=f'Bexleyheath — LSTM Historical Forecast Accuracy (First 7 Days)',
           xlabel='Historical test timestamp', ylabel='Demand (kW)')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    save_figure(fig, figdir / '27_lstm_historical_accuracy_timeline.png')

    # Figure 28 — Horizontal dot plot of a single, clearly defined accuracy
    # measure, evaluated against the same historical test timestamps.
    fig, ax = plt.subplots(figsize=(10, 4.8))
    rates = summary_df['Within_Tolerance_Pct'].to_numpy(dtype=float)
    models = summary_df['Model'].to_list()
    y = np.arange(len(models))
    ax.hlines(y, xmin=0, xmax=rates, color='#c4cfd8', linewidth=2)
    ax.scatter(rates, y, color='#157c8e', s=115, zorder=3)
    for i, rate in enumerate(rates):
        ax.annotate(f'{rate:.1f}%', (rate, i), xytext=(7, 0),
                    textcoords='offset points', va='center')
    ax.set_xlim(0, 112)
    ax.set_yticks(y, models)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle=':', alpha=.35)
    ax.set(title=f'Bexleyheath — Historical Forecast Accuracy: Within ±{tolerance_kw:g} kW',
           xlabel='Held-out predictions within tolerance (%)')
    save_figure(fig, figdir / '28_historical_accuracy_model_comparison.png')

    # Figure 29: day-by-day breakdown for the LSTM, to show WHEN accuracy
    # improved or deteriorated instead of hiding it in one aggregate metric.
    lstm_daily = daily.loc[daily['model'] == 'LSTM'].copy()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(pd.to_datetime(lstm_daily.date), lstm_daily.within_tolerance_pct,
            marker='.', markersize=4, linewidth=1)
    ax.set_ylim(0, 103)
    ax.set(title=f'Bexleyheath — LSTM Daily Historical Accuracy (±{tolerance_kw:g} kW)',
           xlabel='Historical test date', ylabel='Within-tolerance forecasts (%)')
    save_figure(fig, figdir / '29_lstm_daily_historical_accuracy.png')
    print('Historical test accuracy (within ±%.2f kW):' % tolerance_kw)
    print(summary_df[['Model', 'Accurate_Intervals', 'Evaluated_Intervals', 'Within_Tolerance_Pct']].to_string(index=False))
    return summary_df


# =============================================================================
# STEP 14 — PLOT TRAINED TREE MODEL FEATURE IMPORTANCES AND CONFUSION MATRICES
# =============================================================================
def plot_importances_confusion(pred,importance,cap,figdir):
    for name,n in [('Random Forest',13),('XGBoost',14)]:
        df=importance[name]
        fig,ax=plt.subplots(figsize=(8,4));ax.barh(df.feature,df.importance)
        ax.set(title=f'Bexleyheath — {name} Feature Importance',xlabel='Importance')
        save_figure(fig,figdir/f'{n:02d}_{name.lower().replace(" ","_")}_feature_importance.png')
    for name,n in [('Random Forest',15),('XGBoost',16),('LSTM',17)]:
        truth=pred.actual_kw.to_numpy()>cap
        forecast=pred[MODEL_COLS[name]].to_numpy()>cap
        cm=confusion_matrix(truth,forecast,labels=[False,True])
        fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(cm,cmap='Blues')
        threshold = cm.max() / 2.0
        for (i,j),v in np.ndenumerate(cm):
            color = 'white' if v > threshold else 'black'
            ax.text(j,i,str(v),ha='center',va='center',fontsize=14,color=color,fontweight='bold')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['No breach','Breach'],
               yticklabels=['No breach','Breach'],xlabel='Predicted',ylabel='Actual',
               title=f'Bexleyheath — {name} Capacity Breach Confusion Matrix')
        save_figure(fig,figdir/f'{n:02d}_{name.lower().replace(" ","_")}_confusion_matrix.png')


# =============================================================================
# STEP 15 — FORECAST THE NEXT 24 HOURS RECURSIVELY WITH THE TRAINED SHORT LSTM
# =============================================================================
def recursive_24hour(series,model,mu,sd,figdir,tables):
    recent=series.dropna().iloc[-96:].to_numpy(dtype=np.float32).copy()
    # Forecast one 15-minute step, append that prediction, and predict again.
    out=[]
    for _ in range(96):
        v=np.asarray(((recent-mu)/sd)[None,:,None],dtype=np.float32)
        y=float(model.predict(v,verbose=0)[0,0]*sd+mu)
        out.append(max(0,y));recent=np.r_[recent[1:],out[-1]]
    stamp=pd.date_range(series.dropna().index[-1]+pd.Timedelta(minutes=15),periods=96,freq='15min')
    pd.DataFrame({'timestamp':stamp,'forecast_kw':out}).to_csv(tables/'24_hour_recursive_lstm_forecast.csv',index=False)
    fig,ax=plt.subplots(figsize=(11,4));history=series.dropna().iloc[-96:]
    ax.plot(history.index,history.values,label='Most recent observations')
    ax.plot(stamp,out,label='Recursive 24-hour prediction');ax.legend()
    ax.set(title='Bexleyheath — 24-Hour LSTM Recursive Forecast',ylabel='Demand (kW)')
    save_figure(fig,figdir/'22_lstm_24h_recursive_forecast.png')


# =============================================================================
# STEP 16 — BUILD 60-DAY INPUT / 30-DAY TARGET WINDOWS FOR ALL NINE DEPOTS
# =============================================================================
def long_windows(days):
    xx=[];yy=[];ends=[]
    n=len(days)-INPUT_DAYS-OUTPUT_DAYS+1
    for i in range(max(0,n)):
        xx.append(days[i:i+INPUT_DAYS]);yy.append(days[i+INPUT_DAYS:i+INPUT_DAYS+OUTPUT_DAYS].reshape(-1))
        ends.append(i+INPUT_DAYS+OUTPUT_DAYS-1)
    return np.asarray(xx,dtype=np.float32),np.asarray(yy,dtype=np.float32),np.asarray(ends)


# =============================================================================
# STEP 17 — ADD SYNTHETIC WINDOWS TO REAL TRAINING WINDOWS ONLY
# =============================================================================
def combine_long_training(x_real,y_real,x_synth,y_synth):
    # Appending training examples does not rewrite or insert dates in real history.
    return np.concatenate([x_real,x_synth]),np.concatenate([y_real,y_synth])


# =============================================================================
# STEP 18 — TRAIN AN INDEPENDENT 30-DAY DIRECT LSTM FOR EACH DEPOT
# =============================================================================
def train_30day_depot(depot,series,epochs,batch_size,augmented,tables,figdir,compatibility,max_windows=0,accuracy_tolerance_kw=1.0):
    keras=keras_library()
    days,dates=longest_complete_block(series,minimum_days=180)
    x,y,ends=long_windows(days)
    train_cut=int(len(days)*.70);val_cut=int(len(days)*.85)
    tr=ends<train_cut; va=(ends>=train_cut)&(ends<val_cut);te=ends>=val_cut
    if min(tr.sum(),va.sum(),te.sum())<1:raise ValueError(f'{depot}: insufficient split windows')
    mean=float(days[:train_cut].mean());std=max(float(days[:train_cut].std()),1e-6)
    xtr,ytr=x[tr],y[tr];synthetic_count=0
    if augmented and depot in SYNTHETIC_DEPOTS:
        syn,synthetic_dates=make_synthetic_days(days[:train_cut],dates[:train_cut],seed=SEED+DEPOTS.index(depot))
        synthetic_compatibility(days[:train_cut],syn,depot,tables,compatibility)
        sx,sy,_=long_windows(syn)
        synthetic_count=len(sx)
        xtr,ytr=combine_long_training(xtr,ytr,sx,sy)
    if max_windows>0 and len(xtr)>max_windows:
        choices=np.linspace(0,len(xtr)-1,max_windows,dtype=int)
        xtr,ytr=xtr[choices],ytr[choices]
    # A long-term training example is one 60-day input and 30-day target window.
    long_counts = {
        'Depot': depot,
        'Rows_In_Complete_Historical_Block': int(days.size),
        'Real_Training_Days': int(train_cut),
        'Real_Training_Rows_15min': int(train_cut*INTERVALS_PER_DAY),
        'Real_Training_Windows': int(tr.sum()),
        'Synthetic_Training_Days': 365 if augmented and depot in SYNTHETIC_DEPOTS else 0,
        'Synthetic_Training_Windows': int(synthetic_count),
        'Combined_Windows_Before_Sampling': int(tr.sum()+synthetic_count),
        'Windows_Actually_Fitted': int(len(xtr)),
        'Real_Validation_Windows': int(va.sum()),
        'Real_Test_Windows': int(te.sum())}
    long_counts_path = tables/'30day_training_data_counts.csv'
    if long_counts_path.exists():
        previous = pd.read_csv(long_counts_path)
        previous = previous.loc[previous.Depot != depot]
        long_count_frame = pd.concat([previous,pd.DataFrame([long_counts])],ignore_index=True)
    else:
        long_count_frame = pd.DataFrame([long_counts])
    long_count_frame.to_csv(long_counts_path,index=False)
    print('\n30-DAY DATA COUNTS — '+depot+'\n'+pd.DataFrame([long_counts]).to_string(index=False)+'\n',flush=True)
    model=keras.Sequential([keras.layers.Input((INPUT_DAYS,INTERVALS_PER_DAY)),
                            keras.layers.LSTM(48),keras.layers.Dense(256,activation='relu'),
                            keras.layers.Dropout(.2),keras.layers.Dense(OUTPUT_DAYS*INTERVALS_PER_DAY)])
    model.compile(optimizer=keras.optimizers.Adam(.001),loss='mse',metrics=['mae'])
    tic=time.monotonic()
    hist=model.fit((xtr-mean)/std,(ytr-mean)/std,
        validation_data=((x[va]-mean)/std,(y[va]-mean)/std),
        epochs=epochs,batch_size=batch_size,shuffle=False,verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=7,restore_best_weights=True)])
    seconds=time.monotonic()-tic
    predicted=np.maximum(0,model.predict((x[te]-mean)/std,batch_size=16,verbose=0)*std+mean)
    actual=y[te]
    persistence=x[te,-OUTPUT_DAYS:,:].reshape(-1,OUTPUT_DAYS*INTERVALS_PER_DAY)
    mae=float(np.mean(np.abs(predicted-actual)))
    rmse=float(np.sqrt(np.mean((predicted-actual)**2)))
    baseline=float(np.mean(np.abs(persistence-actual)))
    slug=depot.lower().replace(' ','_')
    history=pd.DataFrame(hist.history);history.insert(0,'epoch',np.arange(1,len(history)+1))
    history.to_csv(tables/f'30day_{slug}_training_history.csv',index=False)
    # Show the first complete 30-day future trajectory from an unseen test window.
    start=int(np.flatnonzero(te)[0]); target_start=dates[start+INPUT_DAYS]
    stamp=pd.date_range(target_start,periods=OUTPUT_DAYS*INTERVALS_PER_DAY,freq='15min')
    forecast=pd.DataFrame({'timestamp':stamp,'actual_kw':actual[0],
                           'lstm_kw':predicted[0],'persistence_kw':persistence[0]})
    forecast.to_csv(tables/f'30day_{slug}_heldout_forecast.csv',index=False)
    # Historical 30-day accuracy for the first held-out forecast window.
    # Each point is compared with the actual value from the same timestamp.
    forecast['date'] = forecast.timestamp.dt.date
    forecast['absolute_error_kw'] = (forecast.lstm_kw - forecast.actual_kw).abs()
    forecast['within_tolerance'] = forecast.absolute_error_kw <= accuracy_tolerance_kw
    by_day = forecast.groupby('date', sort=True).agg(
        intervals=('within_tolerance', 'size'),
        within_tolerance_intervals=('within_tolerance', 'sum'),
        daily_mae_kw=('absolute_error_kw', 'mean')
    ).reset_index()
    by_day['within_tolerance_pct'] = 100 * by_day.within_tolerance_intervals / by_day.intervals
    by_day['tolerance_kw'] = accuracy_tolerance_kw
    by_day.to_csv(tables/f'30day_{slug}_historical_accuracy_by_day.csv', index=False)
    fig,ax=plt.subplots(figsize=(12,4));ax.plot(stamp,actual[0],lw=.65,label='Observed')
    ax.plot(stamp,predicted[0],lw=.7,label='30-day LSTM forecast')
    ax.plot(stamp,persistence[0],lw=.65,alpha=.6,label='30-day persistence')
    ax.legend(ncol=3);ax.set(title=f'{depot} — Direct 30-Day Demand Forecast',ylabel='Demand (kW)')
    save_figure(fig,figdir/f'26_30day_forecast_{slug}.png')
    return {'Depot':depot,'LSTM_MAE_kW':mae,'LSTM_RMSE_kW':rmse,
            'Persistence_MAE_kW':baseline,'Test_Windows':int(te.sum()),
            'Real_Training_Windows':int(tr.sum()),'Synthetic_Training_Windows':synthetic_count,
            'Validation_Windows':int(va.sum()),'Training_Seconds':seconds}


# =============================================================================
# STEP 19 — SUMMARISE NINE-DEPOT RESULTS AND COMPARE REAL/SYNTHETIC TRAINING
# =============================================================================
def plot_long_summary(rows,compatibility,tables,figdir):
    df=pd.DataFrame(rows)
    df.to_csv(tables/'30day_all_depots_test_metrics.csv',index=False)
    frame=pd.DataFrame(compatibility)
    if len(frame):frame.to_csv(tables/'synthetic_training_compatibility.csv',index=False)
    if df.empty:return
    fig,ax=plt.subplots(figsize=(11,4));ix=np.arange(len(df));w=.38
    ax.bar(ix-w/2,df.LSTM_MAE_kW,w,label='LSTM');ax.bar(ix+w/2,df.Persistence_MAE_kW,w,label='Persistence')
    ax.set_xticks(ix,df.Depot,rotation=35,ha='right');ax.set(ylabel='Held-out MAE (kW)',title='30-Day LSTM vs Persistence Across Depots')
    ax.legend();save_figure(fig,figdir/'24_30day_lstm_vs_persistence_all_depots.png')
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(df.Depot,df.Real_Training_Windows,label='Real')
    ax.bar(df.Depot,df.Synthetic_Training_Windows,bottom=df.Real_Training_Windows,label='Synthetic')
    ax.tick_params(axis='x',rotation=35);ax.set(title='Real and Synthetic 30-Day Training Windows',ylabel='Training windows')
    ax.legend();save_figure(fig,figdir/'23_real_synthetic_training_integration.png')


# =============================================================================
# STEP 20 — RUN THE COMPLETE WORKFLOW AND WRITE ALL FIGURES AND RESULT TABLES
# =============================================================================
def main():
    p=argparse.ArgumentParser(description='Single-script Optivolt forecasting experiment')
    p.add_argument('--archive',type=Path,required=True,help='ZIP archive of the nine original depot CSVs')
    p.add_argument('--output',type=Path,default=Path('optivolt_output'))
    p.add_argument('--epochs-short',type=int,default=25)
    p.add_argument('--epochs-long',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--capacity-kw',type=float,default=17.0)
    p.add_argument('--accuracy-tolerance-kw',type=float,default=1.0,
                   help='Absolute error threshold (kW) for historical accuracy figures and tables; default 1.0')
    p.add_argument('--long-depots',nargs='+',choices=DEPOTS,default=DEPOTS,
                   help='Which depots to train long-horizon models for (default all nine)')
    p.add_argument('--max-short-lstm-windows',type=int,default=0,help='Optional quick-test sampling; 0 trains on all available windows')
    p.add_argument('--max-long-lstm-windows',type=int,default=0,help='Optional quick-test sampling; 0 trains on all available windows')
    p.add_argument('--no-augment-long',action='store_true',help='Train 30-day models on real-only data')
    args=p.parse_args()
    random.seed(SEED);np.random.seed(SEED)
    figures=args.output/'figures';tables=args.output/'tables'
    figures.mkdir(parents=True,exist_ok=True);tables.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.archive) as z:
        real={depot:read_real(z,depot) for depot in DEPOTS}
    audit_datasets(real,tables)
    print('Loaded nine original depot datasets.')
    s=real['Bexleyheath']
    days,dates=longest_complete_block(s,minimum_days=180)
    train_cut=int(len(days)*.70)
    syn,syn_dates=make_synthetic_days(days[:train_cut],dates[:train_cut])
    comp=[];synthetic_compatibility(days[:train_cut],syn,'Bexleyheath',tables,comp)
    # Short-horizon training and evaluation use Bexleyheath.
    predictions,importance,times,model,mu,sd,tr,va=train_short_term(
        s,syn,syn_dates,args.epochs_short,tables,figures,args.capacity_kw,args.max_short_lstm_windows)
    metrics=calculate_metrics(predictions,times,args.capacity_kw,tables)
    plot_history(s,tr,va,figures)
    plot_comparison(metrics,figures,tables)
    plot_prediction_diagnostics(predictions,figures,args.capacity_kw)
    plot_24hour_absolute_error(predictions,figures,tables)
    plot_historical_accuracy(predictions,figures,tables,args.accuracy_tolerance_kw)
    plot_importances_confusion(predictions,importance,args.capacity_kw,figures)
    recursive_24hour(s,model,mu,sd,figures,tables)
    print('Short-term evaluation complete.\n',metrics[['Model','MAE_kW','RMSE_kW','R2']].to_string(index=False))
    long_rows=[]
    for depot in args.long_depots:
        print(f'\nTraining direct 30-day LSTM: {depot}')
        long_rows.append(train_30day_depot(depot,real[depot],args.epochs_long,
            args.batch_size,not args.no_augment_long,tables,figures,comp,args.max_long_lstm_windows,args.accuracy_tolerance_kw))
    plot_long_summary(long_rows,comp,tables,figures)
    print('\nCompleted. Figures:',figures,'\nResult tables:',tables)


if __name__=='__main__':main()
