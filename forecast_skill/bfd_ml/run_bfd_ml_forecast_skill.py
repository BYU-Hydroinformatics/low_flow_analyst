#!/usr/bin/env /Users/amin/miniconda3/bin/python3
# -*- coding: utf-8 -*-
"""Compare BFS forecast skill to ML-identified baseflow-dominant periods.

Uses the Random Forest BFD model (bfd_ciroh) to flag baseflow-dominant days,
then evaluates pybfs forecast skill on those periods. Saves per-gage metrics
to bfd_ml/output/metrics.csv. Supports resume after interruption.

Data sources:
  - Streamflow:  ../../../usgs_daily_streamflow/<site_no>.csv
  - Calibration: ../../../usgs_calibration_results/params/params_<site_no>.csv
  - Gage list:   ../../../usgs_daily_streamflow/site_info.csv
  - BFD model:   /Users/amin/Downloads/research/projects/bfd_ciroh/ml_model/
"""

import os
import sys
import traceback
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import joblib
import numpy as np
import pandas as pd
import pybfs
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from baseflow.skill import forecast_skill as _orig_forecast_skill

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
LFA_ROOT    = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

STREAMFLOW_DIR = os.path.join(LFA_ROOT, 'usgs_daily_streamflow')
PARAMS_DIR     = os.path.join(LFA_ROOT, 'usgs_calibration_results', 'params')
SITE_INFO_CSV  = os.path.join(STREAMFLOW_DIR, 'site_info.csv')
OUTPUT_DIR     = os.path.join(SCRIPT_DIR, 'output')

BFD_MODEL_DIR  = '/Users/amin/Downloads/research/projects/bfd_ciroh/ml_model'
MODEL_PATH     = os.path.join(BFD_MODEL_DIR, 'random_forest_bfd_model.joblib')
SCALER_PATH    = os.path.join(BFD_MODEL_DIR, 'feature_scaler.joblib')

os.makedirs(OUTPUT_DIR, exist_ok=True)

CFS_TO_M3_PER_DAY = 0.0283168 * 86400

# Features expected by the ML model (order matters)
BFD_FEATURES = [
    'streamflow/Mean',
    'Mean_streamflow',
    'MW5_d2streamflowabs',
    'MW5_streamflow',
    'MW5_dstreamflowabs',
    'r10m',
    'streamflow',
    'streamflow/Chapman',
    'Months',
]

# ── ML model (loaded once) ────────────────────────────────────────────────────
_model  = joblib.load(MODEL_PATH)
_scaler = joblib.load(SCALER_PATH)


# ── Chapman baseflow filter ────────────────────────────────────────────────────
def _chapman(q_series, alpha=0.925):
    vals = q_series.dropna().reset_index(drop=True).tolist()
    if not vals:
        return []
    bf = [vals[0]]
    for cur, prev in zip(vals[1:], vals[:-1]):
        bf.append(
            ((3 * alpha - 1) / (3 - alpha)) * bf[-1]
            + ((1 - alpha) / (3 - alpha)) * (cur + prev)
        )
    return bf


# ── Feature extraction (mirrors create_model.py process_file) ─────────────────
def _extract_features(df_raw):
    """Return DataFrame with BFD_FEATURES columns (and 'date') for ML prediction.

    Input df_raw must have columns 'date' (datetime) and 'streamflow' (cfs).
    Rows with NaN / Inf are dropped; caller must re-align with original dates.
    """
    df = df_raw[['date', 'streamflow']].copy()
    df['streamflow'] = pd.to_numeric(df['streamflow'], errors='coerce')
    df['streamflow'] = df['streamflow'].clip(lower=1e-10).replace(0, 1e-10)

    df['Chapman'] = _chapman(df['streamflow'])
    df['streamflow/Chapman'] = df['streamflow'] / df['Chapman'].replace(0, np.nan)

    # Mean excluding 2-sigma outliers
    m, s = df['streamflow'].mean(), df['streamflow'].std()
    mask_out = (df['streamflow'] > m + 2 * s) | (df['streamflow'] < m - 2 * s)
    mean_sf = df.loc[~mask_out, 'streamflow'].mean()
    if pd.isna(mean_sf) or mean_sf == 0:
        mean_sf = df['streamflow'].mean()
    if pd.isna(mean_sf) or mean_sf == 0:
        mean_sf = 1e-10

    df['Mean_streamflow'] = mean_sf
    df['streamflow/Mean'] = df['streamflow'] / mean_sf

    df['dstreamflow']     = df['streamflow'].diff()
    df['dstreamflow_abs'] = df['dstreamflow'].abs()
    df['d2streamflow']    = df['dstreamflow'].diff()
    df['d2streamflow_abs'] = df['d2streamflow'].abs()

    df['MW5_streamflow']      = df['streamflow'].rolling(5, min_periods=1).mean()
    df['MW5_dstreamflowabs']  = df['dstreamflow_abs'].rolling(5, min_periods=1).mean()
    df['MW5_d2streamflowabs'] = df['d2streamflow_abs'].rolling(5, min_periods=1).mean()

    df['Months'] = df['date'].dt.month

    df['streamflow_monthly'] = df.groupby(df['date'].dt.month)['streamflow'].transform('mean')
    P10 = df['streamflow'].quantile(0.1)
    if P10 == 0 or pd.isna(P10):
        P10 = 1e-10
    df['r10m'] = df['streamflow_monthly'] / P10

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=BFD_FEATURES + ['date'])
    return df[['date'] + BFD_FEATURES]


# ── BFD mask using ML model ────────────────────────────────────────────────────
def bfd_ml_mask(hydrograph_cfs):
    """Return boolean array aligned to hydrograph_cfs index.

    True  = ML model predicts this day is baseflow-dominant.
    False = not baseflow-dominant (or missing data).

    hydrograph_cfs must have columns 'date' (datetime) and 'streamflow' (cfs).
    """
    feat_df = _extract_features(hydrograph_cfs)
    if feat_df.empty:
        return np.zeros(len(hydrograph_cfs), dtype=bool)

    X = feat_df[BFD_FEATURES]
    X_scaled = _scaler.transform(X)
    preds = _model.predict(X_scaled)          # 0 or 1

    # Build a date→prediction lookup, then map back to full index
    pred_map = dict(zip(feat_df['date'].dt.normalize(), preds))
    mask = np.zeros(len(hydrograph_cfs), dtype=bool)
    dates_norm = hydrograph_cfs['date'].dt.normalize()
    for i, d in enumerate(dates_norm):
        mask[i] = bool(pred_map.get(d, 0))
    return mask


# ── Forecast skill with BFD-ML mask ───────────────────────────────────────────
def bfd_ml_forecast_skill(hydrograph_m3, hydrograph_cfs, SBT, basin_char, gw_hyd, flow,
                          min_days=30, max_days=90, train_days=365):
    """Run forecast skill assessment using ML-identified BFD periods.

    Parameters
    ----------
    hydrograph_m3  : DataFrame with 'Date' (datetime) and 'Streamflow' (m³/day)
    hydrograph_cfs : DataFrame with 'date' (datetime) and 'streamflow' (cfs)
    SBT, basin_char, gw_hyd, flow : same as pybfs.bfs / pybfs.forecast
    min_days  : minimum sequence length to qualify
    max_days  : maximum forecast horizon (sequences are truncated)
    train_days: days of prior data used to initialize the forecast

    Returns
    -------
    metrics : dict  {overall_RMSE, overall_MAE}
    n_seq   : int   number of qualifying sequences
    """
    bf_mask = bfd_ml_mask(hydrograph_cfs)

    hydrograph_m3 = hydrograph_m3.reset_index(drop=True)
    dates = pd.to_datetime(hydrograph_m3['Date'].values)
    Q = np.array(hydrograph_m3['Streamflow'], dtype=float)
    n = len(hydrograph_m3)

    # Align mask length (in case cfs and m3 DataFrames differ by a few rows)
    if len(bf_mask) != n:
        # Re-align on dates
        cfs_dates = hydrograph_cfs['date'].dt.normalize().values
        m3_dates  = pd.to_datetime(hydrograph_m3['Date']).dt.normalize().values
        pred_map  = dict(zip(cfs_dates, bf_mask))
        bf_mask   = np.array([bool(pred_map.get(d, False)) for d in m3_dates])

    # Find contiguous BFD sequences
    sequences = []
    i = 0
    while i < n:
        if bf_mask[i]:
            j = i + 1
            while j < n and bf_mask[j]:
                j += 1
            if (j - i) >= min_days:
                sequences.append((i, j - 1))
            i = j
        else:
            i += 1

    residuals_all = []

    for (start_idx, end_idx) in sequences:
        seq_len = min(end_idx - start_idx + 1, max_days)

        train_start = max(0, start_idx - train_days)
        train_slice = hydrograph_m3.iloc[train_start:start_idx].reset_index(drop=True)
        if len(train_slice) == 0:
            continue

        train_bfs = pybfs.bfs(train_slice, SBT, basin_char, gw_hyd, flow)
        last = train_bfs.iloc[-1]
        ini = (
            last['X'], last['Zb.L'], last['Zs.L'],
            last['StBase'], last['StSur'],
            last['SurfaceFlow'], last['Baseflow'], last['Rech'],
        )

        fc_dates = dates[start_idx: start_idx + seq_len]
        fc_input = pd.DataFrame({'date': fc_dates, 'streamflow': np.nan})
        fc_result = pybfs.forecast(fc_input, SBT, basin_char, gw_hyd, flow, ini)

        obs  = Q[start_idx: start_idx + seq_len]
        pred = fc_result['Baseflow'].values
        res  = obs - pred
        valid = res[~np.isnan(res)]
        residuals_all.extend(valid.tolist())

    if residuals_all:
        arr = np.array(residuals_all)
        overall_rmse = float(np.sqrt(np.mean(arr ** 2)))
        overall_mae  = float(np.mean(np.abs(arr)))
    else:
        overall_rmse = np.nan
        overall_mae  = np.nan

    return {'overall_RMSE': overall_rmse, 'overall_MAE': overall_mae}, len(sequences)


# ── USGS API fetch ─────────────────────────────────────────────────────────────
def fetch_from_usgs(site_no, out_path):
    """Download daily streamflow from USGS Water Services and save as CSV.

    Returns True on success, False on failure.
    """
    url = (
        f'https://waterservices.usgs.gov/nwis/dv/'
        f'?format=json&sites={site_no}&parameterCd=00060'
        f'&startDT=1900-01-01&endDT=2030-12-31&siteStatus=all'
    )
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        ts_list = data['value']['timeSeries']
        if not ts_list:
            return False
        values = ts_list[0]['values'][0]['value']
        rows = []
        for v in values:
            try:
                q = float(v['value'])
            except (ValueError, TypeError):
                q = np.nan
            rows.append({'date': v['dateTime'][:10], 'streamflow': q})
        if not rows:
            return False
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        return True
    except Exception as e:
        print(f'  USGS fetch failed: {e}')
        return False


# ── Gage selection ─────────────────────────────────────────────────────────────
def select_all_gages():
    info = pd.read_csv(SITE_INFO_CSV, dtype={'site_no': str})
    selected = []
    for site_no in info['site_no']:
        site_no = str(site_no).zfill(8)
        params_path = os.path.join(PARAMS_DIR, f'params_{site_no}.csv')
        if os.path.exists(params_path):
            selected.append(site_no)
    return selected


# ── Data loaders ───────────────────────────────────────────────────────────────
def load_params(site_no):
    path = os.path.join(PARAMS_DIR, f'params_{site_no}.csv')
    df = pd.read_csv(path)
    row = df.iloc[0]
    area = row['tmp.area']
    basin_char = [area, row['Lb'], row['X1'], row['Wb'], row['POR']]
    gw_hyd = [row['ALPHA'], row['BETA'], row['Ks'], row['Kb'], row['Kz']]
    flow = [row['Qthresh'], row['Rs'], row['Rb1'], row['Rb2'], row['Prec'], row['Frac4Rise']]
    return basin_char, gw_hyd, flow


def load_streamflow(site_no, fetched_list):
    """Load streamflow in both cfs (for ML) and m³/day (for BFS).

    If the local CSV is missing, attempts to fetch from USGS API and records
    the site_no in fetched_list.
    """
    path = os.path.join(STREAMFLOW_DIR, f'{site_no}.csv')
    if not os.path.exists(path):
        success = fetch_from_usgs(site_no, path)
        if success:
            fetched_list.append(site_no)
        else:
            raise FileNotFoundError(f'No streamflow data for {site_no}')

    df = pd.read_csv(path, dtype={'date': str, 'streamflow': str})
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['streamflow_cfs'] = pd.to_numeric(df['streamflow'], errors='coerce')
    df = df.dropna(subset=['streamflow_cfs']).sort_values('date').reset_index(drop=True)

    # cfs version for ML features
    cfs_df = df[['date', 'streamflow_cfs']].rename(columns={'streamflow_cfs': 'streamflow'})

    # m³/day version for BFS
    m3_df = pd.DataFrame({
        'Date': df['date'],
        'Streamflow': df['streamflow_cfs'] * CFS_TO_M3_PER_DAY,
    })

    return cfs_df, m3_df


# ── Main ───────────────────────────────────────────────────────────────────────
def main(limit=None):
    out_path = os.path.join(OUTPUT_DIR, 'metrics.csv')
    fetched_log = os.path.join(OUTPUT_DIR, 'fetched_from_usgs.txt')

    # Resume logic
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path, dtype={'site_no': str})
        done = set(existing['site_no'].str.zfill(8).tolist())
        rows = existing.to_dict('records')
        print(f'Resuming — {len(done)} sites already processed.')
    else:
        done = set()
        rows = []

    fetched_from_usgs = []

    print('Selecting gages...')
    sites = select_all_gages()
    remaining = [s for s in sites if s not in done]

    if limit is not None:
        remaining = remaining[:limit]
        print(f'Running first {limit} gages (test mode).')

    total = len(remaining)
    print(f'Total eligible: {len(sites)}  |  Remaining this run: {total}\n')

    for i, site_no in enumerate(remaining):
        pct = (i + 1) / total * 100
        bar_width = 30
        filled = int(bar_width * (i + 1) / total)
        bar = '#' * filled + '-' * (bar_width - filled)
        print(f'\r[{bar}] {pct:5.1f}%  ({i+1}/{total})  {site_no}', end='  ', flush=True)

        try:
            cfs_df, m3_df = load_streamflow(site_no, fetched_from_usgs)
            if len(m3_df) < 365:
                raise ValueError(f'Insufficient data: {len(m3_df)} days')

            basin_char, gw_hyd, flow = load_params(site_no)
            lb, x1, wb, por = basin_char[1], basin_char[2], basin_char[3], basin_char[4]
            beta, kb = gw_hyd[1], gw_hyd[3]

            SBT = pybfs.base_table(lb, x1, wb, beta, kb, m3_df, por)
            metrics, n_seq = bfd_ml_forecast_skill(
                m3_df, cfs_df, SBT, basin_char, gw_hyd, flow
            )

            rows.append({
                'site_no':      site_no,
                'n_sequences':  n_seq,
                'overall_RMSE': metrics['overall_RMSE'],
                'overall_MAE':  metrics['overall_MAE'],
            })
            print(f'seq={n_seq}  RMSE={metrics["overall_RMSE"]:.1f}  MAE={metrics["overall_MAE"]:.1f}')

        except Exception as e:
            print(f'FAILED: {e}')
            rows.append({
                'site_no':      site_no,
                'n_sequences':  np.nan,
                'overall_RMSE': np.nan,
                'overall_MAE':  np.nan,
            })

        pd.DataFrame(rows, columns=['site_no', 'n_sequences', 'overall_RMSE', 'overall_MAE']
                     ).to_csv(out_path, index=False)

    print(f'\n\nDone. Metrics saved to {out_path}')

    if fetched_from_usgs:
        print(f'\nFetched {len(fetched_from_usgs)} gage(s) from USGS API:')
        for s in fetched_from_usgs:
            print(f'  {s}')
        with open(fetched_log, 'w') as f:
            f.write('\n'.join(fetched_from_usgs) + '\n')
        print(f'  (also saved to {fetched_log})')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only the first N gages (test mode)')
    args = parser.parse_args()
    main(limit=args.limit)
