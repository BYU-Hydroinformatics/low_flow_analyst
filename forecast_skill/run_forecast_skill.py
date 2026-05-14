# -*- coding: utf-8 -*-
"""Run forecast_skill() for all U.S. stream gages with low-flow and save metrics.

Data sources:
  - Streamflow:  ../../usgs_daily_streamflow/<site_no>.csv
  - Calibration: ../../usgs_calibration_results/params/params_<site_no>.csv
  - Gage list:   ../../low_flow_gages.csv
"""

import os
import sys
import traceback

import numpy as np
import pandas as pd
import pybfs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from baseflow.skill import forecast_skill

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, '..', '..')

STREAMFLOW_DIR = os.path.join(ROOT, 'usgs_daily_streamflow')
PARAMS_DIR = os.path.join(ROOT, 'usgs_calibration_results', 'params')
SITE_INFO_CSV = os.path.join(ROOT, 'usgs_daily_streamflow', 'site_info.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')

os.makedirs(OUTPUT_DIR, exist_ok=True)

CFS_TO_M3_PER_DAY = 0.0283168 * 86400


def load_params(site_no):
    path = os.path.join(PARAMS_DIR, f'params_{site_no}.csv')
    df = pd.read_csv(path)
    row = df.iloc[0]
    area = row['tmp.area']
    basin_char = [area, row['Lb'], row['X1'], row['Wb'], row['POR']]
    gw_hyd = [row['ALPHA'], row['BETA'], row['Ks'], row['Kb'], row['Kz']]
    flow = [row['Qthresh'], row['Rs'], row['Rb1'], row['Rb2'], row['Prec'], row['Frac4Rise']]
    return basin_char, gw_hyd, flow


def load_streamflow(site_no):
    path = os.path.join(STREAMFLOW_DIR, f'{site_no}.csv')
    df = pd.read_csv(path, dtype={'date': str, 'streamflow': str})
    df['Date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['Streamflow'] = pd.to_numeric(df['streamflow'], errors='coerce') * CFS_TO_M3_PER_DAY
    df = df.dropna(subset=['Streamflow'])
    df = df[['Date', 'Streamflow']].sort_values('Date').reset_index(drop=True)
    return df


def select_all_gages():
    info = pd.read_csv(SITE_INFO_CSV, dtype={'site_no': str})
    selected = []
    for site_no in info['site_no']:
        site_no = str(site_no).zfill(8)
        params_path = os.path.join(PARAMS_DIR, f'params_{site_no}.csv')
        streamflow_path = os.path.join(STREAMFLOW_DIR, f'{site_no}.csv')
        if os.path.exists(params_path) and os.path.exists(streamflow_path):
            selected.append(site_no)
    return selected


def run_site(site_no):
    hydrograph = load_streamflow(site_no)
    if len(hydrograph) < 365:
        raise ValueError(f'Insufficient data: {len(hydrograph)} days')

    basin_char, gw_hyd, flow = load_params(site_no)
    lb, x1, wb, por = basin_char[1], basin_char[2], basin_char[3], basin_char[4]
    beta, kb = gw_hyd[1], gw_hyd[3]

    SBT = pybfs.base_table(lb, x1, wb, beta, kb, hydrograph, por)
    skill_df, summary_df, metrics = forecast_skill(hydrograph, SBT, basin_char, gw_hyd, flow)

    return metrics, len(summary_df)


def main():
    out_path = os.path.join(OUTPUT_DIR, 'metrics.csv')

    # Resume: load already-processed sites
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path, dtype={'site_no': str})
        done = set(existing['site_no'].str.zfill(8).tolist())
        rows = existing.to_dict('records')
        print(f'Resuming — {len(done)} sites already done.')
    else:
        done = set()
        rows = []

    print('Selecting gages...')
    sites = select_all_gages()
    remaining = [s for s in sites if s not in done]
    print(f'Total eligible: {len(sites)}  |  Remaining: {len(remaining)}\n')

    for i, site_no in enumerate(remaining):
        print(f'[{i+1}/{len(remaining)}] {site_no}', end='  ', flush=True)
        try:
            metrics, n_sequences = run_site(site_no)
            rows.append({
                'site_no': site_no,
                'n_sequences': n_sequences,
                'overall_RMSE': metrics['overall_RMSE'],
                'overall_MAE': metrics['overall_MAE'],
            })
            print(f'seq={n_sequences}  RMSE={metrics["overall_RMSE"]:.1f}  MAE={metrics["overall_MAE"]:.1f}')
        except Exception as e:
            print(f'FAILED: {e}')
            rows.append({
                'site_no': site_no,
                'n_sequences': np.nan,
                'overall_RMSE': np.nan,
                'overall_MAE': np.nan,
            })

        # Save after every site so progress isn't lost
        pd.DataFrame(rows, columns=['site_no', 'n_sequences', 'overall_RMSE', 'overall_MAE']
                     ).to_csv(out_path, index=False)

    print(f'\nDone. Metrics saved to {out_path}')


if __name__ == '__main__':
    main()
