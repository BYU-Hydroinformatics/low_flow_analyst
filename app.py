"""
Low Flow Analyst (LFA)
Interactive map-based viewer for USGS stream gage baseflow separation analysis.
"""
import os
import sys
import json
import math
import time
import ssl
import tempfile
import shutil
import threading
import traceback
import urllib.request
from io import StringIO
from datetime import datetime

# Add project root to path (for local baseflow package)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify, send_from_directory, Response, request
import pandas as pd
import numpy as np

import pybfs
from baseflow.separation import chapman, lh, eckhardt, strict_baseflow
from baseflow.estimate import recession_coefficient, maxmium_BFI
from baseflow.skill import separation_skill, forecast_skill

# Reuse functions from existing scripts
from calibrate_all_gages import calibrate_site, load_drainage_areas
from run_bfs_all_gages import process_site, load_site_params, load_streamflow

# Disable SSL verification for USGS API
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

# ----- Configuration -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STREAMFLOW_DIR = os.path.join(BASE_DIR, "usgs_daily_streamflow")
CALIB_DIR = os.path.join(BASE_DIR, "usgs_calibration_results")
PARAMS_DIR = os.path.join(CALIB_DIR, "params")
BFF_DIR = os.path.join(CALIB_DIR, "bff")
RESULTS_DIR = os.path.join(BASE_DIR, "usgs_bfs_results")
TEMP_DIR = os.path.join(BASE_DIR, "temp_results")

CFS_TO_M3_PER_DAY = 0.0283168 * 86400
SQMI_TO_M2 = 2_589_988.11
FORECAST_DAYS = 90

# ----- In-memory state -----
site_info = {}          # site_no -> {name, lat, lng, drain_area_sqmi, drain_area_m2}
nwm_info = {}           # site_no -> {behavior, nwm_id, stream_order}
low_flow_info = {}      # site_no -> {'has_lowflow': bool, 'max_lowflow_duration': int}
metrics_info = {}       # site_no -> {overall_RMSE, overall_MAE, n_sequences}
processing_status = {}  # site_no -> {stage, progress, message, error, done}
fips_cache = {}         # (lat, lng) -> county FIPS code


def load_site_info_data():
    """Load all gage metadata from site_info.csv at startup."""
    path = os.path.join(STREAMFLOW_DIR, "site_info.csv")
    df = pd.read_csv(path, dtype=str)
    for _, row in df.iterrows():
        site_no = row['site_no']
        try:
            lat = float(row['dec_lat_va'])
            lng = float(row['dec_long_va'])
        except (ValueError, TypeError):
            continue
        drain_area_sqmi = None
        drain_area_m2 = None
        try:
            drain_area_sqmi = float(row['drain_area_va'])
            drain_area_m2 = drain_area_sqmi * SQMI_TO_M2
        except (ValueError, TypeError):
            pass
        site_info[site_no] = {
            'name': row.get('station_nm', ''),
            'lat': lat,
            'lng': lng,
            'drain_area_sqmi': drain_area_sqmi,
            'drain_area_m2': drain_area_m2,
        }
    print(f"Loaded {len(site_info)} gages from site_info.csv")


def load_nwm_data():
    """Load NWM natural/artificial classification and GAGES-II ref status at startup."""
    behav_dir = os.path.join(BASE_DIR, "behavior")
    for filename, behavior in [("NWM_USGS_Natural_Flow.csv", "Natural"),
                               ("NWM_USGS_Artificial_Path.csv", "Artificial")]:
        path = os.path.join(behav_dir, filename)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, na_values=['<Null>', '<null>', ''])
        for _, row in df.iterrows():
            site_no = str(int(row['USGS_ID'])).zfill(8)
            nwm_info[site_no] = {
                'behavior': behavior,
                'nwm_id': int(row['NWM_ID']),
                'stream_order': int(row['Stream Order']) if pd.notna(row.get('Stream Order')) else None,
                'ref_status': None,
            }

    # Load GAGES-II reference/non-reference classification
    ref_path = os.path.join(behav_dir, "GAGES-II_ref_non_ref.csv")
    if os.path.exists(ref_path):
        ref_df = pd.read_csv(ref_path, dtype=str)
        for _, row in ref_df.iterrows():
            site_no = row['STAID']
            if site_no in nwm_info:
                nwm_info[site_no]['ref_status'] = row['CLASS']
            else:
                nwm_info[site_no] = {
                    'behavior': None,
                    'nwm_id': None,
                    'stream_order': None,
                    'ref_status': row['CLASS'],
                }
    print(f"Loaded NWM/behavior info for {len(nwm_info)} gages")


def load_low_flow_data():
    """Load pre-computed low-flow classification at startup."""
    path = os.path.join(BASE_DIR, "low_flow_gages.csv")
    if not os.path.exists(path):
        print("Warning: low_flow_gages.csv not found. Low-flow filter will be unavailable.")
        return
    df = pd.read_csv(path, dtype={'site_no': str})
    for _, row in df.iterrows():
        site_no = row['site_no']
        low_flow_info[site_no] = {
            'has_lowflow': str(row.get('has_lowflow', '')).lower() == 'true',
            'max_lowflow_duration': int(row.get('max_lowflow_duration', 0)),
        }
    count = sum(1 for v in low_flow_info.values() if v['has_lowflow'])
    print(f"Loaded low-flow classification: {count}/{len(low_flow_info)} gages have low-flow periods")


def load_metrics_data():
    """Load pre-computed forecast skill metrics at startup."""
    path = os.path.join(BASE_DIR, "forecast_skill", "output", "metrics.csv")
    if not os.path.exists(path):
        print("Warning: forecast_skill/output/metrics.csv not found.")
        return
    df = pd.read_csv(path, dtype={'site_no': str})
    for _, row in df.iterrows():
        site_no = str(row['site_no']).zfill(8)
        try:
            metrics_info[site_no] = {
                'overall_RMSE': float(row['overall_RMSE']) if pd.notna(row.get('overall_RMSE')) else None,
                'overall_MAE': float(row['overall_MAE']) if pd.notna(row.get('overall_MAE')) else None,
                'n_sequences': int(row['n_sequences']) if pd.notna(row.get('n_sequences')) else None,
            }
        except (ValueError, TypeError):
            pass
    print(f"Loaded forecast metrics for {len(metrics_info)} gages")


def get_gage_status(site_no):
    """Check calibration status for a gage."""
    params_path = os.path.join(PARAMS_DIR, f"params_{site_no}.csv")
    if os.path.exists(params_path):
        return "calibrated"
    return "pending"


# =====================================================
# USGS Data Fetching (for Update Data feature)
# =====================================================

def fetch_usgs_streamflow(site_no):
    """Download all available daily streamflow from USGS WaterServices.
    Returns a DataFrame with columns [date, streamflow] (in cfs), or None on failure.
    """
    base_url = 'https://waterservices.usgs.gov/nwis/dv/?'
    start_date = '1800-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    url = (
        f'{base_url}sites={site_no}&parameterCd=00060'
        f'&startDT={start_date}&endDT={end_date}&format=rdb'
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read().decode('utf-8')

    lines = content.split('\n')
    data_lines = [line for line in lines if not line.startswith('#')]
    if len(data_lines) <= 2:
        return None

    df = pd.read_csv(StringIO('\n'.join(data_lines)), delimiter='\t', dtype=str)
    if len(df) <= 1:
        return None
    df = df.iloc[1:, :]

    # Drop unnecessary columns
    columns_to_remove = ['site_no', 'agency_cd']
    for col in df.columns:
        if '_00060_00003_cd' in col:
            columns_to_remove.append(col)
    df = df.drop(columns=[c for c in columns_to_remove if c in df.columns], errors='ignore')

    # Rename to standard names
    rename_dict = {}
    if 'datetime' in df.columns:
        rename_dict['datetime'] = 'date'
    for col in df.columns:
        if '_00060_00003' in col and '_00060_00003_cd' not in col:
            rename_dict[col] = 'streamflow'
    df = df.rename(columns=rename_dict)

    if 'date' not in df.columns or 'streamflow' not in df.columns:
        return None
    return df[['date', 'streamflow']].reset_index(drop=True)


def fetch_usgs_site_info(site_no):
    """Fetch site metadata (including drainage area) from USGS.
    Returns dict with drain_area_sqmi or None.
    """
    url = (
        f"https://waterservices.usgs.gov/nwis/site/?format=rdb"
        f"&sites={site_no}&siteOutput=expanded"
    )
    with urllib.request.urlopen(url, timeout=30) as response:
        content = response.read().decode('utf-8')

    lines = content.split('\n')
    data_lines = [line for line in lines if not line.startswith('#')]
    if len(data_lines) <= 2:
        return None

    df = pd.read_csv(StringIO('\n'.join(data_lines)), delimiter='\t', dtype=str)
    if len(df) <= 1:
        return None
    df = df.iloc[1:, :]

    result = {}
    if 'drain_area_va' in df.columns:
        try:
            result['drain_area_sqmi'] = float(df.iloc[0]['drain_area_va'])
        except (ValueError, TypeError):
            pass
    if 'station_nm' in df.columns:
        result['station_nm'] = df.iloc[0]['station_nm']
    return result


# ----- In-memory BFS result cache -----
bfs_cache = {}  # key -> {data: dict, timestamp: float}
BFS_CACHE_TTL = 300  # seconds


def run_bfs_on_the_fly(site_no, sf_dir=None, p_dir=None, b_dir=None):
    """Run BFS analysis on-the-fly from calibration parameters.

    Returns a dict ready for JSON response, or None on failure.
    Uses sf_dir/p_dir/b_dir if provided (for updated/temp data),
    otherwise defaults to the main directories.
    """
    sf_dir = sf_dir or STREAMFLOW_DIR
    p_dir = p_dir or PARAMS_DIR
    b_dir = b_dir or BFF_DIR

    cache_key = f"{site_no}:{sf_dir}:{p_dir}"
    if cache_key in bfs_cache:
        cached = bfs_cache[cache_key]
        if time.time() - cached['timestamp'] < BFS_CACHE_TTL:
            return cached['data']

    params_path = os.path.join(p_dir, f"params_{site_no}.csv")
    bff_path = os.path.join(b_dir, f"bff_{site_no}.csv")
    streamflow_path = os.path.join(sf_dir, f"{site_no}.csv")

    if not all(os.path.exists(p) for p in [params_path, bff_path, streamflow_path]):
        return None

    basin_char, gw_hyd, flow = load_site_params(params_path)
    bff_df = pd.read_csv(bff_path)
    streamflow_df = load_streamflow(streamflow_path)

    if len(streamflow_df) < 365:
        return None

    # Run BFS
    lb, x1, wb, por = basin_char[1], basin_char[2], basin_char[3], basin_char[4]
    beta, kb = gw_hyd[1], gw_hyd[3]
    SBT = pybfs.base_table(lb, x1, wb, beta, kb, streamflow_df, por)
    bfs_out = pybfs.bfs(streamflow_df, SBT, basin_char, gw_hyd, flow)

    # Downsample for large datasets
    total = len(bfs_out)
    step = 1
    bfs_display = bfs_out
    if total > 5000:
        step = max(1, total // 5000)
        bfs_display = bfs_out.iloc[::step].reset_index(drop=True)

    result = {
        'dates': pd.to_datetime(bfs_display['Date']).dt.strftime('%Y-%m-%d').tolist(),
        'qob': (bfs_display['Qob'] / 86400).round(4).tolist(),
        'baseflow': (bfs_display['Baseflow'] / 86400).round(4).tolist(),
        'surface_flow': (bfs_display['SurfaceFlow'] / 86400).round(4).tolist(),
        'direct_runoff': (bfs_display['DirectRunoff'] / 86400).round(4).tolist(),
        'total_points': total,
    }

    # BFF fractions
    try:
        row = bff_df.iloc[0]
        result['bff'] = round(float(row['BFF']), 4)
        result['sff'] = round(float(row['SFF']), 4)
        result['drf'] = round(float(row['DRF']), 4)
    except Exception:
        pass

    # Annual BFI
    bfs_out_copy = bfs_out.copy()
    bfs_out_copy['Year'] = pd.to_datetime(bfs_out_copy['Date']).dt.year
    annual = bfs_out_copy.groupby('Year').agg({'Qob': 'sum', 'Baseflow': 'sum'})
    annual['BFI'] = (annual['Baseflow'] / annual['Qob']).clip(0, 1)
    annual = annual[annual['Qob'] > 0]
    result['annual_years'] = annual.index.tolist()
    result['annual_bfi'] = annual['BFI'].round(4).tolist()
    result['mean_bfi'] = round(float(annual['BFI'].mean()), 4)

    # Confidence intervals
    try:
        ci_table, ci_df = pybfs.bf_ci(bfs_out)
        ci_display = ci_df
        if total > 5000:
            ci_display = ci_df.iloc[::step].reset_index(drop=True)
        lo_col = next((c for c in ci_display.columns if '0.05' in c or '05' in c), None)
        hi_col = next((c for c in ci_display.columns if '0.95' in c or '95' in c), None)
        if lo_col and hi_col:
            result['ci_lower'] = (ci_display[lo_col] / 86400).round(4).tolist()
            result['ci_upper'] = (ci_display[hi_col] / 86400).round(4).tolist()
    except Exception:
        pass

    # Forecast
    try:
        last_row = bfs_out.iloc[-1]
        ini = (
            last_row['X'], last_row['Zb.L'], last_row['Zs.L'],
            last_row['StBase'], last_row['StSur'],
            last_row['SurfaceFlow'], last_row['Baseflow'], last_row['Rech'],
        )
        last_date = pd.to_datetime(bfs_out['Date'].iloc[-1])
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                       periods=FORECAST_DAYS, freq='D')
        forecast_input = pd.DataFrame({'date': forecast_dates, 'streamflow': np.nan})
        forecast_out = pybfs.forecast(forecast_input, SBT, basin_char, gw_hyd, flow, ini)
        result['forecast_dates'] = pd.to_datetime(forecast_out['Date']).dt.strftime('%Y-%m-%d').tolist()
        result['forecast_baseflow'] = (forecast_out['Baseflow'] / 86400).round(4).tolist()
    except Exception:
        pass

    # Traditional baseflow separation methods
    try:
        Q = bfs_out['Qob'].values.astype(float)
        strict = strict_baseflow(Q)
        a = recession_coefficient(Q, strict)
        b_lh = lh(Q)
        BFImax = maxmium_BFI(Q, b_lh, a)
        bf_chapman = chapman(Q, a)
        bf_eckhardt = eckhardt(Q, a, BFImax)
        if total > 5000:
            bf_chapman = bf_chapman[::step]
            b_lh = b_lh[::step]
            bf_eckhardt = bf_eckhardt[::step]
        result['bf_chapman'] = (bf_chapman / 86400).round(4).tolist()
        result['bf_lh'] = (b_lh / 86400).round(4).tolist()
        result['bf_eckhardt'] = (bf_eckhardt / 86400).round(4).tolist()
    except Exception:
        pass

    # Cache the result
    bfs_cache[cache_key] = {'data': result, 'timestamp': time.time()}
    return result


# =====================================================
# API Routes
# =====================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/gages')
def api_gages():
    """Return all gages with lat/lng and processing status."""
    gages = []
    for site_no, info in site_info.items():
        entry = {
            'site_no': site_no,
            'name': info['name'],
            'lat': info['lat'],
            'lng': info['lng'],
            'status': get_gage_status(site_no),
        }
        if site_no in nwm_info:
            nwm = nwm_info[site_no]
            entry['behavior'] = nwm.get('behavior')
            entry['ref_status'] = nwm.get('ref_status')
        entry['has_lowflow'] = low_flow_info.get(site_no, {}).get('has_lowflow', False)
        drain = info.get('drain_area_sqmi')
        entry['drain_area_sqmi'] = drain if (drain is not None and math.isfinite(drain)) else None
        if site_no in metrics_info:
            m = metrics_info[site_no]
            rmse = m.get('overall_RMSE')
            mae  = m.get('overall_MAE')
            entry['overall_RMSE'] = rmse if (rmse is not None and math.isfinite(rmse)) else None
            entry['overall_MAE']  = mae  if (mae  is not None and math.isfinite(mae))  else None
            entry['n_sequences']  = m.get('n_sequences')
        gages.append(entry)
    return jsonify(gages)


@app.route('/api/gage/<site_no>/info')
def api_gage_info(site_no):
    """Return detailed info for a specific gage."""
    if site_no not in site_info:
        return jsonify({'error': 'Gage not found'}), 404
    info = site_info[site_no]
    result = {
        'site_no': site_no,
        'name': info['name'],
        'lat': info['lat'],
        'lng': info['lng'],
        'drain_area_sqmi': info['drain_area_sqmi'],
        'status': get_gage_status(site_no),
    }
    if site_no in nwm_info:
        nwm = nwm_info[site_no]
        if nwm['nwm_id'] is not None:
            result['nwm_id'] = nwm['nwm_id']
        if nwm['stream_order'] is not None:
            result['stream_order'] = nwm['stream_order']
        if nwm['behavior'] is not None:
            result['river_behavior'] = nwm['behavior']
        if nwm['ref_status'] is not None:
            result['ref_status'] = nwm['ref_status']

    bff_path = os.path.join(BFF_DIR, f"bff_{site_no}.csv")
    if os.path.exists(bff_path):
        try:
            bff_df = pd.read_csv(bff_path)
            row = bff_df.iloc[0]
            result['bff'] = round(float(row['BFF']), 4)
            result['sff'] = round(float(row['SFF']), 4)
            result['drf'] = round(float(row['DRF']), 4)
        except Exception:
            pass
    return jsonify(result)


@app.route('/api/gage/<site_no>/plots')
def api_gage_plots(site_no):
    """Return list of available plot files for a gage."""
    # Check temp results first (for updated data runs)
    source = request.args.get('source', 'original')
    if source == 'updated':
        site_dir = os.path.join(TEMP_DIR, site_no)
    else:
        site_dir = os.path.join(RESULTS_DIR, site_no)

    plot_files = [
        'baseflow_separation.png',
        'flow_fractions.png',
        'annual_bfi.png',
        'confidence_intervals.png',
        'forecast.png',
    ]
    available = []
    for f in plot_files:
        if os.path.exists(os.path.join(site_dir, f)):
            available.append({
                'name': f.replace('.png', '').replace('_', ' ').title(),
                'url': f'/plots/{site_no}/{f}?source={source}',
            })
    return jsonify(available)


@app.route('/plots/<site_no>/<filename>')
def serve_plot(site_no, filename):
    """Serve a plot image file."""
    source = request.args.get('source', 'original')
    if source == 'updated':
        site_dir = os.path.join(TEMP_DIR, site_no)
    else:
        site_dir = os.path.join(RESULTS_DIR, site_no)
    return send_from_directory(site_dir, filename)


# =====================================================
# Interactive Plot Data API
# =====================================================

@app.route('/api/gage/<site_no>/data')
def api_gage_data(site_no):
    """Return BFS results as JSON for interactive Plotly charts.

    Runs BFS on-the-fly from calibration parameters instead of reading
    pre-computed results.
    """
    source = request.args.get('source', 'original')
    try:
        if source == 'updated':
            temp_site_dir = os.path.join(TEMP_DIR, site_no)
            result = run_bfs_on_the_fly(
                site_no,
                sf_dir=temp_site_dir,
                p_dir=temp_site_dir,
                b_dir=temp_site_dir,
            )
        else:
            result = run_bfs_on_the_fly(site_no)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'BFS computation failed: {str(e)}'}), 500

    if result is None:
        # Provide a specific reason
        params_path = os.path.join(PARAMS_DIR, f"params_{site_no}.csv")
        bff_path = os.path.join(BFF_DIR, f"bff_{site_no}.csv")
        streamflow_path = os.path.join(STREAMFLOW_DIR, f"{site_no}.csv")
        missing = []
        if not os.path.exists(params_path):
            missing.append('calibration parameters')
        if not os.path.exists(bff_path):
            missing.append('baseflow fractions')
        if not os.path.exists(streamflow_path):
            missing.append('streamflow data')
        if missing:
            reason = f"Missing: {', '.join(missing)}"
        else:
            reason = "Insufficient streamflow data (need at least 365 days)"
        return jsonify({'error': reason}), 404

    return safe_jsonify(result)


def safe_jsonify(obj):
    """Return a JSON response that converts NaN/Infinity to null (valid JSON)."""
    def sanitize(v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        if isinstance(v, list):
            return [sanitize(x) for x in v]
        if isinstance(v, dict):
            return {k: sanitize(val) for k, val in v.items()}
        # Handle numpy types
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            f = float(v)
            return None if (np.isnan(f) or np.isinf(f)) else f
        return v
    return Response(
        json.dumps(sanitize(obj)),
        mimetype='application/json'
    )


# =====================================================
# Processing Endpoints (original data)
# =====================================================

@app.route('/api/gage/<site_no>/process', methods=['POST'])
def api_process_gage(site_no):
    """Trigger calibration for a gage (BFS runs on-the-fly when viewing)."""
    if site_no not in site_info:
        return jsonify({'error': 'Gage not found'}), 404
    if get_gage_status(site_no) == "calibrated":
        return jsonify({'status': 'already_done'})
    if site_no in processing_status and not processing_status[site_no].get('done'):
        return jsonify({'status': 'already_processing'})

    info = site_info[site_no]
    if info['drain_area_m2'] is None:
        return jsonify({'error': 'No drainage area available for this gage'}), 400
    streamflow_path = os.path.join(STREAMFLOW_DIR, f"{site_no}.csv")
    if not os.path.exists(streamflow_path):
        return jsonify({'error': 'No streamflow data file for this gage'}), 400

    processing_status[site_no] = {
        'stage': 'starting', 'progress': 0,
        'message': 'Starting calibration...', 'error': None, 'done': False,
    }
    thread = threading.Thread(target=run_gage_processing, args=(site_no,), daemon=True)
    thread.start()
    return jsonify({'status': 'started'})


def run_gage_processing(site_no):
    """Background task: calibrate parameters. BFS runs on-the-fly when viewing."""
    try:
        info = site_info[site_no]
        area_m2 = info['drain_area_m2']
        streamflow_path = os.path.join(STREAMFLOW_DIR, f"{site_no}.csv")

        params_path = os.path.join(PARAMS_DIR, f"params_{site_no}.csv")
        if not os.path.exists(params_path):
            processing_status[site_no].update({
                'stage': 'calibrating', 'progress': 10,
                'message': 'Calibrating parameters... This may take ~90 seconds.',
            })
            os.makedirs(PARAMS_DIR, exist_ok=True)
            os.makedirs(BFF_DIR, exist_ok=True)

            bf_params, bff = calibrate_site(site_no, streamflow_path, area_m2)
            if bf_params is None:
                processing_status[site_no].update({
                    'stage': 'error', 'progress': 0,
                    'message': 'Calibration failed - insufficient or invalid data.',
                    'error': 'Calibration returned no results', 'done': True,
                })
                return
            bf_params.to_csv(params_path, index=False, float_format='%.6g')
            bff.to_csv(os.path.join(BFF_DIR, f"bff_{site_no}.csv"),
                       index=False, float_format='%.6g')

        processing_status[site_no].update({
            'stage': 'done', 'progress': 100,
            'message': 'Calibration complete! Charts will be generated on view.',
            'done': True,
        })
    except Exception as e:
        traceback.print_exc()
        processing_status[site_no].update({
            'stage': 'error', 'progress': 0,
            'message': f'Error: {str(e)}', 'error': str(e), 'done': True,
        })


# =====================================================
# Update Data Feature (live USGS fetch + temp processing)
# =====================================================

@app.route('/api/gage/<site_no>/update', methods=['POST'])
def api_update_gage(site_no):
    """Fetch latest USGS data and run full analysis in a temp directory."""
    if site_no not in site_info:
        return jsonify({'error': 'Gage not found'}), 404

    status_key = f"update_{site_no}"
    if status_key in processing_status and not processing_status[status_key].get('done'):
        return jsonify({'status': 'already_processing'})

    processing_status[status_key] = {
        'stage': 'starting', 'progress': 0,
        'message': 'Starting data update...', 'error': None, 'done': False,
    }
    thread = threading.Thread(target=run_update_processing, args=(site_no,), daemon=True)
    thread.start()
    return jsonify({'status': 'started'})


def run_update_processing(site_no):
    """Background: fetch fresh USGS streamflow and reuse existing calibration parameters."""
    status_key = f"update_{site_no}"
    try:
        # Step 1: Fetch latest streamflow from USGS
        processing_status[status_key].update({
            'stage': 'downloading', 'progress': 10,
            'message': 'Downloading latest streamflow data from USGS...',
        })
        fresh_df = fetch_usgs_streamflow(site_no)
        if fresh_df is None or len(fresh_df) == 0:
            processing_status[status_key].update({
                'stage': 'error', 'progress': 0,
                'message': 'Could not retrieve streamflow data from USGS.',
                'error': 'No data returned', 'done': True,
            })
            return

        n_records = len(fresh_df)
        processing_status[status_key].update({
            'stage': 'downloaded', 'progress': 60,
            'message': f'Downloaded {n_records:,} daily records. Loading existing calibration parameters...',
        })

        # Step 2: Verify existing calibration parameters exist
        params_path = os.path.join(PARAMS_DIR, f"params_{site_no}.csv")
        bff_path = os.path.join(BFF_DIR, f"bff_{site_no}.csv")
        if not os.path.exists(params_path) or not os.path.exists(bff_path):
            processing_status[status_key].update({
                'stage': 'error', 'progress': 0,
                'message': 'No calibration parameters found. Use Recalibrate to generate them.',
                'error': 'Missing calibration parameters', 'done': True,
            })
            return

        # Step 3: Set up temp directory and save fresh streamflow
        temp_site_dir = os.path.join(TEMP_DIR, site_no)
        os.makedirs(temp_site_dir, exist_ok=True)
        fresh_df.to_csv(os.path.join(temp_site_dir, f"{site_no}.csv"), index=False)

        # Copy existing params/bff into temp so run_bfs_on_the_fly can find them
        shutil.copy2(params_path, os.path.join(temp_site_dir, f"params_{site_no}.csv"))
        shutil.copy2(bff_path, os.path.join(temp_site_dir, f"bff_{site_no}.csv"))

        processing_status[status_key].update({
            'stage': 'done', 'progress': 100,
            'message': f'Streamflow updated! ({n_records:,} records) Using existing calibration parameters.',
            'done': True,
            'n_records': n_records,
        })

    except Exception as e:
        traceback.print_exc()
        processing_status[status_key].update({
            'stage': 'error', 'progress': 0,
            'message': f'Update error: {str(e)}', 'error': str(e), 'done': True,
        })


# =====================================================
# Recalibrate Feature (fresh USGS fetch + full recalibration)
# =====================================================

@app.route('/api/gage/<site_no>/recalibrate', methods=['POST'])
def api_recalibrate_gage(site_no):
    """Fetch latest USGS data and run full recalibration in a temp directory."""
    if site_no not in site_info:
        return jsonify({'error': 'Gage not found'}), 404

    status_key = f"recalibrate_{site_no}"
    if status_key in processing_status and not processing_status[status_key].get('done'):
        return jsonify({'status': 'already_processing'})

    processing_status[status_key] = {
        'stage': 'starting', 'progress': 0,
        'message': 'Starting recalibration...', 'error': None, 'done': False,
    }
    thread = threading.Thread(target=run_recalibrate_processing, args=(site_no,), daemon=True)
    thread.start()
    return jsonify({'status': 'started'})


def run_recalibrate_processing(site_no):
    """Background: fetch fresh USGS data, recalibrate, and save results to temp directory."""
    status_key = f"recalibrate_{site_no}"
    try:
        # Step 1: Fetch latest streamflow from USGS
        processing_status[status_key].update({
            'stage': 'downloading', 'progress': 5,
            'message': 'Downloading latest streamflow data from USGS...',
        })
        fresh_df = fetch_usgs_streamflow(site_no)
        if fresh_df is None or len(fresh_df) == 0:
            processing_status[status_key].update({
                'stage': 'error', 'progress': 0,
                'message': 'Could not retrieve streamflow data from USGS.',
                'error': 'No data returned', 'done': True,
            })
            return

        n_records = len(fresh_df)
        processing_status[status_key].update({
            'stage': 'downloaded', 'progress': 10,
            'message': f'Downloaded {n_records:,} daily records. Fetching site metadata...',
        })

        # Step 2: Fetch fresh drainage area from USGS
        fresh_info = fetch_usgs_site_info(site_no)
        info = site_info[site_no]
        drain_area_sqmi = info['drain_area_sqmi']
        if fresh_info and 'drain_area_sqmi' in fresh_info and fresh_info['drain_area_sqmi']:
            drain_area_sqmi = fresh_info['drain_area_sqmi']

        if drain_area_sqmi is None or drain_area_sqmi <= 0:
            processing_status[status_key].update({
                'stage': 'error', 'progress': 0,
                'message': 'No drainage area available (neither local nor from USGS).',
                'error': 'Missing drainage area', 'done': True,
            })
            return

        area_m2 = drain_area_sqmi * SQMI_TO_M2

        # Step 3: Set up temp directory and save fresh streamflow
        temp_site_dir = os.path.join(TEMP_DIR, site_no)
        os.makedirs(temp_site_dir, exist_ok=True)
        temp_streamflow_path = os.path.join(temp_site_dir, f"{site_no}.csv")
        fresh_df.to_csv(temp_streamflow_path, index=False)

        # Step 4: Calibrate on fresh data
        processing_status[status_key].update({
            'stage': 'calibrating', 'progress': 15,
            'message': f'Calibrating on {n_records:,} records... This may take ~90 seconds.',
        })
        bf_params, bff = calibrate_site(site_no, temp_streamflow_path, area_m2)
        if bf_params is None:
            processing_status[status_key].update({
                'stage': 'error', 'progress': 0,
                'message': 'Calibration failed on updated data.',
                'error': 'Calibration returned no results', 'done': True,
            })
            return

        # Save new params/bff to temp
        bf_params.to_csv(os.path.join(temp_site_dir, f"params_{site_no}.csv"),
                         index=False, float_format='%.6g')
        bff.to_csv(os.path.join(temp_site_dir, f"bff_{site_no}.csv"),
                   index=False, float_format='%.6g')

        processing_status[status_key].update({
            'stage': 'done', 'progress': 100,
            'message': f'Recalibration complete! ({n_records:,} records, area: {drain_area_sqmi:.1f} mi²).',
            'done': True,
            'drain_area_sqmi': drain_area_sqmi,
            'n_records': n_records,
        })

    except Exception as e:
        traceback.print_exc()
        processing_status[status_key].update({
            'stage': 'error', 'progress': 0,
            'message': f'Recalibration error: {str(e)}', 'error': str(e), 'done': True,
        })


@app.route('/api/gage/<site_no>/progress')
def api_gage_progress(site_no):
    """SSE endpoint for real-time progress updates."""
    source = request.args.get('source', 'original')
    if source == 'updated':
        status_key = f"update_{site_no}"
    elif source == 'recalibrated':
        status_key = f"recalibrate_{site_no}"
    else:
        status_key = site_no

    def generate():
        while True:
            status = processing_status.get(status_key, {
                'stage': 'unknown', 'progress': 0,
                'message': 'No processing in progress.', 'done': True,
            })
            yield f"data: {json.dumps(status)}\n\n"
            if status.get('done'):
                break
            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# =====================================================
# Drought Conditions (US Drought Monitor)
# =====================================================

def get_county_fips(lat, lng):
    """Convert lat/lon to a 5-digit county FIPS code using FCC Census API."""
    key = (round(lat, 4), round(lng, 4))
    if key in fips_cache:
        return fips_cache[key]
    url = f"https://geo.fcc.gov/api/census/area?lat={lat}&lon={lng}&format=json"
    with urllib.request.urlopen(url, timeout=15) as response:
        data = json.loads(response.read().decode('utf-8'))
    results = data.get('results', [])
    if not results:
        return None
    fips = results[0].get('county_fips')
    if fips:
        fips_cache[key] = fips
    return fips


def fetch_drought_data(fips, start_date, end_date):
    """Fetch drought severity percentages from USDM for a county."""
    url = (
        f"https://usdmdataservices.unl.edu/api/CountyStatistics/"
        f"GetDroughtSeverityStatisticsByAreaPercent"
        f"?aoi={fips}&startdate={start_date}&enddate={end_date}&statisticsType=1"
    )
    req = urllib.request.Request(url, headers={'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode('utf-8'))


@app.route('/api/gage/<site_no>/drought')
def api_gage_drought(site_no):
    """Return drought severity timeline for the gage's county."""
    if site_no not in site_info:
        return jsonify({'error': 'Gage not found'}), 404

    info = site_info[site_no]
    try:
        fips = get_county_fips(info['lat'], info['lng'])
    except Exception as e:
        return jsonify({'error': f'FIPS lookup failed: {str(e)}'}), 500

    if not fips:
        return jsonify({'error': 'Could not determine county for this location'}), 404

    end = datetime.now()
    start = datetime(2000, 1, 4)  # USDM records begin Jan 4, 2000
    start_str = start.strftime('%-m/%-d/%Y')
    end_str = end.strftime('%-m/%-d/%Y')

    try:
        raw = fetch_drought_data(fips, start_str, end_str)
    except Exception as e:
        return jsonify({'error': f'Drought data fetch failed: {str(e)}'}), 500

    if not raw:
        return jsonify({'error': 'No drought data available for this county'}), 404

    dates = []
    d0, d1, d2, d3, d4, none_pct = [], [], [], [], [], []
    for entry in raw:
        dates.append(entry.get('MapDate', entry.get('mapDate', '')))
        d0.append(float(entry.get('D0', entry.get('d0', 0))))
        d1.append(float(entry.get('D1', entry.get('d1', 0))))
        d2.append(float(entry.get('D2', entry.get('d2', 0))))
        d3.append(float(entry.get('D3', entry.get('d3', 0))))
        d4.append(float(entry.get('D4', entry.get('d4', 0))))
        none_pct.append(float(entry.get('None', entry.get('nothing', 0))))

    # Determine current drought level from the latest entry
    current_level = 'No Drought'
    if dates:
        latest = len(dates) - 1
        if d4[latest] > 0: current_level = 'D4 - Exceptional Drought'
        elif d3[latest] > 0: current_level = 'D3 - Extreme Drought'
        elif d2[latest] > 0: current_level = 'D2 - Severe Drought'
        elif d1[latest] > 0: current_level = 'D1 - Moderate Drought'
        elif d0[latest] > 0: current_level = 'D0 - Abnormally Dry'

    return jsonify({
        'fips': fips,
        'dates': dates,
        'D0': d0, 'D1': d1, 'D2': d2, 'D3': d3, 'D4': d4,
        'none': none_pct,
        'current_level': current_level,
    })


# =====================================================
# Flow Anomaly Detection
# =====================================================

def detect_anomalies(site_no, sf_dir=None, p_dir=None, b_dir=None):
    """Detect extreme low-flow and high-flow events by running BFS on-the-fly."""
    data = run_bfs_on_the_fly(site_no, sf_dir=sf_dir, p_dir=p_dir, b_dir=b_dir)
    if data is None:
        return []

    # Reconstruct arrays from the on-the-fly result (already in m³/s)
    dates_list = data['dates']
    qob_ms = np.array([v if v is not None else np.nan for v in data['qob']], dtype=float)

    # Compute the downsampling step used in run_bfs_on_the_fly so duration
    # thresholds and reported values reflect actual days, not downsampled indices.
    import math
    total_points = data.get('total_points', len(dates_list))
    step = max(1, round(total_points / max(len(dates_list), 1)))
    # Minimum consecutive downsampled points that represent 7 real days
    min_duration = max(1, math.ceil(7 / step))

    # Percentile thresholds
    p5 = np.nanpercentile(qob_ms, 5)
    p95 = np.nanpercentile(qob_ms, 95)

    events = []

    # Find consecutive runs below/above threshold
    for label, mask, threshold in [
        ('low_flow', qob_ms <= p5, p5),
        ('high_flow', qob_ms >= p95, p95),
    ]:
        runs = []
        in_run = False
        start_idx = 0
        for i in range(len(mask)):
            if mask[i] and not in_run:
                in_run = True
                start_idx = i
            elif not mask[i] and in_run:
                in_run = False
                runs.append((start_idx, i - 1))
        if in_run:
            runs.append((start_idx, len(mask) - 1))

        for s, e in runs:
            duration = e - s + 1
            if duration < min_duration:
                continue
            actual_days = duration * step  # convert to real calendar days
            segment = qob_ms[s:e+1]
            event = {
                'type': label,
                'start_date': str(dates_list[s])[:10],
                'end_date': str(dates_list[e])[:10],
                'duration': actual_days,
                'min_flow': round(float(np.nanmin(segment)), 4),
                'max_flow': round(float(np.nanmax(segment)), 4),
                'mean_flow': round(float(np.nanmean(segment)), 4),
                'threshold': round(float(threshold), 4),
                'severity': 0.0,
                'drought_context': '',
            }
            # Severity score: actual days weighted by how far from threshold
            if label == 'low_flow' and threshold > 0:
                event['severity'] = actual_days * (1 - event['mean_flow'] / threshold)
            else:
                event['severity'] = actual_days * (event['mean_flow'] / max(threshold, 0.001))
            events.append(event)

    # Sort by severity descending, keep top 10
    events.sort(key=lambda x: x['severity'], reverse=True)
    events = events[:10]

    # Try to correlate with drought data
    info = site_info.get(site_no)
    if info:
        try:
            fips = get_county_fips(info['lat'], info['lng'])
            if fips:
                # Fetch full drought record once
                drought_raw = fetch_drought_data(fips, '1/4/2000',
                                                 datetime.now().strftime('%-m/%-d/%Y'))
                if drought_raw:
                    # Build a lookup: date_str -> worst drought level
                    drought_lookup = {}
                    for entry in drought_raw:
                        md = str(entry.get('MapDate', entry.get('mapDate', '')))
                        if len(md) == 8:
                            dkey = md[:4] + '-' + md[4:6] + '-' + md[6:8]
                        else:
                            dkey = md
                        d4v = float(entry.get('D4', entry.get('d4', 0)))
                        d3v = float(entry.get('D3', entry.get('d3', 0)))
                        d2v = float(entry.get('D2', entry.get('d2', 0)))
                        d1v = float(entry.get('D1', entry.get('d1', 0)))
                        d0v = float(entry.get('D0', entry.get('d0', 0)))
                        if d4v > 0: drought_lookup[dkey] = ('D4', d4v)
                        elif d3v > 0: drought_lookup[dkey] = ('D3', d3v)
                        elif d2v > 0: drought_lookup[dkey] = ('D2', d2v)
                        elif d1v > 0: drought_lookup[dkey] = ('D1', d1v)
                        elif d0v > 0: drought_lookup[dkey] = ('D0', d0v)

                    drought_dates = sorted(drought_lookup.keys())

                    for event in events:
                        estart = event['start_date']
                        eend = event['end_date']
                        # Find worst drought during event window
                        worst_level = None
                        worst_pct = 0
                        level_order = {'D0': 0, 'D1': 1, 'D2': 2, 'D3': 3, 'D4': 4}
                        for dd in drought_dates:
                            if estart <= dd <= eend:
                                lv, pct = drought_lookup[dd]
                                if worst_level is None or level_order[lv] > level_order.get(worst_level, -1):
                                    worst_level = lv
                                    worst_pct = pct
                        if worst_level:
                            labels = {'D0': 'Abnormally Dry', 'D1': 'Moderate Drought',
                                      'D2': 'Severe Drought', 'D3': 'Extreme Drought',
                                      'D4': 'Exceptional Drought'}
                            event['drought_context'] = (
                                f"This coincided with {worst_level} ({labels[worst_level]}) "
                                f"conditions affecting {worst_pct:.0f}% of the county."
                            )
                            event['drought_level'] = worst_level
        except Exception:
            pass  # Drought correlation is best-effort

    # Generate narrative text
    for event in events:
        if event['type'] == 'low_flow':
            narrative = (
                f"Extreme low-flow period from {event['start_date']} to {event['end_date']} "
                f"({event['duration']} days). Flow dropped to {event['min_flow']} m\u00b3/s, "
                f"well below the 5th percentile threshold of {event['threshold']} m\u00b3/s."
            )
        else:
            narrative = (
                f"Extreme high-flow event from {event['start_date']} to {event['end_date']} "
                f"({event['duration']} days). Peak flow reached {event['max_flow']} m\u00b3/s, "
                f"exceeding the 95th percentile threshold of {event['threshold']} m\u00b3/s."
            )
        if event['drought_context']:
            narrative += ' ' + event['drought_context']
        event['narrative'] = narrative

    return events


@app.route('/api/gage/<site_no>/anomalies')
def api_gage_anomalies(site_no):
    """Detect and narrate flow anomalies for a gage."""
    if site_no not in site_info:
        return jsonify({'error': 'Gage not found'}), 404
    if get_gage_status(site_no) != 'calibrated':
        return jsonify({'error': 'Not calibrated. Run calibration first.'}), 400

    # Prefer temp (updated/recalibrated) data if available, otherwise use main dirs
    temp_site_dir = os.path.join(TEMP_DIR, site_no)
    temp_sf = os.path.join(temp_site_dir, f"{site_no}.csv")
    main_sf = os.path.join(STREAMFLOW_DIR, f"{site_no}.csv")

    if os.path.exists(temp_sf):
        sf_dir, p_dir, b_dir = temp_site_dir, temp_site_dir, temp_site_dir
    elif os.path.exists(main_sf):
        sf_dir, p_dir, b_dir = None, None, None  # use defaults
    else:
        return jsonify({'error': 'No streamflow data for this gage. Click Update to fetch it from USGS first.'}), 404

    try:
        events = detect_anomalies(site_no, sf_dir=sf_dir, p_dir=p_dir, b_dir=b_dir)
        return jsonify({'events': events})
    except Exception as e:
        return jsonify({'error': f'Anomaly detection failed: {str(e)}'}), 500


# =====================================================
# Skill Assessment
# =====================================================

def compute_skill_assessment(site_no, sf_dir=None, p_dir=None, b_dir=None):
    """Run separation and forecast skill assessment, returning a JSON-ready dict."""
    sf_dir = sf_dir or STREAMFLOW_DIR
    p_dir = p_dir or PARAMS_DIR
    b_dir = b_dir or BFF_DIR

    params_path = os.path.join(p_dir, f"params_{site_no}.csv")
    streamflow_path = os.path.join(sf_dir, f"{site_no}.csv")

    if not all(os.path.exists(p) for p in [params_path, streamflow_path]):
        return None

    basin_char, gw_hyd, flow = load_site_params(params_path)
    streamflow_df = load_streamflow(streamflow_path)

    if len(streamflow_df) < 365:
        return None

    lb, x1, wb, por = basin_char[1], basin_char[2], basin_char[3], basin_char[4]
    beta, kb = gw_hyd[1], gw_hyd[3]
    SBT = pybfs.base_table(lb, x1, wb, beta, kb, streamflow_df, por)

    # Separation skill
    sep_skill_df, sep_metrics = separation_skill(streamflow_df, SBT, basin_char, gw_hyd, flow)

    sep_dates = pd.to_datetime(sep_skill_df['Date']).dt.strftime('%Y-%m-%d').tolist()
    sep_q = (sep_skill_df['Q'] / 86400).round(4).tolist()
    sep_bf_bfs = (sep_skill_df['BF_bfs'] / 86400).round(4).tolist()
    sep_res = (sep_skill_df['RES'].where(sep_skill_df['BF_strict']) / 86400).round(4).tolist()
    sep_strict = sep_skill_df['BF_strict'].tolist()

    # Forecast skill
    fc_skill_df, fc_summary_df, fc_metrics = forecast_skill(streamflow_df, SBT, basin_char, gw_hyd, flow)

    fc_dates = pd.to_datetime(fc_skill_df['Date']).dt.strftime('%Y-%m-%d').tolist()
    fc_q = (fc_skill_df['Q'] / 86400).round(4).tolist()
    fc_fc = (fc_skill_df['FC'] / 86400).round(4).tolist()
    fc_res = (fc_skill_df['RES'] / 86400).round(4).tolist()
    fc_seq = fc_skill_df['SEQ'].tolist()

    fc_summary = []
    for _, row in fc_summary_df.iterrows():
        fc_summary.append({
            'seq': int(row['SEQ']),
            'start_date': pd.to_datetime(row['START']).strftime('%Y-%m-%d'),
            'end_date': pd.to_datetime(row['END']).strftime('%Y-%m-%d'),
            'len': int(row['LEN']),
            'sat': round(float(row['SAT']), 3),
            'rmse': round(float(row['RMSE']) / 86400, 4) if not np.isnan(row['RMSE']) else None,
            'mae': round(float(row['MAE']) / 86400, 4) if not np.isnan(row['MAE']) else None,
        })

    return {
        'separation': {
            'rmse': round(sep_metrics['RMSE'] / 86400, 4) if sep_metrics['RMSE'] is not None and not np.isnan(sep_metrics['RMSE']) else None,
            'mae': round(sep_metrics['MAE'] / 86400, 4) if sep_metrics['MAE'] is not None and not np.isnan(sep_metrics['MAE']) else None,
            'n_days': sep_metrics['n_days'],
            'frac_strict': round(sep_metrics['frac_strict'], 4) if sep_metrics['frac_strict'] is not None and not np.isnan(sep_metrics['frac_strict']) else None,
            'dates': sep_dates,
            'q': sep_q,
            'bf_bfs': sep_bf_bfs,
            'residuals': sep_res,
            'strict_mask': sep_strict,
        },
        'forecast': {
            'overall_rmse': round(fc_metrics['overall_RMSE'] / 86400, 4) if fc_metrics['overall_RMSE'] is not None and not np.isnan(fc_metrics['overall_RMSE']) else None,
            'overall_mae': round(fc_metrics['overall_MAE'] / 86400, 4) if fc_metrics['overall_MAE'] is not None and not np.isnan(fc_metrics['overall_MAE']) else None,
            'n_sequences': len(fc_summary),
            'dates': fc_dates,
            'q': fc_q,
            'fc': fc_fc,
            'residuals': fc_res,
            'seq': fc_seq,
            'summary': fc_summary,
        },
    }


@app.route('/api/gage/<site_no>/skill')
def api_gage_skill(site_no):
    """Run separation and forecast skill assessment for a gage."""
    if site_no not in site_info:
        return jsonify({'error': 'Gage not found'}), 404
    if get_gage_status(site_no) != 'calibrated':
        return jsonify({'error': 'Not calibrated. Run calibration first.'}), 400

    temp_site_dir = os.path.join(TEMP_DIR, site_no)
    temp_sf = os.path.join(temp_site_dir, f"{site_no}.csv")
    main_sf = os.path.join(STREAMFLOW_DIR, f"{site_no}.csv")

    if os.path.exists(temp_sf):
        sf_dir, p_dir, b_dir = temp_site_dir, temp_site_dir, temp_site_dir
    elif os.path.exists(main_sf):
        sf_dir, p_dir, b_dir = None, None, None
    else:
        return jsonify({'error': 'No streamflow data for this gage. Click Update to fetch it from USGS first.'}), 404

    try:
        result = compute_skill_assessment(site_no, sf_dir=sf_dir, p_dir=p_dir, b_dir=b_dir)
        if result is None:
            return jsonify({'error': 'Insufficient data for skill assessment (need at least 365 days).'}), 400
        return safe_jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Skill assessment failed: {str(e)}'}), 500


# ----- Startup -----
load_site_info_data()
load_nwm_data()
load_low_flow_data()
load_metrics_data()

if __name__ == '__main__':
    os.makedirs(TEMP_DIR, exist_ok=True)
    app.run(debug=True, port=5000, threaded=True)
