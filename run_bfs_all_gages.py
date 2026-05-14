import pandas as pd
import numpy as np
import os
import sys
import traceback
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import plotly.graph_objects as go

import pybfs

# Unit conversions
CFS_TO_M3_PER_DAY = 0.0283168 * 86400  # 1 cfs = 2446.58 m³/day
FORECAST_DAYS = 90  # Number of days to forecast beyond last observation


def load_site_params(params_path):
    """Load calibrated parameters from a per-site params CSV."""
    df = pd.read_csv(params_path)
    row = df.iloc[0]

    area = row['tmp.area']
    basin_char = [area, row['Lb'], row['X1'], row['Wb'], row['POR']]
    gw_hyd = [row['ALPHA'], row['BETA'], row['Ks'], row['Kb'], row['Kz']]
    flow = [row['Qthresh'], row['Rs'], row['Rb1'], row['Rb2'], row['Prec'], row['Frac4Rise']]

    return basin_char, gw_hyd, flow


def load_streamflow(streamflow_path):
    """Load and prepare streamflow data for pybfs."""
    df = pd.read_csv(streamflow_path, dtype={'date': str, 'streamflow': str})
    df['Date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['Streamflow'] = pd.to_numeric(df['streamflow'], errors='coerce') * CFS_TO_M3_PER_DAY
    df = df.dropna(subset=['Streamflow'])
    df = df[['Date', 'Streamflow']].sort_values('Date').reset_index(drop=True)
    return df


def plot_baseflow_separation(streamflow_df, bfs_out, site_no, save_path):
    """Plot observed streamflow vs baseflow + surface flow + direct runoff."""
    fig, ax = plt.subplots(figsize=(14, 6))

    dates = pd.to_datetime(bfs_out['Date'])
    qob = bfs_out['Qob'] / 86400  # m³/day -> m³/s
    baseflow = bfs_out['Baseflow'] / 86400
    surface = bfs_out['SurfaceFlow'] / 86400
    direct = bfs_out['DirectRunoff'] / 86400

    ax.plot(dates, qob, color='black', linewidth=0.8, label='Observed Streamflow')
    ax.fill_between(dates, 0, baseflow, color='#2196F3', alpha=0.5, label='Baseflow')
    ax.fill_between(dates, baseflow, baseflow + surface, color='#4CAF50', alpha=0.5, label='Surface Flow')
    ax.fill_between(dates, baseflow + surface, baseflow + surface + direct, color='#FF9800', alpha=0.4, label='Direct Runoff')

    ax.set_xlabel('Date')
    ax.set_ylabel('Flow (m\u00b3/s)')
    ax.set_title(f'Baseflow Separation - Site {site_no}')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_flow_components_pie(bfs_out, bff_df, site_no, save_path):
    """Pie chart of baseflow, surface flow, and direct runoff fractions."""
    fig, ax = plt.subplots(figsize=(7, 7))

    row = bff_df.iloc[0]
    fractions = [row['BFF'], row['SFF'], row['DRF']]
    labels = [
        f'Baseflow ({fractions[0]:.1%})',
        f'Surface Flow ({fractions[1]:.1%})',
        f'Direct Runoff ({fractions[2]:.1%})'
    ]
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    ax.pie(fractions, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 11})
    ax.set_title(f'Flow Component Fractions - Site {site_no}')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_forecast_result(streamflow_df, bfs_out, forecast_out, site_no, save_path):
    """Interactive Plotly forecast: last year of training + forecast period, saved as HTML."""
    train_dates = pd.to_datetime(bfs_out['Date'])
    cutoff = train_dates.iloc[-1] - pd.Timedelta(days=365)
    mask = train_dates >= cutoff

    fc_dates = pd.to_datetime(forecast_out['Date'])
    fc_start = fc_dates.iloc[0]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_dates[mask],
        y=bfs_out.loc[mask, 'Qob'] / 86400,
        name='Observed Streamflow',
        mode='lines',
        line=dict(color='#37474F', width=1),
    ))

    fig.add_trace(go.Scatter(
        x=train_dates[mask],
        y=bfs_out.loc[mask, 'Baseflow'] / 86400,
        name='Baseflow (training)',
        mode='lines',
        line=dict(color='#2196F3', width=2),
    ))

    fig.add_trace(go.Scatter(
        x=fc_dates,
        y=forecast_out['Baseflow'] / 86400,
        name='Baseflow (forecast)',
        mode='lines',
        line=dict(color='#2196F3', width=2, dash='dash'),
    ))

    fig.add_vline(
        x=fc_start,
        line=dict(color='gray', dash='dot', width=1.5),
        annotation_text='Forecast start',
        annotation_position='top right',
    )

    fig.update_layout(
        title=f'Baseflow Forecast ({FORECAST_DAYS} days) \u2014 Site {site_no}',
        xaxis_title='Date',
        yaxis_title='Flow (m\u00b3/s)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
        template='plotly_white',
        height=500,
    )

    fig.write_html(save_path)


def plot_confidence_intervals(bfs_out, ci_df, site_no, save_path):
    """Plot baseflow with confidence interval band."""
    fig, ax = plt.subplots(figsize=(14, 6))

    dates = pd.to_datetime(bfs_out['Date'])
    qob = bfs_out['Qob'] / 86400

    ax.plot(dates, qob, color='black', linewidth=0.8, label='Observed Streamflow')

    if 'Qci.05' in ci_df.columns and 'Qci.95' in ci_df.columns:
        ci_lower = ci_df['Qci.05'] / 86400
        ci_upper = ci_df['Qci.95'] / 86400
        ax.fill_between(dates[:len(ci_lower)], ci_lower, ci_upper,
                        color='#2196F3', alpha=0.25, label='90% Credible Interval')

    baseflow = bfs_out['Baseflow'] / 86400
    ax.plot(dates, baseflow, color='#2196F3', linewidth=1, label='Baseflow')

    ax.set_xlabel('Date')
    ax.set_ylabel('Flow (m\u00b3/s)')
    ax.set_title(f'Baseflow with Credible Intervals - Site {site_no}')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_annual_baseflow_index(bfs_out, site_no, save_path):
    """Bar chart of annual Baseflow Index (BFI = baseflow/total flow) per year."""
    df = bfs_out[['Date', 'Qob', 'Baseflow']].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year

    annual = df.groupby('Year').agg({'Qob': 'sum', 'Baseflow': 'sum'})
    annual['BFI'] = annual['Baseflow'] / annual['Qob']
    annual['BFI'] = annual['BFI'].clip(0, 1)
    # Drop years with very little data
    annual = annual[annual['Qob'] > 0]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(annual.index, annual['BFI'], color='#2196F3', alpha=0.7)
    ax.axhline(annual['BFI'].mean(), color='red', linestyle='--', linewidth=1,
               label=f'Mean BFI = {annual["BFI"].mean():.3f}')
    ax.set_xlabel('Year')
    ax.set_ylabel('Baseflow Index (BFI)')
    ax.set_title(f'Annual Baseflow Index - Site {site_no}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def process_site(site_no, streamflow_dir, params_dir, bff_dir, output_dir):
    """
    Run full BFS analysis for a single site:
    1. Baseflow separation
    2. Flow component fractions (pie chart)
    3. Annual BFI
    4. Confidence intervals
    5. Baseflow forecast
    6. Save all CSV outputs and plots
    """
    site_dir = os.path.join(output_dir, site_no)
    os.makedirs(site_dir, exist_ok=True)

    # Check if already processed (done marker)
    done_marker = os.path.join(site_dir, '.done')
    if os.path.exists(done_marker):
        return True

    # Load data
    params_path = os.path.join(params_dir, f"params_{site_no}.csv")
    bff_path = os.path.join(bff_dir, f"bff_{site_no}.csv")
    streamflow_path = os.path.join(streamflow_dir, f"{site_no}.csv")

    if not all(os.path.exists(p) for p in [params_path, bff_path, streamflow_path]):
        print(f"  Missing files for site {site_no}, skipping")
        return False

    basin_char, gw_hyd, flow = load_site_params(params_path)
    bff_df = pd.read_csv(bff_path)
    streamflow_df = load_streamflow(streamflow_path)

    if len(streamflow_df) < 365:
        print(f"  Site {site_no}: insufficient data ({len(streamflow_df)} days), skipping")
        return False

    # 1. Generate baseflow table and run BFS
    lb, x1, wb, por = basin_char[1], basin_char[2], basin_char[3], basin_char[4]
    beta, kb = gw_hyd[1], gw_hyd[3]

    SBT = pybfs.base_table(lb, x1, wb, beta, kb, streamflow_df, por)
    bfs_out = pybfs.bfs(streamflow_df, SBT, basin_char, gw_hyd, flow)

    # Save BFS results CSV
    bfs_out.to_csv(os.path.join(site_dir, 'bfs_results.csv'), index=False)

    # 2. Baseflow separation plot
    plot_baseflow_separation(streamflow_df, bfs_out, site_no,
                             os.path.join(site_dir, 'baseflow_separation.png'))

    # 3. Flow component pie chart
    plot_flow_components_pie(bfs_out, bff_df, site_no,
                             os.path.join(site_dir, 'flow_fractions.png'))

    # 4. Annual BFI
    plot_annual_baseflow_index(bfs_out, site_no,
                               os.path.join(site_dir, 'annual_bfi.png'))

    # 5. Confidence intervals
    try:
        ci_table, ci_df = pybfs.bf_ci(bfs_out)
        ci_table.to_csv(os.path.join(site_dir, 'ci_table.csv'), index=False)
        ci_df.to_csv(os.path.join(site_dir, 'ci_daily.csv'), index=False)
        plot_confidence_intervals(bfs_out, ci_df, site_no,
                                  os.path.join(site_dir, 'confidence_intervals.png'))
    except Exception:
        pass  # CI may fail for some sites

    # 6. Forecast
    try:
        last_row = bfs_out.iloc[-1]
        ini = (
            last_row['X'],
            last_row['Zb.L'],
            last_row['Zs.L'],
            last_row['StBase'],
            last_row['StSur'],
            last_row['SurfaceFlow'],
            last_row['Baseflow'],
            last_row['Rech'],
        )

        last_date = pd.to_datetime(bfs_out['Date'].iloc[-1])
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                       periods=FORECAST_DAYS, freq='D')
        forecast_input = pd.DataFrame({
            'date': forecast_dates,
            'streamflow': np.nan
        })

        forecast_out = pybfs.forecast(forecast_input, SBT, basin_char, gw_hyd, flow, ini)
        forecast_out.to_csv(os.path.join(site_dir, 'forecast.csv'), index=False)

        plot_forecast_result(streamflow_df, bfs_out, forecast_out, site_no,
                             os.path.join(site_dir, 'forecast.html'))
    except Exception:
        pass  # Forecast may fail for some sites

    # Mark as done
    with open(done_marker, 'w') as f:
        f.write('done')

    return True


def main():
    streamflow_dir = "./usgs_daily_streamflow"
    calib_dir = "./usgs_calibration_results"
    output_dir = "./usgs_bfs_results"

    params_dir = os.path.join(calib_dir, "params")
    bff_dir = os.path.join(calib_dir, "bff")
    os.makedirs(output_dir, exist_ok=True)

    # Get list of calibrated sites
    param_files = [f for f in os.listdir(params_dir)
                   if f.startswith('params_') and f.endswith('.csv')]
    all_sites = sorted([f.replace('params_', '').replace('.csv', '') for f in param_files])
    print(f"Found {len(all_sites)} calibrated sites")

    # Resume: check which are already done
    already_done = set()
    if os.path.exists(output_dir):
        for d in os.listdir(output_dir):
            if os.path.isfile(os.path.join(output_dir, d, '.done')):
                already_done.add(d)

    sites_to_process = [s for s in all_sites if s not in already_done]
    print(f"Already processed: {len(already_done)}")
    print(f"Remaining: {len(sites_to_process)}")

    succeeded = len(already_done)
    failed = 0

    for idx, site_no in enumerate(sites_to_process):
        print(f"\n[{idx + 1}/{len(sites_to_process)}] Processing site {site_no}...")
        try:
            if process_site(site_no, streamflow_dir, params_dir, bff_dir, output_dir):
                succeeded += 1
                print(f"  Site {site_no}: done")
            else:
                failed += 1
        except Exception as e:
            print(f"  Site {site_no}: failed - {e}")
            traceback.print_exc()
            failed += 1

    print(f"\nComplete. {succeeded} succeeded, {failed} failed.")
    print(f"Results saved to: {output_dir}/")
    print(f"Each site folder contains:")
    print(f"  - bfs_results.csv: daily baseflow separation data")
    print(f"  - baseflow_separation.png: streamflow vs flow components")
    print(f"  - flow_fractions.png: pie chart of BFF/SFF/DRF")
    print(f"  - annual_bfi.png: annual baseflow index bar chart")
    print(f"  - confidence_intervals.png: baseflow with 90% credible interval")
    print(f"  - ci_table.csv / ci_daily.csv: credible interval data")
    print(f"  - forecast.csv / forecast.html: {FORECAST_DAYS}-day baseflow forecast (interactive Plotly)")


if __name__ == '__main__':
    main()
