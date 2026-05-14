# Low Flow Analyst (LFA)

An interactive web application for exploring and analyzing baseflow separation across thousands of USGS stream gages in the contiguous United States. Built on the [PyBFS](https://github.com/BYU-Hydroinformatics/pybfs) physically-based algorithm and backed by USGS WaterServices, this tool lets researchers and practitioners calibrate models, visualize flow components, detect anomalies, and fetch live data — all from a browser.

---

## Quick Start

```bash
git clone git@github.com:BYU-Hydroinformatics/low-flow-analyst.git
cd lfa
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. The app ships with pre-calibrated results for 8,729 USGS gages, so no data pipeline steps are required to start exploring.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Pipeline](#data-pipeline)
  - [Step 1 — Download USGS Streamflow](#step-1--download-usgs-streamflow)
  - [Step 2 — Detect Low-Flow Gages](#step-2--detect-low-flow-gages)
  - [Step 3 — Calibrate All Gages](#step-3--calibrate-all-gages)
  - [Step 4 — Pre-compute BFS Results (optional)](#step-4--pre-compute-bfs-results-optional)
  - [Step 5 — Run the App](#step-5--run-the-app)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Directory Structure](#directory-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Low Flow Analyst (LFA) combines a physically-based baseflow separation model with real-time USGS data access into a single Flask application. The map view displays every active USGS stream gage that has daily discharge records, color-coded by calibration status and watershed behavior. Clicking a gage loads interactive Plotly charts without any page reload, running BFS on-the-fly from pre-calibrated parameters stored on disk.

The backend pipeline consists of four standalone scripts that can be run once to bootstrap the dataset, after which the web app serves everything dynamically.

---

## Features

### Baseflow Separation
- **PyBFS algorithm** — physically-based coupled surface/subsurface reservoir model separating total streamflow into baseflow, surface flow, and direct runoff
- **Traditional method comparisons** — Chapman, Lyne-Hollick (LH), and Eckhardt filter overlaid on the same chart
- **90% Bayesian credible intervals** on baseflow estimates via `pybfs.bf_ci`

### Forecasting
- **90-day baseflow forecast** projected from the last observed state, displayed alongside the historical record

### Analytics
- **Annual Baseflow Index (BFI)** — year-by-year bar chart with long-term mean
- **Flow component fractions** — BFF / SFF / DRF displayed as a donut summary
- **Flow anomaly detection** — top extreme low-flow and high-flow events identified by percentile thresholds, ranked by severity, with narrative descriptions

### Data Integration
- **US Drought Monitor (USDM)** — county-level weekly drought severity timeline correlated with detected low-flow anomalies
- **National Water Model (NWM)** classification — Natural vs. Artificial channel behavior
- **GAGES-II** reference / non-reference status for each gage
- **Live USGS update** — fetch the latest full discharge record from USGS WaterServices, recalibrate, and view updated results in the browser without touching local files

### Map Interface
- Leaflet-based interactive map with marker clustering
- **Symbolize by metric** — color-ramp the map by Forecast RMSE, Forecast MAE, number of evaluated sequences, or drainage area; quantile-scaled gradient legend updates in the panel
- Filter panel: calibration status, NWM behavior, GAGES-II class, low-flow gages
- Side panel with gage metadata, calibration trigger, and tabbed chart view

---

## Architecture

```
lfa/
├── app.py                      # Flask application (entry point)
├── usgs_download_all_daily.py  # Pipeline step 1: download USGS data
├── detect_low_flow_gages.py    # Pipeline step 2: classify low-flow gages
├── calibrate_all_gages.py      # Pipeline step 3: calibrate BFS parameters
├── run_bfs_all_gages.py        # Pipeline step 4: pre-compute results & plots
├── templates/
│   └── index.html              # Single-page map application
├── behavior/
│   ├── NWM_USGS_Natural_Flow.csv
│   ├── NWM_USGS_Artificial_Path.csv
│   └── GAGES-II_ref_non_ref.csv
├── pybfs/                      # PyBFS library (submodule)
├── baseflow/                   # Baseflow library (submodule)
├── usgs_daily_streamflow/      # Downloaded gage CSVs (generated)
├── usgs_calibration_results/   # Calibrated parameters (generated)
├── usgs_bfs_results/           # Pre-computed plots & CSVs (generated)
├── temp_results/               # Live-update working directory (generated)
└── low_flow_gages.csv          # Low-flow classification (generated)
```

The app reads calibration parameters from `usgs_calibration_results/params/` and runs BFS in-process on every chart request. Results are cached in memory for 5 minutes (`BFS_CACHE_TTL`). The "Update Data" feature runs a full download + calibration cycle in a background thread, writing to `temp_results/` so original data is never modified.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.9 |
| pip | ≥ 22 |

Python package dependencies (install via `requirements.txt`):

```
flask
pandas
numpy
matplotlib
scipy
statsmodels
numba
tqdm
plotly
```

The `pybfs` and `baseflow` libraries ship as subdirectories under the project root and are added to `sys.path` automatically by `app.py`.

---

## Installation

```bash
# Clone the repository
git clone git@github.com:BYU-Hydroinformatics/low-flow-analyst.git
cd lfa

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `numba` requires a C compiler. On macOS install Xcode Command Line Tools (`xcode-select --install`). On Linux ensure `gcc` is available.

---

## Data Pipeline

Run these scripts once to build the dataset. Each step is idempotent — re-running skips already-completed work so you can safely resume after interruptions.

### Step 1 — Download USGS Streamflow

Downloads all available daily discharge records for every active USGS stream gage in CONUS (48 states + DC).

```bash
python usgs_download_all_daily.py
```

**Output** (`usgs_daily_streamflow/`):
- `site_info.csv` — gage metadata (coordinates, drainage area, …)
- `<site_no>.csv` — one CSV per gage with `date` and `streamflow` (cfs) columns
- `gauges_with_data.csv` — list of gages that returned data
- `gauges_without_data.csv` — list of gages with no available records

This step queries the USGS WaterServices REST API state-by-state with a 1-second delay between requests and a 0.5-second delay per gage download to respect rate limits. Expect several hours for a full CONUS run (~10 000 gages).

---

### Step 2 — Detect Low-Flow Gages

Classifies gages that exhibit sustained low-flow periods (e.g. regulated rivers, intermittent streams). Uses multiprocessing to analyze all gages in parallel.

```bash
python detect_low_flow_gages.py [--pct 10] [--cv 0.15] [--dur 60]
```

| Flag | Default | Description |
|---|---|---|
| `--pct` | 10 | Percentile threshold (flow below this is "low") |
| `--cv` | 0.15 | Max coefficient of variation to consider flow "constant" |
| `--dur` | 60 | Minimum consecutive days below threshold |

**Output:** `low_flow_gages.csv` with columns `site_no`, `has_lowflow`, `max_lowflow_duration`, `max_run_cv`, `threshold_cfs`, `record_days`.

---

### Step 3 — Calibrate All Gages

Runs the PyBFS calibration routine for every gage that has both streamflow data and a known drainage area. Saves per-site parameter files used by the web app. Skips already-calibrated sites and logs failures for easy resumption.

```bash
python calibrate_all_gages.py
```

**Output** (`usgs_calibration_results/`):
- `params/params_<site_no>.csv` — calibrated basin and groundwater parameters
- `bff/bff_<site_no>.csv` — baseflow, surface flow, and direct runoff fractions
- `all_params.csv` — combined parameter table for all sites
- `all_bff.csv` — combined BFF table
- `calibration_failed.csv` — sites that could not be calibrated

Calibration takes roughly 60–90 seconds per gage on a modern CPU.

---

### Step 4 — Pre-compute BFS Results (optional)

Generates static PNG plots and result CSVs for every calibrated gage. The web app runs BFS on-the-fly, so this step is optional but useful for producing publication-quality figures in bulk.

```bash
python run_bfs_all_gages.py
```

**Output per gage** (`usgs_bfs_results/<site_no>/`):
- `bfs_results.csv` — daily baseflow separation time series
- `baseflow_separation.png` — observed streamflow vs. flow components
- `flow_fractions.png` — BFF / SFF / DRF pie chart
- `annual_bfi.png` — annual Baseflow Index bar chart
- `confidence_intervals.png` — baseflow with 90% credible interval band
- `ci_table.csv` / `ci_daily.csv` — credible interval tables
- `forecast.html` / `forecast.csv` — 90-day baseflow forecast (interactive Plotly)

A `.done` marker file is written per site so the script can be interrupted and resumed safely.

---

### Step 5 — Run the App

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser. The app starts by loading all gage metadata, NWM behavior data, and low-flow classifications into memory.

For production deployments, serve with Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## Usage

### Map View

The map loads all gages as clustered markers. The default symbolization shows calibration status:

| Color | Meaning |
|---|---|
| Green | Calibrated — charts available |
| Orange | Pending — calibration not yet run |

Use the **Symbolize by** dropdown in the legend to switch to a continuous color ramp based on:
- **Forecast RMSE / MAE** — green (low error) → yellow → red (high error)
- **N Sequences** — light → dark blue (number of forecast windows evaluated)
- **Drainage Area** — light → dark green (mi²)

Grey markers indicate gages with no data for the selected metric. The legend shows a gradient bar with p5–p95 range labels.

Use the **filter toggles** to show only:
- Reference-only gages (GAGES-II)
- Natural-channel gages (NWM classification)
- Low-flow gages
- Favorited gages

### Gage Detail Panel

Click any marker to open the side panel:

1. **Info tab** — site name, USGS ID, coordinates, drainage area, NWM behavior, GAGES-II class, calibration status
2. **Charts tab** — interactive Plotly charts loaded on demand:
   - Baseflow separation time series with traditional method overlay
   - Flow component fractions (BFF / SFF / DRF)
   - Annual BFI bar chart
   - Confidence interval band
   - 90-day forecast
   - County drought monitor timeline
3. **Anomalies tab** — ranked extreme flow events with narrative descriptions and drought context

### Calibrating a Gage

If a gage shows "Pending," click **Run Calibration**. A progress bar tracks the background process (~90 seconds). Once complete, the charts load automatically.

### Live Data Update

Click **Update Data** on any gage to:
1. Download the complete historical record fresh from USGS
2. Re-calibrate on the updated data
3. View charts based on the new calibration (written to `temp_results/`)

Original calibration files are never overwritten.

---

## API Reference

All endpoints return JSON unless noted otherwise.

### GET `/api/gages`

Returns all gages with coordinates, status, NWM behavior, low-flow flag, drainage area, and pre-computed forecast skill metrics.

```json
[
  {
    "site_no": "12167000",
    "name": "SAUK RIVER NEAR SAUK CITY, WA",
    "lat": 48.47,
    "lng": -121.59,
    "status": "calibrated",
    "behavior": "Natural",
    "ref_status": "Ref",
    "has_lowflow": false,
    "drain_area_sqmi": 714.0,
    "overall_RMSE": 42381.5,
    "overall_MAE": 31204.8,
    "n_sequences": 97
  }
]
```

Metric fields are `null` for gages not covered by `forecast_skill/output/metrics.csv`.

### GET `/api/gage/<site_no>/info`

Returns detailed metadata and BFF/SFF/DRF fractions for a single gage.

### GET `/api/gage/<site_no>/data?source=original|updated`

Runs BFS on-the-fly and returns time series data for all interactive charts:

```json
{
  "dates": ["2000-01-01", "..."],
  "qob": [12.4, "..."],
  "baseflow": [8.1, "..."],
  "surface_flow": [3.1, "..."],
  "direct_runoff": [1.2, "..."],
  "ci_lower": [7.5, "..."],
  "ci_upper": [8.7, "..."],
  "forecast_dates": ["2025-01-01", "..."],
  "forecast_baseflow": [7.9, "..."],
  "annual_years": [2000, 2001, "..."],
  "annual_bfi": [0.652, "..."],
  "mean_bfi": 0.641,
  "bff": 0.641, "sff": 0.253, "drf": 0.106
}
```

Large datasets (> 5 000 points) are automatically downsampled for chart performance; `total_points` reports the full record length.

### POST `/api/gage/<site_no>/process`

Triggers background calibration. Returns `{"status": "started"}` or `{"status": "already_done"}`.

### GET `/api/gage/<site_no>/progress?source=original|updated`

Server-Sent Events (SSE) stream reporting calibration / update progress:

```json
{"stage": "calibrating", "progress": 10, "message": "Calibrating...", "done": false}
```

### POST `/api/gage/<site_no>/update`

Fetches the latest USGS data and runs a full calibration cycle in `temp_results/`.

### GET `/api/gage/<site_no>/drought`

Returns US Drought Monitor weekly severity percentages (D0–D4) for the gage's county since January 2000, plus the current drought level string.

### GET `/api/gage/<site_no>/anomalies`

Detects and ranks extreme low-flow and high-flow events, correlates with drought data, and returns narrative descriptions.

```json
{
  "events": [
    {
      "type": "low_flow",
      "start_date": "2015-07-01",
      "end_date": "2015-09-28",
      "duration": 89,
      "min_flow": 0.12,
      "threshold": 0.43,
      "severity": 58.3,
      "drought_context": "This coincided with D3 (Extreme Drought) conditions affecting 74% of the county.",
      "narrative": "Extreme low-flow period from 2015-07-01 to 2015-09-28..."
    }
  ]
}
```

### GET `/plots/<site_no>/<filename>?source=original|updated`

Serves pre-computed static plot PNGs.

---

## Directory Structure

```
lfa/
├── app.py                          # Flask application
├── usgs_download_all_daily.py      # Step 1: download USGS data
├── detect_low_flow_gages.py        # Step 2: low-flow classification
├── calibrate_all_gages.py          # Step 3: BFS calibration
├── run_bfs_all_gages.py            # Step 4: batch BFS + plots
│
├── templates/
│   └── index.html                  # Map SPA (Leaflet + Plotly)
│
├── behavior/                       # Static reference datasets
│   ├── NWM_USGS_Natural_Flow.csv
│   ├── NWM_USGS_Artificial_Path.csv
│   └── GAGES-II_ref_non_ref.csv
│
├── forecast_skill/                 # Forecast skill evaluation
│   ├── run_forecast_skill.py       # Batch script to compute metrics
│   └── output/metrics.csv          # Pre-computed per-gage RMSE / MAE / N sequences
│
├── pybfs/                          # PyBFS library
├── baseflow/                       # Baseflow separation library
│
├── low_flow_gages.csv              # Generated by step 2
│
├── usgs_daily_streamflow/          # Generated by step 1
│   ├── site_info.csv
│   ├── gauges_with_data.csv
│   └── <site_no>.csv  (one per gage)
│
├── usgs_calibration_results/       # Generated by step 3
│   ├── params/params_<site_no>.csv
│   ├── bff/bff_<site_no>.csv
│   ├── all_params.csv
│   ├── all_bff.csv
│   └── calibration_failed.csv
│
├── usgs_bfs_results/               # Generated by step 4 (optional)
│   └── <site_no>/
│       ├── bfs_results.csv
│       ├── baseflow_separation.png
│       ├── flow_fractions.png
│       ├── annual_bfi.png
│       ├── confidence_intervals.png
│       ├── forecast.png
│       └── ci_table.csv
│
└── temp_results/                   # Live-update scratch space
    └── <site_no>/
```

---

## Configuration

Key constants at the top of `app.py`:

| Constant | Default | Description |
|---|---|---|
| `FORECAST_DAYS` | 90 | Number of days projected in the baseflow forecast |
| `BFS_CACHE_TTL` | 300 | Seconds to cache on-the-fly BFS results in memory |
| `CFS_TO_M3_PER_DAY` | 2446.58 | Unit conversion factor (cfs → m³/day) |
| `SQMI_TO_M2` | 2 589 988.11 | Unit conversion factor (mi² → m²) |

Directory paths (`STREAMFLOW_DIR`, `CALIB_DIR`, `RESULTS_DIR`, `TEMP_DIR`) are derived from the location of `app.py` and can be adjusted at the top of the file if you keep data in a different location.

---

## Contributing

1. Fork the repository and create a feature branch.
2. Follow existing code style (no external formatter required).
3. Keep new dependencies minimal — the app intentionally avoids heavy frameworks.
4. Open a pull request with a clear description of the change and any relevant screenshots.

For bug reports, please include the USGS site number that triggered the issue, the Python version, and the full traceback.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

The [PyBFS](pybfs/) and [baseflow](baseflow/) libraries ship as subdirectories and carry their own respective licenses.

---

## Acknowledgements

- **PyBFS** — BYU Hydroinformatics, physically-based baseflow separation algorithm
- **USGS WaterServices** — daily streamflow data and site metadata
- **US Drought Monitor (USDM)** — county-level drought severity data
- **FCC Census Bureau Area API** — lat/lon to county FIPS conversion
- **National Water Model (NWM)** — channel behavior classification
- **GAGES-II** — reference/non-reference gage classification
