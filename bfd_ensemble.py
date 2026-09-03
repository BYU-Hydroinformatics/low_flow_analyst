"""
BFD Ensemble panel support.

Wires the LFA app to the shared `bfd_db` labeling code in us_baseflow_db
(RF-BFD classifier, baseflowx digital filters, PyBFS) so every gage can show
which "voters" call each day baseflow-dominant, with dmin/t_c/threshold/
voting-fraction all adjustable at request time.

Design: separation itself (RF predict, the 4 baseflowx filters incl. Boughton
calibration, PyBFS) is the expensive part and does not depend on the tunable
parameters, so it is run once per gage and cached to disk as raw per-method
baseflow estimates. Turning those into 0/1 labels, an ensemble vote, and
pooled events is cheap arithmetic done fresh on every request.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

BFD_DB_SRC = "/Users/amin/Downloads/research/projects/us_baseflow_db/src"

import sys
if BFD_DB_SRC not in sys.path:
    sys.path.insert(0, BFD_DB_SRC)

from bfd_db.config import PipelineConfig
from bfd_db.labeling import baseflowx_methods, pybfs_labeling, rf_bfd
from bfd_db.events.pooling import pool_events

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STREAMFLOW_DIR = os.path.join(BASE_DIR, "usgs_daily_streamflow")
CACHE_DIR = os.path.join(BASE_DIR, "bfd_ensemble_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_DEFAULT_CONFIG = PipelineConfig()

# Ratio-based voters (raw qb_<method> cached, threshold applied at request time).
RATIO_METHODS = list(_DEFAULT_CONFIG.baseflowx_methods)  # eckhardt, chapman, lyne_hollick, boughton

VOTER_LABELS = {
    "rf_bfd": "RF-BFD",
    "eckhardt": "Eckhardt",
    "chapman": "Chapman",
    "lyne_hollick": "Lyne-Hollick",
    "boughton": "Boughton",
    "pybfs": "PyBFS",
}

_rf_model = None
_rf_scaler = None


def _get_rf_model():
    global _rf_model, _rf_scaler
    if _rf_model is None:
        _rf_model, _rf_scaler = rf_bfd.load_rf_bfd_model()
    return _rf_model, _rf_scaler


class GageNotFoundError(Exception):
    pass


def _load_gage_q(site_no: str) -> pd.Series:
    """Load daily streamflow (cfs) for one gage as a DatetimeIndex Series."""
    path = os.path.join(STREAMFLOW_DIR, f"{site_no}.csv")
    if not os.path.exists(path):
        raise GageNotFoundError(f"No streamflow data for gage {site_no}")

    df = pd.read_csv(path, dtype={"date": str, "streamflow": str})
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["streamflow"] = pd.to_numeric(df["streamflow"], errors="coerce")
    df = df.dropna(subset=["streamflow"]).sort_values("date")
    return df.set_index("date")["streamflow"]


def get_or_compute_raw(site_no: str, force: bool = False) -> pd.DataFrame:
    """Return the cached (or freshly computed) raw per-method voter table.

    Columns: Q, rf_bfd (fixed 0/1 label), qb_eckhardt, qb_chapman,
    qb_lyne_hollick, qb_boughton, qb_pybfs (present only if PyBFS is
    calibrated for this gage).
    """
    cache_path = os.path.join(CACHE_DIR, f"{site_no}.parquet")
    if os.path.exists(cache_path) and not force:
        return pd.read_parquet(cache_path)

    q = _load_gage_q(site_no)

    rf_model, rf_scaler = _get_rf_model()
    rf_label = rf_bfd.label_gage(q, rf_model, rf_scaler)

    bfx_df = baseflowx_methods.label_gage(
        q, methods=RATIO_METHODS, bf_ratio_threshold=_DEFAULT_CONFIG.bf_ratio_threshold
    )

    raw = pd.DataFrame(index=q.index)
    raw["Q"] = q
    raw["rf_bfd"] = rf_label
    for method in RATIO_METHODS:
        raw[f"qb_{method}"] = bfx_df[f"qb_{method}"]

    pybfs_params = pybfs_labeling.load_pybfs_params(site_no)
    if pybfs_params is not None:
        try:
            raw["qb_pybfs"] = pybfs_labeling.run_pybfs_separation(q, pybfs_params)
        except Exception:
            pass  # gage has params but PyBFS failed to converge; drop that voter

    raw.to_parquet(cache_path)
    return raw


def apply_params(
    raw: pd.DataFrame,
    bf_ratio_threshold: float,
    voting_fraction: float,
    t_c: int,
    d_min: int,
) -> dict:
    """Derive labels/ensemble/events from cached raw voter data for the given params."""
    labels = pd.DataFrame(index=raw.index)
    labels["rf_bfd"] = raw["rf_bfd"]
    for method in RATIO_METHODS:
        col = f"qb_{method}"
        if col in raw.columns:
            ratio = raw[col] / raw["Q"].replace(0, np.nan)
            labels[method] = (ratio >= bf_ratio_threshold).astype(int).fillna(0)
    if "qb_pybfs" in raw.columns:
        ratio = raw["qb_pybfs"] / raw["Q"].replace(0, np.nan)
        labels["pybfs"] = (ratio >= bf_ratio_threshold).astype(int).fillna(0)

    vote_frac = labels.mean(axis=1)
    ensemble = (vote_frac >= voting_fraction).astype(int)

    events = pool_events(ensemble, t_c=t_c, d_min=d_min)

    voters = list(labels.columns)
    total_points = len(raw)

    # Downsample for chart transport/rendering on long records (matches the
    # app's existing >5000-point convention, e.g. api_gage_data); events and
    # stats below are still computed from the full-resolution series above.
    step = max(1, total_points // 5000) if total_points > 5000 else 1
    idx = raw.index[::step]
    dates = idx.strftime("%Y-%m-%d").tolist()
    q_ds = raw["Q"].loc[idx]
    labels_ds = labels.loc[idx]
    vote_frac_ds = vote_frac.loc[idx]
    ensemble_ds = ensemble.loc[idx]

    return {
        "dates": dates,
        "q": q_ds.round(3).tolist(),
        "voters": voters,
        "voter_labels": {v: VOTER_LABELS[v] for v in voters},
        "labels": {v: labels_ds[v].astype(int).tolist() for v in voters},
        "vote_fraction": vote_frac_ds.round(3).tolist(),
        "ensemble": ensemble_ds.astype(int).tolist(),
        "total_points": total_points,
        "events": [
            {
                "start_date": row.start_date.strftime("%Y-%m-%d"),
                "end_date": row.end_date.strftime("%Y-%m-%d"),
                "duration": int(row.duration),
                "n_bfd_days": int(row.n_bfd_days),
                "gap_days": int(row.gap_days),
            }
            for row in events.itertuples()
        ],
        "stats": {
            "n_days": len(raw),
            "bfd_fraction": round(float(ensemble.mean()), 4),
            "n_events": len(events),
            "mean_event_duration": round(float(events["duration"].mean()), 1) if len(events) else None,
        },
    }


def get_ensemble_result(
    site_no: str,
    bf_ratio_threshold: float = 0.95,
    voting_fraction: float = 0.5,
    t_c: int = 3,
    d_min: int = 7,
) -> dict:
    raw = get_or_compute_raw(site_no)
    return apply_params(raw, bf_ratio_threshold, voting_fraction, t_c, d_min)
