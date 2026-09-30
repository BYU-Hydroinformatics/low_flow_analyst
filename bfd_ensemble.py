"""
BFD Ensemble panel support.

Wires the LFA app to the shared `bfd_db` labeling code in us_baseflow_db
(RF-BFD classifier, baseflowx digital filters, PyBFS, magnitude-adaptive
threshold, ensemble vote) so every gage can show which "voters" call each day
baseflow-dominant, with dmin/t_c/threshold/voting-fraction all adjustable at
request time.

The labeling *algorithm* lives entirely in `bfd_db.labeling.ensemble` and
`bfd_db.events.pooling` — see docs/adaptive-bfd-threshold.md in us_baseflow_db.
This module is a thin adapter: it loads gage streamflow, caches the expensive
per-method raw baseflow estimates to disk (separation itself does not depend
on the tunable threshold/voting/pooling parameters, so it is run once per gage
and reused), and shapes the result for the API/chart. Turning cached raw
estimates into 0/1 labels, an ensemble vote, and pooled events is cheap
arithmetic done fresh on every request.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from bfd_db.config import PipelineConfig
from bfd_db.labeling import baseflowx_methods, pybfs_labeling, rf_bfd
from bfd_db.labeling.ensemble import (
    FLATNESS_CV_THRESHOLD,
    FLATNESS_WINDOW,
    HIGH_FLOW_ANCHOR,
    LOW_FLOW_ANCHOR,
    anchor_flows,
    apply_flatness_override,
    effective_threshold,
    flatness_vote,
    force_not_bfd,
    label_from_ratio,
    voter_agreement_stats,
    zero_flow_mask,
)
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
    "flatness": "Flatness",
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

    raw = pd.DataFrame(index=q.index)
    raw["Q"] = q
    raw["rf_bfd"] = rf_label
    for method in RATIO_METHODS:
        raw[f"qb_{method}"] = baseflowx_methods.separate_baseflow(q, method)

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
    bf_ratio_threshold_low: float | None = None,
    low_flow_anchor: float = LOW_FLOW_ANCHOR,
    high_flow_anchor: float = HIGH_FLOW_ANCHOR,
    use_flatness_vote: bool = False,
    flatness_window: int = FLATNESS_WINDOW,
    flatness_cv_threshold: float = FLATNESS_CV_THRESHOLD,
) -> dict:
    """Derive labels/ensemble/events from cached raw voter data for the given
    params, shape it for the API/chart. All labeling math (threshold ramp,
    ratio clamp, voting) comes from bfd_db.labeling.ensemble.
    """
    t_eff = effective_threshold(
        raw["Q"], bf_ratio_threshold, bf_ratio_threshold_low,
        low_flow_anchor, high_flow_anchor,
    )
    q_lo_cfs, q_hi_cfs = anchor_flows(raw["Q"], low_flow_anchor, high_flow_anchor)

    labels = pd.DataFrame(index=raw.index)
    labels["rf_bfd"] = raw["rf_bfd"]
    for method in RATIO_METHODS:
        col = f"qb_{method}"
        if col in raw.columns:
            labels[method] = label_from_ratio(raw[col], raw["Q"], t_eff)
    if "qb_pybfs" in raw.columns:
        labels["pybfs"] = label_from_ratio(raw["qb_pybfs"], raw["Q"], t_eff)

    # No flow, no baseflow-dominance: same rule as label_gage_all_methods()
    # (see docs/adaptive-bfd-threshold.md §5.1 in us_baseflow_db).
    zero_flow = zero_flow_mask(raw["Q"])
    labels = labels.apply(lambda col: force_not_bfd(col, zero_flow))

    vote_frac = labels.mean(axis=1)
    ensemble = (vote_frac >= voting_fraction).astype(int)

    # Flatness override: rescues flat recessions the digital filters lag
    # behind on, without diluting the majority vote on every other day (an
    # equal-weight vote here would — see apply_flatness_override()'s
    # docstring). Added to `labels`/voter_stats for display only after the
    # override, so it never feeds back into vote_frac above.
    if use_flatness_vote:
        flatness = flatness_vote(raw["Q"], flatness_window, flatness_cv_threshold)
        flatness = force_not_bfd(flatness, zero_flow)
        labels["flatness"] = flatness
        ensemble = apply_flatness_override(ensemble, flatness)
    ensemble = force_not_bfd(ensemble, zero_flow)

    events = pool_events(ensemble, t_c=t_c, d_min=d_min)

    voters = list(labels.columns)
    total_points = len(raw)

    # Per-voter scorecard over the FULL record (not the downsampled display
    # series below).
    voter_stats = voter_agreement_stats(labels, ensemble)

    # Downsample for chart transport/rendering on long records (matches the
    # app's existing >5000-point convention, e.g. api_gage_data); events and
    # stats above are still computed from the full-resolution series.
    step = max(1, total_points // 5000) if total_points > 5000 else 1
    idx = raw.index[::step]
    dates = idx.strftime("%Y-%m-%d").tolist()
    q_ds = raw["Q"].loc[idx]
    labels_ds = labels.loc[idx]
    vote_frac_ds = vote_frac.loc[idx]
    ensemble_ds = ensemble.loc[idx]

    # Per-method baseflow curves for the hydrograph overlay. Only the ratio
    # voters have one: RF-BFD is a classifier and emits labels, not a series.
    # NaNs (filter warm-up, a PyBFS run that didn't converge) become null so
    # Plotly draws a gap instead of the JSON carrying a bare NaN literal.
    def _series(col):
        return [None if pd.isna(v) else round(float(v), 3) for v in raw[col].loc[idx]]

    baseflow = {
        col[3:]: _series(col) for col in raw.columns if col.startswith("qb_")
    }

    return {
        "dates": dates,
        "q": q_ds.round(3).tolist(),
        "baseflow": baseflow,
        "threshold": t_eff.loc[idx].round(4).tolist(),
        "threshold_range": [
            round(float(t_eff.min()), 4),
            round(float(t_eff.max()), 4),
        ],
        "anchor_flows": [
            None if not np.isfinite(q_lo_cfs) else round(q_lo_cfs, 3),
            None if not np.isfinite(q_hi_cfs) else round(q_hi_cfs, 3),
        ],
        "voters": voters,
        "voter_labels": {v: VOTER_LABELS[v] for v in voters},
        "labels": {v: labels_ds[v].astype(int).tolist() for v in voters},
        "vote_fraction": vote_frac_ds.round(3).tolist(),
        "zero_flow": zero_flow.loc[idx].astype(int).tolist(),
        "ensemble": ensemble_ds.astype(int).tolist(),
        "voter_stats": voter_stats,
        "total_points": total_points,
        "events": [
            {
                "start_date": row.start_date.strftime("%Y-%m-%d"),
                "end_date": row.end_date.strftime("%Y-%m-%d"),
                "duration": int(row.duration),
                "n_bfd_days": int(row.n_bfd_days),
                "gap_days": int(row.gap_days),
                # Days inside the event each voter called BFD (full-resolution
                # labels), for the hover card; the ensemble's own count is
                # n_bfd_days above.
                "voter_days": {
                    v: int(labels[v].loc[row.start_date:row.end_date].sum())
                    for v in voters
                },
                "mean_q": round(float(raw["Q"].loc[row.start_date:row.end_date].mean()), 3),
            }
            for row in events.itertuples()
        ],
        "stats": {
            "n_days": len(raw),
            "bfd_fraction": round(float(ensemble.mean()), 4),
            "n_events": len(events),
            "zero_flow_days": int(zero_flow.sum()),
            "mean_event_duration": round(float(events["duration"].mean()), 1) if len(events) else None,
        },
    }


def get_ensemble_result(
    site_no: str,
    bf_ratio_threshold: float = 0.95,
    voting_fraction: float = 0.5,
    t_c: int = 3,
    d_min: int = 7,
    bf_ratio_threshold_low: float = 0.75,
    low_flow_anchor: float = LOW_FLOW_ANCHOR,
    high_flow_anchor: float = HIGH_FLOW_ANCHOR,
    use_flatness_vote: bool = False,
    flatness_window: int = FLATNESS_WINDOW,
    flatness_cv_threshold: float = FLATNESS_CV_THRESHOLD,
) -> dict:
    raw = get_or_compute_raw(site_no)
    return apply_params(
        raw, bf_ratio_threshold, voting_fraction, t_c, d_min,
        bf_ratio_threshold_low=bf_ratio_threshold_low,
        low_flow_anchor=low_flow_anchor,
        high_flow_anchor=high_flow_anchor,
        use_flatness_vote=use_flatness_vote,
        flatness_window=flatness_window,
        flatness_cv_threshold=flatness_cv_threshold,
    )
