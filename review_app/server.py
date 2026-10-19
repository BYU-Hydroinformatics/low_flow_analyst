"""
Event review app: show BFD events (zoomed + whole hydrograph) one at a time and
record the user's judgment, so threshold / anchor / t_c / d_min can be tuned
against human opinion instead of guesswork.

Run:  venv/bin/python review_app/server.py     (http://localhost:5055)

Reuses bfd_ensemble.apply_params() so what is shown is exactly what the LFA app
and bfd_db produce for the same parameters. Judgments are appended to
review_app/judgments/<site_no>.jsonl together with the params in force.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import bfd_ensemble as be  # noqa: E402
from bfd_db.labeling.ensemble import (  # noqa: E402
    effective_threshold, label_from_ratio, flatness_vote, force_not_bfd,
    apply_flatness_override, zero_flow_mask,
)
from bfd_db.events.pooling import pool_events  # noqa: E402
import plotly  # noqa: E402

JUDGE_DIR = os.path.join(HERE, "judgments")
os.makedirs(JUDGE_DIR, exist_ok=True)
PLOTLY_DIR = os.path.join(os.path.dirname(plotly.__file__), "package_data")

app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")

DEFAULTS = dict(
    bf_ratio_threshold=be.DEFAULT_BF_RATIO_THRESHOLD, bf_ratio_threshold_low=be.DEFAULT_BF_RATIO_THRESHOLD_LOW,
    low_flow_anchor=be.DEFAULT_LOW_FLOW_ANCHOR, high_flow_anchor=be.DEFAULT_HIGH_FLOW_ANCHOR,
    voting_fraction=0.5, t_c=be.DEFAULT_T_C, d_min=be.DEFAULT_D_MIN,
    use_flatness_vote=False, flatness_window=15, flatness_cv_threshold=0.05,
)
N_EVENTS = 12
N_PROBES = 4


def parse_params(src) -> dict:
    p = dict(DEFAULTS)
    for k, d in DEFAULTS.items():
        if k in src and src[k] not in (None, ""):
            v = src[k]
            if isinstance(d, bool):
                p[k] = str(v).lower() in ("1", "true", "yes", "on")
            elif isinstance(d, int):
                p[k] = int(float(v))
            else:
                p[k] = float(v)
    return p


def compute(site_no: str, p: dict) -> dict:
    """Labels, ensemble, events and per-day series for a gage under params p."""
    raw = be.get_or_compute_raw(site_no)
    q = raw["Q"]
    t_eff = effective_threshold(q, p["bf_ratio_threshold"], p["bf_ratio_threshold_low"],
                                p["low_flow_anchor"], p["high_flow_anchor"])
    labels = pd.DataFrame(index=raw.index)
    labels["rf_bfd"] = raw["rf_bfd"]
    ratios = {}
    for m in be.RATIO_METHODS + ["pybfs"]:
        col = f"qb_{m}"
        if col in raw.columns:
            labels[m] = label_from_ratio(raw[col], q, t_eff)
            ratios[m] = (raw[col] / q.where(q > 0)).clip(upper=1.2)
    zf = zero_flow_mask(q)
    labels = labels.apply(lambda c: force_not_bfd(c, zf))
    vote = labels.mean(axis=1)
    ens = (vote >= p["voting_fraction"]).astype(int)
    if p["use_flatness_vote"]:
        fl = force_not_bfd(flatness_vote(q, p["flatness_window"], p["flatness_cv_threshold"]), zf)
        labels["flatness"] = fl
        ens = apply_flatness_override(ens, fl)
    ens = force_not_bfd(ens, zf)
    events = pool_events(ens, t_c=p["t_c"], d_min=p["d_min"])
    return dict(raw=raw, q=q, t_eff=t_eff, labels=labels, ratios=ratios, vote=vote,
                ens=ens, events=events)


def pick_items(site_no: str, p: dict, seed: int) -> list[dict]:
    """Stratified sample: clear, borderline, short, long, plus 'probe' windows
    that are NOT events but look close, to expose false negatives."""
    c = compute(site_no, p)
    ev, vote, ens = c["events"], c["vote"], c["ens"]
    rng = np.random.default_rng(seed)
    items = []
    if len(ev):
        ev = ev.copy()
        ev["mean_vote"] = [vote.loc[r.start_date:r.end_date].mean() for r in ev.itertuples()]
        ev["gap_frac"] = ev["gap_days"] / ev["duration"]
        ev["month"] = ev["start_date"].dt.month
        ev["idx"] = range(len(ev))
        chosen: list[int] = []

        def take(df, n):
            df = df[~df["idx"].isin(chosen)]
            if len(df):
                chosen.extend(rng.choice(df["idx"].values, size=min(n, len(df)), replace=False).tolist())

        n = len(ev)
        take(ev[ev["mean_vote"] <= ev["mean_vote"].quantile(0.25)], 3)        # borderline
        take(ev[ev["duration"] <= max(p["d_min"] * 1.5, ev["duration"].quantile(0.15))], 2)  # near d_min
        take(ev[ev["duration"] >= ev["duration"].quantile(0.85)], 2)          # long
        take(ev[ev["gap_frac"] >= ev["gap_frac"].quantile(0.75)], 2)          # lots of pooled gaps
        take(ev, N_EVENTS)                                                    # fill with typical
        chosen = chosen[:N_EVENTS]
        for i in sorted(chosen, key=lambda i: ev.iloc[i]["start_date"]):
            r = ev.iloc[i]
            items.append(dict(kind="event", start=r["start_date"].strftime("%Y-%m-%d"),
                              end=r["end_date"].strftime("%Y-%m-%d"),
                              duration=int(r["duration"]), n_bfd_days=int(r["n_bfd_days"]),
                              gap_days=int(r["gap_days"]), mean_vote=round(float(r["mean_vote"]), 2)))
    # probes: near-miss non-event stretches
    in_event = pd.Series(False, index=ens.index)
    for r in c["events"].itertuples():
        in_event.loc[r.start_date:r.end_date] = True
    near = ((vote >= max(0.0, p["voting_fraction"] - 0.2)) & ~in_event & (c["q"] > 0)).astype(int)
    runs = pool_events(near, t_c=2, d_min=5)
    if len(runs):
        runs = runs.copy()
        runs["score"] = [vote.loc[r.start_date:r.end_date].mean() * np.sqrt(r.duration) for r in runs.itertuples()]
        top = runs.sort_values("score", ascending=False).head(N_PROBES * 3)
        pick = top.sample(n=min(N_PROBES, len(top)), random_state=seed)
        for r in pick.sort_values("start_date").itertuples():
            items.append(dict(kind="probe", start=r.start_date.strftime("%Y-%m-%d"),
                              end=r.end_date.strftime("%Y-%m-%d"), duration=int(r.duration),
                              n_bfd_days=0, gap_days=0,
                              mean_vote=round(float(vote.loc[r.start_date:r.end_date].mean()), 2)))
    for it in items:
        it["id"] = f'{it["kind"]}:{it["start"]}:{it["end"]}'
    return items


def _r(a, nd=3):
    return [None if (v is None or not np.isfinite(v)) else round(float(v), nd) for v in a]


@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.route("/plotly.min.js")
def plotly_js():
    return send_from_directory(PLOTLY_DIR, "plotly.min.js")


@app.route("/api/gages")
def api_gages():
    cached = sorted(f[:-8] for f in os.listdir(be.CACHE_DIR) if f.endswith(".parquet"))
    done = {}
    for f in os.listdir(JUDGE_DIR):
        if f.endswith(".jsonl"):
            ids = {}
            with open(os.path.join(JUDGE_DIR, f)) as fh:
                for line in fh:
                    j = json.loads(line)
                    ids[j["item_id"]] = 1
            done[f[:-6]] = len(ids)
    return jsonify(cached=cached, judged=done)


@app.route("/api/gage/<site_no>/full")
def api_full(site_no):
    try:
        p = parse_params(request.args)
        c = compute(site_no, p)
    except be.GageNotFoundError as e:
        return jsonify(error=str(e)), 404
    q = c["q"]
    return jsonify(
        dates=q.index.strftime("%Y-%m-%d").tolist(), q=_r(q.values),
        ens=c["ens"].astype(int).tolist(),
        events=[[r.start_date.strftime("%Y-%m-%d"), r.end_date.strftime("%Y-%m-%d")]
                for r in c["events"].itertuples()],
        anchor_flows=[float(x) for x in be.anchor_flows(q, p["low_flow_anchor"], p["high_flow_anchor"])],
        n_events=len(c["events"]), params=p,
    )


@app.route("/api/gage/<site_no>/items")
def api_items(site_no):
    try:
        p = parse_params(request.args)
        seed = int(request.args.get("seed", 1))
        return jsonify(items=pick_items(site_no, p, seed), params=p)
    except be.GageNotFoundError as e:
        return jsonify(error=str(e)), 404


@app.route("/api/gage/<site_no>/window")
def api_window(site_no):
    p = parse_params(request.args)
    start, end = request.args["start"], request.args["end"]
    c = compute(site_no, p)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    dur = (e - s).days + 1
    pad = pd.Timedelta(days=int(max(20, 0.6 * dur)))
    lo, hi = max(c["q"].index.min(), s - pad), min(c["q"].index.max(), e + pad)
    sl = slice(lo, hi)
    raw = c["raw"].loc[sl]
    return jsonify(
        dates=raw.index.strftime("%Y-%m-%d").tolist(), q=_r(raw["Q"].values),
        baseflow={m: _r(raw[f"qb_{m}"].values) for m in be.RATIO_METHODS + ["pybfs"] if f"qb_{m}" in raw.columns},
        ratio={m: _r(v.loc[sl].values, 3) for m, v in c["ratios"].items()},
        threshold=_r(c["t_eff"].loc[sl].values, 3),
        labels={m: c["labels"][m].loc[sl].astype(int).tolist() for m in c["labels"].columns},
        vote=_r(c["vote"].loc[sl].values, 3), ens=c["ens"].loc[sl].astype(int).tolist(),
        labels_all=list(c["labels"].columns),
    )


@app.route("/api/judge", methods=["POST"])
def api_judge():
    j = request.get_json()
    j["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(JUDGE_DIR, f'{j["site_no"]}.jsonl'), "a") as fh:
        fh.write(json.dumps(j) + "\n")
    return jsonify(ok=True)


@app.route("/api/gage/<site_no>/judgments")
def api_judgments(site_no):
    path = os.path.join(JUDGE_DIR, f"{site_no}.jsonl")
    latest = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                j = json.loads(line)
                latest[j["item_id"]] = j
    return jsonify(latest)


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5055)), debug=False, threaded=True)
