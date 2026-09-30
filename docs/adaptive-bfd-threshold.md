# Magnitude-Adaptive BFD Threshold — LFA panel

How the BFD ensemble panel exposes the magnitude-adaptive threshold. The
**algorithm itself** — the ramp, the anchors, the ratio clamp, edge cases,
and why a flat 0.95 under-labels low flow — lives in
[`us_baseflow_db/docs/adaptive-bfd-threshold.md`](https://github.com/BYU-Hydroinformatics/us_baseflow_db/blob/main/docs/adaptive-bfd-threshold.md)
(`bfd_db.labeling.ensemble`). This page covers only what's specific to this
app: the API route, caching, and the UI.

Implemented in [`bfd_ensemble.py`](../bfd_ensemble.py); exposed through
`/api/gage/<site_no>/bfd_ensemble` in [`app.py`](../app.py) and the BFD panel
in [`templates/index.html`](../templates/index.html).

---

## 1. Division of labor

- **`bfd_db.labeling.ensemble`** (installed as `bfd-db[labeling]`, see
  `requirements.txt`) owns the algorithm: `anchor_flows()`,
  `effective_threshold()`, `label_from_ratio()`, `ensemble_vote()`,
  `voter_agreement_stats()`, and `label_gage_all_methods()` for batch use (the
  national database build).
- **`bfd_ensemble.py`** in this repo is a thin adapter: it loads a gage's
  streamflow, caches the expensive per-method raw baseflow estimates
  (`bfd_ensemble_cache/<site_no>.parquet` — separation doesn't depend on the
  tunable threshold/voting/pooling params, so it's computed once and reused),
  and shapes `bfd_db`'s output into the JSON this app's chart consumes.
  Turning cached raw estimates into labels/ensemble/events on each request is
  cheap arithmetic, all of it delegated to `bfd_db`.

A change to the labeling algorithm (a new anchor default, a different clamp
rule, a new voter) belongs in `us_baseflow_db`, gets a test there, and shows
up here after bumping the `bfd-db` dependency (see §4). A change to caching,
downsampling, or the response shape belongs here.

---

## 2. API surface

### Request

`GET /api/gage/<site_no>/bfd_ensemble`

| Param | Default | Meaning |
|---|---|---|
| `bf_ratio_threshold` | `0.95` | `t_hi`, threshold at/above the high anchor |
| `bf_ratio_threshold_low` | `0.75` | `t_lo`, threshold at/below the low anchor |
| `low_flow_anchor` | `0.10` | low anchor, as a flow quantile |
| `high_flow_anchor` | `0.50` | high anchor, as a flow quantile |
| `voting_fraction` | `0.5` | fraction of voters needed for an ensemble BFD |
| `t_c` | `3` | event pooling gap tolerance, days |
| `d_min` | `7` | event pooling minimum duration, days |

### Response fields

| Field | Shape | Meaning |
|---|---|---|
| `threshold` | `float[]`, one per displayed date | `t_eff` on that day, downsampled with the rest of the series |
| `threshold_range` | `[min, max]` | realized span of `t_eff` over the full record |
| `anchor_flows` | `[q_lo, q_hi]` cfs, either may be `null` | what the anchor percentiles resolve to on this gage |
| `voter_stats` | `{voter: {bfd_fraction, agreement}}` | computed over the full record, not the downsampled series |
| `zero_flow` | `int[]`, one per displayed date | 1 on days with `Q <= 0`; every voter and the ensemble are forced to not-BFD there (`bfd_db` rule, see its doc §5.1) |
| `events[].voter_days` | `{voter: int}` | days inside the event each voter called BFD (full-resolution labels), for the hover card |
| `events[].mean_q` | `float` | mean flow (cfs) over the event |
| `stats.zero_flow_days` | `int` | zero-flow days in the full record |

A defaults asymmetry worth knowing: `bfd_ensemble.apply_params()` takes
`bf_ratio_threshold_low=None` and falls back to the flat `bf_ratio_threshold`
(old behavior), so direct callers of that function are unaffected by default.
`get_ensemble_result()` and the HTTP route default it to `0.75` — adaptive by
default. This mirrors `bfd_db`'s own `PipelineConfig` default (`None` = flat,
used by the frozen-snapshot database build).

---

## 3. UI surface

Four sliders in the BFD panel drive the threshold:

| Control | Range | Step | Display |
|---|---|---|---|
| X% Threshold — high flow | 0.50 – 1.00 | 0.01 | `95%` |
| X% Threshold — low flow | 0.50 – 1.00 | 0.01 | `75%` |
| Low-flow anchor | 0.00 – 0.90 | 0.01 | `P10` |
| High-flow anchor | 0.10 – 1.00 | 0.01 | `P50` |

Feedback surfaces:

- **Threshold Range card** — realized span (`75%–95%`) with the resolved anchor
  flows beneath it (`9.96 → 46.50 cfs`).
- **Hydrograph hover** — `threshold here: 89%` for the hovered day, alongside
  each voter's call.
- **Voter chips** — per-method BFD rate and agreement with the ensemble
  (`voter_stats`), both computed over the full record.
- **Event hover** — hovering a pooled event band shows its dates, day counts,
  and each voter's days called BFD; the same breakdown appears in the
  forecast-skill modal.
- **Focus period** — date inputs / zoom limit the hydrograph, the voter strip
  and the forecast-skill events to one window; the per-voter stats stay
  full-record.
- **Method checkboxes** — each separation-method chip toggles that method's
  baseflow curve on the hydrograph (RF-BFD and flatness have no curve).

Slider changes are debounced 250 ms and re-fetch; nothing is recomputed
server-side beyond the cheap arithmetic in `bfd_db.labeling.ensemble` and
`bfd_ensemble.apply_params()`.

---

## 4. Picking up a `bfd_db` change

`requirements.txt` pins `bfd-db[labeling]` to a git ref (currently `@main`).
To pick up an algorithm change made in `us_baseflow_db`:

```bash
# local dev against a sibling checkout, for iterating on both at once:
pip install -e "../us_baseflow_db[labeling]"

# or, to match what a deploy will install:
pip install --upgrade "bfd-db[labeling] @ git+https://github.com/BYU-Hydroinformatics/us_baseflow_db.git@main"
```

For a deploy, pin `requirements.txt` to the commit or tag being shipped
(`...us_baseflow_db.git@<sha-or-tag>`) rather than floating `@main`, so a push
to `us_baseflow_db` doesn't silently change this app's labels on its next
Render rebuild.

---

## 5. Reproducing the numbers in `us_baseflow_db`'s doc

```python
import bfd_ensemble as be

raw = be.get_or_compute_raw('03198350')

# flat (old behaviour) vs adaptive
flat     = be.apply_params(raw, 0.95, 0.5, 3, 7)
adaptive = be.apply_params(raw, 0.95, 0.5, 3, 7, bf_ratio_threshold_low=0.75)

# anchor sensitivity
wide = be.apply_params(raw, 0.95, 0.5, 3, 7, bf_ratio_threshold_low=0.75,
                       low_flow_anchor=0.25, high_flow_anchor=0.75)

print(flat['stats'], adaptive['stats'], wide['anchor_flows'])
```

The underlying `effective_threshold()` / `anchor_flows()` functions themselves
are re-exported from `bfd_db.labeling.ensemble` — see that package's docs for
the math and the plateau/results tables.
