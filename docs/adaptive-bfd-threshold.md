# Magnitude-Adaptive BFD Threshold

How the BFD ensemble decides whether a separation method calls a given day
baseflow-dominant, and why that decision is no longer a single fixed number.

Implemented in [`bfd_ensemble.py`](../bfd_ensemble.py); exposed through
`/api/gage/<site_no>/bfd_ensemble` in [`app.py`](../app.py) and the BFD panel in
[`templates/index.html`](../templates/index.html).

---

## 1. Summary

Each ratio-based voter used to label a day BFD when

```
qb_method / Q  >=  0.95
```

with `0.95` fixed for every day of every gage. That test is now **ramped by flow
magnitude**: strict at high flow, relaxed at low flow, log-interpolated between
two anchors taken from the gage's own flow duration curve.

Nothing about the cached separation changes. The raw per-method `qb_*` series in
`bfd_ensemble_cache/*.parquet` are untouched — the threshold is applied at
request time in `apply_params()`, so every parameter below is live-tunable
without recomputing anything.

---

## 2. The problem it solves

The fixed `0.95` test was not merely strict — for most voters it was
**unreachable**. Every digital filter plateaus at its own ceiling of `qb/Q` well
below 1.0, because `BFI_max`-style parameters structurally cap the baseflow
fraction the method can report.

Median `qb/Q` by flow band, gage 03198350:

| Flow band | Eckhardt | Chapman | Lyne-Hollick | Boughton | PyBFS |
|---|---|---|---|---|---|
| Lowest 5% | 0.867 | 0.752 | 0.903 | 0.760 | 0.590 |
| P5–P20 | 0.875 | 0.755 | 0.855 | 0.763 | 0.629 |
| P20–P50 | 0.857 | 0.730 | 0.734 | 0.738 | 0.636 |
| Highest 10% | 0.383 | 0.271 | 0.348 | 0.285 | 0.327 |

Two things to read out of this:

1. **The filters respond correctly in direction.** The ratio climbs steeply as
   flow falls — Eckhardt goes 0.383 at storm flow to 0.867 at the lowest flows.
   The signal is there.
2. **They plateau, and the plateau is method-specific and below 0.95.** Eckhardt
   tops out near 0.87, Chapman near 0.75, PyBFS near 0.60, and they stay flat
   across the entire lower half of the record. No amount of "more baseflow-like"
   pushes them to 0.95.

So on a flat, low, obviously baseflow-dominant recession, most voters still
failed the test. In the lowest 5% of flows only 31% of days passed for Eckhardt
and 7% for PyBFS. The threshold was above the ceiling of the instrument.

---

## 3. The algorithm

### 3.1 The ramp

For each day with flow `Q`:

```
s(Q)      = clip( (ln Q − ln q_lo) / (ln q_hi − ln q_lo), 0, 1 )
t_eff(Q)  = t_lo + (t_hi − t_lo) · s(Q)
```

where

- `t_hi` = `bf_ratio_threshold` — the threshold at and above the high anchor
- `t_lo` = `bf_ratio_threshold_low` — the threshold at and below the low anchor
- `q_lo`, `q_hi` = the two anchor flows in cfs (§3.2)

`s` is clipped, so the curve is flat outside the anchors: every day at or above
`q_hi` gets exactly `t_hi`, every day at or below `q_lo` gets exactly `t_lo`.

**Interpolation is on log flow, not linear flow.** Streamflow is log-distributed
— on these gages the median spans 0.61 to 6,710 cfs across sites and three orders
of magnitude within a single site. A linear ramp would spend nearly all its range
on storm flows and collapse the entire low-flow region into the first pixel.

### 3.2 The anchors

Both anchors are **quantiles of the gage's own non-zero flow**, not absolute cfs:

```python
q_pos = q.replace(0, np.nan)
q_lo  = q_pos.quantile(low_flow_anchor)    # default 0.10
q_hi  = q_pos.quantile(high_flow_anchor)   # default 0.50
```

Anchoring on the gage's own flow duration curve is what lets a single setting
travel across a network of gages with wildly different magnitudes. An absolute
cutoff ("relax below 50 cfs") is meaningless on a river whose median is 6,710 cfs
and on one whose median is 0.61 cfs.

> **Naming convention — read this carefully.** These are percentiles of the flow
> *distribution* (non-exceedance), so `low_flow_anchor = 0.10` is the **10th
> percentile of flow**, which in the usual hydrologic convention is **Q90** — the
> flow exceeded 90% of the time. The UI labels them `P10` / `P50` and the
> *Threshold Range* card prints the cfs each one resolves to, precisely so this
> convention never has to be guessed at.

Zeros are excluded before taking quantiles, so an intermittent record's
zero-flow days do not drag the low anchor to 0 and flatten the ramp.

### 3.3 Worked example

Gage 03198350, defaults `t_hi = 0.95`, `t_lo = 0.75`, `P10/P50`
→ `q_lo = 9.96 cfs`, `q_hi = 46.50 cfs`:

| Q (cfs) | s | t_eff | note |
|---|---|---|---|
| 5.00 | 0.000 | 0.7500 | below low anchor — clipped |
| 9.96 | 0.000 | 0.7500 | at low anchor |
| 30.10 | 0.718 | 0.8935 | mid-ramp |
| 46.50 | 1.000 | 0.9500 | at high anchor |
| 500.00 | 1.000 | 0.9500 | above high anchor — clipped |

The mid-ramp value is what the chart hover reports as `threshold here: 89%`.

---

## 4. The ratio clamp

The ratio fed to the test is clamped at 1.0:

```python
ratio = (raw[col] / raw["Q"].replace(0, np.nan)).clip(upper=1.0)
```

Separated baseflow exceeding total flow is unphysical but does occur, PyBFS
especially. Frequency of `qb_pybfs > Q`:

| Gage | Days affected | Median overshoot | Max |
|---|---|---|---|
| 03198350 | 4.5% | 1.11× | 3.1× |
| 13077000 | 30.1% | 1.39× | 81.1× |
| 10263000 | 24.5% | 1.85× | 46.7× |
| 10263500 | 8.9% | 1.14× | 3.3× |

The rule adopted is **any day whose separated baseflow meets or exceeds total
flow counts as fully baseflow-dominant**.

**This clamp does not change any label.** A ratio above 1.0 already passed every
threshold ≤ 1.0, so the vote is identical with and without it — verified: PyBFS
holds at 6.5% BFD on 03198350 either way. What it buys is that an 81× value is
no longer carried around as a number, so the behaviour is explicit rather than
incidental, and stays correct if a threshold of exactly 1.0 is ever used.

It is **not** a fix for the underlying problem. See §9.

---

## 5. Edge cases

| Condition | Behaviour |
|---|---|
| `t_lo > t_hi` | `t_lo` clamped down to `t_hi` — the ramp flattens rather than inverting |
| `low_flow_anchor` / `high_flow_anchor` outside [0, 1] | clipped to [0, 1] |
| `high_flow_anchor <= low_flow_anchor` | `q_hi <= q_lo`, no usable ramp → flat `t_hi` for every day |
| Near-constant or all-zero record | quantiles collapse, `q_hi > q_lo > 0` fails → flat `t_hi` |
| Non-finite anchor flows | flat `t_hi` |
| `Q = 0` on a day | `t_eff` filled with `t_hi`; ratio is NaN, comparison is False, day labels **not BFD** |
| `qb` is NaN (filter warm-up, non-converged PyBFS) | comparison False → **not BFD** |

Every fallback lands on the *strict* flat threshold, so a misconfiguration
under-labels rather than silently over-labels. The fallback is visible in the
response: `threshold_range` comes back as `[t_hi, t_hi]` and the UI card shows
`95%–95%`.

---

## 6. API surface

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

### Response fields added

| Field | Shape | Meaning |
|---|---|---|
| `threshold` | `float[]`, one per displayed date | `t_eff` on that day, downsampled with the rest of the series |
| `threshold_range` | `[min, max]` | realized span of `t_eff` over the full record |
| `anchor_flows` | `[q_lo, q_hi]` cfs, either may be `null` | what the anchor percentiles resolve to on this gage |

### A defaults asymmetry worth knowing

`apply_params()` takes `bf_ratio_threshold_low=None` and falls back to
`bf_ratio_threshold`, i.e. **flat, old behaviour**, so existing direct callers of
that function are unaffected. `get_ensemble_result()` and the HTTP route default
it to `0.75`, i.e. **adaptive**. If you call `apply_params()` directly and want
the ramp, you must pass `bf_ratio_threshold_low` explicitly.

---

## 7. UI surface

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
- **Voter chips** — per-method BFD rate and agreement with the ensemble, both
  computed over the full record, not the downsampled display series.

Slider changes are debounced 250 ms and re-fetch; nothing is recomputed
server-side beyond the cheap arithmetic in `apply_params()`.

---

## 8. Results

Defaults `0.95 / 0.75` at `P10/P50`, versus the old flat `0.95`:

| Gage | Anchor flows (cfs) | BFD flat → adaptive | Events flat → adaptive |
|---|---|---|---|
| 03198350 | 9.96 → 46.5 | 17.7% → **29.3%** | 132 → 179 |
| 13077000 | 434 → 6,710 | 17.8% → **26.5%** | 335 → 333 |
| 13181000 | 106 → 221 | 21.0% → **41.4%** | 284 → 421 |

The check that matters is the storm end, which must **not** move. On 03198350 the
highest-decile BFD rate is 0.3% before and 0.3% after; on 13077000, 2.0% and
2.0%. The relaxation buys low-flow sensitivity without letting storm days leak
in.

---

## 9. Choosing anchors

The defaults are not universal. What matters is the **spread** between the two
anchor flows, and that varies a lot by gage:

| Gage | P10 → P50 | Spread |
|---|---|---|
| 03198350 | 9.96 → 46.5 cfs | 4.7× |
| 13181000 | 106 → 221 cfs | 2.1× |

A compressed spread means the ramp does almost nothing: on 13181000, a flat
recession sitting at ~150–200 cfs is already near the *top* of a 106→221 ramp and
gets a near-strict threshold, which is exactly the failure the ramp was built to
fix. Widening the anchors resolves it:

| Anchors | Anchor flows (cfs) | BFD | Events |
|---|---|---|---|
| P10/P50 (default) | 106 → 221 | 41.4% | 421 |
| P25/P75 | 149 → 700 | 61.1% | 406 |
| P40/P90 | 187 → 2,320 | 66.4% | 408 |
| P2/P20 | 71 → 136 | 27.5% | 324 |

**Rule of thumb:** check the *Threshold Range* card's cfs values first. If the two
anchor flows are within a factor of ~3, the ramp is too tight to do useful work —
widen the high anchor. Regulated rivers and snowmelt-dominated gages tend to have
compressed low-flow duration curves and want wider anchors than the default.

---

## 10. Known limitations

**PyBFS is an unconditional vote at low flow on some gages.** The clamp makes
`qb > Q` days count as BFD, but on 13077000 that is 30.1% of the record, sitting
at median flow-duration position 0.206 — the lower fifth of the flow range,
exactly where discrimination matters. On those days PyBFS is not voting on
evidence. The worst cases are sudden collapses in observed flow (52–82 cfs on a
river with a 6,710 cfs median, clustered in October and mid-January) that a
smooth recession model cannot follow — plausibly diversion shutoff or
ice-affected gauging. This is a separation/calibration problem, not a threshold
problem, and remains open. Note the overshoot is *not* a unit error: ratios run
1.0–81× with a median of 1.39×, whereas a cfs↔m³/s slip would be a constant
35.31×.

**No flatness or recession criterion.** The ramp keys on flow *magnitude* only.
A small storm peak occurring during a low-flow period gets the relaxed threshold
purely because the flow is low. Adding a slope gate — relax only when
`|d ln Q/dt|` is small and the flow is not rising — was considered and
deliberately deferred; it is the natural next refinement if false BFD days show
up on minor low-flow events.

**RF-BFD is unaffected by any of this.** It is a classifier that emits 0/1 labels
directly and never sees a ratio or a threshold. Its column is passed through
untouched, which is why its BFD fraction is identical before and after the
change, and why moving the sliders never moves its strip row.

**Anchors are per-request, not per-gage-persisted.** There is no stored
per-gage anchor calibration; the sliders reset to `P10/P50` each time the panel
is opened.

---

## 11. Reproducing the numbers in this document

```python
import numpy as np, bfd_ensemble as be

raw = be.get_or_compute_raw('03198350')

# flat (old behaviour) vs adaptive
flat     = be.apply_params(raw, 0.95, 0.5, 3, 7)
adaptive = be.apply_params(raw, 0.95, 0.5, 3, 7, bf_ratio_threshold_low=0.75)

# anchor sensitivity
wide = be.apply_params(raw, 0.95, 0.5, 3, 7, bf_ratio_threshold_low=0.75,
                       low_flow_anchor=0.25, high_flow_anchor=0.75)

print(flat['stats'], adaptive['stats'], wide['anchor_flows'])

# the threshold curve on its own
t = be.effective_threshold(raw['Q'], 0.95, 0.75, 0.10, 0.50)
print(be.anchor_flows(raw['Q'], 0.10, 0.50), t.min(), t.max())
```

The plateau table in §2 comes from binning `qb_method / Q` by
`Q.rank(pct=True)` and taking the median per band.
