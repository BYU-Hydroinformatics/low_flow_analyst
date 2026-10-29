# RF-referenced threshold tuning

How the app's magnitude-adaptive threshold defaults were chosen, 2026-10-12.

## Goal

Pick `bf_ratio_threshold` (t_hi), `bf_ratio_threshold_low` (t_lo),
`low_flow_anchor` and `high_flow_anchor` so the digital-filter voters
(Eckhardt, Chapman, Lyne-Hollick, Boughton, PyBFS) label baseflow days the
way RF-BFD does. The voting fraction (0.5), t_c (3) and d_min (7) are held
fixed during tuning. The app defaults were then set to t_c 10 and d_min 21
days by choice, not by this search.

## Procedure

1. Cache raw per-method baseflow and RF-BFD labels for 134 gages (64 already
   cached plus 70 random ones with at least 10 years of record).
2. Split gages at random: 89 for tuning, 45 held out.
3. Draw 3000 random (t_hi, t_lo, low anchor, high anchor) sets, then
   hill-climb from the 5 best.
4. Score a set by Cohen's kappa between the majority vote of the filter
   voters (RF-BFD excluded) and the RF-BFD daily label, averaged over gages.
5. Report the held-out score.

## Result

| | t_hi | t_lo | low anchor | high anchor | held-out kappa |
|---|---|---|---|---|---|
| Previous defaults | 0.95 | 0.75 | 0.10 | 0.50 | 0.13 |
| New defaults | 1.00 | 0.02 | 0.33 | 0.43 | 0.29 |

With the voting fraction free, the best set reached 0.33 (t_hi 1.0,
t_lo 0.02, anchors 0.33/0.38, voting fraction 0.85). A separate threshold per
method did no better (0.32).

## Later change: t_hi 1.00 -> 0.96 (2026-10-28)

The app default for t_hi was relaxed to 0.96 by choice. On the 145 cached gages
(all gages, not a held-out split) this lowers the mean kappa of the filter vote
vs RF-BFD from 0.340 to 0.327, raises the share of days above the high anchor
labeled BFD from 7.4 % to 11.6 %, and the share of all days from 42.7 % to
45.3 %. Days below the low anchor are unaffected (98.5 %).

## Caveats

- RF-BFD is a reference, not ground truth. In the manual event review on
  10172800, 10310500 and 12332000, RF-BFD (d_min 7, t_c 3) also covers 25 of
  the 28 events that were rated Bad.
- Thresholds on the filter ratios reach kappa of about 0.3 at most. RF-BFD
  also uses flow relative to the gage mean, short-term slope and month.
- The tuned threshold is close to a step: below about the 33rd flow
  percentile almost any ratio counts, above the 43rd a ratio near 1.0 is
  needed.
- In the app's full ensemble (RF-BFD included) the new defaults give longer
  events: mean duration 60 -> 69 d (10172800), 35 -> 81 d (10310500),
  27 -> 45 d (12332000), 16 -> 46 d (06752280).
- Only the LFA app's defaults changed. `bfd_db` defaults and the national
  database build are untouched.
