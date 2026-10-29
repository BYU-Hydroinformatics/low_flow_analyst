# Note to Amin — BFD threshold defaults retuned, and an event review app

2026-10-19

The BFD ensemble defaults changed. They now live as `DEFAULT_*` constants in
`bfd_ensemble.py`, used by both API routes, the BFD panel (`templates/index.html`)
and the review app. New values (old in parentheses): t_hi 1.00 (0.95), t_lo 0.02
(0.75), low anchor 0.33 (0.10), high anchor 0.43 (0.50), t_c 10 (3), d_min 21 (7).
This is the LFA app only; the `bfd_db` defaults and the national database build are
unchanged. Committed and pushed as 1959198f.

**How they were chosen.** 134 gages (64 cached plus 70 random with at least 10 years),
split 89 tuning / 45 held out. 3000 random draws plus a hill-climb over t_hi, t_lo and
both anchors, voting fraction fixed at 0.5. Score is Cohen's kappa between the vote of
the 5 digital-filter voters (RF-BFD excluded) and the RF-BFD daily labels. Held-out
kappa went from 0.13 to 0.29 (voting fraction free reached 0.33; per-method thresholds
0.32). t_c 10 and d_min 21 were set by choice, not tuned. Details are in
`docs/rf-referenced-threshold-tuning.md`.

**Why a reference.** Expert review of 3 gages (10172800, 10310500, 12332000; 48
judgments in `review_app/judgments/`, git-ignored) rated 28 of 36 events Bad, and no
parameter set reproduced those ratings.

**Open items.**
- RF-BFD is a calibration reference, not ground truth. It also covers 25 of the 28
  events rated Bad (with d_min 7, t_c 3).
- The new defaults make events longer in the full ensemble (10172800 mean duration
  60 to 69 d at the old pooling), and the kappa was measured with t_c 3 / d_min 7,
  not 10 / 21. The new defaults have not been re-reviewed in the review app.
- Next candidates: re-review with boundary marks, per-gage tuning as an upper bound,
  and adding flow-relative-to-mean and slope features in `us_baseflow_db`.

**Review app.** `venv/bin/python review_app/server.py`, then http://localhost:5055.
It serves 12 sampled events plus 4 near-miss probes per gage.

---

# Note to Amin — bfd_db is now a real dependency, not a sys.path hack

2026-09-17

The BFD ensemble panel's threshold-adaptation logic (magnitude-adaptive
threshold, ratio clamp, per-voter agreement stats — added 2026-09-09) was
written directly in this repo's `bfd_ensemble.py` instead of in
`us_baseflow_db`, which is where the rest of the ensemble-voting algorithm
lives and where the national database build reads it from. That's now fixed:
the algorithm moved to `bfd_db.labeling.ensemble` (with tests, and
`docs/adaptive-bfd-threshold.md` in that repo), and `bfd_ensemble.py` here is
back to being a thin adapter — gage loading, raw-estimate caching, response
shaping, nothing else.

This also resolves the open item from the note below ("BFD ensemble needs
`bfd_db` to be installable"): `us_baseflow_db` had a broken
`build-backend` in its `pyproject.toml` (not a real entry point — nothing had
ever actually tried to `pip install` it), fixed to `setuptools.build_meta`.
`requirements.txt` now installs it as a normal git dependency:

```
bfd-db[labeling] @ git+https://github.com/BYU-Hydroinformatics/us_baseflow_db.git@main
```

The `BFD_DB_SRC` env var and the `sys.path.insert` it drove are gone —
`bfd_ensemble.py` just does `from bfd_db.labeling import ensemble` like any
other dependency. `[labeling]` is a new extra on `bfd-db` that pulls in only
what the algorithm needs (baseflowx, pybfs, scikit-learn, joblib), not the
database-build-only stack (geopandas, tqdm, click) or plotting (matplotlib) —
worth keeping in mind given the free-tier memory ceiling below.

**Before this next deploys**, pin that git ref to a commit or tag instead of
floating `@main` — see `docs/adaptive-bfd-threshold.md` §4 — so a future push
to `us_baseflow_db` can't silently change this app's live labels. Also added:
`pyarrow` was missing from `requirements.txt` even though `bfd_ensemble.py`'s
caching has always needed it for `to_parquet`/`read_parquet` — it happened to
be present locally as someone else's transitive dependency, which would not
have been true on Render.

---

# Note to Amin — LFA is now deployed, and two of your paths moved

2026-09-03

The Low Flow Analyst runs as a public web service at
<https://low-flow-analyst.onrender.com>, deployed from `main` on Render the same way
Baseflow Explorer is. Every push to `main` rebuilds it.

Getting it to boot required changing two things you own. Your local setup should behave
exactly as before — both changes keep your paths as the defaults — but you should know
they happened.

## The two paths

`app.py` and `bfd_ensemble.py` each hardcoded a directory under
`/Users/amin/Downloads/research/projects/`. Both now read an environment variable and
fall back to that same path:

| File | Variable | Default (unchanged) |
|---|---|---|
| `app.py` | `BFD_MODEL_DIR` | `.../bfd_ciroh/ml_model` |
| `bfd_ensemble.py` | `BFD_DB_SRC` | `.../us_baseflow_db/src` |

Set neither and you get the old behavior.

## The ensemble import is now optional

`app.py` imported `bfd_ensemble` unconditionally, and `bfd_ensemble` imports `bfd_db`
from that hardcoded `sys.path` entry. Anywhere but your laptop that raises
`ModuleNotFoundError` before Flask ever starts, so gunicorn's worker could not boot and
the whole app was down — not just the ensemble panel. It is now wrapped in
`try/except ImportError`, and `/api/gage/<site_no>/bfd_ensemble` returns a 503 naming the
missing package when it is absent. Everything else serves normally.

## What we need from you

Two features are dark on the deployment, and only you can turn them on:

1. **BFD ensemble** needs `bfd_db` to be installable — a published package, a git
   dependency, or vendored into this repo. A `sys.path` entry pointing at a local
   checkout cannot travel.
2. **BFD-ML skill and metrics** need `random_forest_bfd_model.joblib` and
   `feature_scaler.joblib` somewhere the container can read, with `BFD_MODEL_DIR` set to
   that directory. If the files are small enough to commit, that is the simplest route;
   otherwise they can go on a Render disk.

## One dependency pin

`requirements.txt` now caps `pandas<3.0`. The container was resolving pandas 3.0.5 against
the 2.x we develop on, and 3.0 is a breaking release. Raise the cap when we have tested
against it.

## Notes on the deployment itself

It runs on Render's free instance, which means 0.1 CPU, 512 MB, and a spin-down after 15
idle minutes. Memory is fine. Speed is the cost: `/api/gages` takes about 10 seconds to
serialize 9,539 records, and the first chart after a wake-up takes about 75 seconds while
numba compiles. Both return correct results. Moving to a paid instance under
**Settings → Compute** would cut both.

Anything written at runtime — `temp_results/`, fetched streamflow, parameters from
**Recalibrate** — lives on ephemeral disk and disappears on the next deploy.
