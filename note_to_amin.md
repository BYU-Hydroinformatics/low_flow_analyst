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
