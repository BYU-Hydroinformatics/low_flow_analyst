# BFD event review app

Show sampled baseflow events (zoomed + whole record), record human judgments to tune
thresholds / anchors / t_c / d_min.

    cd lfa && venv/bin/python review_app/server.py     # http://localhost:5055

Judgments: `review_app/judgments/<site_no>.jsonl` (append-only, latest per `item_id` wins;
each row stores the params in force). Per gage: 12 events (borderline, near-d_min, long,
gap-heavy, typical) + 4 "probe" near-miss stretches that are NOT events.
