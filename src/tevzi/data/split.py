"""Temporal split by submission time, purging and the burst flag.

Days are counted from the first submission in the trace (T0). Periods:
warm-up [0, 7), train [7, 45), validation [45, 52), test [52, end).

Purging: a train or validation job is purged if it ends at or after the end
of its period, so no label in a period depends on time after that period.
The test period runs to the end of the trace and is not purged.

The burst (days 58.8-60.5, one user, mostly Failed jobs) is flagged so that
test results can be reported with and without it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

T0 = 494_319          # first submission in the trace (s)
DAY = 86_400

PERIODS = {           # name: (start_day, end_day)
    "warmup": (0.0, 7.0),
    "train": (7.0, 45.0),
    "val": (45.0, 52.0),
    "test": (52.0, np.inf),
}
PURGED_PERIODS = ("train", "val")
BURST_DAYS = (58.8, 60.5)


def to_day(t: pd.Series) -> pd.Series:
    """Seconds since trace start -> days since the first submission."""
    return (t - T0) / DAY


def assign_periods(jobs: pd.DataFrame) -> pd.DataFrame:
    """Period, purge flag and burst flag per job.

    ``jobs`` needs ``job_name``, ``submit_time`` and ``end_time``.
    Returns ``job_name, submit_day, period, purged, in_burst``.
    """
    out = jobs[["job_name"]].copy()
    out["submit_day"] = to_day(jobs["submit_time"])

    edges = [PERIODS[p][0] for p in PERIODS] + [np.inf]
    out["period"] = pd.cut(out["submit_day"], bins=edges, right=False,
                           labels=list(PERIODS)).astype(str)
    out.loc[out["submit_day"] < 0, "period"] = "warmup"   # safety: before T0

    period_end = out["period"].map({p: e for p, (_, e) in PERIODS.items()})
    end_day = to_day(jobs["end_time"])
    out["purged"] = out["period"].isin(PURGED_PERIODS) & ~(end_day < period_end)

    lo, hi = BURST_DAYS
    out["in_burst"] = out["submit_day"].between(lo, hi, inclusive="left")
    return out


def split_summary(periods: pd.DataFrame, run_time: pd.Series) -> pd.DataFrame:
    """Jobs kept and purged per period, with run-time quantiles of each group.

    ``run_time`` must be aligned with ``periods`` (same index).
    """
    df = periods.assign(run_time=run_time.to_numpy())
    rows = []
    for period in PERIODS:
        g = df[df["period"].eq(period)]
        for purged, h in g.groupby("purged"):
            rt = h["run_time"]
            rows.append({
                "period": period,
                "group": "purged" if purged else "kept",
                "jobs": len(h),
                "run_time_median": rt.median(),
                "run_time_p90": rt.quantile(0.9),
                "run_time_p99": rt.quantile(0.99),
                "in_burst": int(h["in_burst"].sum()),
            })
    return pd.DataFrame(rows)