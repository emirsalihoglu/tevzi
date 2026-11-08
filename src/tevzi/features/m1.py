"""M1 (static job) features, computed only from information known at submission.

Feature groups (each group is an ablation unit, see ``FEATURE_GROUPS``):
A requests, B job structure, C identity, D history, E time.

History (group D) uses an as-of join: for a job submitted at time t, only
events that became known strictly before t are used.
- Run-time history: Terminated jobs, known at their end time.
- Usage history: model-set jobs with usage, known at their end time.
- Submission history: all submitted jobs, known at their submission time.
Job status is never used (it is only known after the job ends).

Categorical columns (user, workload, GPU type, main task) are left as raw
strings; encoders must be fit on the training period only (Stage 3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tevzi.data.loader import load_table

HOUR = 3_600
DAY = 86_400

FEATURE_GROUPS = {
    "A_requests": [
        "main_plan_gpu", "main_plan_cpu", "main_plan_mem", "main_inst_num",
        "req_gpu_total", "req_cpu_total", "req_mem_total", "main_gpu_type",
        "gpu_fractional", "mem_per_gpu", "cpu_per_gpu",
    ],
    "B_structure": ["n_tasks", "has_ps", "has_evaluator", "n_ps_inst", "total_inst", "main_task"],
    "C_identity": ["user", "workload", "workload_missing"],
    "D_history": [
        "grp_rt_n", "grp_rt_mean", "grp_rt_std", "grp_rt_last", "grp_rt_last2",
        "usr_rt_n", "usr_rt_mean", "usr_rt_std", "usr_rt_last", "usr_rt_last2",
        "grp_use_n", "grp_gpu_mean", "grp_mem_mean", "grp_gpu_zero_share",
        "usr_use_n", "usr_gpu_mean", "usr_mem_mean", "usr_gpu_zero_share",
        "grp_n_prev_submit", "grp_seen_before", "grp_time_since_prev", "grp_submit_last_1h",
    ],
    "E_time": ["hour_sin", "hour_cos", "dow_sin", "dow_cos"],
}
CATEGORICAL = ["main_gpu_type", "main_task", "user", "workload"]


# ----------------------------------------------------------------------------
# Generic as-of helpers (pure functions, unit-tested for leakage)
# ----------------------------------------------------------------------------

def history_stats(events: pd.DataFrame, queries: pd.DataFrame, key: str,
                  values: list[str], prefix: str) -> pd.DataFrame:
    """Per-key statistics of events known strictly before each query time.

    events : columns ``key``, ``t_avail`` (time the event becomes known), ``values``
    queries: columns ``qid``, ``key``, ``t`` (query time)
    Returns a frame indexed by ``qid`` with ``{prefix}_n`` and, for each value v,
    ``{prefix}_{v}_mean``, ``_std`` (n >= 2), ``_last`` (most recent) and
    ``_last2`` (mean of the two most recent).
    """
    ev = events[[key, "t_avail", *values]].dropna(subset=[key, "t_avail"])
    ev = ev.sort_values("t_avail", kind="mergesort").reset_index(drop=True)
    g = ev.groupby(key, sort=False)
    cum = pd.DataFrame({key: ev[key], "t_avail": ev["t_avail"], "_n": g.cumcount() + 1})
    for v in values:
        cum[f"_{v}_s"] = g[v].cumsum()
        cum[f"_{v}_s2"] = (ev[v] ** 2).groupby(ev[key], sort=False).cumsum()
        cum[f"_{v}_last"] = ev[v]
        cum[f"_{v}_prev"] = g[v].shift(1)

    q = queries[["qid", key, "t"]].sort_values("t", kind="mergesort")
    m = pd.merge_asof(q, cum, left_on="t", right_on="t_avail", by=key,
                      allow_exact_matches=False, direction="backward")

    n = m["_n"].fillna(0)
    out = pd.DataFrame({"qid": m["qid"], f"{prefix}_n": n})
    for v in values:
        mean = m[f"_{v}_s"] / n.where(n > 0)
        var = (m[f"_{v}_s2"] / n.where(n > 0) - mean ** 2).clip(lower=0)
        out[f"{prefix}_{v}_mean"] = mean
        out[f"{prefix}_{v}_std"] = np.sqrt(var).where(n >= 2)
        out[f"{prefix}_{v}_last"] = m[f"_{v}_last"]
        out[f"{prefix}_{v}_last2"] = (
            (m[f"_{v}_last"] + m[f"_{v}_prev"]) / 2
        ).fillna(m[f"_{v}_last"])
    return out.set_index("qid")


def submission_stats(subs: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """Per-key submission history strictly before each submission.

    subs: columns ``qid``, ``key``, ``t`` (submission time), one row per job.
    Returns, indexed by ``qid``: ``{prefix}_n_prev_submit``,
    ``{prefix}_time_since_prev`` and ``{prefix}_submit_last_1h``
    (submissions in [t - 1h, t)).
    """
    s = subs[["qid", key, "t"]].dropna(subset=[key, "t"])
    u = (s.groupby([key, "t"], sort=False).size().rename("k").reset_index()
           .sort_values([key, "t"], kind="mergesort").reset_index(drop=True))
    u["cum"] = u.groupby(key, sort=False)["k"].cumsum()          # submissions with time <= t
    u["n_before"] = u["cum"] - u["k"]                            # submissions with time <  t
    u["prev_t"] = u.groupby(key, sort=False)["t"].shift(1)       # previous distinct time

    s = s.merge(u[[key, "t", "n_before", "prev_t"]], on=[key, "t"], how="left")

    q = s[["qid", key]].assign(t0=s["t"] - HOUR).sort_values("t0", kind="mergesort")
    right = u[[key, "t", "cum"]].rename(columns={"t": "t_right"}).sort_values("t_right", kind="mergesort")
    m = pd.merge_asof(q, right, left_on="t0", right_on="t_right", by=key,
                      allow_exact_matches=False, direction="backward")
    before_window = m.set_index("qid")["cum"].fillna(0)

    s = s.set_index("qid")
    return pd.DataFrame({
        f"{prefix}_n_prev_submit": s["n_before"],
        f"{prefix}_time_since_prev": s["t"] - s["prev_t"],
        f"{prefix}_submit_last_1h": s["n_before"] - before_window.reindex(s.index),
    })


# ----------------------------------------------------------------------------
# Feature groups
# ----------------------------------------------------------------------------

def _structure(data_dir: Path, job_names: pd.Series) -> pd.DataFrame:
    task = load_table("task", data_dir, columns=["job_name", "task_name", "inst_num"])
    task = task[task["job_name"].isin(job_names)]
    is_ps = task["task_name"].eq("ps")
    t = pd.DataFrame({
        "job_name": task["job_name"],
        "is_ps": is_ps,
        "is_eval": task["task_name"].eq("evaluator"),
        "ps_inst": task["inst_num"].where(is_ps, 0),
        "inst": task["inst_num"],
    })
    return t.groupby("job_name", sort=False).agg(
        has_ps=("is_ps", "any"), has_evaluator=("is_eval", "any"),
        n_ps_inst=("ps_inst", "sum"), total_inst=("inst", "sum"),
    )


def _cyclic(x: pd.Series, period: float, name: str) -> pd.DataFrame:
    angle = 2 * np.pi * (x % period) / period
    return pd.DataFrame({f"{name}_sin": np.sin(angle), f"{name}_cos": np.cos(angle)})


def build_features(jobs: pd.DataFrame, targets: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """M1 features for every job in ``targets`` (the model set).

    ``jobs`` comes from cleaning.clean (all jobs), ``targets`` from
    targets.build_targets (model set, main-task columns included).
    """
    data_dir = Path(data_dir)
    tags = load_table("group_tag", data_dir, columns=["job_name", "group", "workload"])
    base = jobs[["job_name", "user", "submit_time", "end_time", "first_start",
                 "n_tasks", "req_gpu_total", "req_cpu_total", "req_mem_total",
                 "m2_arrival", "m2_occupancy"]].merge(tags, on="job_name", how="left")
    base["group"] = base["group"].fillna(base["job_name"])      # unknown group = no shared history

    f = targets[["job_name", "main_task", "main_inst_num", "main_plan_cpu",
                 "main_plan_mem", "main_plan_gpu", "main_gpu_type"]].merge(
        base, on="job_name", how="left")

    # A. Requests
    f["gpu_fractional"] = (f["main_plan_gpu"] % 100).ne(0)
    gpus = f["main_plan_gpu"] / 100
    f["mem_per_gpu"] = f["main_plan_mem"] / gpus
    f["cpu_per_gpu"] = f["main_plan_cpu"] / gpus

    # B. Structure
    f = f.merge(_structure(data_dir, f["job_name"]), left_on="job_name", right_index=True, how="left")

    # C. Identity
    f["workload_missing"] = f["workload"].isna()
    f["workload"] = f["workload"].fillna("missing")

    # D. History
    q = f[["job_name", "group", "user", "submit_time"]].rename(columns={"job_name": "qid", "submit_time": "t"})

    done = base[base["m2_occupancy"]]
    rt_events = pd.DataFrame({
        "group": done["group"], "user": done["user"], "t_avail": done["end_time"],
        "rt": np.log1p(done["end_time"] - done["first_start"]),
    })
    use = targets.loc[targets["has_usage"], ["job_name", "gpu_ratio", "mem_ratio"]].merge(
        base[["job_name", "group", "user", "end_time"]], on="job_name", how="left")
    use_events = pd.DataFrame({
        "group": use["group"], "user": use["user"], "t_avail": use["end_time"],
        "gpu": use["gpu_ratio"], "mem": use["mem_ratio"],
        "gpuzero": use["gpu_ratio"].eq(0).astype(float),
    })

    hist = []
    for key, p in (("group", "grp"), ("user", "usr")):
        h = history_stats(rt_events, q, key, ["rt"], f"{p}_rt")
        hist.append(h[[f"{p}_rt_n", f"{p}_rt_rt_mean", f"{p}_rt_rt_std",
                       f"{p}_rt_rt_last", f"{p}_rt_rt_last2"]].set_axis(
            [f"{p}_rt_n", f"{p}_rt_mean", f"{p}_rt_std", f"{p}_rt_last", f"{p}_rt_last2"], axis=1))
        u = history_stats(use_events, q, key, ["gpu", "mem", "gpuzero"], f"{p}_use")
        hist.append(u[[f"{p}_use_n", f"{p}_use_gpu_mean", f"{p}_use_mem_mean",
                       f"{p}_use_gpuzero_mean"]].set_axis(
            [f"{p}_use_n", f"{p}_gpu_mean", f"{p}_mem_mean", f"{p}_gpu_zero_share"], axis=1))

    subs = base.loc[base["m2_arrival"], ["job_name", "group", "submit_time"]].rename(
        columns={"job_name": "qid", "submit_time": "t"})
    sub = submission_stats(subs, "group", "grp")
    hist.append(sub)
    hist = pd.concat(hist, axis=1)
    f = f.merge(hist, left_on="job_name", right_index=True, how="left")
    f["grp_seen_before"] = f["grp_n_prev_submit"].gt(0)

    # E. Time (relative clock of the trace)
    f = pd.concat([f.reset_index(drop=True),
                   _cyclic(f["submit_time"], DAY, "hour").reset_index(drop=True),
                   _cyclic(f["submit_time"], 7 * DAY, "dow").reset_index(drop=True)], axis=1)

    cols = [c for group in FEATURE_GROUPS.values() for c in group]
    return f[["job_name", *cols]]