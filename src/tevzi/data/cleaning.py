"""Job-level cleaning for the Alibaba GPU trace.

Nothing is deleted here. Every job gets one boolean flag per exclusion rule
(R1-R7), the first rule it fails (``first_rule``) and membership flags for the
analysis sets:

- ``m2_arrival``   : jobs with a known submission time (exogenous M2 channels)
- ``m2_occupancy`` : Terminated jobs with valid requests and run interval,
                     CPU-only jobs included (endogenous M2 channels, background)
- ``model``        : jobs that pass all rules (prediction models)

Units follow the trace: ``plan_cpu`` / ``cpu_usage`` in percent of a core,
``plan_gpu`` / ``gpu_wrk_util`` in percent of a GPU, memory in GB,
machine capacities in cores, GB and GPU count.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tevzi.data.loader import load_table

RULES = {
    "R1_not_terminated": "Job status is not Terminated",
    "R2_cpu_only": "No task requests a GPU",
    "R3_missing_request": "Missing CPU/memory request, or GPU type without plan_gpu",
    "R4_zero_cpu_request": "A task requests 0 CPU",
    "R5_request_over_capacity": "A request exceeds the largest machine of the requested type",
    "R6_time_inconsistent": "Run starts before submission, or run duration <= 0",
    "R7_usage_over_capacity": "Observed usage exceeds the capacity of its machine",
}
RULE_ORDER = list(RULES)

# Rules that make requests or the run interval unusable (used for m2_occupancy)
_OCCUPANCY_RULES = [
    "R1_not_terminated",
    "R3_missing_request",
    "R4_zero_cpu_request",
    "R5_request_over_capacity",
    "R6_time_inconsistent",
]

# Requested types that may run on any machine
_ANY_MACHINE_TYPES = ("MISC", "None")


def _type_capacities(spec: pd.DataFrame) -> pd.DataFrame:
    """Largest capacity per requested type; MISC and None may use any machine."""
    cap_cols = ["cap_cpu", "cap_mem", "cap_gpu"]
    caps = spec.groupby("gpu_type")[cap_cols].max()
    overall = spec[cap_cols].max()
    for t in _ANY_MACHINE_TYPES:
        caps.loc[t] = overall
    return caps


def _task_summary(data_dir: Path, caps: pd.DataFrame) -> pd.DataFrame:
    """Per-job request flags and total requests from the task table."""
    task = load_table(
        "task", data_dir,
        columns=["job_name", "inst_num", "plan_cpu", "plan_mem", "plan_gpu", "gpu_type"],
    )
    gpu_type = task["gpu_type"].astype(str).where(task["gpu_type"].notna(), "None")
    gpu_typed = gpu_type.ne("None")
    plan_gpu_missing = task["plan_gpu"].isna()

    task_caps = caps.reindex(gpu_type.to_numpy())
    task_caps.index = task.index
    over = (
        (task["plan_gpu"] > 100 * task_caps["cap_gpu"])
        | (task["plan_cpu"] > 100 * task_caps["cap_cpu"])
        | (task["plan_mem"] > task_caps["cap_mem"])
    )

    t = pd.DataFrame({
        "job_name": task["job_name"],
        "wants_gpu": gpu_typed & (plan_gpu_missing | task["plan_gpu"].gt(0)),
        "missing": task["plan_cpu"].isna() | task["plan_mem"].isna() | (gpu_typed & plan_gpu_missing),
        "cpu_zero": task["plan_cpu"].eq(0),
        "over": over,
        "gpu_tot": task["plan_gpu"].fillna(0) * task["inst_num"],
        "cpu_tot": task["plan_cpu"] * task["inst_num"],
        "mem_tot": task["plan_mem"] * task["inst_num"],
    })
    return t.groupby("job_name", sort=False).agg(
        n_tasks=("wants_gpu", "size"),
        any_gpu=("wants_gpu", "any"),
        any_missing=("missing", "any"),
        any_cpu_zero=("cpu_zero", "any"),
        any_over_cap=("over", "any"),
        req_gpu_total=("gpu_tot", "sum"),
        req_cpu_total=("cpu_tot", "sum"),
        req_mem_total=("mem_tot", "sum"),
    )


def _instance_summary(data_dir: Path) -> pd.DataFrame:
    """First instance start and last instance end per job."""
    inst = load_table("instance", data_dir, columns=["job_name", "start_time", "end_time"])
    return inst.groupby("job_name", sort=False).agg(
        first_start=("start_time", "min"),
        last_end=("end_time", "max"),
    )


def _usage_over_capacity(data_dir: Path, spec: pd.DataFrame) -> pd.Series:
    """True for jobs with any sensor record above the capacity of its machine."""
    s = load_table(
        "sensor", data_dir,
        columns=["job_name", "machine", "cpu_usage", "gpu_wrk_util", "max_mem"],
    )
    s = s.merge(spec[["machine", "cap_cpu", "cap_mem", "cap_gpu"]], on="machine", how="left")
    over = (
        (s["cpu_usage"] > 100 * s["cap_cpu"])
        | (s["gpu_wrk_util"] > 100 * s["cap_gpu"])
        | (s["max_mem"] > s["cap_mem"])
    )
    return over.groupby(s["job_name"], sort=False).any().rename("usage_over_cap")


def build_job_table(data_dir: Path) -> pd.DataFrame:
    """One row per job: status, times, request flags, totals and usage check."""
    data_dir = Path(data_dir)
    spec = load_table("machine_spec", data_dir,
                      columns=["machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu"])
    caps = _type_capacities(spec)

    jobs = load_table("job", data_dir,
                      columns=["job_name", "user", "status", "start_time", "end_time"])
    jobs = jobs.rename(columns={"start_time": "submit_time"})
    jobs = jobs.merge(_task_summary(data_dir, caps), left_on="job_name", right_index=True, how="left")
    jobs = jobs.merge(_instance_summary(data_dir), left_on="job_name", right_index=True, how="left")
    jobs = jobs.merge(_usage_over_capacity(data_dir, spec), left_on="job_name", right_index=True, how="left")
    return jobs


def apply_rules(jobs: pd.DataFrame) -> pd.DataFrame:
    """Add one flag per rule and the first rule each job fails ('kept' if none)."""
    jobs = jobs.copy()
    jobs["R1_not_terminated"] = jobs["status"].ne("Terminated")
    jobs["R2_cpu_only"] = jobs["any_gpu"].eq(False)          # jobs with tasks but no GPU request
    jobs["R3_missing_request"] = jobs["any_missing"].eq(True) | jobs["n_tasks"].isna()
    jobs["R4_zero_cpu_request"] = jobs["any_cpu_zero"].eq(True)
    jobs["R5_request_over_capacity"] = jobs["any_over_cap"].eq(True)
    jobs["R6_time_inconsistent"] = (
        jobs["first_start"].isna()
        | jobs["end_time"].isna()
        | (jobs["first_start"] < jobs["submit_time"])
        | (jobs["end_time"] <= jobs["first_start"])
    )
    jobs["R7_usage_over_capacity"] = jobs["usage_over_cap"].eq(True)

    hits = jobs[RULE_ORDER]
    jobs["first_rule"] = np.where(hits.any(axis=1), hits.idxmax(axis=1), "kept")
    return jobs


def assign_sets(jobs: pd.DataFrame) -> pd.DataFrame:
    """Add membership flags for the analysis sets (see module docstring)."""
    jobs = jobs.copy()
    jobs["m2_arrival"] = jobs["submit_time"].notna()
    jobs["m2_occupancy"] = ~jobs[_OCCUPANCY_RULES].any(axis=1)
    jobs["model"] = jobs["first_rule"].eq("kept")
    return jobs


def exclusion_flow(jobs: pd.DataFrame) -> pd.DataFrame:
    """Exclusion flow table: jobs removed at each rule (in order) and jobs remaining.

    ``flagged_any_order`` counts all jobs failing the rule, regardless of order.
    """
    remaining = len(jobs)
    rows = [{"step": "All jobs", "description": "", "removed": 0,
             "remaining": remaining, "flagged_any_order": 0}]
    for rule in RULE_ORDER:
        removed = int(jobs["first_rule"].eq(rule).sum())
        remaining -= removed
        rows.append({"step": rule, "description": RULES[rule], "removed": removed,
                     "remaining": remaining, "flagged_any_order": int(jobs[rule].sum())})
    return pd.DataFrame(rows)


def clean(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full cleaning: job table with flags and sets, plus the flow table."""
    jobs = assign_sets(apply_rules(build_job_table(data_dir)))
    return jobs, exclusion_flow(jobs)