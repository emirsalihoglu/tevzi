"""Main task selection and prediction targets for the model set.

Main task: among the GPU-requesting tasks of a job, the one with the largest
total GPU request (plan_gpu x inst_num). Ties are broken by higher
per-instance plan_gpu, then role (worker > chief > evaluator > ps; other task
names count as worker), then task name, so the choice is deterministic.

Targets (one row per job):
- ``run_time``     : end_time - first instance start (s); ``log_run_time`` = log1p
- ``gpu_ratio``    : max gpu_wrk_util / plan_gpu over all instances and attempts
                     of the main task
- ``mem_ratio``    : max max_mem / plan_mem, same aggregation
- ``wait_time``    : first instance start - submission (reference only, not a target)
- ``has_usage``    : the main task has at least one sensor record with both values
- ``usage_coverage``: share of main-task attempts that have a sensor record
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tevzi.data.loader import load_table

ROLE_RANK = {"worker": 0, "chief": 1, "evaluator": 2, "ps": 3}  # lower = preferred on ties


def select_main_task(data_dir: Path, job_names: pd.Series | None = None) -> pd.DataFrame:
    """One row per job with its main task and the main task's requests."""
    task = load_table(
        "task", data_dir,
        columns=["job_name", "task_name", "inst_num", "plan_cpu", "plan_mem", "plan_gpu", "gpu_type"],
    )
    task = task[task["plan_gpu"].gt(0)]
    if job_names is not None:
        task = task[task["job_name"].isin(job_names)]
    task = task.assign(
        gpu_total=task["plan_gpu"] * task["inst_num"],
        role_rank=task["task_name"].map(ROLE_RANK).fillna(0),
    )
    order = task.sort_values(
        ["job_name", "gpu_total", "plan_gpu", "role_rank", "task_name"],
        ascending=[True, False, False, True, True],
    )
    main = order.drop_duplicates("job_name")
    return main.rename(columns={
        "task_name": "main_task", "inst_num": "main_inst_num", "plan_cpu": "main_plan_cpu",
        "plan_mem": "main_plan_mem", "plan_gpu": "main_plan_gpu", "gpu_type": "main_gpu_type",
    })[["job_name", "main_task", "main_inst_num", "main_plan_cpu",
        "main_plan_mem", "main_plan_gpu", "main_gpu_type"]].reset_index(drop=True)


def main_task_usage(data_dir: Path, main: pd.DataFrame) -> pd.DataFrame:
    """Max GPU utilisation and peak memory of the main task, plus sensor coverage."""
    keys = main[["job_name", "main_task"]].rename(columns={"main_task": "task_name"})

    sensor = load_table(
        "sensor", data_dir, columns=["job_name", "task_name", "gpu_wrk_util", "max_mem"],
    ).merge(keys, on=["job_name", "task_name"], how="inner")
    usage = sensor.groupby("job_name", sort=False).agg(
        gpu_util_max=("gpu_wrk_util", "max"),
        max_mem_max=("max_mem", "max"),
        n_sensor=("job_name", "size"),
    )
    del sensor

    attempts = load_table(
        "instance", data_dir, columns=["job_name", "task_name"],
    ).merge(keys, on=["job_name", "task_name"], how="inner")
    n_attempts = attempts.groupby("job_name", sort=False).size().rename("n_attempts")
    del attempts

    return usage.join(n_attempts, how="outer")


def build_targets(jobs: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Targets for jobs in the model set (``jobs`` comes from cleaning.clean)."""
    data_dir = Path(data_dir)
    base = jobs.loc[jobs["model"], ["job_name", "submit_time", "first_start", "end_time"]]
    main = select_main_task(data_dir, base["job_name"])
    usage = main_task_usage(data_dir, main)

    t = base.merge(main, on="job_name", how="left")
    t = t.merge(usage, left_on="job_name", right_index=True, how="left")

    t["run_time"] = t["end_time"] - t["first_start"]
    t["log_run_time"] = np.log1p(t["run_time"])
    t["wait_time"] = t["first_start"] - t["submit_time"]
    t["gpu_ratio"] = t["gpu_util_max"] / t["main_plan_gpu"]
    t["mem_ratio"] = t["max_mem_max"] / t["main_plan_mem"]
    t["has_usage"] = t["gpu_util_max"].notna() & t["max_mem_max"].notna()
    t["usage_coverage"] = (t["n_sensor"].fillna(0) / t["n_attempts"]).clip(upper=1.0)
    return t.drop(columns=["submit_time", "first_start", "end_time"])