"""M2 (cluster-state time series) on a 5-minute grid, plus the exact state at submission.

Time semantics (leakage rule): the state "at time t" only uses events that
happened strictly before t.
- A job counts as running at t if it started before t and has not ended before t.
- A job counts as queued at t if it was submitted before t and has not started before t.
- Arrivals at t are the submissions in [t - STEP, t).
A job's window uses the last ``n_steps`` grid points strictly before its
submission, followed by the exact state at its submission time.

Job sets (decision D): arrivals use all jobs with a submission time
(``m2_arrival``); running and queued state use Terminated jobs with a valid
interval (``m2_occupancy``), CPU-only jobs included in CPU/memory channels.

Pools: GPU pools by machine type. Occupancy uses the pool a job actually ran
in; queue and arrivals use the GPU type a job requests. A job's own pool is
its requested type; MISC (any type) maps to "ALL" (all GPU pools together).
All channels are ratios to capacity, except the arrival count (log1p).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tevzi.data.loader import load_table
from tevzi.data.split import T0

STEP = 300
GPU_POOLS = ("MISC", "P100", "T4", "V100", "V100M32")      # machine types with GPUs
POOL_KEYS = ("ALL", "P100", "T4", "V100", "V100M32")       # a job's own pool
CLUSTER_CHANNELS = ["arr_count", "arr_gpu", "occ_gpu", "occ_cpu", "occ_mem", "pressure", "use_gpu"]
POOL_CHANNELS = ["pool_occ", "pool_pressure", "pool_arr_gpu", "pool_use_gpu"]
CHANNELS = CLUSTER_CHANNELS + POOL_CHANNELS
ONE_GPU = 100.0                                              # plan_gpu units


# ----------------------------------------------------------------------------
# Step functions evaluated at arbitrary times (pure, unit-tested)
# ----------------------------------------------------------------------------

class Level:
    """Sum of weights of intervals active at t: start < t and not (end < t)."""

    def __init__(self, starts, ends, weights):
        s, e = np.asarray(starts, float), np.asarray(ends, float)
        w = np.asarray(weights, float)
        o = np.argsort(s, kind="mergesort")
        self._s, self._cs = s[o], np.concatenate([[0.0], np.cumsum(w[o])])
        o = np.argsort(e, kind="mergesort")
        self._e, self._ce = e[o], np.concatenate([[0.0], np.cumsum(w[o])])

    def __call__(self, t):
        t = np.asarray(t, float)
        return (self._cs[np.searchsorted(self._s, t, side="left")]
                - self._ce[np.searchsorted(self._e, t, side="left")])


class Window:
    """Sum of weights of events with time in [t - step, t)."""

    def __init__(self, times, weights, step: float = STEP):
        x = np.asarray(times, float)
        w = np.asarray(weights, float)
        o = np.argsort(x, kind="mergesort")
        self._x, self._c, self._step = x[o], np.concatenate([[0.0], np.cumsum(w[o])]), step

    def __call__(self, t):
        t = np.asarray(t, float)
        return (self._c[np.searchsorted(self._x, t, side="left")]
                - self._c[np.searchsorted(self._x, t - self._step, side="left")])


# ----------------------------------------------------------------------------
# Core: channels at arbitrary times from a standardised job frame
# ----------------------------------------------------------------------------

def make_state(frame: pd.DataFrame, caps: dict) -> dict:
    """Build the step functions once.

    frame columns: submit, start, end, gpu, cpu, mem, use_gpu (float),
    req_type (requested GPU type or 'None'), pool (machine type ran in),
    arrival (bool), occupancy (bool).
    caps: gpu, cpu, mem (cluster) and one GPU capacity per entry of POOL_KEYS.
    """
    a = frame[frame["arrival"]]
    o = frame[frame["occupancy"]]
    on_gpu_pool = o["pool"].isin(GPU_POOLS)
    st = {
        "caps": caps,
        "arr_count": Window(a["submit"], np.ones(len(a))),
        "arr_gpu": Window(a["submit"], a["gpu"]),
        "occ_gpu": Level(o["start"], o["end"], o["gpu"]),
        "occ_cpu": Level(o["start"], o["end"], o["cpu"]),
        "occ_mem": Level(o["start"], o["end"], o["mem"]),
        "queue_gpu": Level(o["submit"], o["start"], o["gpu"]),
        "use_gpu": Level(o["start"], o["end"], o["use_gpu"]),
        "pool": {},
    }
    for key in POOL_KEYS:
        ran = on_gpu_pool if key == "ALL" else o["pool"].eq(key)
        req_o = o["req_type"].ne("None") if key == "ALL" else o["req_type"].eq(key)
        req_a = a["req_type"].ne("None") if key == "ALL" else a["req_type"].eq(key)
        st["pool"][key] = {
            "occ": Level(o.loc[ran, "start"], o.loc[ran, "end"], o.loc[ran, "gpu"]),
            "use": Level(o.loc[ran, "start"], o.loc[ran, "end"], o.loc[ran, "use_gpu"]),
            "queue": Level(o.loc[req_o, "submit"], o.loc[req_o, "start"], o.loc[req_o, "gpu"]),
            "arr": Window(a.loc[req_a, "submit"], a.loc[req_a, "gpu"]),
        }
    return st


def cluster_channels(st: dict, t) -> pd.DataFrame:
    c = st["caps"]
    occ = st["occ_gpu"](t)
    free = np.maximum(c["gpu"] - occ, ONE_GPU)
    return pd.DataFrame({
        "arr_count": np.log1p(st["arr_count"](t)),
        "arr_gpu": st["arr_gpu"](t) / c["gpu"],
        "occ_gpu": occ / c["gpu"],
        "occ_cpu": st["occ_cpu"](t) / c["cpu"],
        "occ_mem": st["occ_mem"](t) / c["mem"],
        "pressure": st["queue_gpu"](t) / free,
        "use_gpu": st["use_gpu"](t) / c["gpu"],
    })


def pool_channels(st: dict, key: str, t) -> pd.DataFrame:
    cap = st["caps"][key]
    p = st["pool"][key]
    occ = p["occ"](t)
    free = np.maximum(cap - occ, ONE_GPU)
    return pd.DataFrame({
        "pool_occ": occ / cap,
        "pool_pressure": p["queue"](t) / free,
        "pool_arr_gpu": p["arr"](t) / cap,
        "pool_use_gpu": p["use"](t) / cap,
    })


def grid_times(t_end: float) -> np.ndarray:
    return T0 + STEP * np.arange(max(int((t_end - T0) // STEP), 0) + 2)


def build_grid(st: dict, t_end: float) -> pd.DataFrame:
    """All channels at every grid point; pool channels as '<key>:<channel>'."""
    t = grid_times(t_end)
    parts = [cluster_channels(st, t)]
    for key in POOL_KEYS:
        parts.append(pool_channels(st, key, t).add_prefix(f"{key}:"))
    g = pd.concat(parts, axis=1)
    g.insert(0, "t", t)
    return g


def own_pool(req_type: pd.Series) -> pd.Series:
    return req_type.where(req_type.isin(POOL_KEYS[1:]), "ALL")


def build_snapshots(st: dict, queries: pd.DataFrame) -> pd.DataFrame:
    """Exact state at each query's submission time, with the query's own pool.

    queries: job_name, submit, req_type. Adds pool_key and k_last (last grid
    index strictly before submission).
    """
    q = queries[["job_name", "submit"]].copy().reset_index(drop=True)
    q["pool_key"] = own_pool(queries["req_type"].reset_index(drop=True))
    q["k_last"] = (np.ceil((q["submit"] - T0) / STEP) - 1).astype("int64")
    snap = pd.concat([q, cluster_channels(st, q["submit"])], axis=1)
    pool = pd.DataFrame(index=q.index, columns=POOL_CHANNELS, dtype=float)
    for key in POOL_KEYS:
        m = q["pool_key"].eq(key)
        if m.any():
            pool.loc[m, :] = pool_channels(st, key, q.loc[m, "submit"]).to_numpy()
    return pd.concat([snap, pool], axis=1)


def windows(grid: pd.DataFrame, snap: pd.DataFrame, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Model input (jobs, n_steps + 1, channels): the last n_steps grid points
    strictly before submission (oldest first) and the exact state last.
    Returns the array and a (jobs, n_steps) mask of valid (non-padded) steps.
    """
    K = len(grid)
    cl = grid[CLUSTER_CHANNELS].to_numpy(np.float32)
    k = snap["k_last"].to_numpy()
    idx = k[:, None] - np.arange(n_steps - 1, -1, -1)[None, :]
    valid = (idx >= 0) & (idx < K)
    idx = np.clip(idx, 0, K - 1)
    nc = len(CLUSTER_CHANNELS)
    X = np.zeros((len(snap), n_steps + 1, len(CHANNELS)), np.float32)
    X[:, :n_steps, :nc] = cl[idx]
    keys = snap["pool_key"].to_numpy()
    for key in POOL_KEYS:
        m = keys == key
        if m.any():
            pg = grid[[f"{key}:{c}" for c in POOL_CHANNELS]].to_numpy(np.float32)
            X[m, :n_steps, nc:] = pg[idx[m]]
    X[:, :n_steps][~valid] = 0.0
    X[:, n_steps, :] = snap[CHANNELS].to_numpy(np.float32)
    return X, valid


def fit_scaler(X: np.ndarray, valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std over all valid steps (fit on the training period only)."""
    rows = X.reshape(-1, X.shape[-1])
    if valid is not None:
        full = np.concatenate([valid, np.ones((len(valid), 1), bool)], axis=1).reshape(-1)
        rows = rows[full]
    mean, std = rows.mean(axis=0), rows.std(axis=0)
    return mean, np.where(std > 0, std, 1.0)


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


# ----------------------------------------------------------------------------
# Trace adapter
# ----------------------------------------------------------------------------

def _requested_type(data_dir: Path, job_names: pd.Series) -> pd.Series:
    task = load_table("task", data_dir, columns=["job_name", "inst_num", "plan_gpu", "gpu_type"])
    task = task[task["job_name"].isin(job_names) & task["plan_gpu"].gt(0)]
    task = task[task["gpu_type"].astype(str).ne("None")]
    task = task.assign(tot=task["plan_gpu"] * task["inst_num"]).sort_values(
        ["job_name", "tot"], ascending=[True, False], kind="mergesort")
    return task.drop_duplicates("job_name").set_index("job_name")["gpu_type"].astype(str)


def _actual_pool(data_dir: Path, job_names: pd.Series) -> pd.Series:
    """GPU pool a job ran in: the most frequent GPU machine type among its
    instances (CPU machines, e.g. for parameter servers, are ignored)."""
    spec = load_table("machine_spec", data_dir, columns=["machine", "gpu_type"])
    inst = load_table("instance", data_dir, columns=["job_name", "machine"])
    inst = inst[inst["job_name"].isin(job_names)].dropna(subset=["machine"])
    inst = inst.merge(spec, on="machine", how="inner")
    inst = inst[inst["gpu_type"].isin(GPU_POOLS)]
    counts = inst.groupby(["job_name", "gpu_type"], sort=False).size().rename("n").reset_index()
    counts = counts.sort_values(["job_name", "n", "gpu_type"], ascending=[True, False, True], kind="mergesort")
    return counts.drop_duplicates("job_name").set_index("job_name")["gpu_type"].astype(str)


def capacities(data_dir: Path) -> dict:
    spec = load_table("machine_spec", data_dir, columns=["gpu_type", "cap_cpu", "cap_mem", "cap_gpu"])
    gpu_by_type = spec.groupby("gpu_type")["cap_gpu"].sum() * ONE_GPU
    caps = {"gpu": float(spec["cap_gpu"].sum() * ONE_GPU),
            "cpu": float(spec["cap_cpu"].sum() * 100.0),
            "mem": float(spec["cap_mem"].sum()),
            "ALL": float(gpu_by_type.reindex(GPU_POOLS).fillna(0).sum())}
    for key in POOL_KEYS[1:]:
        caps[key] = float(gpu_by_type.get(key, 0.0))
    return caps


def job_frame(jobs: pd.DataFrame, targets: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Standardised frame for make_state from cleaning/targets outputs.

    use_gpu = main-task peak GPU utilisation x main-task instances (jobs with
    usage only; 0 otherwise, also in the simulator for background jobs).
    """
    data_dir = Path(data_dir)
    j = jobs[jobs["m2_arrival"] | jobs["m2_occupancy"]]
    use = targets.loc[targets["has_usage"], ["job_name", "gpu_util_max", "main_inst_num"]]
    use = use.assign(use_gpu=use["gpu_util_max"] * use["main_inst_num"]).set_index("job_name")["use_gpu"]
    f = pd.DataFrame({
        "job_name": j["job_name"].to_numpy(),
        "submit": j["submit_time"].to_numpy(float),
        "start": j["first_start"].to_numpy(float),
        "end": j["end_time"].to_numpy(float),
        "gpu": j["req_gpu_total"].fillna(0).to_numpy(float),
        "cpu": j["req_cpu_total"].fillna(0).to_numpy(float),
        "mem": j["req_mem_total"].fillna(0).to_numpy(float),
        "arrival": j["m2_arrival"].to_numpy(bool),
        "occupancy": j["m2_occupancy"].to_numpy(bool),
    })
    f["use_gpu"] = f["job_name"].map(use).fillna(0.0).to_numpy(float)
    f["req_type"] = f["job_name"].map(_requested_type(data_dir, f["job_name"])).fillna("None")
    f["pool"] = f["job_name"].map(_actual_pool(data_dir, f.loc[f["occupancy"], "job_name"])).fillna("unknown")
    return f


def build_m2(jobs: pd.DataFrame, targets: pd.DataFrame, data_dir: Path):
    """Grid (all channels at 5-min points) and per-job snapshots for the model set."""
    f = job_frame(jobs, targets, data_dir)
    caps = capacities(data_dir)
    st = make_state(f, caps)
    grid = build_grid(st, float(np.nanmax(f["end"])))
    q = f.loc[f["job_name"].isin(targets["job_name"]), ["job_name", "submit", "req_type"]]
    snap = build_snapshots(st, q)
    return grid, snap, caps