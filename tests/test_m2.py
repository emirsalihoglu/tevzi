"""Leakage tests for M2: the state at time t only uses events strictly before t,
and a job's window only uses grid points strictly before its submission."""

import numpy as np
import pandas as pd

from tevzi.data.split import T0
from tevzi.features.m2 import (CHANNELS, STEP, Level, Window, build_grid,
                               build_snapshots, make_state, windows)


def test_level_is_strict_at_start_and_end():
    lv = Level([10.0], [20.0], [1.0])
    assert lv(10.0) == 0          # starting at t is not known at t
    assert lv(15.0) == 1
    assert lv(20.0) == 1          # ending at t is not known at t
    assert lv(20.5) == 0


def test_window_counts_only_the_previous_step():
    w = Window([0.0, 299.0, 300.0], [1.0, 1.0, 1.0], step=300)
    assert w(300.0) == 2          # events in [0, 300)
    assert w(600.0) == 1          # events in [300, 600)
    assert w(0.0) == 0


def _frame():
    s = T0 + 10 * STEP
    return pd.DataFrame({
        "job_name": ["a", "b", "c"],
        "submit": [s, s + 100, s + 5000],     # c is submitted in the future
        "start": [s + 10, s + 100 + 60, s + 5100],
        "end": [s + 4000, s + 3000, s + 9000],
        "gpu": [100.0, 50.0, 800.0], "cpu": [600.0, 400.0, 100.0], "mem": [30.0, 10.0, 5.0],
        "use_gpu": [40.0, 0.0, 500.0],
        "req_type": ["V100", "MISC", "V100"], "pool": ["V100", "MISC", "V100"],
        "arrival": [True, True, True], "occupancy": [True, True, True],
    })


CAPS = {"gpu": 1000.0, "cpu": 10000.0, "mem": 100.0,
        "ALL": 1000.0, "P100": 200.0, "T4": 200.0, "V100": 400.0, "V100M32": 200.0}


def test_snapshot_does_not_see_itself_or_the_future():
    f = _frame()
    st = make_state(f, CAPS)
    snap = build_snapshots(st, f[["job_name", "submit", "req_type"]]).set_index("job_name")
    # job b: a has arrived (100 s earlier) and is running; b itself and c are not counted
    assert snap.loc["b", "arr_count"] == np.log1p(1)
    assert snap.loc["b", "arr_gpu"] == 100.0 / 1000.0
    assert snap.loc["b", "occ_gpu"] == 100.0 / 1000.0
    assert snap.loc["b", "use_gpu"] == 40.0 / 1000.0
    assert snap.loc["b", "pressure"] == 0.0           # b's own queue time has not started
    # job a: nothing before it
    assert snap.loc["a", "arr_count"] == 0 and snap.loc["a", "occ_gpu"] == 0
    # own pool: V100 for a, ALL for MISC request b
    assert snap.loc["a", "pool_key"] == "V100" and snap.loc["b", "pool_key"] == "ALL"


def test_queue_pressure_counts_waiting_jobs_only():
    f = _frame()
    st = make_state(f, CAPS)
    t = f.loc[1, "submit"] + 30                       # b submitted, not started yet
    q = pd.DataFrame({"job_name": ["x"], "submit": [t], "req_type": ["MISC"]})
    s = build_snapshots(st, q).iloc[0]
    assert s["pressure"] == 50.0 / (1000.0 - 100.0)   # b waiting, a running
    assert s["pool_pressure"] == 50.0 / (1000.0 - 100.0)


def test_window_uses_only_grid_points_before_submission():
    f = _frame()
    st = make_state(f, CAPS)
    grid = build_grid(st, float(f["end"].max()))
    snap = build_snapshots(st, f[["job_name", "submit", "req_type"]])
    X, valid = windows(grid, snap, n_steps=4)
    assert X.shape == (3, 5, len(CHANNELS))
    for i, k in enumerate(snap["k_last"]):
        assert grid["t"].iloc[k] < snap["submit"].iloc[i]
        assert grid["t"].iloc[k + 1] >= snap["submit"].iloc[i]
    # job c (future) never appears in a's or b's windows
    occ_col = CHANNELS.index("occ_gpu")
    assert X[0, :, occ_col].max() * 1000.0 <= 100.0 + 50.0 + 1e-6
    assert valid.all()


def test_window_pads_before_trace_start():
    f = _frame()
    st = make_state(f, CAPS)
    grid = build_grid(st, float(f["end"].max()))
    q = pd.DataFrame({"job_name": ["early"], "submit": [T0 + 2 * STEP + 1], "req_type": ["T4"]})
    snap = build_snapshots(st, q)
    X, valid = windows(grid, snap, n_steps=5)
    assert valid[0].tolist() == [False, False, True, True, True]
    assert (X[0, :2] == 0).all()