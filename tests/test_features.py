"""Leakage tests for M1 history features: only information known strictly
before a job's submission may be used."""

import numpy as np
import pandas as pd

from tevzi.features.m1 import history_stats, submission_stats


def _events():
    # group g: job A ends at t=100 (rt=1), job B ends at t=300 (rt=3)
    return pd.DataFrame({"key": ["g", "g", "h"], "t_avail": [100, 300, 50], "rt": [1.0, 3.0, 9.0]})


def _query(t, key="g"):
    return pd.DataFrame({"qid": ["q"], "key": [key], "t": [t]})


def test_job_not_finished_yet_is_not_counted():
    out = history_stats(_events(), _query(99), "key", ["rt"], "p")
    assert out.loc["q", "p_n"] == 0
    assert np.isnan(out.loc["q", "p_rt_mean"])


def test_event_at_same_second_is_not_counted():
    out = history_stats(_events(), _query(100), "key", ["rt"], "p")
    assert out.loc["q", "p_n"] == 0


def test_only_past_events_are_counted():
    out = history_stats(_events(), _query(200), "key", ["rt"], "p")
    assert out.loc["q", "p_n"] == 1
    assert out.loc["q", "p_rt_last"] == 1.0
    assert np.isnan(out.loc["q", "p_rt_std"])          # std needs n >= 2


def test_stats_after_both_events():
    out = history_stats(_events(), _query(301), "key", ["rt"], "p")
    assert out.loc["q", "p_n"] == 2
    assert out.loc["q", "p_rt_mean"] == 2.0
    assert out.loc["q", "p_rt_std"] == 1.0
    assert out.loc["q", "p_rt_last"] == 3.0
    assert out.loc["q", "p_rt_last2"] == 2.0


def test_other_keys_do_not_leak():
    out = history_stats(_events(), _query(1000, key="z"), "key", ["rt"], "p")
    assert out.loc["q", "p_n"] == 0


def test_order_of_completion_not_submission():
    # B finished before A although submitted later: 'last' must be the most recently finished
    ev = pd.DataFrame({"key": ["g", "g"], "t_avail": [500, 200], "rt": [5.0, 2.0]})
    out = history_stats(ev, _query(300), "key", ["rt"], "p")
    assert out.loc["q", "p_n"] == 1
    assert out.loc["q", "p_rt_last"] == 2.0


def test_submission_history_is_strictly_before():
    subs = pd.DataFrame({
        "qid": ["a", "b", "c", "d", "e"],
        "key": ["g", "g", "g", "g", "x"],
        "t":   [0, 0, 1800, 7200, 0],
    })
    out = submission_stats(subs, "key", "p")
    assert out.loc["a", "p_n_prev_submit"] == 0          # same-second submissions do not see each other
    assert out.loc["b", "p_n_prev_submit"] == 0
    assert np.isnan(out.loc["a", "p_time_since_prev"])
    assert out.loc["c", "p_n_prev_submit"] == 2
    assert out.loc["c", "p_time_since_prev"] == 1800
    assert out.loc["c", "p_submit_last_1h"] == 2          # a, b in [t-1h, t)
    assert out.loc["d", "p_n_prev_submit"] == 3
    assert out.loc["d", "p_submit_last_1h"] == 0          # nothing in [3600, 7200)
    assert out.loc["e", "p_n_prev_submit"] == 0