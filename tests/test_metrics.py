import numpy as np
import pandas as pd
import pytest

from tevzi.evaluation.metrics import (
    HOUR, SubsetThresholds, kendall_within_windows, make_subsets,
    paired_comparison, runtime_report, sjf_regret, usage_metrics,
)


def _jobs(n=400, seed=0):
    rng = np.random.default_rng(seed)
    submit = np.sort(rng.uniform(0, 10 * HOUR, n))
    run_time = np.exp(rng.normal(6, 2, n))
    return submit, run_time


def test_sjf_regret_oracle_and_fifo_bounds():
    submit, rt = _jobs()
    perfect = sjf_regret(rt, rt, submit)
    assert perfect['regret'] == pytest.approx(1.0)
    assert perfect['gap_closed'] == pytest.approx(1.0)
    fifo_like = sjf_regret(rt, submit, submit)       # order by arrival = FIFO
    assert fifo_like['regret'] == pytest.approx(fifo_like['fifo'])
    assert fifo_like['gap_closed'] == pytest.approx(0.0)
    reverse = sjf_regret(rt, -rt, submit)            # longest first
    assert reverse['regret'] > reverse['fifo'] > 1.0
    assert reverse['gap_closed'] < 0


def test_sjf_regret_is_invariant_to_monotone_transform():
    submit, rt = _jobs(seed=1)
    noisy = rt * np.exp(np.random.default_rng(2).normal(0, 1, len(rt)))
    a = sjf_regret(rt, noisy, submit)
    b = sjf_regret(rt, np.log1p(noisy), submit)
    assert a['regret'] == pytest.approx(b['regret'])
    assert 1.0 < a['regret'] < a['fifo']


def test_sjf_regret_penalises_long_job_mistakes_more_than_mae():
    # Same absolute log error, placed on a short job vs on a long job.
    submit = np.zeros(3)
    rt = np.array([10.0, 100.0, 10_000.0])
    y = np.log1p(rt)
    short_wrong = y.copy(); short_wrong[0] += 6.0     # short job looks long
    long_wrong = y.copy(); long_wrong[2] -= 6.0       # long job looks short
    r_short = sjf_regret(rt, short_wrong, submit)['regret']
    r_long = sjf_regret(rt, long_wrong, submit)['regret']
    assert r_long > r_short


def test_kendall_only_compares_jobs_in_same_window():
    # Two windows; inside each the order is perfect, across windows reversed.
    submit = np.array([0, 10, 20, HOUR + 0, HOUR + 10, HOUR + 20], dtype=float)
    y = np.array([4, 5, 6, 1, 2, 3], dtype=float)
    p = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    assert kendall_within_windows(y, p, submit) == pytest.approx(1.0)


def test_usage_metrics_under_prediction():
    y = np.array([0.0, 0.0, 0.5, 1.0])
    p = np.array([0.1, 0.0, 0.25, 1.5])
    m = usage_metrics(y, p)
    assert m['under_rate'] == pytest.approx(0.25)
    assert m['mean_shortfall'] == pytest.approx(0.25 / 4)
    assert m['mae_zero'] == pytest.approx(0.05)
    assert m['zero_share'] == pytest.approx(0.5)


def test_paired_comparison_uses_paired_noise():
    # Seed effects are large but shared; B is consistently 3 % better.
    seed_effect = np.array([0.0, 0.05, -0.04, 0.03, -0.02])
    a = 1.0 + seed_effect
    b = a * 0.97
    res = paired_comparison(a, b, min_rel_gain=0.02)
    assert res['b_better'] and res['ci_low'] > 0
    worse = paired_comparison(a, a * 1.01)
    assert not worse['b_better']
    with pytest.raises(ValueError):
        paired_comparison(a[:2], b[:2])


def test_subsets_and_report_run():
    submit, rt = _jobs(n=200, seed=3)
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({
        'pool_key': rng.choice(['ALL', 'T4', 'V100', 'V100M32'], len(rt)),
        'pool_pressure': rng.exponential(0.1, len(rt)),
        'grp_rt_n': rng.integers(0, 5, len(rt)),
        'run_time': rt,
        'in_burst': rng.random(len(rt)) < 0.1,
        'usage_coverage': rng.choice([1.0, 0.5], len(rt)),
    })
    thr = SubsetThresholds.fit(frame['pool_pressure'])
    subs = make_subsets(frame, thr)
    assert subs['busy'].sum() + subs['calm'].sum() == len(frame)
    assert subs['hist_none'].sum() + subs['hist_some'].sum() == len(frame)
    y = np.log1p(rt)
    rep = runtime_report(y, y + rng.normal(0, 0.5, len(y)), submit, subs)
    assert rep.loc['all', 'n'] == len(frame)
    assert 1.0 <= rep.loc['all', 'sjf_regret'] < rep.loc['all', 'sjf_fifo']