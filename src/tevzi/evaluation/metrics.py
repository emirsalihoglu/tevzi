"""Evaluation protocol for Stage 3 (prediction models).

Written before any model results are seen. Contents:

* run-time metrics: log-scale MAE/RMSE/R2, Spearman, Kendall tau within
  submission windows, and the scheduling-aware proxy ``sjf_regret``;
* usage metrics: MAE, under-prediction rate and size, zero / positive split;
* evaluation subsets (pool, group history, long jobs, burst, busy, coverage);
* paired multi-seed comparison of two models.

Model selection rule (decision g, 12.11.2026): run-time models are selected
on ``sjf_regret`` (lower is better) on the validation period; usage models on
MAE and under-prediction rate. Final candidates are re-checked with simulated
JCT in Stages 4-5, which wins if the two disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

HOUR = 3600
DAY = 86_400


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _as_float(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _windows(submit, window: float) -> np.ndarray:
    """Integer id of the submission window each job falls in."""
    return np.floor(_as_float(submit) / window).astype(np.int64)


# ----------------------------------------------------------------------------
# Run-time metrics
# ----------------------------------------------------------------------------

def regression_metrics(y, p) -> dict:
    """MAE, RMSE, R2 and Spearman rho (inputs on the same scale, e.g. log1p)."""
    y, p = _as_float(y), _as_float(p)
    err = p - y
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        'mae': float(np.mean(np.abs(err))),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'r2': float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        'spearman': float(stats.spearmanr(y, p).statistic) if len(y) > 1 else np.nan,
    }


def kendall_within_windows(y, p, submit, window: float = HOUR) -> float:
    """Kendall tau-b computed among jobs submitted in the same window.

    Averaged over windows, weighted by the number of job pairs in each window.
    Only pairs that a scheduler could actually have to order are compared.
    """
    y, p, w = _as_float(y), _as_float(p), _windows(submit, window)
    total, weight = 0.0, 0.0
    for idx in pd.Series(np.arange(len(w))).groupby(w).indices.values():
        n = len(idx)
        if n < 2:
            continue
        tau = stats.kendalltau(y[idx], p[idx]).statistic
        if np.isnan(tau):  # all values tied inside the window
            continue
        pairs = n * (n - 1) / 2
        total += tau * pairs
        weight += pairs
    return float(total / weight) if weight > 0 else np.nan


def _sum_completion_times(durations: np.ndarray) -> float:
    """Sum of completion times when jobs run back to back in the given order."""
    return float(np.cumsum(durations).sum())


def sjf_regret(run_time_s, score, submit, window: float = HOUR) -> dict:
    """Scheduling-aware proxy for model selection (decision g).

    Jobs are grouped by submission window. Inside each window they are run
    back to back on one server in the order given by ``score`` (shortest
    first; ties broken by submission order). The total completion time is
    compared with the oracle order (true run times) and with FIFO.

    ``run_time_s`` must be true run times in seconds. ``score`` can be any
    monotone transform of the predicted run time (e.g. log1p predictions).

    Returns
    -------
    regret      : cost(model) / cost(oracle); 1.0 is perfect, lower is better.
    fifo        : cost(FIFO) / cost(oracle), for reference.
    gap_closed  : (cost(FIFO) - cost(model)) / (cost(FIFO) - cost(oracle)).
    """
    rt, sc, sub = _as_float(run_time_s), _as_float(score), _as_float(submit)
    if np.any(rt < 0):
        raise ValueError('run_time_s must be non-negative seconds')
    w = _windows(sub, window)
    cost = {'model': 0.0, 'oracle': 0.0, 'fifo': 0.0}
    for idx in pd.Series(np.arange(len(w))).groupby(w).indices.values():
        fifo_order = idx[np.argsort(sub[idx], kind='stable')]
        r = rt[fifo_order]
        cost['fifo'] += _sum_completion_times(r)
        cost['oracle'] += _sum_completion_times(np.sort(r))
        cost['model'] += _sum_completion_times(r[np.argsort(sc[fifo_order], kind='stable')])
    span = cost['fifo'] - cost['oracle']
    return {
        'regret': cost['model'] / cost['oracle'] if cost['oracle'] > 0 else np.nan,
        'fifo': cost['fifo'] / cost['oracle'] if cost['oracle'] > 0 else np.nan,
        'gap_closed': (cost['fifo'] - cost['model']) / span if span > 0 else np.nan,
    }


def runtime_report(y_log, p_log, submit, subsets: dict | None = None,
                   window: float = HOUR) -> pd.DataFrame:
    """All run-time metrics, overall and per subset. Inputs are log1p seconds."""
    y_log, p_log, submit = _as_float(y_log), _as_float(p_log), _as_float(submit)
    subsets = subsets or {'all': np.ones(len(y_log), dtype=bool)}
    rows = []
    for name, m in subsets.items():
        m = np.asarray(m, dtype=bool)
        if m.sum() < 2:
            rows.append({'subset': name, 'n': int(m.sum())})
            continue
        row = {'subset': name, 'n': int(m.sum())}
        row.update(regression_metrics(y_log[m], p_log[m]))
        row['kendall_hour'] = kendall_within_windows(y_log[m], p_log[m], submit[m], window)
        sjf = sjf_regret(np.expm1(y_log[m]), p_log[m], submit[m], window)
        row.update({f'sjf_{k}': v for k, v in sjf.items()})
        rows.append(row)
    return pd.DataFrame(rows).set_index('subset')


# ----------------------------------------------------------------------------
# Usage metrics
# ----------------------------------------------------------------------------

def usage_metrics(y, p) -> dict:
    """MAE, under-prediction rate and mean shortfall; MAE on zero / positive y.

    Under-prediction (p < y) is the costly direction: the scheduler reserves
    less than the job uses, which leads to slowdown or OOM in the simulator.
    """
    y, p = _as_float(y), _as_float(p)
    under = p < y
    zero = y == 0
    return {
        'mae': float(np.mean(np.abs(p - y))),
        'under_rate': float(under.mean()),
        'mean_shortfall': float(np.mean(np.where(under, y - p, 0.0))),
        'mae_zero': float(np.mean(np.abs(p[zero] - y[zero]))) if zero.any() else np.nan,
        'mae_pos': float(np.mean(np.abs(p[~zero] - y[~zero]))) if (~zero).any() else np.nan,
        'zero_share': float(zero.mean()),
    }


def usage_report(y, p, subsets: dict | None = None) -> pd.DataFrame:
    """Usage metrics overall and per subset; rows with missing y are skipped."""
    y, p = _as_float(y), _as_float(p)
    ok = np.isfinite(y)
    subsets = subsets or {'all': np.ones(len(y), dtype=bool)}
    rows = []
    for name, m in subsets.items():
        m = np.asarray(m, dtype=bool) & ok
        row = {'subset': name, 'n': int(m.sum())}
        if m.any():
            row.update(usage_metrics(y[m], p[m]))
        rows.append(row)
    return pd.DataFrame(rows).set_index('subset')


# ----------------------------------------------------------------------------
# Evaluation subsets
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class SubsetThresholds:
    """Thresholds fitted on the training period only."""
    busy_pressure: float  # pool_pressure at or above this = busy

    @classmethod
    def fit(cls, train_pool_pressure, q: float = 0.75) -> 'SubsetThresholds':
        return cls(busy_pressure=float(np.nanquantile(_as_float(train_pool_pressure), q)))


def make_subsets(frame: pd.DataFrame, thr: SubsetThresholds,
                 hist_col: str = 'grp_rt_n') -> dict:
    """Pre-registered evaluation subsets (boolean masks aligned with ``frame``).

    Expected columns: ``pool_key``, ``pool_pressure`` (M2 snapshot),
    ``hist_col`` (group history count), ``run_time`` (seconds), ``in_burst``,
    ``usage_coverage``.
    """
    f = frame
    n_hist = f[hist_col].fillna(0).to_numpy()
    rt = f['run_time'].to_numpy()
    busy = f['pool_pressure'].to_numpy() >= thr.busy_pressure
    burst = f['in_burst'].fillna(False).to_numpy().astype(bool)
    subsets = {
        'all': np.ones(len(f), dtype=bool),
        'no_burst': ~burst,
        'burst': burst,
        'hist_none': n_hist == 0,
        'hist_some': n_hist > 0,
        'busy': busy,
        'calm': ~busy,
        'rt_ge_1h': rt >= HOUR,
        'rt_ge_1d': rt >= DAY,
        'full_coverage': f['usage_coverage'].to_numpy() == 1,
    }
    for pool in sorted(f['pool_key'].dropna().unique()):
        subsets[f'pool_{pool}'] = f['pool_key'].to_numpy() == pool
    subsets['pool_V100_busy'] = (
        np.isin(f['pool_key'].to_numpy(), ['V100', 'V100M32']) & busy)
    return subsets


# ----------------------------------------------------------------------------
# Paired multi-seed comparison
# ----------------------------------------------------------------------------

def paired_comparison(metric_a, metric_b, lower_is_better: bool = True,
                      min_rel_gain: float = 0.0, alpha: float = 0.05) -> dict:
    """Compare model B with baseline A over seeds (same seed = same pair).

    Noise is defined as the spread of the *paired* difference B - A across
    seeds (seed-specific randomness common to both models cancels out).

    B is declared better if (i) the (1 - alpha) t-interval of the mean paired
    improvement excludes zero and (ii) the relative gain is at least
    ``min_rel_gain`` (e.g. 0.02 for 2 %). Requires at least 3 seeds;
    the protocol uses 5.
    """
    a, b = _as_float(metric_a), _as_float(metric_b)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 3:
        raise ValueError('need two equal-length 1-D arrays with at least 3 seeds')
    improvement = (a - b) if lower_is_better else (b - a)   # > 0 means B is better
    n = len(improvement)
    mean = float(improvement.mean())
    se = float(improvement.std(ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(1 - alpha / 2, df=n - 1) * se)
    rel_gain = mean / abs(float(a.mean())) if a.mean() != 0 else np.nan
    return {
        'n_seeds': n,
        'mean_a': float(a.mean()),
        'mean_b': float(b.mean()),
        'improvement': mean,
        'rel_gain': rel_gain,
        'ci_low': mean - half,
        'ci_high': mean + half,
        'b_better': bool(mean - half > 0 and rel_gain >= min_rel_gain),
    }