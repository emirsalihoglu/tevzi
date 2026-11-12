"""Gradient boosting models for Stage 3.3 (Approach 1, 1-nouser and 1b).

Design (12.11.2026, fixed before results):

* Features: M1 (numeric as-is, categoricals as ``CategoryMap`` codes); 1b adds
  the M2 submission snapshot (11 channels, prefix ``m2_``).
* Early stopping on an inner hold-out = the last training days
  (``submit_day >= INNER_START_DAY``), then refit on all training rows with
  the best number of trees. The validation period is never used for fitting.
* Small fixed grid, searched once per target on variant 1 with a single seed
  and without the refit step (run time: ``sjf_regret``, usage: MAE). The
  chosen setting is then used for every variant (1, 1-nouser, 1b), each
  trained with 5 seeds, so variants differ only in their features.
* Loss: L2 for all targets (target structure is decided in Stage 3.4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tevzi.features.m1 import CATEGORICAL
from tevzi.features.m2 import CHANNELS
from tevzi.models.preprocess import CategoryMap

INNER_START_DAY = 41.0
SEEDS = (0, 1, 2, 3, 4)
GRID = (
    {'num_leaves': 63, 'min_child_samples': 20},
    {'num_leaves': 63, 'min_child_samples': 100},
    {'num_leaves': 255, 'min_child_samples': 20},
    {'num_leaves': 255, 'min_child_samples': 100},
)
BASE_PARAMS = {
    'objective': 'l2',
    'learning_rate': 0.1,
    'n_estimators': 3000,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'cat_smooth': 10,
    'cat_l2': 10,
    'min_data_per_group': 50,
    'verbose': -1,
}
EARLY_STOPPING_ROUNDS = 100


def build_features(m1: pd.DataFrame, cmap: CategoryMap, snap: pd.DataFrame | None = None,
                   drop: tuple = ()) -> pd.DataFrame:
    """Feature matrix indexed by ``job_name`` in the row order of ``m1``.

    ``snap`` (optional) adds the M2 snapshot channels; it is aligned by
    ``job_name``, so its row order does not matter. ``drop`` removes columns
    (e.g. ``('user',)`` for variant 1-nouser).
    """
    num = m1.drop(columns=['job_name', *CATEGORICAL])
    cats = cmap.as_category(m1)
    X = pd.concat([num.reset_index(drop=True), cats.reset_index(drop=True)], axis=1)
    X.index = pd.Index(m1['job_name'].to_numpy(), name='job_name')
    if snap is not None:
        s = snap.set_index('job_name')[list(CHANNELS)].add_prefix('m2_')
        missing = ~X.index.isin(s.index)
        if missing.any():
            raise ValueError(f'{missing.sum()} jobs have no M2 snapshot')
        X = X.join(s)
    return X.drop(columns=list(drop))


def _regressor(params: dict, seed: int, n_estimators: int | None = None):
    import lightgbm as lgb
    p = {**BASE_PARAMS, **params, 'random_state': seed}
    if n_estimators is not None:
        p['n_estimators'] = n_estimators
    return lgb.LGBMRegressor(**p)


def fit_gbm(X_tr: pd.DataFrame, y_tr, inner, params: dict, seed: int, refit: bool = True):
    """Early-stop on the inner hold-out, then refit on all training rows.

    Rows with a missing target (e.g. no usage data) are ignored. With
    ``refit=False`` the early-stopped model itself is returned (grid search).
    Returns ``(model, best_iteration)``.
    """
    import lightgbm as lgb
    y_tr = np.asarray(y_tr, dtype=float)
    inner = np.asarray(inner, dtype=bool)
    ok = np.isfinite(y_tr)
    fit_rows, stop_rows = ok & ~inner, ok & inner
    if stop_rows.sum() == 0 or fit_rows.sum() == 0:
        raise ValueError('inner hold-out and fitting rows must both be non-empty')
    probe = _regressor(params, seed)
    probe.fit(X_tr[fit_rows], y_tr[fit_rows],
              eval_set=[(X_tr[stop_rows], y_tr[stop_rows])],
              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    best = int(probe.best_iteration_ or BASE_PARAMS['n_estimators'])
    if not refit:
        return probe, best
    model = _regressor(params, seed, n_estimators=best)
    model.fit(X_tr[ok], y_tr[ok])
    return model, best


def grid_search(X_tr, y_tr, inner, X_va, score_fn, grid=GRID, seed: int = 0) -> pd.DataFrame:
    """Fit every grid setting with one seed; ``score_fn(pred_va)`` (lower is better)."""
    rows = []
    for i, params in enumerate(grid):
        model, best = fit_gbm(X_tr, y_tr, inner, params, seed, refit=False)
        rows.append({'setting': i, **params, 'best_iter': best,
                     'score': float(score_fn(model.predict(X_va)))})
    return pd.DataFrame(rows).sort_values('score').reset_index(drop=True)


def train_seeds(X_tr, y_tr, inner, X_va, params: dict, seeds=SEEDS):
    """Train the chosen setting with several seeds.

    Returns ``(pred_va, best_iters, models)`` with ``pred_va`` shaped
    (n_seeds, n_val).
    """
    preds, iters, models = [], [], []
    for seed in seeds:
        model, best = fit_gbm(X_tr, y_tr, inner, params, seed)
        preds.append(model.predict(X_va))
        iters.append(best)
        models.append(model)
    return np.vstack(preds), iters, models