import numpy as np
import pandas as pd
import pytest

from tevzi.features.m1 import CATEGORICAL
from tevzi.features.m2 import CHANNELS
from tevzi.models.preprocess import CategoryMap
from tevzi.models.gbm import build_features


def _m1(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'job_name': [f'j{i}' for i in range(n)],
                       'x1': rng.normal(size=n), 'x2': rng.lognormal(size=n)})
    for c in CATEGORICAL:
        df[c] = rng.choice(['a', 'b', 'c'], n)
    return df


def _snap(m1, seed=1):
    rng = np.random.default_rng(seed)
    s = pd.DataFrame({'job_name': m1['job_name']})
    for ch in CHANNELS:
        s[ch] = rng.normal(size=len(m1))
    return s.sample(frac=1, random_state=seed)          # shuffled on purpose


def test_build_features_layout_and_drop():
    m1 = _m1()
    cmap = CategoryMap.fit(m1, CATEGORICAL, min_count=1)
    X = build_features(m1, cmap)
    assert list(X.index) == list(m1['job_name'])
    assert set(CATEGORICAL) <= set(X.columns)
    assert all(str(X[c].dtype) == 'category' for c in CATEGORICAL)
    assert 'user' not in build_features(m1, cmap, drop=('user',)).columns


def test_build_features_aligns_snapshot_by_job_name():
    m1 = _m1()
    snap = _snap(m1)
    cmap = CategoryMap.fit(m1, CATEGORICAL, min_count=1)
    X = build_features(m1, cmap, snap=snap)
    expected = snap.set_index('job_name').loc[m1['job_name'], CHANNELS[0]].to_numpy()
    assert np.allclose(X[f'm2_{CHANNELS[0]}'].to_numpy(), expected)
    with pytest.raises(ValueError):
        build_features(m1, cmap, snap=snap.iloc[:-5])


def test_fit_gbm_early_stops_refits_and_ignores_missing_targets():
    pytest.importorskip('lightgbm')
    from tevzi.models.gbm import fit_gbm, train_seeds
    m1 = _m1(n=2000)
    cmap = CategoryMap.fit(m1, CATEGORICAL, min_count=1)
    X = build_features(m1, cmap)
    rng = np.random.default_rng(3)
    y = 2 * X['x1'].to_numpy() + rng.normal(0, 0.1, len(X))
    y[:100] = np.nan                                   # e.g. jobs without usage data
    inner = np.arange(len(X)) >= 1600
    model, best = fit_gbm(X, y, inner, {'num_leaves': 15, 'min_child_samples': 20}, seed=0)
    assert 1 <= best < 3000
    assert model.n_estimators == best
    assert np.corrcoef(model.predict(X.iloc[1600:]), y[1600:])[0, 1] > 0.9
    preds, iters, _ = train_seeds(X, y, inner, X.iloc[:50], {'num_leaves': 15}, seeds=(0, 1))
    assert preds.shape == (2, 50) and len(iters) == 2
    again, _, _ = train_seeds(X, y, inner, X.iloc[:50], {'num_leaves': 15}, seeds=(0,))
    assert np.allclose(again[0], preds[0])             # same seed -> same model
    probe, best2 = fit_gbm(X, y, inner, {'num_leaves': 15}, seed=0, refit=False)
    assert best2 == iters[0] and probe.n_estimators == 3000   # early-stopped, not refitted