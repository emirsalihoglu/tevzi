import numpy as np
import pandas as pd
import pytest

from tevzi.models.preprocess import OTHER, CategoryMap, NumericScaler


def _train():
    rng = np.random.default_rng(0)
    n = 1000
    return pd.DataFrame({
        'user': ['u1'] * 600 + ['u2'] * 380 + ['u_rare'] * 15 + [None] * 5,
        'gpu': rng.choice(['T4', 'V100'], n),
        'req': rng.lognormal(3, 2, n),             # skewed, non-negative -> log
        'hour': np.sin(rng.uniform(0, 6.3, n)),    # symmetric, negative -> no log
        'flag': rng.integers(0, 2, n),             # binary -> no log
        'hist': np.where(rng.random(n) < 0.3, np.nan, rng.normal(50, 5, n)),
    })


def test_category_map_rare_unseen_missing_go_to_other():
    cm = CategoryMap.fit(_train(), ['user', 'gpu'], min_count=20)
    assert cm.mapping['user'] == {'u1': 1, 'u2': 2}           # u_rare (15) dropped
    assert cm.cardinalities == {'user': 3, 'gpu': 3}
    val = pd.DataFrame({'user': ['u2', 'u_rare', 'new_user', None, 'u1'],
                        'gpu': ['V100', 'T4', 'A100', 'T4', None]})
    codes = cm.transform(val)
    assert codes['user'].tolist() == [2, OTHER, OTHER, OTHER, 1]
    assert codes['gpu'].iloc[2] == OTHER and codes['gpu'].iloc[4] == OTHER


def test_category_map_uses_training_rows_only():
    train = _train()
    val = pd.DataFrame({'user': ['new_user'] * 500, 'gpu': ['T4'] * 500})
    cm = CategoryMap.fit(train, ['user'])
    assert 'new_user' not in cm.mapping['user']
    assert (cm.transform(val)['user'] == OTHER).all()


def test_as_category_has_fixed_categories_across_frames():
    cm = CategoryMap.fit(_train(), ['user'])
    a = cm.as_category(pd.DataFrame({'user': ['u1', 'u1']}))
    b = cm.as_category(pd.DataFrame({'user': ['zzz']}))
    assert list(a['user'].cat.categories) == list(b['user'].cat.categories) == [0, 1, 2]


def test_numeric_scaler_log_rule_standardisation_and_na():
    train = _train()
    cols = ['req', 'hour', 'flag', 'hist']
    sc = NumericScaler.fit(train, cols)
    assert sc.log_cols == ['req']                   # hist is symmetric -> no log
    assert sc.na_cols == ['hist']
    out = sc.transform(train)
    assert list(out.columns) == cols + ['hist_isna']
    for c in cols:
        observed = out[c][train[c].notna()] if c == 'hist' else out[c]
        assert observed.mean() == pytest.approx(0, abs=1e-5)
        assert observed.std(ddof=0) == pytest.approx(1, abs=1e-4)
    assert (out.loc[train['hist'].isna(), 'hist'] == 0).all()
    assert out['hist_isna'].sum() == train['hist'].isna().sum()


def test_numeric_scaler_is_fitted_on_training_only_and_handles_new_data():
    train = _train()
    sc = NumericScaler.fit(train, ['req', 'hour', 'hist'])
    before = (dict(sc.mean), dict(sc.std))
    val = pd.DataFrame({'req': [1e9, -5.0, np.nan], 'hour': [0.0, 1.0, -1.0],
                        'hist': [np.nan, 10.0, np.inf]})
    out = sc.transform(val)
    assert (dict(sc.mean), dict(sc.std)) == before   # transform never refits
    assert np.isfinite(out.to_numpy()).all()
    assert out.dtypes.unique().tolist() == [np.float32]