"""Preprocessing for Stage 3 models (decision a, 12.11.2026).

Everything is fitted on the training period only and then applied unchanged
to validation and test.

* ``CategoryMap``: categorical levels -> integer codes. Code 0 is reserved for
  "other": levels seen fewer than ``min_count`` times in training, levels never
  seen in training (e.g. new users) and missing values. Used as LightGBM
  categorical codes (``as_category``) and as embedding indices in DL models.
* ``NumericScaler`` (DL only; GBM uses raw values): log1p for non-negative,
  non-binary columns with training skewness > ``skew_threshold``; then
  standardisation with training mean / std; missing values -> 0 (the training
  mean) plus an ``<col>_isna`` indicator for every column that has missing
  values in training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

OTHER = 0


@dataclass
class CategoryMap:
    """Training-only mapping of categorical levels to codes (0 = other)."""
    mapping: dict = field(default_factory=dict)   # column -> {level: code >= 1}
    min_count: int = 20

    @classmethod
    def fit(cls, train: pd.DataFrame, cols, min_count: int = 20) -> 'CategoryMap':
        mapping = {}
        for c in cols:
            counts = train[c].dropna().astype(str).value_counts()
            kept = counts[counts >= min_count]
            # Deterministic order: most frequent first, ties by level name
            order = sorted(kept.index, key=lambda lv: (-kept[lv], lv))
            mapping[c] = {lv: i + 1 for i, lv in enumerate(order)}
        return cls(mapping=mapping, min_count=min_count)

    @property
    def cardinalities(self) -> dict:
        """Number of codes per column, including the 'other' code."""
        return {c: len(m) + 1 for c, m in self.mapping.items()}

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c, m in self.mapping.items():
            s = frame[c]
            codes = s.astype(str).map(m).where(s.notna())
            out[c] = codes.fillna(OTHER).astype(np.int64).to_numpy()
        return pd.DataFrame(out, index=frame.index)

    def as_category(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Codes as pandas categoricals with fixed categories (for LightGBM)."""
        codes = self.transform(frame)
        for c, k in self.cardinalities.items():
            codes[c] = pd.Categorical(codes[c], categories=range(k))
        return codes


@dataclass
class NumericScaler:
    """Training-only numeric transform for neural networks."""
    cols: list = field(default_factory=list)
    log_cols: list = field(default_factory=list)
    mean: dict = field(default_factory=dict)
    std: dict = field(default_factory=dict)
    na_cols: list = field(default_factory=list)

    @classmethod
    def fit(cls, train: pd.DataFrame, cols, skew_threshold: float = 2.0) -> 'NumericScaler':
        log_cols, mean, std, na_cols = [], {}, {}, []
        for c in cols:
            x = train[c].to_numpy(dtype=float)
            finite = x[np.isfinite(x)]
            if np.isnan(x).any():
                na_cols.append(c)
            if len(finite) == 0:
                raise ValueError(f'column {c} has no finite training values')
            binary = np.unique(finite).size <= 2
            skew = stats.skew(finite) if finite.std() > 0 else 0.0
            if finite.min() >= 0 and not binary and skew > skew_threshold:
                log_cols.append(c)
                finite = np.log1p(finite)
            mean[c] = float(finite.mean())
            sd = float(finite.std())
            std[c] = sd if sd > 0 else 1.0
        return cls(cols=list(cols), log_cols=log_cols, mean=mean, std=std, na_cols=na_cols)

    @property
    def out_cols(self) -> list:
        return self.cols + [f'{c}_isna' for c in self.na_cols]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in self.cols:
            x = frame[c].to_numpy(dtype=float)
            if c in self.log_cols:
                x = np.log1p(np.clip(x, 0, None))   # unseen negatives are clipped
            z = (x - self.mean[c]) / self.std[c]
            out[c] = np.where(np.isfinite(z), z, 0.0)
        for c in self.na_cols:
            out[f'{c}_isna'] = frame[c].isna().to_numpy().astype(float)
        result = pd.DataFrame(out, index=frame.index)[self.out_cols].astype(np.float32)
        assert np.isfinite(result.to_numpy()).all()
        return result