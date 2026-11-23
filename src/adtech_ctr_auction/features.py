"""Feature preprocessing for the Criteo sample.

Honest handling of the raw file, documented so the numbers are
reproducible:

* Integer features (I1..I13): missing values are filled with the
  training median, then standardised (logistic regression only).
  No clipping, no winsorising — the raw magnitudes are kept.
* Categorical features (C1..C26): hashed 32-bit hex strings. Missing
  (empty in the raw file) becomes the literal level ``"MISSING"``.
  - Logistic regression: one-hot of the most frequent levels only
    (``max_categories`` per column, default 50; everything rarer
    folds into ``"OTHER"``). Rare hashed levels carry no signal a
    linear model can use, and full one-hot would be millions of
    columns.
  - Gradient boosting: ordinal encoding (integer codes, unknown /
    missing = -1) with HistGradientBoosting's native categorical
    support, so the tree can split on the raw hashed levels.
    HistGradientBoosting caps categorical cardinality at max_bins
    (255), so only the ``max_categories`` most frequent levels per
    column are kept as their own codes; rarer and unseen levels fold
    into a single OTHER code (see TopKOrdinalEncoder).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CAT_COLS, INT_COLS

MISSING_LEVEL = "MISSING"
OTHER_LEVEL = "OTHER"


class TopKOrdinalEncoder(BaseEstimator, TransformerMixin):
    """Ordinal-encode categoricals, keeping only the most frequent levels.

    Fits the top ``max_categories`` levels per column on the training
    data (missing filled with ``MISSING`` first, so it competes on
    frequency like any other level). Each kept level gets its own
    integer code; every rarer or unseen level maps to a single OTHER
    code. With ``max_categories=254`` the output cardinality is at
    most 255, which is HistGradientBoosting's ``max_bins`` cap for
    categorical features.
    """

    def __init__(self, max_categories: int = 254):
        self.max_categories = max_categories

    def fit(self, X, y=None):
        Xf = pd.DataFrame(X).fillna(MISSING_LEVEL).astype(str)
        self.maps_ = []
        for j in range(Xf.shape[1]):
            top = list(Xf.iloc[:, j].value_counts().head(self.max_categories).index)
            self.maps_.append({v: i for i, v in enumerate(top)})
        return self

    def transform(self, X):
        Xf = pd.DataFrame(X).fillna(MISSING_LEVEL).astype(str)
        out = np.empty(Xf.shape, dtype=np.int64)
        for j, mp in enumerate(self.maps_):
            other = len(mp)
            out[:, j] = [mp.get(v, other) for v in Xf.iloc[:, j]]
        return out


def _prep_cats(df: pd.DataFrame, top_levels: dict[str, set] | None) -> pd.DataFrame:
    """Fill missing cats and fold rare levels into OTHER (LR path)."""
    out = df[CAT_COLS].fillna(MISSING_LEVEL).astype(str)
    if top_levels is not None:
        for c in CAT_COLS:
            keep = top_levels[c]
            out[c] = np.where(out[c].isin(keep), out[c], OTHER_LEVEL)
    return out


def fit_top_levels(df: pd.DataFrame, max_categories: int = 50) -> dict[str, set]:
    """Most frequent levels per categorical column (train only)."""
    top = {}
    filled = df[CAT_COLS].fillna(MISSING_LEVEL).astype(str)
    for c in CAT_COLS:
        top[c] = set(filled[c].value_counts().head(max_categories).index)
    return top


def build_lr_features(max_categories: int = 50):
    """Preprocessor for logistic regression: scaled numerics + top-K one-hot."""
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_LEVEL)),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    max_categories=max_categories,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, INT_COLS),
            ("cat", categorical, CAT_COLS),
        ]
    )


def build_gbm_features(max_categories: int = 254):
    """Preprocessor for HistGradientBoosting: median-imputed numerics +
    top-K ordinal-encoded categoricals (see TopKOrdinalEncoder)."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = TopKOrdinalEncoder(max_categories=max_categories)
    return ColumnTransformer(
        [
            ("num", numeric, INT_COLS),
            ("cat", categorical, CAT_COLS),
        ]
    )


def gbm_categorical_indices(n_numeric: int = 13) -> list[int]:
    """Column positions of the categorical features after the transformer.

    ColumnTransformer concatenates numeric block first, then categorical.
    """
    return list(range(n_numeric, n_numeric + len(CAT_COLS)))
