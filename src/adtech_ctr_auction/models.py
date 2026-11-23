"""CTR models: logistic regression and gradient boosting.

Both models are plain scikit-learn, CPU-only. Evaluation is a single
held-out split (no cross-validation — one GBM fit on 200k rows is
already minutes on CPU); the split is stratified and seeded so the
numbers reproduce exactly.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from .features import build_gbm_features, build_lr_features, gbm_categorical_indices


def train_logistic_regression(X_train, y_train, max_categories: int = 50, C: float = 1.0):
    pipe = Pipeline(
        [
            ("prep", build_lr_features(max_categories=max_categories)),
            (
                "clf",
                LogisticRegression(
                    C=C, max_iter=1000, solver="lbfgs",
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def train_gbm(X_train, y_train, max_iter: int = 300, learning_rate: float = 0.06,
              max_leaf_nodes: int = 31, random_state: int = 42):
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        categorical_features=gbm_categorical_indices(),
        random_state=random_state,
    )
    pipe = Pipeline([("prep", build_gbm_features()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate(model, X_test, y_test) -> dict:
    """AUC, log loss, Brier score and a 10-bin reliability curve."""
    proba = model.predict_proba(X_test)[:, 1]
    proba = np.clip(proba, 1e-7, 1 - 1e-7)
    auc = float(roc_auc_score(y_test, proba))
    ll = float(log_loss(y_test, proba))
    brier = float(brier_score_loss(y_test, proba))
    # Reliability curve: 10 equal-width probability bins.
    bins = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.digitize(proba, bins) - 1, 0, 9)
    curve = []
    for b in range(10):
        mask = idx == b
        curve.append(
            {
                "bin": b,
                "bin_lo": float(bins[b]),
                "bin_hi": float(bins[b + 1]),
                "n": int(mask.sum()),
                "mean_pred": float(proba[mask].mean()) if mask.any() else float("nan"),
                "frac_pos": float(y_test[mask].mean()) if mask.any() else float("nan"),
            }
        )
    # Expected calibration error (count-weighted |mean_pred - frac_pos|).
    n = len(y_test)
    ece = float(
        sum(
            (c["n"] / n) * abs(c["mean_pred"] - c["frac_pos"])
            for c in curve
            if c["n"] > 0
        )
    )
    return {
        "auc": auc,
        "logloss": ll,
        "brier": brier,
        "ece": ece,
        "mean_pred": float(proba.mean()),
        "base_rate": float(np.mean(y_test)),
        "reliability_curve": curve,
    }
