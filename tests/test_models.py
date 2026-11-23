import numpy as np

from adtech_ctr_auction.models import evaluate, train_gbm, train_logistic_regression


def test_models_train_and_evaluate(small_frame):
    X = small_frame.drop(columns=["label"])
    y = small_frame["label"].to_numpy()
    for train_fn in (train_logistic_regression, train_gbm):
        model = train_fn(X, y)
        metrics = evaluate(model, X, y)
        assert 0.0 <= metrics["auc"] <= 1.0
        assert metrics["logloss"] > 0
        assert 0.0 <= metrics["brier"] <= 1.0
        assert 0.0 <= metrics["ece"] <= 1.0
        assert len(metrics["reliability_curve"]) == 10
        assert sum(c["n"] for c in metrics["reliability_curve"]) == len(y)
        assert abs(metrics["base_rate"] - y.mean()) < 1e-9


def test_evaluate_perfect_separation():
    # A hand-built model stub: predict_proba returns the label.
    class Stub:
        def predict_proba(self, X):
            y = X[:, 0]
            return np.column_stack([1 - y, y])

    X = np.array([[1.0], [0.0], [1.0], [0.0]])
    y = np.array([1, 0, 1, 0])
    m = evaluate(Stub(), X, y)
    assert m["auc"] == 1.0
    assert m["logloss"] < 0.01
