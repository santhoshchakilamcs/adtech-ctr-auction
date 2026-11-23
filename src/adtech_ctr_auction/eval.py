"""End-to-end evaluation: load Criteo sample, train CTR models, run the
auction simulator, and return one results dict (written to results.json
by run_eval.py).

Split: FORWARD IN TIME. The sample's readme states the rows are
chronologically ordered, so train is the first ``train_rows`` rows and
test is the next ``test_rows`` rows (later in time). No shuffling, no
stratification — this is the honest adtech setup (predict future
traffic from past traffic), and it is stated as such.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve

from .auction import BidLandscape, run_auction, sweep_reserve_prices
from .data import DEFAULT_TEST_ROWS, DEFAULT_TRAIN_ROWS, describe, load_rows
from .models import evaluate, train_gbm, train_logistic_regression

VALUE_PER_CLICK = 2.50  # assumed advertiser value per click ($), documented
SEED = 42

# Marketplace constraints, inspired by how Netflix's ad tier actually
# runs (see README for sources). These are POLICY choices applied to
# the simulator, not measurements from Netflix's systems:
# - Ad-load cap: Basic With Ads runs about 4-5 minutes of ads per
#   hour (Deadline, Nov 3 2022). Here: at most 35% of a bucket's
#   impressions may be won.
# - Brand safety: political, firearms and tobacco ads are excluded
#   (Deadline, Nov 3 2022). The Criteo sample has NO advertiser
#   category column, so categories below are SYNTHETIC — assigned
#   deterministically from a seeded RNG with plausible shares; the
#   filter's mechanics are real, the labels are not measured.
# - Reserve price: a floor on the clearing price, swept to find the
#   revenue-maximising level (the dynamic-pricing / yield lever).
ADLOAD_CAP_FRAC = 0.35
BLOCKED_CATEGORIES = ("political", "firearms", "tobacco")
CATEGORY_SHARES = {
    "beer": 0.10,
    "automotive": 0.15,
    "fast_casual": 0.12,
    "travel": 0.18,
    "consumer_packaged_goods": 0.41,
    "political": 0.02,
    "firearms": 0.01,
    "tobacco": 0.01,
}
RESERVE_FLOORS = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00]


def synthetic_categories(n: int, seed: int = SEED) -> np.ndarray:
    """Deterministic synthetic advertiser categories per impression.

    Shares follow the launch mix reported for Netflix's ad tier (beer,
    cars, fast-casual, travel, consumer packaged goods) plus the
    excluded set (political, firearms, tobacco). Labels are synthetic:
    the Criteo sample has no category column.
    """
    rng = np.random.default_rng(seed)
    names = list(CATEGORY_SHARES)
    probs = np.array([CATEGORY_SHARES[c] for c in names], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(np.array(names, dtype=object), size=n, p=probs)


def _roc_points(y_test, proba, n_points: int = 200) -> list[list[float]]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    idx = np.unique(np.linspace(0, len(fpr) - 1, n_points).astype(int))
    return [[float(fpr[i]), float(tpr[i])] for i in idx]


def run_eval(train_rows: int = DEFAULT_TRAIN_ROWS,
             test_rows: int = DEFAULT_TEST_ROWS,
             data_path: str = "data/raw/dac_sample.txt",
             gbm_max_iter: int = 300) -> dict:
    total = train_rows + test_rows
    df = load_rows(data_path, n_rows=total)
    stats = describe(df)
    X = df.drop(columns=["label"])
    y = df["label"].to_numpy()
    # Forward-in-time split: rows are chronologically ordered.
    X_train, X_test = X.iloc[:train_rows], X.iloc[train_rows:total]
    y_train, y_test = y[:train_rows], y[train_rows:total]
    results: dict = {
        "data": {
            **stats,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_ctr": float(y_train.mean()),
            "test_ctr": float(y_test.mean()),
        },
        "params": {
            "value_per_click_usd": VALUE_PER_CLICK,
            "gbm_max_iter": gbm_max_iter,
            "lr_max_categories": 50,
            "seed": SEED,
        },
    }

    lr = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate(lr, X_test, y_test)
    lr_metrics["roc_curve"] = _roc_points(
        y_test, lr.predict_proba(X_test)[:, 1]
    )
    results["logistic_regression"] = lr_metrics

    gbm = train_gbm(X_train, y_train, max_iter=gbm_max_iter)
    gbm_metrics = evaluate(gbm, X_test, y_test)
    gbm_metrics["roc_curve"] = _roc_points(
        y_test, gbm.predict_proba(X_test)[:, 1]
    )
    results["gbm"] = gbm_metrics

    # Auction: use the GBM's predicted CTR (the better model) on the
    # held-out impressions, in file order as the arrival order.
    pctr = gbm.predict_proba(X_test)[:, 1]
    bids = pctr * VALUE_PER_CLICK
    landscape = BidLandscape.fit_from_bids(bids, competition_factor=1.0)
    results["auction_landscape"] = {
        "mu": landscape.mu,
        "sigma": landscape.sigma,
        "median_competing_bid_usd": landscape.median_bid(),
        "competition_factor": 1.0,
    }
    # 1) Unconstrained (no pacing, effectively unlimited budget): the
    #    natural spend / win rate of value-based bidding.
    free = run_auction(
        pctr, VALUE_PER_CLICK, landscape,
        budget=1e12, n_buckets=24, pacing=False, seed=SEED,
    )
    # 2) Paced: budget = half the unconstrained spend, 24 buckets.
    paced_budget = 0.5 * free["spend"]
    paced = run_auction(
        pctr, VALUE_PER_CLICK, landscape,
        budget=paced_budget, n_buckets=24, pacing=True, seed=SEED,
    )
    results["auction_unconstrained"] = free
    results["auction_paced"] = paced

    # 3) Marketplace constraints (Netflix-informed policy layer).
    # Synthetic categories, seeded and deterministic.
    cats = synthetic_categories(len(pctr), seed=SEED)
    results["marketplace"] = {
        "adload_cap_frac": ADLOAD_CAP_FRAC,
        "blocked_categories": list(BLOCKED_CATEGORIES),
        "category_shares": CATEGORY_SHARES,
        "reserve_floors": RESERVE_FLOORS,
        "categories_synthetic": True,
    }
    # 3a) Brand-safe paced run: blocked categories never win.
    brand_safe = run_auction(
        pctr, VALUE_PER_CLICK, landscape,
        budget=paced_budget, n_buckets=24, pacing=True, seed=SEED,
        categories=cats, blocked_categories=set(BLOCKED_CATEGORIES),
    )
    results["auction_brand_safe"] = brand_safe
    # 3b) Ad-load-capped paced run: at most 35% of a bucket's
    # impressions may be won (member-experience constraint).
    bucket_size = int(np.ceil(len(pctr) / 24))
    cap = max(1, int(round(ADLOAD_CAP_FRAC * bucket_size)))
    capped = run_auction(
        pctr, VALUE_PER_CLICK, landscape,
        budget=paced_budget, n_buckets=24, pacing=True, seed=SEED,
        max_wins_per_bucket=cap,
    )
    capped["adload_cap"] = cap
    results["auction_adload_capped"] = capped
    # 3c) Reserve-price sweep on the unconstrained auction.
    results["auction_reserve_sweep"] = sweep_reserve_prices(
        pctr, VALUE_PER_CLICK, landscape, RESERVE_FLOORS, seed=SEED,
    )
    return results
