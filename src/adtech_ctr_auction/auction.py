"""Second-price auction simulator with pacing and win-rate forecasting.

What is real and what is modelled — read this before quoting any
number from this module:

* The Criteo display-ads sample has NO bid, cost, or auction column.
  There is no real bid landscape in the data. The competing-bid
  landscape below is therefore a PARAMETRIC MODEL, not a measurement:
  the highest competing bid for an impression is drawn from a
  LogNormal distribution whose log-mean / log-std are calibrated to
  the distribution of our own value-based bids (median scaled by
  ``competition_factor``). This is the standard textbook treatment
  (bid-landscape forecasting assumes a parametric competing-bid
  distribution), but it must not be presented as validated against
  real auction logs — it is not.
* What the experiment DOES measure honestly: given that landscape
  model, how well does the analytic win-rate forecast
  ``P(competing bid < our bid)`` predict the win rate the Monte Carlo
  simulator actually produces, bid-decile by bid-decile — and how
  does a pacing controller change spend, win rate and clearing price
  under a fixed budget. The forecast error is the forecaster's error
  against simulated ground truth under the stated model, not against
  production auctions.

Mechanics (per impression, in arrival order):
* bid = pCTR * value_per_click * pacing_multiplier (value-based
  bidding: bid your expected value per impression).
* second price: if bid > highest competing bid, we win and pay the
  competing bid; otherwise we pay nothing and lose.
* pacing: the day is split into ``n_buckets`` equal arrival buckets
  with a per-bucket spend target of budget / n_buckets. After each
  bucket the multiplier is scaled by target/actual spend (clipped),
  so overspend shades bids down and underspend shades them up.
  When the budget is exhausted, bidding stops.
* reserve price (``reserve_price``): a floor on the clearing price.
  We win only if bid > max(competing bid, reserve) and pay
  max(competing bid, reserve) — the standard second-price auction
  with a reserve. This is the yield lever: a floor trades win rate
  for clearing price, and the revenue-maximising floor is measured
  by sweeping it (see ``sweep_reserve_prices``).
* ad-load cap (``max_wins_per_bucket``): a member-experience
  constraint. At most that many ads may be won per arrival bucket;
  further would-be wins in the bucket are suppressed and counted.
  This mirrors a real ad tier's ad-load limit (Netflix's Basic With
  Ads runs about 4-5 minutes of ads per hour).
* brand safety (``categories`` + ``blocked_categories``): impressions
  whose advertiser category is blocked are never bid on. The Criteo
  sample carries NO advertiser category, so categories used here are
  SYNTHETIC — assigned deterministically in eval.py from a seeded
  RNG, with the blocked set (political, firearms, tobacco) taken
  from Netflix's launch exclusions. The filter's mechanics are
  real; the category labels are not measured.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import lognorm, norm


class BidLandscape:
    """LogNormal model of the highest competing bid."""

    def __init__(self, mu: float, sigma: float):
        self.mu = float(mu)
        self.sigma = float(sigma)

    @classmethod
    def fit_from_bids(cls, bids: np.ndarray, competition_factor: float = 1.0):
        """Calibrate to the log-scale location/spread of ``bids``.

        The competitor median is our median bid times
        ``competition_factor`` (1.0 = equally strong competition);
        the spread is the spread of our own value-based bids.
        """
        bids = np.asarray(bids, dtype=float)
        bids = bids[bids > 0]
        if len(bids) == 0:
            raise ValueError("no positive bids to calibrate the landscape")
        log_bids = np.log(bids)
        mu = float(log_bids.mean() + np.log(competition_factor))
        sigma = float(log_bids.std())
        if sigma <= 0:
            sigma = 0.5
        return cls(mu, sigma)

    def win_prob(self, bid: np.ndarray | float) -> np.ndarray | float:
        """P(competing bid < bid): the analytic win-rate forecast."""
        return norm.cdf((np.log(np.maximum(bid, 1e-12)) - self.mu) / self.sigma)

    def sample_competing_bid(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.lognormal(mean=self.mu, sigma=self.sigma, size=n)

    def median_bid(self) -> float:
        return float(np.exp(self.mu))


def forecast_win_rates(bids: np.ndarray, landscape: BidLandscape) -> np.ndarray:
    return np.asarray(landscape.win_prob(np.asarray(bids, dtype=float)))


def run_auction(
    pctr: np.ndarray,
    value_per_click: float,
    landscape: BidLandscape,
    budget: float,
    n_buckets: int = 24,
    pacing: bool = True,
    seed: int = 42,
    reserve_price: float = 0.0,
    max_wins_per_bucket: int | None = None,
    categories: np.ndarray | None = None,
    blocked_categories: set[str] | frozenset[str] | None = None,
) -> dict:
    """Simulate a second-price auction over impressions in arrival order.

    Optional marketplace constraints (all default to off, reproducing
    the original behaviour): ``reserve_price`` (floor on the clearing
    price), ``max_wins_per_bucket`` (ad-load cap per arrival bucket),
    and ``categories``/``blocked_categories`` (brand-safety filter;
    categories are synthetic — see module docstring).

    Returns spend, wins, clearing prices, the pacing trajectory, and
    the forecast-vs-simulated win-rate comparison.
    """
    pctr = np.asarray(pctr, dtype=float)
    n = len(pctr)
    rng = np.random.default_rng(seed)
    base_bids = pctr * value_per_click
    if categories is not None:
        categories = np.asarray(categories, dtype=object)
        if len(categories) != n:
            raise ValueError("categories must have one entry per impression")
    blocked = set(blocked_categories or set())

    bucket_id = np.minimum((np.arange(n) * n_buckets) // max(n, 1), n_buckets - 1)
    target_per_bucket = budget / n_buckets

    multiplier = 1.0
    bids = np.zeros(n)
    competing = np.zeros(n)
    won = np.zeros(n, dtype=bool)
    paid = np.zeros(n)
    spend = 0.0
    trajectory = []
    suppressed_by_cap = 0
    blocked_by_brand = 0

    for b in range(n_buckets):
        mask = bucket_id == b
        idx = np.where(mask)[0]
        if len(idx) == 0:
            trajectory.append(
                {"bucket": b, "multiplier": multiplier, "spend": 0.0,
                 "wins": 0, "impressions": 0, "suppressed": 0, "blocked": 0}
            )
            continue
        room = budget - spend
        if room <= 0:
            trajectory.append(
                {"bucket": b, "multiplier": 0.0, "spend": 0.0,
                 "wins": 0, "impressions": int(len(idx)),
                 "suppressed": 0, "blocked": 0}
            )
            continue
        bucket_bids = base_bids[idx] * multiplier
        bucket_comp = landscape.sample_competing_bid(rng, len(idx))
        # Brand safety: blocked categories are never bid on.
        if categories is not None and blocked:
            bucket_blocked = np.array(
                [c in blocked for c in categories[idx]], dtype=bool
            )
        else:
            bucket_blocked = np.zeros(len(idx), dtype=bool)
        blocked_by_brand += int(bucket_blocked.sum())
        floor = np.maximum(bucket_comp, reserve_price)
        bucket_won = (~bucket_blocked) & (bucket_bids > floor)
        bucket_paid = np.where(bucket_won, floor, 0.0)
        # Ad-load cap: keep only the first max_wins_per_bucket wins
        # in arrival order; the rest are suppressed.
        bucket_suppressed = 0
        if max_wins_per_bucket is not None:
            win_positions = np.where(bucket_won)[0]
            if len(win_positions) > max_wins_per_bucket:
                drop = win_positions[max_wins_per_bucket:]
                bucket_suppressed = int(len(drop))
                bucket_won[drop] = False
                bucket_paid[drop] = 0.0
        suppressed_by_cap += bucket_suppressed
        bucket_spend = float(bucket_paid.sum())
        if bucket_spend > room:  # budget binds mid-bucket: take wins in order
            order = np.where(bucket_won)[0]
            kept = []
            running = 0.0
            for j in order:
                if running + bucket_paid[j] <= room:
                    kept.append(j)
                    running += bucket_paid[j]
                else:
                    break
            keep_mask = np.zeros(len(idx), dtype=bool)
            keep_mask[kept] = True
            bucket_won = bucket_won & keep_mask
            bucket_paid = np.where(bucket_won, floor, 0.0)
            bucket_spend = float(bucket_paid.sum())
        bids[idx] = bucket_bids
        competing[idx] = bucket_comp
        won[idx] = bucket_won
        paid[idx] = bucket_paid
        spend += bucket_spend
        trajectory.append(
            {
                "bucket": b,
                "multiplier": float(multiplier),
                "spend": bucket_spend,
                "wins": int(bucket_won.sum()),
                "impressions": int(len(idx)),
                "suppressed": bucket_suppressed,
                "blocked": int(bucket_blocked.sum()),
            }
        )
        if pacing:
            if bucket_spend > 0:
                ratio = target_per_bucket / bucket_spend
            else:
                ratio = 1.5  # no spend at all: shade bids up
            multiplier = float(np.clip(multiplier * np.clip(ratio, 0.5, 1.5), 0.1, 2.0))

    pred_win = forecast_win_rates(bids, landscape)
    actual_win_rate = float(won.mean())
    pred_win_rate = float(pred_win.mean())
    # Bid-decile forecast error: empirical vs predicted win rate per decile.
    deciles = []
    order = np.argsort(bids)
    chunks = np.array_split(order, 10)
    for d, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        deciles.append(
            {
                "decile": d,
                "n": int(len(chunk)),
                "mean_bid": float(bids[chunk].mean()),
                "pred_win_rate": float(pred_win[chunk].mean()),
                "actual_win_rate": float(won[chunk].mean()),
            }
        )
    decile_mae = float(
        np.mean([abs(x["pred_win_rate"] - x["actual_win_rate"]) for x in deciles])
    )
    n_wins = int(won.sum())
    return {
        "n_impressions": n,
        "budget": float(budget),
        "spend": float(spend),
        "budget_used_frac": float(spend / budget) if budget > 0 else 0.0,
        "wins": n_wins,
        "pred_win_rate": pred_win_rate,
        "actual_win_rate": actual_win_rate,
        "win_rate_abs_error": float(abs(pred_win_rate - actual_win_rate)),
        "decile_mae": decile_mae,
        "deciles": deciles,
        "avg_clearing_price": float(paid[won].mean()) if n_wins else 0.0,
        "avg_bid": float(bids.mean()),
        "median_competing_bid": landscape.median_bid(),
        "pacing": pacing,
        "reserve_price": float(reserve_price),
        "max_wins_per_bucket": (
            int(max_wins_per_bucket) if max_wins_per_bucket is not None else None
        ),
        "wins_suppressed_by_adload_cap": int(suppressed_by_cap),
        "impressions_blocked_by_brand_safety": int(blocked_by_brand),
        "trajectory": trajectory,
    }


def sweep_reserve_prices(
    pctr: np.ndarray,
    value_per_click: float,
    landscape: BidLandscape,
    floors: list[float],
    seed: int = 42,
) -> dict:
    """Sweep a reserve-price floor over the unconstrained auction.

    Each floor runs with effectively unlimited budget and no pacing,
    so the comparison is purely floor vs no-floor. Returns per-floor
    spend (platform revenue), wins and average clearing price, plus
    the revenue-maximising floor. Revenue here is simulated spend
    under the parametric landscape, not real money.
    """
    rows = []
    for floor in floors:
        res = run_auction(
            pctr, value_per_click, landscape,
            budget=1e12, n_buckets=24, pacing=False, seed=seed,
            reserve_price=float(floor),
        )
        rows.append(
            {
                "reserve_price": float(floor),
                "spend": res["spend"],
                "wins": res["wins"],
                "avg_clearing_price": res["avg_clearing_price"],
                "actual_win_rate": res["actual_win_rate"],
            }
        )
    best = max(rows, key=lambda r: r["spend"])
    return {
        "floors": rows,
        "best_reserve_price": best["reserve_price"],
        "best_spend": best["spend"],
        "best_wins": best["wins"],
        "no_floor_spend": rows[0]["spend"] if rows else 0.0,
    }
