import numpy as np

from adtech_ctr_auction.auction import BidLandscape, forecast_win_rates, run_auction


def test_win_prob_monotonic_in_bid():
    land = BidLandscape(mu=0.0, sigma=0.5)
    bids = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    probs = forecast_win_rates(bids, land)
    assert np.all(np.diff(probs) > 0)
    assert 0.0 < probs[0] < probs[-1] <= 1.0  # CDF saturates to 1.0 at extreme bids
    assert abs(float(land.win_prob(np.exp(0.0))) - 0.5) < 1e-9


def test_fit_from_bids_uses_median():
    bids = np.array([1.0, 2.0, 4.0, 8.0])
    land = BidLandscape.fit_from_bids(bids, competition_factor=1.0)
    assert abs(land.median_bid() - float(np.exp(np.log(bids).mean()))) < 1e-9
    land2 = BidLandscape.fit_from_bids(bids, competition_factor=2.0)
    assert abs(land2.median_bid() - 2.0 * land.median_bid()) < 1e-9


def test_auction_second_price_and_budget():
    rng = np.random.default_rng(7)
    pctr = rng.uniform(0.05, 0.5, size=2000)
    land = BidLandscape.fit_from_bids(pctr * 2.5)
    res = run_auction(pctr, 2.5, land, budget=1e12, pacing=False, seed=1)
    assert res["wins"] > 0 and res["wins"] < res["n_impressions"]
    assert res["avg_clearing_price"] > 0
    assert res["avg_clearing_price"] < res["avg_bid"]
    assert 0.0 < res["actual_win_rate"] < 1.0
    # Forecast tracks simulation: decile MAE small on 2000 draws.
    assert res["decile_mae"] < 0.10
    assert res["win_rate_abs_error"] < 0.05


def test_auction_deterministic_with_seed():
    rng = np.random.default_rng(3)
    pctr = rng.uniform(0.05, 0.5, size=1000)
    land = BidLandscape.fit_from_bids(pctr * 2.5)
    a = run_auction(pctr, 2.5, land, budget=100.0, pacing=True, seed=42)
    b = run_auction(pctr, 2.5, land, budget=100.0, pacing=True, seed=42)
    assert a["spend"] == b["spend"] and a["wins"] == b["wins"]


def test_pacing_respects_budget_and_shades_bids():
    rng = np.random.default_rng(11)
    pctr = rng.uniform(0.05, 0.5, size=5000)
    land = BidLandscape.fit_from_bids(pctr * 2.5)
    free = run_auction(pctr, 2.5, land, budget=1e12, pacing=False, seed=5)
    paced = run_auction(pctr, 2.5, land, budget=0.5 * free["spend"],
                        pacing=True, seed=5)
    assert paced["spend"] <= paced["budget"] + 1e-9
    assert paced["spend"] > 0.4 * paced["budget"]  # pacing spends most of it
    mults = [t["multiplier"] for t in paced["trajectory"]]
    assert all(0.1 <= m <= 2.0 for m in mults)
    assert len(paced["trajectory"]) == 24
    # Pacing changes behaviour: fewer wins than the free run.
    assert paced["wins"] < free["wins"]


def test_reserve_price_raises_clearing_price_and_cuts_wins():
    from adtech_ctr_auction.auction import sweep_reserve_prices

    rng = np.random.default_rng(21)
    pctr = rng.uniform(0.05, 0.5, size=5000)
    land = BidLandscape.fit_from_bids(pctr * 2.5)
    base = run_auction(pctr, 2.5, land, budget=1e12, pacing=False, seed=9)
    floored = run_auction(pctr, 2.5, land, budget=1e12, pacing=False,
                          seed=9, reserve_price=0.40)
    assert floored["wins"] <= base["wins"]
    assert floored["avg_clearing_price"] >= base["avg_clearing_price"]
    assert floored["reserve_price"] == 0.40
    # Sweep: a very high floor must earn less than no floor.
    sweep = sweep_reserve_prices(pctr, 2.5, land, [0.0, 0.2, 0.4, 2.0],
                                 seed=9)
    assert sweep["floors"][0]["spend"] > sweep["floors"][-1]["spend"]
    assert sweep["best_reserve_price"] in (0.0, 0.2, 0.4)


def test_adload_cap_bounds_wins_per_bucket():
    rng = np.random.default_rng(31)
    pctr = rng.uniform(0.05, 0.5, size=4800)
    land = BidLandscape.fit_from_bids(pctr * 2.5)
    cap = 50
    res = run_auction(pctr, 2.5, land, budget=1e12, pacing=False, seed=13,
                      max_wins_per_bucket=cap)
    assert all(t["wins"] <= cap for t in res["trajectory"])
    assert res["wins_suppressed_by_adload_cap"] > 0  # cap binds here
    free = run_auction(pctr, 2.5, land, budget=1e12, pacing=False, seed=13)
    assert res["wins"] < free["wins"]
    assert res["wins"] + res["wins_suppressed_by_adload_cap"] == free["wins"]


def test_brand_safety_blocks_categories():
    rng = np.random.default_rng(41)
    pctr = rng.uniform(0.05, 0.5, size=3000)
    land = BidLandscape.fit_from_bids(pctr * 2.5)
    cats = rng.choice(
        np.array(["beer", "travel", "political", "tobacco"], dtype=object),
        size=3000, p=[0.4, 0.4, 0.1, 0.1],
    )
    res = run_auction(pctr, 2.5, land, budget=1e12, pacing=False, seed=17,
                      categories=cats,
                      blocked_categories={"political", "tobacco"})
    blocked_n = int(sum(1 for c in cats if c in ("political", "tobacco")))
    assert res["impressions_blocked_by_brand_safety"] == blocked_n
    assert blocked_n > 0
    free = run_auction(pctr, 2.5, land, budget=1e12, pacing=False, seed=17)
    assert res["wins"] <= free["wins"]
