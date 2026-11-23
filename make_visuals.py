#!/usr/bin/env python3
"""Regenerate results/*.svg + results/report.html from results.json.

Usage: python make_visuals.py [--results results.json]
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"wrote {path}")


def roc_svg(results, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    for key, label in (("logistic_regression", "Logistic regression"),
                       ("gbm", "HistGradientBoosting")):
        pts = results[key]["roc_curve"]
        fpr = [p[0] for p in pts]
        tpr = [p[1] for p in pts]
        ax.plot(fpr, tpr, label=f"{label} (AUC {results[key]['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — CTR prediction, held-out test set")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    _save(fig, out)


def reliability_svg(results, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    for key, label, marker in (("logistic_regression", "Logistic regression", "o"),
                               ("gbm", "HistGradientBoosting", "s")):
        curve = results[key]["reliability_curve"]
        xs = [c["mean_pred"] for c in curve if c["n"] > 0]
        ys = [c["frac_pos"] for c in curve if c["n"] > 0]
        ax.plot(xs, ys, marker=marker, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted CTR (bin)")
    ax.set_ylabel("Observed click rate (bin)")
    ax.set_title("Reliability curve — 10 equal-width bins")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, out)


def win_rate_svg(results, out):
    a = results["auction_paced"]
    deciles = a["deciles"]
    xs = [d["decile"] for d in deciles]
    pred = [d["pred_win_rate"] for d in deciles]
    actual = [d["actual_win_rate"] for d in deciles]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    width = 0.35
    ax.bar([x - width / 2 for x in xs], pred, width, label="Forecast win rate")
    ax.bar([x + width / 2 for x in xs], actual, width, label="Simulated actual win rate")
    ax.set_xlabel("Bid decile (0 = lowest bids)")
    ax.set_ylabel("Win rate")
    ax.set_title("Bid-landscape win-rate forecast vs simulated actual (paced)")
    ax.set_xticks(xs)
    ax.legend()
    ax.set_ylim(0, 1)
    _save(fig, out)


def pacing_svg(results, out):
    a = results["auction_paced"]
    traj = a["trajectory"]
    xs = [t["bucket"] for t in traj]
    spend = [t["spend"] for t in traj]
    mult = [t["multiplier"] for t in traj]
    target = a["budget"] / len(traj)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True)
    ax1.bar(xs, spend, label="Spend per bucket")
    ax1.axhline(target, color="red", ls="--", lw=1,
                label=f"Target ${target:.2f}/bucket")
    ax1.set_ylabel("Spend ($)")
    ax1.set_title("Pacing: spend per bucket vs target")
    ax1.legend()
    ax2.plot(xs, mult, marker="o")
    ax2.set_xlabel("Arrival bucket (24 across the test set)")
    ax2.set_ylabel("Pacing multiplier")
    ax2.set_title("Pacing multiplier trajectory")
    ax2.set_ylim(0, 2.1)
    _save(fig, out)


def reserve_svg(results, out):
    s = results["auction_reserve_sweep"]
    floors = [r["reserve_price"] for r in s["floors"]]
    spend = [r["spend"] for r in s["floors"]]
    wins = [r["wins"] for r in s["floors"]]
    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax1.plot(floors, spend, marker="o", label="Revenue (spend)")
    ax1.axvline(s["best_reserve_price"], color="red", ls="--", lw=1,
                label=f"Best floor ${s['best_reserve_price']:.2f}")
    ax1.set_xlabel("Reserve price ($)")
    ax1.set_ylabel("Revenue ($)")
    ax1.set_title("Reserve-price sweep — revenue vs floor (unconstrained)")
    ax2 = ax1.twinx()
    ax2.plot(floors, wins, marker="s", color="green", label="Wins")
    ax2.set_ylabel("Wins")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    _save(fig, out)


def report_html(results, out):
    lr = results["logistic_regression"]
    gbm = results["gbm"]
    a = results["auction_paced"]
    f = results["auction_unconstrained"]
    bs = results["auction_brand_safe"]
    cap = results["auction_adload_capped"]
    sweep = results["auction_reserve_sweep"]
    d = results["data"]
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Adtech CTR + Auction — Results</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:2em auto;padding:0 1em}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 10px;text-align:left}}
img{{max-width:100%}}</style></head><body>
<h1>Adtech CTR Prediction + Second-Price Auction</h1>
<p>Personal project by Santhosh Chakilam. Data: Criteo Display Advertising
Challenge 100k-row sample (Microsoft Recommenders <code>dac_sample</code>).
{d['n_rows']} rows loaded (CTR {d['ctr']:.4f}); train {d['train_rows']} /
test {d['test_rows']} stratified random split (seed 42). Value per click
${results['params']['value_per_click_usd']:.2f} (assumed).</p>
<h2>CTR prediction (held-out test set)</h2>
<table><tr><th>Model</th><th>AUC</th><th>Log loss</th><th>Brier</th><th>ECE</th></tr>
<tr><td>Logistic regression</td><td>{lr['auc']:.4f}</td><td>{lr['logloss']:.4f}</td>
<td>{lr['brier']:.4f}</td><td>{lr['ece']:.4f}</td></tr>
<tr><td>HistGradientBoosting</td><td>{gbm['auc']:.4f}</td><td>{gbm['logloss']:.4f}</td>
<td>{gbm['brier']:.4f}</td><td>{gbm['ece']:.4f}</td></tr></table>
<p><img src="roc_comparison.svg" alt="ROC curves"></p>
<p><img src="reliability.svg" alt="Reliability curve"></p>
<h2>Second-price auction (GBM pCTR, LogNormal bid landscape)</h2>
<p>Competing-bid landscape: LogNormal(mu={results['auction_landscape']['mu']:.3f},
sigma={results['auction_landscape']['sigma']:.3f}), median competing bid
${results['auction_landscape']['median_competing_bid_usd']:.3f}. The landscape is a
parametric model calibrated to our own value-based bids — there is no bid
column in the Criteo sample, so this is NOT validated against real auction logs.</p>
<table><tr><th>Run</th><th>Spend</th><th>Budget</th><th>Wins</th>
<th>Pred win rate</th><th>Actual win rate</th><th>Abs error</th><th>Decile MAE</th></tr>
<tr><td>Unconstrained</td><td>${f['spend']:.2f}</td><td>—</td><td>{f['wins']}</td>
<td>{f['pred_win_rate']:.3f}</td><td>{f['actual_win_rate']:.3f}</td>
<td>{f['win_rate_abs_error']:.4f}</td><td>{f['decile_mae']:.4f}</td></tr>
<tr><td>Paced (budget ${a['budget']:.2f})</td><td>${a['spend']:.2f}</td>
<td>${a['budget']:.2f}</td><td>{a['wins']}</td>
<td>{a['pred_win_rate']:.3f}</td><td>{a['actual_win_rate']:.3f}</td>
<td>{a['win_rate_abs_error']:.4f}</td><td>{a['decile_mae']:.4f}</td></tr>
<tr><td>Brand-safe paced</td><td>${bs['spend']:.2f}</td>
<td>${bs['budget']:.2f}</td><td>{bs['wins']}</td>
<td>{bs['pred_win_rate']:.3f}</td><td>{bs['actual_win_rate']:.3f}</td>
<td>{bs['win_rate_abs_error']:.4f}</td><td>{bs['decile_mae']:.4f}</td></tr>
<tr><td>Ad-load-capped paced (cap {cap['adload_cap']}/bucket)</td>
<td>${cap['spend']:.2f}</td>
<td>${cap['budget']:.2f}</td><td>{cap['wins']}</td>
<td>{cap['pred_win_rate']:.3f}</td><td>{cap['actual_win_rate']:.3f}</td>
<td>{cap['win_rate_abs_error']:.4f}</td><td>{cap['decile_mae']:.4f}</td></tr></table>
<p>Brand safety blocked {bs['impressions_blocked_by_brand_safety']} impressions
(synthetic categories: political, firearms, tobacco). The ad-load cap
suppressed {cap['wins_suppressed_by_adload_cap']} would-be wins.</p>
<p><img src="win_rate_forecast.svg" alt="Win-rate forecast vs simulated"></p>
<p><img src="pacing.svg" alt="Pacing trajectory"></p>
<h2>Reserve-price sweep (unconstrained auction)</h2>
<table><tr><th>Floor ($)</th><th>Revenue ($)</th><th>Wins</th>
<th>Avg clearing price ($)</th></tr>
{''.join(f"<tr><td>{r['reserve_price']:.2f}</td><td>{r['spend']:.2f}</td><td>{r['wins']}</td><td>{r['avg_clearing_price']:.3f}</td></tr>" for r in sweep['floors'])}</table>
<p>Best floor ${sweep['best_reserve_price']:.2f} at revenue
${sweep['best_spend']:.2f} vs ${sweep['no_floor_spend']:.2f} with no floor.</p>
<p><img src="reserve_sweep.svg" alt="Reserve-price sweep"></p>
<h2>Honest limits</h2>
<ul><li>The bid landscape is a parametric LogNormal model, not measured
from real auctions; forecast error is measured against Monte Carlo
simulation under that model.</li>
<li>Advertiser categories are synthetic (seeded RNG, plausible
shares); the Criteo sample has no category column. The brand-safety
filter's mechanics are real, the labels are not measured.</li>
<li>Ad-load cap (35% of a bucket's impressions) and reserve-price
floors are policy choices applied to the simulator, not production
measurements; spend and clearing prices are simulated dollars.</li>
<li>Forward-in-time split (rows are chronologically ordered, train is
the first 80k rows, test the next 20k); no timestamp column exists, so
the split is by row order only.</li>
<li>Single 80k/20k split of the 100k-row sample; results on the full
45.8M-row Criteo train set are NOT claimed (GPU / bigger-machine work).</li>
<li>Hashed categoricals carry no semantics; rare levels are folded into
OTHER for both models (top-50 linear, top-254 GBM).</li></ul>
</body></html>
"""
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    args = ap.parse_args()
    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)
    os.makedirs("results", exist_ok=True)
    roc_svg(results, "results/roc_comparison.svg")
    reliability_svg(results, "results/reliability.svg")
    win_rate_svg(results, "results/win_rate_forecast.svg")
    pacing_svg(results, "results/pacing.svg")
    reserve_svg(results, "results/reserve_sweep.svg")
    report_html(results, "results/report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
