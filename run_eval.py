#!/usr/bin/env python3
"""Run the full evaluation and write results.json.

Trains logistic regression + HistGradientBoosting on an 80k-row
subsample of the Criteo 100k-row sample, evaluates on a 20k-row held-out
split, and runs the second-price auction simulator (unconstrained and
paced). CPU-only; the GBM fit takes a few minutes.

Usage:
    python run_eval.py [--train-rows 80000] [--test-rows 20000]
                       [--data data/raw/dac_sample.txt] [--gbm-max-iter 300]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from adtech_ctr_auction.eval import run_eval  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-rows", type=int, default=80_000)
    ap.add_argument("--test-rows", type=int, default=20_000)
    ap.add_argument("--data", default="data/raw/dac_sample.txt")
    ap.add_argument("--gbm-max-iter", type=int, default=300)
    ap.add_argument("--results", default="results.json")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print(f"data file not found: {args.data}")
        print("run: python scripts/download_data.py")
        return 1

    t0 = time.time()
    results = run_eval(
        train_rows=args.train_rows,
        test_rows=args.test_rows,
        data_path=args.data,
        gbm_max_iter=args.gbm_max_iter,
    )
    results["runtime_seconds"] = round(time.time() - t0, 1)
    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    d = results["data"]
    print(f"data: {d['n_rows']} rows loaded, CTR {d['ctr']:.4f} "
          f"(train {d['train_rows']} / test {d['test_rows']})")
    for name in ("logistic_regression", "gbm"):
        m = results[name]
        print(f"{name}: AUC {m['auc']:.4f}, logloss {m['logloss']:.4f}, "
              f"Brier {m['brier']:.4f}, ECE {m['ece']:.4f}")
    for key in ("auction_unconstrained", "auction_paced"):
        a = results[key]
        print(f"{key}: spend ${a['spend']:.2f} of ${a['budget']:.2f}, "
              f"wins {a['wins']}, pred win rate {a['pred_win_rate']:.3f}, "
              f"actual {a['actual_win_rate']:.3f}, "
              f"abs err {a['win_rate_abs_error']:.4f}, "
              f"decile MAE {a['decile_mae']:.4f}")
    bs = results["auction_brand_safe"]
    print(f"auction_brand_safe: spend ${bs['spend']:.2f}, wins {bs['wins']}, "
          f"blocked {bs['impressions_blocked_by_brand_safety']} impressions")
    cap = results["auction_adload_capped"]
    print(f"auction_adload_capped: spend ${cap['spend']:.2f}, wins {cap['wins']}, "
          f"suppressed {cap['wins_suppressed_by_adload_cap']} wins "
          f"(cap {cap['adload_cap']}/bucket)")
    sw = results["auction_reserve_sweep"]
    print(f"auction_reserve_sweep: best floor ${sw['best_reserve_price']:.2f}, "
          f"revenue ${sw['best_spend']:.2f} vs ${sw['no_floor_spend']:.2f} no floor")
    print(f"wrote {args.results} in {results['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
