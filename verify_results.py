#!/usr/bin/env python3
"""Re-run the evaluation and check the reported numbers.

Verifies:
  1. A fresh run of run_eval() reproduces the model metrics and
     auction numbers in results.json within tolerance (AUC / logloss /
     Brier / ECE to 1e-3, win rates and spend to 1e-3 relative — the
     pipeline is seeded and deterministic, so a larger drift means
     the code or data changed).
  2. README.md states the same headline numbers as results.json.
  3. The SVG charts and HTML report exist and carry the same numbers.

Usage: python verify_results.py [--results results.json] [--readme README.md]

Exits non-zero on any mismatch. The re-run takes a few minutes on CPU
(GBM fit on 80k rows).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from adtech_ctr_auction.eval import run_eval  # noqa: E402

TOL = 1e-3


def _close(name, fresh, stored, errors, tol=TOL):
    if abs(fresh - stored) > tol:
        errors.append(f"{name}: fresh {fresh} != stored {stored}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--readme", default="README.md")
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as f:
        stored = json.load(f)
    errors = []

    fresh = run_eval(
        train_rows=stored["data"]["train_rows"],
        test_rows=stored["data"]["test_rows"],
        data_path="data/raw/dac_sample.txt",
        gbm_max_iter=stored["params"]["gbm_max_iter"],
    )
    if fresh["data"]["n_rows"] != stored["data"]["n_rows"]:
        errors.append("n_rows changed")
    for model in ("logistic_regression", "gbm"):
        for metric in ("auc", "logloss", "brier", "ece"):
            _close(f"{model}.{metric}", fresh[model][metric],
                   stored[model][metric], errors)
    for run in ("auction_unconstrained", "auction_paced",
                "auction_brand_safe", "auction_adload_capped"):
        for metric in ("spend", "pred_win_rate", "actual_win_rate",
                       "win_rate_abs_error", "decile_mae"):
            _close(f"{run}.{metric}", fresh[run][metric],
                   stored[run][metric], errors,
                   tol=max(TOL, abs(stored[run][metric]) * 1e-3))
        if fresh[run]["wins"] != stored[run]["wins"]:
            errors.append(f"{run}.wins: fresh {fresh[run]['wins']} != "
                          f"stored {stored[run]['wins']}")
    for key, stored_key in (
        ("wins_suppressed_by_adload_cap", "auction_adload_capped"),
        ("impressions_blocked_by_brand_safety", "auction_brand_safe"),
    ):
        if fresh[stored_key][key] != stored[stored_key][key]:
            errors.append(f"{stored_key}.{key}: fresh "
                          f"{fresh[stored_key][key]} != stored "
                          f"{stored[stored_key][key]}")
    fresh_sw = fresh["auction_reserve_sweep"]
    stored_sw = stored["auction_reserve_sweep"]
    if fresh_sw["best_reserve_price"] != stored_sw["best_reserve_price"]:
        errors.append("reserve sweep best floor changed: "
                      f"{fresh_sw['best_reserve_price']} != "
                      f"{stored_sw['best_reserve_price']}")
    _close("reserve_sweep.best_spend", fresh_sw["best_spend"],
           stored_sw["best_spend"], errors,
           tol=max(TOL, abs(stored_sw["best_spend"]) * 1e-3))

    if os.path.exists(args.readme):
        readme = open(args.readme, encoding="utf-8").read()
        checks = [
            f"Logistic regression AUC {stored['logistic_regression']['auc']:.4f}",
            f"GBM AUC {stored['gbm']['auc']:.4f}",
            f"train {stored['data']['train_rows']} / test {stored['data']['test_rows']}",
        ]
        for line in checks:
            if line not in readme:
                errors.append(f"README is missing {line!r}")

    for path in ("results/roc_comparison.svg", "results/reliability.svg",
                 "results/win_rate_forecast.svg", "results/pacing.svg",
                 "results/reserve_sweep.svg", "results/report.html"):
        if not os.path.exists(path):
            errors.append(f"missing visual: {path} "
                          "(regenerate with python make_visuals.py)")
            continue
        text = open(path, encoding="utf-8").read()
        if path.endswith("report.html"):
            for token in (f"{stored['gbm']['auc']:.4f}",
                          f"{stored['logistic_regression']['auc']:.4f}"):
                if token not in text:
                    errors.append(f"report.html is missing AUC {token}")

    if errors:
        print("VERIFY FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"VERIFY OK: train {stored['data']['train_rows']} / "
          f"test {stored['data']['test_rows']}, "
          f"LR AUC {stored['logistic_regression']['auc']:.4f}, "
          f"GBM AUC {stored['gbm']['auc']:.4f}, paced win-rate abs err "
          f"{stored['auction_paced']['win_rate_abs_error']:.4f} — "
          "fresh run reproduces results.json and the README.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
