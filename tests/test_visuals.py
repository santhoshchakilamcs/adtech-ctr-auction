import json
import os
import subprocess
import sys

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..")


def _fake_results():
    curve = [
        {"bin": b, "bin_lo": b / 10, "bin_hi": (b + 1) / 10, "n": 100,
         "mean_pred": (b + 0.5) / 10, "frac_pos": (b + 0.5) / 10}
        for b in range(10)
    ]
    roc = [[i / 100, i / 100] for i in range(101)]
    return {
        "data": {"n_rows": 1000, "ctr": 0.25, "train_rows": 800,
                 "test_rows": 200},
        "params": {"value_per_click_usd": 2.5},
        "logistic_regression": {"auc": 0.70, "logloss": 0.55, "brier": 0.16,
                                "ece": 0.01, "reliability_curve": curve,
                                "roc_curve": roc},
        "gbm": {"auc": 0.75, "logloss": 0.52, "brier": 0.15, "ece": 0.01,
                "reliability_curve": curve, "roc_curve": roc},
        "auction_landscape": {"mu": -0.5, "sigma": 0.8,
                              "median_competing_bid_usd": 0.6},
        "auction_unconstrained": {
            "spend": 100.0, "budget": 1e12, "wins": 50,
            "pred_win_rate": 0.5, "actual_win_rate": 0.5,
            "win_rate_abs_error": 0.01, "decile_mae": 0.02,
            "deciles": [{"decile": d, "n": 20, "mean_bid": 0.5,
                         "pred_win_rate": 0.5, "actual_win_rate": 0.5}
                        for d in range(10)],
            "trajectory": [{"bucket": b, "multiplier": 1.0, "spend": 4.0,
                            "wins": 2, "impressions": 8} for b in range(24)],
        },
        "auction_paced": {
            "spend": 50.0, "budget": 50.0, "wins": 25,
            "pred_win_rate": 0.4, "actual_win_rate": 0.4,
            "win_rate_abs_error": 0.01, "decile_mae": 0.02,
            "deciles": [{"decile": d, "n": 20, "mean_bid": 0.5,
                         "pred_win_rate": 0.4, "actual_win_rate": 0.4}
                        for d in range(10)],
            "trajectory": [{"bucket": b, "multiplier": 1.0, "spend": 2.0,
                            "wins": 1, "impressions": 8} for b in range(24)],
        },
    }


def test_make_visuals_from_fake_results(tmp_path):
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(_fake_results()))
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    # make_visuals writes into ./results relative to CWD; run it there.
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import make_visuals; make_visuals.main()" % REPO
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, "--results", str(results_path)],
        cwd=str(tmp_path), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    for name in ("roc_comparison.svg", "reliability.svg",
                 "win_rate_forecast.svg", "pacing.svg", "report.html"):
        path = out_dir / name
        assert path.exists(), f"missing {name}"
        assert path.stat().st_size > 0
    html = (out_dir / "report.html").read_text()
    assert "0.7500" in html  # GBM AUC appears in the report
