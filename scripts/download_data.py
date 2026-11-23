#!/usr/bin/env python3
"""Download the Criteo 100k-row sample (Microsoft Recommenders
dac_sample.tar.gz) into data/raw/dac_sample.txt, and write a 3,000-row
committed sample to data/criteo_sample_3000.tsv.

Usage: python scripts/download_data.py [--data-dir data]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adtech_ctr_auction.data import describe, download_sample, load_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    path = download_sample(args.data_dir)
    df = load_rows(path, n_rows=100_000)
    stats = describe(df)
    print(f"rows: {stats['n_rows']}, clicks: {stats['n_clicks']}, "
          f"CTR: {stats['ctr']:.4f}")
    print(f"int missing (mean): {stats['int_missing_mean']:.3f}, "
          f"max: {stats['int_missing_max']:.3f}")
    print(f"cat missing (mean): {stats['cat_missing_mean']:.3f}, "
          f"max: {stats['cat_missing_max']:.3f}")

    sample_path = os.path.join(args.data_dir, "criteo_sample_3000.tsv")
    head = load_rows(path, n_rows=3000)
    head.to_csv(sample_path, sep="\t", header=False, index=False)
    size_kb = os.path.getsize(sample_path) / 1024
    print(f"committed sample: {sample_path} ({size_kb:.0f} KB, "
          f"{len(head)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
