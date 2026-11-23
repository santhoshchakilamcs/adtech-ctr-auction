"""Data loading for the Criteo Display Advertising Challenge sample.

Source: the public Criteo display-ads data (Kaggle 2014 challenge),
via the Microsoft Recommenders sample ``dac_sample.tar.gz`` hosted on
Hugging Face (``Recommenders/criteo``). The archive is a 100,000-row
subsample of the full 45.8M-row ``train.txt`` — note: the file is
100k rows, not 1M; the "1M-row" label sometimes attached to this
sample is wrong, and this repo reports the measured 100,000. The file
is tab-separated with no header: column 0 is the click label,
columns 1-13 are integer features (I1..I13), columns 14-39 are hashed
categorical features (C1..C26, 32-bit hex strings, empty = missing).
Per the sample's own readme, rows are chronologically ordered, so the
evaluation uses a forward-in-time split (first 80k train, next 20k
test), not a random split.

The full file is downloaded at build time into ``data/raw/`` (not
committed). Only a 5,000-row sample is committed under ``data/``.
"""

from __future__ import annotations

import os
import tarfile
import urllib.request

# Microsoft Recommenders' 100k-row Criteo sample (8.79 MB tar.gz).
SAMPLE_URL = (
    "https://huggingface.co/datasets/Recommenders/criteo/"
    "resolve/main/dac_sample.tar.gz"
)

INT_COLS = [f"I{i}" for i in range(1, 14)]
CAT_COLS = [f"C{i}" for i in range(1, 27)]
COLUMNS = ["label"] + INT_COLS + CAT_COLS
N_COLUMNS = 40

DEFAULT_TRAIN_ROWS = 80_000
DEFAULT_TEST_ROWS = 20_000


def raw_path(data_dir: str) -> str:
    return os.path.join(data_dir, "raw", "dac_sample.txt")


def download_sample(data_dir: str, url: str = SAMPLE_URL) -> str:
    """Download dac_sample.tar.gz and extract dac_sample.txt into data/raw/.

    Returns the path of the extracted file. Idempotent: skips the
    download when dac_sample.txt already exists.
    """
    dest = raw_path(data_dir)
    if os.path.exists(dest):
        return dest
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    archive = os.path.join(raw_dir, "dac_sample.tar.gz")
    if not os.path.exists(archive):
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, archive)
        print(f"  saved {archive} ({os.path.getsize(archive) / 1e6:.1f} MB)")
    print("extracting dac_sample.txt ...")
    with tarfile.open(archive, "r:gz") as tf:
        # The archive contains dac_sample.txt at its top level.
        member = next(
            (m for m in tf.getmembers()
             if os.path.basename(m.name) == "dac_sample.txt"),
            None,
        )
        if member is None:
            raise ValueError(f"dac_sample.txt not found inside {archive}")
        tf.extract(member, raw_dir, filter="data")
    print(f"  extracted {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
    return dest


def load_rows(path: str, n_rows: int | None = None, skip_rows: int = 0):
    """Load the first ``n_rows`` rows (after ``skip_rows``) of train.txt.

    Returns a pandas DataFrame with columns label, I1..I13, C1..C26.
    Categorical columns are read as strings; empty fields stay NaN
    (missing) exactly as in the raw file. Integer columns are read as
    float so that their missing values survive as NaN.
    """
    import pandas as pd

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=COLUMNS,
        nrows=n_rows,
        skiprows=skip_rows,
        dtype={c: str for c in CAT_COLS},
        na_values=[""],
        keep_default_na=True,
        low_memory=False,
    )
    if df.shape[1] != N_COLUMNS:
        raise ValueError(f"expected {N_COLUMNS} columns, got {df.shape[1]}")
    df["label"] = df["label"].astype(int)
    for c in INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def describe(df) -> dict:
    """Basic facts about a loaded frame (CTR, missingness)."""
    import numpy as np

    n = len(df)
    ctr = float(df["label"].mean())
    int_missing = {c: float(df[c].isna().mean()) for c in INT_COLS}
    cat_missing = {c: float(df[c].isna().mean()) for c in CAT_COLS}
    return {
        "n_rows": n,
        "n_clicks": int(df["label"].sum()),
        "ctr": ctr,
        "int_missing_mean": float(np.mean(list(int_missing.values()))),
        "int_missing_max": float(max(int_missing.values())),
        "cat_missing_mean": float(np.mean(list(cat_missing.values()))),
        "cat_missing_max": float(max(cat_missing.values())),
        "int_missing": int_missing,
        "cat_missing": cat_missing,
    }
