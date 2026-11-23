import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adtech_ctr_auction.data import CAT_COLS, INT_COLS  # noqa: E402


@pytest.fixture
def small_frame():
    rng = np.random.default_rng(0)
    n = 200
    data = {"label": rng.integers(0, 2, size=n)}
    for c in INT_COLS:
        vals = rng.normal(0, 10, size=n)
        vals[rng.random(n) < 0.1] = np.nan  # missing, as in the raw file
        data[c] = vals
    for c in CAT_COLS:
        cats = rng.choice(["a1b2c3d4", "e5f6a7b8", "c9d0e1f2", None], size=n)
        data[c] = cats
    return pd.DataFrame(data)
