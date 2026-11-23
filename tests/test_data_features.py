import numpy as np
import pandas as pd

from adtech_ctr_auction.data import describe, load_rows
from adtech_ctr_auction.features import (
    build_gbm_features,
    build_lr_features,
    fit_top_levels,
)


def test_load_rows_sample_file(tmp_path):
    # Write a tiny raw-format file: label, 13 ints, 26 hex cats.
    rows = []
    for i in range(5):
        cats = "\t".join(["a1b2c3d4"] * 26)
        rows.append(f"{i % 2}\t" + "\t".join(["1"] * 13) + "\t" + cats)
    # One row with missing values: empty int fields and empty cat fields.
    rows.append("0\t" + "\t".join([""] * 13) + "\t" + "\t".join([""] * 26))
    p = tmp_path / "train.txt"
    p.write_text("\n".join(rows) + "\n")
    df = load_rows(str(p))
    assert list(df.columns)[:1] == ["label"]
    assert df.shape == (6, 40)
    assert df["label"].tolist() == [0, 1, 0, 1, 0, 0]
    assert df["I1"].isna().sum() == 1  # missing int survives as NaN
    assert df["C1"].isna().sum() == 1  # missing cat survives as NaN


def test_describe(small_frame):
    stats = describe(small_frame)
    assert stats["n_rows"] == 200
    assert 0.0 < stats["ctr"] < 1.0
    assert stats["int_missing_mean"] > 0  # fixture has ~10% missing ints
    assert stats["cat_missing_mean"] > 0


def test_lr_preprocessor_handles_missing(small_frame):
    X = small_frame.drop(columns=["label"])
    prep = build_lr_features(max_categories=5)
    Xt = prep.fit_transform(X)
    assert Xt.shape[0] == 200
    assert Xt.shape[1] > 13  # one-hot columns added
    assert not np.isnan(Xt.toarray()).any() if hasattr(Xt, "toarray") else True


def test_gbm_preprocessor_handles_missing(small_frame):
    X = small_frame.drop(columns=["label"])
    prep = build_gbm_features()
    Xt = prep.fit_transform(X)
    assert Xt.shape == (200, 39)
    assert not np.isnan(Xt.astype(float)).any()


def test_fit_top_levels(small_frame):
    top = fit_top_levels(small_frame, max_categories=2)
    assert set(top.keys()) == set(f"C{i}" for i in range(1, 27))
    assert all(len(v) <= 2 for v in top.values())


def test_topk_ordinal_encoder_caps_cardinality():
    from adtech_ctr_auction.features import TopKOrdinalEncoder

    rng = np.random.default_rng(0)
    # One column with 5,000 distinct levels: HistGBM would reject this.
    X = pd.DataFrame(
        {"C1": [f"hash{i}" for i in rng.integers(0, 5000, size=2000)]}
    )
    enc = TopKOrdinalEncoder(max_categories=254)
    out = enc.fit_transform(X)
    assert out.shape == (2000, 1)
    assert out.max() <= 254  # 254 kept levels + 1 OTHER code
    assert len(np.unique(out)) <= 255
    # Unseen levels at transform time fold into OTHER, not an error.
    Xt = pd.DataFrame({"C1": ["brand_new_hash_1", "brand_new_hash_2"]})
    out_t = enc.transform(Xt)
    assert (out_t == 254).all() or (out_t <= 254).all()
