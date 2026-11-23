# Data

`criteo_sample_3000.tsv` — the first 3,000 rows of the Criteo Display
Advertising Challenge 100k-row sample (`dac_sample.tar.gz` from Microsoft
Recommenders, hosted on Hugging Face as `Recommenders/criteo`), tab-separated,
no header: label, I1..I13 (integer features), C1..C26 (hashed categorical
features, hex strings, empty = missing). Committed so the format and the
preprocessing can be inspected without downloading anything.

The full 100k-row file is downloaded at build time by
`python scripts/download_data.py` into `data/raw/dac_sample.txt` (gitignored,
~24 MB uncompressed). The evaluation loads all 100,000 rows of it
(80,000 train / 20,000 test, forward in time).
