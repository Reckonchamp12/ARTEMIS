# Data

ARTEMIS does not ship with any datasets. All benchmarks use publicly
available data from Kaggle or other open repositories. Follow the
instructions below to download and prepare each dataset.

---

## 1. Optiver Realized Volatility

**Competition:** [Optiver Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction)

```bash
kaggle competitions download -c optiver-realized-volatility-prediction
unzip optiver-realized-volatility-prediction.zip -d data/optiver/
```

Expected files after download:
- `data/optiver/book_train.parquet`
- `data/optiver/trade_train.parquet`
- `data/optiver/train.csv`

The benchmark loads these directly; no manual pre-processing needed.

---

## 2. Crypto LOB (Synthetic / Real-World)

We use the **Crypto Limit Order Book** dataset referenced in the notebook.

If you have the raw snapshot CSV:
```bash
mkdir -p data/crypto_lob/
python -c "
import numpy as np, pandas as pd
df = pd.read_csv('crypto_lob_raw.csv')
# columns: bid_px_1..10, bid_sz_1..10, ask_px_1..10, ask_sz_1..10, mid_price
features = df.iloc[:, :-1].values.astype('float32')
mid_px   = df['mid_price'].values.astype('float64')
np.save('data/crypto_lob/lob_features.npy', features)
np.save('data/crypto_lob/mid_prices.npy',   mid_px)
"
```

---

## 3. Time-IMM EPA-Air Quality

**Source:** [US EPA Air Quality System (AQS)](https://www.epa.gov/aqs)
or [Kaggle EPA dataset](https://www.kaggle.com/datasets/sogun3/uspollution)

```bash
kaggle datasets download -d sogun3/uspollution
unzip uspollution.zip -d data/timeimm_raw/
python scripts/prep_timeimm.py --raw data/timeimm_raw/ --out data/timeimm/
```

Expected output files:
- `data/timeimm/features.npy`   — shape (N, F) normalised
- `data/timeimm/temperature.npy` — shape (N,) raw °C

---

## 4. DSLOB (Synthetic LOB for Ablation)

**Source:** [DSLOB: A Synthetic Limit Order Book Dataset](https://arxiv.org/abs/2202.02521)

```bash
# Download from Zenodo (DOI: 10.5281/zenodo.6081090)
wget https://zenodo.org/record/6081090/files/DSLOB.zip
unzip DSLOB.zip -d data/dslob_raw/
python scripts/prep_dslob.py --raw data/dslob_raw/ --out data/dslob/
```

Expected output:
- `data/dslob/lob_snapshots.npy`  — crash-regime snapshots
- `data/dslob/mid_returns.npy`

---

## 5. Jane Street Market Prediction

**Competition:** [Jane Street Market Prediction](https://www.kaggle.com/competitions/jane-street-market-prediction)

```bash
kaggle competitions download -c jane-street-market-prediction
unzip jane-street-market-prediction.zip -d data/janestreet_raw/
python scripts/prep_janestreet.py --raw data/janestreet_raw/ --out data/janestreet/
```

Expected output:
- `data/janestreet/features.npy`  — shape (N, 130)
- `data/janestreet/resp.npy`      — shape (N,)
- `data/janestreet/weight.npy`    — shape (N,)

---

## Directory layout after preparation

```
data/
├── optiver/
│   ├── book_train.parquet
│   ├── trade_train.parquet
│   └── train.csv
├── crypto_lob/
│   ├── lob_features.npy
│   └── mid_prices.npy
├── timeimm/
│   ├── features.npy
│   └── temperature.npy
├── dslob/
│   ├── lob_snapshots.npy
│   └── mid_returns.npy
└── janestreet/
    ├── features.npy
    ├── resp.npy
    └── weight.npy
```
