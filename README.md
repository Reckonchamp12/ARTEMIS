# ARTEMIS: Adaptive Real-Time Market Intelligence System

A hybrid deep learning architecture for financial time-series forecasting that combines neural SDEs, Mamba-style state space models, and physics-informed constraints.

---

## Overview

ARTEMIS is benchmarked against 6 baselines across multiple financial datasets:

| Model | Type | Description |
|---|---|---|
| **ARTEMIS** | Hybrid SDE + Attention | Encoder → SDE → Prediction head with PDE/MPR losses |
| LSTM | Recurrent | Bidirectional LSTM baseline |
| Transformer | Attention | Vanilla encoder-only transformer |
| NS-Transformer | Attention | Non-stationary transformer with de-stationing normalisation |
| Informer | Sparse Attention | ProbSparse attention for long sequences |
| Chronos-2 | Foundation Model | Amazon Chronos-T5/Bolt zero-shot + fine-tune |
| XGBoost | Gradient Boosting | Tabular baseline with aggregated window features |

---

## Datasets

| Dataset | Task | Metric |
|---|---|---|
| **Optiver (LOB sequences)** | Realized volatility regression | RMSE, RankIC, Directional Acc, Weighted R² |
| **Crypto LOB (ADA/BTC/ETH)** | Mid-price direction (binary) | AUC-ROC, PR-AUC, F1, Directional Acc |
| **Time-IMM EPA-Air** | Next-hour temperature regression | RMSE, RankIC, Delta Dir Acc, Weighted R² |
| **Jane Street** | Return prediction (regression) | RMSE, RankIC, Directional Acc, Weighted R² |

> ℹ️ **Club Loan dataset benchmarks are excluded from this release.**

---

## Benchmark Results

### Optiver — Realized Volatility (Test Set)

| Model | RMSE ↓ | RankIC ↑ | Dir. Acc ↑ | Weighted R² ↑ |
|---|---|---|---|---|
| **XGBoost** | **0.2777** | **0.8489** | **0.8538** | **0.7447** |
| Transformer | 0.5422 | 0.3583 | 0.6162 | 0.0268 |
| ARTEMIS | 0.5553 | -0.0555 | 0.4582 | -0.0208 |
| LSTM | 0.5570 | 0.0000 | 0.0000 | -0.0271 |
| NS-Transformer | 0.7019 | 0.2474 | 0.6057 | -0.6308 |
| Chronos-2 | 4.9538 | -0.1384 | 0.4047 | -80.232 |
| Informer | 1.8411 | -0.1465 | 0.5679 | -10.220 |

### Crypto LOB — Mid-Price Direction (Test Set)

| Model | AUC-ROC ↑ | PR-AUC ↑ | Directional Acc ↑ |
|---|---|---|---|
| **XGBoost** | **0.5124** | **0.5178** | 0.5027 |
| LSTM | 0.5019 | 0.5127 | 0.5037 |
| Transformer | 0.5013 | 0.5050 | 0.5018 |
| ARTEMIS | 0.4951 | 0.5027 | 0.4976 |
| Chronos-2 | 0.5012 | 0.5061 | 0.5019 |
| NS-Transformer | 0.4956 | 0.5031 | 0.4942 |
| Informer | 0.4950 | 0.5062 | 0.4932 |

### Time-IMM EPA-Air — Next-Hour Temperature (Test Set)

| Model | RMSE ↓ | RankIC ↑ | Dir. Acc ↑ | Weighted R² ↑ |
|---|---|---|---|---|
| **XGBoost** | **3.437** | 0.946 | 0.905 | **0.898** |
| Informer | 4.011 | 0.928 | 0.890 | 0.861 |
| Transformer | 4.420 | **0.969** | **0.922** | 0.831 |
| ARTEMIS | 4.691 | 0.904 | 0.860 | 0.810 |
| LSTM | 19.580 | 0.493 | 0.533 | -2.314 |
| NS-Transformer | 40.469 | 0.257 | 0.599 | -13.158 |
| Chronos-2 | 79.255 | 0.943 | 0.907 | -53.302 |

---

## ARTEMIS Architecture

```
Input (N, T, D)
     │
     ▼
┌─────────────────────────────┐
│    Time Encoder             │
│  LOB features + mask +      │
│  Fourier time basis → Z     │
│  (N, T, L)                  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Neural SDE (Euler-Maruyama)│
│  dZ = μ(Z,t)dt + σ(Z,t)dW  │
│  μ: drift network           │
│  σ: diffusion network       │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Prediction Head            │
│  Z_sde[:, -1] → Linear → ŷ │
└─────────────────────────────┘

Loss = MSE + λ_PDE · L_PDE + λ_MPR · L_MPR + λ_cons · L_consistency
```

**Key components:**

- **Time Encoder**: Projects raw features + Fourier time basis into latent space Z ∈ ℝᴸ
- **Drift Network (μ)**: Learned deterministic component of the SDE
- **Diffusion Network (σ)**: Stochastic/volatility component; magnitude separates market regimes without supervision
- **Value Network**: Used in HJB PDE residual loss for physics consistency
- **PDE Loss**: Hamilton-Jacobi-Bellman residual via Hutchinson trace estimator
- **MPR Loss**: Market price of risk regularisation
- **Consistency Loss**: SDE path smoothness constraint

---

## Ablation Study Results (DSLOB dataset)

| Variant | RMSE ↓ | MAE ↓ | RankIC | Dir. Acc |
|---|---|---|---|---|
| A0 Full ARTEMIS | 0.2666 | 0.2513 | -0.0590 | 0.6489 |
| A1 No SDE | **0.0224** | **0.0199** | -0.0752 | 0.6459 |
| A2 No PDE loss | 0.0723 | 0.0576 | -0.0471 | 0.5032 |
| A3 No MPR loss | 0.0685 | 0.0555 | -0.0224 | 0.5682 |
| A4 No physics (MSE only) | 0.0399 | 0.0323 | +0.0306 | 0.4177 |
| A5 No consistency | 0.1529 | 0.1300 | -0.0557 | 0.3754 |
| A6 MLP baseline | 1.8491 | 1.8274 | +0.0090 | 0.3504 |

---

## Repository Structure

```
artemis-repo/
├── artemis/
│   ├── model.py            ARTEMIS full architecture
│   ├── losses.py           PDE, MPR, consistency losses
│   └── encoder.py          Time encoder with Fourier basis
├── benchmarks/
│   ├── run_optiver.py      Optiver realized volatility benchmark
│   ├── run_crypto_lob.py   Crypto LOB direction benchmark
│   ├── run_timeimm.py      Time-IMM EPA-Air benchmark
│   └── baselines.py        All 6 baseline models
├── configs/
│   ├── optiver.yaml        Optiver training config
│   ├── crypto_lob.yaml     Crypto LOB training config
│   └── timeimm.yaml        Time-IMM training config
├── data/
│   └── README.md           Dataset download instructions
├── results/
│   └── tables/             Pre-computed result tables (CSV)
├── scripts/
│   ├── train.py            Generic training script
│   └── evaluate.py         Evaluation script
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-org/artemis.git
cd artemis
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.x, XGBoost, scikit-learn, scipy, numpy, pandas, matplotlib

---

## Usage

### Train ARTEMIS on a dataset

```bash
python scripts/train.py --config configs/optiver.yaml --model artemis
```

### Run full benchmark (all 7 models)

```bash
python benchmarks/run_optiver.py --data_root /path/to/optiver/data
python benchmarks/run_crypto_lob.py --data_root /path/to/crypto/data
python benchmarks/run_timeimm.py --data_root /path/to/timeimm/data
```

### Reproduce ablation study

```bash
python benchmarks/run_optiver.py --ablation --model artemis
```

---

## Citation

```bibtex
@misc{artemis2025,
  title  = {ARTEMIS: Adaptive Real-Time Market Intelligence System},
  year   = {2025},
  note   = {GitHub repository}
}
```
