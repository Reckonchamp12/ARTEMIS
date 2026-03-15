"""
Time-IMM EPA-Air benchmark — next-hour temperature forecasting.

Dataset : EPA Air Quality hourly measurements (temperature, humidity,
          wind speed, pressure, pollutant concentrations).
Task    : Predict next-hour ambient temperature (regression).
Split   : Chronological 70 / 15 / 15 train / val / test.

Each model sees a 24-hour look-back window of all sensor channels
and must output a single scalar prediction for t+1.
"""

import argparse
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from benchmarks.baselines import (
    LSTMModel,
    TransformerModel,
    NSTransformerModel,
    InformerModel,
    Chronos2Wrapper,
    XGBoostModel,
)
from benchmarks.metrics import rmse, mae, rank_ic, weighted_r2, print_summary_table
from artemis.model import ARTEMIS
from artemis.losses import artemis_loss

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
CFG = {
    "seq_len":    24,    # 24-hour look-back
    "d_model":    64,
    "n_heads":    4,
    "n_layers":   2,
    "dropout":    0.1,
    "batch_size": 256,
    "epochs":     40,
    "lr":         3e-4,
    "patience":   6,
    "device":     "cuda" if torch.cuda.is_available() else "cpu",
}


# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------

def sliding_window(series: np.ndarray, targets: np.ndarray, seq_len: int):
    """Convert a multivariate time series into (X, y) window pairs."""
    X, y = [], []
    for i in range(seq_len, len(series)):
        X.append(series[i - seq_len : i])
        y.append(targets[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_timeimm_data(data_path: str):
    """
    Load EPA-Air arrays.

    Expected files inside `data_path`:
        features.npy  — shape (N, F)  float32, z-score normalised
        temperature.npy — shape (N,)  float32, raw °C values

    Returns (train_ds, val_ds, test_ds, y_mean, y_std) so RMSE
    can be reported in original °C units.
    """
    features    = np.load(f"{data_path}/features.npy")
    temperature = np.load(f"{data_path}/temperature.npy")

    # normalise targets for stable training; we'll invert later
    y_mean = float(temperature.mean())
    y_std  = float(temperature.std()) + 1e-8
    temperature_norm = (temperature - y_mean) / y_std

    X, y = sliding_window(features, temperature_norm, CFG["seq_len"])

    n = len(X)
    t1, t2 = int(n * 0.70), int(n * 0.85)

    def to_ds(xi, yi):
        return TensorDataset(
            torch.from_numpy(xi),
            torch.from_numpy(yi).unsqueeze(-1),
        )

    tr = to_ds(X[:t1],   y[:t1])
    va = to_ds(X[t1:t2], y[t1:t2])
    te = to_ds(X[t2:],   y[t2:])
    return tr, va, te, y_mean, y_std


# ------------------------------------------------------------------
# Training utilities
# ------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, y_mean, y_std,
                use_artemis_loss=False):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        if use_artemis_loss:
            loss = artemis_loss(pred, yb, model, xb)
        else:
            loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def predict(model, loader, device, y_mean, y_std):
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb).squeeze(-1).cpu().numpy()
        preds.append(out)
        targets.append(yb.squeeze(-1).numpy())
    preds   = np.concatenate(preds)   * y_std + y_mean   # invert normalisation
    targets = np.concatenate(targets) * y_std + y_mean
    return preds, targets


def train_neural(model, train_ds, val_ds, cfg, y_mean, y_std,
                 use_artemis_loss=False):
    device = cfg["device"]
    model  = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_val, best_state, no_improve = np.inf, None, 0

    for epoch in range(cfg["epochs"]):
        train_epoch(model, train_loader, optimizer, device, y_mean, y_std,
                    use_artemis_loss=use_artemis_loss)
        scheduler.step()

        val_preds, val_targets = predict(model, val_loader, device, y_mean, y_std)
        val_rmse = rmse(val_targets, val_preds)

        if val_rmse < best_val:
            best_val  = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    return model


# ------------------------------------------------------------------
# Main benchmark
# ------------------------------------------------------------------

def run_benchmark(data_path: str):
    print("\n" + "=" * 60)
    print("  Time-IMM EPA-Air — Next-Hour Temperature Benchmark")
    print("=" * 60)

    train_ds, val_ds, test_ds, y_mean, y_std = load_timeimm_data(data_path)
    n_feat = train_ds.tensors[0].shape[-1]

    print(f"Samples — Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print(f"Features: {n_feat}   Seq len: {CFG['seq_len']}h")

    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
    d = CFG["d_model"]

    # XGBoost — flattened sequences
    X_train_np = train_ds.tensors[0].numpy().reshape(len(train_ds), -1)
    y_train_np = (train_ds.tensors[1].numpy().flatten() * y_std + y_mean)
    X_test_np  = test_ds.tensors[0].numpy().reshape(len(test_ds), -1)
    y_test_np  = (test_ds.tensors[1].numpy().flatten() * y_std + y_mean)

    results = {}

    # --- XGBoost ---
    print("\n[1/7] XGBoost")
    t0 = time.time()
    xgb = XGBoostModel(task="regression")
    xgb.fit(X_train_np, y_train_np)
    xgb_preds = xgb.predict(X_test_np)
    results["XGBoost"] = {
        "rmse":    rmse(y_test_np, xgb_preds),
        "mae":     mae(y_test_np, xgb_preds),
        "rank_ic": rank_ic(y_test_np, xgb_preds),
        "wr2":     weighted_r2(y_test_np, xgb_preds),
    }
    print(f"  done in {time.time()-t0:.1f}s  RMSE={results['XGBoost']['rmse']:.4f}")

    # --- neural models ---
    neural_cfgs = [
        ("LSTM",           LSTMModel(n_feat, d, n_layers=2, dropout=0.1, output_dim=1), False),
        ("Transformer",    TransformerModel(n_feat, d, n_heads=4, n_layers=2, seq_len=CFG["seq_len"], output_dim=1), False),
        ("NS-Transformer", NSTransformerModel(n_feat, d, n_heads=4, n_layers=2, seq_len=CFG["seq_len"], output_dim=1), False),
        ("Informer",       InformerModel(n_feat, d, n_heads=4, n_layers=2, seq_len=CFG["seq_len"], output_dim=1), False),
        ("Chronos-2",      Chronos2Wrapper(n_feat, d, seq_len=CFG["seq_len"], output_dim=1), False),
        ("ARTEMIS",        ARTEMIS(input_dim=n_feat, d_model=d, n_heads=4, n_layers=2, seq_len=CFG["seq_len"], output_dim=1), True),
    ]

    for i, (name, model, use_al) in enumerate(neural_cfgs, start=2):
        print(f"\n[{i}/7] {name}")
        t0 = time.time()
        try:
            trained = train_neural(model, train_ds, val_ds, CFG, y_mean, y_std,
                                   use_artemis_loss=use_al)
            preds, targets = predict(trained, test_loader, CFG["device"], y_mean, y_std)
            results[name] = {
                "rmse":    rmse(targets, preds),
                "mae":     mae(targets, preds),
                "rank_ic": rank_ic(targets, preds),
                "wr2":     weighted_r2(targets, preds),
            }
            print(f"  done in {time.time()-t0:.1f}s  RMSE={results[name]['rmse']:.4f}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results[name] = {"rmse": float("nan"), "mae": float("nan"),
                             "rank_ic": float("nan"), "wr2": float("nan")}

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Results — Time-IMM EPA-Air (Test Set, units: °C)")
    print("=" * 60)
    rows = []
    for name, m in results.items():
        rows.append({
            "Model":   name,
            "RMSE":    f"{m['rmse']:.4f}",
            "MAE":     f"{m['mae']:.4f}",
            "RankIC":  f"{m['rank_ic']:.4f}",
            "WR²":     f"{m['wr2']:.4f}",
        })
    print_summary_table(rows)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time-IMM EPA-Air benchmark")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to directory with features.npy and temperature.npy")
    args = parser.parse_args()
    run_benchmark(args.data_path)
