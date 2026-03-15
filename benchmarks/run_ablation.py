"""
Optiver Realized Volatility — 7-Model Benchmark
================================================
All seven models trained and evaluated under identical conditions on
the Optiver realized volatility prediction dataset.

Dataset layout expected at `data_path`:
    X_train.npy, X_val.npy, X_test.npy  — shape (N, seq_len, n_features)
    y_train.npy, y_val.npy, y_test.npy  — shape (N,)  log-volatility targets

These are generated from the raw Kaggle parquet files using standard
LOB feature engineering (wap, bid-ask spread, order imbalance, etc.)
See data/README.md for download and preparation instructions.

Usage:
    python benchmarks/run_optiver.py --data_path data/optiver/
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
    LSTMModel, TransformerModel, NSTransformerModel,
    InformerModel, Chronos2Wrapper, XGBoostModel,
)
from benchmarks.metrics import rmse, mae, rank_ic, weighted_r2, directional_accuracy, print_summary_table
from artemis.model import ARTEMIS
from artemis.losses import artemis_loss

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CFG = {
    "d_model":    64,
    "n_heads":    4,
    "n_layers":   2,
    "dropout":    0.1,
    "batch_size": 512,
    "epochs":     60,
    "lr":         3e-4,
    "patience":   8,
    "device":     "cuda" if torch.cuda.is_available() else "cpu",
    "lambda_pde":  0.10,
    "lambda_mpr":  0.05,
    "lambda_cons": 0.02,
}


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_optiver_data(data_path: str):
    """
    Load pre-processed Optiver arrays.

    Returns (train_ds, val_ds, test_ds) as TensorDatasets.
    """
    def load_split(split):
        X = np.load(f"{data_path}/X_{split}.npy").astype(np.float32)
        y = np.load(f"{data_path}/y_{split}.npy").astype(np.float32)
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    return load_split("train"), load_split("val"), load_split("test")


# ------------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, is_artemis=False):
    model.train()
    total, n = 0.0, 0
    criterion = nn.MSELoss()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        if is_artemis:
            loss = artemis_loss(pred, yb, model, xb,
                                lambda_pde=CFG["lambda_pde"],
                                lambda_mpr=CFG["lambda_mpr"],
                                lambda_cons=CFG["lambda_cons"])
        else:
            loss = criterion(pred.float(), yb.float())
        if not torch.isfinite(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(yb)
        n     += len(yb)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        out = model(xb.to(device)).cpu().numpy()
        preds.append(out)
        targets.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def train_neural(model, train_ds, val_ds, cfg, is_artemis=False):
    device = cfg["device"]
    model  = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = nn.MSELoss()

    best_val, best_state, no_improve = np.inf, None, 0

    for epoch in range(cfg["epochs"]):
        train_epoch(model, train_loader, optimizer, device, is_artemis)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred.float(), yb.float()).item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    return model


def compute_metrics(targets, preds):
    return {
        "rmse":    rmse(targets, preds),
        "mae":     mae(targets, preds),
        "rank_ic": rank_ic(targets, preds),
        "wr2":     weighted_r2(targets, preds),
        "dir_acc": directional_accuracy(targets, preds),
    }


# ------------------------------------------------------------------
# Main benchmark
# ------------------------------------------------------------------

def run_benchmark(data_path: str):
    print("\n" + "=" * 60)
    print("  Optiver Realized Volatility Benchmark")
    print("=" * 60)

    train_ds, val_ds, test_ds = load_optiver_data(data_path)
    n_feat  = train_ds.tensors[0].shape[-1]
    seq_len = train_ds.tensors[0].shape[1]
    d       = CFG["d_model"]

    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print(f"Features: {n_feat}   Seq len: {seq_len}")

    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=0)
    results = {}

    # XGBoost
    print("\n[1/7] XGBoost")
    t0 = time.time()
    X_tr = train_ds.tensors[0].numpy().reshape(len(train_ds), -1)
    y_tr = train_ds.tensors[1].numpy()
    X_te = test_ds.tensors[0].numpy().reshape(len(test_ds), -1)
    y_te = test_ds.tensors[1].numpy()

    xgb = XGBoostModel(task="regression", n_estimators=1000, learning_rate=0.02)
    xgb.fit(X_tr, y_tr)
    xgb_preds = xgb.predict(X_te)
    results["XGBoost"] = compute_metrics(y_te, xgb_preds)
    print(f"  done in {time.time()-t0:.1f}s  RMSE={results['XGBoost']['rmse']:.4f}")

    # Neural models
    neural_cfgs = [
        ("LSTM",           LSTMModel(n_feat, d, n_layers=2, dropout=0.1, output_dim=1), False),
        ("Transformer",    TransformerModel(n_feat, d, n_heads=4, n_layers=2, seq_len=seq_len, output_dim=1), False),
        ("NS-Transformer", NSTransformerModel(n_feat, d, n_heads=4, n_layers=2, seq_len=seq_len, output_dim=1), False),
        ("Informer",       InformerModel(n_feat, d, n_heads=4, n_layers=2, seq_len=seq_len, output_dim=1), False),
        ("Chronos-2",      Chronos2Wrapper(n_feat, d, n_heads=4, n_layers=2, seq_len=seq_len, output_dim=1), False),
        ("ARTEMIS",        ARTEMIS(input_dim=n_feat, d_model=d, n_heads=4, n_layers=2, seq_len=seq_len, output_dim=1), True),
    ]

    for i, (name, model, is_al) in enumerate(neural_cfgs, start=2):
        print(f"\n[{i}/7] {name}")
        t0 = time.time()
        try:
            trained = train_neural(model, train_ds, val_ds, CFG, is_artemis=is_al)
            preds, targets = evaluate(trained, test_loader, CFG["device"])
            results[name]  = compute_metrics(targets, preds)
            r = results[name]
            print(f"  done in {time.time()-t0:.1f}s  RMSE={r['rmse']:.4f}  RankIC={r['rank_ic']:.4f}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results[name] = {k: float("nan") for k in ["rmse","mae","rank_ic","wr2","dir_acc"]}

    # Print table
    print("\n" + "=" * 60)
    print("  Results — Optiver Realized Volatility (Test Set)")
    print("=" * 60)
    rows = []
    for name, m in results.items():
        rows.append({
            "Model":   name,
            "RMSE":    f"{m['rmse']:.4f}",
            "MAE":     f"{m['mae']:.4f}",
            "RankIC":  f"{m['rank_ic']:.4f}",
            "WR²":     f"{m['wr2']:.4f}",
            "DirAcc":  f"{m['dir_acc']:.4f}",
        })
    print_summary_table(rows)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optiver benchmark")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Directory with X_*.npy and y_*.npy arrays")
    args = parser.parse_args()
    run_benchmark(args.data_path)
