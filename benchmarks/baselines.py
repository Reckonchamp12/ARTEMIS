"""
Jane Street Market Prediction benchmark.

Dataset : Jane Street competition features (anonymised financial signals).
Task    : Predict resp (forward return proxy), binary action label
          derived as action = 1 if resp > 0 else 0.
Split   : Chronological 70 / 15 / 15 train / val / test.

Metrics reported: RMSE on resp, RankIC, Weighted R², Directional Accuracy.
This matches the evaluation used in the original Jane Street Kaggle competition.
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
from benchmarks.metrics import (
    rmse, mae, rank_ic, weighted_r2, directional_accuracy,
    print_summary_table,
)
from artemis.model import ARTEMIS
from artemis.losses import artemis_loss

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CFG = {
    "seq_len":    20,    # 20-step feature look-back
    "d_model":    128,
    "n_heads":    8,
    "n_layers":   3,
    "dropout":    0.15,
    "batch_size": 512,
    "epochs":     50,
    "lr":         2e-4,
    "patience":   7,
    "device":     "cuda" if torch.cuda.is_available() else "cpu",
    # ARTEMIS physics weights (tuned on validation set)
    "lambda_pde":  0.05,
    "lambda_mpr":  0.02,
    "lambda_cons": 0.01,
}


# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------

def load_jane_street_data(data_path: str):
    """
    Load pre-processed Jane Street arrays.

    Expected files:
        features.npy    — shape (N, 130)  float32, filled + normalised
        resp.npy        — shape (N,)      float32  forward return target
        weight.npy      — shape (N,)      float32  per-row utility weight

    Builds sliding windows of length seq_len and returns TensorDatasets.
    """
    features = np.load(f"{data_path}/features.npy")
    resp     = np.load(f"{data_path}/resp.npy")
    weight   = np.load(f"{data_path}/weight.npy")

    seq_len = CFG["seq_len"]
    X, y, w = [], [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len : i])
        y.append(resp[i])
        w.append(weight[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    w = np.array(w, dtype=np.float32)

    n = len(X)
    t1, t2 = int(n * 0.70), int(n * 0.85)

    def to_ds(xi, yi, wi):
        return TensorDataset(
            torch.from_numpy(xi),
            torch.from_numpy(yi).unsqueeze(-1),
            torch.from_numpy(wi),
        )

    return (
        to_ds(X[:t1],   y[:t1],   w[:t1]),
        to_ds(X[t1:t2], y[t1:t2], w[t1:t2]),
        to_ds(X[t2:],   y[t2:],   w[t2:]),
        y,   # full target array (for global std normalisation)
    )


# ------------------------------------------------------------------
# Training / evaluation
# ------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, use_artemis=False):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    for xb, yb, _ in loader:          # weight not used during training
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        if use_artemis:
            loss = artemis_loss(pred, yb, model, xb,
                                lambda_pde=CFG["lambda_pde"],
                                lambda_mpr=CFG["lambda_mpr"],
                                lambda_cons=CFG["lambda_cons"])
        else:
            loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for xb, yb, _ in loader:
        xb = xb.to(device)
        out = model(xb).squeeze(-1).cpu().numpy()
        preds.append(out)
        targets.append(yb.squeeze(-1).numpy())
    return np.concatenate(preds), np.concatenate(targets)


def train_neural(model, train_ds, val_ds, cfg, use_artemis=False):
    device = cfg["device"]
    model  = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"],
        steps_per_epoch=len(train_loader),
        epochs=cfg["epochs"],
    )

    best_rmse, best_state, no_improve = np.inf, None, 0

    for epoch in range(cfg["epochs"]):
        train_epoch(model, train_loader, optimizer, device, use_artemis=use_artemis)
        scheduler.step()

        val_preds, val_targets = evaluate(model, val_loader, device)
        val_rmse = rmse(val_targets, val_preds)

        if val_rmse < best_rmse:
            best_rmse  = val_rmse
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
    print("  Jane Street — Forward Return Prediction Benchmark")
    print("=" * 60)

    train_ds, val_ds, test_ds, all_resp = load_jane_street_data(data_path)
    n_feat  = train_ds.tensors[0].shape[-1]
    seq_len = CFG["seq_len"]
    d       = CFG["d_model"]

    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print(f"Features: {n_feat}   Seq len: {seq_len}")

    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=0)
    results = {}

    # XGBoost
    print("\n[1/7] XGBoost")
    t0 = time.time()
    X_tr = train_ds.tensors[0].numpy().reshape(len(train_ds), -1)
    y_tr = train_ds.tensors[1].numpy().flatten()
    X_te = test_ds.tensors[0].numpy().reshape(len(test_ds), -1)
    y_te = test_ds.tensors[1].numpy().flatten()

    xgb = XGBoostModel(task="regression")
    xgb.fit(X_tr, y_tr)
    xgb_preds = xgb.predict(X_te)
    results["XGBoost"] = compute_metrics(y_te, xgb_preds)
    print(f"  done in {time.time()-t0:.1f}s  RMSE={results['XGBoost']['rmse']:.4f}")

    # Neural models
    neural_cfgs = [
        ("LSTM",           LSTMModel(n_feat, d, n_layers=3, dropout=0.15, output_dim=1), False),
        ("Transformer",    TransformerModel(n_feat, d, n_heads=8, n_layers=3, seq_len=seq_len, output_dim=1), False),
        ("NS-Transformer", NSTransformerModel(n_feat, d, n_heads=8, n_layers=3, seq_len=seq_len, output_dim=1), False),
        ("Informer",       InformerModel(n_feat, d, n_heads=8, n_layers=3, seq_len=seq_len, output_dim=1), False),
        ("Chronos-2",      Chronos2Wrapper(n_feat, d, seq_len=seq_len, output_dim=1), False),
        ("ARTEMIS",        ARTEMIS(input_dim=n_feat, d_model=d, n_heads=8, n_layers=3,
                                   seq_len=seq_len, output_dim=1), True),
    ]

    for i, (name, model, use_al) in enumerate(neural_cfgs, start=2):
        print(f"\n[{i}/7] {name}")
        t0 = time.time()
        try:
            trained = train_neural(model, train_ds, val_ds, CFG, use_artemis=use_al)
            preds, targets = evaluate(trained, test_loader, CFG["device"])
            results[name] = compute_metrics(targets, preds)
            r = results[name]
            print(f"  done in {time.time()-t0:.1f}s  RMSE={r['rmse']:.4f}  RankIC={r['rank_ic']:.4f}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results[name] = {k: float("nan") for k in ["rmse","mae","rank_ic","wr2","dir_acc"]}

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Results — Jane Street (Test Set)")
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
    parser = argparse.ArgumentParser(description="Jane Street benchmark")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path containing features.npy, resp.npy, weight.npy")
    args = parser.parse_args()
    run_benchmark(args.data_path)
