"""
ARTEMIS ablation study on DSLOB (Dynamic Synthetic LOB).

We compare seven ARTEMIS variants to isolate the contribution of
each architectural component. The DSLOB crash-regime subset is used
because it contains the non-stationary dynamics that ARTEMIS was
specifically designed to handle.

Ablation variants
-----------------
A0  Full ARTEMIS          — all components active
A1  No SDE                — remove stochastic diffusion term (σ=0)
A2  No PDE loss           — λ_pde = 0
A3  No MPR constraint     — λ_mpr = 0
A4  No physics at all     — λ_pde = λ_mpr = λ_cons = 0 (pure MSE)
A5  No consistency loss   — λ_cons = 0
A6  MLP backbone          — replace transformer encoder with 2-layer MLP
"""

import argparse
import copy
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from benchmarks.metrics import rmse, mae, rank_ic, print_summary_table
from artemis.model import ARTEMIS
from artemis.losses import artemis_loss

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CFG = {
    "seq_len":    30,
    "d_model":    64,
    "n_heads":    4,
    "n_layers":   2,
    "dropout":    0.1,
    "batch_size": 256,
    "epochs":     40,
    "lr":         3e-4,
    "patience":   5,
    "device":     "cuda" if torch.cuda.is_available() else "cpu",
    # base physics weights
    "lambda_pde":  0.10,
    "lambda_mpr":  0.05,
    "lambda_cons": 0.02,
}


# ------------------------------------------------------------------
# MLP baseline for A6
# ------------------------------------------------------------------

class MLPBaseline(nn.Module):
    """Simple 2-layer MLP that takes a flattened LOB window."""

    def __init__(self, input_dim, seq_len, hidden=256, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * seq_len, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------
# Data loader
# ------------------------------------------------------------------

def load_dslob_data(data_path: str):
    """
    Load DSLOB crash-regime arrays.

    Expected files:
        lob_snapshots.npy  — shape (N, F)   float32, normalised
        mid_returns.npy    — shape (N,)     float32
    """
    lob  = np.load(f"{data_path}/lob_snapshots.npy")
    rets = np.load(f"{data_path}/mid_returns.npy")

    seq_len = CFG["seq_len"]
    X, y = [], []
    for i in range(seq_len, len(lob)):
        X.append(lob[i - seq_len : i])
        y.append(rets[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    n = len(X)
    t1, t2 = int(n * 0.70), int(n * 0.85)

    def to_ds(xi, yi):
        return TensorDataset(torch.from_numpy(xi),
                             torch.from_numpy(yi).unsqueeze(-1))

    return to_ds(X[:t1], y[:t1]), to_ds(X[t1:t2], y[t1:t2]), to_ds(X[t2:], y[t2:])


# ------------------------------------------------------------------
# Train / eval
# ------------------------------------------------------------------

def make_loss_fn(lambda_pde, lambda_mpr, lambda_cons, use_artemis_loss, model):
    """Return a loss function with the right physics weights."""
    if not use_artemis_loss:
        return nn.MSELoss()

    def loss_fn(pred, target, xb):
        return artemis_loss(pred, target, model, xb,
                            lambda_pde=lambda_pde,
                            lambda_mpr=lambda_mpr,
                            lambda_cons=lambda_cons)

    return loss_fn


def train_model(model, train_ds, val_ds, loss_cfg: dict):
    device = CFG["device"]
    model  = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])

    use_artemis = loss_cfg.get("use_artemis_loss", False)
    lp  = loss_cfg.get("lambda_pde",  0.0)
    lm  = loss_cfg.get("lambda_mpr",  0.0)
    lc  = loss_cfg.get("lambda_cons", 0.0)
    mse = nn.MSELoss()

    best_rmse, best_state, no_improve = np.inf, None, 0

    for _ in range(CFG["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            if use_artemis:
                loss = artemis_loss(pred, yb, model, xb,
                                    lambda_pde=lp, lambda_mpr=lm, lambda_cons=lc)
            else:
                loss = mse(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb.to(device)).squeeze(-1).cpu().numpy()
                preds.append(out)
                targets.append(yb.squeeze(-1).numpy())
        val_rmse = rmse(np.concatenate(targets), np.concatenate(preds))

        if val_rmse < best_rmse:
            best_rmse  = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= CFG["patience"]:
                break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def test_model(model, test_ds):
    device = CFG["device"]
    model.eval().to(device)
    loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
    preds, targets = [], []
    for xb, yb in loader:
        out = model(xb.to(device)).squeeze(-1).cpu().numpy()
        preds.append(out)
        targets.append(yb.squeeze(-1).numpy())
    return np.concatenate(preds), np.concatenate(targets)


# ------------------------------------------------------------------
# Ablation variants
# ------------------------------------------------------------------

def build_variant(name: str, n_feat: int) -> tuple:
    """
    Returns (model, loss_config_dict).
    loss_config determines which loss terms are active.
    """
    d = CFG["d_model"]
    seq = CFG["seq_len"]

    base_model = lambda no_sde=False: ARTEMIS(
        input_dim=n_feat, d_model=d, n_heads=4, n_layers=2,
        seq_len=seq, output_dim=1, no_sde=no_sde
    )

    variants = {
        "A0 Full": (
            base_model(),
            {"use_artemis_loss": True,
             "lambda_pde": CFG["lambda_pde"],
             "lambda_mpr": CFG["lambda_mpr"],
             "lambda_cons": CFG["lambda_cons"]},
        ),
        "A1 NoSDE": (
            base_model(no_sde=True),
            {"use_artemis_loss": True,
             "lambda_pde": CFG["lambda_pde"],
             "lambda_mpr": CFG["lambda_mpr"],
             "lambda_cons": CFG["lambda_cons"]},
        ),
        "A2 NoPDE": (
            base_model(),
            {"use_artemis_loss": True,
             "lambda_pde": 0.0,
             "lambda_mpr": CFG["lambda_mpr"],
             "lambda_cons": CFG["lambda_cons"]},
        ),
        "A3 NoMPR": (
            base_model(),
            {"use_artemis_loss": True,
             "lambda_pde": CFG["lambda_pde"],
             "lambda_mpr": 0.0,
             "lambda_cons": CFG["lambda_cons"]},
        ),
        "A4 NoPhysics": (
            base_model(),
            {"use_artemis_loss": False},
        ),
        "A5 NoConsistency": (
            base_model(),
            {"use_artemis_loss": True,
             "lambda_pde": CFG["lambda_pde"],
             "lambda_mpr": CFG["lambda_mpr"],
             "lambda_cons": 0.0},
        ),
        "A6 MLP": (
            MLPBaseline(n_feat, seq, hidden=256, output_dim=1),
            {"use_artemis_loss": False},
        ),
    }
    return variants[name]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run_ablation(data_path: str):
    print("\n" + "=" * 60)
    print("  ARTEMIS Ablation Study — DSLOB Crash Regime")
    print("=" * 60)

    train_ds, val_ds, test_ds = load_dslob_data(data_path)
    n_feat = train_ds.tensors[0].shape[-1]
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print(f"Features per step: {n_feat}")

    results = {}
    variant_names = ["A0 Full", "A1 NoSDE", "A2 NoPDE", "A3 NoMPR",
                     "A4 NoPhysics", "A5 NoConsistency", "A6 MLP"]

    for i, vname in enumerate(variant_names):
        print(f"\n[{i+1}/{len(variant_names)}] {vname}")
        t0 = time.time()
        try:
            model, loss_cfg = build_variant(vname, n_feat)
            trained = train_model(model, train_ds, val_ds, loss_cfg)
            preds, targets = test_model(trained, test_ds)
            results[vname] = {
                "rmse":    rmse(targets, preds),
                "mae":     mae(targets, preds),
                "rank_ic": rank_ic(targets, preds),
            }
            print(f"  done in {time.time()-t0:.1f}s  RMSE={results[vname]['rmse']:.4f}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results[vname] = {"rmse": float("nan"), "mae": float("nan"),
                              "rank_ic": float("nan")}

    print("\n" + "=" * 60)
    print("  Ablation Results — DSLOB (Test Set)")
    print("=" * 60)
    rows = []
    for name, m in results.items():
        rows.append({
            "Variant": name,
            "RMSE":    f"{m['rmse']:.4f}",
            "MAE":     f"{m['mae']:.4f}",
            "RankIC":  f"{m['rank_ic']:.4f}",
        })
    print_summary_table(rows)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARTEMIS ablation on DSLOB")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Directory with lob_snapshots.npy and mid_returns.npy")
    args = parser.parse_args()
    run_ablation(args.data_path)
