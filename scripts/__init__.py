"""
scripts/train.py — train ARTEMIS (or any baseline) on a given dataset.

Usage examples
--------------
# Train ARTEMIS on Optiver data
python scripts/train.py --dataset optiver --data_path data/optiver/ --model artemis

# Train LSTM on Time-IMM data
python scripts/train.py --dataset timeimm --data_path data/timeimm/ --model lstm

# Train XGBoost on Jane Street data
python scripts/train.py --dataset janestreet --data_path data/janestreet/ --model xgboost

Supported datasets : optiver, crypto_lob, timeimm, janestreet
Supported models   : artemis, lstm, transformer, ns_transformer,
                     informer, chronos2, xgboost
"""

import argparse
import os
import random
import time

import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_model(model_name: str, n_feat: int, seq_len: int, task: str = "regression"):
    from artemis.model import ARTEMIS
    from benchmarks.baselines import (
        LSTMModel, TransformerModel, NSTransformerModel,
        InformerModel, Chronos2Wrapper, XGBoostModel,
    )

    output_dim = 1
    d_model    = 64

    name = model_name.lower()
    if name == "artemis":
        return ARTEMIS(input_dim=n_feat, d_model=d_model, n_heads=4,
                       n_layers=2, seq_len=seq_len, output_dim=output_dim)
    elif name == "lstm":
        return LSTMModel(n_feat, d_model, n_layers=2, dropout=0.1, output_dim=output_dim)
    elif name == "transformer":
        return TransformerModel(n_feat, d_model, n_heads=4, n_layers=2,
                                seq_len=seq_len, output_dim=output_dim)
    elif name == "ns_transformer":
        return NSTransformerModel(n_feat, d_model, n_heads=4, n_layers=2,
                                  seq_len=seq_len, output_dim=output_dim)
    elif name == "informer":
        return InformerModel(n_feat, d_model, n_heads=4, n_layers=2,
                             seq_len=seq_len, output_dim=output_dim)
    elif name == "chronos2":
        return Chronos2Wrapper(n_feat, d_model, seq_len=seq_len, output_dim=output_dim)
    elif name == "xgboost":
        return XGBoostModel(task=task)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_data(dataset: str, data_path: str):
    """Route to the correct data loader based on dataset name."""
    if dataset == "optiver":
        from benchmarks.run_optiver import load_optiver_data
        return load_optiver_data(data_path)
    elif dataset == "crypto_lob":
        from benchmarks.run_crypto_lob import load_crypto_lob_data
        return load_crypto_lob_data(data_path)
    elif dataset == "timeimm":
        from benchmarks.run_timeimm import load_timeimm_data
        return load_timeimm_data(data_path)
    elif dataset == "janestreet":
        from benchmarks.run_jane_street import load_jane_street_data
        return load_jane_street_data(data_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def save_checkpoint(model, path: str, epoch: int, val_loss: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "val_loss":  val_loss,
        "state_dict": model.state_dict(),
    }, path)
    print(f"  checkpoint saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="Train ARTEMIS or a baseline model")
    parser.add_argument("--dataset",   type=str, required=True,
                        choices=["optiver", "crypto_lob", "timeimm", "janestreet"])
    parser.add_argument("--data_path", type=str, required=True,
                        help="Directory containing the dataset .npy files")
    parser.add_argument("--model",     type=str, default="artemis",
                        help="Model name (artemis / lstm / transformer / etc.)")
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--batch_size",type=int, default=256)
    parser.add_argument("--patience",  type=int, default=7)
    parser.add_argument("--save_dir",  type=str, default="checkpoints/")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Dataset: {args.dataset}  |  Model: {args.model}")

    # load data
    data = load_data(args.dataset, args.data_path)
    train_ds, val_ds, test_ds = data[0], data[1], data[2]

    n_feat  = train_ds.tensors[0].shape[-1]
    seq_len = train_ds.tensors[0].shape[1]
    print(f"Features: {n_feat}   Seq len: {seq_len}")
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    model = get_model(args.model, n_feat, seq_len)

    # XGBoost has its own fit/predict API
    if args.model.lower() == "xgboost":
        X_tr = train_ds.tensors[0].numpy().reshape(len(train_ds), -1)
        y_tr = train_ds.tensors[1].numpy().flatten()
        model.fit(X_tr, y_tr)
        print("XGBoost training complete.")
        return

    # Neural training loop
    from torch.utils.data import DataLoader
    import torch.nn as nn
    from artemis.losses import artemis_loss

    model = model.to(device)
    use_artemis = args.model.lower() == "artemis"

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_val, best_epoch, no_improve = np.inf, 0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        t0 = time.time()
        for xb, yb, *_ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            if use_artemis:
                loss = artemis_loss(pred, yb, model, xb)
            else:
                loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        scheduler.step()
        tr_loss /= len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, *_ in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
        val_loss /= len(val_loader)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={tr_loss:.6f}  val_loss={val_loss:.6f}  "
              f"({elapsed:.1f}s)")

        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch
            no_improve = 0
            ckpt_path  = os.path.join(args.save_dir, f"{args.model}_{args.dataset}_best.pt")
            save_checkpoint(model, ckpt_path, epoch, val_loss)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

    print(f"\nBest val loss: {best_val:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
