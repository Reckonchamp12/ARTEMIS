"""
scripts/evaluate.py — evaluate a trained checkpoint on a test set.

Usage
-----
python scripts/evaluate.py \
    --dataset optiver \
    --data_path data/optiver/ \
    --model artemis \
    --checkpoint checkpoints/artemis_optiver_best.pt
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmarks.metrics import (
    rmse, mae, rank_ic, weighted_r2,
    directional_accuracy, print_summary_table,
)


def load_model(model_name: str, checkpoint_path: str, n_feat: int, seq_len: int):
    from artemis.model import ARTEMIS
    from benchmarks.baselines import (
        LSTMModel, TransformerModel, NSTransformerModel,
        InformerModel, Chronos2Wrapper,
    )

    d_model = 64
    name = model_name.lower()

    if name == "artemis":
        model = ARTEMIS(input_dim=n_feat, d_model=d_model, n_heads=4,
                        n_layers=2, seq_len=seq_len, output_dim=1)
    elif name == "lstm":
        model = LSTMModel(n_feat, d_model, n_layers=2, dropout=0.1, output_dim=1)
    elif name == "transformer":
        model = TransformerModel(n_feat, d_model, n_heads=4, n_layers=2,
                                 seq_len=seq_len, output_dim=1)
    elif name == "ns_transformer":
        model = NSTransformerModel(n_feat, d_model, n_heads=4, n_layers=2,
                                   seq_len=seq_len, output_dim=1)
    elif name == "informer":
        model = InformerModel(n_feat, d_model, n_heads=4, n_layers=2,
                              seq_len=seq_len, output_dim=1)
    elif name == "chronos2":
        model = Chronos2Wrapper(n_feat, d_model, seq_len=seq_len, output_dim=1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(val_loss={ckpt['val_loss']:.6f})")
    return model


@torch.no_grad()
def run_evaluation(model, test_ds, device="cpu"):
    model.eval().to(device)
    loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=0)
    preds, targets = [], []
    for batch in loader:
        xb = batch[0].to(device)
        yb = batch[1].squeeze(-1).numpy()
        out = model(xb).squeeze(-1).cpu().numpy()
        preds.append(out)
        targets.append(yb)
    return np.concatenate(preds), np.concatenate(targets)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint")
    parser.add_argument("--dataset",    type=str, required=True,
                        choices=["optiver", "crypto_lob", "timeimm", "janestreet"])
    parser.add_argument("--data_path",  type=str, required=True)
    parser.add_argument("--model",      type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load test data
    if args.dataset == "optiver":
        from benchmarks.run_optiver import load_optiver_data
        _, _, test_ds = load_optiver_data(args.data_path)
    elif args.dataset == "crypto_lob":
        from benchmarks.run_crypto_lob import load_crypto_lob_data
        _, _, test_ds = load_crypto_lob_data(args.data_path)
    elif args.dataset == "timeimm":
        from benchmarks.run_timeimm import load_timeimm_data
        _, _, test_ds, y_mean, y_std = load_timeimm_data(args.data_path)
    elif args.dataset == "janestreet":
        from benchmarks.run_jane_street import load_jane_street_data
        _, _, test_ds, _ = load_jane_street_data(args.data_path)

    n_feat  = test_ds.tensors[0].shape[-1]
    seq_len = test_ds.tensors[0].shape[1]

    model = load_model(args.model, args.checkpoint, n_feat, seq_len)
    preds, targets = run_evaluation(model, test_ds, device)

    # invert normalisation for timeimm
    if args.dataset == "timeimm":
        preds   = preds   * y_std + y_mean
        targets = targets * y_std + y_mean

    metrics = {
        "RMSE":   rmse(targets, preds),
        "MAE":    mae(targets, preds),
        "RankIC": rank_ic(targets, preds),
        "WR²":    weighted_r2(targets, preds),
        "DirAcc": directional_accuracy(targets, preds),
    }

    print(f"\n{'='*50}")
    print(f"  {args.model.upper()} on {args.dataset} — Test Results")
    print(f"{'='*50}")
    rows = [{"Metric": k, "Value": f"{v:.4f}"} for k, v in metrics.items()]
    print_summary_table(rows)


if __name__ == "__main__":
    main()
