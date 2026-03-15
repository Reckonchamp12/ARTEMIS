"""
Benchmark Baselines
===================
All six baseline models for the ARTEMIS benchmark suite.

Every neural model shares the same call signature:

    pred = model(x)      x: (B, T, D)  →  pred: (B,) or (B, output_dim)

This lets the benchmark runners treat all models uniformly.

Models
------
1. LSTMModel           — stacked LSTM, final hidden state → head
2. TransformerModel    — standard encoder-only transformer
3. NSTransformerModel  — non-stationary transformer (Liu et al., NeurIPS 2022)
4. InformerModel       — ProbSparse attention (Zhou et al., AAAI 2021)
5. Chronos2Wrapper     — T5-style univariate encoder (Chronos-2 proxy)
6. XGBoostModel        — gradient-boosted trees (works on flattened features)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _xavier_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# 1. LSTM
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    """
    Stacked bidirectional LSTM.  Predicts from the final hidden state.

    Parameters
    ----------
    input_dim  : feature dimension D
    hidden_dim : LSTM hidden size (maps to d_model in callers)
    n_layers   : number of stacked LSTM layers
    dropout    : dropout probability (applied between layers)
    output_dim : prediction size (1 = regression, 2 = binary cls)
    task       : 'reg' or 'cls' — affects head output (unused at model level)
    """

    def __init__(self, input_dim, hidden_dim=64, n_layers=2,
                 dropout=0.1, output_dim=1, task="reg"):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim
        _xavier_init(self)

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.lstm(x)                     # (B, T, H)
        pred   = self.head(out[:, -1, :])          # (B, output_dim)
        if self.output_dim == 1:
            return pred.squeeze(-1)
        return pred

    def encode(self, x):
        """Return the final hidden state for classification head attachment."""
        out, _ = self.lstm(x)
        return out[:, -1, :]


# ---------------------------------------------------------------------------
# 2. Vanilla Transformer
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerModel(nn.Module):
    """
    Encoder-only transformer.  Uses the last timestep representation
    for prediction (no aggregation pooling — keeps it comparable to ARTEMIS).
    """

    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=2,
                 seq_len=30, dropout=0.1, output_dim=1, task="reg"):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe   = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder   = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head      = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_dim))
        self.output_dim = output_dim
        _xavier_init(self)

    def forward(self, x):
        h    = self.pe(self.proj(x))
        h    = self.encoder(h)
        pred = self.head(h[:, -1, :])
        if self.output_dim == 1:
            return pred.squeeze(-1)
        return pred

    def encode(self, x):
        h = self.pe(self.proj(x))
        return self.encoder(h)[:, -1, :]


# ---------------------------------------------------------------------------
# 3. Non-Stationary Transformer
# ---------------------------------------------------------------------------

class NSTransformerModel(nn.Module):
    """
    Non-Stationary Transformer: per-window z-score normalisation before
    encoding, then re-stationary scaling of the prediction output.

    Reference: Liu et al., "Non-stationary Transformers", NeurIPS 2022.
    """

    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=2,
                 seq_len=30, dropout=0.1, output_dim=1, task="reg"):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe   = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.scale_head = nn.Linear(d_model, 1)      # predicts scale factor
        self.shift_head = nn.Linear(d_model, 1)      # predicts shift
        self.head       = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_dim))
        self.output_dim = output_dim
        _xavier_init(self)

    def forward(self, x):
        # per-window normalisation
        mu    = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True) + 1e-8
        xs    = (x - mu) / sigma

        h       = self.pe(self.proj(xs))
        h       = self.encoder(h)
        h_last  = h[:, -1, :]

        if self.output_dim == 1:
            # re-station the scalar prediction
            pred  = self.head(h_last).squeeze(-1)
            scale = self.scale_head(h_last).squeeze(-1)
            shift = self.shift_head(h_last).squeeze(-1)
            return pred * scale + shift
        return self.head(h_last)

    def encode(self, x):
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True) + 1e-8
        return self.encoder(self.pe(self.proj((x - mu) / sigma)))[:, -1, :]


# ---------------------------------------------------------------------------
# 4. Informer (ProbSparse attention)
# ---------------------------------------------------------------------------

class ProbSparseAttention(nn.Module):
    """
    Samples u = factor·log(T) query positions and computes attention
    only for those, reducing complexity from O(T²) to O(T log T).

    Reference: Zhou et al., "Informer", AAAI 2021.
    """

    def __init__(self, d_model, n_heads, factor=5):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.factor  = factor
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        Q = self.W_q(x).view(B, T, H, Dh).transpose(1, 2)
        K = self.W_k(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, Dh).transpose(1, 2)

        u   = max(1, min(int(self.factor * math.log(T + 1)), T))
        idx = torch.randperm(T, device=x.device)[:u]

        scores = (Q[:, :, idx, :] @ K.transpose(-2, -1)) / Dh ** 0.5
        attn   = F.softmax(scores, dim=-1)
        ctx    = attn @ V                                    # (B, H, u, Dh)

        out = ctx.mean(dim=2, keepdim=True).expand(B, H, T, Dh).clone()
        out[:, :, idx, :] = ctx
        return self.W_o(out.transpose(1, 2).contiguous().view(B, T, D))


class InformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.drop(self.attn(x)))
        return self.norm2(x + self.drop(self.ffn(x)))


class InformerModel(nn.Module):
    """Informer encoder with ProbSparse attention, last-token prediction."""

    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=2,
                 seq_len=30, dropout=0.1, output_dim=1, task="reg"):
        super().__init__()
        self.proj   = nn.Linear(input_dim, d_model)
        self.pe     = PositionalEncoding(d_model, dropout)
        self.blocks = nn.ModuleList([
            InformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.head       = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_dim))
        self.output_dim = output_dim
        _xavier_init(self)

    def forward(self, x):
        h = self.pe(self.proj(x))
        for blk in self.blocks:
            h = blk(h)
        pred = self.head(h[:, -1, :])
        if self.output_dim == 1:
            return pred.squeeze(-1)
        return pred

    def encode(self, x):
        h = self.pe(self.proj(x))
        for blk in self.blocks:
            h = blk(h)
        return h[:, -1, :]


# ---------------------------------------------------------------------------
# 5. Chronos-2 proxy
# ---------------------------------------------------------------------------

class Chronos2Wrapper(nn.Module):
    """
    Lightweight T5-style encoder used as a Chronos-2 stand-in when the
    Amazon pretrained weights are unavailable.  Accepts multivariate input
    but reduces to the first channel (matching the univariate Chronos design).
    """

    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=2,
                 seq_len=30, dropout=0.1, output_dim=1, task="reg"):
        super().__init__()
        self.val_emb = nn.Linear(1, d_model)
        self.pos_emb = nn.Embedding(seq_len + 4, d_model)  # +4 for safety
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, batch_first=True, norm_first=True,
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head       = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_dim))
        self.output_dim = output_dim
        _xavier_init(self)

    def forward(self, x):
        # x: (B, T, D)  — use first channel as univariate proxy
        if x.dim() == 3:
            x = x[:, :, 0]           # (B, T)
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h   = self.val_emb(x.unsqueeze(-1)) + self.pos_emb(pos)
        h   = self.encoder(h)
        pred = self.head(h[:, -1, :])
        if self.output_dim == 1:
            return pred.squeeze(-1)
        return pred

    def encode(self, x):
        if x.dim() == 3:
            x = x[:, :, 0]
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h   = self.val_emb(x.unsqueeze(-1)) + self.pos_emb(pos)
        return self.encoder(h)[:, -1, :]


# ---------------------------------------------------------------------------
# 6. XGBoost wrapper
# ---------------------------------------------------------------------------

class XGBoostModel:
    """
    Thin wrapper around XGBoost (falls back to sklearn GBT if not installed).
    Works on flattened (B, T*D) feature arrays.

    Parameters
    ----------
    task : 'regression' or 'classification'
    """

    def __init__(self, task="regression", n_estimators=500, max_depth=6,
                 learning_rate=0.05, subsample=0.8, colsample_bytree=0.8):
        self.task   = task
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method="hist",
            verbosity=0,
        )
        self._model = None

    def fit(self, X, y):
        try:
            import xgboost as xgb
            if self.task == "classification":
                self._model = xgb.XGBClassifier(
                    **self.params, use_label_encoder=False, eval_metric="logloss"
                )
            else:
                self._model = xgb.XGBRegressor(**self.params)
        except ImportError:
            from sklearn.ensemble import (
                GradientBoostingClassifier, GradientBoostingRegressor,
            )
            klass = GradientBoostingClassifier if self.task == "classification" \
                    else GradientBoostingRegressor
            self._model = klass(
                n_estimators=min(self.params["n_estimators"], 200),
                max_depth=self.params["max_depth"],
                learning_rate=self.params["learning_rate"],
                subsample=self.params["subsample"],
            )
        self._model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        assert self._model is not None, "Call .fit() before .predict()"
        return self._model.predict(X).astype(np.float32)

    def predict_proba(self, X) -> np.ndarray:
        """Returns probability for the positive class (shape: (N,))."""
        assert self._model is not None, "Call .fit() before .predict_proba()"
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)[:, 1].astype(np.float32)
        # regression fallback — shouldn't happen for cls task
        return self._model.predict(X).astype(np.float32)
