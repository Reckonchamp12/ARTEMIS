"""
ARTEMIS — Adaptive Real-Time Market Intelligence System
=======================================================
Hybrid Neural SDE + Transformer architecture for financial time-series.

Architecture overview:
    Input (B, T, D)
        │
    TimeEncoder  ──  Fourier time basis + observation mask
        │
    Transformer  ──  multi-head self-attention over the time axis
        │
    Z ∈ R^(B, T, L)   latent trajectory
        │
    ┌───┴───────────────────────┐
    │   DriftNet  μ(Z, t)       │  Neural SDE step (Euler-Maruyama)
    │   DiffusionNet σ(Z, t)    │  Z_sde = Z + μ·dt + σ·√dt·ε
    └───────────────────────────┘
        │
    ValueNet V(Z, t)  ← HJB PDE loss only, not used in forward()
        │
    Prediction head → (B, output_dim)

Physics losses are computed in artemis/losses.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class TimeEncoder(nn.Module):
    """
    Maps (x, mask) → latent trajectory Z ∈ R^(B, T, d_model).

    Concatenates:
        x          (B, T, D)    raw features
        mask       (B, T, D)    observation mask  (1=observed, 0=missing)
        time_feats (B, T, 2K)   Fourier basis of the time index

    Then projects to d_model via a small MLP.
    """

    def __init__(self, input_dim, d_model, hidden_dim, num_basis=32, dropout=0.1):
        super().__init__()
        self.num_basis = num_basis
        self.register_buffer("freqs", torch.linspace(1.0, num_basis, num_basis))
        in_feat = input_dim * 2 + 2 * num_basis
        self.proj = nn.Sequential(
            nn.Linear(in_feat, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def _fourier(self, t_norm):
        angles = t_norm.unsqueeze(-1) * self.freqs * 2 * math.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, x, mask):
        B, T, _ = x.shape
        t_norm = (
            torch.arange(T, device=x.device, dtype=x.dtype) / max(T - 1, 1)
        ).unsqueeze(0).expand(B, -1)
        time_feats = self._fourier(t_norm)
        return self.proj(torch.cat([x, mask, time_feats], dim=-1))


class DriftNet(nn.Module):
    """μ(Z, t) — deterministic part of the neural SDE."""

    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, z, tv):
        return self.net(torch.cat([z, tv], dim=-1))


class DiffusionNet(nn.Module):
    """σ(Z, t) — stochastic diffusion; Softplus keeps σ > 0."""

    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, d_model),
            nn.Softplus(),
        )

    def forward(self, z, tv):
        return self.net(torch.cat([z, tv], dim=-1))


class ValueNet(nn.Module):
    """
    V(Z, t) — scalar value function used in the HJB PDE residual loss.
    This module is not called during the standard forward pass; it is only
    accessed by artemis_loss() in artemis/losses.py.
    """

    def __init__(self, d_model, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, t):
        # t can be scalar, (B,), or (B, 1) — normalise to (B, 1) or (..., 1)
        if t.dim() == 0 or (t.dim() == 1 and t.shape[0] != z.shape[0]):
            t = t.expand(z.shape[0])
        if t.dim() < z.dim():
            t = t.view(*t.shape, *([1] * (z.dim() - t.dim())))
        if t.shape[-1] != 1:
            t = t.unsqueeze(-1)
        t = t.expand(*z.shape[:-1], 1).to(z.dtype)
        return self.net(torch.cat([z, t], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# ARTEMIS full model
# ---------------------------------------------------------------------------

class ARTEMIS(nn.Module):
    """
    Full ARTEMIS model. Shares the same call signature as all baseline models:

        pred = model(x)          x: (B, T, D)  →  pred: (B,) or (B, output_dim)

    Parameters
    ----------
    input_dim  : number of input features per timestep
    d_model    : latent / model dimension  (default 64)
    n_heads    : transformer attention heads
    n_layers   : transformer encoder layers
    seq_len    : expected input sequence length (informational, not hard-required)
    output_dim : prediction size (1 for regression, 2 for binary classification)
    dropout    : dropout probability
    no_sde     : ablation flag — if True, diffusion term σ is zeroed (A1 variant)
    task       : 'reg' (regression) or 'cls' (classification)
    """

    def __init__(
        self,
        input_dim,
        d_model    = 64,
        n_heads    = 4,
        n_layers   = 2,
        seq_len    = 30,
        output_dim = 1,
        dropout    = 0.1,
        no_sde     = False,
        task       = "reg",
    ):
        super().__init__()
        self.d_model    = d_model
        self.no_sde     = no_sde
        self.task       = task
        self.output_dim = output_dim

        hidden = d_model * 2

        # 1. Time encoder
        self.encoder = TimeEncoder(input_dim, d_model, hidden,
                                   num_basis=32, dropout=dropout)

        # 2. Transformer backbone
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # 3. Neural SDE components
        self.drift     = DriftNet(d_model, hidden)
        self.diffusion = DiffusionNet(d_model, hidden)

        # 4. Value network (physics loss only — not used in forward)
        self.value_net = ValueNet(d_model, hidden_dim=d_model)

        # 5. Prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    def forward(self, x, mask=None):
        """
        x    : (B, T, D)   input features
        mask : (B, T, D)   observation mask; auto-created (all ones) if None

        Returns
        -------
        Regression (output_dim=1) : (B,)
        Classification (output_dim>1) : (B, output_dim)
        """
        B, T, D = x.shape
        if mask is None:
            mask = torch.ones_like(x)

        # 1. encode
        Z = self.encoder(x, mask)                         # (B, T, L)
        Z = self.transformer(Z)                           # (B, T, L)

        # 2. SDE step (Euler-Maruyama, vectorised over time)
        dt = 1.0 / max(T - 1, 1)
        tv = (
            torch.linspace(0, 1, T, device=x.device, dtype=x.dtype)
            .view(1, T, 1).expand(B, T, 1)
        )
        mu    = self.drift(Z, tv)                         # (B, T, L)
        sigma = self.diffusion(Z, tv)                     # (B, T, L)

        if self.no_sde:
            Z_sde = Z + mu * dt
        else:
            eps   = torch.randn_like(Z)
            Z_sde = Z + mu * dt + sigma * (dt ** 0.5) * eps

        # 3. Predict from last latent state
        out = self.head(Z_sde[:, -1, :])                  # (B, output_dim)

        if self.output_dim == 1:
            out = out.squeeze(-1)                          # (B,)

        return out

    # ------------------------------------------------------------------
    # Helpers used by artemis_loss()
    # ------------------------------------------------------------------

    def get_sde_components(self, x, mask=None):
        """
        Returns (Z_last, mu_last, sigma_last) for physics loss computation.
        Shape: each (B, d_model).
        """
        B, T, _ = x.shape
        if mask is None:
            mask = torch.ones_like(x)
        Z  = self.transformer(self.encoder(x, mask))      # (B, T, L)
        tv = torch.ones(B, 1, device=x.device, dtype=x.dtype).unsqueeze(-1)
        mu    = self.drift(Z[:, -1, :], tv[:, 0, :])
        sigma = self.diffusion(Z[:, -1, :], tv[:, 0, :])
        return Z[:, -1, :], mu, sigma

    @torch.no_grad()
    def encode(self, x, mask=None):
        """Deterministic latent state at the last timestep (no SDE noise)."""
        if mask is None:
            mask = torch.ones_like(x)
        return self.transformer(self.encoder(x, mask))[:, -1, :]
