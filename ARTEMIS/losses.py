"""
ARTEMIS Physics Losses
======================
Three regularisation terms stacked on top of standard MSE/BCE:

  1. PDE loss  — Hamilton-Jacobi-Bellman residual via Hutchinson trace
                 estimator (avoids the O(L²) full Hessian)
  2. MPR loss  — Market Price of Risk: penalises |μ/σ| > threshold
                 (continuous-time analogue of a Sharpe ratio cap)
  3. Consistency — keep the SDE path close to the deterministic
                   encoder path across all timesteps

Public API
----------
    artemis_loss(pred, target, model, x, lambda_pde, lambda_mpr, lambda_cons)
        → scalar loss tensor

This matches the call sites in all benchmark runners.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PDE Loss — HJB residual
# ---------------------------------------------------------------------------

def _pde_loss(model, x: torch.Tensor, n_pts: int = 2, max_sub: int = 16) -> torch.Tensor:
    """
    Hamilton-Jacobi-Bellman residual:

        L_PDE = E[ (∂V/∂t  +  μ·∇V  +  ½ tr(σ²·∇²V))² ]

    The Laplacian trace is approximated via one Rademacher random vector v:
        tr(σ²·∇²V) ≈ vᵀ ∂(σ²·(∇V·v)) / ∂z

    Parameters
    ----------
    model  : ARTEMIS instance (needs .drift, .diffusion, .value_net)
    x      : (B, T, D)  raw input (used to derive latent Z)
    n_pts  : how many random time-points to sample per call
    max_sub: sub-batch size for grad computations (memory control)
    """
    device = x.device

    with torch.enable_grad():
        mask = torch.ones_like(x)
        Z_full = model.transformer(model.encoder(x, mask))  # (B, T, L)

    B, T, L = Z_full.shape
    residuals = []

    for _ in range(n_pts):
        ti = torch.randint(0, T, (1,)).item()
        sb = min(B, max_sub)

        t_val = torch.full(
            (sb,), ti / max(T - 1, 1),
            device=device, dtype=torch.float32,
        ).requires_grad_(True)

        z_s = Z_full[:sb, ti, :].float().detach().requires_grad_(True)

        V = model.value_net(z_s, t_val)         # (sb,)

        dV_dt, = torch.autograd.grad(
            V.sum(), t_val, create_graph=True,
            retain_graph=True, allow_unused=True,
        )
        dV_dz, = torch.autograd.grad(
            V.sum(), z_s, create_graph=True,
            retain_graph=True, allow_unused=True,
        )

        if dV_dt is None:
            dV_dt = torch.zeros(sb, device=device)
        if dV_dz is None:
            dV_dz = torch.zeros_like(z_s)

        # drift and diffusion at this state/time
        tv_exp = t_val.detach().view(sb, 1, 1).expand(sb, 1, 1)
        mu_s  = model.drift(z_s.unsqueeze(1), tv_exp).squeeze(1)
        sig_s = model.diffusion(z_s.unsqueeze(1), tv_exp).squeeze(1)

        # Hutchinson Laplacian estimate
        v   = torch.randint(0, 2, (sb, L), device=device).float() * 2 - 1
        gvp = (dV_dz * sig_s.pow(2) * v).sum()
        J_v, = torch.autograd.grad(
            gvp, z_s, create_graph=False,
            retain_graph=True, allow_unused=True,
        )
        lap = 0.5 * (v * (J_v if J_v is not None else torch.zeros_like(z_s))).sum(-1)

        residual = dV_dt + (mu_s * dV_dz).sum(-1) + lap
        residuals.append(residual.pow(2).mean())

    if not residuals:
        return torch.tensor(0.0, device=device)
    return torch.stack(residuals).mean()


# ---------------------------------------------------------------------------
# MPR Loss — Market Price of Risk
# ---------------------------------------------------------------------------

def _mpr_loss(mu: torch.Tensor, sigma: torch.Tensor,
              threshold: float = 5.0) -> torch.Tensor:
    """
    Penalise ||λ||² > threshold²  where  λ = μ / σ  (elementwise).

    Enforces that the implied Sharpe ratio in latent space stays bounded,
    preventing drift/diffusion imbalance from distorting the SDE dynamics.

    Parameters
    ----------
    mu        : (B, L) drift at the last timestep
    sigma     : (B, L) diffusion at the last timestep (> 0 by Softplus)
    threshold : maximum allowed ||λ||
    """
    lam    = mu / sigma.clamp(min=1e-3)
    excess = F.relu(lam.pow(2).sum(-1) - threshold ** 2)
    return excess.mean()


# ---------------------------------------------------------------------------
# Consistency Loss
# ---------------------------------------------------------------------------

def _consistency_loss(model, x: torch.Tensor) -> torch.Tensor:
    """
    SDE path should remain close to the deterministic encoder trajectory.
    This prevents the stochastic term from overwhelming the signal.

    L_cons = MSE( Z_sde[:, 1:], stop_grad(Z_enc[:, 1:]) )
    """
    mask  = torch.ones_like(x)
    Z_enc = model.transformer(model.encoder(x, mask))       # (B, T, L)

    B, T, _ = Z_enc.shape
    dt  = 1.0 / max(T - 1, 1)
    tv  = (
        torch.linspace(0, 1, T, device=x.device, dtype=x.dtype)
        .view(1, T, 1).expand(B, T, 1)
    )
    mu    = model.drift(Z_enc, tv)
    sigma = model.diffusion(Z_enc, tv)
    eps   = torch.randn_like(Z_enc)
    Z_sde = Z_enc + mu * dt + sigma * (dt ** 0.5) * eps

    return F.mse_loss(Z_sde[:, 1:, :], Z_enc[:, 1:, :].detach())


# ---------------------------------------------------------------------------
# Combined loss — public API
# ---------------------------------------------------------------------------

def artemis_loss(
    pred:         torch.Tensor,
    target:       torch.Tensor,
    model,
    x:            torch.Tensor,
    lambda_pde:   float = 0.05,
    lambda_mpr:   float = 0.05,
    lambda_cons:  float = 0.02,
    mpr_threshold: float = 5.0,
) -> torch.Tensor:
    """
    Full ARTEMIS training loss.

        L = MSE(pred, target)
            + lambda_pde  · L_PDE
            + lambda_mpr  · L_MPR
            + lambda_cons · L_cons

    Parameters
    ----------
    pred, target : model output and ground truth  (B,) or (B, 1)
    model        : ARTEMIS instance
    x            : raw input tensor (B, T, D)  — used to compute physics terms
    lambda_*     : weights for each physics term (set to 0 to disable)

    Returns
    -------
    Scalar loss tensor (differentiable w.r.t. all model parameters).
    """
    device = pred.device

    # task loss
    l_task = F.mse_loss(pred.float(), target.float())

    # PDE loss
    if lambda_pde > 0:
        try:
            l_pde = _pde_loss(model, x)
        except Exception:
            l_pde = torch.tensor(0.0, device=device)
    else:
        l_pde = torch.tensor(0.0, device=device)

    # MPR loss — need SDE components at last timestep
    if lambda_mpr > 0:
        try:
            _, mu_last, sigma_last = model.get_sde_components(x)
            l_mpr = _mpr_loss(mu_last, sigma_last, mpr_threshold)
        except Exception:
            l_mpr = torch.tensor(0.0, device=device)
    else:
        l_mpr = torch.tensor(0.0, device=device)

    # Consistency loss
    if lambda_cons > 0:
        try:
            l_cons = _consistency_loss(model, x)
        except Exception:
            l_cons = torch.tensor(0.0, device=device)
    else:
        l_cons = torch.tensor(0.0, device=device)

    return l_task + lambda_pde * l_pde + lambda_mpr * l_mpr + lambda_cons * l_cons
