from __future__ import annotations

from dataclasses import dataclass

import torch

from .losses import decoder_log_joint


@dataclass
class HMCSummary:
    samples: torch.Tensor
    acceptance_rate: float
    final_step_size: float


def _grad_log_joint(model, x: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = z.detach().requires_grad_(True)
    logp = decoder_log_joint(model, x, z)
    grad = torch.autograd.grad(logp.sum(), z, create_graph=False)[0]
    return logp.detach(), grad.detach()


def hmc_step(
    model,
    x: torch.Tensor,
    z: torch.Tensor,
    step_size: float,
    n_leapfrog: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    momentum = torch.randn_like(z)
    current_z = z.detach()
    current_momentum = momentum
    current_logp, grad = _grad_log_joint(model, x, current_z)

    z_new = current_z
    p_new = current_momentum + 0.5 * step_size * grad
    for i in range(n_leapfrog):
        z_new = z_new + step_size * p_new
        new_logp, grad = _grad_log_joint(model, x, z_new)
        if i != n_leapfrog - 1:
            p_new = p_new + step_size * grad
    p_new = p_new + 0.5 * step_size * grad
    p_new = -p_new

    current_h = -current_logp + 0.5 * current_momentum.pow(2).sum(dim=1)
    proposed_h = -new_logp + 0.5 * p_new.pow(2).sum(dim=1)
    log_accept_prob = torch.clamp(current_h - proposed_h, max=0.0)
    accept = torch.log(torch.rand_like(log_accept_prob)) < log_accept_prob
    z_out = torch.where(accept[:, None], z_new.detach(), current_z)
    return z_out, accept


def hmc_sample(
    model,
    x: torch.Tensor,
    num_samples: int,
    burn_in: int,
    step_size: float,
    n_leapfrog: int,
    target_acceptance: float | None = None,
    adapt: bool = True,
    init_z: torch.Tensor | None = None,
) -> HMCSummary:
    model.eval()
    z = torch.randn(x.shape[0], model.z_dim, device=x.device) if init_z is None else init_z.to(x.device)
    samples = []
    accepts = []
    current_step = float(step_size)
    total_steps = int(burn_in + num_samples)
    for step in range(total_steps):
        z, accept = hmc_step(model, x, z, current_step, n_leapfrog)
        accept_rate = float(accept.float().mean().detach().cpu())
        accepts.append(accept_rate)
        if adapt and target_acceptance is not None and step < burn_in:
            if accept_rate > target_acceptance:
                current_step *= 1.02
            else:
                current_step *= 0.98
            current_step = min(max(current_step, 1e-4), 1.0)
        if step >= burn_in:
            samples.append(z.detach().clone())
    return HMCSummary(
        samples=torch.stack(samples, dim=0),
        acceptance_rate=float(sum(accepts) / max(len(accepts), 1)),
        final_step_size=current_step,
    )


def full_cov_gaussian_logpdf(samples_for_fit: torch.Tensor, samples_to_score: torch.Tensor, jitter: float = 1e-4) -> torch.Tensor:
    """Fit one full-covariance Gaussian per datapoint and score posterior samples.

    Args:
        samples_for_fit: shape [S_fit, B, D].
        samples_to_score: shape [S_score, B, D].
    Returns:
        Tensor with shape [S_score, B].
    """
    s_fit, batch, dim = samples_for_fit.shape
    mean = samples_for_fit.mean(dim=0)
    centered = samples_for_fit - mean.unsqueeze(0)
    cov = torch.einsum("sbd,sbe->bde", centered, centered) / float(max(s_fit - 1, 1))
    eye = torch.eye(dim, device=samples_for_fit.device).unsqueeze(0)
    cov = cov + jitter * eye
    chol = torch.linalg.cholesky(cov)
    diff = samples_to_score - mean.unsqueeze(0)
    solved = torch.cholesky_solve(diff.permute(1, 2, 0), chol).permute(2, 0, 1)
    quad = (diff * solved).sum(dim=2)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=1, dim2=2)).sum(dim=1)
    log_norm = dim * torch.log(torch.tensor(2.0 * torch.pi, device=samples_for_fit.device)) + logdet
    return -0.5 * (quad + log_norm.unsqueeze(0))


@torch.no_grad()
def bridge_log_marginal_from_posterior_samples(
    model,
    x: torch.Tensor,
    fit_samples: torch.Tensor,
    score_samples: torch.Tensor,
) -> torch.Tensor:
    log_q = full_cov_gaussian_logpdf(fit_samples, score_samples)
    flat_z = score_samples.reshape(-1, score_samples.shape[-1])
    flat_x = x.unsqueeze(0).expand(score_samples.shape[0], -1, -1).reshape(-1, x.shape[-1])
    log_joint = decoder_log_joint(model, flat_x, flat_z).reshape(score_samples.shape[0], x.shape[0])
    return -torch.logsumexp(log_q - log_joint, dim=0) + torch.log(
        torch.tensor(float(score_samples.shape[0]), device=x.device)
    )
