from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .models import bernoulli_log_prob_from_logits, diagonal_gaussian_log_prob, standard_normal_log_prob


def elbo_components_from_logits(
    logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-datapoint negative ELBO, BCE reconstruction term, and KL."""
    bce = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1)
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - torch.exp(logvar), dim=1)
    return bce + kl, bce, kl


def map_l2_penalty(parameters: Iterable[torch.Tensor], scale: float) -> torch.Tensor:
    params = list(parameters)
    if not params:
        raise ValueError("Cannot compute MAP L2 penalty over an empty parameter list.")
    penalty = torch.zeros((), device=params[0].device)
    for param in params:
        penalty = penalty + param.pow(2).sum()
    return 0.5 * float(scale) * penalty


def vae_negative_elbo_loss(
    model: nn.Module,
    x: torch.Tensor,
    map_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits, mu, logvar, _ = model(x)
    nelbo, bce, kl = elbo_components_from_logits(logits, x, mu, logvar)
    objective = nelbo.mean() + map_l2_penalty(model.parameters(), map_scale)
    metrics = {
        "negative_elbo": float(nelbo.mean().detach().cpu()),
        "bce": float(bce.mean().detach().cpu()),
        "kl": float(kl.mean().detach().cpu()),
        "lower_bound": float((-nelbo.mean()).detach().cpu()),
    }
    return objective, metrics


def decoder_log_joint(model: nn.Module, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    logits = model.decode_logits(z)
    return standard_normal_log_prob(z) + bernoulli_log_prob_from_logits(logits, x)


def encoder_log_prob(model: nn.Module, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    mu, logvar = model.encode(x)
    return diagonal_gaussian_log_prob(z, mu, logvar)
