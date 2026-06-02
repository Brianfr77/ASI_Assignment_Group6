from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    """One-hidden-layer MLP VAE matching the AEVB MNIST experiment."""

    def __init__(self, z_dim: int, hidden_dim: int = 500, input_dim: int = 784, init_std: float = 0.01):
        super().__init__()
        self.z_dim = int(z_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_dim = int(input_dim)
        self.encoder_hidden = nn.Linear(input_dim, hidden_dim)
        self.encoder_mu = nn.Linear(hidden_dim, z_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, z_dim)
        self.decoder_hidden = nn.Linear(z_dim, hidden_dim)
        self.decoder_logits = nn.Linear(hidden_dim, input_dim)
        self.reset_parameters(init_std)

    def reset_parameters(self, init_std: float) -> None:
        for param in self.parameters():
            nn.init.normal_(param, mean=0.0, std=init_std)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.encoder_hidden(x))
        return self.encoder_mu(h), self.encoder_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.decoder_hidden(z))
        return self.decoder_logits(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_logits(z)
        return logits, mu, logvar, z

    def encoder_parameters(self):
        yield from self.encoder_hidden.parameters()
        yield from self.encoder_mu.parameters()
        yield from self.encoder_logvar.parameters()

    def decoder_parameters(self):
        yield from self.decoder_hidden.parameters()
        yield from self.decoder_logits.parameters()


class GenerativeModel(nn.Module):
    """Decoder-only model used by the MCEM baseline."""

    def __init__(self, z_dim: int, hidden_dim: int = 100, input_dim: int = 784, init_std: float = 0.01):
        super().__init__()
        self.z_dim = int(z_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_dim = int(input_dim)
        self.decoder_hidden = nn.Linear(z_dim, hidden_dim)
        self.decoder_logits = nn.Linear(hidden_dim, input_dim)
        self.reset_parameters(init_std)

    def reset_parameters(self, init_std: float) -> None:
        for param in self.parameters():
            nn.init.normal_(param, mean=0.0, std=init_std)

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.decoder_hidden(z))
        return self.decoder_logits(h)


def bernoulli_log_prob_from_logits(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return -F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1)


def standard_normal_log_prob(z: torch.Tensor) -> torch.Tensor:
    return -0.5 * (z.pow(2) + math.log(2.0 * math.pi)).sum(dim=1)


def diagonal_gaussian_log_prob(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (math.log(2.0 * math.pi) + logvar + (z - mu).pow(2) / torch.exp(logvar)).sum(dim=1)


def sample_bernoulli_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.bernoulli(torch.sigmoid(logits))
