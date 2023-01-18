from __future__ import annotations

from functools import partial 
from torch import Tensor
from torch.nn import Module
from typing import Callable
from .utils import Loadable

import torch


BetaSchedule = Callable[[int], Tensor]


def linear_beta_schedule(n_timestep: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> Tensor:
    return torch.linspace(beta_start, beta_end, n_timestep + 1).float()


def latent_linear_beta_schedule(n_timestep: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> Tensor:
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, n_timestep + 1).float() ** 2


class DDPM(Module, Loadable):
    def __init__(self, n_timestep: int, schedule: BetaSchedule) -> None:
        super().__init__()
        self.n_timestep = n_timestep
        
        betas = schedule(n_timestep)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_betas", torch.sqrt(betas))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("beta_over_sqrt_one_minus_alphas_cumprod", betas / torch.sqrt(1.0 - alphas_cumprod))

    def __len__(self) -> int:
        return self.n_timestep

    def _extract(self, x: Tensor, t: Tensor, size: tuple) -> Tensor:
        return x.gather(-1, t).reshape(t.size(0), *([1] * (len(size) - 1)))

    def forward_diffusion(self, x_start: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        extract = partial(self._extract, t=t, size=x_start.size())
        noise = torch.randn_like(x_start) if noise is None else noise
        return extract(self.sqrt_alphas_cumprod) * x_start + extract(self.sqrt_one_minus_alphas_cumprod) * noise
        
    @torch.inference_mode()
    def backward_diffusion(self, eps: Tensor, x: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        extract = partial(self._extract, t=t, size=x.size())
        noise = (torch.randn_like(x) if noise is None else noise) * (t > 1).to(x.dtype)[:, None, None, None]
        return extract(self.sqrt_recip_alphas) * (x - eps * extract(self.beta_over_sqrt_one_minus_alphas_cumprod)) + extract(self.sqrt_betas) * noise


class DDIM(DDPM):
    def __init__(self, n_timestep: int, schedule: BetaSchedule, eta: float) -> None:
        super().__init__(n_timestep, schedule)
        self.eta = eta
        self.register_buffer("one_minus_alphas_cumprod", 1.0 - self.alphas_cumprod)

    @torch.inference_mode()
    def backward_diffusion(self, eps: Tensor, x: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        extract = partial(self._extract, t=t, size=x.size())
        extract_prev = partial(self._extract, t=t - 1, size=x.size())
        noise = (torch.randn_like(x) if noise is None else noise) * (t > 1).to(x.dtype)[:, None, None, None]
        xO_t = (x - eps * extract(self.sqrt_one_minus_alphas_cumprod)) / extract(self.sqrt_alphas_cumprod)
        c1 = self.eta * torch.sqrt((1 - extract(self.alphas_cumprod) / extract_prev(self.alphas_cumprod)) * extract_prev(self.one_minus_alphas_cumprod) / extract(self.one_minus_alphas_cumprod))
        c2 = torch.sqrt(extract_prev(self.one_minus_alphas_cumprod) - c1 ** 2)
        return extract_prev(self.sqrt_alphas_cumprod) * xO_t + c1 * noise + c2 * eps