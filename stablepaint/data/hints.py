from __future__ import annotations

from functools import partial
from torch import Tensor

import scipy.stats as stats
import torch


def random_hints(x: Tensor, mu: float, sigma: float) -> Tensor:
    H, W = x.shape[-2:]
    X = stats.truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma)
    m = torch.rand(H, W, dtype=x.dtype).ge(X.rvs(1)[0])
    h = torch.zeros((4, H, W), dtype=x.dtype)
    h[-1], h[:3, m] = m, x[:3, m]
    return h


class HintMethod:
    RANDOM  = partial(random_hints, mu=1.0, sigma=5e-3)