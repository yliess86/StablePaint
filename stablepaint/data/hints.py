from __future__ import annotations

from functools import partial
from torch import Tensor
from typing import Callable

import random
import scipy.stats as stats
import torch


def identity_hints(x: Tensor) -> Tensor:
    return torch.cat((x, torch.ones_like(x)[:1]), dim=0)
    

def random_hints(x: Tensor, mu: float, sigma: float) -> Tensor:
    H, W = x.shape[-2:]
    X = stats.truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma)
    m = torch.rand(H, W, dtype=x.dtype).ge(X.rvs(1)[0])
    h = torch.zeros((4, H, W), dtype=x.dtype)
    h[-1], h[:3, m] = m, x[:3, m]
    return h


HintMethod = Callable[[Tensor], Tensor]
class HintMethods:
    IDENTITY: HintMethod = identity_hints
    RANDOM  : HintMethod = partial(random_hints, mu=1.0, sigma=5e-3)

    @staticmethod
    def mix(a: HintMethod, b: HintMethod, p: float) -> HintMethod:
        return lambda x: (a(x) if random.random() < p else b(x))