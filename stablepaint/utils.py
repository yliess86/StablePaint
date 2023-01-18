from __future__ import annotations

from PIL import Image
from torch import Tensor
from torch.optim import Optimizer
from torchvision.utils import make_grid
from typing import (Generator, Iterable)

import math


def cycle(iterable: Iterable) -> Generator:
    iterator = iter(iterable)
    while True:
        try: yield next(iterator)
        except StopIteration: iterator = iter(iterable)


class WarmupCosineDecayLR:
    def __init__(self, optim: Optimizer, lr: float, min_lr: float, warmup_steps: int, decay_steps: int) -> None:
        self.optim = optim
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

    def compute(self, step: int) -> float:
        if step < self.warmup_steps: return self.lr * step / self.warmup_steps
        if step > self.decay_steps: return self.min_lr
        decay_ratio = (step - self.warmup_steps) / (self.decay_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.lr - self.min_lr)

    def update(self, step: int) -> None:
        lr = self.compute(step)
        for param_group in self.optim.param_groups:
            param_group["lr"] = lr


def to_pil(x: Tensor) -> Image.Image:
    grid = make_grid((0.5 + 0.5 * x).clip_(0.0, 1.0))
    return Image.fromarray((grid.permute(1, 2, 0) * 255.0).byte().cpu().numpy())