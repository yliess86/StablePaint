from __future__ import annotations

from torch import Tensor
from torch.nn import (Conv2d, GroupNorm, Module, ReflectionPad2d, Sequential, SiLU)

import torch


Swish = SiLU
Identity = lambda x: x


class Norm(GroupNorm):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_groups=32, num_channels=num_channels, eps=1e-5, affine=True)


class DepthWiseSeparableConv2d(Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True) -> None:
        super().__init__()
        self.depth = Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.point = Conv2d(in_channels, out_channels, 1, bias=bias)


class Upsample2x(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.repeat_interleave(2, 2).repeat_interleave(2, 3)


class Upsample(Sequential):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.up = Upsample2x()
        self.conv = DepthWiseSeparableConv2d(in_channels, in_channels, 3, 1, 1)
        

class Downsample(Sequential):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.pad = ReflectionPad2d(1)
        self.conv = DepthWiseSeparableConv2d(in_channels, in_channels, 3, 2, 0)


class DiagonalGaussianDistribution:
    def __init__(self, params: Tensor):
        self.mean, self.log_var = params.chunk(2, dim=1)
        self.log_var = self.log_var.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.log_var)
        self.var = torch.exp(self.log_var)

    def sample(self) -> Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self) -> Tensor:
        return 0.5 * (self.mean.pow(2) + self.var - 1.0 - self.log_var).sum(dim=[1, 2, 3])