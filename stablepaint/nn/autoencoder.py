from __future__ import annotations

from lpips import LPIPS
from torch import Tensor
from torch.nn import (BatchNorm2d, Module, Parameter, Sequential)
from typing import NamedTuple
from .modules import (DepthWiseSeparableConv2d, DiagonalGaussianDistribution, Downsample, Identity, Norm, Swish, Upsample)
from .utils import (Jitable, Loadable)

import torch
import torch.nn.init as init
import xformers.ops as xops


class AttentionBlock(Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int) -> None:
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        self.norm = Norm(dim)
        self.qkv = DepthWiseSeparableConv2d(dim, 3 * n_heads * head_dim, 1, bias=False)
        self.proj = DepthWiseSeparableConv2d(n_heads * head_dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.size()
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(B, C, H * W).permute(0, 2, 1).reshape(x.size(0), -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, C, H * W).permute(0, 2, 1).reshape(x.size(0), -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, C, H * W).permute(0, 2, 1).reshape(x.size(0), -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        h = xops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)
        h = h.permute(0, 2, 1, 3).reshape(x.size(0), -1, self.n_heads * self.head_dim).permute(0, 2, 1).reshape(B, C, H, W)
        return self.proj(h) + x


class ResnetBlock(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_ = Sequential(Norm( in_channels), Swish(), DepthWiseSeparableConv2d( in_channels, out_channels, 3, 1, 1, bias=False))
        self.out = Sequential(Norm(out_channels), Swish(), DepthWiseSeparableConv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        self.skip = DepthWiseSeparableConv2d(in_channels, out_channels, 1) if in_channels != out_channels else Identity

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.in_(x)) + self.skip(x)


class Encoder(Sequential, Loadable, Jitable):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__(
            DepthWiseSeparableConv2d(in_channels, 128, 3, 1, 1),
            ResnetBlock(128, 128),
            ResnetBlock(128, 128),
            Downsample(128),
            ResnetBlock(128, 256),
            ResnetBlock(256, 256),
            Downsample(256),
            ResnetBlock(256, 512),
            ResnetBlock(512, 512),
            Downsample(512),
            ResnetBlock(512, 512),
            AttentionBlock(512, 8, 64),
            ResnetBlock(512, 512),
            Norm(512),
            Swish(),
            DepthWiseSeparableConv2d(512, 8, 3, 1, 1),
            DepthWiseSeparableConv2d(8, 8, 1),
        )
        self[-2].apply(self.init_weights)
        self.encode = lambda x: DiagonalGaussianDistribution(self(x))

    @torch.no_grad()
    def init_weights(self, module: Module) -> None:
        if isinstance(module, DepthWiseSeparableConv2d):
            init.constant_(module.point.weight.data, 0.0)
            init.constant_(module.point.bias.data, 0.0)


class Decoder(Sequential, Loadable, Jitable):
    def __init__(self) -> None:
        super().__init__(
            DepthWiseSeparableConv2d(4, 4, 1),
            DepthWiseSeparableConv2d(4, 512, 3, 1, 1),
            ResnetBlock(512, 512),
            AttentionBlock(512, 8, 64),
            ResnetBlock(512, 512),
            ResnetBlock(512, 512),
            Upsample(512),
            ResnetBlock(512, 512),
            ResnetBlock(512, 512),
            ResnetBlock(512, 512),
            Upsample(512),
            ResnetBlock(512, 256),
            ResnetBlock(256, 256),
            ResnetBlock(256, 256),
            Upsample(256),
            ResnetBlock(256, 128),
            ResnetBlock(128, 128),
            ResnetBlock(128, 128),
            Norm(128),
            Swish(),
            DepthWiseSeparableConv2d(128, 3, 3, 1, 1),
        )
        self.decode = lambda z: self(z)


class Discriminator(Sequential):
    def __init__(self) -> None:
        super().__init__(
            DepthWiseSeparableConv2d(  3,  64, 3, 2, 1), Swish(),
            DepthWiseSeparableConv2d( 64, 128, 3, 2, 1, bias=False), BatchNorm2d(128), Swish(),
            DepthWiseSeparableConv2d(128, 256, 3, 2, 1, bias=False), BatchNorm2d(256), Swish(),
            DepthWiseSeparableConv2d(256, 512, 3, 2, 1, bias=False), BatchNorm2d(512), Swish(),
            DepthWiseSeparableConv2d(512, 512, 3, 2, 1, bias=False), BatchNorm2d(512), Swish(),
            DepthWiseSeparableConv2d(512,   1, 3, 1, 1)
        )
        self.apply(self.init_weights)

    @torch.no_grad()
    def init_weights(self, module: Module) -> None:
        if isinstance(module, DepthWiseSeparableConv2d):
            init.normal_(module.point.weight.data, 0.0, 0.02)
        if isinstance(module, BatchNorm2d):
            init.normal_(module.weight.data, 1.0, 0.02)
            init.constant_(module.bias.data, 0.0)


class LPIPSWithDiscriminator(Module):
    class Weights(NamedTuple):
        kl         : float = 1e-6
        disc       : float = 0.5
        disc_factor: float = 1.0
        perc       : float = 1.0

    def __init__(self, weights: Weights, last: Parameter, start_disc: int = 0) -> None:
        super().__init__()
        self.weights = weights
        self.last = last
        self.start_disc = start_disc
        self.disc = Discriminator()
        self.perc = LPIPS(net="vgg").eval()
        self.logvar = Parameter(torch.zeros(size=(), dtype=torch.float32, requires_grad=False))

    def _adaptive_weight(self, nll_loss: Tensor, g_loss: Tensor) -> Tensor:
        grad = lambda loss: torch.norm(torch.autograd.grad(loss, self.last, retain_graph=True)[0])
        return self.weights.disc * torch.clamp(grad(nll_loss) / (grad(g_loss) + 1e-6), 0.0, 1e6).detach()

    def forward(self, illu: Tensor, reco: Tensor, post: DiagonalGaussianDistribution, step: int) -> tuple[Tensor, Tensor]:
        illu, reco = illu.contiguous(), reco.contiguous()
        
        if step >= self.start_disc:
            loss_rec = torch.abs(illu - reco) + self.weights.perc * self.perc(illu, reco)
            loss_nll = torch.sum(loss_rec / torch.exp(self.logvar) + self.logvar) / illu.size(0)
            loss_kl = torch.sum(post.kl()) / illu.size(0)
            loss_g = -torch.mean(self.disc(reco))
            loss_autoencoder = loss_nll + self.weights.kl * loss_kl + self._adaptive_weight(loss_nll, loss_g) * self.weights.disc_factor * loss_g
            
            r_logits, f_logits = self.disc(illu.detach()), self.disc(reco.detach())
            loss_discriminator = self.weights.disc_factor * 0.5 * (torch.mean(torch.relu(1. - r_logits)) + torch.mean(torch.relu(1. + f_logits)))

        else:
            loss_rec = torch.abs(illu - reco) + self.weights.perc * self.perc(illu, reco)
            loss_nll = torch.sum(loss_rec / torch.exp(self.logvar) + self.logvar) / illu.size(0)
            loss_kl = torch.sum(post.kl()) / illu.size(0)
            loss_autoencoder = loss_nll + self.weights.kl * loss_kl
            
            loss_discriminator = torch.tensor(0.0, dtype=illu.dtype, requires_grad=True) * 0.0

        return loss_autoencoder, loss_discriminator