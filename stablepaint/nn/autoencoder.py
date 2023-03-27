from __future__ import annotations

from lpips import LPIPS
from torch import Tensor
from torch.nn import (BatchNorm2d, Conv2d, Module, Parameter, Sequential)
from typing import NamedTuple
from .modules import (DiagonalGaussianDistribution, Downsample, Identity, Norm, Swish, Upsample)
from .utils import (Jitable, Loadable, scale)

import torch
import torch.nn.init as init
import torch.nn.functional as F


class AttentionBlock(Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int) -> None:
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        self.norm = Norm(dim)
        self.qkv = Conv2d(dim, 3 * n_heads * head_dim, 1, bias=False)
        self.proj = Conv2d(n_heads * head_dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.size()
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(B, C, H * W).permute(0, 2, 1).reshape(x.size(0), -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, C, H * W).permute(0, 2, 1).reshape(x.size(0), -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, C, H * W).permute(0, 2, 1).reshape(x.size(0), -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        h = h.permute(0, 2, 1, 3).reshape(x.size(0), -1, self.n_heads * self.head_dim).permute(0, 2, 1).reshape(B, C, H, W)
        return self.proj(h) + x


class ResnetBlock(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_ = Sequential(Norm( in_channels), Swish(), Conv2d( in_channels, out_channels, 3, 1, 1, bias=False))
        self.out = Sequential(Norm(out_channels), Swish(), Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        self.skip = Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else Identity

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.in_(x)) + self.skip(x)


class Encoder(Sequential, Loadable, Jitable):
    def __init__(self, in_channels: int = 3, t: float = 1.0) -> None:
        h1, h2, h3 = map(lambda x: scale(x, t), (128, 256, 512))
        super().__init__(
            Conv2d(in_channels, h1, 3, 1, 1),
            ResnetBlock(h1, h1), ResnetBlock(h1, h1), Downsample(h1),
            ResnetBlock(h1, h2), ResnetBlock(h2, h2), Downsample(h2),
            ResnetBlock(h2, h3), ResnetBlock(h3, h3), Downsample(h3),
            ResnetBlock(h3, h3), AttentionBlock(h3, 8, h3 // 8), ResnetBlock(h3, h3),
            Norm(h3), Swish(), Conv2d(h3, 8, 3, 1, 1),
            Conv2d(8, 8, 1),
        )
        self[-2].apply(self.init_weights)
        self.encode = lambda x: DiagonalGaussianDistribution(self(x))

    @torch.no_grad()
    def init_weights(self, module: Module) -> None:
        if isinstance(module, Conv2d):
            init.constant_(module.weight.data, 0.0)
            init.constant_(module.bias.data, 0.0)

    def load(self: Module, ckpt: dict[str, Tensor]) -> Module:
        if self[0].weight.data.shape[1] > ckpt["0.weight"].data.shape[1]:
            print(f"Input Channels Differs [{self[0].weight.data.shape[1]} > {ckpt['0.weight'].data.shape[1]}]")
            print("Peforming Partial Initialisation...")
            self[0].weight.data[:, :3] = ckpt["0.weight"].data[:, :3]
            self[0].weight.data[:, -1] = ckpt["0.weight"].data[:, -1]
            ckpt["0.weight"].data = self[0].weight.data
        self.load_state_dict(ckpt, strict=False)
        return self


class Decoder(Sequential, Loadable, Jitable):
    def __init__(self, t: float = 1.0) -> None:
        h1, h2, h3 = map(lambda x: scale(x, t), (128, 256, 512))
        super().__init__(
            Conv2d(4, 4, 1),
            Conv2d(4, h3, 3, 1, 1),
            ResnetBlock(h3, h3), AttentionBlock(h3, 8, h3 // 8), ResnetBlock(h3, h3), ResnetBlock(h3, h3),
            Upsample(h3), ResnetBlock(h3, h3), ResnetBlock(h3, h3), ResnetBlock(h3, h3),
            Upsample(h3), ResnetBlock(h3, h2), ResnetBlock(h2, h2), ResnetBlock(h2, h2),
            Upsample(h2), ResnetBlock(h2, h1), ResnetBlock(h1, h1), ResnetBlock(h1, h1),
            Norm(h1), Swish(), Conv2d(h1, 3, 3, 1, 1),
        )
        self.decode = lambda z: self(z)


class Discriminator(Sequential):
    def __init__(self, t: float = 1.0) -> None:
        h0, h1, h2, h3 = map(lambda x: scale(x, t), (64, 128, 256, 512))
        super().__init__(
            Conv2d(            3, h0, 3, 2, 1), Swish(),
            Conv2d(h0, h1, 3, 2, 1, bias=False), BatchNorm2d(h1), Swish(),
            Conv2d(h1, h2, 3, 2, 1, bias=False), BatchNorm2d(h2), Swish(),
            Conv2d(h2, h3, 3, 2, 1, bias=False), BatchNorm2d(h3), Swish(),
            Conv2d(h3, h3, 3, 2, 1, bias=False), BatchNorm2d(h3), Swish(),
            Conv2d(h3,             1, 3, 1, 1)
        )
        self.apply(self.init_weights)

    @torch.no_grad()
    def init_weights(self, module: Module) -> None:
        if isinstance(module, Conv2d):
            init.normal_(module.weight.data, 0.0, 0.02)
        if isinstance(module, BatchNorm2d):
            init.normal_(module.weight.data, 1.0, 0.02)
            init.constant_(module.bias.data, 0.0)


class LPIPSWithDiscriminator(Module):
    class Weights(NamedTuple):
        kl         : float = 1e-6
        disc       : float = 0.5
        disc_factor: float = 1.0
        perc       : float = 1.0

    def __init__(self, last: Parameter, start_disc: int = 0, t: float = 1.0, weights: Weights = Weights()) -> None:
        super().__init__()
        self.weights = weights
        self.last = last
        self.start_disc = start_disc
        self.disc = Discriminator(t=t)
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