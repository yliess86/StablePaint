from __future__ import annotations

from torch import Tensor
from torch.nn import (Conv2d, GELU, LayerNorm, Linear, Module, ModuleList, Sequential)
from .modules import (Downsample, Identity, Norm, Swish, Upsample)
from .utils import (Loadable, scale)

import torch
import torch.nn.functional as F


class ResBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int):
        super().__init__()
        self.emb = Sequential(Swish(), Linear(emb_dim, out_channels))
        self.in_ = Sequential(Norm( in_channels), Swish(), Conv2d( in_channels, out_channels, 3, 1, 1, bias=False))
        self.out = Sequential(Norm(out_channels), Swish(), Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        self.skip = Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else Identity

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.out(self.in_(x) + self.emb(t)[:, :, None, None]) + self.skip(x)


class CrossAttention(Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int) -> None:
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        self.norm = LayerNorm(dim, eps=1e-5)
        self.q = Linear(dim, head_dim * n_heads, bias=False)
        self.kv = Linear(dim, 2 * head_dim * n_heads, bias=False)
        self.out = Linear(head_dim * n_heads, dim)

    def forward(self, x: Tensor, ctx: Tensor | None = None) -> Tensor:
        x = self.norm(x)
        ctx = x if ctx is None else ctx
        q, (k, v) = self.q(x), self.kv(ctx).chunk(2, dim=1)
        q = q.reshape(x.size(0), -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(x.size(0), -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(x.size(0), -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        with torch.backends.cuda.sdp_kernel(enable_flash=False):
            h = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        h = h.permute(0, 2, 1, 3).reshape(x.size(0), -1, self.n_heads * self.head_dim)
        return self.out(h)


class GeGLU(Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = LayerNorm(in_dim, eps=1e-5)
        self.proj = Linear(in_dim, out_dim * 2)
        self.gelu = GELU()
        
    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(self.norm(x)).chunk(2, dim=-1)
        return x * self.gelu(gate)


class TransformerBlock(Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int) -> None:
        super().__init__()
        self.cross_1 = CrossAttention(dim, n_heads, head_dim)
        self.cross_2 = CrossAttention(dim, n_heads, head_dim)
        self.geglu = GeGLU(dim, dim * 4)
        self.out = Linear(dim * 4, dim)

    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        x = self.cross_1(x, None) + x
        x = self.cross_2(x, ctx) + x
        x = self.out(self.geglu(x)) + x
        return x


class SpatialTransformer(Module):
    def __init__(self, channels: int, n_heads: int) -> None:
        super().__init__()
        self.ctx = Conv2d(4 + 4, channels, 1)
        self.norm = Norm(channels)
        self.proj_in = Conv2d(channels, channels, 1)
        self.block = TransformerBlock(channels, n_heads, channels // n_heads)
        self.proj_out = Conv2d(channels, channels, 1)

    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        B, C, H, W = x.size()
        ctx = F.avg_pool2d(ctx, (ctx.size(-2) // H, ctx.size(-1) // W))
        l = self.ctx(ctx).reshape(B, C, H * W).permute(0, 2, 1)
        h = self.proj_in(self.norm(x)).reshape(B, C, H * W).permute(0, 2, 1)
        h = self.block(h, l).permute(0, 2, 1).reshape(B, C, H, W)
        return self.proj_out(h) + x


class PositionalEncoding(Module):
    def __init__(self, dim: int, max_period: int = 10_000) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.pow(max_period, -torch.arange(0, dim // 2, dtype=torch.float64) / (dim // 2)).float())

    def forward(self, t: Tensor) -> Tensor:
        x = t[:, None] * self.freqs[None, :]
        return torch.cat((torch.cos(x), torch.sin(x)), dim=-1)



class UNet(Module, Loadable):
    def __init__(self, t: float = 1.0):
        super().__init__()
        h0, h1, h2 = map(lambda x: scale(x, t), (256, 512, 1_024))
        self.pos_enc = PositionalEncoding(h2)
        self.time_emb = Sequential(Linear(h2, h2), Swish(), Linear(h2, h2))
        self.encoder = ModuleList([
            ModuleList([Conv2d(4, h0, 3, 1, 1)]),
            ModuleList([ResBlock(h0, h0, h2), SpatialTransformer(h0, 8)]),
            ModuleList([ResBlock(h0, h0, h2), SpatialTransformer(h0, 8)]),
            ModuleList([Downsample(h0)]),
            ModuleList([ResBlock(h0, h1, h2), SpatialTransformer(h1, 8)]),
            ModuleList([ResBlock(h1, h1, h2), SpatialTransformer(h1, 8)]),
            ModuleList([Downsample(h1)]),
            ModuleList([ResBlock(h1, h2, h2), SpatialTransformer(h2, 8)]),
            ModuleList([ResBlock(h2, h2, h2), SpatialTransformer(h2, 8)]),
            ModuleList([Downsample(h2)]),
            ModuleList([ResBlock(h2, h2, h2)]),
            ModuleList([ResBlock(h2, h2, h2)]),
        ])
        self.bottleneck = ModuleList([
            ResBlock(h2, h2, h2), SpatialTransformer(h2, 8),
            ResBlock(h2, h2, h2),
        ])
        self.decoder = ModuleList([
            ModuleList([ResBlock(h2 + h2, h2, h2)]),
            ModuleList([ResBlock(h2 + h2, h2, h2)]),
            ModuleList([ResBlock(h2 + h2, h2, h2), Upsample(h2)]),
            ModuleList([ResBlock(h2 + h2, h2, h2), SpatialTransformer(h2, 8)]),
            ModuleList([ResBlock(h2 + h2, h2, h2), SpatialTransformer(h2, 8)]),
            ModuleList([ResBlock(h2 + h1, h2, h2), SpatialTransformer(h2, 8), Upsample(h2)]),
            ModuleList([ResBlock(h2 + h1, h1, h2), SpatialTransformer(h1, 8)]),
            ModuleList([ResBlock(h1 + h1, h1, h2), SpatialTransformer(h1, 8)]),
            ModuleList([ResBlock(h0 + h1, h1, h2), SpatialTransformer(h1, 8), Upsample(h1)]),
            ModuleList([ResBlock(h0 + h1, h0, h2), SpatialTransformer(h0, 8)]),
            ModuleList([ResBlock(h0 + h0, h0, h2), SpatialTransformer(h0, 8)]),
            ModuleList([ResBlock(h0 + h0, h0, h2), SpatialTransformer(h0, 8)]),
        ])
        self.out = Sequential(Norm(h0), Swish(), Conv2d(h0, 4, 3, 1, 1))

    def apply(self, modules: ModuleList, x: Tensor, t: Tensor, ctx: Tensor) -> Tensor:
        for module in modules:
            if isinstance(module, ResBlock): x = module(x, t)
            elif isinstance(module, SpatialTransformer): x = module(x, ctx)
            else: x = module(x)
        return x

    def forward(self, x: Tensor, t: Tensor, ctx: Tensor) -> Tensor:
        t = self.time_emb(self.pos_enc(t))
        residuals = []
        for modules in self.encoder:
            x = self.apply(modules, x, t, ctx)
            residuals.append(x)
        x = self.apply(self.bottleneck, x, t, ctx)
        for modules in self.decoder:
            x = torch.cat((x, residuals.pop()), dim=1)
            x = self.apply(modules, x, t, ctx)
        x = self.out(x)
        return x