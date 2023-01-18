from __future__ import annotations

from torch import Tensor
from torch.nn import Module
from torch.jit import ScriptModule
from tqdm import tqdm
from typing import Protocol

import torch
import torch.jit as jit
import torchinfo


class Loadable(Protocol):
    def load(self: Module, ckpt: dict[str, Tensor]) -> Module:
        self.load_state_dict(ckpt, strict=False)
        return self


class Jitable(Protocol):
    @torch.inference_mode()
    def jit(self: Module, *input_shape: tuple, warmups: int) -> ScriptModule:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        example = torch.randn(1, *input_shape, dtype=dtype, device=device)
        jit_module = jit.optimize_for_inference(jit.trace(self.eval(), example))
        for _ in tqdm(range(warmups), desc=f"[{self.__class__.__name__}] Warmups"): jit_module(example)
        return jit_module


class Summarizable(Protocol):
    @torch.inference_mode()
    def summary(self, *input_shape: tuple) -> None:
        input_shape = 1, *input_shape
        torchinfo.summary(self, input_shape, depth=10)