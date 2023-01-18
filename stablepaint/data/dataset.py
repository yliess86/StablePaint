from __future__ import annotations

from itertools import chain
from pathlib import Path
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from .hints import HintMethod
from .sketch import sketch
from .smooth import l0_smoothing
from ..nn.autoencoder import Encoder
from ..nn.modules import DiagonalGaussianDistribution

import numpy as np
import torch
import torchvision.transforms.functional as F


def crop_anchors(h: int, w: int, size: int, train: bool) -> tuple[int, int]:
    if train: anchor = lambda x: np.random.randint(0, x - size) if ((x - size) > 0) else 0
    else: anchor = lambda x: (x // 2 - size // 2) if ((x // 2 - size // 2) > 0) else 0
    return anchor(h), anchor(w)


class SketchDataset(Dataset):
    def __init__(self, path: str, process_size: int, crop_size: int, train: bool, jobs: int) -> None:
        super().__init__()
        self.path = path
        self.process_size = process_size
        self.crop_size = crop_size
        self.train = train
        self.jobs = jobs
        self.files = sorted(list(chain(Path(self.path).rglob("*.png"), Path(self.path).rglob("*.jpg"))))
        self.cached_files = [file.with_suffix(".pt") for file in self.files]
        self.prepare()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tensor | tuple[Tensor, Tensor]:
        illustration, *sketches, _ = torch.load(self.cached_files[idx])
        # TODO: Replace by -> illustration, sketches, _ = torch.load(self.cached_files[idx])
        sketch = sketches[np.random.choice(len(sketches)) if self.train else 1]
        
        row, col = crop_anchors(*illustration.shape[-2:], self.crop_size, self.train)
        prep = lambda x: 2.0 * x[:, row:row + self.crop_size, col:col + self.crop_size].to(torch.float32) / 255.0 - 1.0
        return prep(illustration), prep(sketch)

    def prepare_file(self, idx: int) -> None:
        file, cached_file = self.files[idx], self.cached_files[idx]
        if cached_file.is_file():
            return

        img = np.array(F.resize(Image.open(file).convert("RGB"), self.process_size))
        illustration = F.resize(torch.from_numpy(img.astype(np.uint8)).permute(2, 0, 1), self.crop_size)
        sketches = [F.resize(torch.from_numpy(sketch(img, s, 4.5, 0.95, -1.0, 10e9, 2).astype(np.uint8)).unsqueeze(0), self.crop_size) for s in [0.3, 0.4, 0.5]]
        smoothed = F.resize(torch.from_numpy(l0_smoothing(img).astype(np.uint8)).permute(2, 0, 1), self.crop_size)
        torch.save([illustration, sketches, smoothed], cached_file)

    def prepare(self) -> None:
        if len(list(filter(lambda p: p.is_file(), self.cached_files))) != len(self.cached_files):
            for idx in tqdm(range(len(self)), desc="Preparing Dataset"): self.prepare_file(idx)


class LatentSketchDataset(Dataset):
    def __init__(self, path: str, illustration_encoder: Encoder, lineart_encoder: Encoder, hint_method: HintMethod, train: bool, jobs: int) -> None:
        super().__init__()
        self.path = path
        self.illustration_encoder = illustration_encoder
        self.lineart_encoder = lineart_encoder
        self.hint_method = hint_method
        self.train = train
        self.jobs = jobs
        self.files = sorted(list(chain(Path(self.path).rglob("*.png"), Path(self.path).rglob("*.jpg"))))
        self.cached_files = [file.with_suffix(".pt") for file in self.files]
        self.encoded_files = [cached_file.with_suffix(".enc.pt") for cached_file in self.cached_files]
        self.prepare()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        param_xs, param_ls, smoothed = torch.load(self.encoded_files[idx])
        param_x = param_xs[np.random.choice(len(param_xs)) if self.train else 1]
        param_l = param_ls[np.random.choice(len(param_ls)) if self.train else 1]

        crop_size = min(*param_x.shape[-2:])
        row, col = crop_anchors(*param_x.shape[-2:], crop_size, self.train)
        crop = lambda x: x[:, row:row + crop_size, col:col + crop_size]
        param_x, param_l, smoothed = map(crop, (param_x, param_l, smoothed))
        
        z_x = DiagonalGaussianDistribution(param_x.unsqueeze(0)).sample().squeeze(0) if self.train else param_x.chunk(2, dim=0)[0]
        z_l = DiagonalGaussianDistribution(param_l.unsqueeze(0)).sample().squeeze(0) if self.train else param_l.chunk(2, dim=0)[0]
        h = self.hint_method(smoothed)
        return z_x, z_l, h

    @torch.inference_mode()
    def prepare_file(self, idx: int) -> None:
        cached_file, encoded_file = self.cached_files[idx], self.encoded_files[idx]
        if encoded_file.is_file():
            return

        device, dtype = next(self.illustration_encoder.parameters()).device, next(self.illustration_encoder.parameters()).dtype
        prep = lambda x: 2.0 * x.to(dtype) / 255.0 - 1.0
        
        illustration, *sketches, smoothed = torch.load(cached_file)
        # TODO: Replace by -> illustration, sketches, smoothed = torch.load(cached_file)
        ls = [prep(sketch) for sketch in sketches]
        xs = [torch.cat((prep(illustration), l), dim=0) for l in ls]
        param_xs = [self.illustration_encoder(x.unsqueeze(0).to(device=device)).squeeze(0).cpu() for x in xs]
        param_ls = [self.lineart_encoder(l.unsqueeze(0).to(device=device)).squeeze(0).cpu() for l in ls]
        torch.save([param_xs, param_ls, F.resize(prep(smoothed), param_xs[0].shape[-2:])], encoded_file)

    def prepare(self) -> None:
        if len(list(filter(lambda p: p.is_file(), self.encoded_files))) != len(self.encoded_files):
            for idx in tqdm(range(len(self)), desc="Preparing Latent Dataset"): self.prepare_file(idx)