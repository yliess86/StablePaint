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

import numpy as np
import torch
import torchvision.transforms.functional as F


def crop_anchors(h: int, w: int, size: int, train: bool) -> tuple[int, int]:
    if train: anchor = lambda x: np.random.randint(0, x - size) if ((x - size) > 0) else 0
    else: anchor = lambda x: (x // 2 - size // 2) if ((x // 2 - size // 2) > 0) else 0
    return anchor(h), anchor(w)


class SketchDataset(Dataset):
    def __init__(self, path: str, process_size: int, crop_size: int, hint_method: HintMethod, train: bool, jobs: int) -> None:
        super().__init__()
        self.path = path
        self.process_size = process_size
        self.crop_size = crop_size
        self.hint_method = hint_method
        self.train = train
        self.jobs = jobs
        self.files = sorted(list(chain(Path(self.path).rglob("*.png"), Path(self.path).rglob("*.jpg"))))
        self.cached_files = [file.with_suffix(".pt") for file in self.files]
        self.prepare()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        illustration, sketches, smoothed = torch.load(self.cached_files[idx])
        sketch = sketches[np.random.choice(len(sketches)) if self.train else 1]
        
        row, col = crop_anchors(*illustration.shape[-2:], self.crop_size, self.train)
        prep = lambda x: 2.0 * x[:, row:row + self.crop_size, col:col + self.crop_size].to(torch.float32) / 255.0 - 1.0
        interpolate = lambda x, scale_factor: torch.nn.functional.interpolate(x[None, ...], scale_factor=scale_factor)[0]

        x, l, s = map(prep, (illustration, sketch, smoothed))
        h = interpolate(self.hint_method(interpolate(s, 1 / 4)), 4)
        return x, l, h

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