"""Species multi-label dataset (whole images): letterboxed images cached in RAM, GPU-free tensor augmentation."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

import dataset

Image.MAX_IMAGE_PIXELS = None


def letterbox(path, size):
    im = Image.open(path)
    im.draft("RGB", (size, size))  # JPEG DCT downscale: much faster for 9000 px sources
    im = im.convert("RGB")
    s = size / max(im.size)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1)  # uint8 CHW


def load_items(size):
    """{split: (uint8 images [N,3,S,S], multi-hot [N,C], stems)}, class names; one sample per whole image."""
    task = "full_body_detection"
    names = dataset.class_names(task)
    dataset.check(task)
    out = {}
    with ThreadPoolExecutor(8) as ex:
        for split in dataset.SPLITS:
            pairs = dataset.items(task, split)
            y = torch.zeros(len(pairs), len(names))
            for r, (_, lab) in enumerate(pairs):
                for c, _ in dataset.read_boxes(lab):
                    y[r, c] = 1
            imgs = list(ex.map(lambda pair: letterbox(pair[0], size), pairs))
            out[split] = (torch.stack(imgs), y, [p.stem for p, _ in pairs])
    return out, names


DEFAULT_AUG = {"translate": 0.1, "scale": [0.8, 1.2], "brightness": 0.4, "saturation": 0.7, "hue": 0.015}


class Aug:
    """Guideline augmentation: rot90 k, flips p=.5, random scale / translate, mild HSV jitter."""

    def __init__(self, mean, std, cfg=None):
        c = {**DEFAULT_AUG, **(cfg or {})}
        self.t = v2.Compose([
            v2.RandomAffine(degrees=0, translate=(c["translate"], c["translate"]), scale=tuple(c["scale"]), fill=114),
            v2.ColorJitter(brightness=c["brightness"], saturation=c["saturation"], hue=c["hue"]),
        ])
        self.mean, self.std = mean, std

    def __call__(self, x):
        x = torch.rot90(x, int(torch.randint(0, 4, ())), (1, 2))
        if torch.rand(()) < 0.5:
            x = x.flip(2)
        if torch.rand(()) < 0.5:
            x = x.flip(1)
        return (self.t(x).float() / 255 - self.mean) / self.std


class SpeciesSet(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, mean, std, train, aug=None):
        self.x, self.y, self.train = imgs, labels, train
        self.mean, self.std = mean, std
        self.aug = Aug(mean, std, aug)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.aug(self.x[i]) if self.train else (self.x[i].float() / 255 - self.mean) / self.std
        return x, self.y[i]


def train_stats(imgs):
    """Per-channel mean/std over the training split only, shaped [3,1,1]."""
    f = imgs.float() / 255
    return f.mean((0, 2, 3)).view(3, 1, 1), f.std((0, 2, 3)).view(3, 1, 1)
