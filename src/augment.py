"""Per-sample image augmentation ported from onsite_training_studio (training_geometry.py / training.py).

Every augmentation is switched on independently per sample with its own probability (p_*), so each sample gets a
random subset. All geometric ones (translate, scale, rotate, shear) are sampled first and merged into ONE affine matrix
(T @ S @ R about the image centre) warped once; then brightness (gain + bias), additive Gaussian noise and blur (Gaussian
or median, odd kernel) follow, in that order. Every range lives in the `augment` section of the config.
The discrete 90-degree rotations and flips of the classifier are applied before this (cls_data.Aug).
"""
import cv2
import numpy as np
import torch

DEFAULT = {
    "p_brightness": 0.55, "p_translate": 0.55, "p_noise": 0.45, "p_blur": 0.35,
    "p_rotate": 0.45, "p_shear": 0.30, "p_scale": 0.55,
    "brightness": {"gain_min": 0.75, "gain_max": 1.25, "bias_min": -18, "bias_max": 18},
    "translate": {"frac_x": 0.10, "frac_y": 0.10},
    "rotate": {"deg_min": -12, "deg_max": 12},
    "scale": {"min": 0.85, "max": 1.20},
    "shear": {"shx_min": -0.08, "shx_max": 0.08, "shy_min": -0.05, "shy_max": 0.05},
    "noise": {"sigma_min": 3.0, "sigma_max": 18.0},
    "blur": {"kernels": [3, 5, 7]},
    "border": "reflect",  # what fills the corners a warp uncovers: reflect (as the original) | grey
}
BORDER = {"reflect": cv2.BORDER_REFLECT_101, "grey": cv2.BORDER_CONSTANT}


def merged(cfg):
    """DEFAULT overridden by cfg, one level deep (a partial sub-dict keeps the other defaults)."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT.items()}
    for k, v in (cfg or {}).items():
        out[k] = {**out[k], **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def build_affine(w, h, rng, do_translate, do_rotate, do_shear, do_scale, c):
    """2x3 matrix: translate @ shear @ rotate-scale, about the image centre."""
    cx, cy = w / 2.0, h / 2.0
    angle = float(rng.uniform(c["rotate"]["deg_min"], c["rotate"]["deg_max"])) if do_rotate else 0.0
    scale = float(rng.uniform(c["scale"]["min"], c["scale"]["max"])) if do_scale else 1.0
    tx = float(rng.uniform(-c["translate"]["frac_x"], c["translate"]["frac_x"]) * w) if do_translate else 0.0
    ty = float(rng.uniform(-c["translate"]["frac_y"], c["translate"]["frac_y"]) * h) if do_translate else 0.0
    shx = float(rng.uniform(c["shear"]["shx_min"], c["shear"]["shx_max"])) if do_shear else 0.0
    shy = float(rng.uniform(c["shear"]["shy_min"], c["shear"]["shy_max"])) if do_shear else 0.0
    r = np.vstack([cv2.getRotationMatrix2D((cx, cy), angle, scale), [0, 0, 1]])
    s = np.array([[1, shx, -shx * cy], [shy, 1, -shy * cx], [0, 0, 1]], np.float32)
    t = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], np.float32)
    return (t @ s @ r)[:2].astype(np.float32)


def brightness(img, rng, c):
    gain, bias = rng.uniform(c["gain_min"], c["gain_max"]), rng.uniform(c["bias_min"], c["bias_max"])
    return np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


def noise(img, rng, c):
    sigma = rng.uniform(c["sigma_min"], c["sigma_max"])
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def blur(img, rng, c):
    k = int(rng.choice(c["kernels"]))
    return cv2.GaussianBlur(img, (k, k), 0) if rng.random() < 0.5 else cv2.medianBlur(img, k)


class Augmenter:
    """Callable: uint8 RGB HWC array -> augmented uint8 array. One RNG per process (seeded from the torch worker seed)."""

    def __init__(self, cfg=None):
        self.c = merged(cfg)
        self.rng = None

    def _rng(self):
        if self.rng is None:  # created lazily so every DataLoader worker gets its own stream
            self.rng = np.random.default_rng(torch.initial_seed() % 2**32)
        return self.rng

    def __call__(self, img):
        c, rng = self.c, self._rng()
        on = {k: rng.random() < c[f"p_{k}"] for k in ("brightness", "translate", "noise", "blur", "rotate", "shear", "scale")}
        if on["translate"] or on["rotate"] or on["shear"] or on["scale"]:
            h, w = img.shape[:2]
            m = build_affine(w, h, rng, on["translate"], on["rotate"], on["shear"], on["scale"], c)
            img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=BORDER[c["border"]],
                                 borderValue=(114, 114, 114))
        if on["brightness"]:
            img = brightness(img, rng, c["brightness"])
        if on["noise"]:
            img = noise(img, rng, c["noise"])
        if on["blur"]:
            img = blur(img, rng, c["blur"])
        return img
