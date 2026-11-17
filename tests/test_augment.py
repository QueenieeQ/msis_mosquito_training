import numpy as np
import pytest
import torch

import augment
import cls_data


def img(seed=0, size=64):
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


def test_merged_keeps_defaults_for_a_partial_subdict():
    c = augment.merged({"rotate": {"deg_max": 30}, "p_blur": 0.0})
    assert c["rotate"] == {"deg_min": -12, "deg_max": 30} and c["p_blur"] == 0.0 and c["noise"]["sigma_min"] == 3.0


def test_all_probabilities_zero_is_identity():
    cfg = {f"p_{k}": 0.0 for k in ("brightness", "translate", "noise", "blur", "rotate", "shear", "scale")}
    x = img()
    assert np.array_equal(augment.Augmenter(cfg)(x), x)


def test_all_probabilities_one_changes_the_image_and_keeps_shape_and_dtype():
    cfg = {f"p_{k}": 1.0 for k in ("brightness", "translate", "noise", "blur", "rotate", "shear", "scale")}
    x = img()
    y = augment.Augmenter(cfg)(x)
    assert y.shape == x.shape and y.dtype == np.uint8 and not np.array_equal(x, y)


def test_affine_is_identity_without_any_geometric_aug_and_scales_about_the_centre():
    rng, c = np.random.default_rng(0), augment.merged(None)
    assert np.allclose(augment.build_affine(64, 64, rng, False, False, False, False, c), [[1, 0, 0], [0, 1, 0]], atol=1e-6)
    c["scale"] = {"min": 2.0, "max": 2.0}
    m = augment.build_affine(64, 64, rng, False, False, False, True, c)
    assert np.allclose(m @ [32, 32, 1], [32, 32], atol=1e-4) and np.allclose(m @ [48, 32, 1], [64, 32], atol=1e-4)


def test_brightness_noise_and_blur_stay_in_range_and_follow_their_settings():
    rng, x = np.random.default_rng(1), img()
    b = augment.brightness(x, rng, {"gain_min": 1.0, "gain_max": 1.0, "bias_min": 10, "bias_max": 10})
    assert np.array_equal(b, np.clip(x.astype(int) + 10, 0, 255))
    n = augment.noise(x, rng, {"sigma_min": 10, "sigma_max": 10})
    assert 5 < np.abs(n.astype(int) - x).mean() < 15
    assert augment.blur(x, rng, {"kernels": [7]}).std() < x.std()


def test_grey_border_fills_uncovered_corners_with_114():
    cfg = {**{f"p_{k}": 0.0 for k in ("brightness", "translate", "noise", "blur", "rotate", "shear")}, "p_scale": 1.0,
           "scale": {"min": 0.5, "max": 0.5}, "border": "grey"}
    y = augment.Augmenter(cfg)(np.full((64, 64, 3), 200, np.uint8))
    assert (y[0, 0] == 114).all() and (y[32, 32] == 200).all()


def test_training_aug_returns_a_normalised_tensor_and_is_random():
    x = torch.from_numpy(img()).permute(2, 0, 1).contiguous()
    mean, std = torch.full((3, 1, 1), 0.5), torch.full((3, 1, 1), 0.25)
    a = cls_data.Aug(mean, std)
    out = [a(x) for _ in range(6)]
    assert out[0].shape == (3, 64, 64) and out[0].dtype == torch.float32 and torch.isfinite(out[0]).all()
    assert any(not torch.equal(out[0], o) for o in out[1:])
