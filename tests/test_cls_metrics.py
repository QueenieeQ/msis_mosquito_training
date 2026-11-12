import pytest
import torch
import torch.nn.functional as F

from cls_metrics import class_weights, fit_temperature, focal, lr_at, macro_ap


def test_class_weights_favour_rare_classes():
    y = torch.zeros(200, 3)
    y[:150, 0] = 1
    y[:10, 1] = 1  # class 2 never appears
    w = class_weights(y)
    assert w.mean() == pytest.approx(1.0)
    assert w[0] < w[1] < w[2]
    assert w[1] / w[0] == pytest.approx((1 - 0.95 ** 150) / (1 - 0.95 ** 10), rel=1e-4)


def test_focal_gamma_zero_is_weighted_bce():
    torch.manual_seed(0)
    logits, y = torch.randn(8, 4), (torch.rand(8, 4) > 0.5).float()
    w = torch.tensor([1.0, 2.0, 0.5, 1.5])
    expected = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).mean()
    assert focal(logits, y, w, gamma=0.0) == pytest.approx(expected.item(), rel=1e-5)


def test_focal_down_weights_easy_examples():
    y, w = torch.ones(1, 1), torch.ones(1)
    easy, hard = torch.tensor([[4.0]]), torch.tensor([[-4.0]])
    bce = lambda x: F.binary_cross_entropy_with_logits(x, y).item()  # noqa: E731
    assert focal(easy, y, w).item() / bce(easy) < 0.01
    assert focal(hard, y, w).item() / bce(hard) > 0.9
    assert focal(hard, y, w).item() > focal(easy, y, w).item()


def test_macro_ap_perfect_worst_and_skips_empty_class():
    y = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert macro_ap(y.clone(), y) == pytest.approx(1.0)
    # class 0 ranked fully inverted: AP = (1/3 + 2/4) / 2; class 1 perfect; class 2 has no positives
    s = y.clone()
    s[:, 0] = 1 - y[:, 0]
    assert macro_ap(s, y) == pytest.approx(((1 / 3 + 2 / 4) / 2 + 1.0) / 2)
    assert macro_ap(torch.rand(4, 3), torch.zeros(4, 3)) == 0


def test_lr_schedule_warmup_then_cosine_to_floor():
    m = [lr_at(e, 80, 5) for e in range(80)]
    assert m[:5] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert m[5] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(m[5:], m[6:]))
    assert m[-1] == pytest.approx(0.01)  # 1e-4 -> 1e-6
    assert lr_at(0, 1, 5) == pytest.approx(0.2) and lr_at(3, 2, 1) == pytest.approx(0.01)


def test_fit_temperature_recovers_scale():
    torch.manual_seed(0)
    clean = torch.randn(4000, 5) * 2
    y = (torch.rand_like(clean) < clean.sigmoid()).float()
    assert fit_temperature(clean * 1.7, y) == pytest.approx(1.7, abs=0.15)
    assert fit_temperature(clean, y) == pytest.approx(1.0, abs=0.1)
