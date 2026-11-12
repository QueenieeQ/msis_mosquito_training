import pytest
import torch
from PIL import Image

import common
import cls_data
import train_classifier as tc
from cls_metrics import class_weights


def test_letterbox_keeps_aspect_and_pads_grey(tmp_path):
    Image.new("RGB", (200, 100), (255, 0, 0)).save(tmp_path / "a.png")
    x = cls_data.letterbox(tmp_path / "a.png", 64)
    assert x.shape == (3, 64, 64) and x.dtype == torch.uint8
    assert x[:, 0, 0].tolist() == [114, 114, 114] and x[:, 63, 63].tolist() == [114, 114, 114]
    assert x[:, 32, 32].tolist() == [255, 0, 0]
    assert (x[0] == 255).sum().item() == 64 * 32


def test_load_items_follows_folders_and_builds_multi_hot(yolo_input):
    data, names = cls_data.load_items(32)
    assert names == ["Culex spp.", "Ae. albopictus", "No Mosq."]  # data.yaml order
    assert {k: len(v[2]) for k, v in data.items()} == {"train": 8, "val": 4, "test": 4}
    for split, (imgs, y, stems) in data.items():
        assert imgs.shape == (len(stems), 3, 32, 32) and y.shape == (len(stems), 3)
        assert (y.sum(1) == 1).all() and all(s.startswith(split) for s in stems)


def test_train_stats_and_dataset_normalisation():
    torch.manual_seed(0)
    imgs = torch.randint(0, 256, (6, 3, 32, 32), dtype=torch.uint8)
    y = torch.eye(3).repeat(2, 1)
    mean, std = cls_data.train_stats(imgs)
    assert mean.shape == std.shape == (3, 1, 1)
    val = cls_data.SpeciesSet(imgs, y, mean, std, train=False)
    x = torch.stack([val[i][0] for i in range(6)])
    assert x.mean((0, 2, 3)).abs().max() < 1e-4 and (x.std((0, 2, 3)) - 1).abs().max() < 1e-3
    xt, yt = cls_data.SpeciesSet(imgs, y, mean, std, train=True)[0]
    assert xt.shape == (3, 32, 32) and xt.dtype == torch.float32 and torch.isfinite(xt).all()
    assert torch.equal(yt, y[0])


def test_model_is_swin_v2_t_with_new_head_and_two_lr_groups():
    model = tc.build_model(5, pretrained=False)
    assert type(model).__name__ == "SwinTransformer" and model.head.out_features == 5
    assert any("logit_scale" in n for n, _ in model.named_parameters())  # V2 cosine attention
    model.eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 128, 128)).shape == (1, 5)
    opt = tc.build_optimizer(model)
    back, head = opt.param_groups
    assert (back["lr"], head["lr"]) == (1e-5, 1e-4) and head["weight_decay"] == 0.05
    assert sum(p.numel() for p in head["params"]) == sum(p.numel() for p in model.head.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    assert sum(p.numel() for g in opt.param_groups for p in g["params"]) == n_all


class Tiny(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.head = torch.nn.Linear(3, n)

    def forward(self, x):
        return self.head(x.mean((2, 3)))


def loader(x, y):
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=8)


def test_train_epoch_reduces_loss_on_cpu():
    torch.manual_seed(0)
    y = torch.eye(3).repeat(8, 1)
    x = y[:, :, None, None].expand(-1, -1, 4, 4) * 3 + 0.1 * torch.randn(24, 3, 4, 4)
    model, dev = Tiny(3), torch.device("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    w = class_weights(y)
    losses = [tc.train_epoch(model, loader(x, y), opt, w, dev) for _ in range(15)]
    assert losses[-1] < 0.3 * losses[0]
    logits = tc.predict(model, loader(x, y), dev)
    assert logits.shape == (24, 3) and (logits.argmax(1) == y.argmax(1)).all()


def test_predict_tta_averages_four_rotations():
    class Corner(torch.nn.Module):
        def forward(self, x):
            return x[:, :, 0, 0]

    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).view(2, 3, 4, 4)
    ld, dev = loader(x, torch.zeros(2, 3)), torch.device("cpu")
    corners = (x[:, :, 0, 0] + x[:, :, 0, 3] + x[:, :, 3, 0] + x[:, :, 3, 3]) / 4
    assert torch.allclose(tc.predict(Corner(), ld, dev, tta=True), corners)
    assert torch.equal(tc.predict(Corner(), ld, dev), x[:, :, 0, 0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_swin_v2_t_step_under_bf16_autocast_at_640():
    dev = torch.device("cuda:0")
    model = tc.build_model(4, pretrained=False).to(dev)
    opt = tc.build_optimizer(model)
    y = torch.eye(4)[:2]
    loss = tc.train_epoch(model, [(torch.randn(2, 3, 640, 640), y)], opt, torch.ones(4, device=dev), dev)
    assert loss == loss and loss > 0


def test_crop_dataset_one_sample_per_box_and_cache_follows_data(yolo_input):
    import crop_data
    data, names = crop_data.load_items(32)
    assert {k: v[0].shape[0] for k, v in data.items()} == {"train": 8, "val": 4, "test": 4}
    assert data["train"][0].shape == (8, 3, 32, 32) and (data["train"][1].sum(1) == 1).all()
    n_cache = len(list((common.RUNS / "crop_cache").glob("*.pt")))
    assert n_cache == 3
    crop_data.load_items(32)  # unchanged data: reuses the cache
    assert len(list((common.RUNS / "crop_cache").glob("*.pt"))) == 3
    lab = yolo_input / "full_body_detection" / "labels" / "train" / "train_0.txt"
    lab.write_text("0 0.3 0.3 0.2 0.2\n1 0.7 0.7 0.2 0.2\n")  # a second box in one image
    data, _ = crop_data.load_items(32)
    assert data["train"][0].shape[0] == 9 and len(list((common.RUNS / "crop_cache").glob("*.pt"))) == 3
