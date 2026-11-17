"""Task C: Swin Transformer V2 (tiny / small / base, `arch` in the config; ImageNet-1K) multi-label species classifier per the guideline.

Every setting comes from the `species` section of config/config_training.yaml; the flags below only
override it for quick experiments. AdamW with separate backbone / head LR, linear warmup + cosine,
focal loss with effective-number class weights, early stopping and best.pt selection on the val AP of the `focus` classes (all-class macro-AP if empty), temperature scaling
on val, 4-rotation TTA on test. best.pt holds everything inference needs: weights, class names,
train-split mean/std, image size and temperature.
"""
import argparse
import json

import torch
from torchvision.models import (Swin_V2_B_Weights, Swin_V2_S_Weights, Swin_V2_T_Weights, swin_v2_b, swin_v2_s,
                                swin_v2_t)

import crop_data
from cls_data import SpeciesSet, load_items, train_stats
from cls_metrics import class_weights, fit_temperature, focal, focus_ap, lr_at, macro_ap
import common


ARCHS = {"swin_v2_t": (swin_v2_t, Swin_V2_T_Weights), "swin_v2_s": (swin_v2_s, Swin_V2_S_Weights),
         "swin_v2_b": (swin_v2_b, Swin_V2_B_Weights)}  # tiny 28M / small 50M / base 88M parameters, ImageNet-1K


def build_model(n_classes, pretrained=True, arch="swin_v2_t"):
    fn, weights = ARCHS[arch]
    model = fn(weights=weights.IMAGENET1K_V1 if pretrained else None)
    model.head = torch.nn.Linear(model.head.in_features, n_classes)
    return model


def build_optimizer(model, lr_head=1e-4, lr_backbone=1e-5, weight_decay=0.05):
    head = list(model.head.parameters())
    back = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    return torch.optim.AdamW([{"params": back, "lr": lr_backbone}, {"params": head, "lr": lr_head}],
                             weight_decay=weight_decay)


def amp(dev):
    return torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")


def train_epoch(model, loader, opt, w, dev, gamma=2.0):
    model.train()
    tot = 0.0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        with amp(dev):
            out = model(x)
        loss = focal(out.float(), y, w, gamma)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
    return tot / max(len(loader), 1)


@torch.no_grad()
def predict(model, loader, dev, tta=False):
    """Logits for a loader; tta averages the 0/90/180/270 degree rotations."""
    model.eval()
    outs = []
    for x, _ in loader:
        x = x.to(dev)
        views = [torch.rot90(x, k, (2, 3)) for k in (range(4) if tta else [0])]
        with amp(dev):
            outs.append(torch.stack([model(v).float() for v in views]).mean(0).cpu())
    return torch.cat(outs)


OVERRIDES = {"arch": str, "data": str, "imgsz": int, "epochs": int, "batch": int, "device": str, "name": str, "limit": int,
             "lr_head": float, "lr_backbone": float, "weight_decay": float}


def settings(overrides=None):
    """Species config with non-None CLI overrides applied."""
    cfg = dict(common.CFG["species"])
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key, typ in OVERRIDES.items():
        ap.add_argument(f"--{key}", type=typ, help=f"override {key} of the config")
    a = ap.parse_args()
    c = settings({k: getattr(a, k) for k in OVERRIDES})
    torch.manual_seed(c["seed"])
    run = common.RUNS / c["name"]
    run.mkdir(parents=True, exist_ok=True)

    crops = c.get("data", "images") == "crops"  # per-mosquito crops (pipeline input) vs whole images
    data, names = crop_data.load_items(c["imgsz"], c["crop_margin"]) if crops else load_items(c["imgsz"])
    if c["limit"]:
        data = {k: tuple(x[:c["limit"]] for x in v) for k, v in data.items()}
    mean, std = train_stats(data["train"][0])
    sets = {k: SpeciesSet(v[0], v[1], mean, std, k == "train", c["augment"]) for k, v in data.items()}
    ld = {k: torch.utils.data.DataLoader(s, c["batch"], shuffle=k == "train", drop_last=k == "train",
                                         num_workers=c["workers"], pin_memory=True,
                                         persistent_workers=k == "train" and c["workers"] > 0)
          for k, s in sets.items()}

    dev = torch.device(c["device"])
    model = build_model(len(names), c["pretrained"], c["arch"]).to(dev)
    opt = build_optimizer(model, c["lr_head"], c["lr_backbone"], c["weight_decay"])
    base = [g["lr"] for g in opt.param_groups]
    w = class_weights(data["train"][1], c["class_weight_beta"]).to(dev)
    ckpt = {"arch": c["arch"], "classes": names, "mean": mean, "std": std, "imgsz": c["imgsz"], "temperature": 1.0,
            "input": "crop" if crops else "image"}

    focus = c.get("focus") or []
    best, bad = -1.0, 0
    for ep in range(c["epochs"]):
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * lr_at(ep, c["epochs"], c["warmup_epochs"], c["lr_final"] / c["lr_head"])
        loss = train_epoch(model, ld["train"], opt, w, dev, c["focal_gamma"])
        p_val = predict(model, ld["val"], dev).sigmoid()
        ap_val = macro_ap(p_val, data["val"][1])
        foc = focus_ap(p_val, data["val"][1], names, focus)
        score = sum(foc.values()) / len(foc) if foc else ap_val
        print(f"epoch {ep + 1}/{c['epochs']} loss {loss:.4f} val_mAP {ap_val:.4f}"
              + "".join(f" {n} {v:.3f}" for n, v in foc.items()), flush=True)
        if score > best:
            best, bad = score, 0
            torch.save({**ckpt, "model": model.state_dict()}, run / "best.pt")
        else:
            bad += 1
            if bad >= c["patience"]:
                print("early stop")
                break

    ckpt = torch.load(run / "best.pt", weights_only=True)
    model.load_state_dict(ckpt["model"])
    ckpt["temperature"] = temp = fit_temperature(predict(model, ld["val"], dev), data["val"][1])
    torch.save(ckpt, run / "best.pt")
    res = {"val_select_score": best, "temperature": temp, "classes": names, "focus": focus}
    res["val_macro_ap"] = macro_ap(predict(model, ld["val"], dev).sigmoid(), data["val"][1])
    for tta in ([False, True] if c["tta"] else [False]):
        s = (predict(model, ld["test"], dev, tta) / temp).sigmoid()
        k = "tta4" if tta else "plain"
        res["test_macro_ap_" + k] = macro_ap(s, data["test"][1])
        res["test_focus_ap_" + k] = focus_ap(s, data["test"][1], names, focus)
    (run / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print({k: v for k, v in res.items() if k != "classes"})


if __name__ == "__main__":
    main()
