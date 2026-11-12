"""Compare a freshly trained Swin V2-T species classifier (best.pt) with the old 4-class SwinV2-B weights.

Both are scored on the SAME held-out images (split from make_splits.py, default `test`), per image, on the old model's
4 classes (Anopheles / Culex tritaeniorhynchus / Non / Others): the new 14 species are collapsed onto them
(compare_classes.py) and the old model runs through its full app pipeline (compare_old.py).
Writes <run>/compare_old.json + compare_old.md and appends a row to <runs>/compare_history.csv.
run_training.sh calls this after the species task finishes; settings are the `compare` section of the config.

Usage: compare_models.py [--run NAME_OR_DIR] [--old-cls-model FILE] [--split test] [--device cuda:3] [--thr 0.5]
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch

import common
import dataset
from cls_data import SpeciesSet, letterbox
from compare_classes import COARSE, collapse, metrics, table, truth
from compare_old import run_old
from train_classifier import build_model, predict


def split_items(split):
    """[(stem, image file, species names)] of one split of the input folder's full_body_detection, sorted by stem."""
    names = dataset.class_names("full_body_detection")
    return sorted((img.stem, img, sorted({names[c] for c, _ in dataset.read_boxes(lab)}))
                  for img, lab in dataset.items("full_body_detection", split))


def new_scores(ckpt_path, images, device):
    """{'plain' | 'tta4': [N, 4] scores} and the checkpoint, scoring with the temperature stored in best.pt."""
    ck = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    model = build_model(len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["model"])
    dev = torch.device(device)
    model.to(dev)
    x = torch.stack([letterbox(p, ck["imgsz"]) for p in images])
    ds = SpeciesSet(x, torch.zeros(len(images), 1), ck["mean"], ck["std"], False)
    loader = torch.utils.data.DataLoader(ds, common.CFG["species"]["batch"], shuffle=False)
    out = {}
    for name, tta in (("plain", False), ("tta4", True)):
        out[name] = collapse((predict(model, loader, dev, tta) / ck["temperature"]).sigmoid(), ck["classes"])
    return out, ck


def main():
    cfg = common.CFG.get("compare", {})
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=common.CFG["species"]["name"], help="run name under <runs>/ or a run directory")
    ap.add_argument("--old-cls-model", nargs="+", default=[cfg.get("old_cls_model")],
                    help="old classifier(s) (.onnx or mmpretrain .pth); several are scored side by side")
    ap.add_argument("--old-infer-dir", default=str(common._path(cfg.get("old_infer_dir", "../mosquitto_msislab"))))
    ap.add_argument("--split", default=cfg.get("split", "test"), choices=["train", "val", "test"])
    ap.add_argument("--device", default=cfg.get("device") or common.CFG["species"]["device"])
    ap.add_argument("--thr", type=float, default=cfg.get("threshold", 0.5), help="decision threshold for precision/recall/F1")
    a = ap.parse_args()

    run = Path(a.run) if Path(a.run).is_dir() else common.RUNS / a.run
    items = split_items(a.split)
    images = [f for _, f, _ in items]
    y = torch.stack([truth(s) for _, _, s in items])
    print(f"{len(items)} {a.split} images; new model {run / 'best.pt'}", flush=True)

    ck = torch.load(run / "best.pt", weights_only=True, map_location="cpu")
    if ck.get("input") == "crop":  # crop classifier: only meaningful inside the detector pipeline, so score it there
        sc, _ = run_old(images, a.old_infer_dir, run / "best.pt", a.device, common.RUNS / "compare_cache", ck["classes"])
        new = {"pipeline": collapse(sc, ck["classes"])}
    else:
        new, ck = new_scores(run / "best.pt", images, a.device)
    rows, old_names = {}, []
    for cls_model in a.old_cls_model:
        old, old_path = run_old(images, a.old_infer_dir, cls_model, a.device, common.RUNS / "compare_cache")
        rows[f"old: {Path(old_path).name}"] = metrics(old, y, a.thr)
        old_names.append(Path(old_path).name)
    for k, v in new.items():
        rows[f"new: {run.name} ({k})"] = metrics(v, y, a.thr)
    md = f"Split `{a.split}`, {len(items)} images, threshold {a.thr}; classes {', '.join(COARSE)}\n\n{table(rows)}\n"
    print(md)
    (run / "compare_old.md").write_text(md, encoding="utf-8")
    (run / "compare_old.json").write_text(json.dumps({"split": a.split, "threshold": a.thr, "models": rows},
                                                     ensure_ascii=False, indent=1))
    hist = common.RUNS / "compare_history.csv"
    new_file = not hist.exists()
    with open(hist, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["time", "run", "old_model", "split", "old_macro_ap", "new_macro_ap_plain", "new_macro_ap_tta4"])
        vals = list(rows.values())
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), run.name, "+".join(old_names), a.split,
                    "/".join(f"{v['macro_ap']:.4f}" for v in vals[:len(a.old_cls_model)]),
                    *(round(v["macro_ap"], 4) for v in vals[len(a.old_cls_model):][:2])])


if __name__ == "__main__":
    main()
