"""Train Task A (full-body HBB, YOLOv8) or Task B (oriented body boxes, YOLOv8-OBB).

Every setting comes from the `full` / `obb` section of config/config_training.yaml; the flags below
only override it for quick experiments. OBB uses Ultralytics' probabilistic-IoU loss, HBB uses CIoU.
Validation runs every epoch. Ultralytics has no 90-degree rotation augmentation: flips cover 0/180.
"""
import argparse
import json

from ultralytics import YOLO

import common
from yolo_dataset import build

OVERRIDES = {"epochs": int, "batch": int, "device": str, "name": str, "fraction": float}


def settings(task, overrides=None):
    """Task config with non-None CLI overrides applied."""
    cfg = dict(common.CFG[task])
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["full", "obb"], required=True)
    for key, typ in OVERRIDES.items():
        ap.add_argument(f"--{key}", type=typ, help=f"override {key} of the config")
    a = ap.parse_args()
    cfg = settings(a.task, {k: getattr(a, k) for k in OVERRIDES})

    data, name, device = build(a.task), cfg["name"], str(cfg["device"])
    model = YOLO(str(common._path(cfg["weights"])))
    model.train(
        data=str(data), epochs=cfg["epochs"], batch=cfg["batch"], imgsz=cfg["imgsz"], device=device,
        workers=cfg["workers"], patience=cfg["patience"], cache=cfg["cache"], fraction=cfg["fraction"],
        project=str(common.RUNS), name=name, exist_ok=True, **cfg["train"],
    )
    best = YOLO(str(common.RUNS / name / "weights" / "best.pt"))
    m = best.val(data=str(data), split="test", imgsz=cfg["imgsz"], device=device.split(",")[0],
                 project=str(common.RUNS), name=f"{name}_test", exist_ok=True)
    res = {"map50": float(m.box.map50), "map50_95": float(m.box.map)}
    (common.RUNS / name / "test_metrics.json").write_text(json.dumps(res, indent=1))
    print("TEST", res)


if __name__ == "__main__":
    main()
