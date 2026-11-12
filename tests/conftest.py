"""Fixtures: import path for the training modules and a tiny synthetic export on tmp_path."""
import json
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import common  # noqa: E402

FULL_CLASSES = ["Culex spp.", "Ae. albopictus", "No Mosq.", "Unknown"]
BODY_CLASSES = ["Culex spp._body", "Ae. albopictus_body", "No_mosq_body", "unknown_body"]
ANNOTATORS = ["a@x.kr", "b@x.kr"]


def region(from_name, label, rotation=0):
    return {"from_name": from_name, "type": "rectanglelabels", "original_width": 64, "original_height": 48,
            "value": {"x": 10, "y": 10, "width": 20, "height": 20, "rotation": rotation, "rectanglelabels": [label]}}


def record(task_id, image_path, result, who):
    return {"task_id": task_id, "image_path": image_path,
            "annotations": [
                {"result": [region("label", "Unknown")], "was_cancelled": True,
                 "updated_at": "2026-11-09T00:00:00Z", "completed_by_email": "skip@x.kr"},
                {"result": result, "was_cancelled": False,
                 "updated_at": "2026-11-08T00:00:00Z", "completed_by_email": who}]}


def write_item(root, task_dir, stem, rec, txt):
    d = root / task_dir
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "annotations").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), (40 + rec["task_id"] % 200, 90, 160)).save(d / "images" / f"{stem}.jpg")
    (d / "annotations" / f"{stem}.json").write_text(json.dumps(rec, ensure_ascii=False))
    if txt is not None:
        (d / "annotations" / f"{stem}.txt").write_text(txt)


@pytest.fixture
def export(tmp_path, monkeypatch):
    """40 photos; each has a 원본 (full boxes) and a 라벨링 (body boxes) copy like the real export."""
    root, here = tmp_path / "consolidated_dataset", tmp_path / "work"
    for i in range(40):
        cls = i % 3
        who = ANNOTATORS[i % 2]
        orig = f"member_3/원본/site/{i:04d}_({i}).jpg"
        lab = f"member_4/라벨링/site/{i:04d}.jpg"
        r_full = record(i, orig, [region("label", FULL_CLASSES[cls])], who)
        r_body = record(1000 + i, lab, [region("label_body", BODY_CLASSES[cls], rotation=30)], who)
        write_item(root, "full_body_detection", f"p1_t{i}_{i:04d}", r_full, f"{cls} 0.2 0.2 0.2 0.2\n")
        write_item(root, "species_classification", f"p1_t{i}_{i:04d}", r_full, None)
        write_item(root, "body_oriented_detection", f"p2_t{1000 + i}_{i:04d}", r_body,
                   f"{cls} 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n")
    (root / "full_body_detection" / "classes.txt").write_text("\n".join(FULL_CLASSES) + "\n")
    (root / "body_oriented_detection" / "classes.txt").write_text("\n".join(BODY_CLASSES) + "\n")
    monkeypatch.setattr(common, "LABELSTUDIO", root)
    monkeypatch.setattr(common, "YOLO_DATASETS", here / "datasets")
    monkeypatch.setattr(common, "RUNS", here / "runs")
    monkeypatch.setattr(common, "SPLIT_FILE", here / "splits" / "split.json")
    return root


@pytest.fixture
def yolo_input(tmp_path, monkeypatch):
    """A small input folder: 8 train / 4 val / 4 test images per task, 3 species, class i%3 (OBB: one class `body`)."""
    root = tmp_path / "input"
    names = {"full_body_detection": FULL_CLASSES[:3], "body_oriented_detection": ["body"]}
    for task, cls_names in names.items():
        (root / task).mkdir(parents=True)
        (root / task / "data.yaml").write_text(yaml.safe_dump({"names": dict(enumerate(cls_names))}))
        n = 0
        for split, count in (("train", 8), ("val", 4), ("test", 4)):
            for sub in ("images", "labels"):
                (root / task / sub / split).mkdir(parents=True)
            for i in range(count):
                stem = f"{split}_{i}"
                Image.new("RGB", (64, 48), (40 + 10 * n, 90, 160)).save(root / task / "images" / split / f"{stem}.jpg")
                cls = n % 3 if task == "full_body_detection" else 0
                box = "0.5 0.5 0.4 0.4" if task == "full_body_detection" else "0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7"
                (root / task / "labels" / split / f"{stem}.txt").write_text(f"{cls} {box}\n")
                n += 1
    monkeypatch.setattr(common, "INPUT", root)
    monkeypatch.setattr(common, "YOLO_DATASETS", tmp_path / "gen")
    monkeypatch.setattr(common, "RUNS", tmp_path / "runs")
    return root


real_data = pytest.mark.skipif(not (common.LABELSTUDIO / "full_body_detection" / "classes.txt").exists(),
                               reason="consolidated Label Studio export not present")
