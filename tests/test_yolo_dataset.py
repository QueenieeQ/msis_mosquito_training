import pytest
import yaml

import common
import dataset
import import_labelstudio
import make_splits
import yolo_dataset
from conftest import BODY_CLASSES, FULL_CLASSES


def test_build_points_the_data_yaml_at_the_input_folder(yolo_input):
    for task, task_dir, names in (("full", "full_body_detection", FULL_CLASSES[:3]), ("obb", "body_oriented_detection", ["body"])):
        yml = yolo_dataset.build(task)
        cfg = yaml.safe_load(yml.read_text())
        assert cfg["path"] == str((yolo_input / task_dir).resolve()) and cfg["train"] == "images/train"
        assert cfg["names"] == dict(enumerate(names))


def test_check_counts_images_and_boxes(yolo_input):
    res = dataset.check("full_body_detection")
    assert {s: r["images"] for s, r in res.items()} == {"train": 8, "val": 4, "test": 4}
    assert sum(res["train"]["boxes"].values()) == 8 and set(res["test"]["boxes"]) <= set(FULL_CLASSES[:3])
    assert "Culex spp." in dataset.summary("full_body_detection")


def test_adding_an_image_is_picked_up_without_any_rebuild(yolo_input):
    root = yolo_input / "full_body_detection"
    (root / "images" / "train" / "new.png").write_bytes((root / "images" / "train" / "train_0.jpg").read_bytes())
    (root / "labels" / "train" / "new.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    assert dataset.check("full_body_detection")["train"]["images"] == 9


@pytest.mark.parametrize("change,message", [
    (lambda r: (r / "labels" / "train" / "train_0.txt").write_text("7 0.5 0.5 0.2 0.2\n"), "class id 7"),
    (lambda r: (r / "labels" / "train" / "train_0.txt").write_text("0 0.5 0.5\n"), "expected 5 numbers"),
    (lambda r: [p.unlink() for p in (r / "images" / "val").iterdir()], "images/val has no images"),
    (lambda r: (r / "data.yaml").unlink(), "missing"),
])
def test_check_reports_what_to_fix(yolo_input, change, message):
    change(yolo_input / "full_body_detection")
    with pytest.raises(dataset.DatasetError, match=message):
        dataset.check("full_body_detection")


def test_image_without_label_file_counts_as_no_objects(yolo_input):
    (yolo_input / "full_body_detection" / "labels" / "train" / "train_0.txt").unlink()
    assert dataset.check("full_body_detection")["train"]["images"] == 8
    assert dataset.read_boxes(yolo_input / "full_body_detection" / "labels" / "train" / "train_0.txt") == []


def test_import_labelstudio_writes_the_input_layout(export, tmp_path):
    out = tmp_path / "imported"
    split, _ = make_splits.make()
    for t in import_labelstudio.TASKS:
        import_labelstudio.export_task(t, split, out)
    for t in import_labelstudio.TASKS:
        imgs = {s: {p.stem for p in (out / t / "images" / s).iterdir()} for s in dataset.SPLITS}
        labs = {s: {p.stem for p in (out / t / "labels" / s).iterdir()} for s in dataset.SPLITS}
        assert imgs == labs and sum(len(v) for v in imgs.values()) == 40
        recs = common.load_records(t)
        assert all(split[recs[s]["image_path"]] == sp for sp, ss in imgs.items() for s in ss)
    assert yaml.safe_load((out / "full_body_detection" / "data.yaml").read_text())["names"] == dict(enumerate(FULL_CLASSES))
    obb = out / "body_oriented_detection"
    assert yaml.safe_load((obb / "data.yaml").read_text())["names"] == {0: "body"}
    assert {ln.split()[0] for f in (obb / "labels").rglob("*.txt") for ln in f.read_text().splitlines()} == {"0"}
    assert not any(f.is_symlink() for f in (obb / "labels").rglob("*.txt"))
    assert all(f.is_symlink() for f in (obb / "images").rglob("*.jpg"))
