"""The YAML is the only place training settings live: check its shape and how the scripts read it."""
import inspect

import pytest

import common
import train_classifier
import train_yolo

YOLO_KEYS = {"name", "weights", "device", "epochs", "batch", "imgsz", "patience", "workers", "cache", "fraction", "train"}
SPECIES_KEYS = {"name", "pretrained", "device", "seed", "epochs", "warmup_epochs", "batch", "imgsz", "patience",
                "workers", "lr_head", "lr_backbone", "lr_final", "weight_decay", "focal_gamma", "class_weight_beta",
                "tta", "limit", "augment", "focus", "data", "crop_margin", "arch"}


def test_config_file_location_and_sections():
    assert common.CONFIG_FILE == common.ROOT / "config" / "config_training.yaml"
    cfg = common.load_config()
    assert set(cfg) == {"paths", "run", "split", "full", "obb", "species", "telegram", "sweep", "compare"}
    assert set(cfg["run"]["tasks"]) <= {"full", "obb", "species"} and cfg["run"]["tasks"]
    assert sum(cfg["split"]["ratios"].values()) == pytest.approx(1.0)
    assert list(cfg["split"]["ratios"]) == ["train", "val", "test"]


def test_paths_resolve_from_config():
    p = common.load_config()["paths"]
    assert common._path(p["runs"]).is_absolute() and common._path("x") == common.ROOT / "x"
    assert common._path("/abs/x").as_posix() == "/abs/x"
    assert common.SPLIT_FILE.name == "split.json"


@pytest.mark.parametrize("task", ["full", "obb"])
def test_yolo_sections_are_complete_and_numeric(task):
    c = common.load_config()[task]
    assert YOLO_KEYS <= set(c)
    assert isinstance(c["train"]["lr0"], float) and isinstance(c["train"]["weight_decay"], float)
    assert c["train"]["lr0"] * c["train"]["lrf"] == pytest.approx(1e-6)
    assert c["train"]["optimizer"] == "AdamW" and c["train"]["cos_lr"] is True
    reserved = {"data", "epochs", "batch", "imgsz", "device", "workers", "patience", "cache", "fraction",
                "project", "name", "exist_ok"}
    assert not reserved & set(c["train"]), "these keys belong to the task level, not train:"


def test_run_names_differ_and_classes_come_from_the_input_folder():
    cfg = common.load_config()
    assert "single_class" not in cfg["obb"] and "single_class" not in cfg["full"]
    assert len({cfg[t]["name"] for t in ("full", "obb", "species")}) == 3


def test_species_section_is_complete_and_numeric():
    c = common.load_config()["species"]
    assert set(c) == SPECIES_KEYS
    for k in ("lr_head", "lr_backbone", "lr_final", "weight_decay", "focal_gamma", "class_weight_beta"):
        assert isinstance(c[k], float), k
    assert c["lr_final"] < c["lr_backbone"] <= c["lr_head"]
    assert c["augment"]["scale"]["min"] < c["augment"]["scale"]["max"]
    assert 0 <= c["augment"]["p_rotate"] <= 1 and c["augment"]["border"] in ("reflect", "grey")


def test_cli_overrides_only_replace_given_values():
    y = train_yolo.settings("obb", {"epochs": 3, "name": None, "device": None})
    assert y["epochs"] == 3 and y["name"] == common.CFG["obb"]["name"]
    assert common.CFG["obb"]["epochs"] != 3  # config itself is not mutated
    s = train_classifier.settings({"limit": 8, "epochs": None})
    assert s["limit"] == 8 and s["epochs"] == common.CFG["species"]["epochs"]


def test_scripts_hold_no_hard_coded_hyperparameters():
    for mod in (train_yolo, train_classifier):
        src = inspect.getsource(mod.main)
        for literal in ("1e-4", "1e-5", "0.05", "lr0=", "default=80", "default=16", "default=640"):
            assert literal not in src, (mod.__name__, literal)
