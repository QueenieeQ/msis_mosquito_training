"""Dropping a Label Studio export into the input folder converts it on first use."""
import json
import shutil

import pytest

import common
import dataset
import ingest


@pytest.fixture
def drop(export, tmp_path, monkeypatch):
    """Empty input folder; returns a function that copies the synthetic export in at a given sub-path."""
    inp = tmp_path / "input"
    inp.mkdir()
    monkeypatch.setattr(common, "INPUT", inp)
    monkeypatch.setattr(common, "SPLIT_FILE", tmp_path / "splits" / "split.json")
    monkeypatch.setattr(ingest, "_ready", set())
    src = common.LABELSTUDIO

    def put(sub):
        shutil.copytree(src, inp / sub)
        return inp
    return put


@pytest.mark.parametrize("sub", ["consolidated_dataset", "export_2026/consolidated_dataset"])
def test_export_dropped_in_input_is_converted_and_checked(drop, sub):
    inp = drop(sub)
    res = dataset.check("full_body_detection")
    assert sum(r["images"] for r in res.values()) == 40 and all(r["images"] for r in res.values())
    assert dataset.check("body_oriented_detection")["train"]["images"] > 0
    assert json.loads((inp / ingest.MARKER).read_text())["split"]


def test_raw_task_folders_placed_directly_in_input_are_tucked_away_and_converted(drop):
    inp = drop("tmp_export")
    for t in ingest.RAW_DIRS:
        shutil.move(str(inp / "tmp_export" / t), str(inp / t))
    (inp / "tmp_export").rmdir()
    assert sum(r["images"] for r in dataset.check("full_body_detection").values()) == 40
    assert (inp / "consolidated_dataset" / "full_body_detection" / "annotations").is_dir()


def test_same_export_is_converted_once_and_new_export_replaces_with_backup(drop):
    inp = drop("consolidated_dataset")
    dataset.check("full_body_detection")
    edited = inp / "full_body_detection" / "labels" / "train"
    next(edited.glob("*.txt")).write_text("0 0.5 0.5 0.1 0.1\n" * 2)
    ingest._ready.clear()
    dataset.check("full_body_detection")  # same export: the edit survives
    assert not (inp / "_previous").exists() and "0 0.5 0.5 0.1 0.1\n" * 2 in {f.read_text() for f in edited.glob("*.txt")}
    extra = inp / "consolidated_dataset" / "full_body_detection" / "annotations"
    one = next(extra.glob("*.json"))
    (extra / "p9_t9_extra.json").write_text(one.read_text())
    (extra / "p9_t9_extra.txt").write_text((extra / f"{one.stem}.txt").read_text())
    img = inp / "consolidated_dataset" / "full_body_detection" / "images"
    shutil.copy(next(img.glob(f"{one.stem}.*")), img / "p9_t9_extra.jpg")
    ingest._ready.clear()
    dataset.check("full_body_detection")  # changed export: reconverted, old folders kept
    assert (inp / "_previous").is_dir() and "0 0.5 0.5 0.1 0.1\n" * 2 not in {f.read_text() for f in edited.glob("*.txt")}


def test_no_export_leaves_prepared_data_untouched(drop):
    assert ingest.find_export(common.INPUT) is None
    ingest.ensure_ready()
    assert not (common.INPUT / ingest.MARKER).exists()
