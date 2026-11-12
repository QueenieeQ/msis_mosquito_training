"""Integrity checks on the real export and the committed split (skipped when the data is absent)."""
from collections import Counter, defaultdict

import pytest

import common
from conftest import real_data

pytestmark = real_data
MEMBERS = ["dogyunkim", "insungpark", "jihoon.an", "minhyuk.lee"]


@pytest.mark.parametrize("task,fields", [("full_body_detection", 5), ("body_oriented_detection", 9)])
def test_yolo_labels_are_valid(task, fields):
    n_cls = len([c for c in (common.LABELSTUDIO / task / "classes.txt").read_text().split("\n") if c])
    files = sorted((common.LABELSTUDIO / task / "annotations").glob("*.txt"))
    assert files
    for f in files:
        lines = f.read_text().splitlines()
        assert lines, f"{f.name}: empty label file"
        for line in lines:
            parts = line.split()
            assert len(parts) == fields, f.name
            assert 0 <= int(parts[0]) < n_cls, f.name
            v = [float(x) for x in parts[1:]]
            assert min(v) >= 0 and max(v) <= 1, f"{f.name}: coordinates outside the image"
            if fields == 5:
                assert v[2] > 0 and v[3] > 0, f"{f.name}: zero-size box"


@pytest.mark.parametrize("task", common.TASK_DIRS)
def test_every_record_has_image_and_primary_annotation(task):
    recs = common.load_records(task)
    assert recs
    for stem, rec in recs.items():
        assert common.image_file(task, stem) is not None, stem
        assert common.primary_annotation(rec)["result"], stem


@pytest.fixture(scope="module")
def split():
    if not common.SPLIT_FILE.exists():
        pytest.skip("run make_splits.py first")
    return common.load_split()


def test_split_covers_every_image(split):
    paths = {r["image_path"] for t in common.TASK_DIRS for r in common.load_records(t).values()}
    assert paths == set(split)
    assert set(split.values()) == {"train", "val", "test"}


def test_split_ratios(split):
    n = Counter(split.values())
    assert n["train"] / len(split) == pytest.approx(0.70, abs=0.03)
    assert n["val"] / len(split) == pytest.approx(0.15, abs=0.03)
    assert n["test"] / len(split) == pytest.approx(0.15, abs=0.03)


def test_no_photo_crosses_splits(split):
    by_group = defaultdict(set)
    for path, s in split.items():
        by_group[common.group_key(path)].add(s)
    leaks = [k for k, v in by_group.items() if len(v) > 1]
    assert not leaks, f"{len(leaks)} photos in more than one split, e.g. {leaks[:3]}"


def test_each_member_is_spread_over_all_splits(split):
    per = defaultdict(Counter)
    for task in common.TASK_DIRS:
        for rec in common.load_records(task).values():
            who = common.primary_annotation(rec).get("completed_by_email") or ""
            per[who.split("@")[0]][rec["image_path"]] = 1
    for m in MEMBERS:
        c = Counter(split[p] for p in per[m])
        assert c["train"] / sum(c.values()) == pytest.approx(0.70, abs=0.05), (m, c)
        assert c["val"] and c["test"], (m, c)


def test_real_input_folder_is_valid_when_present():
    import dataset
    if not (common.INPUT / "full_body_detection" / "data.yaml").exists():
        pytest.skip("input/datasets not present")
    for task in dataset.FIELDS:
        assert all(r["images"] for r in dataset.check(task).values())
