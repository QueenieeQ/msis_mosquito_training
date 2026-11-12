import json
from collections import Counter, defaultdict

import pytest

import common
import make_splits


def synthetic_groups(n=600):
    species = ["a"] * 6 + ["b"] * 3 + ["c"]
    return {f"g{i}": {"species": species[i % 10], "annotator": f"u{i % 4}"} for i in range(n)}


def test_assign_ratios_and_coverage():
    groups = synthetic_groups()
    split = make_splits.assign(groups)
    assert set(split) == set(groups)
    n = Counter(split.values())
    assert n["train"] / 600 == pytest.approx(0.70, abs=0.03)
    assert n["val"] / 600 == pytest.approx(0.15, abs=0.03)
    assert n["test"] / 600 == pytest.approx(0.15, abs=0.03)


def test_assign_is_deterministic_and_seeded():
    groups = synthetic_groups()
    assert make_splits.assign(groups, 0) == make_splits.assign(dict(reversed(list(groups.items()))), 0)
    assert make_splits.assign(groups, 0) != make_splits.assign(groups, 1)


def test_assign_stratifies_species_and_annotator():
    groups = synthetic_groups()
    split = make_splits.assign(groups)
    for field in ("species", "annotator"):
        per = defaultdict(Counter)
        for k, g in groups.items():
            per[g[field]][split[k]] += 1
        for value, c in per.items():
            assert c["train"] / sum(c.values()) == pytest.approx(0.70, abs=0.06), (field, value)
            assert c["val"] and c["test"], (field, value)


def test_tiny_stratum_goes_to_train_first():
    split = make_splits.assign({"only": {"species": "rare", "annotator": "u"}})
    assert split == {"only": "train"}


def test_make_keeps_pairs_together_and_writes_files(export):
    split, report = make_splits.make()
    assert len(split) == 80 and report["groups"] == 40
    by_group = defaultdict(set)
    for path, s in split.items():
        by_group[common.group_key(path)].add(s)
    assert all(len(v) == 1 for v in by_group.values())
    assert json.loads(common.SPLIT_FILE.read_text()) == split
    assert sum(report["per_split"].values()) == 80
    assert set(split.values()) == {"train", "val", "test"}
