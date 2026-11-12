import pytest
import torch

from compare_classes import COARSE, coarse_index, collapse, metrics, truth


def test_coarse_mapping():
    assert [coarse_index(s) for s in ("Anopheles spp.", "Culex tritaeniorhynchus", "No Mosq.")] == [0, 1, 2]
    assert [coarse_index(s) for s in ("Culex spp.", "Unknown", "Ae. albopictus")] == [3, 3, 3]


def test_truth_is_multi_hot():
    assert truth(["Culex spp.", "Ae. albopictus", "Anopheles spp."]).tolist() == [1, 0, 0, 1]
    assert truth(["No Mosq."]).tolist() == [0, 0, 1, 0]


def test_collapse_takes_max_per_coarse_class():
    classes = ["Ae. albopictus", "Anopheles spp.", "Culex spp.", "No Mosq."]
    out = collapse(torch.tensor([[0.2, 0.9, 0.7, 0.1]]), classes)
    assert out.tolist()[0] == pytest.approx([0.9, 0.0, 0.1, 0.7])


def test_metrics_perfect_and_wrong():
    y = torch.eye(4)
    perfect = metrics(y.clone(), y)
    assert perfect["macro_ap"] == perfect["macro_f1"] == perfect["exact_match"] == 1.0
    wrong = metrics(1 - y, y)
    assert wrong["macro_f1"] == 0.0 and wrong["exact_match"] == 0.0
    assert set(perfect["classes"]) == set(COARSE)


def test_metrics_skip_classes_without_positives():
    y = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    m = metrics(torch.tensor([[0.9, 0, 0, 0], [0.8, 0, 0, 0]]), y)
    assert m["macro_ap"] == 1.0 and m["classes"]["Non"]["support"] == 0
