import common
from conftest import region, record


def test_group_key_pairs_original_with_labeled_copy():
    a = common.group_key("member_5/원본/1. site(08.19~09.15)/041842659901_(14).jpg")
    b = common.group_key("member_3/라벨링/1. site(08.19~09.15)/041842659901.jpg")
    assert a == b == "1. site(08.19~09.15)/041842659901"


def test_group_key_separates_folders_and_names():
    assert common.group_key("m/원본/siteA/1.jpg") != common.group_key("m/원본/siteB/1.jpg")
    assert common.group_key("m/원본/siteA/1.jpg") != common.group_key("m/원본/siteA/2.jpg")
    # only a trailing _(N) is stripped
    assert common.group_key("m/원본/s/064234_(51).jpg") == "s/064234"
    assert common.group_key("m/원본/s/06_(5)_x.jpg") == "s/06_(5)_x"


def test_primary_annotation_skips_cancelled_and_takes_latest():
    rec = record(1, "p.jpg", [region("label", "Culex spp.")], "a@x.kr")
    rec["annotations"].append({"result": [], "was_cancelled": False, "updated_at": "2026-11-10T00:00:00Z",
                               "completed_by_email": "late@x.kr"})
    assert common.primary_annotation(rec)["completed_by_email"] == "late@x.kr"


def test_canonical_species_normalises_body_and_case(export):
    canon = common.load_canon()
    assert common.canonical_species("Culex spp._body", canon) == "Culex spp."
    assert common.canonical_species("culex SPP._body", canon) == "Culex spp."
    assert common.canonical_species("No_mosq_body", canon) == "No Mosq."
    assert common.canonical_species("unknown_body", canon) == "Unknown"
    assert common.canonical_species("Brand new_body", canon) == "Brand new"


def test_species_counts_prefers_full_boxes_then_body(export):
    canon = common.load_canon()
    both = record(1, "p.jpg", [region("label", "Culex spp."), region("label", "Culex spp."),
                               region("label_body", "Ae. albopictus_body")], "a@x.kr")
    assert common.species_counts(both, canon) == {"Culex spp.": 2}
    body = record(2, "q.jpg", [region("label_body", "Ae. albopictus_body")], "a@x.kr")
    assert common.species_counts(body, canon) == {"Ae. albopictus": 1}


def test_load_records_and_image_file(export):
    recs = common.load_records("full_body_detection")
    assert len(recs) == 40
    stem = next(iter(recs))
    assert common.image_file("full_body_detection", stem).suffix == ".jpg"
    assert common.image_file("full_body_detection", "missing") is None
