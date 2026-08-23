"""Tests for road-hazard class normalization."""

from urban_intelligence.classes import class_matches_filter, normalize_class_name


def test_rdd2022_codes_normalize_to_friendly_names() -> None:
    assert normalize_class_name("D00") == "longitudinal_crack"
    assert normalize_class_name("d10") == "transverse_crack"
    assert normalize_class_name("D20") == "alligator_crack"
    assert normalize_class_name("D40") == "pothole"


def test_friendly_aliases_remain_unchanged() -> None:
    assert normalize_class_name("pothole") == "pothole"
    assert normalize_class_name("Longitudinal_Crack") == "longitudinal_crack"


def test_coco_classes_remain_unchanged() -> None:
    assert normalize_class_name("car") == "car"
    assert normalize_class_name("person") == "person"
    assert normalize_class_name("bus") == "bus"
    assert normalize_class_name("truck") == "truck"


def test_class_filter_accepts_friendly_names_for_raw_codes() -> None:
    allowed = {"pothole", "longitudinal_crack"}
    assert class_matches_filter("D40", allowed)
    assert class_matches_filter("D00", allowed)
    assert not class_matches_filter("D10", allowed)


def test_class_filter_keeps_coco_behavior() -> None:
    allowed = {"car", "person"}
    assert class_matches_filter("car", allowed)
    assert class_matches_filter("person", allowed)
    assert not class_matches_filter("truck", allowed)
