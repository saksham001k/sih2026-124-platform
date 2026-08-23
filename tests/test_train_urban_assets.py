import pytest

from train_urban_assets import (
    REQUIRED_CLASSES,
    dataset_class_names,
    validate_dataset_classes,
)


def test_accepts_list_and_index_mapping_class_names() -> None:
    names = sorted(REQUIRED_CLASSES)
    assert dataset_class_names({"names": names}) == names
    assert dataset_class_names({"names": {str(i): name for i, name in enumerate(names)}}) == names


def test_rejects_missing_or_duplicate_classes() -> None:
    with pytest.raises(ValueError, match="missing"):
        validate_dataset_classes(["waterlogging"])
    duplicate = sorted(REQUIRED_CLASSES) + ["waterlogging"]
    with pytest.raises(ValueError, match="unique"):
        validate_dataset_classes(duplicate)
