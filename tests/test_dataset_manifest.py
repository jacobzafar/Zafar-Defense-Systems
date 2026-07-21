import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.datasets.manifest import (
    DATASET_REGISTRY,
    DatasetManifest,
    LicenseStatus,
    get_manifest,
    list_datasets,
)

EXPECTED_DATASETS = {"anti-uav", "dut-anti-uav", "drone-vs-bird", "visiodect"}


def test_registry_contains_all_named_public_datasets():
    assert EXPECTED_DATASETS <= set(DATASET_REGISTRY)


def test_list_datasets_matches_registry():
    assert set(list_datasets()) == set(DATASET_REGISTRY)


@pytest.mark.parametrize("name", sorted(EXPECTED_DATASETS))
def test_every_entry_has_required_fields(name):
    manifest = get_manifest(name)
    assert isinstance(manifest, DatasetManifest)
    assert manifest.name
    assert manifest.description
    assert manifest.classes
    assert isinstance(manifest.license, LicenseStatus)
    assert manifest.license_notes


@pytest.mark.parametrize("name", sorted(EXPECTED_DATASETS))
def test_no_entry_ships_a_local_path_or_fabricated_url(name):
    """Infrastructure only: nothing here should claim data is already
    present or point at a URL nobody has verified."""
    manifest = get_manifest(name)
    assert manifest.local_path is None
    assert manifest.source_url is None


@pytest.mark.parametrize("name", sorted(EXPECTED_DATASETS))
def test_unverified_entries_are_not_silently_marked_safe(name):
    """None of these have been confirmed by a maintainer yet — the
    registry must not claim otherwise."""
    manifest = get_manifest(name)
    assert manifest.license is LicenseStatus.UNVERIFIED


def test_is_available_is_false_without_a_local_path():
    manifest = get_manifest("anti-uav")
    assert manifest.is_available() is False


def test_get_manifest_unknown_name_raises():
    with pytest.raises(KeyError, match="Unknown dataset"):
        get_manifest("not-a-real-dataset")


def test_drone_vs_bird_documents_bird_as_hard_negative_source():
    manifest = get_manifest("drone-vs-bird")
    assert "bird" in manifest.classes
