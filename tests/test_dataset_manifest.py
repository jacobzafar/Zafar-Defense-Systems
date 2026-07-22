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
    assert manifest.license_id
    assert manifest.commercial_ok in (True, False, None)


# dut-anti-uav is the one entry with a verified source_url and a
# local_path set per-split by scripts/convert_dut_anti_uav.py — every
# other entry remains a pure infrastructure placeholder.
_STILL_PLACEHOLDER_DATASETS = sorted(EXPECTED_DATASETS - {"dut-anti-uav"})


@pytest.mark.parametrize("name", _STILL_PLACEHOLDER_DATASETS)
def test_no_entry_ships_a_local_path_or_fabricated_url(name):
    """Infrastructure only: nothing here should claim data is already
    present or point at a URL nobody has verified."""
    manifest = get_manifest(name)
    assert manifest.local_path is None
    assert manifest.source_url is None


def test_dut_anti_uav_source_url_is_the_verified_official_repo():
    """The one entry with a real source_url — verified live against the
    dataset's official GitHub repo, not guessed."""
    manifest = get_manifest("dut-anti-uav")
    assert manifest.source_url == "https://github.com/wangdongdut/DUT-Anti-UAV"


@pytest.mark.parametrize("name", sorted(EXPECTED_DATASETS))
def test_unverified_entries_are_not_silently_marked_safe(name):
    """None of these have been confirmed by a maintainer yet — the
    registry must not claim otherwise."""
    manifest = get_manifest(name)
    assert manifest.license is LicenseStatus.UNVERIFIED


@pytest.mark.parametrize("name", sorted(EXPECTED_DATASETS))
def test_commercial_ok_defaults_to_unknown_not_a_guess(name):
    """`commercial_ok` must never be True/False without an explicit,
    verified license_id backing it up — every entry today is still
    pending that, dut-anti-uav included (see its license_notes)."""
    manifest = get_manifest(name)
    assert manifest.commercial_ok is None
    assert manifest.license_id == "UNVERIFIED"


def test_is_available_is_false_without_a_local_path():
    manifest = get_manifest("anti-uav")
    assert manifest.is_available() is False


def test_get_manifest_unknown_name_raises():
    with pytest.raises(KeyError, match="Unknown dataset"):
        get_manifest("not-a-real-dataset")


def test_drone_vs_bird_documents_bird_as_hard_negative_source():
    manifest = get_manifest("drone-vs-bird")
    assert "bird" in manifest.classes
