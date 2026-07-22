"""Tests for detector/datasets/dut_anti_uav.py against a tiny synthetic
fixture mimicking the real DUT Anti-UAV VOC-XML layout (img/ + xml/) —
not the real ~10,000-image dataset, which is downloaded separately (see
docs/datasets.md) and gitignored, never committed.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.datasets.dut_anti_uav import DutAnnotationFormatError, convert_split
from detector.datasets.loader import load_manifest_dataset

VOC_XML_TEMPLATE = """<annotation>
	<folder>{split}</folder>
	<filename>{stem}.jpg</filename>
	<size>
		<width>{width}</width>
		<height>{height}</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
{objects}</annotation>
"""

OBJECT_TEMPLATE = """	<object>
		<name>{name}</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{xmin}</xmin>
			<ymin>{ymin}</ymin>
			<xmax>{xmax}</xmax>
			<ymax>{ymax}</ymax>
		</bndbox>
	</object>
"""


def _write_raw_split(raw_dir: Path) -> None:
    """Build a 4-image synthetic split mirroring real DUT quirks:
    one normal box, one small (<32x32) box, one multi-object image, and
    one degenerate (zero-height) box that must be skipped, not kept.
    """
    np = pytest.importorskip("numpy")
    pil_image = pytest.importorskip("PIL.Image")

    img_dir = raw_dir / "img"
    xml_dir = raw_dir / "xml"
    img_dir.mkdir(parents=True)
    xml_dir.mkdir(parents=True)

    cases = {
        "00001": (500, 300, [OBJECT_TEMPLATE.format(name="UAV", xmin=100, ymin=50, xmax=200, ymax=150)]),
        "00002": (500, 300, [OBJECT_TEMPLATE.format(name="UAV", xmin=10, ymin=10, xmax=30, ymax=30)]),  # small: 20x20=400px^2
        "00003": (
            500,
            300,
            [
                OBJECT_TEMPLATE.format(name="UAV", xmin=10, ymin=10, xmax=100, ymax=100),
                OBJECT_TEMPLATE.format(name="UAV", xmin=200, ymin=200, xmax=260, ymax=260),
            ],
        ),
        "00004": (500, 300, [OBJECT_TEMPLATE.format(name="UAV", xmin=50, ymin=50, xmax=50, ymax=90)]),  # degenerate: zero width
    }

    for stem, (width, height, object_blocks) in cases.items():
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        pil_image.fromarray(arr).save(img_dir / f"{stem}.jpg")
        xml_dir.joinpath(f"{stem}.xml").write_text(
            VOC_XML_TEMPLATE.format(split=raw_dir.name, stem=stem, width=width, height=height, objects="".join(object_blocks))
        )


def test_convert_split_produces_expected_stats(tmp_path):
    raw_dir = tmp_path / "raw" / "val"
    _write_raw_split(raw_dir)

    stats = convert_split(raw_dir, tmp_path / "converted")

    assert stats.num_images == 4
    assert stats.num_drone_instances == 4  # 1 + 1 + 2 + 0 (degenerate skipped)
    assert stats.num_small_instances == 1  # only 00002's 20x20 box
    assert stats.num_degenerate_boxes_skipped == 1
    # 00004's only box is degenerate and skipped, so it has zero valid UAV
    # boxes left -- correctly counted as a hard negative, not silently kept.
    assert stats.num_hard_negative_images == 1


def test_converted_output_parses_via_the_unified_loader(tmp_path):
    raw_dir = tmp_path / "raw" / "val"
    _write_raw_split(raw_dir)
    output_dir = tmp_path / "converted"
    convert_split(raw_dir, output_dir)

    samples = load_manifest_dataset(output_dir)
    assert len(samples) == 4
    assert all(s.image_path.exists() for s in samples)

    hard_negatives = [s for s in samples if s.is_hard_negative]
    assert len(hard_negatives) == 1  # 00004: only box was degenerate, skipped

    drone_box_count = sum(1 for s in samples for b in s.boxes if b.category == "drone")
    assert drone_box_count == 4


def test_source_class_uav_is_mapped_to_drone(tmp_path):
    raw_dir = tmp_path / "raw" / "val"
    _write_raw_split(raw_dir)
    output_dir = tmp_path / "converted"
    convert_split(raw_dir, output_dir)

    annotations = json.loads((output_dir / "annotations.json").read_text())["annotations"]
    categories = {a["category"] for a in annotations}
    assert categories == {"drone"}


def test_conversion_stats_json_is_written(tmp_path):
    raw_dir = tmp_path / "raw" / "val"
    _write_raw_split(raw_dir)
    output_dir = tmp_path / "converted"
    stats = convert_split(raw_dir, output_dir)

    on_disk = json.loads((output_dir / "conversion_stats.json").read_text())
    assert on_disk == stats.as_dict()


def test_missing_img_or_xml_subdir_raises_file_not_found(tmp_path):
    raw_dir = tmp_path / "raw" / "val"
    raw_dir.mkdir(parents=True)
    (raw_dir / "img").mkdir()
    # no xml/ subdir
    with pytest.raises(FileNotFoundError, match="img.*xml"):
        convert_split(raw_dir, tmp_path / "converted")


def test_empty_xml_dir_raises_file_not_found(tmp_path):
    raw_dir = tmp_path / "raw" / "val"
    (raw_dir / "img").mkdir(parents=True)
    (raw_dir / "xml").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No .xml"):
        convert_split(raw_dir, tmp_path / "converted")


def test_malformed_xml_raises_dut_annotation_format_error(tmp_path):
    raw_dir = tmp_path / "raw" / "val"
    img_dir = raw_dir / "img"
    xml_dir = raw_dir / "xml"
    img_dir.mkdir(parents=True)
    xml_dir.mkdir(parents=True)

    np = pytest.importorskip("numpy")
    pil_image = pytest.importorskip("PIL.Image")
    pil_image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(img_dir / "00001.jpg")
    xml_dir.joinpath("00001.xml").write_text("<annotation><size><width>10</width></size></annotation>")

    with pytest.raises(DutAnnotationFormatError, match="size"):
        convert_split(raw_dir, tmp_path / "converted")


def test_missing_image_for_annotation_raises_file_not_found(tmp_path):
    raw_dir = tmp_path / "raw" / "val"
    img_dir = raw_dir / "img"
    xml_dir = raw_dir / "xml"
    img_dir.mkdir(parents=True)
    xml_dir.mkdir(parents=True)

    xml_dir.joinpath("00001.xml").write_text(
        VOC_XML_TEMPLATE.format(split="val", stem="00001", width=10, height=10, objects="")
    )
    # no matching img/00001.jpg written

    with pytest.raises(FileNotFoundError, match="00001.jpg"):
        convert_split(raw_dir, tmp_path / "converted")
