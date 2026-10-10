"""Tests for tools/import_reviewed.py: reviewed CVAT / Label Studio exports
back into a clip's annotations.json."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import import_reviewed  # noqa: E402
from detector.datasets.loader import load_manifest_dataset  # noqa: E402
from tests.conftest import synthetic_clip_box  # noqa: E402


def _frames(clip):
    return json.loads((clip / "frames.json").read_text())["images"]


def _cvat_coco_export(clip, path, skip_drone_on=()):
    """What CVAT's COCO 1.0 export looks like after a reviewer boxed the
    drone on every frame (except `skip_drone_on`: reviewed, no drone)."""
    images, annotations = [], []
    for f in _frames(clip):
        images.append({"id": f["id"] + 100, "file_name": f["file_name"], "width": f["width"], "height": f["height"]})
        if f["file_name"] in skip_drone_on:
            continue
        x1, y1, x2, y2 = synthetic_clip_box(f["source_frame"])
        annotations.append(
            {"id": len(annotations) + 1, "image_id": f["id"] + 100, "category_id": 1, "segmentation": [],
             "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0}
        )  # fmt: skip
    export = {"categories": [{"id": 1, "name": "drone"}, {"id": 2, "name": "bird"}], "images": images, "annotations": annotations}
    path.write_text(json.dumps(export))
    return path


def test_cvat_export_round_trips_into_a_loadable_dataset(ingested_clip, tmp_path):
    first = _frames(ingested_clip)[0]["file_name"]
    export = _cvat_coco_export(ingested_clip, tmp_path / "instances_default.json", skip_drone_on={first})

    import_reviewed.import_reviewed(ingested_clip, export)

    samples = {s.image_path.name: s for s in load_manifest_dataset(ingested_clip)}
    assert len(samples) == 10
    assert samples[first].is_hard_negative  # reviewed, no drone: a real hard negative
    frame = _frames(ingested_clip)[3]
    [box] = samples[frame["file_name"]].boxes
    assert (box.x1, box.y1, box.x2, box.y2, box.category) == (*synthetic_clip_box(frame["source_frame"]), "drone")

    meta = json.loads((ingested_clip / "clip_meta.json").read_text())
    assert meta["label_status"] == "reviewed" and meta["num_frames_reviewed"] == 10


def _labelstudio_export(clip, path, reviewed_count):
    tasks = []
    for i, f in enumerate(_frames(clip)):
        x1, y1, x2, y2 = synthetic_clip_box(f["source_frame"])
        result = {
            "type": "rectanglelabels", "original_width": 320, "original_height": 240,
            "value": {"x": x1 / 3.2, "y": y1 / 2.4, "width": (x2 - x1) / 3.2, "height": (y2 - y1) / 2.4,
                      "rotation": 0, "rectanglelabels": ["drone"]},
        }  # fmt: skip
        task = {
            "data": {"image": f"/data/local-files/?d=c0001/images/{f['file_name']}"},
            "predictions": [{"result": [dict(result, value=dict(result["value"], x=0.0))]}],  # a wrong pre-label
        }
        if i < reviewed_count:
            task["annotations"] = [{"result": [result], "was_cancelled": False}]
        tasks.append(task)
    path.write_text(json.dumps(tasks))
    return path


def test_labelstudio_reads_human_annotations_never_predictions(ingested_clip, tmp_path):
    export = _labelstudio_export(ingested_clip, tmp_path / "ls.json", reviewed_count=10)
    import_reviewed.import_reviewed(ingested_clip, export)

    frame = _frames(ingested_clip)[5]
    sample = next(s for s in load_manifest_dataset(ingested_clip) if s.image_path.name == frame["file_name"])
    [box] = sample.boxes
    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx(synthetic_clip_box(frame["source_frame"]))


def test_partially_reviewed_clip_is_refused_unless_allowed(ingested_clip, tmp_path):
    export = _labelstudio_export(ingested_clip, tmp_path / "ls.json", reviewed_count=4)
    with pytest.raises(import_reviewed.ImportReviewedError, match="6 of 10 frames"):
        import_reviewed.import_reviewed(ingested_clip, export)
    assert not (ingested_clip / "annotations.json").exists()

    dataset = import_reviewed.import_reviewed(ingested_clip, export, allow_partial=True)
    assert len(dataset["images"]) == 4  # unreviewed frames are left out, not written as empty
    assert json.loads((ingested_clip / "clip_meta.json").read_text())["label_status"] == "partially_reviewed"


def test_cancelled_labelstudio_annotation_counts_as_unreviewed(ingested_clip, tmp_path):
    export = _labelstudio_export(ingested_clip, tmp_path / "ls.json", reviewed_count=10)
    tasks = json.loads(export.read_text())
    tasks[0]["annotations"][0]["was_cancelled"] = True
    export.write_text(json.dumps(tasks))
    with pytest.raises(import_reviewed.ImportReviewedError, match="1 of 10 frames"):
        import_reviewed.import_reviewed(ingested_clip, export)


def test_refuses_preannotate_output_and_frames_from_another_clip(ingested_clip, tmp_path):
    prelabels = tmp_path / "coco_predictions.json"
    prelabels.write_text(json.dumps({"info": {"review_status": "unreviewed_predictions"}, "images": [], "annotations": [], "categories": []}))
    with pytest.raises(import_reviewed.ImportReviewedError, match="pre-label file"):
        import_reviewed.import_reviewed(ingested_clip, prelabels)

    export = _cvat_coco_export(ingested_clip, tmp_path / "other.json")
    data = json.loads(export.read_text())
    data["images"][0]["file_name"] = "someotherclip_f000000.jpg"
    export.write_text(json.dumps(data))
    with pytest.raises(import_reviewed.ImportReviewedError, match="not in this clip"):
        import_reviewed.import_reviewed(ingested_clip, export)


def test_boxes_are_clipped_to_the_frame_and_empty_ones_dropped(ingested_clip, tmp_path):
    export = _cvat_coco_export(ingested_clip, tmp_path / "instances_default.json")
    data = json.loads(export.read_text())
    data["annotations"][0]["bbox"] = [300.0, 230.0, 50.0, 50.0]  # runs off the 320x240 frame
    data["annotations"][1]["bbox"] = [10.0, 10.0, 0.0, 5.0]
    export.write_text(json.dumps(data))

    dataset = import_reviewed.import_reviewed(ingested_clip, export)
    assert len(dataset["annotations"]) == 9
    assert dataset["annotations"][0]["bbox"] == [300.0, 230.0, 20.0, 10.0]


def test_cli_exit_codes(ingested_clip, tmp_path):
    export = _cvat_coco_export(ingested_clip, tmp_path / "instances_default.json")
    assert import_reviewed.main(["--clip", str(ingested_clip), "--export", str(export)]) == 0
    assert import_reviewed.main(["--clip", str(ingested_clip), "--export", str(tmp_path / "missing.json")]) == 1
