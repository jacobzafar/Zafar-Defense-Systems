"""Tests for tools/ingest_footage.py on a generated 2-second clip."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import ingest_footage  # noqa: E402
from footage_meta import ClipMetaError, load_clip_dirs  # noqa: E402
from tests.conftest import POSITIVE_CLIP_META, SYNTHETIC_CLIP_FPS, SYNTHETIC_CLIP_SECONDS  # noqa: E402

TOTAL_FRAMES = int(SYNTHETIC_CLIP_FPS * SYNTHETIC_CLIP_SECONDS)


def test_extracts_frames_at_requested_fps_by_timestamp(ingested_clip):
    frames = json.loads((ingested_clip / "frames.json").read_text())["images"]

    # 15 FPS source sampled at 5 FPS: every 3rd source frame, 10 in 2 s.
    assert [f["source_frame"] for f in frames] == list(range(0, TOTAL_FRAMES, 3))
    for f in frames:
        assert f["file_name"] == f"c0001_f{f['source_frame']:06d}.jpg"
        assert (ingested_clip / "images" / f["file_name"]).is_file()
        assert (f["width"], f["height"]) == (320, 240)
    assert sorted(p.name for p in (ingested_clip / "images").iterdir()) == [f["file_name"] for f in frames]


def test_writes_metadata_sidecar_with_source_provenance(ingested_clip, synthetic_video):
    meta = json.loads((ingested_clip / "clip_meta.json").read_text())

    for key, value in POSITIVE_CLIP_META.items():
        assert meta[key] == value
    assert meta["clip_id"] == "c0001"
    assert meta["num_frames_extracted"] == 10
    assert meta["sample_fps"] == 5.0
    assert meta["label_status"] == "unlabeled"
    assert meta["source"]["sha256"] == hashlib.sha256(synthetic_video.read_bytes()).hexdigest()
    assert meta["source"]["fps"] == pytest.approx(SYNTHETIC_CLIP_FPS)
    assert meta["source"]["num_frames_decoded"] == TOTAL_FRAMES
    assert meta["source"]["duration_s"] == pytest.approx(SYNTHETIC_CLIP_SECONDS)


def test_writes_no_annotations_file(ingested_clip):
    # An empty annotations.json would make the loader read every frame as a
    # hard negative — labels only arrive via tools/import_reviewed.py.
    assert not (ingested_clip / "annotations.json").exists()


def test_fps_at_or_above_source_keeps_every_frame(tmp_path, synthetic_video):
    clip = ingest_footage.ingest(synthetic_video, dict(POSITIVE_CLIP_META), tmp_path / "clips", sample_fps=60.0)
    assert len(json.loads((clip / "frames.json").read_text())["images"]) == TOTAL_FRAMES


def test_default_clip_id_comes_from_file_name(tmp_path, synthetic_video):
    clip = ingest_footage.ingest(synthetic_video, dict(POSITIVE_CLIP_META), tmp_path / "clips")
    assert clip.name == "c0001"


def test_refuses_to_overwrite_an_existing_clip(ingested_clip, synthetic_video):
    with pytest.raises(ingest_footage.IngestError, match="already ingested"):
        ingest_footage.ingest(synthetic_video, dict(POSITIVE_CLIP_META), ingested_clip.parent, clip_id="c0001")


def test_unreadable_video_leaves_no_partial_clip(tmp_path):
    bogus = tmp_path / "broken.mp4"
    bogus.write_bytes(b"not a video")
    with pytest.raises(ingest_footage.IngestError):
        ingest_footage.ingest(bogus, dict(POSITIVE_CLIP_META), tmp_path / "clips")
    assert not (tmp_path / "clips").exists() or not any((tmp_path / "clips").iterdir())


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"is_hard_negative": True}, "drone_type 'none'"),
        ({"drone_type": "none"}, "only for hard-negative"),
        ({"lighting": "sunny"}, "lighting"),
        ({"negative_subject": "cat"}, "negative_subject"),
    ],
)
def test_rejects_inconsistent_or_unknown_metadata(tmp_path, synthetic_video, overrides, message):
    with pytest.raises(ClipMetaError, match=message):
        ingest_footage.ingest(synthetic_video, {**POSITIVE_CLIP_META, **overrides}, tmp_path / "clips")


def test_rejects_non_video_extension_and_bad_fps(tmp_path, synthetic_video):
    text = tmp_path / "notes.txt"
    text.write_text("x")
    with pytest.raises(ingest_footage.IngestError, match="extension"):
        ingest_footage.ingest(text, dict(POSITIVE_CLIP_META), tmp_path / "clips")
    with pytest.raises(ingest_footage.IngestError, match="fps"):
        ingest_footage.ingest(synthetic_video, dict(POSITIVE_CLIP_META), tmp_path / "clips", sample_fps=0)


def test_cli_ingests_a_hard_negative_and_load_clip_dirs_finds_it(tmp_path, synthetic_video):
    root = tmp_path / "clips"
    argv = [
        "--video", str(synthetic_video), "--footage-root", str(root), "--clip-id", "gulls",
        "--hard-negative", "--negative-subject", "bird", "--drone-type", "none",
        "--distance-band", "close-lt50m", "--lighting", "daylight-overcast", "--background", "water",
    ]  # fmt: skip
    assert ingest_footage.main(argv) == 0
    assert ingest_footage.main(argv) == 1  # second run: already ingested

    [(clip_dir, meta)] = load_clip_dirs(root)
    assert clip_dir.name == "gulls"
    assert meta["is_hard_negative"] is True and meta["negative_subject"] == "bird"
