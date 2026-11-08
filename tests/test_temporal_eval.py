"""Phase 0 of the temporal / robustness experiment (docs/DECISIONS.md #27):
the derived clip view, simulated sensors and shake, and the new metrics.
Torch-free: the harness runs use the `motion` detector."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cv2 = pytest.importorskip("cv2")

from detector.base import Detection  # noqa: E402
from detector.sensor_sim import SensorSimError, apply_sensor, apply_shake, frame_rng  # noqa: E402
from eval.harness import EvalConfig, EvalConfigError, run_evaluation  # noqa: E402
from eval.metrics import FrameEvalData, PredBox, compute_fp_on_target_frames, compute_track_quality  # noqa: E402
from eval.schema import EvalBox, EvalFrame, EvalSequence, freeze_eval_set, load_eval_manifest  # noqa: E402
from eval.sequences import SequenceGroupingError, group_by_filename_runs  # noqa: E402
from tracker.base import Track  # noqa: E402


def _numbered_eval_set(root: Path, numbers: list[int], prefix: str = "Clip") -> Path:
    """One-frame-per-sequence eval set, as build_frozen_eval_set writes,
    with VisioDECT-style "<prefix> (<n>).jpg" names; a drone square moves
    right by 4px per frame number, frames >= 100 are empty."""
    images = root / "images"
    images.mkdir(parents=True)
    sequences = []
    for i, n in enumerate(numbers):
        arr = np.full((96, 128, 3), 40, np.uint8)
        boxes = []
        if n < 100:
            x = 10 + 4 * (n % 20)
            arr[40:52, x : x + 12] = 230
            boxes.append({"bbox": [x, 40, 12, 12], "category": "drone"})
        name = f"{prefix} ({n}).jpg"
        cv2.imwrite(str(images / name), arr)
        sequences.append({"id": f"img_{i:05d}", "frames": [{"file_name": name, "width": 128, "height": 96, "boxes": boxes}]})
    (root / "sequences.json").write_text(json.dumps({"sequences": sequences}))
    freeze_eval_set(root)
    return root


def test_filename_runs_regroups_by_consecutive_numbers_in_order(tmp_path):
    root = _numbered_eval_set(tmp_path / "set", [3, 1, 2, 7, 8, 101])
    grouped = group_by_filename_runs(load_eval_manifest(root))
    assert [s.sequence_id for s in grouped] == ["Clip#1-3", "Clip#7-8", "Clip#101-101"]
    assert [f.image_path.name for f in grouped[0].frames] == ["Clip (1).jpg", "Clip (2).jpg", "Clip (3).jpg"]
    assert [f.frame_index for f in grouped[0].frames] == [0, 1, 2]


def test_filename_runs_rejects_unnumbered_and_duplicate_frames(tmp_path):
    frame = EvalFrame("s", 0, Path("no_number.jpg"), 10, 10)
    with pytest.raises(SequenceGroupingError):
        group_by_filename_runs([EvalSequence("s", [frame])])
    dup = EvalFrame("s", 0, Path("a (1).jpg"), 10, 10)
    with pytest.raises(SequenceGroupingError):
        group_by_filename_runs([EvalSequence("a", [dup]), EvalSequence("b", [dup])])


def test_grouping_keeps_frame_set_and_makes_continuity_measurable(tmp_path):
    root = _numbered_eval_set(tmp_path / "set", list(range(1, 13)))
    base = dict(eval_set_dir=str(root), detector={"backend": "motion", "min_area_px": 20, "history": 5})
    flat = run_evaluation(EvalConfig(**base))
    clips = run_evaluation(EvalConfig(**base, sequence_grouping="filename_runs"))
    assert flat.num_frames == clips.num_frames == 12
    assert clips.num_sequences == 1
    assert flat.track_continuity["measurable"] is False
    assert clips.track_continuity["measurable"] is True
    assert clips.track_quality["measurable"] is True
    assert clips.recall_by_apparent_size["all"]["count"] == 12


def test_eval_config_rejects_bad_grouping_and_ops(tmp_path):
    with pytest.raises(EvalConfigError):
        EvalConfig(eval_set_dir="x", detector={}, sequence_grouping="by_magic")
    with pytest.raises(SensorSimError):
        EvalConfig(eval_set_dir="x", detector={}, sensor_sim=[{"op": "sharpen"}])
    with pytest.raises(EvalConfigError):
        EvalConfig(eval_set_dir="x", detector={}, shake={"seed": 1})


def test_sensor_ops_are_deterministic_per_frame_and_keep_dtype():
    image = np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    ops = [{"op": "noise", "read_sigma": 5.0, "shot": 0.05}, {"op": "motion_blur", "length_frac": 0.1}, {"op": "jpeg", "quality": 30}]
    a = apply_sensor(image, ops, frame_rng(0, "f (1).jpg"))
    b = apply_sensor(image, ops, frame_rng(0, "f (1).jpg"))
    c = apply_sensor(image, ops, frame_rng(0, "f (2).jpg"))
    assert a.dtype == np.uint8 and a.shape == image.shape
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_resize_gray_invert_and_color():
    image = np.full((40, 60, 3), 100, np.uint8)
    image[..., 2] = 200
    rng = frame_rng(0, "k")
    assert apply_sensor(image, [{"op": "resize", "scale": 0.5}], rng).shape == (20, 30, 3)
    assert apply_sensor(image, [{"op": "resize", "scale": 0.5, "restore": True}], rng).shape == image.shape
    gray = apply_sensor(image, [{"op": "gray"}], rng)
    assert np.all(gray[..., 0] == gray[..., 2])
    assert np.array_equal(apply_sensor(image, [{"op": "invert"}], rng), 255 - image)
    warmed = apply_sensor(image, [{"op": "color", "gamma": 1.0, "wb": [0.5, 1.0, 1.0]}], rng)
    assert warmed[0, 0, 0] == 50 and warmed[0, 0, 1] == 100
    with pytest.raises(SensorSimError):
        apply_sensor(image, [{"op": "jpeg", "q": 10}], rng)


def test_shake_moves_content_by_the_returned_offset():
    image = np.zeros((100, 100, 3), np.uint8)
    image[40:50, 40:50] = 255
    shaken, (dx, dy) = apply_shake(image, frame_rng(0, "k"), 0.1)
    assert (dx, dy) != (0, 0) and abs(dx) <= 10 and abs(dy) <= 10
    ys, xs = np.nonzero(shaken[..., 0] > 128)
    assert (xs.min(), ys.min()) == (40 + dx, 40 + dy)


def test_harness_shifts_ground_truth_with_shake(tmp_path):
    root = _numbered_eval_set(tmp_path / "set", [1, 2])
    config = EvalConfig(
        eval_set_dir=str(root),
        detector={"backend": "motion"},
        shake={"amplitude_frac": 0.1, "seed": 3},
        output_predictions_path=str(tmp_path / "p.jsonl.gz"),
    )
    run_evaluation(config)
    import gzip

    rows = [json.loads(line) for line in gzip.open(tmp_path / "p.jsonl.gz", "rt")]
    for row in rows:
        n = int(row["frame_id"].split("#")[0].split("_")[1]) + 1
        x = 10 + 4 * n
        _, (dx, dy) = apply_shake(np.zeros((96, 128, 3), np.uint8), frame_rng(3, f"shake:Clip ({n}).jpg"), 0.1)
        assert row["gt"][0][:2] == [max(0, x + dx), max(0, 40 + dy)]


def test_fp_on_target_frames_counts_only_unmatched_on_drone_frames():
    gt = [EvalBox(0, 0, 10, 10, "drone")]
    frames = [
        FrameEvalData("a", gt, [PredBox(0, 0, 10, 10, 0.9), PredBox(50, 50, 60, 60, 0.8)]),
        FrameEvalData("b", [], [PredBox(0, 0, 10, 10, 0.9)]),  # empty frame: false alarm, not counted here
    ]
    result = compute_fp_on_target_frames(frames)
    assert result == {"measurable": True, "rate_per_frame": 1.0, "num_target_frames": 1, "num_false_positives": 1}


def _track(tid, x):
    return Track(tid, x, 0.0, x + 0.1, 0.1, 0.9, "drone", 1, 1, 0)


def test_track_quality_hand_computed():
    # 6 frames: GT present on 0-3 and 5; frame 4 empty.
    frames = [
        EvalFrame("s", i, Path(f"s ({i}).jpg"), 100, 100, [EvalBox(0, 0, 10, 10, "drone")] if i != 4 else [])
        for i in range(6)
    ]
    sequence = EvalSequence("s", frames)
    tracks = [
        [],                              # run 1 starts, not yet tracked
        [_track(1, 0.0)],                # tracked (delay 1)
        [_track(9, 0.5)],                # lost; track 9 is false
        [_track(1, 0.0)],                # re-tracked -> 1 fragmentation
        [_track(9, 0.5)],                # empty frame, false box
        [],                              # run 2, never tracked
    ]
    tq = compute_track_quality([sequence], {"s": tracks})
    assert tq["coverage"] == pytest.approx(2 / 5)
    assert tq["num_gt_runs"] == 2 and tq["runs_never_tracked"] == 1
    assert tq["frames_to_first_track_mean"] == 1.0
    assert tq["fragmentations"] == 1
    assert tq["false_track_boxes"]["target_frames"] == 1
    assert tq["false_track_boxes"]["empty_frames"] == 1
    assert tq["num_false_track_ids"] == 1


def test_track_quality_not_measurable_on_single_frame_sequences():
    frame = EvalFrame("s", 0, Path("s (1).jpg"), 10, 10, [EvalBox(0, 0, 5, 5, "drone")])
    tq = compute_track_quality([EvalSequence("s", [frame])], {"s": [[]]})
    assert tq["measurable"] is False and tq["coverage"] is None


def test_detector_reset_is_called_once_per_sequence(tmp_path, monkeypatch):
    root = _numbered_eval_set(tmp_path / "set", [1, 2, 5, 6])
    calls = []

    class Stateful:
        name = "stateful"

        def reset(self):
            calls.append("reset")

        def detect(self, frame):
            self.last_stage_ms = {"stage_a": 1.0}
            return [Detection(0.1, 0.1, 0.2, 0.2, 0.9)]

    monkeypatch.setattr("eval.harness.build_detector", lambda cfg: Stateful())
    card = run_evaluation(EvalConfig(eval_set_dir=str(root), detector={}, sequence_grouping="filename_runs"))
    assert calls == ["reset", "reset"]
    assert card.latency_by_stage["stage_a"]["mean_ms"] == 1.0


def test_compare_renders_one_row_per_card_from_real_harness_output(tmp_path):
    from eval.compare import render

    root = _numbered_eval_set(tmp_path / "set", list(range(1, 6)) + [101, 102])
    card = run_evaluation(
        EvalConfig(eval_set_dir=str(root), detector={"backend": "motion", "min_area_px": 20}, sequence_grouping="filename_runs")
    ).as_dict()
    table = render([("A", card), ("B", card)])
    lines = table.strip().splitlines()
    assert len(lines) == 4
    assert all(line.count("|") == lines[0].count("|") for line in lines)
    assert "/2 |" in lines[2]  # false alarms counted over the 2 empty frames


def test_per_frame_tracks_are_snapshots_not_live_tracker_objects(tmp_path, monkeypatch):
    """Regression: IoUTracker mutates its Track objects in place. Storing them
    as-is gave every frame the track's final position, so a drone tracked
    across a clip looked covered on its last frames only."""
    root = _numbered_eval_set(tmp_path / "set", list(range(1, 11)))

    class Truth:
        name = "truth"

        def detect(self, frame):
            # The drone square moves 4px/frame; find it from the pixels.
            ys, xs = np.nonzero(frame[..., 0] > 128)
            h, w = frame.shape[:2]
            return [Detection(xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h, 0.9)]

    monkeypatch.setattr("eval.harness.build_detector", lambda cfg: Truth())
    card = run_evaluation(EvalConfig(eval_set_dir=str(root), detector={}, sequence_grouping="filename_runs"))
    assert card.track_quality["coverage"] == 1.0
    assert card.track_quality["false_track_boxes"]["target_frames"] == 0
    assert card.track_continuity["overall_continuity"] == 1.0


def test_bench_latency_times_every_detector_every_round():
    from eval.bench_latency import bench

    class Counting:
        def __init__(self):
            self.calls = 0

        def detect(self, frame):
            self.calls += 1
            return []

    a, b = Counting(), Counting()
    results = bench({"a": a, "b": b}, images=[None] * 4, rounds=3, warmup=2)
    assert a.calls == b.calls == 2 + 4 * 3
    assert set(results) == {"a", "b"} and results["a"]["median_ms"] >= 0
