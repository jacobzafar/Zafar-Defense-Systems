"""Derived sequence views of a frozen eval set, for temporal evaluation.

`eval/build_frozen_eval_set.py` wraps every image as its own one-frame
"sequence", so `sequences.json` of e.g. `visiodect-fpv-test` carries no
clip structure, and track continuity is unmeasurable on it. The frame
order is still in the file names: VisioDECT names frames
"<scenario> (<n>).jpg", and the FPV test set is six contiguous 120-frame
blocks (tools/split_visiodect_fpv.py: blocks 2 and 7 of each scenario).

`group_by_filename_runs` regroups the loaded frames into one sequence per
run of consecutive frame numbers under the same name prefix. It is a view
computed at load time: sequences.json is never rewritten, so the frozen
checksum stays valid, and the frame set (hence every per-frame metric of a
stateless detector) is unchanged. A gap in the numbering always starts a
new sequence, so a temporal detector or tracker never carries state across
a cut between held-out blocks.
"""

from __future__ import annotations

import re
from dataclasses import replace

from eval.schema import EvalFrame, EvalSequence

_NUMBERED = re.compile(r"^(?P<prefix>.*?)\s*\((?P<n>\d+)\)\.[^.]+$")

SEQUENCE_GROUPINGS = ("none", "filename_runs")


class SequenceGroupingError(ValueError):
    """Raised when frames cannot be grouped as requested."""


def _key(frame: EvalFrame) -> tuple[str, int]:
    match = _NUMBERED.match(frame.image_path.name)
    if not match:
        raise SequenceGroupingError(f"No '<prefix> (<n>).<ext>' frame number in {frame.image_path.name!r}")
    return match.group("prefix"), int(match.group("n"))


def group_by_filename_runs(sequences: list[EvalSequence]) -> list[EvalSequence]:
    """One sequence per run of consecutive frame numbers per name prefix,
    frames in frame-number order, ids "<prefix>#<first>-<last>"."""
    keyed = sorted(((_key(f), f) for s in sequences for f in s.frames), key=lambda kf: kf[0])
    runs: list[list[tuple[tuple[str, int], EvalFrame]]] = []
    for item in keyed:
        (prefix, n), _ = item
        if runs and runs[-1][-1][0] == (prefix, n):
            raise SequenceGroupingError(f"Duplicate frame number {n} for {prefix!r}")
        if runs and runs[-1][-1][0][0] == prefix and runs[-1][-1][0][1] == n - 1:
            runs[-1].append(item)
        else:
            runs.append([item])

    grouped = []
    for run in runs:
        (prefix, first), (_, last) = run[0][0], run[-1][0]
        sequence_id = f"{prefix}#{first}-{last}"
        frames = [replace(f, sequence_id=sequence_id, frame_index=i) for i, (_, f) in enumerate(run)]
        grouped.append(EvalSequence(sequence_id=sequence_id, frames=frames))
    return grouped


def apply_grouping(sequences: list[EvalSequence], grouping: str) -> list[EvalSequence]:
    if grouping == "none":
        return sequences
    if grouping == "filename_runs":
        return group_by_filename_runs(sequences)
    raise SequenceGroupingError(f"Unknown sequence_grouping {grouping!r}; valid: {', '.join(SEQUENCE_GROUPINGS)}")
