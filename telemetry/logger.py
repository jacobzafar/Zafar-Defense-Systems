"""JSONL event logger.

One JSON object per line, one file per run. Simple to grep, tail, or load
with `pandas.read_json(path, lines=True)` for later review. No external
logging framework required.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

EventType = Literal[
    "run_started",
    "run_stopped",
    "frame_processed",
    "frame_dropped",
    "detection",
    "track_update",
    "inference_error",
    "state_change",
]


@dataclass
class LogEvent:
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    frame_index: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class EventLogger:
    """Append-only JSONL writer for one pipeline run."""

    def __init__(self, log_dir: str | Path = "logs", run_name: str | None = None) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        run_name = run_name or time.strftime("run_%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"{run_name}.jsonl"
        self._file = self.log_path.open("a", encoding="utf-8")

    def log(self, event: LogEvent) -> None:
        self._file.write(json.dumps(asdict(event)) + "\n")
        self._file.flush()

    def log_event(
        self,
        event_type: EventType,
        frame_index: int | None = None,
        **payload: Any,
    ) -> None:
        self.log(LogEvent(event_type=event_type, frame_index=frame_index, payload=payload))

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()
