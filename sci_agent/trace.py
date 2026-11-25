from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


class TrajectoryRecorder:
    """Append complete model and agent/environment events to two JSONL files."""

    def __init__(self, directory: str, context: Optional[Dict[str, Any]] = None):
        os.makedirs(directory, exist_ok=True)
        self.fm_path = os.path.join(directory, "fm_traj.jsonl")
        self.e2e_path = os.path.join(directory, "e2e_traj.jsonl")
        self.run_id = uuid.uuid4().hex
        self.context = context or {}

    def _append(self, path: str, record: Dict[str, Any]) -> None:
        envelope = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timestamp_monotonic_ns": time.monotonic_ns(),
            "run_id": self.run_id,
            "context": self.context,
            **record,
        }
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(envelope, ensure_ascii=False, default=_json_default) + "\n")

    def record_fm(
        self,
        *,
        step: int,
        session_id: Optional[str],
        input: Dict[str, Any],
        output: Any = None,
        error: Optional[str] = None,
    ) -> None:
        self._append(self.fm_path, {
            "step": step,
            "session_id": session_id,
            "input": input,
            "output": output,
            "error": error,
        })

    def record_e2e(
        self,
        *,
        event: str,
        step: int,
        session_id: Optional[str],
        input: Any,
        output: Any,
    ) -> None:
        self._append(self.e2e_path, {
            "event": event,
            "step": step,
            "session_id": session_id,
            "input": input,
            "output": output,
        })
