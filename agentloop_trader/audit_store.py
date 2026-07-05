from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from agentloop_trader.models import AuditEvent


DEFAULT_AUDIT_DIR = Path("audit_logs")
DEFAULT_AUDIT_FILE = "agentloop_audit.jsonl"


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


class JsonlAuditStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else DEFAULT_AUDIT_DIR / DEFAULT_AUDIT_FILE

    def append(self, event: AuditEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = self.event_to_record(event)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def append_many(self, events: list[AuditEvent]) -> None:
        for event in events:
            self.append(event)

    def read_recent(self, limit: int = 100) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        records = []
        for line in lines[-limit:]:
            if not line.strip():
                continue
            records.append(json.loads(line))
        return records

    @staticmethod
    def event_to_record(event: AuditEvent) -> dict:
        return {
            "created_at": event.created_at.isoformat(),
            "event_type": event.event_type,
            "message": event.message,
            "payload": _json_safe(event.payload),
        }

