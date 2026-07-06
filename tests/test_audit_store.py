from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import UTC, datetime

from agentloop_trader.audit_store import JsonlAuditStore
from agentloop_trader.models import AuditEvent


def test_jsonl_audit_store_appends_and_reads_recent_events():
    with TemporaryDirectory() as tmp:
        store = JsonlAuditStore(Path(tmp) / "audit.jsonl")
        store.append(AuditEvent(event_type="one", message="First", payload={"x": 1}))
        store.append(AuditEvent(event_type="two", message="Second", payload={"x": 2}))

        records = store.read_recent(limit=1)

    assert len(records) == 1
    assert records[0]["event_type"] == "two"
    assert records[0]["payload"]["x"] == 2


def test_jsonl_audit_store_serializes_dataclass_payload():
    event = AuditEvent(
        event_type="nested",
        message="Dataclass payload",
        payload={"event": AuditEvent(event_type="inner", message="Inner")},
    )
    with TemporaryDirectory() as tmp:
        store = JsonlAuditStore(Path(tmp) / "audit.jsonl")

        store.append(event)

        records = store.read_recent()
    assert records[0]["payload"]["event"]["event_type"] == "inner"


def test_audit_event_defaults_to_pacific_time():
    event = AuditEvent(event_type="timezone", message="Pacific timestamp")

    assert event.created_at.tzinfo is not None
    assert event.created_at.tzinfo.key == "America/Los_Angeles"


def test_jsonl_audit_store_normalizes_utc_events_to_pacific_time():
    event = AuditEvent(
        event_type="timezone",
        message="UTC event should serialize as Pacific",
        created_at=datetime(2026, 7, 6, 2, 7, 44, tzinfo=UTC),
    )

    record = JsonlAuditStore.event_to_record(event)

    assert record["created_at"].startswith("2026-07-05T19:07:44")
    assert record["created_at"].endswith("-07:00")
