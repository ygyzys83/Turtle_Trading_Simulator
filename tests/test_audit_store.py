from pathlib import Path
from tempfile import TemporaryDirectory

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
