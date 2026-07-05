import tempfile

from agentloop_trader.evidence import (
    approval_ledger_records,
    approval_ledger_summary_records,
    build_evidence_package,
    evidence_package_records,
    write_evidence_package,
)


def test_approval_ledger_extracts_preview_hashes_and_manual_gates():
    rows = approval_ledger_records(
        [
            {
                "created_at": "2026-07-05T12:00:00Z",
                "event_type": "alpaca_paper_order_armed",
                "payload": {"symbol": "AAPL", "side": "buy", "quantity": 40, "preview_hash": "abc"},
            },
            {
                "created_at": "2026-07-05T12:01:00Z",
                "event_type": "alpaca_paper_cancel_submitted",
                "payload": {"symbol": "AAPL", "cancel_preview_hash": "def", "broker_order_id": "order-1"},
            },
            {"event_type": "paper_order_filled", "payload": {}},
        ]
    )
    summary = {row["Metric"]: row["Value"] for row in approval_ledger_summary_records(rows)}

    assert len(rows) == 2
    assert rows[0]["Action"] == "order armed"
    assert rows[0]["Preview Hash"] == "abc"
    assert rows[1]["Broker Write"]
    assert rows[1]["Manual Gate Required"]
    assert summary["Rows With Preview Hash"] == 2


def test_evidence_package_can_be_written_locally():
    package = build_evidence_package(
        session_id="session-1",
        manifest={"session_id": "session-1"},
        audit_records=[{"event_type": "run_manifest_recorded"}],
        approval_ledger=[{"Action": "order armed"}],
        tracked_orders=[{"broker_order_id": "order-1"}],
        automation_snapshots=[{"candidate_count": 1}],
        readiness_rows=[{"Check": "Live broker writes blocked", "Passed": True}],
        risk_halts=[{"Halt Reason": "None", "Active": False}],
    )
    rows = {row["Field"]: row["Value"] for row in evidence_package_records(package)}

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = write_evidence_package(package, f"{tmp_dir}/evidence.json")
        text = path.read_text(encoding="utf-8")

    assert rows["Session"] == "session-1"
    assert rows["Audit Records"] == 1
    assert '"session_id": "session-1"' in text
