from agentloop_trader.automation import (
    AutomationDryRunStore,
    automation_evidence_records,
    automation_readiness_records,
    automation_snapshot_record,
    automation_supervisor_dry_run_records,
    build_automation_snapshot,
    build_paper_automation_candidates,
    evidence_dashboard_records,
    paper_automation_candidate_records,
    paper_automation_dry_run,
)
from agentloop_trader.broker_governance import BrokerStateHealth
from agentloop_trader.models import PreflightCheckResult, RiskCheckResult, TradeIntent
import tempfile


def test_paper_automation_dry_run_holds_without_intent():
    decision = paper_automation_dry_run(
        intent=None,
        risk_check=RiskCheckResult(False, ["No trade intent was generated."], {}),
        preflight=PreflightCheckResult(False, {}, ["No trade intent is present."]),
        broker_health=BrokerStateHealth(True, False, []),
        duplicate_reasons=[],
        idempotency_blocked=False,
    )

    assert not decision.ready
    assert decision.action == "hold"


def test_paper_automation_dry_run_reports_candidate_when_all_gates_pass():
    decision = paper_automation_dry_run(
        intent=TradeIntent(symbol="AAPL", side="buy", quantity=1),
        risk_check=RiskCheckResult(True, [], {"risk": True}),
        preflight=PreflightCheckResult(True, {"preflight": True}, []),
        broker_health=BrokerStateHealth(True, False, []),
        duplicate_reasons=[],
        idempotency_blocked=False,
    )

    assert decision.ready
    assert decision.action == "paper_order_candidate"


def test_paper_automation_candidates_include_entry_exit_and_cancel_dry_runs():
    entry_decision = paper_automation_dry_run(
        intent=TradeIntent(symbol="AAPL", side="buy", quantity=10),
        risk_check=RiskCheckResult(True, [], {"risk": True}),
        preflight=PreflightCheckResult(True, {"preflight": True}, []),
        broker_health=BrokerStateHealth(True, False, []),
        duplicate_reasons=[],
        idempotency_blocked=False,
    )

    rows = paper_automation_candidate_records(
        entry_decision=entry_decision,
        entry_symbol="AAPL",
        entry_side="buy",
        entry_quantity=10,
        exit_previews=[{"Symbol": "AAPL", "Side": "sell", "Quantity": 10, "Preview Hash": "exit1", "Valid": True}],
        cancelable_orders=[{"Symbol": "AAPL", "Side": "buy", "Quantity": "10"}],
        exit_blockers={},
    )

    actions = [row["Action"] for row in rows]
    assert actions == ["entry_candidate", "exit_candidate", "cancel_candidate"]
    assert all(row["Dry Run Only"] for row in rows)
    assert all(row["Broker Write Required"] for row in rows)


def test_paper_automation_candidates_hold_blocked_exit_preview():
    candidates = build_paper_automation_candidates(
        entry_decision=paper_automation_dry_run(
            intent=None,
            risk_check=RiskCheckResult(False, ["No trade intent was generated."], {}),
            preflight=PreflightCheckResult(False, {}, ["No trade intent is present."]),
            broker_health=BrokerStateHealth(True, False, []),
            duplicate_reasons=[],
            idempotency_blocked=False,
        ),
        exit_previews=[{"Symbol": "AAPL", "Side": "sell", "Quantity": 10, "Preview Hash": "exit1", "Valid": True}],
        exit_blockers={"exit1": ["Alpaca paper has no open AAPL position to exit."]},
    )

    exit_candidate = candidates[1]
    assert exit_candidate.action == "exit_hold"
    assert not exit_candidate.ready
    assert not exit_candidate.broker_write_required
    assert "no open AAPL position" in exit_candidate.reasons[0]


def test_automation_readiness_records_are_display_ready():
    rows = automation_readiness_records(
        broker_connected=True,
        broker_state_stale=False,
        manual_order_gate_enabled=False,
        kill_switch_enabled=False,
        candidates=[
            {"Ready": True, "Broker Write Required": True},
            {"Ready": False, "Broker Write Required": False},
        ],
    )

    records = {row["Check"]: row for row in rows}
    assert records["Dry Run Only"]["Passed"]
    assert not records["Manual Order Gate"]["Passed"]
    assert records["Ready Candidates"]["Passed"]
    assert "1 ready candidate" in records["Ready Candidates"]["Detail"]


def test_evidence_dashboard_records_count_key_events():
    records = evidence_dashboard_records(
        [
            {"event_type": "alpaca_paper_order_submitted"},
            {"event_type": "alpaca_paper_order_armed"},
            {"event_type": "alpaca_paper_order_state_refreshed"},
            {"event_type": "alpaca_paper_exit_submitted"},
            {"event_type": "alpaca_paper_exit_armed"},
            {"event_type": "alpaca_paper_exit_blocked"},
            {"event_type": "paper_order_filled"},
        ],
        [
            {"broker_order_id": "1", "status": "filled", "lifecycle_status": "filled_at_alpaca", "filled_quantity": "40"},
            {"broker_order_id": "2", "status": "canceled", "lifecycle_status": "canceled_at_alpaca"},
            {"broker_order_id": "3", "status": "accepted", "lifecycle_status": "open_at_alpaca"},
            {"broker_order_id": "4", "status": "accepted", "lifecycle_status": "missing_from_alpaca_orders"},
        ],
    )

    metrics = {record["Metric"]: record["Value"] for record in records}
    assert metrics["Alpaca Paper Orders Submitted"] == 1
    assert metrics["Alpaca Paper Exits Submitted"] == 1
    assert metrics["Alpaca Paper Exits Armed"] == 1
    assert metrics["Alpaca Paper Exits Blocked"] == 1
    assert metrics["Tracked Alpaca Orders"] == 4
    assert metrics["Tracked Alpaca Open Orders"] == 1
    assert metrics["Tracked Alpaca Filled Orders"] == 1
    assert metrics["Tracked Alpaca Filled Quantity"] == "40"
    assert metrics["Tracked Alpaca Canceled Orders"] == 1
    assert metrics["Tracked Alpaca Missing Orders"] == 1
    assert metrics["Local Paper Orders"] == 1


def test_automation_snapshot_records_candidate_evidence():
    snapshot = build_automation_snapshot(
        session_id="session-1",
        candidates=[
            {
                "Action": "entry_candidate",
                "Ready": True,
                "Broker Write Required": True,
                "Reasons": "Dry-run only.",
            },
            {
                "Action": "exit_hold",
                "Ready": False,
                "Broker Write Required": False,
                "Reasons": "Broker state is stale.",
            },
        ],
        readiness=[{"Check": "Dry Run Only", "Passed": True}],
    )

    record = automation_snapshot_record(snapshot)
    evidence = {row["Metric"]: row["Value"] for row in automation_evidence_records([record])}

    assert record["candidate_count"] == 2
    assert record["ready_candidate_count"] == 1
    assert record["broker_write_candidate_count"] == 1
    assert record["hold_count"] == 1
    assert evidence["Dry Run Snapshots"] == 1
    assert evidence["Action: exit_hold"] == 1
    assert evidence["Hold Reason: Broker state is stale."] == 1


def test_automation_dry_run_store_appends_and_reads_recent():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = AutomationDryRunStore(f"{tmp_dir}/dry_runs.jsonl")
        first = build_automation_snapshot("session-1", [], [])
        second = build_automation_snapshot(
            "session-2",
            [{"Action": "cancel_candidate", "Ready": True, "Broker Write Required": True}],
            [],
        )

        store.append(first)
        written = store.append(second)
        rows = store.read_recent(limit=1)

    assert written["session_id"] == "session-2"
    assert rows[0]["session_id"] == "session-2"
    assert rows[0]["candidate_count"] == 1


def test_automation_supervisor_dry_run_halts_before_candidates():
    rows = automation_supervisor_dry_run_records(
        candidates=[{"Ready": True}],
        readiness_rows=[{"Check": "Dry Run Only", "Passed": True}],
        halt_rows=[{"Halt Reason": "Risk breach", "Active": True, "Detail": "Kill switch is enabled."}],
    )
    values = {row["Field"]: row["Value"] for row in rows}

    assert values["Supervisor Mode"] == "dry_run_only"
    assert values["Decision"] == "halt"
    assert values["Broker Writes Submitted"] == 0


def test_automation_supervisor_dry_run_queues_manual_review_when_clear():
    rows = automation_supervisor_dry_run_records(
        candidates=[{"Ready": True}, {"Ready": False}],
        readiness_rows=[{"Check": "Dry Run Only", "Passed": True}],
        halt_rows=[{"Halt Reason": "None", "Active": False}],
    )
    values = {row["Field"]: row["Value"] for row in rows}

    assert values["Decision"] == "would_queue_manual_review"
    assert values["Ready Candidates"] == 1
