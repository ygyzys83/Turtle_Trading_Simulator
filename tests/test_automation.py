from agentloop_trader.automation import (
    AutomationDryRunStore,
    auto_exit_decision,
    auto_exit_decision_records,
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
from datetime import datetime


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


def test_auto_exit_decision_is_ready_when_all_conditions_pass():
    decision = auto_exit_decision(
        automation_level="Auto exits only",
        execution_mode="paper",
        broker_connected=True,
        broker_can_submit=True,
        paper_orders_enabled=True,
        kill_switch_enabled=False,
        broker_state_stale=False,
        market_open=True,
        exit_preview_records=[{"Symbol": "AAPL", "Quantity": "10", "Review ID": "exit-1", "Valid": True}],
        exit_blockers={},
        already_sent_hashes=set(),
    )
    records = {row["Field"]: row["Value"] for row in auto_exit_decision_records(decision)}

    assert decision.ready
    assert decision.status == "Exit ready"
    assert records["Symbol"] == "AAPL"


def test_auto_exit_decision_blocks_manual_mode_and_closed_market():
    decision = auto_exit_decision(
        automation_level="Manual review only",
        execution_mode="paper",
        broker_connected=True,
        broker_can_submit=True,
        paper_orders_enabled=True,
        kill_switch_enabled=False,
        broker_state_stale=False,
        market_open=False,
        exit_preview_records=[{"Symbol": "AAPL", "Quantity": "10", "Review ID": "exit-1", "Valid": True}],
        exit_blockers={},
        already_sent_hashes=set(),
    )

    assert not decision.ready
    assert decision.status == "Off"
    assert "Manual review only" in decision.reasons[0]

def test_auto_exit_decision_blocks_backtest_mode():
    decision = auto_exit_decision(
        automation_level="Auto exits only",
        execution_mode="backtest_only",
        broker_connected=True,
        broker_can_submit=True,
        paper_orders_enabled=True,
        kill_switch_enabled=False,
        broker_state_stale=False,
        market_open=True,
        exit_preview_records=[{"Symbol": "AAPL", "Quantity": "10", "Review ID": "exit-1", "Valid": True}],
        exit_blockers={},
        already_sent_hashes=set(),
    )

    assert not decision.ready
    assert decision.status == "Exit blocked"
    assert "How orders are handled must be Paper trading." in decision.reasons


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
    assert actions == ["Buy ready to send", "Exit ready to send", "Cancel ready to send"]
    assert all(row["No Orders Sent"] for row in rows)
    assert all(row["Would Need Approval"] for row in rows)


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
    assert records["No orders sent"]["Passed"]
    assert not records["Paper orders allowed"]["Passed"]
    assert records["Paper actions ready"]["Passed"]
    assert "1 paper action" in records["Paper actions ready"]["Detail"]


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
    assert metrics["Paper buys sent"] == 1
    assert metrics["Paper exits sent"] == 1
    assert metrics["Paper exits sent"] == 1
    assert metrics["Paper exits blocked"] == 1
    assert metrics["Saved Alpaca orders"] == 4
    assert metrics["Open Alpaca orders"] == 1
    assert metrics["Filled Alpaca orders"] == 1
    assert metrics["Filled Alpaca shares"] == "40"
    assert metrics["Canceled Alpaca orders"] == 1
    assert metrics["Saved orders missing at Alpaca"] == 1
    assert metrics["App paper orders"] == 1


def test_automation_snapshot_records_candidate_evidence():
    snapshot = build_automation_snapshot(
        session_id="session-1",
        candidates=[
            {
                "Action": "Buy ready to send",
                "Ready": True,
                "Would Need Approval": True,
                "Reasons": "Check only.",
            },
            {
                "Action": "Exit blocked",
                "Ready": False,
                "Would Need Approval": False,
                "Reasons": "Broker state is stale.",
            },
        ],
        readiness=[{"Check": "No orders sent", "Passed": True}],
    )

    record = automation_snapshot_record(snapshot)
    evidence = {row["Metric"]: row["Value"] for row in automation_evidence_records([record])}

    assert record["candidate_count"] == 2
    assert not record["created_at"].endswith("+00:00")
    assert datetime.fromisoformat(record["created_at"]).utcoffset().total_seconds() in {-7 * 3600, -8 * 3600}
    assert record["ready_candidate_count"] == 1
    assert record["broker_write_candidate_count"] == 1
    assert record["hold_count"] == 1
    assert evidence["Saved checks"] == 1
    assert evidence["Action type: Exit blocked"] == 1
    assert evidence["Blocked reason: Broker state is stale."] == 1


def test_automation_dry_run_store_appends_and_reads_recent():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = AutomationDryRunStore(f"{tmp_dir}/dry_runs.jsonl")
        first = build_automation_snapshot("session-1", [], [])
        second = build_automation_snapshot(
            "session-2",
            [{"Action": "cancel_candidate", "Ready": True, "Would Need Approval": True}],
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
        readiness_rows=[{"Check": "No orders sent", "Passed": True}],
        halt_rows=[{"Block": "Risk limit hit", "Active": True, "Detail": "Stop trading switch is on."}],
    )
    values = {row["Field"]: row["Value"] for row in rows}

    assert values["Mode"] == "check only"
    assert values["Decision"] == "blocked"
    assert values["Orders sent"] == 0


def test_automation_supervisor_dry_run_is_ready_when_clear():
    rows = automation_supervisor_dry_run_records(
        candidates=[{"Ready": True}, {"Ready": False}],
        readiness_rows=[{"Check": "No orders sent", "Passed": True}],
        halt_rows=[{"Block": "None", "Active": False}],
    )
    values = {row["Field"]: row["Value"] for row in rows}

    assert values["Decision"] == "ready"
    assert values["Paper actions ready"] == 1
