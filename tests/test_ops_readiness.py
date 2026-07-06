from agentloop_trader.models import PreflightCheckResult, RiskCheckResult, RiskLimits, StrategyConfig, TradeIntent
from agentloop_trader.monitoring import MonitoringResult
from agentloop_trader.ops_readiness import (
    market_data_freshness_records,
    paper_account_health_records,
    paper_automation_gate_records,
    restart_recovery_records,
    scheduler_preview_records,
    strategy_state_snapshot_records,
)


def test_market_data_freshness_records_accept_synthetic_data():
    rows = market_data_freshness_records(
        data_source="Synthetic",
        source_caption="synthetic price data",
        row_count=300,
        latest_label="299",
        minimum_rows=200,
    )
    checks = {row["Check"]: row for row in rows}

    assert checks["Data Source Identified"]["Passed"]
    assert checks["Enough Bars Loaded"]["Passed"]
    assert checks["Latest Bar Present"]["Passed"]
    assert "Synthetic data" in checks["Freshness Policy"]["Detail"]


def test_strategy_state_snapshot_records_current_signal_and_gate_state():
    rows = strategy_state_snapshot_records(
        config=StrategyConfig(entry_window=20, exit_window=10),
        live={"signal": "long", "last_p": 150.25},
        intent=TradeIntent(symbol="aapl", side="buy", quantity=10),
        risk_check=RiskCheckResult(True, [], {"risk": True}),
        preflight=PreflightCheckResult(True, {"preflight": True}, []),
    )
    values = {row["Field"]: row["Value"] for row in rows}

    assert values["Signal"] == "LONG"
    assert values["Reference Price"] == "$150.25"
    assert values["Proposed Symbol"] == "AAPL"
    assert values["Risk Approved"]
    assert values["Preflight Ready"]


def test_restart_recovery_records_require_configured_paths():
    rows = restart_recovery_records(
        audit_log_path="",
        broker_state_path="broker_state/orders.json",
        automation_dry_run_path="automation_logs/dry_runs.jsonl",
        run_manifest_path="audit_logs/run_manifests.jsonl",
        audit_records_loaded=2,
        tracked_orders_loaded=1,
        automation_snapshots_loaded=3,
    )
    checks = {row["Check"]: row for row in rows}

    assert not checks["Audit Log Path"]["Passed"]
    assert checks["Broker State Path"]["Passed"]
    assert checks["Automation Snapshots Recoverable"]["Detail"] == "3 snapshot(s) loaded."


def test_paper_account_health_records_surface_monitoring_breach():
    rows = paper_account_health_records(
        paper_cash=10_000,
        paper_equity=50_500,
        starting_cash=50_000,
        local_open_positions=1,
        tracked_alpaca_orders=[{"status": "accepted"}, {"status": "canceled"}],
        monitoring_result=MonitoringResult("BREACH", ["Kill switch is enabled."], {}),
        limits=RiskLimits(max_open_positions=5),
    )
    checks = {row["Check"]: row for row in rows}

    assert checks["Paper Cash Nonnegative"]["Passed"]
    assert checks["Open Positions Within Limit"]["Passed"]
    assert not checks["Monitoring Status"]["Passed"]


def test_scheduler_preview_never_submits_broker_writes():
    rows = scheduler_preview_records(
        interval_minutes=15,
        market_open=True,
        kill_switch_enabled=False,
        ready_candidate_count=1,
        halt_count=0,
    )
    values = {row["Field"]: row["Value"] for row in rows}

    assert values["Scheduler Mode"] == "preview_only"
    assert values["Would Evaluate Strategy"]
    assert values["Would Queue Manual Review"]
    assert values["Broker Writes Submitted"] == 0


def test_paper_automation_gate_blocks_without_dry_run_evidence():
    rows = paper_automation_gate_records(
        broker_connected=True,
        broker_state_stale=False,
        market_data_rows=[{"Check": "Enough Bars Loaded", "Passed": True}],
        account_health_rows=[{"Check": "Monitoring Status", "Passed": True}],
        restart_rows=[
            {"Check": "Audit Log Path", "Passed": True},
            {"Check": "Broker State Path", "Passed": True},
            {"Check": "Automation Dry Run Path", "Passed": True},
            {"Check": "Run Manifest Path", "Passed": True},
        ],
        readiness_rows=[{"Check": "Kill Switch Off", "Passed": True}],
        halt_rows=[{"Halt Reason": "None", "Active": False}],
        dry_run_snapshots_loaded=0,
    )
    checks = {row["Check"]: row for row in rows}

    assert checks["Broker Connected"]["Passed"]
    assert not checks["Ready For Paper Automation"]["Passed"]
    assert "Record Paper Automation Dry Run" in checks["Ready For Paper Automation"]["Detail"]
