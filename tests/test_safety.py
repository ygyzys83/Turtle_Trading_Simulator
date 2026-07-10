from agentloop_trader.brokers import AlpacaBrokerAdapterStub, AlpacaConfig
from agentloop_trader.models import RiskLimits, TradeIntent
from agentloop_trader.risk import check_trade_intent, decide_execution
from agentloop_trader.safety import (
    IMMUTABLE_AGENT_BOUNDARIES,
    broker_state_simulation_records,
    deployment_readiness_records,
    immutable_boundary_records,
    live_mode_lockfile_records,
    pre_live_readiness_report,
    production_readiness_checks,
    write_live_mode_lockfile,
)
from tests.test_brokers import FakeAlpacaClient


def test_safety_boundaries_include_broker_credentials_and_kill_switch():
    assert "broker_credentials" in IMMUTABLE_AGENT_BOUNDARIES
    assert "kill_switch" in IMMUTABLE_AGENT_BOUNDARIES
    assert all(not row["Agent Modifiable"] for row in immutable_boundary_records())


def test_production_readiness_checks_require_shadow_and_manual_live_before_unattended():
    records = production_readiness_checks()
    names = [record["Check"] for record in records]

    assert "Practice mode reviewed" in names
    assert "Automation check reviewed" in names
    assert "Paper exit tested" in names
    assert "Manual live approval tested" in names


def test_alpaca_live_submission_requires_explicit_live_confirmation():
    adapter = AlpacaBrokerAdapterStub(
        AlpacaConfig(api_key="key", api_secret="secret", paper=False),
        trading_client=FakeAlpacaClient(),
        allow_order_submission=True,
    )
    intent = TradeIntent(symbol="AAPL", side="buy", quantity=1, entry_price=100, stop_loss=95)
    risk = check_trade_intent(intent, 50_000, RiskLimits(allowed_symbols=("AAPL",)))
    decision = decide_execution("paper", risk)

    try:
        adapter.submit_order(intent, decision)
    except RuntimeError as exc:
        assert "live order submission is not enabled" in str(exc)
        return
    raise AssertionError("Live Alpaca submission must require live confirmation.")


def test_pre_live_readiness_report_is_evidence_based():
    rows = pre_live_readiness_report(
        paper_order_submitted=True,
        paper_cancel_submitted=True,
        paper_exit_tested=False,
        paper_fill_reconciled=False,
        automation_dry_run_recorded=True,
        performance_reviewed=False,
        emergency_disable_tested=False,
    )

    checks = {row["Check"]: row for row in rows}
    assert checks["Paper buy sent"]["Passed"]
    assert checks["Paper cancel sent"]["Passed"]
    assert checks["Paper exit sent"]["Status"] == "blocked"
    assert checks["Automation check saved"]["Passed"]
    assert checks["Live orders locked"]["Passed"]


def test_broker_state_simulation_records_include_disconnect_and_kill_switch():
    rows = broker_state_simulation_records()
    scenarios = {row["Scenario"] for row in rows}

    assert "Alpaca disconnected" in scenarios
    assert "Stop trading is on" in scenarios


def test_live_mode_lockfile_records_require_local_lockfile():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = f"{tmp_dir}/LIVE_TRADING_LOCKED.txt"
        missing = {row["Check"]: row for row in live_mode_lockfile_records(path)}
        written_path = write_live_mode_lockfile(path)
        present = {row["Check"]: row for row in live_mode_lockfile_records(path)}

    assert not missing["Live trading locked"]["Passed"]
    assert written_path.name == "LIVE_TRADING_LOCKED.txt"
    assert present["Live trading locked"]["Passed"]
    assert present["Live order wiring locked"]["Passed"]


def test_deployment_readiness_records_block_missing_lockfile():
    rows = deployment_readiness_records(
        env_example_present=True,
        dotenv_ignored=True,
        audit_path_configured=True,
        broker_state_path_configured=True,
        evidence_export_path_configured=True,
        live_lockfile_present=False,
    )
    checks = {row["Check"]: row for row in rows}

    assert checks["Example settings file exists"]["Passed"]
    assert checks["Live trading lock exists"]["Status"] == "blocked"
