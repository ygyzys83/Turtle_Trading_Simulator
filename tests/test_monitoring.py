from agentloop_trader.models import RiskLimits
from agentloop_trader.monitoring import (
    MonitoringResult,
    broker_heartbeat_records,
    daily_risk_records,
    risk_halt_records,
)


def test_daily_risk_records_summarize_local_and_tracked_alpaca_state():
    rows = daily_risk_records(
        local_order_records=[
            {"Status": "filled", "Notional": 1200},
            {"Status": "rejected", "Notional": 500},
        ],
        tracked_alpaca_orders=[
            {"status": "accepted", "quantity": "40"},
            {"status": "filled", "filled_quantity": "10"},
            {"status": "canceled", "quantity": "5"},
        ],
        account_equity=50_000,
        session_pnl=-250,
        portfolio_exposure=10_000,
        limits=RiskLimits(max_portfolio_exposure_pct=75, max_session_loss_pct=2),
    )
    metrics = {row["Metric"]: row["Value"] for row in rows}

    assert metrics["Filled app paper orders"] == 1
    assert metrics["Saved Alpaca active/filled orders"] == 2
    assert metrics["Filled app order value"] == 1200
    assert metrics["Saved Alpaca shares"] == "50"
    assert metrics["Portfolio exposure %"] == 20
    assert metrics["Session loss %"] == 0.5


def test_broker_heartbeat_records_enforce_fresh_state_policy():
    rows = broker_heartbeat_records(
        broker_connected=True,
        broker_state_stale=True,
        broker_reasons=["Alpaca orders are unavailable."],
        market_advisory={"Open": False, "Message": "Market is closed."},
    )
    checks = {row["Check"]: row for row in rows}

    assert checks["Alpaca connected"]["Passed"]
    assert not checks["Alpaca data current"]["Passed"]
    assert not checks["Market open"]["Passed"]
    assert "refreshed" in checks["Alpaca data current"]["Policy"]


def test_risk_halt_records_surface_breaches_and_stale_state():
    rows = risk_halt_records(
        monitoring_result=MonitoringResult("BREACH", ["Stop trading switch is on."], {}),
        broker_connected=False,
        broker_state_stale=True,
        automation_ready_rows=[{"Check": "Stop trading off", "Passed": False, "Detail": "Stop trading is active."}],
    )
    reasons = {row["Block"] for row in rows}

    assert "Risk limit hit" in reasons
    assert "Alpaca disconnected" in reasons
    assert "Refresh Alpaca" in reasons
    assert "Stop trading off" in reasons
