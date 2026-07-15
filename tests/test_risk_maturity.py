from agentloop_trader.models import RiskLimits, TradeIntent
from agentloop_trader.risk import (
    build_preflight_check,
    check_trade_intent,
    decide_execution,
    preflight_records,
    risk_policy_records,
)


def _intent(symbol="AAPL", quantity=10, entry=100, stop=95):
    return TradeIntent(
        symbol=symbol,
        side="buy",
        quantity=quantity,
        entry_price=entry,
        stop_loss=stop,
    )


def test_risk_rejects_when_max_open_positions_would_be_exceeded():
    result = check_trade_intent(
        _intent(symbol="AAPL"),
        account_equity=50_000,
        limits=RiskLimits(allowed_symbols=("AAPL",), max_open_positions=2),
        open_positions={"MSFT", "GOOG"},
        open_position_count=2,
    )

    assert not result.approved
    assert any("Open positions" in reason for reason in result.rejected_reasons)


def test_risk_rejects_symbol_concentration_breach():
    result = check_trade_intent(
        _intent(quantity=100, entry=100, stop=95),
        account_equity=50_000,
        limits=RiskLimits(
            allowed_symbols=("AAPL",),
            max_position_notional_pct=100,
            max_symbol_concentration_pct=10,
        ),
    )

    assert not result.approved
    assert any("AAPL exposure" in reason for reason in result.rejected_reasons)


def test_risk_rejects_session_loss_breach():
    result = check_trade_intent(
        _intent(),
        account_equity=50_000,
        limits=RiskLimits(allowed_symbols=("AAPL",), max_session_loss_pct=2),
        session_pnl=-1_250,
    )

    assert not result.approved
    assert any("Daily loss" in reason for reason in result.rejected_reasons)


def test_daily_loss_limit_stops_new_buys_at_exact_prior_day_equity_threshold():
    result = check_trade_intent(
        _intent(),
        account_equity=98_000,
        limits=RiskLimits(allowed_symbols=("AAPL",), max_session_loss_pct=2),
        session_pnl=-2_000,
    )

    assert not result.approved
    assert not result.checks["session_loss_within_limit"]
    assert any("reached max $2,000.00" in reason for reason in result.rejected_reasons)


def test_preflight_reports_ready_when_risk_execution_broker_and_audit_pass():
    intent = _intent()
    risk = check_trade_intent(
        intent,
        account_equity=50_000,
        limits=RiskLimits(allowed_symbols=("AAPL",)),
    )
    decision = decide_execution("paper", risk)

    preflight = build_preflight_check(
        intent=intent,
        risk_check=risk,
        execution_decision=decision,
        broker_connected=True,
        audit_logging_enabled=True,
    )

    assert preflight.ready
    assert preflight_records(preflight)


def test_preflight_reports_blocked_when_execution_mode_blocks_order():
    intent = _intent()
    risk = check_trade_intent(
        intent,
        account_equity=50_000,
        limits=RiskLimits(allowed_symbols=("AAPL",)),
    )
    decision = decide_execution("backtest_only", risk)

    preflight = build_preflight_check(
        intent=intent,
        risk_check=risk,
        execution_decision=decision,
        broker_connected=True,
        audit_logging_enabled=True,
    )

    assert not preflight.ready
    assert any("Backtest only" in reason for reason in preflight.blocked_reasons)


def test_risk_policy_records_name_alpaca_target():
    records = risk_policy_records(RiskLimits(allowed_symbols=("AAPL",)))

    assert any(row["Policy"] == "Broker target" and "Alpaca" in row["Value"] for row in records)
