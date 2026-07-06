from agentloop_trader.models import RiskLimits, TradeIntent
from agentloop_trader.risk import build_preflight_check, check_trade_intent, decide_execution
from agentloop_trader.shadow import record_shadow_decision, shadow_records


def test_shadow_decision_records_would_trade_context_without_order():
    intent = TradeIntent(symbol="AAPL", side="buy", quantity=10, entry_price=100, stop_loss=95)
    risk = check_trade_intent(intent, 50_000, RiskLimits(allowed_symbols=("AAPL",)))
    decision = decide_execution("shadow", risk)
    preflight = build_preflight_check(
        intent=intent,
        risk_check=risk,
        execution_decision=decision,
        broker_connected=True,
        audit_logging_enabled=True,
    )

    shadow = record_shadow_decision(intent, risk, decision, preflight)
    records = shadow_records([shadow])

    assert shadow.symbol == "AAPL"
    assert shadow.risk_approved
    assert not shadow.preflight_ready
    assert records[0]["Action"] == "BUY"
    assert records[0]["Time"].endswith(("PDT", "PST"))
    assert "manual approval" in records[0]["Blocked Reasons"].lower()


def test_shadow_decision_handles_absent_intent():
    risk = check_trade_intent(None, 50_000, RiskLimits())
    decision = decide_execution("shadow", risk)
    preflight = build_preflight_check(None, risk, decision, broker_connected=True, audit_logging_enabled=True)

    shadow = record_shadow_decision(None, risk, decision, preflight)

    assert shadow.symbol == "NONE"
    assert shadow.intended_action == "none"
    assert not shadow.risk_approved
