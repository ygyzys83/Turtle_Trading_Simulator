from agentloop_trader.models import RiskLimits, TradeIntent
from agentloop_trader.risk import (
    build_preflight_check,
    check_trade_intent,
    constrain_trade_intent_to_limits,
    decide_execution,
)


def test_risk_rejects_trade_without_required_stop():
    intent = TradeIntent(
        symbol="AAPL",
        side="buy",
        quantity=10,
        entry_price=100,
        stop_loss=None,
    )

    result = check_trade_intent(
        intent,
        account_equity=50_000,
        limits=RiskLimits(allowed_symbols=("AAPL",), require_stop_loss=True),
    )

    assert not result.approved
    assert "Stop loss is required." in result.rejected_reasons


def test_backtest_only_mode_never_executes_even_when_risk_approved():
    intent = TradeIntent(
        symbol="AAPL",
        side="buy",
        quantity=10,
        entry_price=100,
        stop_loss=95,
    )
    risk = check_trade_intent(
        intent,
        account_equity=50_000,
        limits=RiskLimits(allowed_symbols=("AAPL",)),
    )
    decision = decide_execution("backtest_only", risk)

    assert risk.approved
    assert not decision.approved_for_execution
    assert not decision.requires_manual_approval


def test_live_with_approval_allows_manual_broker_gate_after_risk_passes():
    intent = TradeIntent(
        symbol="AAPL",
        side="buy",
        quantity=10,
        entry_price=100,
        stop_loss=95,
    )
    risk = check_trade_intent(
        intent,
        account_equity=50_000,
        limits=RiskLimits(allowed_symbols=("AAPL",)),
    )
    decision = decide_execution("live_with_approval", risk)

    assert risk.approved
    assert decision.approved_for_execution
    assert decision.requires_manual_approval
    assert "manual approval" in decision.reason.lower()


def test_preflight_and_execution_preserve_specific_risk_rejection_reason():
    intent = TradeIntent(
        symbol="AAPL",
        side="buy",
        quantity=1_000,
        entry_price=100,
        stop_loss=95,
    )
    risk = check_trade_intent(
        intent,
        account_equity=50_000,
        limits=RiskLimits(allowed_symbols=("AAPL",), max_position_notional_pct=10),
    )
    decision = decide_execution("shadow", risk)
    preflight = build_preflight_check(
        intent=intent,
        risk_check=risk,
        execution_decision=decision,
        broker_connected=True,
        audit_logging_enabled=True,
    )

    assert not risk.approved
    assert any("Estimated notional" in reason for reason in preflight.blocked_reasons)
    assert "Estimated notional" in decision.reason
    assert not any("Risk rules blocked this trade" in reason for reason in preflight.blocked_reasons)


def test_preflight_includes_execution_reason_when_risk_passes_but_mode_blocks():
    intent = TradeIntent(
        symbol="AAPL",
        side="buy",
        quantity=10,
        entry_price=100,
        stop_loss=95,
    )
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

    assert risk.approved
    assert "Backtest only does not send orders." in preflight.blocked_reasons


def test_risk_allows_adding_to_existing_position_when_enabled():
    intent = TradeIntent(
        symbol="AAPL",
        side="buy",
        quantity=10,
        entry_price=100,
        stop_loss=95,
    )

    blocked = check_trade_intent(
        intent,
        account_equity=50_000,
        limits=RiskLimits(allowed_symbols=("AAPL",)),
        open_positions={"AAPL"},
    )
    allowed = check_trade_intent(
        intent,
        account_equity=50_000,
        limits=RiskLimits(
            allowed_symbols=("AAPL",),
            allow_add_to_existing_position=True,
            max_position_notional_pct=100,
            max_symbol_concentration_pct=100,
        ),
        open_positions={"AAPL"},
    )

    assert not blocked.approved
    assert "AAPL already has an open position." in blocked.rejected_reasons
    assert allowed.approved
    assert allowed.checks["no_duplicate_position"]


def test_constrain_trade_intent_reduces_quantity_to_strictest_risk_limit():
    intent = TradeIntent(
        symbol="AAPL",
        side="buy",
        quantity=1_000,
        entry_price=100,
        stop_loss=95,
    )

    adjusted = constrain_trade_intent_to_limits(
        intent,
        account_equity=50_000,
        limits=RiskLimits(
            allowed_symbols=("AAPL",),
            max_risk_per_trade_pct=1,
            max_position_notional_pct=25,
            max_portfolio_exposure_pct=75,
            max_symbol_concentration_pct=35,
        ),
        current_portfolio_notional=0,
        symbol_current_notional=0,
        available_cash=50_000,
    )

    assert adjusted.quantity == 100
    assert "deterministic_risk_sizing" in adjusted.source_signals
    assert check_trade_intent(adjusted, 50_000, RiskLimits(allowed_symbols=("AAPL",))).approved


def test_constrain_trade_intent_respects_cash_and_notional_limits():
    intent = TradeIntent(
        symbol="AAPL",
        side="buy",
        quantity=1_000,
        entry_price=100,
        stop_loss=99,
    )

    adjusted = constrain_trade_intent_to_limits(
        intent,
        account_equity=50_000,
        limits=RiskLimits(
            allowed_symbols=("AAPL",),
            max_risk_per_trade_pct=5,
            max_position_notional_pct=50,
            max_portfolio_exposure_pct=50,
            max_symbol_concentration_pct=50,
        ),
        available_cash=12_000,
    )

    assert adjusted.quantity == 120


def test_constrain_trade_intent_returns_zero_when_no_quantity_is_allowed():
    intent = TradeIntent(
        symbol="AAPL",
        side="buy",
        quantity=10,
        entry_price=100,
        stop_loss=95,
    )

    adjusted = constrain_trade_intent_to_limits(
        intent,
        account_equity=50_000,
        limits=RiskLimits(allowed_symbols=("AAPL",), max_portfolio_exposure_pct=10),
        current_portfolio_notional=5_000,
        available_cash=50_000,
    )

    assert adjusted.quantity == 0
    risk = check_trade_intent(adjusted, 50_000, RiskLimits(allowed_symbols=("AAPL",)))
    assert not risk.approved
    assert "Quantity must be greater than zero." in risk.rejected_reasons


def test_risk_rejects_buy_stop_above_entry():
    intent = TradeIntent(symbol="AAPL", side="buy", quantity=10, entry_price=100, stop_loss=101)

    result = check_trade_intent(intent, 50_000, RiskLimits())

    assert not result.approved
    assert "Stop loss must be below the entry price for a buy order." in result.rejected_reasons


def test_risk_rejects_nonpositive_account_value():
    intent = TradeIntent(symbol="AAPL", side="buy", quantity=10, entry_price=100, stop_loss=95)

    result = check_trade_intent(intent, 0, RiskLimits())

    assert not result.approved
    assert "Account value must be greater than zero." in result.rejected_reasons


def test_deterministic_sizing_blocks_new_buys_after_daily_loss_limit():
    intent = TradeIntent(symbol="AAPL", side="buy", quantity=10, entry_price=100, stop_loss=95)
    adjusted = constrain_trade_intent_to_limits(
        intent,
        account_equity=97_000,
        limits=RiskLimits(max_session_loss_pct=2),
        session_pnl=-3_000,
    )

    assert adjusted.quantity == 0
    assert "daily loss limit" in adjusted.rationale
