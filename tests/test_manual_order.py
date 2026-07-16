import pytest

from agentloop_trader.brokers import AlpacaConfig, build_alpaca_order_preview
from agentloop_trader.manual_order import build_manual_buy_intent
from agentloop_trader.models import ExecutionDecision, RiskCheckResult


def test_manual_crypto_buy_uses_fractional_quantity_gtc_and_atr_stop():
    intent = build_manual_buy_intent(
        symbol="BTCUSD",
        asset_class="crypto",
        current_price=60_000,
        atr=1_000,
        stop_atr_multiplier=1.5,
        order_type="limit",
        requested_dollars=5_000,
        limit_price=59_500,
    )

    assert intent.symbol_clean == "BTC/USD"
    assert intent.quantity == pytest.approx(0.08403361)
    assert intent.time_in_force == "gtc"
    assert intent.limit_price == 59_500
    assert intent.stop_loss == 58_000
    assert intent.proposed_by_agent == "manual_order"


def test_manual_equity_buy_accepts_quantity_and_market_price():
    intent = build_manual_buy_intent(
        symbol="AAPL",
        asset_class="equity",
        current_price=200,
        atr=4,
        stop_atr_multiplier=1.5,
        order_type="market",
        requested_quantity=12,
    )

    assert intent.quantity == 12
    assert intent.time_in_force == "day"
    assert intent.limit_price is None
    assert intent.entry_price == 200
    assert intent.stop_loss == 194


def test_manual_buy_requires_atr_for_saved_risk_plan():
    with pytest.raises(ValueError, match="ATR"):
        build_manual_buy_intent(
            symbol="BTC/USD",
            asset_class="crypto",
            current_price=60_000,
            atr=0,
            stop_atr_multiplier=1.5,
            order_type="market",
            requested_dollars=1_000,
        )


def test_manual_order_preview_identifies_manual_source():
    intent = build_manual_buy_intent(
        symbol="BTC/USD",
        asset_class="crypto",
        current_price=60_000,
        atr=1_000,
        stop_atr_multiplier=1.5,
        order_type="market",
        requested_dollars=1_000,
    )
    risk = RiskCheckResult(True, [], {"approved": True})
    decision = ExecutionDecision("paper", True, False, "Approved", risk)

    preview = build_alpaca_order_preview(intent, decision, AlpacaConfig("key", "secret", paper=True))

    assert preview.valid
    assert preview.order["source"] == "manual_order"
