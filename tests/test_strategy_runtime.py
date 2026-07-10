from agentloop_trader.models import TradeIntent
from agentloop_trader.strategy_runtime import (
    apply_buy_order_style,
    saved_exit_settings_for_symbol,
    trade_intent_from_record,
    trade_intent_to_record,
    update_exit_settings_for_symbol,
)


def _intent():
    return TradeIntent(symbol="AAPL", side="buy", quantity=10, entry_price=100.0, stop_loss=95.0)


def test_apply_buy_order_style_supports_custom_limit_price():
    intent = apply_buy_order_style(_intent(), "Custom limit price", custom_limit_price=99.25)

    assert intent.order_type == "limit"
    assert intent.limit_price == 99.25


def test_apply_buy_order_style_supports_limit_below_current_price():
    intent = apply_buy_order_style(_intent(), "Limit below current price", adjustment_pct=0.5)

    assert intent.order_type == "limit"
    assert intent.limit_price == 99.5


def test_trade_intent_record_roundtrip():
    record = trade_intent_to_record(_intent())

    loaded = trade_intent_from_record(record)

    assert loaded.symbol_clean == "AAPL"
    assert loaded.quantity == 10


def test_exit_settings_update_and_read_latest_for_symbol():
    tracked = [{"symbol": "AAPL", "side": "buy", "strategy_settings": {"interval": "1h"}}]
    updated = update_exit_settings_for_symbol("AAPL", tracked, {"interval": "15m", "auto_exit_enabled": True})

    settings = saved_exit_settings_for_symbol("AAPL", updated)

    assert settings["interval"] == "15m"
    assert settings["auto_exit_enabled"] is True
