import pandas as pd

from agentloop_trader.models import TradeIntent
from agentloop_trader.strategy_runtime import (
    apply_buy_order_style,
    evaluate_exit_settings,
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


def test_exit_settings_high_water_mark_starts_at_entry_time(monkeypatch):
    index = pd.date_range("2026-07-09 15:00", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "Close": [100.0, 105.0, 110.0, 109.0],
            "High": [300.0, 106.0, 111.0, 110.0],
            "Low": [99.0, 104.0, 109.0, 108.0],
            "Volume": [1000, 1000, 1000, 1000],
        },
        index=index,
    )

    monkeypatch.setattr(
        "agentloop_trader.strategy_runtime.selected_strategy_result",
        lambda market_data, settings, account: {"live": {"last_p": 110.0, "last_atr": 1.0, "exit_level": 90.0}},
    )
    settings = {
        "symbol": "AAPL",
        "history": "1y",
        "interval": "1h",
        "price_data_source": "Ticker (Alpaca)",
        "entry_submitted_at": "2026-07-09T16:00:00+00:00",
        "entry_stop_loss": 98.0,
        "entry_stop_distance": 2.0,
        "trail_after_r": 2.0,
        "trailing_atr_multiplier": 3.0,
        "profit_protection_enabled": True,
    }

    details = evaluate_exit_settings(settings, {"Symbol": "AAPL", "Average Entry": "100"}, lambda *_: data)

    assert details["highest_high_since_entry"] == 111.0
    assert details["trailing_stop_price"] == 108.0
