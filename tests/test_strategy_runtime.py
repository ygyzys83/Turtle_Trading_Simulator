import pandas as pd

from agentloop_trader.models import TradeIntent
from agentloop_trader.strategy_runtime import (
    adjust_initial_stop_settings,
    apply_buy_order_style,
    evaluate_exit_settings,
    reprice_trade_intent,
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
    assert intent.entry_price == 99.25


def test_adjust_initial_stop_settings_rebuilds_planned_and_fill_adjusted_risk_distance():
    settings = {
        "entry_atr": 5.561451503208706,
        "planned_entry_price": 313.260009765625,
        "entry_stop_loss": 307.70,
        "entry_stop_distance": 5.56,
        "entry_stop_atr_multiplier": 1.0,
        "last_exit_trigger_price": 308.49,
        "last_exit_trigger_source": "fill-adjusted initial stop",
    }

    adjusted = adjust_initial_stop_settings(settings, 1.5)

    assert adjusted["entry_stop_atr_multiplier"] == 1.5
    assert adjusted["atr_stop_multiplier"] == 1.5
    assert adjusted["entry_stop_distance"] == 5.561451503208706 * 1.5
    assert adjusted["entry_stop_loss"] == 313.260009765625 - (5.561451503208706 * 1.5)
    assert "last_exit_trigger_price" not in adjusted
    assert "last_exit_trigger_source" not in adjusted


def test_apply_buy_order_style_supports_limit_below_current_price():
    intent = apply_buy_order_style(_intent(), "Limit below current price", adjustment_pct=0.5)

    assert intent.order_type == "limit"
    assert intent.limit_price == 99.5
    assert intent.entry_price == 99.5


def test_reprice_trade_intent_preserves_stop_distance():
    intent = reprice_trade_intent(_intent(), 103.0)

    assert intent.entry_price == 103.0
    assert intent.stop_loss == 98.0


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


def test_waiting_add_on_order_does_not_replace_filled_position_exit_plan():
    tracked = [
        {"symbol": "AAPL", "side": "buy", "status": "filled", "exit_settings": {"interval": "1h"}},
        {"symbol": "AAPL", "side": "buy", "status": "accepted", "exit_settings": {"interval": "5m"}},
    ]

    settings = saved_exit_settings_for_symbol("AAPL", tracked)

    assert settings["interval"] == "1h"


def test_combined_position_rebases_original_stop_using_saved_risk_distance(monkeypatch):
    index = pd.date_range("2026-07-09 15:00", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"Close": [109, 110, 111, 112], "High": [110, 111, 112, 113], "Low": [108, 109, 110, 111]},
        index=index,
    )
    monkeypatch.setattr(
        "agentloop_trader.strategy_runtime.selected_strategy_result",
        lambda market_data, settings, account: {"live": {"last_p": 112.0, "last_atr": 2.0, "exit_level": 90.0}},
    )
    settings = {
        "entry_reference_price": 100.0,
        "entry_stop_loss": 96.0,
        "entry_stop_distance": 4.0,
        "auto_exit_enabled": True,
    }

    details = evaluate_exit_settings(settings, {"Symbol": "AAPL", "Average Entry": "110"}, lambda *_: data)

    assert details["original_stop_price"] == 106.0


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


def test_profit_protection_stays_active_after_price_pulls_back(monkeypatch):
    index = pd.date_range("2026-07-09 15:00", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "Close": [100.0, 106.0, 110.0, 101.0],
            "High": [101.0, 107.0, 111.0, 102.0],
            "Low": [99.0, 105.0, 109.0, 100.0],
        },
        index=index,
    )
    monkeypatch.setattr(
        "agentloop_trader.strategy_runtime.selected_strategy_result",
        lambda market_data, settings, account: {"live": {"last_p": 101.0, "last_atr": 1.0, "exit_level": 90.0}},
    )
    settings = {
        "entry_submitted_at": "2026-07-09T15:00:00+00:00",
        "entry_stop_distance": 2.0,
        "breakeven_after_r": 1.0,
        "trail_after_r": 2.0,
        "trailing_atr_multiplier": 3.0,
        "profit_protection_enabled": True,
    }

    details = evaluate_exit_settings(settings, {"Symbol": "AAPL", "Average Entry": "100"}, lambda *_: data)

    assert details["profit_r"] == 0.5
    assert details["highest_profit_r"] == 5.5
    assert details["breakeven_stop_price"] == 100.0
    assert details["trailing_stop_price"] == 108.0
