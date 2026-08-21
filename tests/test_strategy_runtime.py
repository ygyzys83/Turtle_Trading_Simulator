import pandas as pd
import pytest

from agentloop_trader.evaluation import synthetic_ohlc_frame
from agentloop_trader.models import TradeIntent
from agentloop_trader.strategy_runtime import (
    STRATEGY_TYPES,
    adjust_initial_stop_settings,
    apply_buy_order_style,
    exit_details_from_snapshot,
    exit_snapshot_from_details,
    evaluate_exit_settings,
    exit_plan_history_for_interval,
    exit_mode_for_settings,
    latest_atr_snapshot,
    normalize_managed_exit_settings,
    reprice_trade_intent,
    run_strategy_suite,
    selected_strategy_result,
    trade_intent_from_record,
    trade_intent_to_record,
)


def test_managed_crypto_exit_settings_use_the_canonical_alpaca_source():
    settings = normalize_managed_exit_settings(
        {"price_data_source": "Alpaca crypto", "auto_exit_enabled": True},
        {"Symbol": "BTC/USD", "Asset Type": "crypto"},
    )

    assert settings["asset_class"] == "crypto"
    assert settings["price_data_source"] == "Crypto (Alpaca)"


def test_selected_strategy_path_matches_the_same_strategy_in_the_full_suite():
    data = synthetic_ohlc_frame(seed=17).iloc[:360]
    settings = {
        "entry_window": 15,
        "exit_window": 10,
        "atr_stop_multiplier": 1.5,
        "risk_per_trade_pct": 0.5,
        "moving_average_window": 50,
        "pullback_average_length": 20,
        "momentum_turn_length": 10,
    }
    suite = run_strategy_suite(data, settings, 100_000)

    for label, strategy_type in STRATEGY_TYPES.items():
        selected = selected_strategy_result(
            data,
            {**settings, "strategy_type": strategy_type},
            100_000,
        )
        assert selected["stats"] == suite[label]["stats"]
        assert selected["trade_log"] == suite[label]["trade_log"]
        assert selected["live"] == suite[label]["live"]


def test_period_evaluation_start_bar_blocks_warmup_entries_for_every_strategy():
    data = synthetic_ohlc_frame(n=360, seed=19)
    data.attrs["_evaluation_start_bar"] = len(data) - 1
    settings = {
        "entry_window": 15,
        "exit_window": 10,
        "atr_stop_multiplier": 1.5,
        "risk_per_trade_pct": 0.5,
        "moving_average_window": 50,
        "pullback_average_length": 20,
        "momentum_turn_length": 10,
        "rsi_length": 14,
        "rsi_swing_lookback": 24,
    }

    for strategy_type in STRATEGY_TYPES.values():
        result = selected_strategy_result(
            data,
            {**settings, "strategy_type": strategy_type},
            100_000,
        )
        assert result["trade_log"] == []
        assert result["stats"]["total_trades"] == 0


def test_exit_plan_history_matches_selected_position_interval():
    assert exit_plan_history_for_interval("5m") == "1mo"
    assert exit_plan_history_for_interval("4h") == "2y"
    assert exit_plan_history_for_interval("1d") == "5y"


def test_latest_atr_snapshot_uses_latest_completed_bar():
    index = pd.date_range("2026-07-15 16:00", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "Close": [10.0, 11.0, 12.0, 13.0],
            "High": [11.0, 12.0, 13.0, 14.0],
            "Low": [9.0, 10.0, 11.0, 12.0],
        },
        index=index,
    )

    atr, measured_at = latest_atr_snapshot(data, length=3)

    assert atr == pytest.approx(2.0)
    assert measured_at == index[-1].isoformat()


def test_latest_atr_snapshot_rejects_incomplete_history():
    data = pd.DataFrame({"Close": [10.0], "High": [11.0], "Low": [9.0]})

    with pytest.raises(ValueError, match="Not enough completed bars"):
        latest_atr_snapshot(data, length=14)


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


def test_exit_uses_live_alpaca_position_price_when_completed_bar_has_not_reached_stop(monkeypatch):
    index = pd.date_range("2026-07-16 15:00", periods=3, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "Close": [77.50, 76.80, 76.10],
            "High": [78.00, 77.20, 76.50],
            "Low": [77.00, 76.40, 75.90],
        },
        index=index,
    )
    data.attrs["latest_price"] = 76.10
    monkeypatch.setattr(
        "agentloop_trader.strategy_runtime.selected_strategy_result",
        lambda market_data, settings, account: {
            "live": {"last_p": 76.10, "last_atr": 1.84, "exit_level": None}
        },
    )
    settings = {
        "exit_mode": "atr_only",
        "entry_stop_distance": 1.8368117749680912,
        "auto_exit_enabled": True,
    }
    position = {
        "Symbol": "CRWV",
        "Quantity": "64",
        "Market Value": str(64 * 73.20),
        "Average Entry": "77.08",
    }

    details = evaluate_exit_settings(settings, position, lambda *_: data)

    assert details["current_price"] == pytest.approx(73.20)
    assert details["current_price_source"] == "Alpaca position market value"
    assert details["trigger_price"] == pytest.approx(75.2431882250319)
    assert details["trigger_source"] == "fill-adjusted initial stop"
    assert details["ready"] is True
    assert "CRWV is at or below" in details["reason"]


def test_saved_exit_snapshot_renders_with_live_position_price_without_history_download():
    settings = {
        "auto_exit_enabled": True,
        "exit_mode": "atr_only",
        "entry_stop_distance": 1.84,
        "last_exit_trigger_price": 75.24,
        "last_exit_trigger_source": "fill-adjusted initial stop",
        "last_exit_snapshot": exit_snapshot_from_details({
            "ready": False,
            "reason": "Hold.",
            "current_price": 76.10,
            "current_price_source": "Alpaca position market value",
            "trigger_price": 75.24,
            "trigger_source": "fill-adjusted initial stop",
            "original_stop_price": 75.24,
            "profit_r": -0.53,
            "checked_at": "2026-07-16T12:00:00-07:00",
        }),
    }
    position = {
        "Symbol": "CRWV",
        "Quantity": "64",
        "Market Value": str(64 * 73.20),
        "Average Entry": "77.08",
    }

    details = exit_details_from_snapshot(settings, position)

    assert details["snapshot_available"] is True
    assert details["current_price"] == pytest.approx(73.20)
    assert details["ready"] is True
    assert details["trigger_price"] == pytest.approx(75.24)
    assert details["snapshot_checked_at"] == "2026-07-16T12:00:00-07:00"


def test_saved_exit_settings_without_worker_snapshot_still_show_initial_stop():
    settings = {
        "auto_exit_enabled": True,
        "exit_mode": "atr_only",
        "entry_stop_distance": 2.0,
        "interval": "4h",
    }
    position = {
        "Symbol": "IBM",
        "Quantity": "10",
        "Market Value": "2200",
        "Average Entry": "218",
    }

    details = exit_details_from_snapshot(settings, position)

    assert details["snapshot_available"] is False
    assert details["original_stop_price"] == pytest.approx(216)
    assert details["trigger_price"] == pytest.approx(216)
    assert details["ready"] is False
    assert "Start the Background Worker" in details["reason"]


def test_saved_trigger_without_source_cannot_override_initial_stop():
    settings = {
        "auto_exit_enabled": True,
        "exit_mode": "strategy_and_atr",
        "entry_stop_distance": 5934.63,
        "last_exit_trigger_price": 63873.23,
        "interval": "4h",
    }
    position = {
        "Symbol": "BTC/USD",
        "Quantity": "0.076884498",
        "Market Value": str(0.076884498 * 64162.99),
        "Average Entry": "64172.62",
    }

    details = exit_details_from_snapshot(settings, position)

    assert details["snapshot_available"] is False
    assert details["original_stop_price"] == pytest.approx(58237.99)
    assert details["trigger_price"] == pytest.approx(58237.99)
    assert details["trigger_source"] == "fill-adjusted initial stop"
    assert details.get("strategy_exit_price") is None


def test_saved_strategy_trigger_without_snapshot_remains_explicit():
    settings = {
        "auto_exit_enabled": True,
        "exit_mode": "strategy_and_atr",
        "entry_stop_distance": 5934.63,
        "last_exit_trigger_price": 63873.23,
        "last_exit_trigger_source": "strategy exit",
        "interval": "4h",
    }
    position = {
        "Symbol": "BTC/USD",
        "Quantity": "0.076884498",
        "Market Value": str(0.076884498 * 64162.99),
        "Average Entry": "64172.62",
    }

    details = exit_details_from_snapshot(settings, position)

    assert details["snapshot_available"] is False
    assert details["original_stop_price"] == pytest.approx(58237.99)
    assert details["strategy_exit_price"] == pytest.approx(63873.23)
    assert details["trigger_price"] == pytest.approx(63873.23)
    assert details["trigger_source"] == "strategy exit"


def test_snapshot_keeps_price_protection_separate_from_non_price_exit():
    settings = {
        "auto_exit_enabled": True,
        "entry_stop_distance": 5.0,
        "last_exit_snapshot": {
            "ready": True,
            "trigger_price": 95.0,
            "trigger_source": "RSI recovery exit",
            "price_trigger_price": 95.0,
            "price_trigger_source": "fill-adjusted initial stop",
            "original_stop_price": 95.0,
        },
    }
    position = {
        "Symbol": "TEST",
        "Quantity": "10",
        "Market Value": "1050",
        "Average Entry": "100",
    }

    details = exit_details_from_snapshot(settings, position)

    assert details["ready"] is True
    assert details["trigger_source"] == "RSI recovery exit"
    assert details["price_trigger_price"] == 95.0
    assert details["price_trigger_source"] == "fill-adjusted initial stop"


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
        "auto_exit_enabled": True,
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
        "auto_exit_enabled": True,
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


def test_atr_only_manual_position_ignores_strategy_exit_above_current_price(monkeypatch):
    index = pd.date_range("2026-07-14 15:00", periods=3, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"Close": [219.0, 218.8, 218.6], "High": [220.0, 219.2, 219.0], "Low": [218.0, 218.0, 218.2]},
        index=index,
    )
    monkeypatch.setattr(
        "agentloop_trader.strategy_runtime.selected_strategy_result",
        lambda market_data, settings, account: {
            "live": {"last_p": 218.6, "last_atr": 6.24, "exit_level": 287.70, "buy_requirements": {}}
        },
    )
    settings = {
        "exit_mode": "atr_only",
        "entry_stop_distance": 12.48,
        "auto_exit_enabled": True,
    }

    details = evaluate_exit_settings(settings, {"Symbol": "IBM", "Average Entry": "218.92"}, lambda *_: data)

    assert details["ready"] is False
    assert details["strategy_exit_price"] is None
    assert details["trigger_price"] == 206.44
    assert details["trigger_source"] == "fill-adjusted initial stop"


def test_older_manual_order_without_exit_mode_migrates_to_atr_only():
    assert exit_mode_for_settings({"entry_source": "manual order"}) == "atr_only"
    assert exit_mode_for_settings({"entry_source": "worker queue"}) == "strategy_and_atr"


def test_existing_strategy_position_keeps_strategy_exit_behavior(monkeypatch):
    index = pd.date_range("2026-07-14 15:00", periods=3, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"Close": [219.0, 218.8, 218.6], "High": [220.0, 219.2, 219.0], "Low": [218.0, 218.0, 218.2]},
        index=index,
    )
    monkeypatch.setattr(
        "agentloop_trader.strategy_runtime.selected_strategy_result",
        lambda market_data, settings, account: {
            "live": {"last_p": 218.6, "last_atr": 6.24, "exit_level": 287.70, "buy_requirements": {}}
        },
    )
    settings = {"entry_stop_distance": 12.48, "auto_exit_enabled": True}

    details = evaluate_exit_settings(settings, {"Symbol": "IBM", "Average Entry": "218.92"}, lambda *_: data)

    assert details["ready"] is True
    assert details["trigger_price"] == 287.70
    assert details["trigger_source"] == "strategy exit"


def test_partial_entry_bar_high_does_not_false_trigger_break_even(monkeypatch):
    index = pd.DatetimeIndex([pd.Timestamp("2026-07-14T16:00:00+00:00")])
    data = pd.DataFrame(
        {"Close": [32.13], "High": [38.87], "Low": [31.90]},
        index=index,
    )
    monkeypatch.setattr(
        "agentloop_trader.strategy_runtime.selected_strategy_result",
        lambda market_data, settings, account: {
            "live": {"last_p": 32.13, "last_atr": 3.53, "exit_level": 30.0, "buy_requirements": {}}
        },
    )
    settings = {
        "exit_mode": "atr_only",
        "entry_filled_at": "2026-07-14T17:44:58+00:00",
        "entry_stop_distance": 3.53,
        "highest_high_since_entry": 38.87,
        "last_exit_trigger_price": 32.09,
        "last_exit_trigger_source": "break-even stop",
        "breakeven_after_r": 1.0,
        "trail_after_r": 2.0,
        "profit_protection_enabled": True,
        "auto_exit_enabled": True,
    }

    details = evaluate_exit_settings(settings, {"Symbol": "WYFI", "Average Entry": "32.09"}, lambda *_: data)

    assert details["highest_high_since_entry"] == 32.13
    assert details["highest_profit_r"] < 1.0
    assert details["breakeven_stop_price"] is None
    assert details["trigger_price"] == pytest.approx(28.56)
    assert details["trigger_source"] == "fill-adjusted initial stop"
    assert details["state_changed"] is True


def test_rsi_profit_only_exit_waits_above_fee_adjusted_break_even(monkeypatch):
    index = pd.date_range("2026-07-14 15:00", periods=3, freq="5min", tz="UTC")
    data = pd.DataFrame(
        {"Close": [100.0, 99.5, 99.0], "High": [100.2, 99.7, 99.2], "Low": [99.8, 99.3, 98.8]},
        index=index,
    )
    monkeypatch.setattr(
        "agentloop_trader.strategy_runtime.selected_strategy_result",
        lambda market_data, settings, account: {
            "live": {"last_p": 99.0, "last_atr": 1.0, "rsi": 70.0, "exit_level": None}
        },
    )
    settings = {
        "strategy_type": "rsi_scalp",
        "rsi_stop_mode": "no_price_stop",
        "entry_rsi_setup_low": 30.0,
        "rsi_sell_recovery_points": 35.0,
        "rsi_overbought": 70.0,
        "rsi_profit_only_exit": True,
        "rsi_max_holding_enabled": False,
        "auto_exit_enabled": True,
        "asset_class": "equity",
    }
    position = {"Symbol": "AAPL", "Asset Type": "equity", "Quantity": "10", "Average Entry": "100"}

    details = evaluate_exit_settings(settings, position, lambda *_: data)

    assert details["rsi_exit_signal_ready"] is True
    assert details["rsi_profit_condition_ready"] is False
    assert details["rsi_exit_ready"] is False
    assert details["ready"] is False
    assert details["rsi_fee_adjusted_break_even"] > 100
    assert "not above the estimated fee-adjusted break-even" in details["reason"]


def test_atr_stop_remains_independent_of_rsi_profit_requirement(monkeypatch):
    index = pd.date_range("2026-07-14 15:00", periods=3, freq="5min", tz="UTC")
    data = pd.DataFrame(
        {"Close": [100.0, 96.0, 94.0], "High": [100.2, 96.2, 94.2], "Low": [99.8, 95.8, 93.8]},
        index=index,
    )
    monkeypatch.setattr(
        "agentloop_trader.strategy_runtime.selected_strategy_result",
        lambda market_data, settings, account: {
            "live": {"last_p": 94.0, "last_atr": 1.0, "rsi": 40.0, "exit_level": None}
        },
    )
    settings = {
        "strategy_type": "rsi_scalp",
        "rsi_stop_mode": "standard_atr",
        "entry_stop_distance": 5.0,
        "entry_rsi_setup_low": 30.0,
        "rsi_profit_only_exit": True,
        "rsi_max_holding_enabled": False,
        "auto_exit_enabled": True,
    }

    details = evaluate_exit_settings(
        settings,
        {"Symbol": "AAPL", "Quantity": "10", "Average Entry": "100"},
        lambda *_: data,
    )

    assert details["ready"] is True
    assert details["trigger_source"] == "fill-adjusted initial stop"
