import numpy as np
import pandas as pd
import pytest

from agentloop_trader.management_chart import (
    build_management_chart,
    management_chart_history,
    position_level_explanation,
    position_price_levels,
    queued_price_levels,
    trim_management_data,
)


def market_frame(rows: int = 80) -> pd.DataFrame:
    close = np.linspace(100.0, 120.0, rows)
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 0.8,
            "Low": close - 0.9,
            "Close": close,
            "Volume": np.full(rows, 1000),
        },
        index=pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"),
    )


def test_position_chart_makes_governing_strategy_exit_unambiguous():
    levels = position_price_levels(
        entry_price=100.0,
        current_price=108.0,
        settings={"entry_stop_distance": 5.0},
        exit_details={
            "original_stop_price": 95.0,
            "strategy_exit_price": 103.0,
            "trigger_price": 103.0,
            "trigger_source": "strategy exit",
            "highest_high_since_entry": 110.0,
        },
    )

    names = [row["name"] for row in levels]
    assert "Fill-adjusted initial stop" in names
    assert "Strategy exit" not in names
    assert "ACTIVE PRICE STOP - strategy exit" in names
    active = next(row for row in levels if row["name"].startswith("ACTIVE PRICE STOP"))
    assert active["value"] == 103.0
    assert active["width"] > 2
    explanation = position_level_explanation({
        "original_stop_price": 95.0,
        "strategy_exit_price": 103.0,
        "trigger_price": 103.0,
        "trigger_source": "strategy exit",
    })
    assert "highest active protection level controls" in explanation
    assert "strategy exit $103.00" in explanation


def test_queue_chart_includes_numeric_buy_and_projected_stop():
    levels = queued_price_levels(
        current_price=100.0,
        settings={"atr_stop_multiplier": 1.5},
        live={"last_atr": 2.0},
        next_buy_level=105.0,
    )

    values = {row["name"]: row["value"] for row in levels}
    assert values["Next numeric BUY level"] == 105.0
    assert values["ATR-only stop reference if bought now"] == 97.0


def test_trendline_queue_level_is_labeled_as_the_completed_close_requirement():
    levels = queued_price_levels(
        current_price=99.0,
        settings={"atr_stop_multiplier": 1.5},
        live={
            "last_atr": 2.0,
            "trendline_breakout_level": 100.2,
            "trend_filter_level": 105.0,
        },
        next_buy_level=100.2,
    )

    names = [row["name"] for row in levels]
    assert "Required completed-close breakout" in names
    assert "Next numeric BUY level" not in names


def test_profit_levels_and_trend_filter_use_distinct_styles():
    levels = position_price_levels(
        entry_price=100.0,
        current_price=104.0,
        settings={"entry_stop_distance": 5.0},
        exit_details={"original_stop_price": 95.0, "trigger_price": 95.0, "trigger_source": "fill-adjusted initial stop"},
    )
    by_name = {level["name"]: level for level in levels}

    assert by_name["+1R profit level"]["color"] != by_name["+2R profit level"]["color"]
    assert by_name["+1R profit level"]["dash"] != by_name["+2R profit level"]["dash"]


def test_chart_history_is_shorter_but_indicator_warmup_is_preserved():
    assert management_chart_history("4h", "2y") == "1y"
    assert management_chart_history("1d", "5y") == "2y"
    data = market_frame(500)
    data.attrs["source"] = "test"

    trimmed = trim_management_data(
        data,
        {"moving_average_window": 300},
        visible_bars=90,
    )

    assert len(trimmed) == 395
    assert trimmed.attrs["source"] == "test"


def test_non_price_exit_does_not_mislabel_a_dollar_line():
    details = {
        "ready": True,
        "trigger_price": 95.0,
        "trigger_source": "RSI recovery exit",
        "price_trigger_price": 95.0,
        "price_trigger_source": "fill-adjusted initial stop",
        "original_stop_price": 95.0,
    }

    levels = position_price_levels(
        entry_price=100.0,
        current_price=105.0,
        settings={"entry_stop_distance": 5.0},
        exit_details=details,
    )

    names = [row["name"] for row in levels]
    assert "ACTIVE PRICE STOP - fill-adjusted initial stop" in names
    assert not any("RSI recovery exit" in name for name in names)
    explanation = position_level_explanation(details)
    assert "non-price exit is currently ready" in explanation
    assert "RSI recovery exit" in explanation


def test_non_price_exit_without_a_price_stop_is_still_explained():
    explanation = position_level_explanation({
        "ready": True,
        "trigger_price": None,
        "trigger_source": "RSI recovery exit",
    })

    assert "non-price exit is currently ready" in explanation
    assert "does not currently have an active dollar price stop" in explanation


def test_rsi_chart_shows_dynamic_setup_and_saved_exit_levels():
    data = market_frame()
    result = {
        "prices": data["Close"].to_numpy(),
        "smas": [],
        "live": {
            "rsi_setup_low": 22.0,
            "required_rsi_rebound_points": 12.0,
        },
    }

    figure = build_management_chart(
        data,
        {
            "strategy_type": "rsi_scalp",
            "rsi_length": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "_position_current_rsi": 58.0,
            "_position_rsi_sell_level": 63.0,
        },
        result,
        title="TEST | 15m | RSI mean-reversion scalp",
        static_levels=[],
    )

    names = [trace.name for trace in figure.data]
    assert "Current setup low (22)" in names
    assert "Minimum RSI rebound for BUY (34)" in names
    assert "Saved RSI sell trigger (63)" in names
    assert "Current RSI (58)" in names


def test_management_chart_draws_only_saved_strategy_context_and_levels():
    data = market_frame()
    result = {
        "prices": data["Close"].to_numpy(),
        "smas": data["Close"].rolling(20).mean().tolist(),
        "live": {
            "setup_type": "breakout",
            "last_atr": 2.0,
        },
    }
    settings = {
        "strategy_type": "breakout",
        "entry_window": 20,
        "exit_window": 10,
        "moving_average_window": 20,
    }

    figure = build_management_chart(
        data,
        settings,
        result,
        title="TEST | 1h | Breakout continuation",
        static_levels=[{
            "name": "Current quote",
            "value": 121.0,
            "color": "#ffffff",
            "dash": "dot",
            "width": 1.2,
        }],
    )

    names = [trace.name for trace in figure.data]
    assert "Completed price bars" in names
    assert "20-bar trend filter" in names
    assert "20-bar BUY breakout" in names
    assert "10-bar strategy exit" in names
    assert "Current quote ($121.00)" in names
    assert not any("trade" in name.lower() for name in names)


def test_atr_only_position_chart_hides_dormant_strategy_indicators():
    data = market_frame()
    result = {
        "prices": data["Close"].to_numpy(),
        "smas": data["Close"].rolling(20).mean().tolist(),
        "live": {},
    }

    figure = build_management_chart(
        data,
        {"strategy_type": "atr_only", "moving_average_window": 20},
        result,
        title="TEST | 1h | ATR protection only",
        static_levels=[],
    )

    assert [trace.name for trace in figure.data] == ["Completed price bars"]


def test_overlay_last_values_match_live_strategy_levels():
    data = market_frame()
    result = {
        "prices": data["Close"].to_numpy(),
        "smas": data["Close"].rolling(20).mean().tolist(),
        "live": {},
    }

    breakout = build_management_chart(
        data,
        {"strategy_type": "breakout", "entry_window": 20, "exit_window": 10, "moving_average_window": 20},
        result,
        title="Breakout",
        static_levels=[],
    )
    traces = {trace.name: trace for trace in breakout.data}
    assert traces["20-bar BUY breakout"].y[-1] == data["High"].iloc[-21:-1].max()
    assert traces["10-bar strategy exit"].y[-1] == data["Low"].iloc[-11:-1].min()

    pullback = build_management_chart(
        data,
        {
            "strategy_type": "pullback",
            "pullback_average_length": 20,
            "momentum_turn_length": 5,
            "exit_window": 10,
            "moving_average_window": 20,
        },
        result,
        title="Pullback",
        static_levels=[],
    )
    traces = {trace.name: trace for trace in pullback.data}
    assert traces["20-bar pullback average"].y[-1] == pytest.approx(data["Close"].iloc[-20:].mean())
    assert traces["5-bar momentum average"].y[-1] == pytest.approx(data["Close"].iloc[-5:].mean())
    assert traces["10-bar strategy exit"].y[-1] == pytest.approx(data["Close"].iloc[-10:].mean())


def test_trendline_chart_shows_selected_line_buffer_anchors_and_confirming_touches():
    data = market_frame()
    result = {
        "prices": data["Close"].to_numpy(),
        "smas": data["Close"].rolling(20).mean().tolist(),
        "atrs": [2.0] * len(data),
        "live": {
            "trendline_level": 118.0,
            "trendline_slope": -0.1,
            "trendline_anchor_indices": [20, 40],
            "trendline_touch_indices": [60],
            "trendline_tolerance_atr": 0.25,
            "trendline_breakout_buffer_atr": 0.10,
        },
    }

    figure = build_management_chart(
        data,
        {
            "strategy_type": "trendline",
            "entry_window": 20,
            "exit_window": 10,
            "moving_average_window": 20,
        },
        result,
        title="Trendline",
        static_levels=[],
    )

    names = [trace.name for trace in figure.data]
    assert "Selected descending trendline" in names
    assert "Allowed wick tolerance (0.25 ATR)" in names
    assert "Required completed-close breakout (line + 0.10 ATR)" in names
    assert "Trendline anchors" in names
    assert "Additional confirming touches" in names


def test_management_chart_displays_only_the_latest_ninety_bars_by_default():
    data = market_frame(220)
    result = {"prices": data["Close"].to_numpy(), "smas": [], "live": {}}

    figure = build_management_chart(
        data,
        {"strategy_type": "atr_only"},
        result,
        title="Compact chart",
        static_levels=[],
    )

    assert len(figure.data[0].x) == 90
    assert figure.layout.title.text == ""
    assert figure.layout.xaxis.type == "category"


def test_management_chart_accepts_independent_vertical_height():
    data = market_frame(180)
    result = {"prices": data["Close"].to_numpy(), "smas": [], "live": {}}

    figure = build_management_chart(
        data,
        {"strategy_type": "atr_only"},
        result,
        title="Tall chart",
        static_levels=[],
        height=900,
    )

    assert figure.layout.height == 900
