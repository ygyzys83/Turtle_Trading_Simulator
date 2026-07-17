from agentloop_trader.backtest import (
    _BacktestSessionRisk,
    _backtest_session_keys,
    _build_stats,
    _recent_descending_trendline,
    _trendline_crossed,
    DescendingTrendline,
    simulate_rsi_mean_reversion_strategy,
    simulate_trendline_breakout_strategy,
    simulate_trendline_retest_strategy,
    simulate_trend_pullback_strategy,
    simulate_turtle_strategy,
    strategy_comparison_records,
)
from agentloop_trader.models import RiskLimits
import agentloop_trader.backtest as backtest_module
import pandas as pd


def test_backtest_daily_loss_resets_on_each_trading_date():
    prices = list(range(100, 114)) + [105, 107, 109, 101, 103, 105, 97, 99]
    day_one = list(pd.date_range("2026-06-13 09:30", periods=16, freq="5min", tz="UTC"))
    day_two = list(pd.date_range("2026-06-14 09:30", periods=2, freq="5min", tz="UTC"))
    day_three = list(pd.date_range("2026-06-15 09:30", periods=4, freq="5min", tz="UTC"))
    market_data = pd.DataFrame(
        {
            "Open": prices,
            "Close": prices,
            "High": [price + 0.1 for price in prices],
            "Low": [price - 0.1 for price in prices],
            "Volume": [1_000_000] * len(prices),
        },
        index=day_one + day_two + day_three,
    )
    market_data.attrs["symbol"] = "RESET"
    limits = RiskLimits(
        max_risk_per_trade_pct=100,
        max_position_notional_pct=100,
        max_portfolio_exposure_pct=100,
        max_symbol_concentration_pct=100,
        max_session_loss_pct=0.01,
    )

    _, _, _, trade_log, _, _, _ = simulate_turtle_strategy(
        account=50_000,
        entry_w=2,
        exit_w=1,
        atr_mult=1.0,
        risk_pct_dec=0.01,
        ma_w=2,
        market_data=market_data,
        risk_limits=limits,
    )

    assert len(trade_log) >= 2
    assert trade_log[1]["entry_date"].startswith("2026-06-14")


def test_backtest_session_dates_use_exchange_day_for_equities_and_utc_for_crypto():
    market_data = pd.DataFrame(
        {"Close": [100, 101]},
        index=pd.to_datetime(["2026-07-02 00:30:00Z", "2026-07-02 14:30:00Z"]),
    )
    labels = ["2026-07-02 00:30", "2026-07-02 14:30"]

    assert _backtest_session_keys(market_data, labels, "equity") == ["2026-07-01", "2026-07-02"]
    assert _backtest_session_keys(market_data, labels, "crypto") == ["2026-07-02", "2026-07-02"]

    tracker = _BacktestSessionRisk(["2026-07-01", "2026-07-01", "2026-07-02"])
    assert tracker.update(0, 100_000) == 0
    assert tracker.update(1, 97_000) == -3_000
    assert tracker.update(2, 97_000) == 0


def test_turtle_backtest_returns_expected_contract():
    prices, smas, atrs, trade_log, live, stats, labels = simulate_turtle_strategy(
        account=50_000,
        entry_w=20,
        exit_w=10,
        atr_mult=2.0,
        risk_pct_dec=0.01,
        ma_w=200,
        seed=42,
        market_data=None,
    )

    assert len(prices) == len(smas) == len(atrs) == len(labels)
    assert isinstance(trade_log, list)
    assert live["signal"] in {"flat", "long", "exit"}
    assert stats["final_equity"] > 0
    assert "max_drawdown_pct" in stats
    assert "profit_factor" in stats
    assert "exposure_pct" in stats


def test_latest_bar_breakout_generates_trade_intent():
    prices = [100 - i for i in range(20)]
    prices[-1] = 90
    market_data = pd.DataFrame(
        {
            "Close": prices,
            "High": [p + 0.25 for p in prices],
            "Low": [p - 0.25 for p in prices],
        },
        index=pd.date_range("2026-01-01", periods=len(prices), freq="D"),
    )
    market_data.attrs["symbol"] = "TEST"

    _, _, _, _, live, _, _ = simulate_turtle_strategy(
        account=50_000,
        entry_w=3,
        exit_w=2,
        atr_mult=2.0,
        risk_pct_dec=0.01,
        ma_w=3,
        seed=None,
        market_data=market_data,
    )

    assert live["signal"] == "long"
    assert live["trade_intent"] is not None
    assert live["trade_intent"].symbol == "TEST"
    assert live["no_trade_reason"] == "BUY intent is present."
    assert all(live["buy_requirements"].values())


def test_simulated_open_trade_does_not_block_latest_buy_intent():
    prices = list(range(100, 150))
    market_data = pd.DataFrame(
        {
            "Close": prices,
            "High": [p + 0.25 for p in prices],
            "Low": [p - 0.25 for p in prices],
            "Volume": [1_000_000] * len(prices),
        },
        index=pd.date_range("2026-01-01", periods=len(prices), freq="D"),
    )
    market_data.attrs["symbol"] = "REAL"

    _, _, _, _, live, _, _ = simulate_turtle_strategy(
        account=50_000,
        entry_w=5,
        exit_w=3,
        atr_mult=2.0,
        risk_pct_dec=0.01,
        ma_w=5,
        seed=None,
        market_data=market_data,
    )

    assert live["in_simulated_trade"] is True
    assert live["signal"] == "long"
    assert live["trade_intent"] is not None
    assert "simulated open trade" not in live["buy_requirements"]
    assert live["no_trade_reason"] == "BUY intent is present."


def test_latest_bar_breakout_explains_missing_buy_intent():
    prices = [130 - i for i in range(30)]
    market_data = pd.DataFrame(
        {
            "Close": prices,
            "High": [p + 0.25 for p in prices],
            "Low": [p - 0.25 for p in prices],
            "Volume": [1_000_000] * len(prices),
        },
        index=pd.date_range("2026-01-01", periods=len(prices), freq="D"),
    )
    market_data.attrs["symbol"] = "WAIT"

    _, _, _, _, live, _, _ = simulate_turtle_strategy(
        account=50_000,
        entry_w=5,
        exit_w=3,
        atr_mult=2.0,
        risk_pct_dec=0.01,
        ma_w=5,
        seed=None,
        market_data=market_data,
    )

    assert live["trade_intent"] is None
    assert live["buy_requirements"]["Price above 5-bar high"] is False
    assert live["no_trade_reason"] == "No BUY because price is not above the 5-bar entry level."


def test_breakout_channel_uses_prior_highs_not_prior_closes():
    closes = list(range(100, 130))
    highs = [price + 0.25 for price in closes]
    highs[-2] = 140.0
    market_data = pd.DataFrame(
        {"Open": closes, "Close": closes, "High": highs, "Low": [price - 0.25 for price in closes]},
        index=pd.date_range("2026-01-01", periods=len(closes), freq="D"),
    )
    market_data.attrs["symbol"] = "CHANNEL"

    _, _, _, _, live, _, _ = simulate_turtle_strategy(
        account=50_000,
        entry_w=5,
        exit_w=3,
        atr_mult=2.0,
        risk_pct_dec=0.01,
        ma_w=5,
        market_data=market_data,
    )

    assert live["don_high"] == 140.0
    assert live["trade_intent"] is None


def test_real_market_volume_feeds_setup_quality():
    prices = list(range(100, 150))
    market_data = pd.DataFrame(
        {
            "Close": prices,
            "High": [p + 0.25 for p in prices],
            "Low": [p - 0.25 for p in prices],
            "Volume": [1_000_000] * (len(prices) - 1) + [2_000_000],
        },
        index=pd.date_range("2026-01-01", periods=len(prices), freq="D"),
    )
    market_data.attrs["symbol"] = "VOL"

    _, _, _, _, live, _, _ = simulate_turtle_strategy(
        account=50_000,
        entry_w=10,
        exit_w=5,
        atr_mult=2.0,
        risk_pct_dec=0.01,
        ma_w=20,
        seed=None,
        market_data=market_data,
    )

    assert live["volume_status"] in {"Normal", "Strong"}
    assert live["volume_confirmed"] is True
    assert live["liquidity_status"] in {"Good", "Usable", "Thin"}


def test_backtest_trade_rows_obey_position_notional_limit():
    prices = list(range(100, 135)) + [130, 125, 120, 115, 110, 105, 100]
    market_data = pd.DataFrame(
        {
            "Close": prices,
            "High": [p + 0.25 for p in prices],
            "Low": [p - 0.25 for p in prices],
            "Volume": [1_000_000] * len(prices),
        },
        index=pd.date_range("2026-01-01", periods=len(prices), freq="D"),
    )
    market_data.attrs["symbol"] = "CAP"
    limits = RiskLimits(
        max_risk_per_trade_pct=100,
        max_position_notional_pct=10,
        max_portfolio_exposure_pct=100,
        max_symbol_concentration_pct=100,
        max_session_loss_pct=50,
    )

    _, _, _, trade_log, _, _, _ = simulate_turtle_strategy(
        account=50_000,
        entry_w=5,
        exit_w=3,
        atr_mult=2.0,
        risk_pct_dec=0.03,
        ma_w=5,
        seed=None,
        market_data=market_data,
        risk_limits=limits,
    )

    assert trade_log
    first_trade = trade_log[0]
    assert first_trade["notional"] <= 50_000 * 0.10
    assert first_trade["shares"] == int((50_000 * 0.10) // first_trade["entry"])
    assert first_trade["price_risk_dollars"] == round((first_trade["entry"] - first_trade["stop"]) * first_trade["shares"], 2)
    assert first_trade["risk_dollars"] > first_trade["price_risk_dollars"]


def test_trend_pullback_backtest_returns_expected_contract():
    prices, smas, atrs, trade_log, live, stats, labels = simulate_trend_pullback_strategy(
        account=50_000,
        pullback_w=20,
        exit_w=10,
        atr_mult=2.0,
        risk_pct_dec=0.01,
        trend_w=50,
        momentum_w=5,
        seed=42,
        market_data=None,
    )

    assert len(prices) == len(smas) == len(atrs) == len(labels)
    assert isinstance(trade_log, list)
    assert live["strategy_name"] == "Trend pullback continuation"
    assert live["signal"] in {"flat", "long", "exit"}
    assert "rsi_status" in live
    assert stats["final_equity"] > 0


def test_latest_bar_trend_pullback_generates_trade_intent():
    prices = list(range(100, 150))
    prices[-8:] = [145, 143, 141, 139, 138, 137, 136, 140]
    market_data = pd.DataFrame(
        {
            "Close": prices,
            "High": [p + 0.25 for p in prices],
            "Low": [p - 0.25 for p in prices],
            "Volume": [1_000_000] * len(prices),
        },
        index=pd.date_range("2026-01-01", periods=len(prices), freq="D"),
    )
    market_data.attrs["symbol"] = "PULL"

    _, _, _, _, live, _, _ = simulate_trend_pullback_strategy(
        account=50_000,
        pullback_w=10,
        exit_w=5,
        atr_mult=2.0,
        risk_pct_dec=0.01,
        trend_w=20,
        momentum_w=3,
        seed=None,
        market_data=market_data,
    )

    assert live["signal"] == "long"
    assert live["trade_intent"] is not None
    assert live["trade_intent"].symbol == "PULL"
    expected_stop = min(min(prices[-10:]), prices[-1] - 2.0 * live["last_atr"])
    assert live["trade_intent"].stop_loss == round(expected_stop, 2)
    assert live["no_trade_reason"] == "BUY intent is present."
    assert all(live["buy_requirements"].values())


def test_pullback_sell_exit_length_changes_backtest_exits():
    prices = list(range(100, 150))
    prices[-16:] = [145, 143, 141, 139, 138, 137, 136, 140, 142, 144, 146, 145, 143, 141, 139, 137]
    market_data = pd.DataFrame(
        {
            "Close": prices,
            "High": [p + 0.25 for p in prices],
            "Low": [p - 0.25 for p in prices],
            "Volume": [1_000_000] * len(prices),
        },
        index=pd.date_range("2026-01-01", periods=len(prices), freq="D"),
    )
    market_data.attrs["symbol"] = "PULL"

    _, _, _, fast_exit_trades, fast_live, fast_stats, _ = simulate_trend_pullback_strategy(
        account=50_000,
        pullback_w=10,
        exit_w=3,
        atr_mult=2.0,
        risk_pct_dec=0.01,
        trend_w=20,
        momentum_w=3,
        seed=None,
        market_data=market_data,
    )
    _, _, _, slow_exit_trades, slow_live, slow_stats, _ = simulate_trend_pullback_strategy(
        account=50_000,
        pullback_w=10,
        exit_w=15,
        atr_mult=2.0,
        risk_pct_dec=0.01,
        trend_w=20,
        momentum_w=3,
        seed=None,
        market_data=market_data,
    )

    assert fast_live["sell_requirements"] == {"Price below 3-bar exit average": True}
    assert slow_live["sell_requirements"] == {"Price below 15-bar exit average": True}
    assert [trade["exit_date"] for trade in fast_exit_trades] != [trade["exit_date"] for trade in slow_exit_trades]
    assert fast_stats["total_pnl"] != slow_stats["total_pnl"]


def test_latest_bar_trendline_breakout_generates_trade_intent():
    prices = [100, 105, 110, 105, 100, 98, 104, 108, 103, 99, 97, 100, 103, 101, 98, 96, 97, 98, 95, 106]
    market_data = pd.DataFrame(
        {
            "Close": prices,
            "High": [p + 0.25 for p in prices],
            "Low": [p - 0.25 for p in prices],
            "Volume": [1_000_000] * len(prices),
        },
        index=pd.date_range("2026-01-01", periods=len(prices), freq="D"),
    )
    market_data.attrs["symbol"] = "TL"

    _, _, _, _, live, _, _ = simulate_trendline_breakout_strategy(
        account=50_000,
        trendline_w=15,
        exit_w=3,
        atr_mult=2.0,
        risk_pct_dec=0.01,
        ma_w=3,
        seed=None,
        market_data=market_data,
    )

    assert live["signal"] == "long"
    assert live["trade_intent"] is not None
    assert live["trade_intent"].symbol == "TL"
    assert live["buy_requirements"]["Touch-scored descending trendline found in last 15 bars"] is True
    assert live["buy_requirements"]["Completed bar crossed above buffered trendline"] is True
    assert live["trendline_breakout_level"] > live["trendline_level"]
    assert len(live["trendline_anchor_indices"]) == 2
    assert live["no_trade_reason"] == "BUY intent is present."


def test_latest_bar_trendline_retest_generates_trade_intent():
    prices = [100, 105, 110, 105, 100, 98, 104, 108, 103, 99, 97, 100, 103, 101, 98, 96, 94, 106, 98, 103]
    market_data = pd.DataFrame(
        {
            "Close": prices,
            "High": [p + 0.25 for p in prices],
            "Low": [p - 0.25 for p in prices],
            "Volume": [1_000_000] * len(prices),
        },
        index=pd.date_range("2026-01-01", periods=len(prices), freq="D"),
    )
    market_data.attrs["symbol"] = "TLR"

    _, _, _, _, live, _, _ = simulate_trendline_retest_strategy(
        account=50_000,
        trendline_w=15,
        exit_w=3,
        atr_mult=2.0,
        risk_pct_dec=0.01,
        ma_w=3,
        momentum_w=2,
        seed=None,
        market_data=market_data,
    )

    assert live["signal"] == "long"
    assert live["trade_intent"] is not None
    assert live["trade_intent"].symbol == "TLR"
    expected_stop = min(live["trendline_level"] - 0.25 * live["last_atr"], prices[-1] - 2.0 * live["last_atr"])
    assert live["trade_intent"].stop_loss == round(expected_stop, 2)
    assert live["buy_requirements"]["Retest held trendline"] is True
    assert live["retest_ready"] is True
    assert live["no_trade_reason"] == "BUY intent is present."


def test_trendline_breakout_requires_a_new_crossing_not_merely_price_above_line():
    prices = [100, 105, 110, 105, 100, 98, 104, 108, 103, 99, 97, 100, 103, 101, 98, 96, 97, 99, 100, 106]
    market_data = pd.DataFrame(
        {"Close": prices, "High": [p + 0.25 for p in prices], "Low": [p - 0.25 for p in prices]},
        index=pd.date_range("2026-01-01", periods=len(prices), freq="D"),
    )
    market_data.attrs["symbol"] = "TL"

    live = simulate_trendline_breakout_strategy(
        account=50_000, trendline_w=15, exit_w=3, atr_mult=2.0,
        risk_pct_dec=0.01, ma_w=3, market_data=market_data,
    )[4]

    assert live["signal"] == "flat"
    assert live["trendline_break"] is False


def test_touch_scored_trendline_prefers_the_line_confirmed_by_more_pivots():
    highs = [104.0 - 0.5 * index for index in range(25)]
    for index, value in ((2, 110.0), (8, 106.0), (14, 102.0), (20, 98.0)):
        highs[index] = value
    closes = [value - 1.0 for value in highs]
    atrs = [2.0] * len(highs)

    line = _recent_descending_trendline(
        highs, 24, 24, closes=closes, atrs=atrs
    )

    assert line is not None
    assert line.anchors == (2, 20)
    assert line.touch_indices == (8, 14)
    assert line.wick_violations == 0


def test_trendline_selection_does_not_look_at_uncompleted_future_pivots():
    highs = [90.0, 95.0, 110.0, 96.0, 94.0, 92.0, 90.0, 91.0, 106.0, 90.0,
             88.0, 87.0, 86.0, 85.0, 102.0, 84.0, 83.0, 82.0, 81.0, 80.0,
             150.0, 160.0, 170.0]
    closes = [value - 1.0 for value in highs]
    atrs = [2.0] * len(highs)

    original = _recent_descending_trendline(highs, 19, 19, closes=closes, atrs=atrs)
    changed_future = list(highs)
    changed_future[20:] = [300.0, 50.0, 400.0]
    repeated = _recent_descending_trendline(changed_future, 19, 19, closes=closes, atrs=atrs)

    assert original == repeated


def test_trendline_breakout_requires_the_atr_confirmation_buffer():
    line = DescendingTrendline(intercept=100.0, slope=0.0, anchors=(0, 1))
    atrs = [2.0, 2.0]

    below_buffer, required = _trendline_crossed([100.0, 100.1], line, 1, atrs)
    above_buffer, _ = _trendline_crossed([100.0, 100.3], line, 1, atrs)

    assert required == 100.2
    assert below_buffer is False
    assert above_buffer is True


def test_final_backtest_equity_includes_latest_open_position_value():
    prices = list(range(100, 140))
    prices[-1] = 150
    market_data = pd.DataFrame(
        {
            "Open": prices,
            "Close": prices,
            "High": [p + 0.25 for p in prices],
            "Low": [p - 0.25 for p in prices],
            "Volume": [1_000_000] * len(prices),
        },
        index=pd.date_range("2026-01-01", periods=len(prices), freq="D"),
    )
    market_data.attrs["symbol"] = "LATEST"

    _, _, _, _, live, stats, _ = simulate_turtle_strategy(
        account=50_000, entry_w=5, exit_w=3, atr_mult=2.0,
        risk_pct_dec=0.01, ma_w=5, market_data=market_data,
    )

    assert live["in_simulated_trade"] is True
    assert stats["final_equity"] == round(live["balance"], 2)


def test_rsi_profit_only_exit_waits_until_price_is_above_break_even(monkeypatch):
    prices = [100.0] * 14 + [99.0, 100.0, 99.0, 101.0, 101.0]
    data = pd.DataFrame(
        {
            "Open": prices,
            "Close": prices,
            "High": [price + 0.1 for price in prices],
            "Low": [price - 0.1 for price in prices],
            "Volume": [1_000_000] * len(prices),
        },
        index=pd.date_range("2026-01-01", periods=len(prices), freq="5min"),
    )
    data.attrs["symbol"] = "RSI"
    rsi_values = [50.0] * 14 + [30.0, 34.0, 70.0, 71.0, 60.0]
    monkeypatch.setattr(backtest_module, "calc_rsi", lambda values, length: rsi_values)

    ordinary = simulate_rsi_mean_reversion_strategy(
        50_000, 1.5, 0.005, rsi_length=5, rsi_swing_lookback=6,
        rsi_sell_recovery_points=35, rsi_max_holding_enabled=False,
        rsi_stop_mode="no_price_stop", rsi_profit_only_exit=False, market_data=data,
    )[3]
    profit_only = simulate_rsi_mean_reversion_strategy(
        50_000, 1.5, 0.005, rsi_length=5, rsi_swing_lookback=6,
        rsi_sell_recovery_points=35, rsi_max_holding_enabled=False,
        rsi_stop_mode="no_price_stop", rsi_profit_only_exit=True, market_data=data,
    )[3]

    assert ordinary[0]["exit_bar"] == 16
    assert ordinary[0]["exit"] == 99.0
    assert profit_only[0]["exit_bar"] == 17
    assert profit_only[0]["exit"] == 101.0


def test_strategy_comparison_records_are_display_ready():
    rows = strategy_comparison_records({
        "Breakout continuation": {"return_pct": 1.2, "total_trades": 3, "win_rate": 67, "max_drawdown_pct": 4.5, "profit_factor": 1.8},
        "Trend pullback continuation": {"return_pct": 2.4, "total_trades": 4, "win_rate": 75, "max_drawdown_pct": 3.0, "profit_factor": 2.1},
    })

    assert rows[0]["Strategy"] == "Breakout continuation"
    assert rows[1]["Allocated Return"] == "2.4%"
    assert rows[1]["Account Return"] == "2.4%"


def test_rsi_entry_rule_blocks_historical_and_current_entries_when_extended(monkeypatch):
    prices = list(range(100, 150))
    market_data = pd.DataFrame(
        {
            "Close": prices,
            "High": [p + 0.25 for p in prices],
            "Low": [p - 0.25 for p in prices],
            "Volume": [1_000_000] * len(prices),
        },
        index=pd.date_range("2026-01-01", periods=len(prices), freq="D"),
    )
    market_data.attrs["symbol"] = "RSI"
    monkeypatch.setattr(backtest_module, "calc_rsi", lambda values, length: [80.0] * len(values))

    without_filter = simulate_turtle_strategy(
        50_000, 5, 3, 2.0, 0.01, 5, market_data=market_data,
        rsi_entry_filter_enabled=False,
    )[4]
    with_filter = simulate_turtle_strategy(
        50_000, 5, 3, 2.0, 0.01, 5, market_data=market_data,
        rsi_entry_filter_enabled=True,
    )[4]

    assert without_filter["in_simulated_trade"] is True
    assert without_filter["trade_intent"] is not None
    assert with_filter["in_simulated_trade"] is False
    assert with_filter["trade_intent"] is None
    assert "RSI(14)" in with_filter["no_trade_reason"]
    assert any("required 50-70" in label for label in with_filter["buy_requirements"])


def test_backtest_stats_report_account_and_allocated_capital_separately():
    market_data = pd.DataFrame(
        {"Close": [100.0, 101.0]},
        index=pd.to_datetime(["2023-01-01", "2025-01-02"]),
    )
    stats = _build_stats(
        account=100_000,
        final_balance=100_500,
        trade_log=[],
        equity_curve=[100_000, 99_500, 100_500],
        exposure_bars=0,
        total_bars=2,
        risk_limits=RiskLimits(max_symbol_concentration_pct=5),
        market_data=market_data,
    )

    assert stats["return_pct"] == 0.5
    assert stats["allocated_capital"] == 5_000
    assert stats["allocated_return_pct"] == 10.0
    assert stats["allocated_max_drawdown_pct"] == 10.0
    assert stats["annualized_allocated_return_pct"] is not None


def test_all_four_strategies_call_the_same_rsi_entry_gate(monkeypatch):
    calls = []

    def reject_rsi(rsis, index, enabled):
        calls.append(enabled)
        return not enabled

    monkeypatch.setattr(backtest_module, "_rsi_entry_allowed", reject_rsi)
    runners = [
        lambda: simulate_turtle_strategy(50_000, 20, 10, 2.0, 0.01, 50, seed=42, rsi_entry_filter_enabled=True),
        lambda: simulate_trend_pullback_strategy(50_000, 20, 10, 2.0, 0.01, 50, 5, seed=42, rsi_entry_filter_enabled=True),
        lambda: simulate_trendline_breakout_strategy(50_000, 20, 10, 2.0, 0.01, 50, seed=42, rsi_entry_filter_enabled=True),
        lambda: simulate_trendline_retest_strategy(50_000, 20, 10, 2.0, 0.01, 50, 5, seed=42, rsi_entry_filter_enabled=True),
    ]

    for runner in runners:
        calls.clear()
        runner()
        assert calls
        assert all(calls)
