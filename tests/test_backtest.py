from agentloop_trader.backtest import (
    simulate_trendline_breakout_strategy,
    simulate_trendline_retest_strategy,
    simulate_trend_pullback_strategy,
    simulate_turtle_strategy,
    strategy_comparison_records,
)
from agentloop_trader.models import RiskLimits
import pandas as pd


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
    assert first_trade["risk_dollars"] == round((first_trade["entry"] - first_trade["stop"]) * first_trade["shares"], 2)


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
    prices = [100, 105, 110, 105, 100, 98, 104, 108, 103, 99, 97, 100, 103, 101, 98, 96, 97, 99, 100, 106]
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
    assert live["buy_requirements"]["Descending trendline found in last 15 bars"] is True
    assert live["buy_requirements"]["Price above trendline"] is True
    assert live["no_trade_reason"] == "BUY intent is present."


def test_latest_bar_trendline_retest_generates_trade_intent():
    prices = [100, 105, 110, 105, 100, 98, 104, 108, 103, 99, 97, 100, 103, 101, 98, 96, 96, 96, 97, 97.4]
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
    assert live["buy_requirements"]["Retest held trendline"] is True
    assert live["retest_ready"] is True
    assert live["no_trade_reason"] == "BUY intent is present."


def test_strategy_comparison_records_are_display_ready():
    rows = strategy_comparison_records({
        "Breakout continuation": {"return_pct": 1.2, "total_trades": 3, "win_rate": 67, "max_drawdown_pct": 4.5, "profit_factor": 1.8},
        "Trend pullback continuation": {"return_pct": 2.4, "total_trades": 4, "win_rate": 75, "max_drawdown_pct": 3.0, "profit_factor": 2.1},
    })

    assert rows[0]["Strategy"] == "Breakout continuation"
    assert rows[1]["Return"] == "2.4%"
