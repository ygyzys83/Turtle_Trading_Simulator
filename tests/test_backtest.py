from agentloop_trader.backtest import simulate_turtle_strategy
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
