import numpy as np
import pandas as pd

from agentloop_trader.backtest import _rsi_scalp_live_setup, simulate_rsi_mean_reversion_strategy
from agentloop_trader.buy_watchlist import BuyWatchPlan, buy_watch_plan_detail_records
from agentloop_trader.models import RiskLimits
from agentloop_trader.parameter_loop import generate_optimizer_settings
from agentloop_trader.strategy_runtime import STRATEGY_TYPES, evaluate_exit_settings, selected_strategy_result


def oscillating_frame(bars: int = 320) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=bars, freq="15min", tz="UTC")
    close = 100 + np.sin(np.arange(bars) / 3.0) * 7 + np.sin(np.arange(bars) / 11.0) * 2
    data = pd.DataFrame({
        "Open": close,
        "High": close + 0.5,
        "Low": close - 0.5,
        "Close": close,
        "Volume": np.full(bars, 1_000_000),
    }, index=index)
    data.attrs.update({"symbol": "TEST", "asset_class": "equity"})
    return data


def rsi_settings() -> dict:
    return {
        "strategy_label": "RSI mean-reversion scalp",
        "strategy_type": "rsi_scalp",
        "atr_stop_multiplier": 1.5,
        "risk_per_trade_pct": 0.5,
        "rsi_length": 7,
        "rsi_oversold": 35.0,
        "rsi_overbought": 70.0,
        "rsi_decline_points": 25.0,
        "rsi_rebound_points": 3.0,
        "rsi_max_rebound_points": 12.0,
        "rsi_sell_recovery_points": 30.0,
        "rsi_swing_lookback": 24,
        "rsi_stop_mode": "standard_atr",
        "rsi_emergency_atr_multiplier": 5.0,
        "rsi_max_holding_enabled": True,
        "rsi_max_holding_bars": 100,
    }


def test_rsi_scalp_is_a_first_class_strategy_and_backtests():
    assert STRATEGY_TYPES["RSI mean-reversion scalp"] == "rsi_scalp"
    result = selected_strategy_result(oscillating_frame(), rsi_settings(), 100_000)

    assert result["live"]["strategy_name"] == "RSI mean-reversion scalp"
    assert result["live"]["setup_type"] == "rsi_scalp"
    assert result["live"]["rsi_sell_level"] is not None
    assert result["stats"]["total_trades"] > 0
    assert all("entry_rsi_setup_low" in trade for trade in result["trade_log"])


def test_rsi_scalp_rejects_a_completed_bar_that_rebounded_too_far():
    ready, armed, setup_low, _, _, rebound, too_large = _rsi_scalp_live_setup(
        prices=[100.0, 90.0, 89.0, 95.0],
        rsis=[60.0, 20.0, 23.0, 35.0],
        index=3,
        oversold=30.0,
        decline_points=40.0,
        rebound_points=3.0,
        max_rebound_points=12.0,
        swing_lookback=24,
    )

    assert ready is False
    assert armed is False
    assert setup_low == 20.0
    assert rebound == 15.0
    assert too_large is True


def test_rsi_scalp_uses_rsi_recovery_or_time_exit_with_atr_protection():
    result = simulate_rsi_mean_reversion_strategy(
        account=100_000,
        atr_mult=1.5,
        risk_pct_dec=0.005,
        rsi_length=7,
        rsi_oversold=35,
        rsi_overbought=70,
        rsi_decline_points=25,
        rsi_rebound_points=3,
        rsi_sell_recovery_points=30,
        rsi_swing_lookback=24,
        rsi_max_holding_enabled=True,
        rsi_max_holding_bars=12,
        market_data=oscillating_frame(),
    )
    trade_log = result[3]

    assert trade_log
    assert all(trade["stop"] < trade["entry"] for trade in trade_log)
    assert any("RSI recovered" in trade["exit_rule"] or "Maximum hold" in trade["exit_rule"] for trade in trade_log)


def test_rsi_scalp_can_disable_maximum_holding_period():
    result = simulate_rsi_mean_reversion_strategy(
        account=100_000,
        atr_mult=1.5,
        risk_pct_dec=0.005,
        rsi_length=7,
        rsi_oversold=35,
        rsi_overbought=70,
        rsi_decline_points=25,
        rsi_rebound_points=3,
        rsi_sell_recovery_points=30,
        rsi_swing_lookback=24,
        rsi_max_holding_enabled=False,
        rsi_max_holding_bars=1,
        market_data=oscillating_frame(),
    )

    assert result[3]
    assert all("Maximum hold" not in trade["exit_rule"] for trade in result[3])
    assert result[4]["trade_intent"] is None or result[4]["trade_intent"].max_holding_bars is None


def test_rsi_stop_modes_use_selected_protection_and_fixed_allocation():
    common = {
        "account": 100_000,
        "atr_mult": 1.5,
        "risk_pct_dec": 0.005,
        "rsi_length": 7,
        "rsi_oversold": 35,
        "rsi_overbought": 70,
        "rsi_decline_points": 25,
        "rsi_rebound_points": 3,
        "rsi_sell_recovery_points": 30,
        "rsi_swing_lookback": 24,
        "rsi_emergency_atr_multiplier": 5.0,
        "rsi_max_holding_enabled": True,
        "rsi_max_holding_bars": 100,
        "market_data": oscillating_frame(),
        "risk_limits": RiskLimits(
            max_position_notional_pct=5,
            max_symbol_concentration_pct=5,
            max_portfolio_exposure_pct=80,
        ),
    }
    standard = simulate_rsi_mean_reversion_strategy(rsi_stop_mode="standard_atr", **common)
    emergency = simulate_rsi_mean_reversion_strategy(rsi_stop_mode="emergency_atr", **common)
    no_stop = simulate_rsi_mean_reversion_strategy(rsi_stop_mode="no_price_stop", **common)

    assert standard[3] and emergency[3] and no_stop[3]
    assert emergency[3][0]["stop"] < standard[3][0]["stop"]
    assert all(trade["stop"] == 0 for trade in no_stop[3])
    equity_before_trade = 100_000.0
    for trade in no_stop[3]:
        assert trade["notional"] <= equity_before_trade * 0.05 + 1.0
        equity_before_trade += trade["pnl"]


def test_saved_rsi_setup_low_drives_worker_exit():
    data = oscillating_frame(120)
    data.loc[:, "Close"] = np.linspace(90, 120, len(data))
    data.loc[:, "Open"] = data["Close"]
    data.loc[:, "High"] = data["Close"] + 0.5
    data.loc[:, "Low"] = data["Close"] - 0.5

    settings = rsi_settings() | {
        "symbol": "TEST",
        "history": "1mo",
        "interval": "15m",
        "price_data_source": "Ticker (Alpaca)",
        "entry_rsi_setup_low": 25.0,
        "entry_reference_price": 100.0,
        "entry_stop_distance": 3.0,
        "entry_filled_at": data.index[-20].isoformat(),
        "auto_exit_enabled": True,
    }
    details = evaluate_exit_settings(
        settings,
        {"Symbol": "TEST", "Average Entry": 100.0},
        lambda *_: data,
    )

    assert details["ready"] is True
    assert details["trigger_source"] == "RSI recovery exit"
    assert details["rsi_sell_level"] == 55.0


def test_optimizer_excludes_rsi_while_buy_watchlist_preserves_rsi_inputs():
    settings = rsi_settings() | {
        "entry_window": 20,
        "exit_window": 10,
        "moving_average_window": 50,
        "pullback_average_length": 20,
        "momentum_turn_length": 10,
    }
    candidates = generate_optimizer_settings(settings, max_candidates_per_strategy=18)
    rsi_candidates = [row for _, kind, row in candidates if kind == "rsi_scalp"]
    assert not rsi_candidates

    plan = BuyWatchPlan(
        plan_id="test",
        symbol="TEST",
        interval="15m",
        history="1mo",
        price_data_source="Ticker (Alpaca)",
        strategy_label="RSI mean-reversion scalp",
        strategy_settings=settings,
    )
    rows = buy_watch_plan_detail_records(plan)
    labels = {row["Input"] for row in rows}
    assert "RSI swing lookback" in labels
    assert "Stop protection" in labels
    assert "Maximum holding period" in labels
    assert "Maximum RSI rebound allowed for buy" in labels
