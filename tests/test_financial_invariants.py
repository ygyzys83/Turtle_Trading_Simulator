import math

from agentloop_trader.backtest import (
    _profit_protection_stop,
    _protective_stop_fill,
    simulate_trend_pullback_strategy,
    simulate_trendline_breakout_strategy,
    simulate_trendline_retest_strategy,
    simulate_turtle_strategy,
)
from agentloop_trader.models import RiskLimits, TradeIntent
from agentloop_trader.risk import check_trade_intent, constrain_trade_intent_to_limits


def test_all_strategy_trade_rows_reconcile_and_obey_risk_caps():
    account = 100_000
    limits = RiskLimits(
        max_risk_per_trade_pct=1,
        max_position_notional_pct=5,
        max_portfolio_exposure_pct=80,
        max_symbol_concentration_pct=10,
        max_session_loss_pct=20,
    )
    results = [
        simulate_turtle_strategy(account, 20, 10, 2.0, 0.01, 50, seed=42, risk_limits=limits),
        simulate_trend_pullback_strategy(account, 20, 10, 2.0, 0.01, 50, 5, seed=42, risk_limits=limits),
        simulate_trendline_breakout_strategy(account, 20, 10, 2.0, 0.01, 50, seed=42, risk_limits=limits),
        simulate_trendline_retest_strategy(account, 20, 10, 2.0, 0.01, 50, 5, seed=42, risk_limits=limits),
    ]

    for result in results:
        trade_log = result[3]
        stats = result[5]
        assert all(math.isfinite(float(stats[key])) for key in ("final_equity", "total_pnl", "return_pct", "max_drawdown_pct"))
        for trade in trade_log:
            assert trade["notional"] <= account * 0.05 + 0.01
            assert trade["risk_dollars"] <= account * 0.01 + 0.01
            assert trade["pnl"] == round((trade["exit"] - trade["entry"]) * trade["shares"], 2)
            assert trade["max_adverse_pnl"] <= 0


def test_constrained_intents_never_exceed_dollar_limits():
    limits = RiskLimits(
        max_risk_per_trade_pct=1,
        max_position_notional_pct=5,
        max_portfolio_exposure_pct=80,
        max_symbol_concentration_pct=10,
    )
    for entry_price in (10, 50, 100, 500):
        for stop_distance in (0.5, 2, 10):
            intent = TradeIntent(
                symbol="AAPL", side="buy", quantity=1_000_000,
                entry_price=entry_price, stop_loss=entry_price - stop_distance,
            )
            adjusted = constrain_trade_intent_to_limits(intent, 100_000, limits, available_cash=100_000)
            checked = check_trade_intent(adjusted, 100_000, limits, available_cash=100_000)

            assert checked.approved
            assert checked.risk_dollars <= 1_000
            assert checked.notional_dollars <= 5_000


def test_profit_protection_never_moves_a_long_stop_down():
    source, stop = _profit_protection_stop(
        entry_price=100,
        initial_stop_price=96,
        current_stop_price=102,
        current_price=112,
        current_atr=2,
        high_since_entry=115,
        strategy_exit_price=101,
        breakeven_after_r=1,
        trail_after_r=2,
        trailing_atr_multiplier=3,
    )

    assert source == "ATR trail"
    assert stop == 109
    assert stop >= 102


def test_gap_below_stop_fills_at_open_and_normal_breach_fills_at_stop():
    assert _protective_stop_fill(open_price=94, low_price=93, stop_price=95) == 94
    assert _protective_stop_fill(open_price=97, low_price=94, stop_price=95) == 95
    assert _protective_stop_fill(open_price=97, low_price=96, stop_price=95) is None
