import pytest

from agentloop_trader.evaluation import (
    evaluate_period_performance,
    evaluate_walk_forward,
    period_performance_records,
    synthetic_ohlc_frame,
    walk_forward_records,
)
from agentloop_trader.models import RiskLimits


def test_walk_forward_evaluation_returns_train_and_oos_stats():
    result = evaluate_walk_forward(
        account=50_000,
        entry_w=20,
        exit_w=10,
        atr_mult=2.0,
        risk_pct_dec=0.01,
        ma_w=200,
        seed=42,
        market_data=None,
        train_fraction=0.65,
    )

    assert result.train_bars > 0
    assert result.oos_bars > 0
    assert result.verdict in {"Pass", "Inconclusive", "Needs review"}
    assert "total_trades" in result.train_stats
    assert "total_trades" in result.oos_stats


def test_walk_forward_records_include_verdict_and_core_metrics():
    result = evaluate_walk_forward(
        account=50_000,
        entry_w=20,
        exit_w=10,
        atr_mult=2.0,
        risk_pct_dec=0.01,
        ma_w=200,
        seed=42,
        market_data=None,
        train_fraction=0.65,
    )

    records = walk_forward_records(result)

    assert records[0]["Metric"] == "Result"
    assert any(row["Metric"] == "Profit factor" for row in records)
    assert any(row["Metric"] == "Allocated return %" for row in records)
    assert any(row["Metric"] == "Allocated worst drop %" for row in records)


def test_walk_forward_rejects_insufficient_history():
    short_data = synthetic_ohlc_frame(n=120, seed=42)

    try:
        evaluate_walk_forward(
            account=50_000,
            entry_w=20,
            exit_w=10,
            atr_mult=2.0,
            risk_pct_dec=0.01,
            ma_w=200,
            seed=42,
            market_data=short_data,
            train_fraction=0.65,
        )
    except ValueError:
        return

    raise AssertionError("Expected ValueError for insufficient walk-forward history.")


def test_walk_forward_runs_the_selected_strategy_type():
    for strategy_type in ("breakout", "pullback", "trendline", "trendline_retest"):
        result = evaluate_walk_forward(
            account=50_000,
            entry_w=20,
            exit_w=10,
            atr_mult=2.0,
            risk_pct_dec=0.01,
            ma_w=50,
            seed=42,
            train_fraction=0.65,
            strategy_type=strategy_type,
            pullback_w=20,
            momentum_w=5,
        )

        assert result.train_stats["final_equity"] > 0
        assert result.oos_stats["final_equity"] > 0


def test_period_performance_uses_fixed_55_25_20_sections_and_buy_hold_comparison():
    result = evaluate_period_performance(
        account=50_000,
        entry_w=20,
        exit_w=10,
        atr_mult=2.0,
        risk_pct_dec=0.01,
        ma_w=50,
        seed=42,
        risk_limits=RiskLimits(max_symbol_concentration_pct=5.0),
        strategy_type="pullback",
        pullback_w=20,
        momentum_w=5,
    )

    assert [period.label for period in result.periods] == ["Older 55%", "Newer 25%", "Latest 20%"]
    assert sum(period.bars for period in result.periods) == 700
    assert all("allocated_return_pct" in period.stats for period in result.periods)
    assert all(
        period.excess_return_percent
        == round(period.stats["allocated_return_pct"] - period.buy_and_hold_return_percent, 2)
        for period in result.periods
    )

    records = period_performance_records(result)
    assert records[0]["Price section"] == "Older 55%"
    assert "Difference" in records[0]


def test_period_performance_labels_follow_valid_custom_fractions():
    result = evaluate_period_performance(
        account=50_000,
        entry_w=20,
        exit_w=10,
        atr_mult=2.0,
        risk_pct_dec=0.01,
        ma_w=50,
        seed=42,
        strategy_type="pullback",
        older_fraction=0.50,
        latest_fraction=0.20,
    )

    assert [period.label for period in result.periods] == ["Older 50%", "Newer 30%", "Latest 20%"]
    assert all(len(period.start_date) <= 16 for period in result.periods)


def test_period_performance_rejects_invalid_fraction_split():
    with pytest.raises(ValueError, match="leave room"):
        evaluate_period_performance(
            account=50_000,
            entry_w=20,
            exit_w=10,
            atr_mult=2.0,
            risk_pct_dec=0.01,
            ma_w=50,
            seed=42,
            older_fraction=0.80,
            latest_fraction=0.20,
        )
