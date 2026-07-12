import pandas as pd

from agentloop_trader.models import RiskLimits
from agentloop_trader.parameter_loop import buy_and_hold_benchmark
from agentloop_trader.performance import (
    allocation_metrics,
    annualized_return_percent,
    elapsed_years,
    ticker_allocated_capital,
)


def test_allocation_metrics_separate_account_and_ticker_sleeve_returns():
    limits = RiskLimits(max_symbol_concentration_pct=5.0)

    result = allocation_metrics(
        account_equity=100_000,
        total_pnl=500,
        max_drawdown_dollars=1_000,
        risk_limits=limits,
        years=2.0,
    )

    assert result["allocated_capital"] == 5_000
    assert result["account_return_pct"] == 0.5
    assert result["allocated_return_pct"] == 10.0
    assert result["allocated_max_drawdown_pct"] == 20.0
    assert result["annualized_allocated_return_pct"] == 4.88


def test_equal_capital_buy_and_hold_keeps_unallocated_account_cash_idle():
    data = pd.DataFrame(
        {"Close": [100.0, 90.0, 120.0]},
        index=pd.to_datetime(["2023-01-01", "2024-01-01", "2025-01-02"]),
    )

    result = buy_and_hold_benchmark(data, 100_000, allocated_capital=5_000)

    assert result.return_percent == 20.0
    assert result.account_return_percent == 1.0
    assert result.final_equity == 101_000
    assert result.max_drawdown_percent == 10.0
    assert result.annualized_return_percent is not None


def test_buy_and_hold_worst_drop_uses_the_original_ticker_allocation():
    data = pd.DataFrame({"Close": [100.0, 160.0, 110.0, 170.0]})

    result = buy_and_hold_benchmark(data, 100_000, allocated_capital=5_000)

    # The sleeve falls $2,500 from its $8,000 peak; the common denominator is the $5,000 allocation.
    assert result.max_drawdown_percent == 50.0


def test_annualized_return_is_hidden_for_one_year_or_less_and_for_exhausted_capital():
    assert annualized_return_percent(10.0, 1.0) is None
    assert annualized_return_percent(-100.0, 2.0) is None
    assert annualized_return_percent(21.0, 2.0) == 10.0


def test_elapsed_years_uses_actual_timestamps_and_allocation_uses_symbol_limit():
    index = pd.to_datetime(["2020-01-01", "2022-01-01"])
    years = elapsed_years(index)

    assert 1.99 < years < 2.01
    assert ticker_allocated_capital(100_000, RiskLimits(max_symbol_concentration_pct=5)) == 5_000
