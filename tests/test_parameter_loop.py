from dataclasses import replace

import pandas as pd

from agentloop_trader.models import RiskLimits, StrategyConfig
from agentloop_trader.evaluation import synthetic_ohlc_frame
from agentloop_trader.parameter_loop import (
    BOUNDED_ATR_MULTIPLIERS,
    BOUNDED_ENTRY_WINDOWS,
    BOUNDED_EXIT_WINDOWS,
    BOUNDED_MA_WINDOWS,
    CandidateDiagnostics,
    buy_and_hold_benchmark,
    candidate_verdict,
    candidate_verdict_records,
    candidate_records,
    evaluate_parameter_candidates,
    generate_bounded_candidates,
    generate_optimizer_settings,
    optimize_strategy_inputs,
    optimize_strategy_intervals,
    optimizer_candidate_records,
    optimizer_interval_records,
    optimizer_regime_records,
    optimizer_recommendation_records,
    optimizer_robustness_records,
    optimizer_stress_records,
    recommendation_evidence_status,
    recommend_candidate,
    recommendation_summary,
    validate_settings_across_tickers,
)


CURRENT_SETTINGS = {
    "strategy_label": "Trendline retest continuation",
    "strategy_type": "trendline_retest",
    "entry_window": 20,
    "exit_window": 10,
    "atr_stop_multiplier": 2.0,
    "risk_per_trade_pct": 1.0,
    "moving_average_window": 50,
    "pullback_average_length": 20,
    "momentum_turn_length": 5,
}


def test_generate_bounded_candidates_stays_inside_allowed_parameter_sets():
    current = StrategyConfig(
        entry_window=20,
        exit_window=10,
        atr_stop_multiplier=2.0,
        risk_per_trade_pct=1.0,
        moving_average_window=200,
    )

    candidates = generate_bounded_candidates(current, max_candidates=8)

    assert len(candidates) == 8
    assert all(c.entry_window in BOUNDED_ENTRY_WINDOWS for c in candidates)
    assert all(c.exit_window in BOUNDED_EXIT_WINDOWS for c in candidates)
    assert all(c.atr_stop_multiplier in BOUNDED_ATR_MULTIPLIERS for c in candidates)
    assert all(c.moving_average_window in BOUNDED_MA_WINDOWS for c in candidates)


def test_parameter_loop_preserves_risk_setting_and_ranks_candidates():
    current = StrategyConfig(
        entry_window=20,
        exit_window=10,
        atr_stop_multiplier=2.0,
        risk_per_trade_pct=1.0,
        moving_average_window=200,
    )

    candidates = evaluate_parameter_candidates(
        current=current,
        account=50_000,
        risk_pct_dec=0.01,
        seed=42,
        market_data=None,
        train_fraction=0.65,
        max_candidates=4,
    )

    assert len(candidates) == 4
    assert candidates == sorted(candidates, key=lambda c: c.score, reverse=True)
    assert all(c.config.risk_per_trade_pct == current.risk_per_trade_pct for c in candidates)


def test_parameter_loop_recommendation_and_records_are_display_ready():
    current = StrategyConfig(
        entry_window=20,
        exit_window=10,
        atr_stop_multiplier=2.0,
        risk_per_trade_pct=1.0,
        moving_average_window=200,
    )
    candidates = evaluate_parameter_candidates(
        current=current,
        account=50_000,
        risk_pct_dec=0.01,
        seed=42,
        market_data=None,
        train_fraction=0.65,
        max_candidates=4,
    )

    recommended = recommend_candidate(candidates)
    records = candidate_records(candidates)
    summary = recommendation_summary(recommended)

    assert recommended is not None
    assert records
    assert "Try these settings next" in summary


def test_optimizer_searches_all_strategies_and_returns_display_records():
    current_settings = dict(CURRENT_SETTINGS)

    settings = generate_optimizer_settings(current_settings, max_candidates_per_strategy=2)
    strategy_names = {row[0] for row in settings}

    assert strategy_names == {
        "Breakout continuation",
        "Trend pullback continuation",
        "Trendline breakout",
        "Trendline retest continuation",
    }

    result = optimize_strategy_inputs(
        market_data=None,
        current_settings=current_settings,
        account_equity=50_000,
        train_fraction=0.65,
        max_candidates_per_strategy=2,
    )

    assert result.tested_candidates > 0
    assert result.best is not None
    assert result.best.tested_periods >= 2
    assert 0 <= result.best.profitable_test_periods <= result.best.tested_periods
    assert result.candidates == sorted(result.candidates, key=lambda row: row.score, reverse=True)
    assert optimizer_recommendation_records(result)
    assert optimizer_candidate_records(result.candidates)
    assert result.robustness is not None
    assert result.robustness.locked_test is not None
    assert result.best.rolling_windows > 0
    assert result.best.plateau_neighbors > 0
    assert optimizer_robustness_records(result)
    assert optimizer_stress_records(result)
    assert optimizer_regime_records(result)


def test_optimizer_candidate_subset_varies_more_than_one_input_dimension():
    current_settings = {
        "entry_window": 20, "exit_window": 10, "atr_stop_multiplier": 2.0,
        "risk_per_trade_pct": 1.0, "moving_average_window": 50,
        "pullback_average_length": 20, "momentum_turn_length": 5,
    }

    rows = generate_optimizer_settings(current_settings, max_candidates_per_strategy=18)
    breakout = [settings for _, strategy, settings in rows if strategy == "breakout"]

    assert len({row["entry_window"] for row in breakout}) > 1
    assert len({row["atr_stop_multiplier"] for row in breakout}) > 1
    assert len({row["moving_average_window"] for row in breakout}) > 1


def test_optimizer_tests_each_setting_with_rsi_entry_rule_off_and_on():
    rows = generate_optimizer_settings(CURRENT_SETTINGS, max_candidates_per_strategy=2)

    for strategy_type in {row[1] for row in rows}:
        strategy_rows = [settings for _, row_type, settings in rows if row_type == strategy_type]
        assert {settings["rsi_entry_filter_enabled"] for settings in strategy_rows} == {False, True}
        assert len(strategy_rows) == 4


def test_optimizer_cost_stress_never_improves_return_and_is_deterministic():
    result = optimize_strategy_inputs(
        market_data=synthetic_ohlc_frame(seed=19),
        current_settings=CURRENT_SETTINGS,
        account_equity=50_000,
        max_candidates_per_strategy=2,
        bootstrap_samples=200,
    )

    stress_returns = [row["Return Percent"] for row in optimizer_stress_records(result)]
    assert stress_returns == sorted(stress_returns, reverse=True)
    assert result.robustness.bootstrap == optimize_strategy_inputs(
        market_data=synthetic_ohlc_frame(seed=19),
        current_settings=CURRENT_SETTINGS,
        account_equity=50_000,
        max_candidates_per_strategy=2,
        bootstrap_samples=200,
    ).robustness.bootstrap


def test_cross_ticker_check_uses_the_selected_settings_without_reoptimizing():
    result = optimize_strategy_inputs(
        market_data=synthetic_ohlc_frame(seed=42),
        current_settings=CURRENT_SETTINGS,
        account_equity=50_000,
        max_candidates_per_strategy=2,
        bootstrap_samples=100,
    )
    selected_settings = dict(result.best.settings)
    other_tickers = {
        "AAA": synthetic_ohlc_frame(seed=7),
        "BBB": synthetic_ohlc_frame(seed=8),
        "CCC": synthetic_ohlc_frame(seed=9),
    }

    cross_result = validate_settings_across_tickers(selected_settings, other_tickers, 50_000)

    assert cross_result.tested_tickers == 3
    assert len(cross_result.rows) == 3
    assert selected_settings == result.best.settings
    assert all(row["Ticker"] in other_tickers for row in cross_result.rows)


def test_buy_and_hold_benchmark_uses_the_exact_requested_price_range():
    market_data = pd.DataFrame({"Close": [100.0, 110.0, 90.0, 120.0]})

    result = buy_and_hold_benchmark(market_data, 10_000)
    subset = buy_and_hold_benchmark(market_data, 10_000, start=1)

    assert result.return_percent == 20.0
    assert result.max_drawdown_percent == 20.0
    assert result.final_equity == 11_999.71
    assert subset.return_percent == 9.09
    assert subset.bars == 3


def test_optimizer_reports_newer_and_locked_results_against_buy_and_hold():
    result = optimize_strategy_inputs(
        market_data=synthetic_ohlc_frame(seed=31),
        current_settings=CURRENT_SETTINGS,
        account_equity=50_000,
        max_candidates_per_strategy=2,
        bootstrap_samples=100,
    )

    assert result.best is not None
    assert result.best.excess_return_percent == round(
        result.best.test_return_percent - result.best.benchmark_return_percent,
        2,
    )
    locked = result.robustness.locked_test
    assert locked.excess_return_percent == round(
        locked.return_percent - locked.benchmark_return_percent,
        2,
    )
    record_names = {row["Item"] for row in optimizer_recommendation_records(result)}
    assert {"Recommendation status", "Validation and final-period comparison", "RSI entry rule", "Next step"} <= record_names
    assert recommendation_evidence_status(result) in {"Research only", "Ready for paper test"}


def test_optimizer_ranks_allocated_return_but_preserves_account_impact():
    result = optimize_strategy_inputs(
        market_data=synthetic_ohlc_frame(seed=12),
        current_settings=CURRENT_SETTINGS,
        account_equity=100_000,
        risk_limits=RiskLimits(max_symbol_concentration_pct=5.0),
        max_candidates_per_strategy=1,
        bootstrap_samples=50,
    )

    assert result.best.allocated_capital == 5_000
    assert abs(result.best.test_account_return_percent - result.best.test_return_percent * 0.05) <= 0.02
    assert abs(result.best.benchmark_account_return_percent - result.best.benchmark_return_percent * 0.05) <= 0.02


def test_interval_optimizer_returns_one_ranked_result_per_interval():
    result = optimize_strategy_intervals(
        market_data_by_interval={
            "1d": ("5y", synthetic_ohlc_frame(n=500, seed=4)),
            "4h": ("2y", synthetic_ohlc_frame(n=500, seed=5)),
        },
        current_settings=CURRENT_SETTINGS,
        account_equity=50_000,
        max_candidates_per_strategy=1,
        bootstrap_samples=50,
    )

    rows = optimizer_interval_records(result)
    assert result.best_interval in {"1d", "4h"}
    assert len(rows) == 2
    assert {row["Interval"] for row in rows} == {"1d", "4h"}
    assert all(
        "Comparable Buy and Hold %" in row
        and "Comparable Excess %" in row
        and "Long-History Excess %" in row
        for row in rows
    )


def test_interval_optimizer_uses_one_shared_calendar_period_and_long_history_check():
    daily = synthetic_ohlc_frame(n=1_500, seed=14)
    hourly = synthetic_ohlc_frame(n=3_000, seed=15)
    daily.index = pd.date_range("2020-01-01", periods=1_500, freq="D")
    hourly.index = pd.date_range("2020-01-01", periods=3_000, freq="8h")

    result = optimize_strategy_intervals(
        market_data_by_interval={"1d": ("10y", daily), "4h": ("5y", hourly)},
        current_settings=CURRENT_SETTINGS,
        account_equity=50_000,
        max_candidates_per_strategy=1,
        bootstrap_samples=20,
        comparison_years=2,
    )

    assert len({row.comparison_history for row in result.interval_results}) == 1
    assert all(row.durability_trades >= 0 for row in result.interval_results)
    assert all(
        row.comparison_excess_return_percent
        == round(row.comparison_return_percent - row.comparison_benchmark_return_percent, 2)
        for row in result.interval_results
    )
    assert "complete latest" in result.summary.lower()
    assert "middle validation period" in result.summary.lower()
    assert "untouched final period" in result.summary.lower()


def _verdict_result(**overrides):
    result = optimize_strategy_inputs(
        market_data=synthetic_ohlc_frame(n=700, seed=23),
        current_settings=CURRENT_SETTINGS,
        account_equity=100_000,
        risk_limits=RiskLimits(max_symbol_concentration_pct=5.0),
        max_candidates_per_strategy=1,
        bootstrap_samples=20,
    )
    candidate = replace(
        result.best,
        excess_return_percent=overrides.get("validation_excess", 5.0),
        plateau_profitable_percent=overrides.get("nearby_percent", 80.0),
        plateau_neighbors=5,
        test_trades=overrides.get("validation_trades", 30),
        rolling_profitable_windows=overrides.get("rolling_profitable", 3),
        rolling_windows=4,
    )
    locked = replace(
        result.robustness.locked_test,
        excess_return_percent=overrides.get("locked_excess", 3.0),
        return_percent=5.0,
        trades=overrides.get("locked_trades", 20),
        passed=True,
    )
    bootstrap = replace(
        result.robustness.bootstrap,
        loss_probability_percent=overrides.get("loss_probability", 20.0),
    )
    diagnostics = CandidateDiagnostics(
        after_cost_expectancy_percent=overrides.get("expectancy", 0.2),
        best_trade_removed_return_percent=overrides.get("without_best", 5.0),
        best_trade_share_percent=overrides.get("best_share", 30.0),
        account_drawdown_percent=overrides.get("account_drawdown", 0.5),
        slippage_survival_bps=overrides.get("slippage_bps", 20),
        profitable_regime_percent=80.0,
        return_to_drawdown=2.0,
    )
    evidence = replace(result.robustness, locked_test=locked, bootstrap=bootstrap, diagnostics=diagnostics)
    return replace(result, best=candidate, robustness=evidence)


def test_candidate_verdict_marks_broad_evidence_as_strong_and_is_deterministic():
    result = _verdict_result()

    first = candidate_verdict(result, "4h")
    second = candidate_verdict(result, "4h")

    assert first == second
    assert first.tier == "Strong Candidate"
    assert candidate_verdict_records(first)


def test_candidate_verdict_allows_modest_locked_underperformance_as_promising():
    result = _verdict_result(
        validation_excess=2.0,
        locked_excess=-2.0,
        loss_probability=35.0,
        nearby_percent=60.0,
        slippage_bps=10,
        without_best=-0.5,
        best_share=50.0,
        account_drawdown=1.2,
        validation_trades=8,
        locked_trades=5,
        rolling_profitable=2,
    )

    assert candidate_verdict(result, "4h").tier == "Promising Candidate"


def test_candidate_verdict_rejects_negative_after_cost_expectancy():
    result = _verdict_result(expectancy=-0.01)

    verdict = candidate_verdict(result, "4h")

    assert verdict.tier == "Reject"
    assert any("After-cost expectancy" in item for item in verdict.failed)


def test_candidate_verdict_keeps_thin_trade_evidence_in_research_only():
    result = _verdict_result(validation_trades=2, locked_trades=2)

    assert candidate_verdict(result, "1h").tier == "Research Only"
    assert candidate_verdict(result, "1d").tier == "Research Only"
