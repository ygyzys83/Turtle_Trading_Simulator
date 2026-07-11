from agentloop_trader.models import StrategyConfig
from agentloop_trader.evaluation import synthetic_ohlc_frame
from agentloop_trader.parameter_loop import (
    BOUNDED_ATR_MULTIPLIERS,
    BOUNDED_ENTRY_WINDOWS,
    BOUNDED_EXIT_WINDOWS,
    BOUNDED_MA_WINDOWS,
    candidate_records,
    evaluate_parameter_candidates,
    generate_bounded_candidates,
    generate_optimizer_settings,
    optimize_strategy_inputs,
    optimizer_candidate_records,
    optimizer_regime_records,
    optimizer_recommendation_records,
    optimizer_robustness_records,
    optimizer_stress_records,
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
