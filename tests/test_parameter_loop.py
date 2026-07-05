from agentloop_trader.models import StrategyConfig
from agentloop_trader.parameter_loop import (
    BOUNDED_ATR_MULTIPLIERS,
    BOUNDED_ENTRY_WINDOWS,
    BOUNDED_EXIT_WINDOWS,
    BOUNDED_MA_WINDOWS,
    candidate_records,
    evaluate_parameter_candidates,
    generate_bounded_candidates,
    recommend_candidate,
    recommendation_summary,
)


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
    assert "Recommended bounded parameters" in summary

