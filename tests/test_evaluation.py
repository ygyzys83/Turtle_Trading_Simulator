from agentloop_trader.evaluation import (
    evaluate_walk_forward,
    synthetic_ohlc_frame,
    walk_forward_records,
)


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

    assert records[0]["Metric"] == "Verdict"
    assert any(row["Metric"] == "Profit factor" for row in records)
    assert any(row["Metric"] == "Max drawdown %" for row in records)


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
