import numpy as np
import pandas as pd

from agentloop_trader.price_regime import (
    classify_price_regime,
    price_regime_sections,
    strategy_regime_rows,
    summarize_regime_dependency,
)


def _frame(prices, freq="1D"):
    close = pd.Series(prices, index=pd.date_range("2020-01-01", periods=len(prices), freq=freq))
    return pd.DataFrame({
        "Open": close,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": 1_000_000,
    })


def test_price_regime_distinguishes_persistent_uptrend_from_sideways_prices():
    uptrend = classify_price_regime(_frame(np.geomspace(100, 260, 500)))
    sideways = classify_price_regime(_frame(100 + np.sin(np.linspace(0, 30, 500)) * 2))

    assert uptrend.direction in {"Strong uptrend", "Mild uptrend"}
    assert uptrend.path == "Persistent"
    assert sideways.direction == "Sideways"
    assert sideways.path in {"Mixed", "Choppy"}


def test_price_regime_sections_use_the_agreed_55_25_20_split():
    data = _frame(np.geomspace(100, 180, 100))
    sections = price_regime_sections(data)

    assert [row[2].period for row in sections] == ["Older 55%", "Newer 25%", "Latest 20%"]
    assert [row[1] - row[0] for row in sections] == [55, 25, 20]


def test_price_regime_sections_label_configured_split_instead_of_hard_coding_it():
    data = _frame(np.geomspace(100, 180, 100))
    sections = price_regime_sections(data, older_fraction=0.50, latest_fraction=0.20)

    assert [row[2].period for row in sections] == ["Older 50%", "Newer 30%", "Latest 20%"]
    assert [row[1] - row[0] for row in sections] == [50, 30, 20]


def test_price_regime_does_not_annualize_a_window_shorter_than_one_year():
    regime = classify_price_regime(_frame(np.geomspace(100, 120, 100), freq="1h"))

    assert regime.annualized_return_percent is None


def test_price_regime_ignores_non_positive_prices_before_log_math():
    regime = classify_price_regime(_frame([100, 0, -5, 105, 110]))

    assert regime.bars == 3
    assert regime.return_percent == 10.0


def test_strategy_regime_dependency_reports_current_match_without_changing_trades():
    data = _frame(np.geomspace(100, 180, 120))
    trades = [
        {"entry_bar": 10, "pnl": 50},
        {"entry_bar": 35, "pnl": 40},
        {"entry_bar": 75, "pnl": -10},
        {"entry_bar": 105, "pnl": 30},
    ]
    rows = strategy_regime_rows(data, trades, allocated_capital=5_000, blocks=6)
    dependency = summarize_regime_dependency(data, rows)

    assert sum(row["Completed trades"] for row in rows) == len(trades)
    assert dependency.tested_periods == 4
    assert dependency.current_regime
    assert dependency.current_match in {"Strong match", "Direction matches", "Different from the strongest history"}


def test_strategy_period_evaluator_replaces_boundary_trade_attribution():
    data = _frame(np.geomspace(100, 180, 120))
    calls = []

    def evaluate(start, end):
        calls.append((start, end))
        return {"total_trades": 2, "allocated_return_pct": 3.5, "win_rate": 50.0}

    rows = strategy_regime_rows(
        data,
        [{"entry_bar": 1, "pnl": 100_000}],
        allocated_capital=5_000,
        blocks=6,
        period_evaluator=evaluate,
    )

    assert len(calls) == 6
    assert all(row["Completed trades"] == 2 for row in rows)
    assert all(row["_strategy_return"] == 3.5 for row in rows)


def test_dependency_chooses_best_advantage_over_buy_and_hold_not_largest_raw_return():
    data = _frame(np.geomspace(100, 180, 120))
    rows = [
        {
            "Completed trades": 5,
            "_strategy_return": 50.0,
            "_buy_hold_return": 80.0,
            "_excess_return": -30.0,
            "_direction": "Strong uptrend",
            "_label": "Strong uptrend, persistent",
        },
        {
            "Completed trades": 5,
            "_strategy_return": 10.0,
            "_buy_hold_return": -5.0,
            "_excess_return": 15.0,
            "_direction": "Sideways",
            "_label": "Sideways, choppy",
        },
    ]

    dependency = summarize_regime_dependency(data, rows)

    assert dependency.strongest_regime == "Sideways, choppy"
    assert dependency.outperforming_periods == 1


def test_dependency_treats_strong_and_mild_uptrends_as_same_direction_family():
    data = _frame(np.geomspace(100, 180, 120))
    current_direction = classify_price_regime(data.iloc[-24:]).direction
    assert "uptrend" in current_direction.lower()
    rows = [{
        "Completed trades": 5,
        "_strategy_return": 10.0,
        "_buy_hold_return": 0.0,
        "_excess_return": 10.0,
        "_direction": "Mild uptrend" if current_direction == "Strong uptrend" else "Strong uptrend",
        "_label": "Different-strength uptrend, mixed",
    }]

    dependency = summarize_regime_dependency(data, rows)

    assert dependency.current_match == "Direction matches"
