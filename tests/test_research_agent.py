from agentloop_trader.research_agent import (
    build_research_agent_report,
    research_agent_records,
    strategy_fit_records,
)


def test_research_agent_builds_plain_english_trade_thesis():
    report = build_research_agent_report(
        ticker="AAPL",
        selected_strategy="Trendline retest continuation",
        strategy_results={
            "Breakout continuation": {
                "live": {"signal": "flat", "buy_requirements": {"breakout": False, "trend": True}},
                "stats": {"return_pct": 2.0, "total_trades": 3, "win_rate": 50, "max_drawdown_pct": 4.0, "profit_factor": 1.2},
            },
            "Trendline retest continuation": {
                "live": {"signal": "long", "buy_requirements": {"trendline": True, "retest": True, "trend": True}},
                "stats": {"return_pct": 8.0, "total_trades": 5, "win_rate": 60, "max_drawdown_pct": 6.0, "profit_factor": 1.8},
            },
        },
        setup_rows=[
            {"Read": "Overall", "Status": "A", "Plain English": "Setup and risk checks support a buy."},
            {"Read": "Trend", "Status": "Good", "Plain English": "Trend supports a buy."},
            {"Read": "Volatility", "Status": "1.2% ATR", "Plain English": "Normal movement for this ticker."},
            {"Read": "Liquidity", "Status": "Good", "Plain English": "Enough trading activity for cleaner fills."},
        ],
        final_read="TRADE",
        decision_detail="Strategy setup and risk checks passed.",
        next_action="Send the paper buy order to Alpaca.",
    )

    rows = research_agent_records(report)

    assert report.best_strategy == "Trendline retest continuation"
    assert report.event_risk == "Not connected"
    assert any(row["Area"] == "Event risk" and row["Read"] == "Not connected" for row in rows)
    assert any(row["Area"] == "Risk / reward" and "PF" in row["Read"] for row in rows)
    assert len(rows) <= 10


def test_strategy_fit_records_mark_one_best_strategy():
    report = build_research_agent_report(
        ticker="IBM",
        selected_strategy="Breakout continuation",
        strategy_results={
            "Breakout continuation": {
                "live": {"signal": "flat", "buy_requirements": {"breakout": False}},
                "stats": {"return_pct": -1.0, "total_trades": 1, "win_rate": 0, "max_drawdown_pct": 3.0, "profit_factor": 0.0},
            },
            "Trend pullback continuation": {
                "live": {"signal": "long", "buy_requirements": {"pullback": True, "trend": True}},
                "stats": {"return_pct": 3.0, "total_trades": 2, "win_rate": 50, "max_drawdown_pct": 2.0, "profit_factor": 1.1},
            },
        },
        setup_rows=[],
        final_read="WAIT",
        decision_detail="No buy yet.",
        next_action="Keep watching.",
    )

    records = strategy_fit_records(report)

    assert sum(1 for row in records if row["Best"] == "Yes") == 1
    assert next(row for row in records if row["Best"] == "Yes")["Strategy"] == "Trend pullback continuation"


def test_research_read_distinguishes_selected_strategy_from_best_current_fit():
    report = build_research_agent_report(
        ticker="IBM",
        selected_strategy="Trendline retest continuation",
        strategy_results={
            "Trendline breakout": {
                "live": {"signal": "long", "buy_requirements": {"trend": True}},
                "stats": {"return_pct": 4, "total_trades": 4, "win_rate": 50, "profit_factor": 1.5, "max_drawdown_pct": 2},
            },
            "Trendline retest continuation": {
                "live": {"signal": "flat", "buy_requirements": {"trend": True, "retest": False}},
                "stats": {"return_pct": 1, "total_trades": 2, "win_rate": 50, "profit_factor": 1.1, "max_drawdown_pct": 2},
            },
        },
        setup_rows=[],
        final_read="WAIT",
        decision_detail="The selected strategy is waiting.",
        next_action="Wait.",
    )

    records = research_agent_records(report)
    selected = next(row for row in records if row["Area"] == "Selected strategy")
    best = next(row for row in records if row["Area"] == "Best current fit across all strategies")
    assert selected["Read"] == "Trendline retest continuation"
    assert best["Read"] == "Trendline breakout"
