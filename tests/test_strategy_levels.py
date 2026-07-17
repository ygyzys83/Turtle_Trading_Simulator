from agentloop_trader.strategy_levels import build_buy_level_snapshot


def test_trendline_snapshot_uses_buffered_breakout_not_trend_filter_as_buy_level():
    snapshot = build_buy_level_snapshot(
        {
            "setup_type": "trendline",
            "last_p": 99.0,
            "last_sma": 105.0,
            "trend_filter_level": 105.0,
            "trendline_level": 100.0,
            "trendline_breakout_level": 100.2,
            "trendline_quality": "2 anchors + 1 additional confirmed touch",
            "trend_ok": False,
            "buy_requirements": {
                "Touch-scored descending trendline found in last 20 bars": True,
                "Completed bar crossed above buffered trendline": False,
                "20-bar trendline slopes down": True,
                "Retest held trendline": True,
                "Position size above zero": True,
            },
        },
        interval="4h",
    )

    assert snapshot["next_buy_level"] == 100.2
    breakout = next(row for row in snapshot["records"] if row["Required BUY Rule"] == "Completed-close breakout")
    assert breakout["Required"] == "Completed 4h bar closes above $100.20"
    assert "0.10 ATR buffer" in breakout["Plain English"]


def test_missing_trendline_does_not_report_the_trend_filter_as_the_next_buy_level():
    snapshot = build_buy_level_snapshot(
        {
            "setup_type": "trendline",
            "last_p": 99.0,
            "trend_filter_level": 105.0,
            "trend_ok": False,
            "buy_requirements": {
                "Touch-scored descending trendline found in last 20 bars": False,
                "Completed bar crossed above buffered trendline": False,
                "20-bar trendline slopes down": False,
                "Retest held trendline": True,
                "Position size above zero": True,
            },
        },
        interval="4h",
    )

    assert snapshot["next_buy_level"] is None
