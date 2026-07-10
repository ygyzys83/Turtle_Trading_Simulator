from datetime import UTC, datetime

from agentloop_trader.market_data import NewsItem, alpaca_timeframe, build_company_research_context, period_start_time


def test_alpaca_timeframe_maps_intraday_intervals():
    assert alpaca_timeframe("1m") == "1Min"
    assert alpaca_timeframe("15m") == "15Min"
    assert alpaca_timeframe("1h") == "1Hour"
    assert alpaca_timeframe("unknown") == "1Hour"


def test_period_start_time_uses_expected_lookback():
    now = datetime(2026, 7, 10, tzinfo=UTC)

    assert period_start_time("1y", now).year == 2025
    assert period_start_time("ytd", now) == datetime(2026, 1, 1, tzinfo=UTC)


def test_company_context_is_honest_when_news_is_missing():
    context = build_company_research_context("AAPL", news=[])

    assert context.event_risk == "Not connected"
    assert context.fundamentals_status == "Not connected"
    assert "Upcoming earnings are not connected" in context.event_detail


def test_company_context_flags_recent_news_keywords_without_blocking():
    context = build_company_research_context(
        "AAPL",
        news=[NewsItem("AAPL earnings preview", "Revenue guidance due soon", "2026-07-10", "test", "")],
    )

    assert context.event_risk == "Review recent news"
    assert context.news_status == "1 recent headline(s)"
