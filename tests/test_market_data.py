from datetime import UTC, datetime, timedelta
import json

import pandas as pd
import pytest

from agentloop_trader.market_data import NewsItem, alpaca_timeframe, build_company_research_context, completed_price_bars, fetch_alpaca_bars, period_start_time, validate_price_bars


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


def test_price_bars_are_sorted_deduplicated_and_numeric():
    index = pd.to_datetime(["2026-01-02", "2026-01-01", "2026-01-02"])
    data = pd.DataFrame(
        {"Open": [11, 10, 12], "High": [12, 11, 13], "Low": [10, 9, 11], "Close": ["11.5", "10.5", "12.5"], "Volume": [100, 90, 110]},
        index=index,
    )

    clean = validate_price_bars(data, "TEST")

    assert clean.index.is_monotonic_increasing
    assert len(clean) == 2
    assert clean.attrs["symbol"] == "TEST"
    assert clean.iloc[-1]["Close"] == 12.5


def test_price_bars_reject_impossible_ohlc_ranges():
    data = pd.DataFrame(
        {"High": [99], "Low": [101], "Close": [100]},
        index=pd.date_range("2026-01-01", periods=1),
    )

    with pytest.raises(ValueError, match="Invalid OHLC"):
        validate_price_bars(data, "BAD")


def test_completed_bars_exclude_forming_bar_but_keep_latest_price():
    index = pd.to_datetime(["2026-07-10T16:00:00Z", "2026-07-10T17:00:00Z"])
    data = pd.DataFrame(
        {"Open": [100, 101], "High": [102, 104], "Low": [99, 100], "Close": [101, 103]},
        index=index,
    )

    completed = completed_price_bars(data, "1h", now=datetime(2026, 7, 10, 17, 30, tzinfo=UTC))

    assert len(completed) == 1
    assert completed.iloc[-1]["Close"] == 101
    assert completed.attrs["latest_price"] == 103
    assert completed.attrs["latest_high"] == 104


def test_alpaca_history_fetch_continues_past_twenty_pages(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def fake_urlopen(_request, timeout):
        calls.append(timeout)
        page = len(calls)
        timestamp = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=page)
        return FakeResponse({
            "bars": {"QQQ": [{"t": timestamp.isoformat(), "o": 100, "h": 102, "l": 99, "c": 101, "v": 1000}]},
            "next_page_token": f"page-{page + 1}" if page < 21 else None,
        })

    monkeypatch.setattr("agentloop_trader.market_data.urlopen", fake_urlopen)

    result = fetch_alpaca_bars("QQQ", "5y", "4h", "key", "secret")

    assert len(calls) == 21
    assert len(result) == 21
