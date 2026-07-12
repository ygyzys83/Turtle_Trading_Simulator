from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd


ALPACA_DATA_BASE_URL = "https://data.alpaca.markets"
NEW_YORK_TIME = ZoneInfo("America/New_York")
INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}


def validate_price_bars(data: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Return sorted, numeric OHLCV bars or raise on unsafe market data."""
    if data is None or data.empty:
        raise ValueError(f"No price data returned for {symbol or 'the selected ticker'}.")
    required = ["Close", "High", "Low"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    clean = data.copy()
    clean = clean[~clean.index.duplicated(keep="last")].sort_index()
    numeric_columns = [column for column in ("Open", "High", "Low", "Close", "Volume") if column in clean.columns]
    for column in numeric_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=required)
    if clean.empty:
        raise ValueError(f"No usable price rows returned for {symbol or 'the selected ticker'}.")
    invalid_price = (clean[required] <= 0).any(axis=1)
    invalid_range = (clean["High"] < clean["Low"]) | (clean["High"] < clean["Close"]) | (clean["Low"] > clean["Close"])
    if "Open" in clean.columns:
        invalid_range |= (clean["High"] < clean["Open"]) | (clean["Low"] > clean["Open"])
    if bool((invalid_price | invalid_range).any()):
        raise ValueError(f"Invalid OHLC values returned for {symbol or 'the selected ticker'}.")
    if "Volume" in clean.columns:
        clean["Volume"] = clean["Volume"].fillna(0)
        if bool((clean["Volume"] < 0).any()):
            raise ValueError(f"Negative volume returned for {symbol or 'the selected ticker'}.")
    clean.attrs.update(getattr(data, "attrs", {}))
    if symbol:
        clean.attrs["symbol"] = str(symbol).strip().upper()
    return clean


def completed_price_bars(
    data: pd.DataFrame,
    interval: str,
    now: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Keep completed bars while preserving the newest available quote-like values."""
    clean = validate_price_bars(data, str(getattr(data, "attrs", {}).get("symbol", "")))
    latest = clean.iloc[-1]
    attrs = dict(clean.attrs)
    attrs.update({
        "latest_price": float(latest["Close"]),
        "latest_high": float(latest["High"]),
        "latest_low": float(latest["Low"]),
        "latest_bar_time": clean.index[-1].isoformat() if hasattr(clean.index[-1], "isoformat") else str(clean.index[-1]),
    })
    current = pd.Timestamp(now or datetime.now(UTC))
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    current_et = current.tz_convert(NEW_YORK_TIME)

    completed_mask: list[bool] = []
    for raw_timestamp in clean.index:
        timestamp = pd.Timestamp(raw_timestamp)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(NEW_YORK_TIME)
        timestamp_et = timestamp.tz_convert(NEW_YORK_TIME)
        if interval == "1d":
            session_close = timestamp_et.normalize() + pd.Timedelta(hours=16)
            is_complete = timestamp_et.date() < current_et.date() or current_et >= session_close
        else:
            minutes = INTERVAL_MINUTES.get(interval, 60)
            normal_end = timestamp_et + pd.Timedelta(minutes=minutes)
            session_close = timestamp_et.normalize() + pd.Timedelta(hours=16)
            bar_end = min(normal_end, session_close) if timestamp_et.hour >= 9 else normal_end
            is_complete = current_et >= bar_end
        completed_mask.append(bool(is_complete))

    completed = clean.loc[completed_mask].copy()
    if completed.empty:
        raise ValueError("No completed price bars are available for the selected interval.")
    completed.attrs.update(attrs)
    return completed


def alpaca_timeframe(interval: str) -> str:
    return {
        "1m": "1Min",
        "5m": "5Min",
        "15m": "15Min",
        "30m": "30Min",
        "1h": "1Hour",
        "4h": "4Hour",
        "1d": "1Day",
    }.get(interval, "1Hour")


def period_start_time(period: str, now: datetime | None = None) -> datetime:
    current = now or datetime.now(UTC)
    offsets = {
        "1d": 2,
        "5d": 7,
        "1mo": 31,
        "3mo": 93,
        "6mo": 186,
        "1y": 366,
        "2y": 732,
        "5y": 365 * 5 + 2,
        "10y": 365 * 10 + 3,
        "max": 365 * 10 + 3,
    }
    if period == "ytd":
        return datetime(current.year, 1, 1, tzinfo=UTC)
    return current - timedelta(days=offsets.get(period, 366))


def fetch_alpaca_bars(
    symbol: str,
    period: str,
    interval: str,
    api_key: str | None,
    api_secret: str | None,
    *,
    feed: str = "iex",
    timeout: int = 20,
) -> pd.DataFrame:
    clean = str(symbol).strip().upper()
    if not clean:
        raise ValueError("Enter a ticker symbol.")
    if not api_key or not api_secret:
        raise RuntimeError("Alpaca API key and secret are required for Alpaca price data.")

    all_bars: list[dict[str, Any]] = []
    page_token = ""
    start_time = period_start_time(period).isoformat().replace("+00:00", "Z")
    max_pages = 200
    for _ in range(max_pages):
        params: dict[str, Any] = {
            "symbols": clean,
            "timeframe": alpaca_timeframe(interval),
            "start": start_time,
            "adjustment": "all",
            "feed": feed,
            "limit": 10000,
        }
        if page_token:
            params["page_token"] = page_token
        request = Request(
            f"{ALPACA_DATA_BASE_URL}/v2/stocks/bars?{urlencode(params)}",
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
                "Accept": "application/json",
            },
        )
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        all_bars.extend(payload.get("bars", {}).get(clean, []))
        page_token = str(payload.get("next_page_token") or "")
        if not page_token:
            break
    if page_token:
        raise RuntimeError(
            f"Alpaca price history for {clean} exceeded {max_pages} pages; no partial dataset was used."
        )

    if not all_bars:
        raise ValueError(f"No Alpaca price data returned for {clean}.")
    data = pd.DataFrame(
        {
            "Date": [bar.get("t") for bar in all_bars],
            "Open": [bar.get("o") for bar in all_bars],
            "Close": [bar.get("c") for bar in all_bars],
            "High": [bar.get("h") for bar in all_bars],
            "Low": [bar.get("l") for bar in all_bars],
            "Volume": [bar.get("v") for bar in all_bars],
        }
    )
    data["Date"] = pd.to_datetime(data["Date"], utc=True)
    data = data.set_index("Date").sort_index()
    return completed_price_bars(
        validate_price_bars(data[["Open", "Close", "High", "Low", "Volume"]], clean),
        interval,
    )


def fetch_yfinance_bars(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception as exc:
        raise RuntimeError("yfinance is not installed.") from exc
    clean = str(symbol).strip().upper()
    if not clean:
        raise ValueError("Enter a ticker symbol.")
    fetch_interval = "1h" if interval == "4h" else interval
    data = yf.download(
        tickers=clean,
        period=period,
        interval=fetch_interval,
        auto_adjust=True,
        progress=False,
        threads=False,
        prepost=False,
        multi_level_index=False,
    )
    if data is None or data.empty:
        raise ValueError(f"No price data returned for {clean}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    required = ["Close", "High", "Low"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    columns = (["Open"] if "Open" in data.columns else []) + required + (["Volume"] if "Volume" in data.columns else [])
    data = data[columns].dropna(subset=required)
    if interval == "4h":
        aggregations = {"Close": "last", "High": "max", "Low": "min"}
        if "Open" in data.columns:
            aggregations["Open"] = "first"
        if "Volume" in data.columns:
            aggregations["Volume"] = "sum"
        data = data.resample("4h").agg(aggregations).dropna(subset=required)
    return completed_price_bars(validate_price_bars(data, clean), interval)


def fetch_price_bars(
    symbol: str,
    period: str,
    interval: str,
    source: str,
    api_key: str | None = None,
    api_secret: str | None = None,
) -> pd.DataFrame:
    if "alpaca" in str(source).lower():
        return fetch_alpaca_bars(symbol, period, interval, api_key, api_secret)
    if "yfinance" in str(source).lower() or "yahoo" in str(source).lower():
        return fetch_yfinance_bars(symbol, period, interval)
    raise ValueError("Automation and scanning require Alpaca or yfinance ticker data.")


@dataclass(frozen=True)
class NewsItem:
    headline: str
    summary: str
    created_at: str
    source: str
    url: str


@dataclass(frozen=True)
class CompanyResearchContext:
    symbol: str
    event_risk: str
    event_detail: str
    news_status: str
    fundamentals_status: str
    headlines: list[NewsItem]


def fetch_alpaca_news(
    symbol: str,
    api_key: str | None,
    api_secret: str | None,
    *,
    days: int = 7,
    limit: int = 10,
    timeout: int = 15,
) -> list[NewsItem]:
    clean = str(symbol).strip().upper()
    if not clean or not api_key or not api_secret:
        return []
    start = (datetime.now(UTC) - timedelta(days=max(1, days))).isoformat().replace("+00:00", "Z")
    params = urlencode({"symbols": clean, "start": start, "sort": "desc", "limit": max(1, min(limit, 50))})
    request = Request(
        f"{ALPACA_DATA_BASE_URL}/v1beta1/news?{params}",
        headers={
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [
        NewsItem(
            headline=str(item.get("headline") or ""),
            summary=str(item.get("summary") or ""),
            created_at=str(item.get("created_at") or ""),
            source=str(item.get("source") or ""),
            url=str(item.get("url") or ""),
        )
        for item in payload.get("news", [])
        if item.get("headline")
    ]


def build_company_research_context(
    symbol: str,
    api_key: str | None = None,
    api_secret: str | None = None,
    news: list[NewsItem] | None = None,
) -> CompanyResearchContext:
    if news is not None:
        items = list(news)
    else:
        try:
            items = fetch_alpaca_news(symbol, api_key, api_secret)
        except Exception:
            items = []
    combined = " ".join(f"{item.headline} {item.summary}" for item in items).lower()
    event_words = ("earnings", "guidance", "revenue", "profit warning", "sec investigation", "merger", "acquisition")
    event_mentions = [word for word in event_words if word in combined]
    if not items:
        event_risk = "Not connected"
        event_detail = "No recent-news response is available. Upcoming earnings are not connected and do not block trades."
        news_status = "Not connected"
    elif event_mentions:
        event_risk = "Review recent news"
        event_detail = f"Recent headlines mention {', '.join(event_mentions[:3])}. This is context, not an upcoming-event calendar."
        news_status = f"{len(items)} recent headline(s)"
    else:
        event_risk = "No recent headline flag"
        event_detail = "Recent Alpaca headlines contain no obvious event keywords. Upcoming earnings are still not connected."
        news_status = f"{len(items)} recent headline(s)"
    return CompanyResearchContext(
        symbol=str(symbol).strip().upper(),
        event_risk=event_risk,
        event_detail=event_detail,
        news_status=news_status,
        fundamentals_status="Not connected",
        headlines=items,
    )
