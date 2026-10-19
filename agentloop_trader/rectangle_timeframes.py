"""US equity session-aware daily/weekly candles for rectangle research only."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from agentloop_trader.market_data import validate_price_bars


TIMEFRAME_LABELS = {"1d": "Daily", "1w": "Weekly"}


def validate_interval(interval: str) -> str:
    if interval not in TIMEFRAME_LABELS:
        raise ValueError("Rectangle timeframe must be Daily (1d) or Weekly (1w).")
    return interval


def utc_now(now=None) -> pd.Timestamp:
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    return current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")


@lru_cache(maxsize=16)
def session_schedule(first_year: int, last_year: int) -> pd.DataFrame:
    # Imported lazily so a missing optional runtime dependency produces an
    # actionable scanner error instead of preventing the whole app from opening.
    try:
        import exchange_calendars as calendars
    except ImportError:
        raise RuntimeError("Install the updated requirements.txt to enable the rectangle trading calendar.") from None
    return calendars.get_calendar("XNAS", start=f"{first_year}-01-01", end=f"{last_year}-12-31").schedule


def completed_daily_bars(data: pd.DataFrame, now=None) -> pd.DataFrame:
    clean = validate_price_bars(data)
    index = pd.DatetimeIndex(clean.index)
    if index.tz is not None:
        index = index.tz_convert("America/New_York").tz_localize(None)
    clean.index = index.normalize()
    clean = clean.loc[~clean.index.duplicated(keep="last")].sort_index()
    current = utc_now(now)
    schedule = session_schedule(clean.index[0].year, max(clean.index[-1].year, current.year))
    completed = schedule.index[schedule["close"] <= current]
    result = clean.loc[clean.index.isin(completed)].copy()
    if result.empty:
        raise ValueError("No completed stock trading sessions are available.")
    return result


@lru_cache(maxsize=16)
def _weeks(first_year: int, last_year: int):
    # Include whole cross-year weeks at both edges of the requested range.
    schedule = session_schedule(first_year-1, last_year+1)
    groups = []
    for _, group in schedule.groupby(schedule.index.to_period("W-FRI")):
        groups.append((group.index, group["close"].iloc[-1]))
    return tuple(groups)


@lru_cache(maxsize=16)
def _week_metadata(first_year: int, last_year: int):
    weeks = _weeks(first_year, last_year)
    return pd.DataFrame(
        {"sessions": [len(sessions) for sessions, _ in weeks],
         "last_session": [sessions[-1] for sessions, _ in weeks],
         "close": [close for _, close in weeks]},
        index=pd.PeriodIndex([sessions[-1].to_period("W-FRI") for sessions, _ in weeks]),
    )


def weekly_bars(data: pd.DataFrame, now=None) -> pd.DataFrame:
    """Aggregate complete exchange weeks; missing sessions never form a full bar.

    Labels are the actual final session (Thursday on a Good Friday week).
    Every expected session must be present, including holiday/early-close weeks.
    The first partially downloaded week and the ongoing week are excluded.
    """
    frame = completed_daily_bars(data, now)
    current = utc_now(now)
    groups = frame.groupby(frame.index.to_period("W-FRI"))
    aggregations = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in frame:
        aggregations["Volume"] = "sum"
    result = groups.agg(aggregations)
    metadata = _week_metadata(frame.index[0].year, max(frame.index[-1].year, current.year)).reindex(result.index)
    complete = groups.size().eq(metadata.sessions) & (metadata["close"] <= current)
    result = result.loc[complete]
    result.index = pd.DatetimeIndex(metadata.loc[complete, "last_session"], name=frame.index.name)
    result.attrs.update(frame.attrs)
    return result


def analysis_bars(data: pd.DataFrame, interval="1d", now=None) -> pd.DataFrame:
    validate_interval(interval)
    return completed_daily_bars(data, now) if interval == "1d" else weekly_bars(data, now)


def latest_completed_date(interval="1d", now=None) -> pd.Timestamp:
    validate_interval(interval)
    current = utc_now(now)
    if interval == "1d":
        schedule = session_schedule(current.year-1, current.year+1)
        return schedule.index[schedule["close"] <= current][-1]
    completed = [sessions[-1] for sessions, close in _weeks(current.year-1, current.year+1) if close <= current]
    return completed[-1]
