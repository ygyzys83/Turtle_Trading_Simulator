"""Research-only horizontal ranges. No strategy, sizing, or order dependencies."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from agentloop_trader.indicators import calc_atr
from agentloop_trader.rectangle_timeframes import analysis_bars, completed_daily_bars, validate_interval


DETECTOR_VERSION = 3
WINDOWS = tuple(month*21 for month in range(3, 25))
WEEKLY_WINDOWS = tuple(round(month*52/12) for month in range(3, 25))


@dataclass(frozen=True)
class Rectangle:
    symbol: str
    start: str
    end: str
    bars: int
    support: float
    resistance: float
    tolerance: float
    atr: float
    last_close: float
    quality: float
    containment: float
    support_touches: tuple[str, ...]
    resistance_touches: tuple[str, ...]
    interval: str = "1d"

    @property
    def months(self) -> float:
        return round(self.bars / 21 if self.interval == "1d" else self.bars*12/52, 1)

    @property
    def position_percent(self) -> float:
        return 100 * (self.last_close - self.support) / (self.resistance - self.support)

    @property
    def location(self) -> str:
        if self.last_close < self.support - self.tolerance:
            return "Below range"
        if self.last_close > self.resistance + self.tolerance:
            return "Above range"
        if self.position_percent <= 25:
            return "Near support"
        if self.position_percent >= 75:
            return "Near resistance"
        return "Middle"

    def record(self) -> dict[str, Any]:
        return asdict(self)


def rectangle_from_record(row: dict[str, Any]) -> Rectangle:
    values = dict(row)
    for name in ("support_touches", "resistance_touches"):
        values[name] = tuple(values[name])
    result = Rectangle(**values)
    validate_interval(result.interval)
    if not (0 < result.support < result.resistance and result.tolerance > 0):
        raise ValueError("Invalid saved rectangle boundaries.")
    return result


def daily_bars(data: pd.DataFrame, now=None) -> pd.DataFrame:
    """Normalize daily timestamps to session dates after excluding forming bars."""
    return completed_daily_bars(data, now)


def _separated(indices: np.ndarray, gap: int = 5) -> np.ndarray:
    selected: list[int] = []
    for value in indices:
        if not selected or int(value) - selected[-1] >= gap:
            selected.append(int(value))
    return np.asarray(selected, dtype=int)


def _rectangle_quality(containment: float, edge_drift: float, touch_count: int, span: float) -> float:
    # Five points reward >=95% actual-box containment; this is not an eligibility gate.
    # The remaining structural factors still matter and the maximum stays at 100.
    clean_bonus = 5 if containment >= .95 else 0
    return round(25*containment + clean_bonus + 25*(1-min(edge_drift/.30, 1))
                 + 25*min(touch_count/10, 1) + 20*min(span/.7, 1), 1)


def detect_rectangles(symbol: str, data: pd.DataFrame, *, interval="1d", now=None) -> list[Rectangle]:
    """Evaluate bounded 3–24 month windows using only data completed at `now`.

    Months are approximately 21 trading sessions or 52/12 weeks. Thresholds are fixed research
    heuristics, not estimated breakout probabilities. Two bars confirm pivots;
    an unconfirmed recent extreme is never counted as a boundary test.
    """
    validate_interval(interval)
    frame = analysis_bars(data, interval, now)
    windows = WINDOWS if interval == "1d" else WEEKLY_WINDOWS
    if len(frame) < min(windows):
        return []
    close = frame.Close.to_numpy(dtype=float)
    high = frame.High.to_numpy(dtype=float)
    low = frame.Low.to_numpy(dtype=float)
    atrs = calc_atr(close, 14, high, low)
    atr = float(atrs[-1] or 0)
    if not np.isfinite(atr) or atr <= 0:
        return []
    piv_high = np.zeros(len(frame), dtype=bool)
    piv_low = np.zeros(len(frame), dtype=bool)
    for i in range(2, len(frame) - 2):
        piv_high[i] = high[i] >= max(high[i-2:i+3]) and high[i] > max(high[i-2:i])
        piv_low[i] = low[i] <= min(low[i-2:i+3]) and low[i] < min(low[i-2:i])
    matches: list[Rectangle] = []
    dates = frame.index.strftime("%Y-%m-%d").tolist()
    for window in windows:
        if window > len(frame):
            continue
        start = len(frame) - window
        c, h, l = close[start:], high[start:], low[start:]
        # Sparse histories must not turn a year-long sequence into a one-month box.
        if (frame.index[-1] - frame.index[start]).days > window * (2 if interval == "1d" else 8):
            continue
        if interval == "1w" and np.any(np.diff(frame.index[start:].to_period("W-FRI").asi8) != 1):
            continue
        support, resistance = float(np.quantile(l, .10)), float(np.quantile(h, .90))
        width = resistance - support
        if width <= 0 or not .03 <= width / support <= .45 or width < 3 * atr:
            continue
        tolerance = min(max(.35 * atr, .003 * close[-1]), .10 * width)
        lows = np.flatnonzero(piv_low[start:])
        highs = np.flatnonzero(piv_high[start:])
        lows = lows[(lows >= 2) & (np.abs(l[lows] - support) <= 2 * tolerance)]
        highs = highs[(highs >= 2) & (np.abs(h[highs] - resistance) <= 2 * tolerance)]
        gap = 5 if interval == "1d" else 2
        lows, highs = _separated(lows, gap), _separated(highs, gap)
        if len(lows) < 2 or len(highs) < 2:
            continue
        support, resistance = float(np.median(l[lows])), float(np.median(h[highs]))
        width = resistance - support
        if width <= 0 or not .03 <= width / support <= .45 or width < 3 * atr:
            continue
        lows = lows[np.abs(l[lows] - support) <= tolerance]
        highs = highs[np.abs(h[highs] - resistance) <= tolerance]
        if len(lows) < 2 or len(highs) < 2:
            continue
        span = min(lows[-1] - lows[0], highs[-1] - highs[0]) / window
        if span < .35:
            continue
        # Qualification measures closes against the actual plotted rectangle.
        # Zone tolerance still applies to touch matching, wicks and the latest-price check.
        containment = float(np.mean((c >= support) & (c <= resistance)))
        if containment < .90 or not support - tolerance <= c[-1] <= resistance + tolerance:
            continue
        if float(np.mean((l < support - 2*tolerance) | (h > resistance + 2*tolerance))) > .10:
            continue
        if min(l) < support - .4*width or max(h) > resistance + .4*width:
            continue
        drift = abs(float(np.polyfit(np.arange(window), c, 1)[0])) * window / width
        edge_drift = max(
            abs(float(np.polyfit(lows, l[lows], 1)[0])),
            abs(float(np.polyfit(highs, h[highs], 1)[0])),
        ) * window / width
        if drift > .5 or edge_drift > .30:
            continue
        quality = _rectangle_quality(containment, edge_drift, len(lows)+len(highs), span)
        matches.append(Rectangle(
            symbol=str(symbol).strip().upper(), start=dates[start], end=dates[-1], bars=window,
            support=support, resistance=resistance, tolerance=tolerance, atr=atr,
            last_close=float(c[-1]), quality=quality, containment=containment,
            support_touches=tuple(dates[start+i] for i in lows),
            resistance_touches=tuple(dates[start+i] for i in highs),
            interval=interval,
        ))
    return sorted(matches, key=lambda r: (r.quality, r.bars), reverse=True)


def filter_rectangles(rows: list[Rectangle], months=(3, 24), location="All",
                      new_symbols: set[str] | None = None) -> list[Rectangle]:
    """Filter all detected durations, then select one best rectangle per symbol."""
    best: dict[tuple[str, str], Rectangle] = {}
    for row in sorted(rows, key=lambda r: (r.quality, r.bars), reverse=True):
        if not months[0] <= row.months <= months[1]:
            continue
        if location != "All" and row.location != location:
            continue
        if new_symbols is not None and row.symbol not in new_symbols:
            continue
        best.setdefault((row.symbol, row.interval), row)
    return list(best.values())


def rectangle_records(rows: list[Rectangle]) -> list[dict[str, Any]]:
    return [{"Ticker": r.symbol, "Timeframe": "Daily" if r.interval == "1d" else "Weekly", "Range (months)": r.months,
             "Support": round(r.support, 2), "Resistance": round(r.resistance, 2),
             "Price": round(r.last_close, 2), "Price location": r.location,
             "Position in range (%)": round(r.position_percent, 1)} for r in rows]


def refresh_saved_rectangle(original: Rectangle, original_anchor: float, data: pd.DataFrame,
                            *, now=None) -> tuple[Rectangle, str]:
    """Retain saved geometry, explicitly adjusting its basis after corporate actions."""
    frame = analysis_bars(data, original.interval, now)
    if pd.Timestamp(original.end) not in frame.index:
        raise ValueError("Original rectangle date is missing; its price basis cannot be verified.")
    anchor = float(frame.loc[pd.Timestamp(original.end), "Close"])
    if original_anchor <= 0:
        raise ValueError("Invalid saved price basis.")
    factor = anchor / original_anchor
    current = replace(original, support=original.support*factor,
                      resistance=original.resistance*factor, tolerance=original.tolerance*factor,
                      atr=original.atr*factor, last_close=float(frame.Close.iloc[-1]))
    adjustment = "Price basis adjusted for corporate actions." if abs(factor-1) > .00001 else ""
    return current, adjustment
