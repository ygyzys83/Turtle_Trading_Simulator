"""Fixed, optional setup assessments on completed rectangle candles only.

These research heuristics never qualify a rectangle or change its ranking.
Input has already been normalized to the selected timeframe by analysis_bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agentloop_trader.rectangle_timeframes import session_schedule, validate_interval


SIGNAL_LABELS = {
    "higher_lows": "Higher lows",
    "tight_near_resistance": "Tight near resistance",
    "supportive_volume": "Supportive volume",
}


def signal_label(value: bool | None) -> str:
    return "Unknown" if value is None else "✓" if value else "—"


def _complete(frame: pd.DataFrame, count: int, interval: str) -> bool:
    if len(frame) != count:
        return False
    if interval == "1w":
        expected = pd.period_range(frame.index[0].to_period("W-FRI"),
                                   frame.index[-1].to_period("W-FRI"), freq="W-FRI")
        return frame.index.to_period("W-FRI").equals(expected)
    schedule = session_schedule(frame.index[0].year, frame.index[-1].year)
    return frame.index.equals(schedule.loc[frame.index[0]:frame.index[-1]].index)


def analyze_breakout_signals(frame: pd.DataFrame, *, start, end, support: float,
                             resistance: float, atr: float, interval="1d") -> dict:
    """Return serializable metadata; never examine bars after the saved end.

    Higher lows: latest two separated 2-left/2-right confirmed lows in 60 daily
    sessions / 13 weeks, rising by max(0.25 ATR, 2% of box height), with no later
    undercut. Tightness: last 10 daily / 4 weekly candles vs preceding equal
    block; 80% above midpoint, latest close inside upper third, median true range
    at least 20% smaller and <=20% / 35% of box height. Volume: last 20 daily /
    8 weekly bars; median declining-bar volume <=80% of advancing-bar volume,
    with >=3 / >=2 observations per side. Direction uses close-to-close change.
    Weekly volume is normalized per scheduled session to avoid holiday bias.
    """
    validate_interval(interval)
    base = frame.loc[pd.Timestamp(start):pd.Timestamp(end)]
    result = dict(higher_lows=None, tight_near_resistance=None, supportive_volume=None,
                  signal_details=(), higher_low_dates=(), higher_low_prices=(),
                  signals_as_of="")
    reasons = []
    if base.empty:
        result["signal_details"] = ("No completed range candles available.",)*3
        return result
    result["signals_as_of"] = base.index[-1].strftime("%Y-%m-%d")
    width = resistance-support
    daily = interval == "1d"

    lookback = 60 if daily else 13
    recent = base.tail(lookback)
    if not _complete(recent, lookback, interval) or not np.isfinite(atr) or atr <= 0:
        reasons.append(f"Need {lookback} consecutive completed candles and valid ATR.")
    else:
        low = recent.Low.to_numpy(dtype=float)
        pivots = [i for i in range(2, len(low)-2)
                  if low[i] <= min(low[i-2:i+3]) and low[i] < min(low[i-2:i])]
        selected = []
        for i in reversed(pivots):
            if not selected or selected[-1]-i >= (5 if daily else 2):
                selected.append(i)
            if len(selected) == 2:
                break
        if len(selected) < 2:
            reasons.append("Need two separated pullback lows, each confirmed by two later candles.")
        else:
            first, second = sorted(selected)
            rise, required = float(low[second]-low[first]), max(.25*atr, .02*width)
            intact = bool(np.min(low[second:]) >= low[second] and base.Close.iloc[-1] > low[second])
            result["higher_lows"] = bool(rise >= required-1e-10*width and intact)
            result["higher_low_dates"] = tuple(recent.index[[first, second]].strftime("%Y-%m-%d"))
            result["higher_low_prices"] = (float(low[first]), float(low[second]))
            reasons.append(f"Latest confirmed lows changed by {rise/width:.1%} of box height; "
                           f"required rise {required/width:.1%}. "
                           + ("Latest low still holds." if intact else "Latest low has been undercut or lost on the close."))

    count = 10 if daily else 4
    context = base.tail(2*count+1)
    if not _complete(context, 2*count+1, interval):
        reasons.append(f"Need {2*count+1} consecutive completed candles to assess tightening.")
    else:
        current = context.tail(count)
        previous_close = context.Close.shift(1)
        ranges = pd.concat([context.High-context.Low, (context.High-previous_close).abs(),
                            (context.Low-previous_close).abs()], axis=1).max(axis=1).iloc[1:]
        before, after = float(ranges.iloc[:count].median()), float(ranges.iloc[count:].median())
        held = float((current.Close > (support+resistance)/2).mean())
        position = (float(current.Close.iloc[-1])-support)/width
        result["tight_near_resistance"] = bool(
            held >= .8 and 2/3 <= position <= 1 and before > 0
            and after <= .8*before+1e-10*width and after <= (.20 if daily else .35)*width)
        reasons.append(f"Last {count} candles: {held:.0%} of closes above midpoint; "
                       f"latest at {position:.0%} of box height. Median candle range including wicks/gaps "
                       f"{after/width:.1%} of box height vs {before/width:.1%} previously (requires ≥20% contraction "
                       f"and ≤{20 if daily else 35}% of box height).")

    count, minimum = (20, 3) if daily else (8, 2)
    context = base.tail(count+1)
    if not _complete(context, count+1, interval) or "Volume" not in context:
        reasons.append(f"Need {count+1} consecutive completed candles with usable volume.")
    else:
        volumes = context.Volume.iloc[1:].astype(float).copy()
        incomplete_weeks = set(frame.attrs.get("incomplete_volume_weeks", ()))
        partial_volume = not daily and any(str(w) in incomplete_weeks for w in volumes.index.to_period("W-FRI"))
        if partial_volume or not bool((np.isfinite(volumes) & (volumes > 0)).all()):
            reasons.append("Recent volume is missing, zero, or invalid; no positive assessment.")
        else:
            if not daily:
                schedule = session_schedule(context.index[0].year-1, context.index[-1].year+1)
                sessions = schedule.groupby(schedule.index.to_period("W-FRI")).size()
                volumes /= sessions.reindex(volumes.index.to_period("W-FRI")).to_numpy()
            changes = context.Close.diff().iloc[1:]
            up, down = volumes[changes > 0], volumes[changes < 0]
            if min(len(up), len(down)) < minimum:
                reasons.append(f"Need at least {minimum} advancing and {minimum} declining candles with volume.")
            else:
                ratio = float(down.median()/up.median())
                result["supportive_volume"] = bool(ratio <= .8+1e-10)
                reasons.append(f"Last {count} candles: median pullback volume is {ratio:.0%} of advance volume "
                               f"(requires ≤80%; {len(down)} declines / {len(up)} advances). "
                               + ("Weekly volume is compared per scheduled session." if not daily
                                  else "Uses the scan's displayed data feed."))
    result["signal_details"] = tuple(reasons)
    return result
