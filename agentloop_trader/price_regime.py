from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PriceRegime:
    period: str
    start: str
    end: str
    bars: int
    direction: str
    path: str
    label: str
    return_percent: float
    annualized_return_percent: float | None
    annualized_volatility_percent: float
    trend_strength: float
    trend_fit_percent: float


@dataclass(frozen=True)
class RegimeDependency:
    current_regime: str
    strongest_regime: str
    current_match: str
    dependency: str
    positive_excess_concentration_percent: float | None
    outperforming_periods: int
    tested_periods: int
    summary: str


def _date_label(value: Any) -> str:
    try:
        return pd.Timestamp(value).strftime("%b %Y")
    except (TypeError, ValueError):
        return "Not recorded"


def _elapsed_years(index: pd.Index) -> float:
    if len(index) < 2:
        return 0.0
    try:
        elapsed = (pd.Timestamp(index[-1]) - pd.Timestamp(index[0])).total_seconds()
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, elapsed / (365.2425 * 24 * 60 * 60))


def _elapsed_time_axis_years(index: pd.Index) -> np.ndarray:
    """Return each observation's actual elapsed time instead of assuming evenly spaced bars."""
    if len(index) < 2:
        return np.zeros(len(index), dtype=float)
    try:
        timestamps = pd.to_datetime(index, utc=True)
        elapsed = (timestamps - timestamps[0]).total_seconds().to_numpy(dtype=float)
        return elapsed / (365.2425 * 24 * 60 * 60)
    except (TypeError, ValueError, AttributeError):
        return np.linspace(0.0, 1.0, len(index))


def classify_price_regime(data: pd.DataFrame, period: str = "Complete history") -> PriceRegime:
    close = data.get("Close", pd.Series(dtype=float)).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    close = close[close > 0]
    if len(close) < 3:
        return PriceRegime(
            period=period,
            start=_date_label(close.index[0]) if len(close) else "Not recorded",
            end=_date_label(close.index[-1]) if len(close) else "Not recorded",
            bars=len(close),
            direction="Not enough data",
            path="Not enough data",
            label="Not enough data",
            return_percent=0.0,
            annualized_return_percent=None,
            annualized_volatility_percent=0.0,
            trend_strength=0.0,
            trend_fit_percent=0.0,
        )

    years = _elapsed_years(close.index)
    first = float(close.iloc[0])
    last = float(close.iloc[-1])
    total_return = (last / first - 1.0) * 100 if first > 0 else 0.0
    annualized_return = None
    if years > 1.0 and first > 0 and last > 0:
        annualized_return = ((last / first) ** (1.0 / years) - 1.0) * 100

    log_close = np.log(close.to_numpy(dtype=float))
    x = _elapsed_time_axis_years(close.index)
    if not np.isfinite(x).all() or float(x[-1] - x[0]) <= 0:
        x = np.linspace(0.0, years if years > 0 else 1.0, len(log_close))
    slope, intercept = np.polyfit(x, log_close, 1)
    fitted = intercept + slope * x
    residual_sum = float(np.sum((log_close - fitted) ** 2))
    total_sum = float(np.sum((log_close - np.mean(log_close)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 0.0

    log_returns = np.diff(log_close)
    observations_per_year = len(log_returns) / years if years > 0 else len(log_returns)
    volatility = (
        float(np.std(log_returns, ddof=1)) * sqrt(max(1.0, observations_per_year)) * 100
        if len(log_returns) > 1 else 0.0
    )
    annualized_log_trend = float(slope) * 100
    raw_trend_strength = annualized_log_trend / volatility if volatility > 0 else 0.0
    # A small endpoint drift in a noisy range is not a directional trend. Discount
    # the slope when the fitted path explains little of the observed price movement.
    trend_strength = raw_trend_strength * sqrt(max(0.0, min(1.0, r_squared)))

    if trend_strength >= 0.60:
        direction = "Strong uptrend"
    elif trend_strength >= 0.15:
        direction = "Mild uptrend"
    elif trend_strength <= -0.60:
        direction = "Strong downtrend"
    elif trend_strength <= -0.15:
        direction = "Mild downtrend"
    else:
        direction = "Sideways"

    segment_returns: list[float] = []
    for segment in np.array_split(np.arange(len(close)), min(5, len(close))):
        if len(segment) >= 2:
            start_price = float(close.iloc[int(segment[0])])
            end_price = float(close.iloc[int(segment[-1])])
            segment_returns.append(end_price / start_price - 1.0 if start_price > 0 else 0.0)
    expected_sign = 1 if "uptrend" in direction else -1 if "downtrend" in direction else 0
    if expected_sign == 0:
        sign_agreement = sum(abs(value) < 0.02 for value in segment_returns) / max(1, len(segment_returns))
    else:
        sign_agreement = sum((value > 0) == (expected_sign > 0) for value in segment_returns) / max(1, len(segment_returns))
    if r_squared >= 0.55 and sign_agreement >= 0.75:
        path = "Persistent"
    elif r_squared < 0.20 or sign_agreement < 0.50:
        path = "Choppy"
    else:
        path = "Mixed"

    return PriceRegime(
        period=period,
        start=_date_label(close.index[0]),
        end=_date_label(close.index[-1]),
        bars=len(close),
        direction=direction,
        path=path,
        label=f"{direction}, {path.lower()}",
        return_percent=round(total_return, 2),
        annualized_return_percent=None if annualized_return is None else round(annualized_return, 2),
        annualized_volatility_percent=round(volatility, 2),
        trend_strength=round(trend_strength, 3),
        trend_fit_percent=round(max(0.0, min(1.0, r_squared)) * 100, 1),
    )


def price_regime_sections(
    data: pd.DataFrame,
    *,
    older_fraction: float = 0.55,
    latest_fraction: float = 0.20,
) -> list[tuple[int, int, PriceRegime]]:
    length = len(data)
    if length < 3:
        return [(0, length, classify_price_regime(data, "Complete history"))]
    older_end = max(1, min(length - 2, int(length * older_fraction)))
    latest_start = max(older_end + 1, min(length - 1, int(length * (1.0 - latest_fraction))))
    older_percent = int(round(older_fraction * 100))
    latest_percent = int(round(latest_fraction * 100))
    newer_percent = max(0, 100 - older_percent - latest_percent)
    ranges = [
        (f"Older {older_percent}%", 0, older_end),
        (f"Newer {newer_percent}%", older_end, latest_start),
        (f"Latest {latest_percent}%", latest_start, length),
    ]
    return [
        (start, end, classify_price_regime(data.iloc[start:end], label))
        for label, start, end in ranges
        if end - start >= 2
    ]


def rolling_price_regimes(data: pd.DataFrame, blocks: int = 6) -> list[tuple[int, int, PriceRegime]]:
    if len(data) < 3:
        return []
    indices = [segment for segment in np.array_split(np.arange(len(data)), min(blocks, len(data) // 2)) if len(segment) >= 2]
    return [
        (
            int(segment[0]),
            int(segment[-1]) + 1,
            classify_price_regime(data.iloc[int(segment[0]):int(segment[-1]) + 1], f"Period {number}"),
        )
        for number, segment in enumerate(indices, start=1)
    ]


def _trade_stats(trades: list[dict[str, Any]], allocated_capital: float) -> tuple[int, float, float]:
    pnl = [float(row.get("pnl", 0.0)) for row in trades]
    return (
        len(pnl),
        round(sum(pnl) / allocated_capital * 100, 2) if allocated_capital > 0 else 0.0,
        round(sum(value > 0 for value in pnl) / len(pnl) * 100, 1) if pnl else 0.0,
    )


def strategy_regime_rows(
    data: pd.DataFrame,
    trades: list[dict[str, Any]],
    allocated_capital: float,
    *,
    blocks: int = 6,
    period_evaluator: Callable[[int, int], dict[str, Any]] | None = None,
    period_ranges: list[tuple[int, int, PriceRegime]] | None = None,
) -> list[dict[str, Any]]:
    """Compare strategy and buy-and-hold over explicit or evenly split periods."""
    rows: list[dict[str, Any]] = []
    ranges = period_ranges if period_ranges is not None else rolling_price_regimes(data, blocks=blocks)
    for start, end, regime in ranges:
        if period_evaluator is None:
            period_trades = [
                row for row in trades
                if start <= int(row.get("entry_bar", -1)) < end
            ]
            trade_count, strategy_return, win_rate = _trade_stats(period_trades, allocated_capital)
        else:
            evaluated = period_evaluator(start, end)
            trade_count = int(evaluated.get("total_trades", 0))
            strategy_return = round(float(evaluated.get("allocated_return_pct", 0.0)), 2)
            win_rate = round(float(evaluated.get("win_rate", 0.0)), 1)
        segment = data.iloc[start:end]
        close = segment["Close"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        close = close[close > 0]
        buy_hold = (float(close.iloc[-1]) / float(close.iloc[0]) - 1.0) * 100 if len(close) >= 2 and close.iloc[0] > 0 else 0.0
        rows.append({
            "Period": regime.period,
            "Dates": f"{regime.start} to {regime.end}",
            "Price behavior": regime.label,
            "Ticker return": f"{regime.return_percent:+.2f}%",
            "Strategy return": f"{strategy_return:+.2f}%",
            "Buy and hold": f"{buy_hold:+.2f}%",
            "Difference": f"{strategy_return - buy_hold:+.2f}%",
            "Completed trades": trade_count,
            "Win rate": f"{win_rate:.0f}%" if trade_count else "No trades",
            "_strategy_return": strategy_return,
            "_buy_hold_return": buy_hold,
            "_excess_return": strategy_return - buy_hold,
            "_direction": regime.direction,
            "_label": regime.label,
        })
    return rows


def _direction_family(direction: str) -> str:
    lowered = str(direction).lower()
    if "uptrend" in lowered:
        return "Uptrend"
    if "downtrend" in lowered:
        return "Downtrend"
    if "sideways" in lowered:
        return "Sideways"
    return "Unknown"


def summarize_regime_dependency(
    data: pd.DataFrame,
    regime_rows: list[dict[str, Any]],
    *,
    latest_fraction: float = 0.20,
) -> RegimeDependency:
    latest_sections = price_regime_sections(data, latest_fraction=latest_fraction)
    current = latest_sections[-1][2] if latest_sections else classify_price_regime(data)
    populated = [row for row in regime_rows if int(row.get("Completed trades", 0)) > 0]
    if not populated:
        return RegimeDependency(
            current_regime=current.label,
            strongest_regime="Not enough trades",
            current_match="Not enough evidence",
            dependency="Not enough evidence",
            positive_excess_concentration_percent=None,
            outperforming_periods=0,
            tested_periods=0,
            summary="There are not enough completed trades to judge which price behavior favored this strategy.",
        )

    strongest = max(populated, key=lambda row: float(row.get("_excess_return", 0.0)))
    positive = [max(0.0, float(row.get("_excess_return", 0.0))) for row in populated]
    positive_total = sum(positive)
    concentration = max(positive) / positive_total * 100 if positive_total > 0 else None
    outperforming = sum(float(row.get("_excess_return", 0.0)) > 0 for row in populated)
    outperforming_directions = {
        _direction_family(str(row.get("_direction"))) for row in populated
        if float(row.get("_excess_return", 0.0)) > 0
    }
    if not positive_total:
        dependency = "No period beat buy and hold"
    elif len(outperforming_directions) >= 3 and concentration < 55:
        dependency = "Broad across price behavior"
    elif len(outperforming_directions) >= 2 and concentration < 75:
        dependency = "Mixed"
    else:
        dependency = "Concentrated in one type of price behavior"

    if current.label == str(strongest.get("_label")):
        current_match = "Strong match"
    elif _direction_family(current.direction) == _direction_family(str(strongest.get("_direction"))):
        current_match = "Direction matches"
    else:
        current_match = "Different from the strongest history"
    summary = (
        f"The latest price-section behavior is {current.label}. The strategy's highest result relative to buy and hold was in "
        f"{str(strongest.get('_label'))}. Its advantage was {dependency.lower()} and it beat buy and hold in "
        f"{outperforming} of {len(populated)} periods with completed trades."
    )
    return RegimeDependency(
        current_regime=current.label,
        strongest_regime=str(strongest.get("_label")),
        current_match=current_match,
        dependency=dependency,
        positive_excess_concentration_percent=None if concentration is None else round(concentration, 1),
        outperforming_periods=outperforming,
        tested_periods=len(populated),
        summary=summary,
    )


def price_regime_records(
    data: pd.DataFrame,
    *,
    older_fraction: float = 0.55,
    latest_fraction: float = 0.20,
) -> list[dict[str, Any]]:
    full = classify_price_regime(data)
    regimes = [full] + [
        item[2] for item in price_regime_sections(
            data,
            older_fraction=older_fraction,
            latest_fraction=latest_fraction,
        )
    ]
    return [
        {
            "Price section": regime.period,
            "Dates": f"{regime.start} to {regime.end}",
            "Price behavior": regime.label,
            "Ticker return": f"{regime.return_percent:+.2f}%",
            "Annualized return": (
                "Not shown" if regime.annualized_return_percent is None
                else f"{regime.annualized_return_percent:+.2f}%"
            ),
            "Annualized volatility": f"{regime.annualized_volatility_percent:.2f}%",
            "Trend consistency": f"{regime.trend_fit_percent:.0f}%",
        }
        for regime in regimes
    ]
