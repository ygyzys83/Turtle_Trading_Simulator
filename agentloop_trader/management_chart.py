from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from agentloop_trader.indicators import calc_rsi
from agentloop_trader.ui_theme import CHART_COLORS


MANAGEMENT_CHART_HISTORY = {
    "1m": "5d",
    "5m": "1mo",
    "15m": "1mo",
    "30m": "3mo",
    "1h": "6mo",
    "4h": "1y",
    "1d": "2y",
}


def management_chart_history(interval: str, fallback: str = "1y") -> str:
    """Use compact chart-only history without changing the saved trading plan."""
    return MANAGEMENT_CHART_HISTORY.get(str(interval), str(fallback))


def trim_management_data(
    market_data: pd.DataFrame,
    settings: dict[str, Any],
    *,
    visible_bars: int = 90,
) -> pd.DataFrame:
    """Keep the visible bars plus enough warm-up data for every saved indicator."""
    windows = [
        14,
        int(settings.get("entry_window", 20)),
        int(settings.get("exit_window", 10)),
        int(settings.get("moving_average_window", 50)),
        int(settings.get("pullback_average_length", 20)),
        int(settings.get("momentum_turn_length", 10)),
        int(settings.get("rsi_length", 14)),
        int(settings.get("rsi_swing_lookback", 24)),
    ]
    keep = max(40, int(visible_bars)) + max(windows) + 5
    if len(market_data) <= keep:
        return market_data
    attrs = dict(market_data.attrs)
    trimmed = market_data.tail(keep).copy()
    trimmed.attrs.update(attrs)
    return trimmed


def _number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _level(name: str, value: Any, color: str, dash: str = "dash", width: float = 1.2, priority: int = 0) -> dict[str, Any] | None:
    number = _number(value)
    if number is None:
        return None
    return {
        "name": name,
        "value": number,
        "color": color,
        "dash": dash,
        "width": width,
        "priority": priority,
    }


def _deduplicate_levels(levels: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    """Keep one clearly labeled line when two rules currently share a price."""
    by_price: dict[float, dict[str, Any]] = {}
    order: list[float] = []
    for level in levels:
        if level is None:
            continue
        key = round(float(level["value"]), 8)
        if key not in by_price:
            by_price[key] = level
            order.append(key)
        elif int(level.get("priority", 0)) >= int(by_price[key].get("priority", 0)):
            by_price[key] = level
    return [by_price[key] for key in order]


def position_price_levels(
    *,
    entry_price: float | None,
    current_price: float | None,
    settings: dict[str, Any],
    exit_details: dict[str, Any],
) -> list[dict[str, Any]]:
    risk_distance = _number(settings.get("entry_stop_distance"))
    highest_price = _number(exit_details.get("highest_high_since_entry"))
    trigger = _number(exit_details.get("price_trigger_price"))
    if trigger is None:
        trigger = _number(exit_details.get("trigger_price"))
    trigger_source = str(
        exit_details.get("price_trigger_source")
        or exit_details.get("trigger_source")
        or "sell rule"
    ).strip()
    levels: list[dict[str, Any] | None] = [
        _level("Current quote", current_price, CHART_COLORS["price"], "dot", 1.2, 30),
        _level("Average entry", entry_price, CHART_COLORS["entry"], "solid", 1.4, 40),
        _level("Fill-adjusted initial stop", exit_details.get("original_stop_price"), CHART_COLORS["sell"], "dot", 1.3, 50),
        _level("Strategy exit", exit_details.get("strategy_exit_price"), CHART_COLORS["exit"], "dash", 1.5, 55),
        _level("Break-even stop", exit_details.get("breakeven_stop_price"), CHART_COLORS["atr"], "dash", 1.4, 60),
        _level("ATR trailing stop", exit_details.get("trailing_stop_price"), "#B388FF", "dash", 1.5, 65),
        _level("Highest price since entry", highest_price, "#8293A6", "dot", 1.0, 10),
    ]
    if entry_price is not None and risk_distance is not None and risk_distance > 0:
        levels.extend([
            _level("+1R profit level", entry_price + risk_distance, "#F4B942", "dot", 1.2, 15),
            _level("+2R profit level", entry_price + 2 * risk_distance, "#B388FF", "dashdot", 1.3, 15),
        ])
    levels.append(
        _level(
            f"ACTIVE PRICE STOP - {trigger_source}",
            trigger,
            CHART_COLORS["sell"],
            "solid",
            2.4,
            100,
        )
    )
    return _deduplicate_levels(levels)


def queued_price_levels(
    *,
    current_price: float | None,
    settings: dict[str, Any],
    live: dict[str, Any],
    next_buy_level: float | None,
) -> list[dict[str, Any]]:
    current_atr = _number(live.get("last_atr"))
    atr_multiplier = _number(settings.get("atr_stop_multiplier"))
    projected_stop = (
        current_price - current_atr * atr_multiplier
        if current_price is not None and current_atr is not None and atr_multiplier is not None
        else None
    )
    buy_label = "Next numeric BUY level"
    breakout_level = _number(live.get("trendline_breakout_level"))
    trend_filter_level = _number(live.get("trend_filter_level"))
    if next_buy_level is not None and breakout_level is not None and abs(next_buy_level - breakout_level) < 0.01:
        buy_label = "Required completed-close breakout"
    elif next_buy_level is not None and trend_filter_level is not None and abs(next_buy_level - trend_filter_level) < 0.01:
        buy_label = "Required trend filter level"
    return _deduplicate_levels([
        _level("Current quote", current_price, CHART_COLORS["price"], "dot", 1.2, 30),
        _level(buy_label, next_buy_level, CHART_COLORS["entry"], "solid", 2.1, 90),
        _level("ATR-only stop reference if bought now", projected_stop, CHART_COLORS["sell"], "dot", 1.2, 20),
    ])


def position_level_explanation(exit_details: dict[str, Any]) -> str:
    trigger = _number(exit_details.get("price_trigger_price"))
    if trigger is None:
        trigger = _number(exit_details.get("trigger_price"))
    source = str(
        exit_details.get("price_trigger_source")
        or exit_details.get("trigger_source")
        or "sell rule"
    ).strip()
    decision_source = str(exit_details.get("trigger_source") or "").strip()
    non_price_ready = bool(
        exit_details.get("ready")
        and decision_source.lower() in {"rsi recovery exit", "maximum holding period"}
    )
    candidates = [
        ("fill-adjusted initial stop", _number(exit_details.get("original_stop_price"))),
        ("strategy exit", _number(exit_details.get("strategy_exit_price"))),
        ("break-even stop", _number(exit_details.get("breakeven_stop_price"))),
        ("ATR trailing stop", _number(exit_details.get("trailing_stop_price"))),
    ]
    active = [(label, price) for label, price in candidates if price is not None]
    if trigger is None:
        if non_price_ready:
            return (
                f"A non-price exit is currently ready because of the {decision_source}. "
                "This position does not currently have an active dollar price stop."
            )
        return "No active sell price has been calculated."
    comparison = "; ".join(f"{label} ${price:,.2f}" for label, price in active)
    price_text = (
            f"Current automatic price stop: ${trigger:,.2f}, controlled by the {source}. "
        f"For a long position, the highest active protection level controls. Current levels: {comparison}."
        if comparison
        else f"Current automatic price stop: ${trigger:,.2f}, controlled by the {source}."
    )
    if non_price_ready:
        return (
            f"A non-price exit is currently ready because of the {decision_source}. "
            f"Separately, {price_text[0].lower() + price_text[1:]}"
        )
    return price_text


def _rolling(values: pd.Series, window: int, *, operation: str, shift: int = 0) -> pd.Series:
    rolled = values.rolling(max(1, int(window)), min_periods=max(1, int(window)))
    result = rolled.max() if operation == "max" else rolled.min() if operation == "min" else rolled.mean()
    return result.shift(shift) if shift else result


def build_management_chart(
    market_data: pd.DataFrame,
    settings: dict[str, Any],
    result: dict[str, Any],
    *,
    title: str,
    static_levels: list[dict[str, Any]],
    entry_time: Any = None,
    entry_price: float | None = None,
    max_bars: int = 90,
    height: int = 500,
) -> go.Figure:
    """Plot the exact saved strategy context without running any new decision logic."""
    if market_data is None or market_data.empty or "Close" not in market_data:
        raise ValueError("No completed price bars are available for this chart.")
    total = min(len(market_data), len(result.get("prices", [])))
    if total <= 0:
        raise ValueError("The saved strategy did not return chartable prices.")
    frame = market_data.iloc[-total:].copy()
    start = max(0, total - max(40, int(max_bars)))
    shown = frame.iloc[start:]
    x = shown.index
    close = frame["Close"].astype(float)
    high = frame["High"].astype(float) if "High" in frame else close
    low = frame["Low"].astype(float) if "Low" in frame else close
    open_prices = frame["Open"].astype(float) if "Open" in frame else close
    live = dict(result.get("live") or {})
    strategy_type = str(settings.get("strategy_type", "breakout"))

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x,
        open=open_prices.iloc[start:],
        high=high.iloc[start:],
        low=low.iloc[start:],
        close=close.iloc[start:],
        name="Completed price bars",
        increasing_line_color=CHART_COLORS["entry"],
        decreasing_line_color=CHART_COLORS["sell"],
        increasing_fillcolor=CHART_COLORS["entry"],
        decreasing_fillcolor=CHART_COLORS["sell"],
    ))

    smas = list(result.get("smas") or [])[-total:]
    if strategy_type in {"breakout", "pullback", "trendline", "trendline_retest"} and len(smas) == total:
        fig.add_trace(go.Scatter(
            x=x,
            y=smas[start:],
            name=f"{int(settings.get('moving_average_window', 50))}-bar trend filter",
            mode="lines",
            line=dict(color=CHART_COLORS["trend"], width=1.2, dash="dot"),
        ))

    entry_window = int(settings.get("entry_window", 20))
    exit_window = int(settings.get("exit_window", 10))
    if strategy_type == "breakout":
        overlays = [
            (f"{entry_window}-bar BUY breakout", _rolling(high, entry_window, operation="max", shift=1), CHART_COLORS["entry"]),
            (f"{exit_window}-bar strategy exit", _rolling(low, exit_window, operation="min", shift=1), CHART_COLORS["exit"]),
        ]
    elif strategy_type == "pullback":
        pullback_window = int(settings.get("pullback_average_length", 20))
        momentum_window = int(settings.get("momentum_turn_length", 10))
        overlays = [
            (f"{pullback_window}-bar pullback average", _rolling(close, pullback_window, operation="mean"), CHART_COLORS["entry"]),
            (f"{momentum_window}-bar momentum average", _rolling(close, momentum_window, operation="mean"), CHART_COLORS["atr"]),
            (f"{exit_window}-bar strategy exit", _rolling(close, exit_window, operation="mean"), CHART_COLORS["exit"]),
        ]
    elif strategy_type in {"trendline", "trendline_retest"}:
        overlays = [
            (f"{exit_window}-bar strategy exit", _rolling(low, exit_window, operation="min", shift=1), CHART_COLORS["exit"]),
        ]
        trendline_level = _number(live.get("trendline_level"))
        trendline_slope = _number(live.get("trendline_slope"))
        if trendline_level is not None and trendline_slope is not None:
            line_values = pd.Series(np.nan, index=frame.index, dtype="float64")
            anchors = [int(value) for value in live.get("trendline_anchor_indices", []) if value is not None]
            touches = [int(value) for value in live.get("trendline_touch_indices", []) if value is not None]
            line_start = max(0, anchors[0] if anchors else total - entry_window)
            for index in range(line_start, total):
                line_values.iloc[index] = trendline_level + trendline_slope * (index - (total - 1))
            fig.add_trace(go.Scatter(
                x=x,
                y=line_values.iloc[start:],
                name="Selected descending trendline",
                mode="lines",
                line=dict(color=CHART_COLORS["entry"], width=1.8, dash="solid"),
            ))
            result_atrs = list(result.get("atrs") or [])[-total:]
            tolerance_atr = _number(live.get("trendline_tolerance_atr")) or 0.25
            if len(result_atrs) == total:
                tolerance = pd.Series(
                    [
                        float(value) * tolerance_atr if value is not None and np.isfinite(value) else np.nan
                        for value in result_atrs
                    ],
                    index=frame.index,
                    dtype="float64",
                )
                lower = line_values - tolerance
                upper = line_values + tolerance
                fig.add_trace(go.Scatter(
                    x=x, y=lower.iloc[start:], mode="lines", showlegend=False,
                    hoverinfo="skip", line=dict(width=0),
                ))
                fig.add_trace(go.Scatter(
                    x=x, y=upper.iloc[start:], name="Allowed wick tolerance (0.25 ATR)",
                    mode="lines", hoverinfo="skip", line=dict(width=0),
                    fill="tonexty", fillcolor="rgba(57, 208, 122, 0.10)",
                ))
                breakout_buffer_atr = _number(live.get("trendline_breakout_buffer_atr")) or 0.10
                breakout_values = line_values + tolerance * (breakout_buffer_atr / tolerance_atr)
                fig.add_trace(go.Scatter(
                    x=x,
                    y=breakout_values.iloc[start:],
                    name="Required completed-close breakout (line + 0.10 ATR)",
                    mode="lines",
                    line=dict(color="#72A7FF", width=1.3, dash="dot"),
                ))
            for marker_name, indices, symbol, color in (
                ("Trendline anchors", anchors, "diamond", "#F4B942"),
                ("Additional confirming touches", touches, "circle-open", "#B388FF"),
            ):
                visible_indices = [value for value in indices if start <= value < total]
                if visible_indices:
                    fig.add_trace(go.Scatter(
                        x=[frame.index[value] for value in visible_indices],
                        y=[float(high.iloc[value]) for value in visible_indices],
                        name=marker_name,
                        mode="markers",
                        marker=dict(symbol=symbol, size=9, color=color, line=dict(width=1)),
                    ))
        momentum_level = int(settings.get("momentum_turn_length", 10))
        if strategy_type == "trendline_retest":
            overlays.append((f"{momentum_level}-bar momentum average", _rolling(close, momentum_level, operation="mean"), CHART_COLORS["atr"]))
    else:
        overlays = []

    for name, values, color in overlays:
        fig.add_trace(go.Scatter(
            x=x,
            y=values.iloc[start:],
            name=name,
            mode="lines",
            line=dict(color=color, width=1.2, dash="dash"),
        ))

    if strategy_type == "rsi_scalp":
        rsi_length = int(settings.get("rsi_length", 14))
        rsi = calc_rsi(close.to_numpy(dtype=float), rsi_length)
        fig.add_trace(go.Scatter(
            x=x,
            y=rsi[start:],
            name=f"RSI ({rsi_length} bars)",
            mode="lines",
            line=dict(color=CHART_COLORS["atr"], width=1.4),
            yaxis="y2",
        ))
        for label, value, color in (
            ("RSI setup level", settings.get("rsi_oversold", 30), CHART_COLORS["entry"]),
            ("RSI maximum sell level", settings.get("rsi_overbought", 70), CHART_COLORS["sell"]),
            ("Current setup low", live.get("rsi_setup_low"), "#8293A6"),
            (
                "Minimum RSI rebound for BUY",
                (
                    _number(live.get("rsi_setup_low")) + _number(live.get("required_rsi_rebound_points"))
                    if _number(live.get("rsi_setup_low")) is not None
                    and _number(live.get("required_rsi_rebound_points")) is not None
                    else None
                ),
                CHART_COLORS["entry"],
            ),
            ("Saved RSI sell trigger", settings.get("_position_rsi_sell_level"), CHART_COLORS["sell"]),
            ("Current RSI", settings.get("_position_current_rsi"), CHART_COLORS["atr"]),
        ):
            number = _number(value)
            if number is not None:
                fig.add_trace(go.Scatter(
                    x=[x[0], x[-1]], y=[number, number], name=f"{label} ({number:g})",
                    mode="lines", line=dict(color=color, width=1.0, dash="dot"), yaxis="y2",
                ))

    for level in static_levels:
        value = float(level["value"])
        fig.add_trace(go.Scatter(
            x=[x[0], x[-1]],
            y=[value, value],
            name=f"{level['name']} (${value:,.2f})",
            mode="lines",
            line=dict(color=level["color"], width=level["width"], dash=level["dash"]),
        ))

    if entry_time and entry_price is not None:
        try:
            marker_time = pd.Timestamp(entry_time)
            first_time = pd.Timestamp(x[0])
            last_time = pd.Timestamp(x[-1])
            if marker_time.tzinfo is not None and first_time.tzinfo is None:
                marker_time = marker_time.tz_localize(None)
            elif marker_time.tzinfo is None and first_time.tzinfo is not None:
                marker_time = marker_time.tz_localize(first_time.tzinfo)
            elif marker_time.tzinfo is not None and first_time.tzinfo is not None:
                marker_time = marker_time.tz_convert(first_time.tzinfo)
            if first_time <= marker_time <= last_time:
                fig.add_trace(go.Scatter(
                    x=[marker_time], y=[entry_price], name="Position entry",
                    mode="markers", marker=dict(symbol="triangle-up", size=11, color=CHART_COLORS["entry"]),
                ))
        except (TypeError, ValueError, OverflowError):
            pass

    fig.update_layout(
        # Streamlit/Plotly can render a null title as the JavaScript text "undefined".
        # The visible title is rendered immediately above the chart by Streamlit.
        title=dict(text=""),
        height=max(360, min(int(height), 1200)),
        margin=dict(l=16, r=16, t=66, b=32),
        font=dict(color=CHART_COLORS["text"], family="Inter, Segoe UI, sans-serif", size=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=9)),
        xaxis=dict(
            showgrid=False,
            type="category",
            nticks=10,
            rangeslider=dict(visible=False),
            linecolor=CHART_COLORS["border"],
        ),
        yaxis=dict(tickprefix="$", gridcolor=CHART_COLORS["grid"], zeroline=False),
        yaxis2=dict(
            range=[0, 100], overlaying="y", side="right", showgrid=False, zeroline=False,
            visible=strategy_type == "rsi_scalp", title="RSI" if strategy_type == "rsi_scalp" else None,
        ),
        plot_bgcolor=CHART_COLORS["surface"],
        paper_bgcolor=CHART_COLORS["surface"],
        hovermode="x unified",
    )
    return fig
