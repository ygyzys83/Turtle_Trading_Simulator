from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from agentloop_trader.assets import normalize_asset_class
from agentloop_trader.backtest import (
    simulate_rsi_mean_reversion_strategy,
    simulate_trendline_breakout_strategy,
    simulate_trendline_retest_strategy,
    simulate_trend_pullback_strategy,
    simulate_turtle_strategy,
)
from agentloop_trader.fees import fee_adjusted_break_even_price
from agentloop_trader.indicators import calc_atr, calc_rsi
from agentloop_trader.models import RiskLimits, TradeIntent
from agentloop_trader.strategy_levels import build_buy_level_snapshot


STRATEGY_TYPES = {
    "Breakout continuation": "breakout",
    "Trend pullback continuation": "pullback",
    "Trendline breakout": "trendline",
    "Trendline retest continuation": "trendline_retest",
    "RSI mean-reversion scalp": "rsi_scalp",
}


def normalize_managed_exit_settings(
    settings: dict[str, Any] | None,
    position: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep broker-managed exits independent from the research screen's data source."""
    normalized = dict(settings or {})
    if not normalized:
        return normalized
    position = position or {}
    symbol = str(position.get("Symbol") or normalized.get("symbol") or "").strip().upper()
    asset_class = normalize_asset_class(
        position.get("Asset Type") or normalized.get("asset_class"),
        symbol,
    )
    normalized["asset_class"] = asset_class
    normalized["price_data_source"] = (
        "Crypto (Alpaca)" if asset_class == "crypto" else "Ticker (Alpaca)"
    )
    return normalized

EXIT_SNAPSHOT_FIELDS = (
    "ready",
    "reason",
    "current_price",
    "current_price_source",
    "trigger_price",
    "trigger_source",
    "price_trigger_price",
    "price_trigger_source",
    "exit_mode",
    "strategy_exit_price",
    "original_stop_price",
    "breakeven_stop_price",
    "trailing_stop_price",
    "highest_high_since_entry",
    "profit_r",
    "highest_profit_r",
    "current_atr",
    "current_rsi",
    "highest_rsi_since_entry",
    "rsi_sell_level",
    "rsi_exit_signal_ready",
    "rsi_profit_only_exit",
    "rsi_fee_adjusted_break_even",
    "rsi_completed_bar_price",
    "rsi_profit_condition_ready",
    "rsi_exit_ready",
    "bars_since_entry",
    "max_holding_enabled",
    "max_holding_bars",
    "time_exit_ready",
    "interval",
    "last_completed_bar_at",
    "exit_window",
    "atr_multiplier",
    "trailing_atr_multiplier",
    "breakeven_after_r",
    "trail_after_r",
    "buy_level_snapshot",
    "checked_at",
)

EXIT_PLAN_HISTORY_BY_INTERVAL = {
    "1m": "7d",
    "5m": "1mo",
    "15m": "3mo",
    "30m": "3mo",
    "1h": "1y",
    "4h": "2y",
    "1d": "5y",
}


def exit_plan_history_for_interval(interval: str) -> str:
    """Return enough history to calculate position exits without excessive downloads."""
    return EXIT_PLAN_HISTORY_BY_INTERVAL.get(str(interval), "1y")


def latest_atr_snapshot(market_data: pd.DataFrame, length: int = 14) -> tuple[float, str]:
    """Return the latest valid ATR and the completed bar that produced it."""
    required = {"Close", "High", "Low"}
    missing = required.difference(market_data.columns)
    if missing:
        raise ValueError(f"Price data is missing: {', '.join(sorted(missing))}.")
    values = calc_atr(
        market_data["Close"].to_numpy(),
        length,
        highs=market_data["High"].to_numpy(),
        lows=market_data["Low"].to_numpy(),
    )
    for index in range(len(values) - 1, -1, -1):
        value = values[index]
        if value is not None and pd.notna(value) and float(value) > 0:
            timestamp = market_data.index[index]
            measured_at = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
            return float(value), measured_at
    raise ValueError(f"Not enough completed bars were available to calculate {length}-bar ATR.")


def exit_mode_for_settings(settings: dict[str, Any] | None) -> str:
    settings = settings or {}
    explicit = str(settings.get("exit_mode") or "").strip()
    if explicit:
        return explicit
    return "atr_only" if str(settings.get("entry_source") or "").strip().lower() == "manual order" else "strategy_and_atr"


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _position_market_price(position: dict[str, Any]) -> float | None:
    direct = _number(position.get("Current Price"), _number(position.get("current_price")))
    if direct is not None and direct > 0:
        return direct
    quantity = _number(position.get("Quantity"))
    market_value = _number(position.get("Market Value"))
    if quantity is None or market_value is None or quantity == 0:
        return None
    derived = market_value / quantity
    return derived if derived > 0 else None


def exit_snapshot_from_details(details: dict[str, Any]) -> dict[str, Any]:
    """Keep the worker's latest exit calculation small and JSON-serializable."""
    return {
        field: details.get(field)
        for field in EXIT_SNAPSHOT_FIELDS
        if field in details
    }


def exit_details_from_snapshot(
    settings: dict[str, Any] | None,
    position: dict[str, Any],
) -> dict[str, Any]:
    """Render a position from the worker snapshot without downloading price history."""
    if not settings:
        return {
            "ready": False,
            "reason": "No saved exit settings for this position; automatic exit is paused.",
            "trigger_price": None,
            "snapshot_available": False,
        }
    if not bool(settings.get("auto_exit_enabled", False)):
        return {
            "ready": False,
            "reason": "Automatic exit is off for this position.",
            "trigger_price": None,
            "snapshot_available": bool(settings.get("last_exit_snapshot")),
        }

    details = dict(settings.get("last_exit_snapshot") or {})
    snapshot_available = bool(details)
    symbol = str(position.get("Symbol") or settings.get("symbol") or "").strip().upper()
    live_price = _position_market_price(position)
    entry = _number(position.get("Average Entry"), _number(settings.get("entry_reference_price")))
    initial_risk = _number(settings.get("entry_stop_distance"))
    original_stop = entry - initial_risk if entry is not None and initial_risk and initial_risk > 0 else None
    saved_trigger = _number(settings.get("last_exit_trigger_price"))
    saved_trigger_source = str(settings.get("last_exit_trigger_source") or "").strip()
    fallback_trigger = saved_trigger if saved_trigger_source else original_stop
    trigger = _number(
        details.get("trigger_price"),
        fallback_trigger,
    )
    trigger_source = str(
        details.get("trigger_source")
        or saved_trigger_source
        or ("fill-adjusted initial stop" if original_stop is not None else "exit rule")
    )
    normalized_source = trigger_source.strip().lower()
    non_price_trigger = normalized_source in {"rsi recovery exit", "maximum holding period"}
    price_trigger = _number(details.get("price_trigger_price"), trigger)
    price_trigger_source = str(details.get("price_trigger_source") or "").strip()
    if not price_trigger_source and not non_price_trigger:
        price_trigger_source = trigger_source
    if not price_trigger_source and price_trigger is not None:
        price_candidates = [
            ("strategy exit", _number(details.get("strategy_exit_price"))),
            ("fill-adjusted initial stop", _number(details.get("original_stop_price"), original_stop)),
            ("break-even stop", _number(details.get("breakeven_stop_price"))),
            ("ATR trail", _number(details.get("trailing_stop_price"))),
        ]
        matching = [
            label
            for label, value in price_candidates
            if value is not None and abs(value - price_trigger) <= max(0.000001, abs(price_trigger) * 0.00000001)
        ]
        price_trigger_source = matching[0] if matching else "saved price protection"
    if trigger is not None:
        if normalized_source == "strategy exit" and details.get("strategy_exit_price") is None:
            details["strategy_exit_price"] = trigger
        elif normalized_source == "break-even stop" and details.get("breakeven_stop_price") is None:
            details["breakeven_stop_price"] = trigger
        elif normalized_source == "atr trail" and details.get("trailing_stop_price") is None:
            details["trailing_stop_price"] = trigger
    snapshot_ready = bool(details.get("ready", False))
    price_ready = bool(live_price is not None and trigger is not None and live_price <= trigger)
    ready = snapshot_ready if non_price_trigger else price_ready

    details.update({
        "ready": ready,
        "current_price": live_price if live_price is not None else _number(details.get("current_price")),
        "current_price_source": (
            "Alpaca position market value"
            if live_price is not None
            else details.get("current_price_source", "saved worker calculation")
        ),
        "trigger_price": trigger,
        "trigger_source": trigger_source,
        "price_trigger_price": price_trigger,
        "price_trigger_source": price_trigger_source,
        "exit_mode": details.get("exit_mode") or exit_mode_for_settings(settings),
        "original_stop_price": _number(details.get("original_stop_price"), original_stop),
        "interval": details.get("interval") or str(settings.get("interval", "1h")),
        "exit_window": details.get("exit_window") or int(settings.get("exit_window", 10)),
        "snapshot_available": snapshot_available,
        "snapshot_checked_at": details.get("checked_at") or settings.get("last_exit_checked_at"),
    })
    if ready and not non_price_trigger and trigger is not None:
        details["reason"] = f"Exit now because {symbol} is at or below the {trigger_source} at ${trigger:,.2f}."
    elif not snapshot_available:
        details["reason"] = (
            f"Hold. The saved protective stop is ${trigger:,.2f}. "
            "Start the Background Worker or refresh this position to calculate the complete exit plan."
            if trigger is not None
            else "Start the Background Worker or refresh this position to calculate its exit plan."
        )
    elif not ready and trigger is not None and not non_price_trigger:
        details["reason"] = (
            f"Hold. Automatic exit triggers at ${trigger:,.2f} or lower using the latest saved worker calculation."
        )
    return details


def _run_one(
    strategy_type: str,
    market_data: pd.DataFrame,
    settings: dict[str, Any],
    account_equity: float,
    risk_limits: RiskLimits | None,
) -> dict[str, Any]:
    entry_w = int(settings.get("entry_window", 20))
    exit_w = int(settings.get("exit_window", 10))
    atr_mult = float(settings.get("atr_stop_multiplier", 2.0))
    risk_dec = float(settings.get("risk_per_trade_pct", 1.0)) / 100
    ma_w = int(settings.get("moving_average_window", 50))
    pullback_w = int(settings.get("pullback_average_length", 20))
    momentum_w = int(settings.get("momentum_turn_length", 10))
    rsi_entry_filter_enabled = bool(settings.get("rsi_entry_filter_enabled", False))
    common = {
        "account": account_equity,
        "exit_w": exit_w,
        "atr_mult": atr_mult,
        "risk_pct_dec": risk_dec,
        "market_data": market_data,
        "risk_limits": risk_limits,
        "rsi_entry_filter_enabled": rsi_entry_filter_enabled,
    }
    if strategy_type == "rsi_scalp":
        result = simulate_rsi_mean_reversion_strategy(
            account=account_equity,
            atr_mult=atr_mult,
            risk_pct_dec=risk_dec,
            rsi_length=int(settings.get("rsi_length", 14)),
            rsi_oversold=float(settings.get("rsi_oversold", 30.0)),
            rsi_overbought=float(settings.get("rsi_overbought", 70.0)),
            rsi_decline_points=float(settings.get("rsi_decline_points", 40.0)),
            rsi_rebound_points=float(settings.get("rsi_rebound_points", 3.0)),
            rsi_max_rebound_points=float(settings.get("rsi_max_rebound_points", 12.0)),
            rsi_sell_recovery_points=float(settings.get("rsi_sell_recovery_points", 35.0)),
            rsi_swing_lookback=int(settings.get("rsi_swing_lookback", 24)),
            rsi_stop_mode=str(settings.get("rsi_stop_mode", "standard_atr")),
            rsi_emergency_atr_multiplier=float(settings.get("rsi_emergency_atr_multiplier", 5.0)),
            rsi_max_holding_enabled=bool(settings.get("rsi_max_holding_enabled", True)),
            rsi_max_holding_bars=int(settings.get("rsi_max_holding_bars", 100)),
            rsi_profit_only_exit=bool(settings.get("rsi_profit_only_exit", False)),
            market_data=market_data,
            risk_limits=risk_limits,
        )
    elif strategy_type == "pullback":
        result = simulate_trend_pullback_strategy(pullback_w=pullback_w, trend_w=ma_w, momentum_w=momentum_w, **common)
    elif strategy_type == "trendline":
        result = simulate_trendline_breakout_strategy(trendline_w=entry_w, ma_w=ma_w, **common)
    elif strategy_type == "trendline_retest":
        result = simulate_trendline_retest_strategy(trendline_w=entry_w, ma_w=ma_w, momentum_w=momentum_w, **common)
    else:
        result = simulate_turtle_strategy(entry_w=entry_w, ma_w=ma_w, **common)
    prices, smas, atrs, trades, live, stats, labels = result
    return {"prices": prices, "smas": smas, "atrs": atrs, "trade_log": trades, "live": live, "stats": stats, "labels": labels}


def run_strategy_suite(
    market_data: pd.DataFrame,
    settings: dict[str, Any],
    account_equity: float,
    risk_limits: RiskLimits | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        label: _run_one(strategy_type, market_data, settings, account_equity, risk_limits)
        for label, strategy_type in STRATEGY_TYPES.items()
    }


def selected_strategy_result(
    market_data: pd.DataFrame,
    settings: dict[str, Any],
    account_equity: float,
    risk_limits: RiskLimits | None = None,
) -> dict[str, Any]:
    strategy_type = str(settings.get("strategy_type", "trendline_retest"))
    return _run_one(strategy_type, market_data, settings, account_equity, risk_limits)


def trade_intent_to_record(intent: TradeIntent | None) -> dict[str, Any] | None:
    return asdict(intent) if intent is not None else None


def trade_intent_from_record(record: dict[str, Any] | None) -> TradeIntent | None:
    return TradeIntent(**record) if record else None


def apply_buy_order_style(
    intent: TradeIntent | None,
    style: str,
    adjustment_pct: float = 0.0,
    custom_limit_price: float = 0.0,
) -> TradeIntent | None:
    if intent is None or intent.entry_price is None:
        return intent
    style_lower = str(style).lower()
    if style_lower == "market":
        return replace(intent, order_type="market", limit_price=None)
    if "custom" in style_lower:
        limit_price = float(custom_limit_price)
        if limit_price <= 0:
            return replace(intent, order_type="limit", limit_price=None)
    elif "below" in style_lower:
        limit_price = float(intent.entry_price) * (1 - float(adjustment_pct) / 100)
    elif "above" in style_lower:
        limit_price = float(intent.entry_price) * (1 + float(adjustment_pct) / 100)
    else:
        limit_price = float(intent.entry_price)
    limit_price = round(limit_price, 2)
    return replace(intent, order_type="limit", limit_price=limit_price, entry_price=limit_price)


def reprice_trade_intent(intent: TradeIntent | None, current_price: float | None) -> TradeIntent | None:
    """Move a proposed buy to the newest price while preserving its stop distance."""
    if intent is None or intent.side != "buy" or intent.entry_price is None or current_price is None or current_price <= 0:
        return intent
    stop_distance = (
        float(intent.entry_price) - float(intent.stop_loss)
        if intent.stop_loss is not None
        else None
    )
    stop_loss = round(float(current_price) - stop_distance, 2) if stop_distance is not None and stop_distance > 0 else intent.stop_loss
    return replace(intent, entry_price=round(float(current_price), 2), stop_loss=stop_loss)


def _parse_time(value: Any) -> pd.Timestamp | None:
    try:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("America/Los_Angeles")
        return timestamp
    except Exception:
        return None


def adjust_initial_stop_settings(settings: dict[str, Any], atr_multiplier: float) -> dict[str, Any]:
    """Return exit settings with initial risk rebuilt from the saved entry ATR."""
    updated = dict(settings)
    multiplier = float(atr_multiplier)
    if multiplier <= 0:
        raise ValueError("Initial stop ATR multiplier must be positive.")
    entry_atr = _number(updated.get("entry_atr"))
    previous_distance = _number(updated.get("entry_stop_distance"))
    previous_multiplier = _number(
        updated.get("entry_stop_atr_multiplier"),
        _number(updated.get("atr_stop_multiplier")),
    )
    if entry_atr is None and previous_distance is not None and previous_multiplier:
        entry_atr = previous_distance / previous_multiplier
    updated["atr_stop_multiplier"] = multiplier
    updated["entry_stop_atr_multiplier"] = multiplier
    if entry_atr is not None:
        distance = entry_atr * multiplier
        updated["entry_stop_distance"] = distance
        planned_entry = _number(updated.get("planned_entry_price"), _number(updated.get("entry_reference_price")))
        if planned_entry is not None:
            updated["entry_stop_loss"] = planned_entry - distance
    updated.pop("last_exit_trigger_price", None)
    updated.pop("last_exit_trigger_source", None)
    return updated


def evaluate_exit_settings(
    settings: dict[str, Any] | None,
    position: dict[str, Any],
    fetch_bars: Callable[[str, str, str, str], pd.DataFrame],
) -> dict[str, Any]:
    if not settings:
        return {"ready": False, "reason": "No saved exit settings; automatic exit is paused.", "trigger_price": None}
    if not bool(settings.get("auto_exit_enabled", False)):
        return {"ready": False, "reason": "Automatic exit is off for this position.", "trigger_price": None}
    symbol = str(position.get("Symbol") or settings.get("symbol") or "").strip().upper()
    try:
        history = str(settings.get("history", "1y"))
        interval = str(settings.get("interval", "1h"))
        source = str(settings.get("price_data_source", "Ticker (Alpaca)"))
        data = fetch_bars(symbol, history, interval, source)
        account = float(settings.get("account_size", 100000))
        result = selected_strategy_result(data, settings, account)
        live = result["live"]
        exit_mode = exit_mode_for_settings(settings)
        strategy_exit_enabled = exit_mode == "strategy_and_atr"
        position_market_price = _position_market_price(position)
        current_price = _number(
            position_market_price,
            _number(data.attrs.get("latest_price"), _number(live.get("last_p"))),
        )
        current_price_source = (
            "Alpaca position market value"
            if position_market_price is not None
            else "latest price data"
        )
        strategy_exit = _number(live.get("exit_level")) if strategy_exit_enabled else None
        current_atr = _number(live.get("last_atr"))
        current_rsi = _number(live.get("rsi"))
        entry = _number(position.get("Average Entry"), _number(settings.get("entry_reference_price"), current_price))
        rsi_strategy = strategy_exit_enabled and str(settings.get("strategy_type", "")) == "rsi_scalp"
        no_price_stop = rsi_strategy and str(settings.get("rsi_stop_mode", "standard_atr")) == "no_price_stop"
        atr_mult = float(settings.get("atr_stop_multiplier", 2.0))
        initial_risk = _number(settings.get("entry_stop_distance"))
        saved_entry = _number(settings.get("planned_entry_price"), _number(settings.get("entry_reference_price")))
        saved_stop = _number(settings.get("entry_stop_loss"))
        if initial_risk is None and saved_entry is not None and saved_stop is not None:
            initial_risk = saved_entry - saved_stop
        if initial_risk is not None and initial_risk <= 0:
            initial_risk = None
        original_stop = entry - initial_risk if entry is not None and initial_risk is not None else saved_stop
        if original_stop is None and entry is not None and current_atr is not None:
            original_stop = entry - atr_mult * current_atr
        if no_price_stop:
            initial_risk = None
            original_stop = None
        profit_r = (current_price - entry) / initial_risk if current_price is not None and entry is not None and initial_risk else None

        entry_time = _parse_time(settings.get("entry_filled_at") or settings.get("entry_submitted_at"))
        last_completed_bar_at = data.index[-1].isoformat() if not data.empty else None
        highest_rsi_since_entry = current_rsi
        if rsi_strategy and not data.empty and "Close" in data.columns:
            rsi_values = pd.Series(
                calc_rsi(data["Close"].to_numpy(), int(settings.get("rsi_length", 14))),
                index=data.index,
                dtype="float64",
            )
            if entry_time is not None:
                compare_rsi_time = (
                    entry_time.tz_convert(data.index.tz)
                    if getattr(data.index, "tz", None)
                    else entry_time.tz_localize(None)
                )
                rsi_values = rsi_values.loc[rsi_values.index >= compare_rsi_time]
            rsi_values = rsi_values.dropna()
            if not rsi_values.empty:
                highest_rsi_since_entry = float(rsi_values.max())
        high_data = data.tail(1) if entry_time is None else data.iloc[0:0]
        bars_since_entry = 0
        has_complete_post_entry_bar = False
        if entry_time is not None and not data.empty:
            compare = entry_time.tz_convert(data.index.tz) if getattr(data.index, "tz", None) else entry_time.tz_localize(None)
            recent = data.loc[data.index >= compare]
            if not recent.empty:
                high_data = recent
                bars_since_entry = max(0, len(recent) - 1)
                has_complete_post_entry_bar = True
        current_high_candidates = [value for value in (current_price, entry) if value is not None]
        if not high_data.empty and "High" in high_data.columns:
            current_high_candidates.append(_number(high_data["High"].max()))
        if entry_time is None or has_complete_post_entry_bar:
            current_high_candidates.append(_number(data.attrs.get("latest_high")))
        current_high = max([value for value in current_high_candidates if value is not None], default=None)
        saved_high = _number(settings.get("highest_high_since_entry"))
        entry_bar_is_still_incomplete = entry_time is not None and not has_complete_post_entry_bar
        high = (
            current_high
            if entry_bar_is_still_incomplete
            else max([value for value in (current_high, saved_high, entry) if value is not None], default=None)
        )
        highest_profit_r = (high - entry) / initial_risk if high is not None and entry is not None and initial_risk else None
        protect = bool(settings.get("profit_protection_enabled", True)) and not no_price_stop
        breakeven_after = float(settings.get("breakeven_after_r", 1.0))
        trail_after = float(settings.get("trail_after_r", 2.0))
        trail_mult = float(settings.get("trailing_atr_multiplier", 3.0))
        breakeven = entry if protect and highest_profit_r is not None and highest_profit_r >= breakeven_after else None
        atr_trail = high - trail_mult * current_atr if protect and highest_profit_r is not None and highest_profit_r >= trail_after and high is not None and current_atr is not None else None
        saved_trigger_source = str(settings.get("last_exit_trigger_source") or "").strip().lower()
        saved_trigger = None if no_price_stop else _number(settings.get("last_exit_trigger_price"))
        if saved_trigger_source == "break-even stop" and breakeven is None:
            saved_trigger = None
        elif saved_trigger_source == "atr trail" and atr_trail is None:
            saved_trigger = None
        elif saved_trigger_source == "strategy exit" and not strategy_exit_enabled:
            saved_trigger = None
        candidates = [
            ("strategy exit", strategy_exit),
            ("fill-adjusted initial stop", original_stop),
            ("break-even stop", breakeven),
            ("ATR trail", atr_trail),
            (saved_trigger_source or "saved price protection", saved_trigger),
        ]
        usable = [(name, value) for name, value in candidates if value is not None]
        source_name, trigger = max(usable, key=lambda item: item[1]) if usable else ("exit rule", None)
        price_trigger_source = source_name
        price_trigger = trigger
        saved_setup_low = _number(settings.get("entry_rsi_setup_low"))
        rsi_sell_level = (
            min(
                float(settings.get("rsi_overbought", 70.0)),
                saved_setup_low + float(settings.get("rsi_sell_recovery_points", 35.0)),
            )
            if rsi_strategy and saved_setup_low is not None
            else None
        )
        rsi_exit_signal_ready = bool(
            rsi_sell_level is not None and current_rsi is not None and current_rsi >= rsi_sell_level
        )
        rsi_profit_only_exit = bool(settings.get("rsi_profit_only_exit", False))
        rsi_completed_bar_price = _number(live.get("last_p"), current_price)
        quantity = _number(position.get("Quantity"), 0.0) or 0.0
        asset_class = normalize_asset_class(
            str(position.get("Asset Type") or settings.get("asset_class") or ""),
            symbol,
        )
        rsi_fee_adjusted_break_even = (
            fee_adjusted_break_even_price(
                asset_class=asset_class,
                quantity=quantity,
                entry_price=entry,
            )
            if rsi_strategy and entry is not None and quantity > 0
            else entry
        )
        rsi_profit_condition_ready = bool(
            not rsi_profit_only_exit
            or (
                rsi_completed_bar_price is not None
                and rsi_fee_adjusted_break_even is not None
                and rsi_completed_bar_price > rsi_fee_adjusted_break_even
            )
        )
        rsi_exit_ready = rsi_exit_signal_ready and rsi_profit_condition_ready
        max_holding_enabled = bool(settings.get("rsi_max_holding_enabled", True))
        max_holding_bars = int(settings.get("rsi_max_holding_bars", 100))
        time_exit_ready = bool(
            rsi_strategy and max_holding_enabled and entry_time is not None
            and bars_since_entry >= max_holding_bars
        )
        price_exit_ready = bool(current_price is not None and trigger is not None and current_price <= trigger)
        ready = price_exit_ready or rsi_exit_ready or time_exit_ready
        if rsi_exit_ready:
            source_name = "RSI recovery exit"
        elif time_exit_ready:
            source_name = "maximum holding period"
        reason = (
            f"Exit now because RSI is {current_rsi:.1f}, at or above the saved {rsi_sell_level:.1f} exit level."
            if rsi_exit_ready and current_rsi is not None and rsi_sell_level is not None
            else f"Exit now because the position reached its {max_holding_bars}-bar maximum holding period."
            if time_exit_ready
            else
            f"Exit now because {symbol} is at or below the {source_name} at ${trigger:,.2f}."
            if price_exit_ready and trigger is not None
            else (
                f"Hold. RSI reached {rsi_sell_level:.1f}, but the completed-bar close of ${rsi_completed_bar_price:,.2f} is not above the "
                f"estimated fee-adjusted break-even price of ${rsi_fee_adjusted_break_even:,.2f}. "
                "The app will check again after each completed price bar."
            )
            if (
                rsi_exit_signal_ready
                and not rsi_profit_condition_ready
                and rsi_completed_bar_price is not None
                and rsi_sell_level is not None
                and rsi_fee_adjusted_break_even is not None
            )
            else f"Hold. Automatic exit triggers at ${trigger:,.2f} or lower using the highest active protection level."
            if trigger is not None
            else "Hold. The saved RSI recovery and maximum holding-period exits are not triggered."
            if max_holding_enabled
            else "Hold. The saved RSI recovery exit and price protection are not triggered."
        )
        return {
            "ready": ready,
            "reason": reason,
            "current_price": current_price,
            "current_price_source": current_price_source,
            "trigger_price": trigger,
            "trigger_source": source_name,
            "price_trigger_price": price_trigger,
            "price_trigger_source": price_trigger_source,
            "exit_mode": exit_mode,
            "strategy_exit_price": strategy_exit,
            "original_stop_price": original_stop,
            "breakeven_stop_price": breakeven,
            "trailing_stop_price": atr_trail,
            "highest_high_since_entry": high,
            "profit_r": profit_r,
            "highest_profit_r": highest_profit_r,
            "current_atr": current_atr,
            "current_rsi": current_rsi,
            "highest_rsi_since_entry": highest_rsi_since_entry if rsi_strategy else None,
            "rsi_sell_level": rsi_sell_level,
            "rsi_exit_signal_ready": rsi_exit_signal_ready,
            "rsi_profit_only_exit": rsi_profit_only_exit if rsi_strategy else None,
            "rsi_fee_adjusted_break_even": rsi_fee_adjusted_break_even if rsi_strategy else None,
            "rsi_completed_bar_price": rsi_completed_bar_price if rsi_strategy else None,
            "rsi_profit_condition_ready": rsi_profit_condition_ready if rsi_strategy else None,
            "rsi_exit_ready": rsi_exit_ready,
            "bars_since_entry": bars_since_entry,
            "max_holding_enabled": max_holding_enabled if rsi_strategy else None,
            "max_holding_bars": max_holding_bars if rsi_strategy and max_holding_enabled else None,
            "time_exit_ready": time_exit_ready,
            "interval": interval,
            "last_completed_bar_at": last_completed_bar_at,
            "exit_window": int(settings.get("exit_window", 10)),
            "atr_multiplier": atr_mult,
            "trailing_atr_multiplier": trail_mult,
            "breakeven_after_r": breakeven_after,
            "trail_after_r": trail_after,
            "buy_level_snapshot": build_buy_level_snapshot(live, interval=interval, latest_price=current_price),
            "checked_at": datetime.now().astimezone().isoformat(),
            "state_changed": bool(
                abs((high or 0) - (saved_high or 0)) > 0.000001
                or abs((trigger or 0) - (_number(settings.get("last_exit_trigger_price")) or 0)) > 0.000001
            ),
        }
    except Exception as exc:
        return {"ready": False, "reason": f"Could not check saved exit rule: {exc}", "trigger_price": None}
