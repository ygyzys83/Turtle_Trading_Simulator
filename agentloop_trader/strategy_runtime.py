from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from agentloop_trader.backtest import (
    simulate_trendline_breakout_strategy,
    simulate_trendline_retest_strategy,
    simulate_trend_pullback_strategy,
    simulate_turtle_strategy,
)
from agentloop_trader.models import RiskLimits, TradeIntent


STRATEGY_TYPES = {
    "Breakout continuation": "breakout",
    "Trend pullback continuation": "pullback",
    "Trendline breakout": "trendline",
    "Trendline retest continuation": "trendline_retest",
}


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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
    if strategy_type == "pullback":
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


def saved_exit_settings_for_symbol(symbol: str, tracked_orders: list[dict]) -> dict[str, Any] | None:
    clean = str(symbol).strip().upper()
    matches = [
        row for row in tracked_orders
        if str(row.get("symbol", "")).strip().upper() == clean
        and str(row.get("side", "")).strip().lower() == "buy"
        and (row.get("exit_settings") or row.get("strategy_settings"))
    ]
    if not matches:
        return None
    def priority(row: dict[str, Any]) -> int:
        status = str(row.get("status", row.get("Status", ""))).strip().lower().rsplit(".", 1)[-1]
        source = str(row.get("source", "")).strip().lower()
        if source == "position_exit_settings" or status == "managed_exit_settings":
            return 3
        if status in {"filled", "partially_filled"}:
            return 2
        if source == "adopted_alpaca_position":
            return 1
        return 0

    latest = max(enumerate(matches), key=lambda item: (priority(item[1]), item[0]))[1]
    settings = dict(latest.get("exit_settings") or latest.get("strategy_settings") or {})
    settings.setdefault("entry_submitted_at", latest.get("submitted_at", ""))
    settings.setdefault("entry_filled_at", latest.get("filled_at", ""))
    settings.setdefault("entry_broker_order_id", latest.get("broker_order_id", ""))
    return settings


def update_exit_settings_for_symbol(symbol: str, tracked_orders: list[dict], exit_settings: dict[str, Any]) -> list[dict]:
    clean = str(symbol).strip().upper()
    updated: list[dict] = []
    matched = False
    for row in tracked_orders:
        record = dict(row)
        if str(record.get("symbol", "")).strip().upper() == clean and str(record.get("side", "")).lower() == "buy":
            record["exit_settings"] = dict(exit_settings)
            matched = True
        updated.append(record)
    if not matched:
        updated.append({
            "broker_order_id": f"managed-exit-{clean}",
            "symbol": clean,
            "side": "buy",
            "status": "managed_exit_settings",
            "source": "position_exit_settings",
            "exit_settings": dict(exit_settings),
        })
    return updated


def _parse_time(value: Any) -> pd.Timestamp | None:
    try:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("America/Los_Angeles")
        return timestamp
    except Exception:
        return None


def evaluate_exit_settings(
    settings: dict[str, Any] | None,
    position: dict[str, Any],
    fetch_bars: Callable[[str, str, str, str], pd.DataFrame],
) -> dict[str, Any]:
    if not settings:
        return {"ready": False, "reason": "No saved exit settings; automatic exit is paused.", "trigger_price": None}
    if not bool(settings.get("auto_exit_enabled", True)):
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
        current_price = _number(data.attrs.get("latest_price"), _number(live.get("last_p")))
        strategy_exit = _number(live.get("exit_level"))
        current_atr = _number(live.get("last_atr"))
        entry = _number(position.get("Average Entry"), _number(settings.get("entry_reference_price"), current_price))
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
        profit_r = (current_price - entry) / initial_risk if current_price is not None and entry is not None and initial_risk else None

        high_data = data.tail(1)
        entry_time = _parse_time(settings.get("entry_filled_at") or settings.get("entry_submitted_at"))
        if entry_time is not None and not data.empty:
            compare = entry_time.tz_convert(data.index.tz) if getattr(data.index, "tz", None) else entry_time.tz_localize(None)
            recent = data.loc[data.index >= compare]
            if not recent.empty:
                high_data = recent
        current_high = _number(high_data["High"].max()) if "High" in high_data.columns else None
        current_high = max(
            [value for value in (current_high, _number(data.attrs.get("latest_high"))) if value is not None],
            default=None,
        )
        saved_high = _number(settings.get("highest_high_since_entry"))
        high = max([value for value in (current_high, saved_high, entry) if value is not None], default=None)
        highest_profit_r = (high - entry) / initial_risk if high is not None and entry is not None and initial_risk else None
        protect = bool(settings.get("profit_protection_enabled", True))
        breakeven_after = float(settings.get("breakeven_after_r", 1.0))
        trail_after = float(settings.get("trail_after_r", 2.0))
        trail_mult = float(settings.get("trailing_atr_multiplier", 3.0))
        breakeven = entry if protect and highest_profit_r is not None and highest_profit_r >= breakeven_after else None
        atr_trail = high - trail_mult * current_atr if protect and highest_profit_r is not None and highest_profit_r >= trail_after and high is not None and current_atr is not None else None
        saved_trigger = _number(settings.get("last_exit_trigger_price"))
        candidates = [("strategy exit", strategy_exit), ("original stop", original_stop), ("break-even stop", breakeven), ("ATR trail", atr_trail), ("saved trigger", saved_trigger)]
        usable = [(name, value) for name, value in candidates if value is not None]
        source_name, trigger = max(usable, key=lambda item: item[1]) if usable else ("exit rule", None)
        ready = bool(current_price is not None and trigger is not None and current_price <= trigger)
        reason = (
            f"Exit now because {symbol} is at or below the {source_name} at ${trigger:,.2f}."
            if ready and trigger is not None
            else f"Hold. Automatic exit triggers at ${trigger:,.2f} or lower using the highest active protection level."
            if trigger is not None
            else "No exit trigger is available."
        )
        return {
            "ready": ready,
            "reason": reason,
            "current_price": current_price,
            "trigger_price": trigger,
            "trigger_source": source_name,
            "strategy_exit_price": strategy_exit,
            "original_stop_price": original_stop,
            "breakeven_stop_price": breakeven,
            "trailing_stop_price": atr_trail,
            "highest_high_since_entry": high,
            "profit_r": profit_r,
            "highest_profit_r": highest_profit_r,
            "current_atr": current_atr,
            "interval": interval,
            "checked_at": datetime.now().astimezone().isoformat(),
            "state_changed": bool((high or 0) > (saved_high or 0) or (trigger or 0) > (saved_trigger or 0)),
        }
    except Exception as exc:
        return {"ready": False, "reason": f"Could not check saved exit rule: {exc}", "trigger_price": None}
