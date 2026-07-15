from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from agentloop_trader.assets import normalize_asset_class, normalize_symbol
from agentloop_trader.audit_store import JsonlAuditStore
from agentloop_trader.automation_runtime import (
    AutomationControl,
    AutomationControlStore,
    WorkerLock,
    WorkerStatus,
    WorkerStatusStore,
    worker_code_fingerprint,
)
from agentloop_trader.broker_governance import (
    BrokerStateStore,
    OPEN_ORDER_STATUSES,
    adopt_alpaca_position,
    build_exit_intent_from_position,
    market_session_advisory,
    open_exit_order_reasons,
    open_order_exposure_reasons,
    refresh_tracked_alpaca_orders,
)
from agentloop_trader.brokers import (
    AlpacaBrokerAdapterStub,
    AlpacaConfig,
    alpaca_tracked_order_from_broker_order,
    build_alpaca_cancel_preview,
    build_alpaca_order_preview,
)
from agentloop_trader.buy_watchlist import BuyWatchPlan, BuyWatchlistStore
from agentloop_trader.market_data import (
    fetch_alpaca_latest_crypto_trades,
    fetch_alpaca_latest_trades,
    fetch_price_bars,
)
from agentloop_trader.models import AuditEvent, ExecutionDecision, PACIFIC_TIME, RiskCheckResult, RiskLimits
from agentloop_trader.risk import check_trade_intent, constrain_trade_intent_to_limits
from agentloop_trader.strategy_runtime import (
    apply_buy_order_style,
    evaluate_exit_settings,
    reprice_trade_intent,
    selected_strategy_result,
    saved_exit_settings_for_symbol,
    update_exit_settings_for_symbol,
)
from agentloop_trader.strategy_levels import build_buy_level_snapshot


_BAR_CACHE: dict[tuple[str, str, str, str], tuple[float, Any]] = {}
_BAR_CACHE_SECONDS = {"1m": 30, "5m": 60, "15m": 120, "30m": 180, "1h": 300, "4h": 900, "1d": 1800}
SLEEP_RESUME_GRACE_SECONDS = 10.0
UNFILLED_ORDER_END_STATUSES = {"canceled", "expired", "rejected", "done_for_day"}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _enum_value(value: Any) -> str:
    text = str(value or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.lower()


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=PACIFIC_TIME)
    return parsed.astimezone(PACIFIC_TIME)


def _risk_limits(control: AutomationControl) -> RiskLimits:
    payload = dict(control.risk_limits or {})
    allowed = {field.name for field in RiskLimits.__dataclass_fields__.values()}
    return RiskLimits(**{key: value for key, value in payload.items() if key in allowed})


def _account_value(account_records: list[dict], field: str, default: float = 0.0) -> float:
    target = field.lower()
    for row in account_records:
        if str(row.get("Field", "")).strip().lower() == target:
            return _number(row.get("Value"), default)
    return default


def _portfolio_notional(positions: list[dict]) -> float:
    return sum(abs(_number(row.get("Market Value"))) for row in positions)


def _symbol_notional(positions: list[dict], symbol: str) -> float:
    clean = symbol.strip().upper()
    return sum(abs(_number(row.get("Market Value"))) for row in positions if str(row.get("Symbol", "")).strip().upper() == clean)


def _open_buy_order_notional(orders: list[dict], symbol: str = "") -> float:
    clean = symbol.strip().upper()
    total = 0.0
    for row in orders:
        if _enum_value(row.get("Side")) != "buy" or _enum_value(row.get("Status")) not in OPEN_ORDER_STATUSES:
            continue
        row_symbol = str(row.get("Symbol", "")).strip().upper()
        if clean and row_symbol != clean:
            continue
        remaining_quantity = max(0.0, _number(row.get("Quantity")) - _number(row.get("Filled Qty")))
        price = _number(row.get("Limit Price"), _number(row.get("Avg Fill")))
        total += remaining_quantity * max(0.0, price)
    return total


def _open_buy_order_symbols(orders: list[dict]) -> set[str]:
    return {
        str(row.get("Symbol", "")).strip().upper()
        for row in orders
        if _enum_value(row.get("Side")) == "buy" and _enum_value(row.get("Status")) in OPEN_ORDER_STATUSES
    }


def _broker_order_id(order: dict[str, Any] | None) -> str:
    if not order:
        return ""
    return str(
        order.get("Alpaca Order ID")
        or order.get("Broker Order ID")
        or order.get("broker_order_id")
        or ""
    ).strip()


def _order_symbol(order: dict[str, Any]) -> str:
    return str(order.get("Symbol") or order.get("symbol") or "").strip().upper()


def _order_side(order: dict[str, Any]) -> str:
    return _enum_value(order.get("Side") if "Side" in order else order.get("side"))


def _order_status(order: dict[str, Any] | None) -> str:
    if not order:
        return ""
    return _enum_value(order.get("Status") if "Status" in order else order.get("status"))


def _order_filled_quantity(order: dict[str, Any] | None) -> float:
    if not order:
        return 0.0
    return _number(
        order.get("Filled Qty")
        if "Filled Qty" in order
        else order.get("filled_quantity"),
    )


def _tracked_plan_id(order: dict[str, Any]) -> str:
    settings = order.get("strategy_settings") or order.get("exit_settings") or {}
    return str(settings.get("buy_watch_plan_id") or "").strip() if isinstance(settings, dict) else ""


def _order_for_watch_plan(
    plan: BuyWatchPlan,
    orders: list[dict],
    tracked_orders: list[dict],
) -> dict[str, Any] | None:
    """Find the queued setup's own order without adopting unrelated manual orders."""
    all_orders = [*orders, *tracked_orders]
    if plan.active_order_id:
        matches = [row for row in all_orders if _broker_order_id(row) == plan.active_order_id]
        if matches:
            return max(matches, key=lambda row: bool(_order_status(row)))

    tagged = [row for row in tracked_orders if _tracked_plan_id(row) == plan.plan_id]
    if tagged:
        return max(tagged, key=lambda row: _parse_time(row.get("submitted_at") or row.get("Submitted")) or datetime.min.replace(tzinfo=PACIFIC_TIME))

    sent_at = _parse_time(plan.order_sent_at)
    if sent_at is None:
        return None
    legacy_matches: list[tuple[float, dict[str, Any]]] = []
    for row in tracked_orders:
        if _order_symbol(row) != plan.symbol.strip().upper() or _order_side(row) != "buy":
            continue
        if str(row.get("source") or "").strip().lower() == "manual_order":
            continue
        submitted_at = _parse_time(row.get("submitted_at") or row.get("Submitted"))
        if submitted_at is None:
            continue
        difference = abs((submitted_at - sent_at).total_seconds())
        if difference <= 600:
            legacy_matches.append((difference, row))
    return min(legacy_matches, key=lambda item: item[0])[1] if legacy_matches else None


def _order_finished_without_fill(order: dict[str, Any] | None) -> bool:
    return bool(
        order
        and _order_status(order) in UNFILLED_ORDER_END_STATUSES
        and _order_filled_quantity(order) <= 0
    )


def _open_symbols(positions: list[dict]) -> set[str]:
    return {str(row.get("Symbol", "")).strip().upper() for row in positions if _number(row.get("Quantity")) != 0}


def _market_is_open(adapter: AlpacaBrokerAdapterStub) -> bool:
    broker_clock = adapter.market_is_open() if hasattr(adapter, "market_is_open") else None
    return bool(broker_clock) if broker_clock is not None else bool(market_session_advisory().get("Open"))


def _strict_broker_records(adapter: AlpacaBrokerAdapterStub, method_name: str) -> list[dict]:
    method = getattr(adapter, method_name)
    try:
        return method(strict=True)
    except TypeError:
        # Lightweight test adapters may not expose the strict keyword.
        return method()


def _track_broker_order(order: Any, preview_hash: str, strategy_settings: dict[str, Any] | None = None) -> dict:
    record = asdict(alpaca_tracked_order_from_broker_order(order, preview_hash=preview_hash))
    if strategy_settings:
        record["strategy_settings"] = dict(strategy_settings)
        record["exit_settings"] = dict(strategy_settings)
    record["broker_writes_submitted"] = 1
    return record


def _fetcher(control: AutomationControl, config: AlpacaConfig) -> Callable[[str, str, str, str], Any]:
    def fetch(symbol: str, history: str, interval: str, source: str):
        asset_class = normalize_asset_class("crypto" if "crypto" in source.lower() else None, symbol)
        clean_symbol = normalize_symbol(symbol, asset_class)
        key = (clean_symbol, history, interval, source)
        cached = _BAR_CACHE.get(key)
        now = time.monotonic()
        if cached and now - cached[0] < _BAR_CACHE_SECONDS.get(interval, 300):
            data = cached[1]
            return data.copy() if hasattr(data, "copy") else data
        data = fetch_price_bars(
            clean_symbol, history, interval, source, config.api_key, config.api_secret, asset_class,
        )
        _BAR_CACHE[key] = (now, data)
        return data.copy() if hasattr(data, "copy") else data

    return fetch


def _reconcile_positions(positions: list[dict], tracked_orders: list[dict]) -> list[dict]:
    tracked_symbols = {str(row.get("symbol", "")).strip().upper() for row in tracked_orders if row.get("symbol")}
    updated = list(tracked_orders)
    for position in positions:
        symbol = str(position.get("Symbol", "")).strip().upper()
        if symbol and symbol not in tracked_symbols and _number(position.get("Quantity")) > 0:
            updated.append(adopt_alpaca_position(position))
            tracked_symbols.add(symbol)
    return updated


def _cancel_stale_limit_buys(
    control: AutomationControl,
    adapter: AlpacaBrokerAdapterStub,
    orders: list[dict],
    audit_store: JsonlAuditStore,
) -> tuple[int, str]:
    if not control.auto_cancel_limit_buys or not control.paper_orders_enabled:
        return 0, "Limit buy cancel is off."
    if not adapter.config.paper:
        return 0, "Limit buy cancel is paper-only."
    cutoff = datetime.now(PACIFIC_TIME) - timedelta(minutes=max(1, int(control.stale_limit_order_minutes)))
    sent = 0
    for order in orders:
        side = _enum_value(order.get("Side"))
        status = _enum_value(order.get("Status"))
        order_type = _enum_value(order.get("Order Type"))
        submitted = _parse_time(order.get("Submitted"))
        broker_order_id = str(order.get("Alpaca Order ID") or order.get("Broker Order ID") or "").strip()
        if side != "buy" or order_type != "limit" or status not in OPEN_ORDER_STATUSES or not submitted or submitted > cutoff or not broker_order_id:
            continue
        preview = build_alpaca_cancel_preview(order, adapter.config)
        if not preview.valid:
            continue
        adapter.cancel_order(broker_order_id, expected_cancel_hash=preview.preview_hash)
        sent += 1
        audit_store.append(AuditEvent(
            event_type="worker_limit_buy_cancelled",
            message="Background worker canceled a stale Alpaca paper limit buy.",
            payload={"symbol": order.get("Symbol", ""), "broker_order_id": broker_order_id, "review_id": preview.preview_hash},
        ))
    return sent, f"Canceled {sent} stale limit buy order(s)." if sent else "No stale limit buy orders were ready to cancel."


def _cancel_late_rsi_limit_buys(
    control: AutomationControl,
    adapter: AlpacaBrokerAdapterStub,
    orders: list[dict],
    tracked_orders: list[dict],
    fetch_bars: Callable[[str, str, str, str], Any],
    audit_store: JsonlAuditStore,
) -> tuple[int, str]:
    """Cancel an unfilled RSI buy after its completed-bar rebound becomes too large."""
    if control.mode != "Auto entries and exits" or not control.full_automation_enabled:
        return 0, "RSI late-entry protection is watching only."
    if not control.paper_orders_enabled or control.kill_switch_enabled or not adapter.config.paper:
        return 0, "RSI late-entry protection is blocked."

    tracked_by_id = {
        _broker_order_id(row): row
        for row in tracked_orders
        if _broker_order_id(row)
    }
    sent = 0
    for order in orders:
        side = _enum_value(order.get("Side"))
        status = _enum_value(order.get("Status"))
        order_type = _enum_value(order.get("Order Type"))
        broker_order_id = _broker_order_id(order)
        if side != "buy" or order_type != "limit" or status not in OPEN_ORDER_STATUSES or not broker_order_id:
            continue
        tracked = tracked_by_id.get(broker_order_id, {})
        settings = dict(tracked.get("exit_settings") or tracked.get("strategy_settings") or {})
        if str(settings.get("strategy_type", "")) != "rsi_scalp":
            continue
        setup_low = _number(settings.get("entry_rsi_setup_low"), -1.0)
        max_rebound = _number(settings.get("rsi_max_rebound_points"), 12.0)
        if setup_low < 0 or max_rebound <= 0:
            continue
        symbol = str(order.get("Symbol") or settings.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        data = fetch_bars(
            symbol,
            str(settings.get("history", control.history)),
            str(settings.get("interval", control.interval)),
            str(settings.get("price_data_source", control.price_data_source)),
        )
        result = selected_strategy_result(data, settings, _number(settings.get("account_size"), control.account_size))
        current_rsi = _number(result.get("live", {}).get("rsi"), -1.0)
        rebound = current_rsi - setup_low if current_rsi >= 0 else -1.0
        if rebound <= max_rebound:
            continue
        preview = build_alpaca_cancel_preview(order, adapter.config)
        if not preview.valid:
            continue
        adapter.cancel_order(broker_order_id, expected_cancel_hash=preview.preview_hash)
        sent += 1
        audit_store.append(AuditEvent(
            event_type="worker_rsi_late_buy_cancelled",
            message="Background worker canceled an unfilled RSI paper buy after the rebound became too large.",
            payload={
                "symbol": symbol,
                "broker_order_id": broker_order_id,
                "review_id": preview.preview_hash,
                "setup_low_rsi": setup_low,
                "current_rsi": current_rsi,
                "rebound_points": rebound,
                "maximum_rebound_points": max_rebound,
            },
        ))
    return sent, f"Canceled {sent} late RSI limit buy order(s)." if sent else "No RSI limit buys became too late."


def _send_exits(
    control: AutomationControl,
    adapter: AlpacaBrokerAdapterStub,
    positions: list[dict],
    orders: list[dict],
    tracked_orders: list[dict],
    fetch_bars: Callable[[str, str, str, str], Any],
    audit_store: JsonlAuditStore,
) -> tuple[int, list[dict], str]:
    if control.mode not in {"Auto exits only", "Auto exits", "Auto entries and exits"}:
        return 0, tracked_orders, "Auto exits are off."
    if not control.paper_orders_enabled or control.kill_switch_enabled:
        return 0, tracked_orders, "Auto exits are blocked by account switch or Kill Switch."
    if not adapter.config.paper:
        return 0, tracked_orders, "Auto exits are paper-only in this worker."
    sent = 0
    updated = list(tracked_orders)
    for position in positions:
        symbol = str(position.get("Symbol", "")).strip().upper()
        asset_class = normalize_asset_class(position.get("Asset Type"), symbol)
        if not symbol or _number(position.get("Quantity")) <= 0:
            continue
        if asset_class != "crypto" and not _market_is_open(adapter):
            continue
        settings = saved_exit_settings_for_symbol(symbol, updated)
        details = evaluate_exit_settings(settings, position, fetch_bars)
        if settings and details.get("state_changed"):
            refreshed_settings = dict(settings)
            if details.get("highest_high_since_entry") is not None:
                refreshed_settings["highest_high_since_entry"] = details.get("highest_high_since_entry")
            if details.get("trigger_price") is not None:
                refreshed_settings["last_exit_trigger_price"] = details.get("trigger_price")
            updated = update_exit_settings_for_symbol(symbol, updated, refreshed_settings)
        if not details.get("ready"):
            continue
        intent = build_exit_intent_from_position(position)
        if intent is None:
            continue
        risk = RiskCheckResult(True, [], {"worker_exit": True})
        decision = ExecutionDecision("paper", True, False, "Background auto-exit sent by saved exit settings.", risk)
        preview = build_alpaca_order_preview(intent, decision, adapter.config)
        blockers = open_exit_order_reasons(preview, orders)
        if not preview.valid or blockers:
            continue
        order = adapter.submit_order(intent, decision, expected_preview_hash=preview.preview_hash)
        tracked_exit = _track_broker_order(order, preview.preview_hash, settings)
        updated.append(tracked_exit)
        sent += 1
        audit_store.append(AuditEvent(
            event_type="worker_paper_exit_sent",
            message="Background worker sent an Alpaca paper exit.",
            payload={
                "symbol": symbol,
                "quantity": intent.quantity,
                "review_id": preview.preview_hash,
                "broker_order_id": tracked_exit.get("broker_order_id", ""),
                "reason": details.get("reason"),
                "exit_details": {
                    "current_price": details.get("current_price"),
                    "trigger_price": details.get("trigger_price"),
                    "trigger_source": details.get("trigger_source"),
                    "strategy_exit_price": details.get("strategy_exit_price"),
                    "original_stop_price": details.get("original_stop_price"),
                    "breakeven_stop_price": details.get("breakeven_stop_price"),
                    "trailing_stop_price": details.get("trailing_stop_price"),
                    "highest_high_since_entry": details.get("highest_high_since_entry"),
                    "profit_r": details.get("profit_r"),
                    "current_atr": details.get("current_atr"),
                    "interval": details.get("interval"),
                },
            },
        ))
    return sent, updated, f"Sent {sent} auto exit order(s)." if sent else "No auto exits were ready."


def _send_entry(
    control: AutomationControl,
    adapter: AlpacaBrokerAdapterStub,
    positions: list[dict],
    orders: list[dict],
    tracked_orders: list[dict],
    fetch_bars: Callable[[str, str, str, str], Any],
    audit_store: JsonlAuditStore,
    *,
    latest_price: float | None = None,
    require_latest_price: bool = False,
) -> tuple[int, list[dict], str]:
    if control.mode != "Auto entries and exits" or not control.full_automation_enabled:
        return 0, tracked_orders, "Auto entries are off."
    if not control.paper_orders_enabled or control.kill_switch_enabled:
        return 0, tracked_orders, "Auto entries are blocked by account switch or Kill Switch."
    if not adapter.config.paper:
        return 0, tracked_orders, "Auto entries are paper-only in this worker."
    market_open = True if control.asset_class == "crypto" else _market_is_open(adapter)
    if not market_open and not control.allow_limit_buys_outside_market_hours:
        return 0, tracked_orders, "Market is closed; auto buys wait for regular hours."

    account_records = adapter.account_records()
    account_equity = _account_value(account_records, "portfolio value", 0.0)
    available_cash = _account_value(account_records, "cash", 0.0)
    prior_day_equity = _account_value(account_records, "last equity", account_equity)
    if account_equity <= 0 or available_cash < 0:
        return 0, tracked_orders, "Alpaca account value is unavailable; auto buy is paused."
    session_pnl = account_equity - prior_day_equity
    limits = _risk_limits(control)

    data = fetch_bars(control.symbol, control.history, control.interval, control.price_data_source)
    result = selected_strategy_result(data, control.strategy_settings, account_equity, limits)
    live = result.get("live", {})
    intent = live.get("trade_intent")
    if intent is None:
        return 0, tracked_orders, str(live.get("no_trade_reason") or "No BUY setup right now.")
    data_attrs = getattr(data, "attrs", {})
    pricing_price = _number(latest_price, 0.0)
    if require_latest_price and pricing_price <= 0:
        return 0, tracked_orders, "Latest Alpaca trade price is unavailable; queued BUY was not submitted."
    if pricing_price <= 0:
        pricing_price = _number(data_attrs.get("latest_price"), _number(intent.entry_price if intent else 0))
    intent = reprice_trade_intent(intent, pricing_price)
    intent = apply_buy_order_style(intent, control.order_style, control.limit_adjustment_pct, control.custom_limit_price)
    if not market_open and intent is not None and intent.order_type != "limit":
        return 0, tracked_orders, "Outside-hours auto buys must be limit orders."

    intent = constrain_trade_intent_to_limits(
        intent,
        account_equity,
        limits,
        current_portfolio_notional=_portfolio_notional(positions) + _open_buy_order_notional(orders),
        symbol_current_notional=_symbol_notional(positions, intent.symbol_clean if intent else "") + _open_buy_order_notional(orders, intent.symbol_clean if intent else ""),
        session_pnl=session_pnl,
        available_cash=available_cash,
    )
    risk = check_trade_intent(
        intent,
        account_equity,
        limits,
        open_positions=_open_symbols(positions),
        open_position_count=len(_open_symbols(positions) | _open_buy_order_symbols(orders)),
        current_portfolio_notional=_portfolio_notional(positions) + _open_buy_order_notional(orders),
        symbol_current_notional=_symbol_notional(positions, intent.symbol_clean if intent else "") + _open_buy_order_notional(orders, intent.symbol_clean if intent else ""),
        session_pnl=session_pnl,
        available_cash=available_cash,
    )
    decision = ExecutionDecision("paper", risk.approved, False, "Background paper buy approved by deterministic rules." if risk.approved else "; ".join(risk.rejected_reasons), risk)
    preview = build_alpaca_order_preview(intent, decision, adapter.config)
    duplicate_reasons = open_order_exposure_reasons(intent, orders)
    if not preview.valid or duplicate_reasons:
        return 0, tracked_orders, "; ".join(preview.blocked_reasons + duplicate_reasons) or "Auto buy blocked."

    order = adapter.submit_order(intent, decision, expected_preview_hash=preview.preview_hash)
    settings = dict(control.strategy_settings)
    live = result.get("live", {})
    settings.update({
        "symbol": intent.symbol_clean,
        "asset_class": intent.asset_class,
        "history": control.history,
        "interval": control.interval,
        "price_data_source": control.price_data_source,
        "account_size": account_equity,
        "entry_reference_price": intent.entry_price,
        "entry_stop_atr_multiplier": settings.get("atr_stop_multiplier"),
        "entry_stop_loss": intent.stop_loss,
        "entry_stop_distance": abs(float(intent.entry_price or 0) - float(intent.stop_loss or 0)),
        "planned_order_type": intent.order_type,
        "planned_limit_price": intent.limit_price,
        "planned_quantity": intent.quantity,
        "auto_exit_enabled": True,
        "exit_mode": "strategy_and_atr",
        "entry_rsi": live.get("rsi"),
        "entry_rsi_setup_low": live.get("rsi_setup_low") if live.get("rsi_setup_low") is not None else live.get("rsi"),
        "entry_rsi_sell_level": (
            live.get("rsi_sell_level")
            if live.get("rsi_sell_level") is not None
            else min(
                float(settings.get("rsi_overbought", 70.0)),
                float(live.get("rsi")) + float(settings.get("rsi_sell_recovery_points", 35.0)),
            )
            if str(settings.get("strategy_type", "")) == "rsi_scalp" and live.get("rsi") is not None
            else None
        ),
    })
    tracked_orders = list(tracked_orders)
    tracked_orders.append(_track_broker_order(order, preview.preview_hash, settings))
    audit_store.append(AuditEvent(
        event_type="worker_paper_buy_sent",
        message="Background worker sent an Alpaca paper buy.",
        payload={"symbol": intent.symbol_clean, "quantity": intent.quantity, "review_id": preview.preview_hash, "order_type": intent.order_type},
    ))
    return 1, tracked_orders, f"Sent paper buy for {intent.quantity} {intent.symbol_clean}."


def _control_for_watch_plan(control: AutomationControl, plan: BuyWatchPlan) -> AutomationControl:
    saved_risk_limits = dict(plan.risk_limits)
    saved_risk_limits["kill_switch_enabled"] = bool(control.kill_switch_enabled)
    return replace(
        control,
        symbol=plan.symbol,
        asset_class=plan.asset_class,
        price_data_source=plan.price_data_source,
        history=plan.history,
        interval=plan.interval,
        strategy_settings=dict(plan.strategy_settings) | {"buy_watch_plan_id": plan.plan_id},
        risk_limits=saved_risk_limits,
        order_style=plan.order_style,
        limit_adjustment_pct=float(plan.limit_adjustment_pct),
        custom_limit_price=float(plan.custom_limit_price),
    )


def _repeat_signal_state(
    control: AutomationControl,
    adapter: AlpacaBrokerAdapterStub,
    fetch_bars: Callable[[str, str, str, str], Any],
) -> tuple[bool | None, str]:
    """Read the saved strategy without allowing an order submission."""
    try:
        account_equity = _account_value(adapter.account_records(), "portfolio value", 0.0)
        if account_equity <= 0:
            return None, "Alpaca account value is unavailable; the repeating setup cannot check for a new BUY yet."
        data = fetch_bars(control.symbol, control.history, control.interval, control.price_data_source)
        result = selected_strategy_result(data, control.strategy_settings, account_equity, _risk_limits(control))
        return result.get("live", {}).get("trade_intent") is not None, ""
    except Exception as exc:
        return None, f"Could not check whether the prior BUY signal cleared: {exc}"


def _repeat_cooldown_remaining(plan: BuyWatchPlan, now: datetime) -> float:
    cooldown_minutes = max(0.0, _number(plan.strategy_settings.get("reentry_cooldown_minutes"), 0.0))
    completed_at = _parse_time(plan.last_cycle_completed_at)
    if cooldown_minutes <= 0 or completed_at is None:
        return 0.0
    elapsed_minutes = max(0.0, (now - completed_at).total_seconds() / 60)
    return max(0.0, cooldown_minutes - elapsed_minutes)


def _watch_plan_snapshot(
    control: AutomationControl,
    adapter: AlpacaBrokerAdapterStub,
    fetch_bars: Callable[[str, str, str, str], Any],
    latest_price: float | None,
) -> dict[str, Any]:
    account_equity = _account_value(adapter.account_records(), "portfolio value", 0.0)
    data = fetch_bars(control.symbol, control.history, control.interval, control.price_data_source)
    result = selected_strategy_result(data, control.strategy_settings, account_equity, _risk_limits(control))
    return build_buy_level_snapshot(result.get("live", {}), interval=control.interval, latest_price=latest_price)


def _send_watchlist_entries(
    control: AutomationControl,
    adapter: AlpacaBrokerAdapterStub,
    positions: list[dict],
    orders: list[dict],
    tracked_orders: list[dict],
    fetch_bars: Callable[[str, str, str, str], Any],
    audit_store: JsonlAuditStore,
    store: BuyWatchlistStore,
    latest_prices: dict[str, float] | None = None,
    latest_price_error: str = "",
) -> tuple[int, list[dict], str]:
    plans = store.read()
    enabled = [plan for plan in plans if plan.enabled]
    if not enabled:
        return 0, tracked_orders, "Buy watchlist has no enabled setups."
    if control.mode != "Auto entries and exits" or not control.full_automation_enabled:
        now = datetime.now(PACIFIC_TIME).isoformat()
        for plan in enabled:
            store.update(
                plan.plan_id,
                status="Paused",
                detail="Select Auto exits and queued buys and check Allow queued buys to monitor this setup.",
                last_checked_at=now,
            )
        return 0, tracked_orders, "Buy watchlist is paused because automatic entries are off."

    sent = 0
    messages = []
    current_positions = list(positions)
    current_orders = list(orders)
    tracked = list(tracked_orders)
    for plan in enabled:
        checked_at = datetime.now(PACIFIC_TIME).isoformat()
        plan_control = _control_for_watch_plan(control, plan)
        try:
            snapshot = _watch_plan_snapshot(
                plan_control,
                adapter,
                fetch_bars,
                (latest_prices or {}).get(plan.symbol),
            )
            store.update(
                plan.plan_id,
                latest_price=(latest_prices or {}).get(plan.symbol),
                next_buy_level=snapshot.get("next_buy_level"),
                distance_to_buy_pct=snapshot.get("distance_to_buy_pct"),
                buy_requirement_levels=snapshot.get("records") or [],
                last_checked_at=checked_at,
            )
        except Exception as exc:
            store.update(
                plan.plan_id,
                buy_requirement_levels=[],
                last_checked_at=checked_at,
                detail=f"Could not calculate current BUY requirements: {exc}",
            )
        if plan.repeat_after_exit:
            clean_symbol = plan.symbol.strip().upper()
            position_open = clean_symbol in _open_symbols(current_positions)
            buy_order_open = clean_symbol in _open_buy_order_symbols(current_orders)
            owned_order = _order_for_watch_plan(plan, current_orders, tracked)
            owned_order_id = _broker_order_id(owned_order) or plan.active_order_id
            owned_order_open = _order_status(owned_order) in OPEN_ORDER_STATUSES
            owned_order_filled = _order_filled_quantity(owned_order) > 0 or _order_status(owned_order) == "filled"
            if position_open:
                queue_owns_position = bool(
                    plan.cycle_had_filled_position
                    or owned_order_filled
                    or (plan.cycle_state == "order_pending" and owned_order_id)
                )
                store.update(
                    plan.plan_id,
                    cycle_state="position_open" if queue_owns_position else "blocked_by_position",
                    cycle_had_filled_position=queue_owns_position,
                    active_order_id=owned_order_id if queue_owns_position else plan.active_order_id,
                    status="Position open",
                    detail=(
                        "Repeat after exit is On. This queued order filled; waiting for its position to close before looking for another BUY."
                        if queue_owns_position
                        else "A separate position is open for this ticker. This queued setup will resume when that position closes."
                    ),
                    last_checked_at=checked_at,
                )
                messages.append(f"{plan.symbol} {plan.strategy_label}: position open; repeat remains on.")
                continue
            if owned_order_open:
                store.update(
                    plan.plan_id,
                    cycle_state="order_pending",
                    active_order_id=owned_order_id,
                    status="Buy order active",
                    detail="Repeat after exit is On. Waiting for this setup's current buy order to fill or finish.",
                    last_checked_at=checked_at,
                )
                messages.append(f"{plan.symbol} {plan.strategy_label}: buy order active; repeat remains on.")
                continue
            if buy_order_open:
                store.update(
                    plan.plan_id,
                    cycle_state="blocked_by_order",
                    status="Other buy order open",
                    detail="A separate buy order is open for this ticker. This queued setup will resume when that order finishes.",
                    last_checked_at=checked_at,
                )
                messages.append(f"{plan.symbol} {plan.strategy_label}: another buy order is open.")
                continue
            if plan.cycle_state in {"blocked_by_position", "blocked_by_order"}:
                store.update(
                    plan.plan_id,
                    cycle_state="waiting_for_buy",
                    cycle_had_filled_position=False,
                    status="Waiting for BUY",
                    detail="The separate position or order finished. Checking this saved setup for a BUY again.",
                    last_checked_at=checked_at,
                )
                messages.append(f"{plan.symbol} {plan.strategy_label}: separate position or order finished; queue resumed.")
                continue
            if plan.cycle_state == "order_pending":
                if _order_finished_without_fill(owned_order):
                    store.update(
                        plan.plan_id,
                        cycle_state="waiting_for_retry",
                        last_cycle_completed_at=plan.last_cycle_completed_at or checked_at,
                        cycle_had_filled_position=False,
                        active_order_id=owned_order_id,
                        status="Waiting to retry",
                        detail=(
                            "This setup's previous buy order was canceled without filling. It may retry after the saved wait time "
                            "if the BUY requirements still pass; the signal does not need to turn off first."
                        ),
                        last_checked_at=checked_at,
                    )
                    messages.append(f"{plan.symbol} {plan.strategy_label}: unfilled order ended; waiting to retry.")
                    continue
                if owned_order_filled:
                    store.update(
                        plan.plan_id,
                        cycle_state="waiting_for_signal_reset",
                        last_cycle_completed_at=plan.last_cycle_completed_at or checked_at,
                        cycle_had_filled_position=True,
                        active_order_id=owned_order_id,
                        status="Waiting for a new BUY",
                        detail="The queued order filled and its position has closed. The prior BUY signal must turn off before buying again.",
                        last_checked_at=checked_at,
                    )
                    messages.append(f"{plan.symbol} {plan.strategy_label}: filled cycle finished; waiting for a new BUY signal.")
                    continue
                store.update(
                    plan.plan_id,
                    status="Checking prior order",
                    detail="The queued order is no longer open, but its final fill or cancellation status is not available yet.",
                    last_checked_at=checked_at,
                )
                messages.append(f"{plan.symbol} {plan.strategy_label}: waiting for the prior order's final status.")
                continue
            if plan.cycle_state == "position_open":
                store.update(
                    plan.plan_id,
                    cycle_state="waiting_for_signal_reset",
                    last_cycle_completed_at=checked_at,
                    cycle_had_filled_position=True,
                    status="Waiting for a new BUY",
                    detail="The queued position closed. The prior BUY signal must turn off before this setup can buy again.",
                    last_checked_at=checked_at,
                )
                messages.append(f"{plan.symbol} {plan.strategy_label}: filled cycle finished; waiting for the BUY signal to reset.")
                continue
            if plan.cycle_state == "waiting_for_signal_reset":
                if not plan.cycle_had_filled_position and _order_finished_without_fill(owned_order):
                    store.update(
                        plan.plan_id,
                        cycle_state="waiting_for_retry",
                        status="Waiting to retry",
                        detail=(
                            "The previous queued buy was canceled without filling. It may retry after the saved wait time "
                            "without requiring the BUY signal to turn off."
                        ),
                        last_checked_at=checked_at,
                    )
                    messages.append(f"{plan.symbol} {plan.strategy_label}: corrected unfilled order state; waiting to retry.")
                    continue
                signal_active, signal_error = _repeat_signal_state(plan_control, adapter, fetch_bars)
                if signal_active is None:
                    store.update(
                        plan.plan_id,
                        status="Blocked",
                        detail=signal_error,
                        last_checked_at=checked_at,
                    )
                    messages.append(f"{plan.symbol} {plan.strategy_label}: {signal_error}")
                elif signal_active:
                    store.update(
                        plan.plan_id,
                        status="Waiting for a new BUY",
                        detail="Repeat after exit is On. The previous BUY signal is still active and must turn off before rearming.",
                        last_checked_at=checked_at,
                    )
                    messages.append(f"{plan.symbol} {plan.strategy_label}: waiting for the previous BUY signal to turn off.")
                else:
                    store.update(
                        plan.plan_id,
                        cycle_state="waiting_for_buy",
                        cycle_had_filled_position=False,
                        active_order_id="",
                        status="Waiting for BUY",
                        detail="Repeat after exit is On. The prior signal cleared; waiting for a new BUY signal.",
                        last_checked_at=checked_at,
                    )
                    messages.append(f"{plan.symbol} {plan.strategy_label}: signal reset; waiting for a new BUY.")
                continue
            if plan.cycle_state == "waiting_for_retry":
                retry_wait = _repeat_cooldown_remaining(plan, datetime.now(PACIFIC_TIME))
                if retry_wait > 0:
                    store.update(
                        plan.plan_id,
                        status="Waiting to retry",
                        detail=f"The previous buy was canceled without filling. Retrying in about {retry_wait:.0f} minutes if BUY requirements still pass.",
                        last_checked_at=checked_at,
                    )
                    messages.append(f"{plan.symbol} {plan.strategy_label}: waiting before retrying the unfilled buy.")
                else:
                    store.update(
                        plan.plan_id,
                        cycle_state="waiting_for_buy",
                        active_order_id="",
                        cycle_had_filled_position=False,
                        status="Waiting for BUY",
                        detail="The unfilled-order wait is complete. Checking the saved BUY requirements again.",
                        last_checked_at=checked_at,
                    )
                    messages.append(f"{plan.symbol} {plan.strategy_label}: unfilled-order wait complete; queue rearmed.")
                continue
            cooldown_remaining = _repeat_cooldown_remaining(plan, datetime.now(PACIFIC_TIME))
            if cooldown_remaining > 0:
                store.update(
                    plan.plan_id,
                    status="Cooling down",
                    detail=f"Repeat after exit is On. Waiting {cooldown_remaining:.0f} more minutes before another BUY.",
                    last_checked_at=checked_at,
                )
                messages.append(f"{plan.symbol} {plan.strategy_label}: cooling down before another BUY.")
                continue
        if plan.price_data_source not in {"Ticker (Alpaca)", "Crypto (Alpaca)"}:
            message = "Queued automatic buys require Alpaca price data for a current order price."
            store.update(
                plan.plan_id,
                status="Blocked",
                detail=message,
                last_checked_at=checked_at,
            )
            messages.append(f"{plan.symbol} {plan.strategy_label}: {message}")
            continue
        if plan.symbol not in (latest_prices or {}):
            message = (
                f"Latest Alpaca trade request failed: {latest_price_error}"
                if latest_price_error
                else "Latest Alpaca trade price is unavailable; queued BUY was not submitted."
            )
            store.update(
                plan.plan_id,
                status="Blocked",
                detail=message,
                last_checked_at=checked_at,
            )
            messages.append(f"{plan.symbol} {plan.strategy_label}: {message}")
            continue
        count, tracked, message = _send_entry(
            plan_control,
            adapter,
            current_positions,
            current_orders,
            tracked,
            fetch_bars,
            audit_store,
            latest_price=(latest_prices or {}).get(plan.symbol),
            require_latest_price=True,
        )
        if count:
            sent += count
            if plan.repeat_after_exit:
                submitted_order_id = _broker_order_id(tracked[-1]) if tracked else ""
                store.update(
                    plan.plan_id,
                    enabled=True,
                    cycle_state="order_pending",
                    active_order_id=submitted_order_id,
                    cycle_had_filled_position=False,
                    status="Buy order sent",
                    detail="Repeat after exit is On. Waiting for this order and position cycle to finish.",
                    last_checked_at=checked_at,
                    order_sent_at=checked_at,
                )
            else:
                store.update(
                    plan.plan_id,
                    enabled=False,
                    status="Order sent",
                    detail=message,
                    last_checked_at=checked_at,
                    order_sent_at=checked_at,
                )
            current_positions = _strict_broker_records(adapter, "position_records")
            current_orders = _strict_broker_records(adapter, "order_records")
        else:
            waiting = message.startswith("No BUY")
            store.update(
                plan.plan_id,
                status="Waiting for BUY" if waiting else "Blocked",
                detail=(
                    message
                    if waiting and message != "No BUY setup right now."
                    else "The saved strategy's required BUY rules have not passed yet."
                    if waiting
                    else message
                ),
                last_checked_at=checked_at,
            )
        messages.append(f"{plan.symbol} {plan.strategy_label}: {message}")
    return sent, tracked, " | ".join(messages)[:800]


def run_once(
    control: AutomationControl,
    status: WorkerStatus | None = None,
    *,
    adapter: AlpacaBrokerAdapterStub | None = None,
) -> WorkerStatus:
    previous = status or WorkerStatus()
    checked_at = datetime.now(PACIFIC_TIME).isoformat()
    if not control.enabled:
        return replace(previous, running=True, pid=os.getpid(), state="Watching only", last_checked_at=checked_at, last_action="Background worker is off.", loop_count=previous.loop_count + 1, last_error="")

    config = AlpacaConfig.from_env()
    adapter = adapter or AlpacaBrokerAdapterStub(config=config, allow_order_submission=True)
    audit_store = JsonlAuditStore(control.audit_log_path)
    broker_store = BrokerStateStore(control.broker_state_path)
    try:
        if not adapter.config.paper:
            raise RuntimeError("Background worker is paper-only. Set ALPACA_PAPER=true for this worker.")
        positions = _strict_broker_records(adapter, "position_records")
        orders = _strict_broker_records(adapter, "order_records")
        tracked = refresh_tracked_alpaca_orders(broker_store.read(), orders)
        tracked = _reconcile_positions(positions, tracked)

        fetch_bars = _fetcher(control, adapter.config)
        rsi_cancel_count, rsi_cancel_action = _cancel_late_rsi_limit_buys(
            control, adapter, orders, tracked, fetch_bars, audit_store
        )
        if rsi_cancel_count:
            stale_cancel_count, stale_cancel_action = 0, "Stale-limit checks continue on the next worker cycle."
        else:
            stale_cancel_count, stale_cancel_action = _cancel_stale_limit_buys(control, adapter, orders, audit_store)
        cancel_count = rsi_cancel_count + stale_cancel_count
        cancel_action = "; ".join(text for text in (rsi_cancel_action, stale_cancel_action) if text)
        orders = _strict_broker_records(adapter, "order_records")
        tracked = refresh_tracked_alpaca_orders(tracked, orders)
        exit_count, tracked, exit_action = _send_exits(control, adapter, positions, orders, tracked, fetch_bars, audit_store)
        broker_store.replace_all(refresh_tracked_alpaca_orders(tracked, _strict_broker_records(adapter, "order_records")))
        positions = _strict_broker_records(adapter, "position_records")
        orders = _strict_broker_records(adapter, "order_records")
        tracked = refresh_tracked_alpaca_orders(tracked, orders)
        watchlist_store = BuyWatchlistStore(control.buy_watchlist_path)
        watchlist_plans = watchlist_store.read()
        if watchlist_plans:
            latest_prices: dict[str, float] = {}
            latest_price_error = ""
            if control.mode == "Auto entries and exits" and control.full_automation_enabled:
                try:
                    stock_symbols = [
                        plan.symbol for plan in watchlist_plans
                        if plan.enabled and plan.asset_class != "crypto" and plan.price_data_source == "Ticker (Alpaca)"
                    ]
                    crypto_symbols = [
                        plan.symbol for plan in watchlist_plans
                        if plan.enabled and plan.asset_class == "crypto"
                    ]
                    latest_prices = fetch_alpaca_latest_trades(
                        stock_symbols,
                        adapter.config.api_key,
                        adapter.config.api_secret,
                    )
                    latest_prices.update(fetch_alpaca_latest_crypto_trades(
                        crypto_symbols,
                        adapter.config.api_key,
                        adapter.config.api_secret,
                    ))
                except Exception as exc:
                    latest_price_error = str(exc)
            buy_count, tracked, buy_action = _send_watchlist_entries(
                control,
                adapter,
                positions,
                orders,
                tracked,
                fetch_bars,
                audit_store,
                watchlist_store,
                latest_prices=latest_prices,
                latest_price_error=latest_price_error,
            )
        else:
            buy_count = 0
            buy_action = "Buy watchlist is empty; no automatic buys are monitored."
        broker_store.replace_all(refresh_tracked_alpaca_orders(tracked, _strict_broker_records(adapter, "order_records")))

        actions = [text for text in (cancel_action, exit_action, buy_action) if text]
        state = "Ready" if buy_count or exit_count or cancel_count else "Watching"
        return WorkerStatus(
            running=True,
            pid=os.getpid(),
            state=state,
            last_checked_at=checked_at,
            last_action="; ".join(actions)[:800],
            last_error="",
            loop_count=previous.loop_count + 1,
            orders_sent=previous.orders_sent + buy_count,
            cancels_sent=previous.cancels_sent + cancel_count,
            exits_sent=previous.exits_sent + exit_count,
        )
    except Exception as exc:
        return replace(
            previous,
            running=True,
            pid=os.getpid(),
            state="Blocked",
            last_checked_at=checked_at,
            last_action="Worker check failed.",
            last_error=str(exc),
            loop_count=previous.loop_count + 1,
        )


def run_loop(control_path: str | Path, status_path: str | Path, once: bool = False) -> int:
    control_store = AutomationControlStore(control_path)
    status_store = WorkerStatusStore(status_path)
    lock = WorkerLock()
    if not lock.acquire():
        status_store.write(WorkerStatus(running=False, state="Already running", last_error="Another worker lock is active."))
        return 2
    startup_fingerprint = worker_code_fingerprint()
    started_at = datetime.now(PACIFIC_TIME).isoformat()
    status_store.write(
        replace(
            status_store.read(),
            running=True,
            pid=os.getpid(),
            state="Starting",
            code_fingerprint=startup_fingerprint,
            started_at=started_at,
            last_error="",
        )
    )
    try:
        while True:
            if worker_code_fingerprint() != startup_fingerprint:
                status_store.write(
                    replace(
                        status_store.read(),
                        running=False,
                        state="Restart required",
                        last_checked_at=datetime.now(PACIFIC_TIME).isoformat(),
                        last_action="Worker stopped because the app code changed. Start Worker to load the updated logic.",
                        last_error="",
                    )
                )
                break
            control = control_store.read()
            if control.stop_requested:
                status_store.write(
                    replace(
                        status_store.read(),
                        running=True,
                        pid=os.getpid(),
                        state="Stopping",
                        last_checked_at=datetime.now(PACIFIC_TIME).isoformat(),
                        last_action="Stop requested from the app.",
                        last_error="",
                    )
                )
                break
            status = run_once(control, status_store.read())
            status = replace(
                status,
                code_fingerprint=startup_fingerprint,
                started_at=started_at,
            )
            status_store.write(status)
            if once:
                break
            wait_seconds = max(5, int(control.refresh_seconds or 15))
            wait_started_at = time.time()
            stop_requested = _stop_requested_during_wait(control_store, wait_seconds)
            wait_elapsed = time.time() - wait_started_at
            if sleep_resume_detected(wait_elapsed, wait_seconds):
                control_store.write(replace(control_store.read(), enabled=False, stop_requested=True))
                status_store.write(
                    replace(
                        status_store.read(),
                        running=False,
                        state="Stopped after sleep",
                        last_checked_at=datetime.now(PACIFIC_TIME).isoformat(),
                        last_action="Background worker stopped after the computer slept or execution was suspended.",
                        last_error="",
                    )
                )
                break
            if stop_requested:
                continue
    finally:
        lock.release()
        final = status_store.read()
        final_state = final.state if final.state in {"Restart required", "Stopped after sleep"} else "Stopped"
        status_store.write(replace(final, running=False, state=final_state))
    return 0


def _stop_requested_during_wait(control_store: AutomationControlStore, seconds: int) -> bool:
    for _ in range(max(1, int(seconds))):
        time.sleep(1)
        if control_store.read().stop_requested:
            return True
    return False


def sleep_resume_detected(
    elapsed_seconds: float,
    expected_wait_seconds: float,
    grace_seconds: float = SLEEP_RESUME_GRACE_SECONDS,
) -> bool:
    """Fail closed when the worker resumes after a system sleep or long suspension."""
    return float(elapsed_seconds) > float(expected_wait_seconds) + float(grace_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Trading Simulator paper automation worker.")
    parser.add_argument("--control", default=str(AutomationControlStore().path))
    parser.add_argument("--status", default=str(WorkerStatusStore().path))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    return run_loop(args.control, args.status, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
