from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from agentloop_trader.audit_store import JsonlAuditStore
from agentloop_trader.automation_runtime import (
    AutomationControl,
    AutomationControlStore,
    WorkerLock,
    WorkerStatus,
    WorkerStatusStore,
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
from agentloop_trader.market_data import fetch_alpaca_latest_trades, fetch_price_bars
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


_BAR_CACHE: dict[tuple[str, str, str, str], tuple[float, Any]] = {}
_BAR_CACHE_SECONDS = {"1m": 30, "5m": 60, "15m": 120, "30m": 180, "1h": 300, "4h": 900, "1d": 1800}


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
        key = (symbol.strip().upper(), history, interval, source)
        cached = _BAR_CACHE.get(key)
        now = time.monotonic()
        if cached and now - cached[0] < _BAR_CACHE_SECONDS.get(interval, 300):
            data = cached[1]
            return data.copy() if hasattr(data, "copy") else data
        data = fetch_price_bars(symbol, history, interval, source, config.api_key, config.api_secret)
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
    if not _market_is_open(adapter):
        return 0, tracked_orders, "Market is closed; auto exits wait for regular hours."

    sent = 0
    updated = list(tracked_orders)
    for position in positions:
        symbol = str(position.get("Symbol", "")).strip().upper()
        if not symbol or _number(position.get("Quantity")) <= 0:
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
    market_open = _market_is_open(adapter)
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
    intent = result.get("live", {}).get("trade_intent")
    if intent is None:
        return 0, tracked_orders, "No BUY setup right now."
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
    settings.update({
        "symbol": intent.symbol_clean,
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
        price_data_source=plan.price_data_source,
        history=plan.history,
        interval=plan.interval,
        strategy_settings=dict(plan.strategy_settings),
        risk_limits=saved_risk_limits,
        order_style=plan.order_style,
        limit_adjustment_pct=float(plan.limit_adjustment_pct),
        custom_limit_price=float(plan.custom_limit_price),
    )


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
    *,
    max_to_send: int,
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
                detail="Select Auto entries and exits and check Enable Automation to monitor this setup.",
                last_checked_at=now,
            )
        return 0, tracked_orders, "Buy watchlist is paused because automatic entries are off."

    sent = 0
    messages = []
    current_positions = list(positions)
    current_orders = list(orders)
    tracked = list(tracked_orders)
    for plan in enabled:
        if sent >= max(0, max_to_send):
            break
        if plan.price_data_source != "Ticker (Alpaca)":
            checked_at = datetime.now(PACIFIC_TIME).isoformat()
            message = "Queued automatic buys require Ticker (Alpaca) for a current order price."
            store.update(
                plan.plan_id,
                status="Blocked",
                detail=message,
                last_checked_at=checked_at,
            )
            messages.append(f"{plan.symbol} {plan.strategy_label}: {message}")
            continue
        if plan.symbol not in (latest_prices or {}):
            checked_at = datetime.now(PACIFIC_TIME).isoformat()
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
        plan_control = _control_for_watch_plan(control, plan)
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
        checked_at = datetime.now(PACIFIC_TIME).isoformat()
        if count:
            sent += count
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
            waiting = message == "No BUY setup right now."
            store.update(
                plan.plan_id,
                status="Waiting for BUY" if waiting else "Blocked",
                detail=(
                    "The saved strategy's required BUY rules have not passed yet."
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

        cancel_count, cancel_action = _cancel_stale_limit_buys(control, adapter, orders, audit_store)
        orders = _strict_broker_records(adapter, "order_records")
        tracked = refresh_tracked_alpaca_orders(tracked, orders)
        fetch_bars = _fetcher(control, adapter.config)
        exit_count, tracked, exit_action = _send_exits(control, adapter, positions, orders, tracked, fetch_bars, audit_store)
        broker_store.replace_all(refresh_tracked_alpaca_orders(tracked, _strict_broker_records(adapter, "order_records")))
        positions = _strict_broker_records(adapter, "position_records")
        orders = _strict_broker_records(adapter, "order_records")
        tracked = refresh_tracked_alpaca_orders(tracked, orders)
        watchlist_store = BuyWatchlistStore(control.buy_watchlist_path)
        watchlist_plans = watchlist_store.read()
        max_session_buys = max(1, int(_number(control.strategy_settings.get("max_auto_buys_per_session"), 3)))
        remaining_buys = max(0, max_session_buys - previous.orders_sent)
        if remaining_buys <= 0:
            buy_count, buy_action = 0, f"Automatic buys reached the session limit of {max_session_buys}."
        elif watchlist_plans:
            latest_prices: dict[str, float] = {}
            latest_price_error = ""
            if control.mode == "Auto entries and exits" and control.full_automation_enabled:
                try:
                    latest_prices = fetch_alpaca_latest_trades(
                        [plan.symbol for plan in watchlist_plans if plan.enabled and plan.price_data_source == "Ticker (Alpaca)"],
                        adapter.config.api_key,
                        adapter.config.api_secret,
                    )
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
                max_to_send=remaining_buys,
            )
        else:
            latest_price = None
            require_latest_price = control.price_data_source == "Ticker (Alpaca)"
            if require_latest_price and control.mode == "Auto entries and exits" and control.full_automation_enabled:
                try:
                    latest_price = fetch_alpaca_latest_trades(
                        [control.symbol],
                        adapter.config.api_key,
                        adapter.config.api_secret,
                    ).get(control.symbol.strip().upper())
                except Exception:
                    latest_price = None
            buy_count, tracked, buy_action = _send_entry(
                control,
                adapter,
                positions,
                orders,
                tracked,
                fetch_bars,
                audit_store,
                latest_price=latest_price,
                require_latest_price=require_latest_price,
            )
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
    try:
        while True:
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
            status_store.write(status)
            if once:
                break
            if _stop_requested_during_wait(control_store, max(5, int(control.refresh_seconds or 15))):
                continue
    finally:
        lock.release()
        final = status_store.read()
        status_store.write(replace(final, running=False, state="Stopped"))
    return 0


def _stop_requested_during_wait(control_store: AutomationControlStore, seconds: int) -> bool:
    for _ in range(max(1, int(seconds))):
        time.sleep(1)
        if control_store.read().stop_requested:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Trading Simulator paper automation worker.")
    parser.add_argument("--control", default=str(AutomationControlStore().path))
    parser.add_argument("--status", default=str(WorkerStatusStore().path))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    return run_loop(args.control, args.status, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
