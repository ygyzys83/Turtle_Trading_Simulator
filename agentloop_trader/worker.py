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
from agentloop_trader.market_data import fetch_price_bars
from agentloop_trader.models import AuditEvent, ExecutionDecision, PACIFIC_TIME, RiskCheckResult, RiskLimits
from agentloop_trader.risk import check_trade_intent, constrain_trade_intent_to_limits
from agentloop_trader.strategy_runtime import (
    apply_buy_order_style,
    evaluate_exit_settings,
    selected_strategy_result,
    saved_exit_settings_for_symbol,
    update_exit_settings_for_symbol,
)


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


def _open_symbols(positions: list[dict]) -> set[str]:
    return {str(row.get("Symbol", "")).strip().upper() for row in positions if _number(row.get("Quantity")) > 0}


def _track_broker_order(order: Any, preview_hash: str, strategy_settings: dict[str, Any] | None = None) -> dict:
    record = asdict(alpaca_tracked_order_from_broker_order(order, preview_hash=preview_hash))
    if strategy_settings:
        record["strategy_settings"] = dict(strategy_settings)
        record["exit_settings"] = dict(strategy_settings)
    record["broker_writes_submitted"] = 1
    return record


def _fetcher(control: AutomationControl, config: AlpacaConfig) -> Callable[[str, str, str, str], Any]:
    def fetch(symbol: str, history: str, interval: str, source: str):
        return fetch_price_bars(symbol, history, interval, source, config.api_key, config.api_secret)

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
    if control.mode not in {"Auto exits", "Auto entries and exits"}:
        return 0, tracked_orders, "Auto exits are off."
    if not control.paper_orders_enabled or control.kill_switch_enabled:
        return 0, tracked_orders, "Auto exits are blocked by account switch or Kill Switch."
    if not adapter.config.paper:
        return 0, tracked_orders, "Auto exits are paper-only in this worker."
    if not bool(market_session_advisory().get("Open")):
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
        updated.append(_track_broker_order(order, preview.preview_hash, settings))
        sent += 1
        audit_store.append(AuditEvent(
            event_type="worker_paper_exit_sent",
            message="Background worker sent an Alpaca paper exit.",
            payload={"symbol": symbol, "quantity": intent.quantity, "review_id": preview.preview_hash, "reason": details.get("reason")},
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
) -> tuple[int, list[dict], str]:
    if control.mode != "Auto entries and exits" or not control.full_automation_enabled:
        return 0, tracked_orders, "Auto entries are off."
    if not control.paper_orders_enabled or control.kill_switch_enabled:
        return 0, tracked_orders, "Auto entries are blocked by account switch or Kill Switch."
    if not adapter.config.paper:
        return 0, tracked_orders, "Auto entries are paper-only in this worker."
    session = market_session_advisory()
    if not bool(session.get("Open")) and not control.allow_limit_buys_outside_market_hours:
        return 0, tracked_orders, "Market is closed; auto buys wait for regular hours."

    data = fetch_bars(control.symbol, control.history, control.interval, control.price_data_source)
    result = selected_strategy_result(data, control.strategy_settings, max(1.0, float(control.account_size)), _risk_limits(control))
    intent = result.get("live", {}).get("trade_intent")
    if intent is None:
        return 0, tracked_orders, "No BUY setup right now."
    intent = apply_buy_order_style(intent, control.order_style, control.limit_adjustment_pct, control.custom_limit_price)
    if not bool(session.get("Open")) and intent is not None and intent.order_type != "limit":
        return 0, tracked_orders, "Outside-hours auto buys must be limit orders."

    account_records = adapter.account_records()
    account_equity = _account_value(account_records, "portfolio value", control.account_size)
    available_cash = _account_value(account_records, "cash", 0.0)
    limits = _risk_limits(control)
    intent = constrain_trade_intent_to_limits(
        intent,
        account_equity,
        limits,
        current_portfolio_notional=_portfolio_notional(positions),
        symbol_current_notional=_symbol_notional(positions, intent.symbol_clean if intent else ""),
        available_cash=available_cash,
    )
    risk = check_trade_intent(
        intent,
        account_equity,
        limits,
        open_positions=_open_symbols(positions),
        open_position_count=len(_open_symbols(positions)),
        current_portfolio_notional=_portfolio_notional(positions),
        symbol_current_notional=_symbol_notional(positions, intent.symbol_clean if intent else ""),
        available_cash=available_cash,
    )
    decision = ExecutionDecision("paper", risk.approved, False, "Background paper buy approved by deterministic rules." if risk.approved else "; ".join(risk.rejected_reasons), risk)
    preview = build_alpaca_order_preview(intent, decision, adapter.config)
    duplicate_reasons = open_order_exposure_reasons(intent, orders, allow_duplicate=limits.allow_add_to_existing_position)
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
        positions = adapter.position_records()
        orders = adapter.order_records()
        tracked = refresh_tracked_alpaca_orders(broker_store.read(), orders)
        tracked = _reconcile_positions(positions, tracked)

        cancel_count, cancel_action = _cancel_stale_limit_buys(control, adapter, orders, audit_store)
        orders = adapter.order_records()
        tracked = refresh_tracked_alpaca_orders(tracked, orders)
        fetch_bars = _fetcher(control, adapter.config)
        exit_count, tracked, exit_action = _send_exits(control, adapter, positions, orders, tracked, fetch_bars, audit_store)
        positions = adapter.position_records()
        orders = adapter.order_records()
        tracked = refresh_tracked_alpaca_orders(tracked, orders)
        buy_count, tracked, buy_action = _send_entry(control, adapter, positions, orders, tracked, fetch_bars, audit_store)
        broker_store.replace_all(refresh_tracked_alpaca_orders(tracked, adapter.order_records()))

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
            time.sleep(max(5, int(control.refresh_seconds or 15)))
    finally:
        lock.release()
        final = status_store.read()
        status_store.write(replace(final, running=False, state="Stopped"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Trading Simulator paper automation worker.")
    parser.add_argument("--control", default=str(AutomationControlStore().path))
    parser.add_argument("--status", default=str(WorkerStatusStore().path))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    return run_loop(args.control, args.status, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
