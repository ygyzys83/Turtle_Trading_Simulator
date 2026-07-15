from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from agentloop_trader.assets import normalize_asset_class
from agentloop_trader.fees import estimate_alpaca_order_fees
from agentloop_trader.models import PACIFIC_TIME


@dataclass(frozen=True)
class PaperSessionSnapshot:
    session_id: str
    started_at: str
    mode: str
    paper_cash: float
    paper_equity: float
    session_pnl: float
    local_orders: list[dict]
    local_positions: list[dict]
    tracked_alpaca_orders: list[dict]
    audit_records: list[dict]


def new_session_id() -> str:
    return f"paper-{datetime.now(PACIFIC_TIME).strftime('%Y%m%d-%H%M%S')}-{str(uuid4())[:8]}"


def session_summary_records(snapshot: PaperSessionSnapshot) -> list[dict]:
    event_types = [record.get("event_type", "") for record in snapshot.audit_records]
    local_filled = [order for order in snapshot.local_orders if _enum_value(order.get("Status")) == "filled"]
    broker_orders = _broker_order_records(snapshot.tracked_alpaca_orders)
    alpaca_statuses = [_enum_value(order.get("status", "")) for order in broker_orders]
    return [
        {"Metric": "Session ID", "Value": snapshot.session_id},
        {"Metric": "Started", "Value": snapshot.started_at},
        {"Metric": "Mode", "Value": snapshot.mode},
        {"Metric": "Activity records", "Value": len(snapshot.audit_records)},
        {"Metric": "App paper orders", "Value": len(snapshot.local_orders)},
        {"Metric": "Filled app paper orders", "Value": len(local_filled)},
        {"Metric": "Open app paper positions", "Value": len(snapshot.local_positions)},
        {"Metric": "Saved Alpaca orders", "Value": len(broker_orders)},
        {"Metric": "Filled Alpaca orders", "Value": alpaca_statuses.count("filled")},
        {"Metric": "Canceled Alpaca orders", "Value": alpaca_statuses.count("canceled") + alpaca_statuses.count("cancelled")},
        {"Metric": "Paper buys sent", "Value": event_types.count("alpaca_paper_order_submitted") + event_types.count("auto_paper_entry_submitted") + event_types.count("worker_paper_buy_sent")},
        {"Metric": "Paper cancels sent", "Value": event_types.count("alpaca_paper_cancel_submitted") + event_types.count("worker_limit_buy_cancelled") + event_types.count("worker_rsi_late_buy_cancelled")},
        {"Metric": "Paper exits sent", "Value": event_types.count("alpaca_paper_exit_submitted") + event_types.count("auto_paper_exit_submitted") + event_types.count("worker_paper_exit_sent")},
    ]


def session_timeline_records(audit_records: list[dict], limit: int = 100) -> list[dict]:
    rows = []
    for record in audit_records[-limit:]:
        payload = record.get("payload", {}) if isinstance(record.get("payload", {}), dict) else {}
        rows.append(
            {
                "Time": record.get("created_at", ""),
                "Record Type": record.get("event_type", ""),
                "Symbol": payload.get("symbol", payload.get("Symbol", "")),
                "Side": payload.get("side", payload.get("Side", "")),
                "Quantity": payload.get("quantity", payload.get("Quantity", "")),
                "Status": payload.get("status", payload.get("cancel_status", "")),
                "Review ID": payload.get("preview_hash", payload.get("cancel_preview_hash", "")),
                "Message": record.get("message", ""),
            }
        )
    return rows


def automatic_exit_records(
    audit_records: list[dict],
    tracked_orders: list[dict] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Explain recent automatic exits without requiring the raw activity log."""
    tracked_by_id = {
        str(order.get("broker_order_id", "")): order
        for order in (tracked_orders or [])
        if str(order.get("broker_order_id", ""))
    }
    rows: list[dict[str, Any]] = []
    automatic_types = {"worker_paper_exit_sent", "auto_paper_exit_submitted"}
    for record in reversed(audit_records):
        if record.get("event_type") not in automatic_types:
            continue
        payload = record.get("payload", {}) if isinstance(record.get("payload"), dict) else {}
        details = payload.get("exit_details", {}) if isinstance(payload.get("exit_details"), dict) else {}
        broker_order_id = str(payload.get("broker_order_id", ""))
        tracked = tracked_by_id.get(broker_order_id, {})
        rows.append(
            {
                "Time": record.get("created_at", ""),
                "Ticker": payload.get("symbol", ""),
                "Shares": _format_number(_as_float(payload.get("quantity"))),
                "Decision Price": _optional_money(details.get("current_price")),
                "Sell Trigger": _optional_money(details.get("trigger_price")),
                "Trigger Rule": details.get("trigger_source") or "Not recorded by older in-page exit",
                "Position Cycle": payload.get("position_cycle_id", "Not recorded"),
                "Average Entry": _optional_money(payload.get("position_average_entry")),
                "Current Profit in R": _optional_r(details.get("profit_r")),
                "Highest Profit in R": _optional_r(details.get("highest_profit_r")),
                "Alpaca Fill": _optional_money(tracked.get("average_fill_price")),
                "Reason": payload.get("reason") or record.get("message", ""),
                "Alpaca Order ID": broker_order_id,
            }
        )
        if len(rows) >= max(1, int(limit)):
            break
    return rows


def paper_performance_records(snapshot: PaperSessionSnapshot) -> list[dict]:
    local_filled = [order for order in snapshot.local_orders if _enum_value(order.get("Status")) == "filled"]
    local_notional = sum(_as_float(order.get("Notional")) for order in local_filled)
    open_local_notional = sum(_as_float(position.get("Book Value")) for position in snapshot.local_positions)
    return [
        {"Metric": "Simulator cash", "Value": _money(snapshot.paper_cash)},
        {"Metric": "Simulator equity", "Value": _money(snapshot.paper_equity)},
        {"Metric": "Session P&L", "Value": _money(snapshot.session_pnl)},
        {"Metric": "Filled app order value", "Value": _money(local_notional)},
        {"Metric": "Open app position value", "Value": _money(open_local_notional)},
        {"Metric": "Filled simulator orders", "Value": len(local_filled)},
        {"Metric": "Open simulator positions", "Value": len(snapshot.local_positions)},
    ]


def alpaca_paper_activity_records(snapshot: PaperSessionSnapshot) -> list[dict]:
    broker_orders = _broker_order_records(snapshot.tracked_alpaca_orders)
    alpaca_filled = [order for order in broker_orders if _enum_value(order.get("status")) == "filled"]
    alpaca_filled_qty = sum(_as_float(order.get("filled_quantity") or order.get("quantity")) for order in alpaca_filled)
    alpaca_canceled = [
        order for order in broker_orders
        if _enum_value(order.get("status")) in {"canceled", "cancelled"}
    ]
    alpaca_waiting = [
        order for order in broker_orders
        if _enum_value(order.get("status")) in {"accepted", "new", "pending_new", "partially_filled"}
    ]
    estimated_fees, fee_orders, missing_fee_orders = _filled_order_fee_summary(alpaca_filled)
    fee_value = _money(estimated_fees) if fee_orders else "Not available for saved fills"
    if fee_orders and missing_fee_orders:
        fee_value += f" ({missing_fee_orders} older fill(s) missing details)"
    return [
        {"Metric": "Saved Alpaca orders", "Value": len(broker_orders)},
        {"Metric": "Filled Alpaca shares", "Value": _format_number(alpaca_filled_qty)},
        {"Metric": "Filled Alpaca orders", "Value": len(alpaca_filled)},
        {"Metric": "Canceled Alpaca orders", "Value": len(alpaca_canceled)},
        {"Metric": "Alpaca orders waiting to fill", "Value": len(alpaca_waiting)},
        {"Metric": "Estimated live fees on filled orders", "Value": fee_value},
    ]


def paper_trading_review_records(
    snapshot: PaperSessionSnapshot,
    alpaca_position_count: int = 0,
    alpaca_account_value: float | None = None,
) -> list[dict]:
    event_types = [record.get("event_type", "") for record in snapshot.audit_records]
    buy_count = (
        event_types.count("alpaca_paper_order_submitted")
        + event_types.count("auto_paper_entry_submitted")
        + event_types.count("worker_paper_buy_sent")
    )
    exit_count = (
        event_types.count("alpaca_paper_exit_submitted")
        + event_types.count("auto_paper_exit_submitted")
        + event_types.count("worker_paper_exit_sent")
    )
    cancel_count = (
        event_types.count("alpaca_paper_cancel_submitted")
        + event_types.count("worker_limit_buy_cancelled")
        + event_types.count("worker_rsi_late_buy_cancelled")
    )
    blocked_count = sum(1 for event_type in event_types if "blocked" in event_type or "rejected" in event_type)
    broker_orders = _broker_order_records(snapshot.tracked_alpaca_orders)
    waiting_orders = [
        order for order in broker_orders
        if _enum_value(order.get("status")) in {"accepted", "new", "pending_new", "partially_filled"}
    ]
    account_value = _money(alpaca_account_value) if alpaca_account_value is not None else "Not connected"
    alpaca_filled = [order for order in broker_orders if _enum_value(order.get("status")) == "filled"]
    estimated_fees, fee_orders, _ = _filled_order_fee_summary(alpaca_filled)
    fee_read = _money(estimated_fees) if fee_orders else "Not available"
    if blocked_count:
        next_step = "Review blocked orders before changing automation."
    elif waiting_orders:
        next_step = "Watch or cancel waiting limit orders."
    elif alpaca_position_count:
        next_step = "Let saved exit settings manage open positions."
    else:
        next_step = "Find the next clean setup or review today."
    return [
        {"Area": "Account", "Read": account_value, "Plain English": "Current Alpaca paper account value when connected."},
        {"Area": "Open positions", "Read": alpaca_position_count, "Plain English": "Current Alpaca paper positions being managed."},
        {"Area": "Waiting orders", "Read": len(waiting_orders), "Plain English": "Limit orders still waiting at Alpaca."},
        {"Area": "Paper buys sent", "Read": buy_count, "Plain English": "Manual and automatic paper buys recorded this session."},
        {"Area": "Paper exits sent", "Read": exit_count, "Plain English": "Manual and automatic paper exits recorded this session."},
        {"Area": "Paper cancels sent", "Read": cancel_count, "Plain English": "Paper order cancels recorded this session."},
        {"Area": "Issues to review", "Read": blocked_count, "Plain English": "Blocked or rejected order events recorded this session."},
        {
            "Area": "Estimated live fees",
            "Read": fee_read,
            "Plain English": "What current Alpaca trading fees would approximately cost. Alpaca paper may not deduct the same amount.",
        },
        {"Area": "Next step", "Read": next_step, "Plain English": "The useful action from here."},
    ]


def _filled_order_fee_summary(orders: list[dict]) -> tuple[float, int, int]:
    total = 0.0
    estimated_count = 0
    missing_count = 0
    for order in orders:
        side = _enum_value(order.get("side", ""))
        quantity = _as_float(order.get("filled_quantity") or order.get("quantity"))
        price = _as_float(order.get("average_fill_price") or order.get("filled_avg_price"))
        if side not in {"buy", "sell"} or quantity <= 0 or price <= 0:
            missing_count += 1
            continue
        total += estimate_alpaca_order_fees(
            asset_class=normalize_asset_class(order.get("asset_class"), order.get("symbol", "")),
            side=side,
            quantity=quantity,
            price=price,
        ).total
        estimated_count += 1
    return round(total, 2), estimated_count, missing_count


def paper_testing_progress_records(
    audit_records: list[dict],
    tracked_alpaca_orders: list[dict],
    target_days: int = 10,
) -> list[dict]:
    paper_event_types = {
        "alpaca_paper_order_submitted",
        "auto_paper_entry_submitted",
        "worker_paper_buy_sent",
        "alpaca_paper_exit_submitted",
        "auto_paper_exit_submitted",
        "worker_paper_exit_sent",
        "alpaca_paper_cancel_submitted",
        "worker_limit_buy_cancelled",
        "worker_rsi_late_buy_cancelled",
    }
    paper_records = [record for record in audit_records if record.get("event_type") in paper_event_types]
    active_dates = sorted(
        {
            str(record.get("created_at", ""))[:10]
            for record in paper_records
            if str(record.get("created_at", ""))[:10]
        }
    )
    event_types = [record.get("event_type", "") for record in audit_records]
    buy_count = event_types.count("alpaca_paper_order_submitted") + event_types.count("auto_paper_entry_submitted") + event_types.count("worker_paper_buy_sent")
    exit_count = event_types.count("alpaca_paper_exit_submitted") + event_types.count("auto_paper_exit_submitted") + event_types.count("worker_paper_exit_sent")
    blocked_count = sum(1 for event_type in event_types if "blocked" in event_type or "rejected" in event_type)
    filled_orders = [
        order for order in _broker_order_records(tracked_alpaca_orders)
        if _enum_value(order.get("status")) == "filled"
    ]
    ready_for_review = len(active_dates) >= target_days and buy_count > 0 and exit_count > 0 and blocked_count == 0
    return [
        {"Check": "Paper trading days", "Read": f"{len(active_dates)}/{target_days}", "Plain English": "Days with recorded paper order activity."},
        {"Check": "Paper buys tested", "Read": buy_count, "Plain English": "Paper buy orders sent through Alpaca."},
        {"Check": "Paper exits tested", "Read": exit_count, "Plain English": "Paper exit orders sent through Alpaca."},
        {"Check": "Filled paper orders", "Read": len(filled_orders), "Plain English": "Saved Alpaca paper orders marked filled."},
        {"Check": "Issues to review", "Read": blocked_count, "Plain English": "Blocked or rejected order records to inspect."},
        {
            "Check": "Testing status",
            "Read": "Ready to review live setup" if ready_for_review else "Keep paper testing",
            "Plain English": "Review live setup only after enough paper days, at least one buy, at least one exit, and no unresolved issues.",
        },
    ]


def _enum_value(value: Any) -> str:
    text = str(value or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.strip().lower()


def _broker_order_records(records: list[dict]) -> list[dict]:
    return [
        order for order in records
        if str(order.get("source") or "").strip().lower()
        not in {"position_plan", "position_observation"}
    ]


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.8f}".rstrip("0").rstrip(".")


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _optional_money(value: Any) -> str:
    if value in (None, ""):
        return "Not recorded"
    try:
        return _money(float(value))
    except (TypeError, ValueError):
        return "Not recorded"


def _optional_r(value: Any) -> str:
    if value in (None, ""):
        return "Not recorded"
    try:
        return f"{float(value):.2f}R"
    except (TypeError, ValueError):
        return "Not recorded"
