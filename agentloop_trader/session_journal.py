from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


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
    return f"paper-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{str(uuid4())[:8]}"


def session_summary_records(snapshot: PaperSessionSnapshot) -> list[dict]:
    event_types = [record.get("event_type", "") for record in snapshot.audit_records]
    local_filled = [order for order in snapshot.local_orders if _enum_value(order.get("Status")) == "filled"]
    alpaca_statuses = [_enum_value(order.get("status", "")) for order in snapshot.tracked_alpaca_orders]
    return [
        {"Metric": "Session ID", "Value": snapshot.session_id},
        {"Metric": "Started At", "Value": snapshot.started_at},
        {"Metric": "Mode", "Value": snapshot.mode},
        {"Metric": "Audit Events", "Value": len(snapshot.audit_records)},
        {"Metric": "Local Paper Orders", "Value": len(snapshot.local_orders)},
        {"Metric": "Local Filled Orders", "Value": len(local_filled)},
        {"Metric": "Local Open Positions", "Value": len(snapshot.local_positions)},
        {"Metric": "Tracked Alpaca Orders", "Value": len(snapshot.tracked_alpaca_orders)},
        {"Metric": "Tracked Alpaca Filled Orders", "Value": alpaca_statuses.count("filled")},
        {"Metric": "Tracked Alpaca Canceled Orders", "Value": alpaca_statuses.count("canceled") + alpaca_statuses.count("cancelled")},
        {"Metric": "Alpaca Submit Events", "Value": event_types.count("alpaca_paper_order_submitted")},
        {"Metric": "Alpaca Cancel Events", "Value": event_types.count("alpaca_paper_cancel_submitted")},
        {"Metric": "Alpaca Exit Events", "Value": event_types.count("alpaca_paper_exit_submitted")},
    ]


def session_timeline_records(audit_records: list[dict], limit: int = 100) -> list[dict]:
    rows = []
    for record in audit_records[-limit:]:
        payload = record.get("payload", {}) if isinstance(record.get("payload", {}), dict) else {}
        rows.append(
            {
                "Created At": record.get("created_at", ""),
                "Event Type": record.get("event_type", ""),
                "Symbol": payload.get("symbol", payload.get("Symbol", "")),
                "Side": payload.get("side", payload.get("Side", "")),
                "Quantity": payload.get("quantity", payload.get("Quantity", "")),
                "Status": payload.get("status", payload.get("cancel_status", "")),
                "Preview Hash": payload.get("preview_hash", payload.get("cancel_preview_hash", "")),
                "Message": record.get("message", ""),
            }
        )
    return rows


def paper_performance_records(snapshot: PaperSessionSnapshot) -> list[dict]:
    local_filled = [order for order in snapshot.local_orders if _enum_value(order.get("Status")) == "filled"]
    local_notional = sum(_as_float(order.get("Notional")) for order in local_filled)
    alpaca_filled = [order for order in snapshot.tracked_alpaca_orders if _enum_value(order.get("status")) == "filled"]
    alpaca_filled_qty = sum(_as_float(order.get("filled_quantity") or order.get("quantity")) for order in alpaca_filled)
    alpaca_canceled = [
        order for order in snapshot.tracked_alpaca_orders
        if _enum_value(order.get("status")) in {"canceled", "cancelled"}
    ]
    open_local_notional = sum(_as_float(position.get("Book Value")) for position in snapshot.local_positions)
    return [
        {"Metric": "Paper Cash", "Value": _money(snapshot.paper_cash)},
        {"Metric": "Paper Equity", "Value": _money(snapshot.paper_equity)},
        {"Metric": "Session P&L", "Value": _money(snapshot.session_pnl)},
        {"Metric": "Local Filled Notional", "Value": _money(local_notional)},
        {"Metric": "Local Open Position Notional", "Value": _money(open_local_notional)},
        {"Metric": "Tracked Alpaca Filled Quantity", "Value": _format_number(alpaca_filled_qty)},
        {"Metric": "Tracked Alpaca Filled Orders", "Value": len(alpaca_filled)},
        {"Metric": "Tracked Alpaca Canceled Orders", "Value": len(alpaca_canceled)},
        {"Metric": "Tracked Alpaca Open Orders", "Value": sum(_enum_value(order.get("status")) in {"accepted", "new", "pending_new", "partially_filled"} for order in snapshot.tracked_alpaca_orders)},
    ]


def _enum_value(value: Any) -> str:
    text = str(value or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.strip().lower()


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
