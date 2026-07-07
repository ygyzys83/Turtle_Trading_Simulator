from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from agentloop_trader.brokers import AlpacaConfig, AlpacaOrderPreview, build_alpaca_order_preview
from agentloop_trader.models import ExecutionDecision, PACIFIC_TIME, RiskCheckResult, TradeIntent


DEFAULT_BROKER_STATE_PATH = Path("broker_state") / "alpaca_paper_orders.json"


@dataclass(frozen=True)
class BrokerStateHealth:
    ready: bool
    stale: bool
    reasons: list[str]


OPEN_ORDER_STATUSES = {"accepted", "new", "pending_new", "partially_filled"}
TERMINAL_ORDER_STATUSES = {"filled", "canceled", "cancelled", "expired", "rejected"}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_symbol_set(position_records: list[dict]) -> set[str]:
    return {str(row.get("Symbol", "")).strip().upper() for row in position_records if row.get("Symbol")}


def _enum_value(value: Any) -> str:
    text = str(value or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.strip().lower()


def broker_state_health(alpaca_connected: bool, position_records: list[dict] | None, order_records: list[dict] | None) -> BrokerStateHealth:
    reasons = []
    if not alpaca_connected:
        reasons.append("Alpaca paper account is not connected.")
    if position_records is None:
        reasons.append("Alpaca positions could not be refreshed.")
    if order_records is None:
        reasons.append("Alpaca orders could not be refreshed.")
    return BrokerStateHealth(ready=not reasons, stale=bool(reasons), reasons=reasons)


def duplicate_exposure_reasons(intent: TradeIntent | None, alpaca_positions: list[dict], allow_duplicate: bool = False) -> list[str]:
    if allow_duplicate or intent is None or intent.side != "buy":
        return []
    symbols = _as_symbol_set(alpaca_positions)
    if intent.symbol_clean in symbols:
        return [f"Alpaca paper already has an open {intent.symbol_clean} position."]
    return []


def preview_already_tracked(preview_hash: str, tracked_orders: list[dict]) -> bool:
    active_statuses = {"", "accepted", "new", "pending_new", "partially_filled", "filled"}
    for order in tracked_orders:
        if order.get("preview_hash") == preview_hash and _enum_value(order.get("status", "")) in active_statuses:
            return True
    return False


def open_order_exposure_reasons(intent: TradeIntent | None, alpaca_orders: list[dict], allow_duplicate: bool = False) -> list[str]:
    if allow_duplicate or intent is None or intent.side != "buy":
        return []
    reasons = []
    for order in alpaca_orders:
        symbol = str(order.get("Symbol", "")).strip().upper()
        side = _enum_value(order.get("Side", ""))
        status = _enum_value(order.get("Status", ""))
        if symbol == intent.symbol_clean and side == "buy" and status in OPEN_ORDER_STATUSES:
            reasons.append(f"Alpaca Orders already has an open {intent.symbol_clean} buy order with status {status}.")
    return list(dict.fromkeys(reasons))


def exit_position_reasons(exit_preview: AlpacaOrderPreview | None, position_records: list[dict]) -> list[str]:
    if exit_preview is None:
        return ["No Alpaca paper exit preview is selected."]
    symbol = str(exit_preview.order.get("symbol", "")).strip().upper()
    quantity = _as_float(exit_preview.order.get("quantity"))
    for position in position_records:
        if str(position.get("Symbol", "")).strip().upper() != symbol:
            continue
        position_qty = _as_float(position.get("Quantity"))
        if position_qty <= 0:
            return [f"Alpaca paper has no open {symbol} position to exit."]
        if quantity > position_qty:
            return [f"Exit quantity {quantity:g} exceeds Alpaca paper {symbol} position quantity {position_qty:g}."]
        return []
    return [f"Alpaca paper has no open {symbol} position to exit."]


def open_exit_order_reasons(exit_preview: AlpacaOrderPreview | None, alpaca_orders: list[dict]) -> list[str]:
    if exit_preview is None:
        return []
    symbol = str(exit_preview.order.get("symbol", "")).strip().upper()
    reasons = []
    for order in alpaca_orders:
        order_symbol = str(order.get("Symbol", "")).strip().upper()
        side = _enum_value(order.get("Side", ""))
        status = _enum_value(order.get("Status", ""))
        if order_symbol == symbol and side == "sell" and status in OPEN_ORDER_STATUSES:
            reasons.append(f"Alpaca Orders already has an open {symbol} sell order with status {status}.")
    return list(dict.fromkeys(reasons))


def cancelable_alpaca_order_records(alpaca_orders: list[dict]) -> list[dict]:
    rows = []
    for order in alpaca_orders:
        status = _enum_value(order.get("Status", ""))
        broker_order_id = str(order.get("Alpaca Order ID") or order.get("Broker Order ID") or "").strip()
        if status not in OPEN_ORDER_STATUSES or not broker_order_id:
            continue
        rows.append(
            {
                "Order ID": order.get("Order ID", ""),
                "Alpaca Order ID": broker_order_id,
                "Symbol": str(order.get("Symbol", "")).strip().upper(),
                "Side": _enum_value(order.get("Side", "")),
                "Quantity": order.get("Quantity", ""),
                "Status": status,
                "Submitted": order.get("Submitted", ""),
            }
        )
    return rows


def refresh_tracked_alpaca_orders(tracked_orders: list[dict], alpaca_orders: list[dict]) -> list[dict]:
    now = datetime.now(PACIFIC_TIME).isoformat()
    alpaca_by_id = {
        str(order.get("Alpaca Order ID") or order.get("Broker Order ID") or "").strip(): order
        for order in alpaca_orders
        if order.get("Alpaca Order ID") or order.get("Broker Order ID")
    }
    refreshed = []
    for tracked in tracked_orders:
        record = dict(tracked)
        broker_order_id = str(record.get("broker_order_id") or record.get("Alpaca Order ID") or record.get("Broker Order ID") or "").strip()
        alpaca_order = alpaca_by_id.get(broker_order_id)
        if record.get("adopted") and record.get("source") == "adopted_alpaca_position":
            record["lifecycle_status"] = "adopted_alpaca_position"
        elif not broker_order_id:
            record["lifecycle_status"] = "missing_broker_order_id"
        elif alpaca_order is None:
            record["lifecycle_status"] = "missing_from_alpaca_orders"
        else:
            status = _enum_value(alpaca_order.get("Status", ""))
            record.update(
                {
                    "broker_order_id": broker_order_id,
                    "symbol": str(alpaca_order.get("Symbol", record.get("symbol", ""))).strip().upper(),
                    "side": _enum_value(alpaca_order.get("Side", record.get("side", ""))),
                    "quantity": str(alpaca_order.get("Quantity", record.get("quantity", ""))).strip(),
                    "status": status,
                    "alpaca_status_raw": str(alpaca_order.get("Status", "")).strip(),
                    "submitted_at": str(alpaca_order.get("Submitted", record.get("submitted_at", ""))).strip(),
                    "filled_at": str(alpaca_order.get("Filled", record.get("filled_at", ""))).strip(),
                    "filled_quantity": str(alpaca_order.get("Filled Qty", record.get("filled_quantity", ""))).strip(),
                    "average_fill_price": str(alpaca_order.get("Avg Fill", record.get("average_fill_price", ""))).strip(),
                    "lifecycle_status": _lifecycle_status(status),
                }
            )
        record["last_synced_at"] = now
        refreshed.append(record)
    return refreshed


def alpaca_order_lifecycle_records(tracked_orders: list[dict], alpaca_orders: list[dict]) -> list[dict]:
    refreshed = refresh_tracked_alpaca_orders(tracked_orders, alpaca_orders)
    return [
        {
            "Order ID": str(order.get("broker_order_id", ""))[:8],
            "Alpaca Order ID": order.get("broker_order_id", ""),
            "Symbol": order.get("symbol", ""),
            "Side": order.get("side", ""),
            "Quantity": order.get("quantity", ""),
            "Alpaca Status": _enum_value(order.get("status", "")),
            "Tracking Status": _display_order_status(str(order.get("lifecycle_status", ""))),
            "Filled Qty": order.get("filled_quantity", ""),
            "Avg Fill": order.get("average_fill_price", ""),
            "Last Updated": order.get("last_synced_at", ""),
        }
        for order in refreshed
    ]


def alpaca_order_lifecycle_summary_records(tracked_orders: list[dict]) -> list[dict]:
    statuses = [_enum_value(order.get("status", "")) for order in tracked_orders]
    lifecycle_statuses = [str(order.get("lifecycle_status", "")) for order in tracked_orders]
    return [
        {"Metric": "Saved Alpaca orders", "Value": len(tracked_orders)},
        {"Metric": "Open orders at Alpaca", "Value": lifecycle_statuses.count("open_at_alpaca")},
        {"Metric": "Filled orders at Alpaca", "Value": statuses.count("filled")},
        {"Metric": "Positions manually added to app", "Value": lifecycle_statuses.count("adopted_alpaca_position")},
        {"Metric": "Canceled orders at Alpaca", "Value": statuses.count("canceled") + statuses.count("cancelled")},
        {"Metric": "Rejected orders at Alpaca", "Value": statuses.count("rejected")},
        {"Metric": "Saved orders missing at Alpaca", "Value": lifecycle_statuses.count("missing_from_alpaca_orders")},
    ]


def alpaca_position_lifecycle_records(position_records: list[dict], tracked_orders: list[dict]) -> list[dict]:
    filled_buy_orders = [
        order for order in tracked_orders
        if _enum_value(order.get("status", "")) == "filled" and _enum_value(order.get("side", "")) == "buy"
    ]
    filled_by_symbol: dict[str, list[dict]] = {}
    for order in filled_buy_orders:
        symbol = str(order.get("symbol", "")).strip().upper()
        if symbol:
            filled_by_symbol.setdefault(symbol, []).append(order)

    rows = []
    position_symbols = set()
    for position in position_records:
        symbol = str(position.get("Symbol", "")).strip().upper()
        if not symbol:
            continue
        position_symbols.add(symbol)
        matched_orders = filled_by_symbol.get(symbol, [])
        filled_qty = sum(_as_float(order.get("filled_quantity") or order.get("quantity")) for order in matched_orders)
        latest_order = matched_orders[-1] if matched_orders else {}
        position_qty = _as_float(position.get("Quantity"))
        adopted_match = bool(matched_orders) and all(_is_adopted_position_order(order) for order in matched_orders)
        tracking_status = (
            "adopted_alpaca_position"
            if adopted_match
            else "position_matched_to_filled_order"
            if matched_orders
            else "untracked_alpaca_position"
        )
        rows.append(
            {
                "Symbol": symbol,
                "Position Qty": position.get("Quantity", ""),
                "Position Value": position.get("Market Value", ""),
                "Avg Entry": position.get("Average Entry", ""),
                "Tracked Order Qty": _format_number(filled_qty) if matched_orders else "",
                "Tracked Avg Fill": latest_order.get("average_fill_price", ""),
                "Matched Saved Orders": len(matched_orders),
                "Exit Ready": position_qty > 0,
                "Tracking Status": _display_position_status(tracking_status),
            }
        )

    for symbol, orders in sorted(filled_by_symbol.items()):
        if symbol in position_symbols:
            continue
        filled_qty = sum(_as_float(order.get("filled_quantity") or order.get("quantity")) for order in orders)
        latest_order = orders[-1]
        rows.append(
            {
                "Symbol": symbol,
                "Position Qty": "",
                "Position Value": "",
                "Avg Entry": "",
                "Tracked Order Qty": _format_number(filled_qty),
                "Tracked Avg Fill": latest_order.get("average_fill_price", ""),
                "Matched Saved Orders": len(orders),
                "Exit Ready": False,
                "Tracking Status": _display_position_status("filled_order_without_open_position"),
            }
        )
    return rows


def alpaca_position_lifecycle_summary_records(position_lifecycle_rows: list[dict]) -> list[dict]:
    statuses = [str(row.get("Tracking Status", "")) for row in position_lifecycle_rows]
    return [
        {"Metric": "Open Alpaca positions", "Value": sum(bool(row.get("Position Qty")) for row in position_lifecycle_rows)},
        {"Metric": "Positions matched to app orders", "Value": statuses.count("Matched to app order")},
        {"Metric": "Positions manually added to app", "Value": statuses.count("Tracked manually")},
        {"Metric": "Positions needing app tracking", "Value": statuses.count("Needs app tracking")},
        {"Metric": "Filled app orders without a position", "Value": statuses.count("No open Alpaca position")},
        {"Metric": "Exit orders ready to review", "Value": sum(bool(row.get("Exit Ready")) for row in position_lifecycle_rows)},
    ]


def _lifecycle_status(status: str) -> str:
    if status in OPEN_ORDER_STATUSES:
        return "open_at_alpaca"
    if status in {"filled"}:
        return "filled_at_alpaca"
    if status in {"canceled", "cancelled"}:
        return "canceled_at_alpaca"
    if status in TERMINAL_ORDER_STATUSES:
        return "terminal_at_alpaca"
    return "unknown_at_alpaca"


def _display_order_status(status: str) -> str:
    return {
        "open_at_alpaca": "Open at Alpaca",
        "filled_at_alpaca": "Filled at Alpaca",
        "canceled_at_alpaca": "Canceled at Alpaca",
        "terminal_at_alpaca": "Closed at Alpaca",
        "missing_from_alpaca_orders": "Saved in app, missing at Alpaca",
        "adopted_alpaca_position": "Position added to app",
        "unknown_at_alpaca": "Unknown at Alpaca",
    }.get(status, status or "")


def _display_position_status(status: str) -> str:
    return {
        "position_matched_to_filled_order": "Matched to app order",
        "adopted_alpaca_position": "Tracked manually",
        "untracked_alpaca_position": "Needs app tracking",
        "filled_order_without_open_position": "No open Alpaca position",
    }.get(status, status or "")


def _is_adopted_position_order(order: dict) -> bool:
    return bool(order.get("adopted")) and str(order.get("source", "")) == "adopted_alpaca_position"


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.8f}".rstrip("0").rstrip(".")


def reconcile_alpaca_positions(position_records: list[dict], tracked_orders: list[dict]) -> list[dict]:
    tracked_symbols = {str(order.get("symbol", "")).strip().upper() for order in tracked_orders}
    rows = []
    for position in position_records:
        symbol = str(position.get("Symbol", "")).strip().upper()
        rows.append(
            {
                "Symbol": symbol,
                "Alpaca Qty": position.get("Quantity", ""),
                "Market Value": position.get("Market Value", ""),
                "Tracked By App": symbol in tracked_symbols,
                "Status": "matched" if symbol in tracked_symbols else "unmatched_alpaca_position",
            }
        )
    for symbol in sorted(s for s in tracked_symbols if s not in _as_symbol_set(position_records)):
        rows.append(
            {
                "Symbol": symbol,
                "Alpaca Qty": "",
                "Market Value": "",
                "Tracked By App": True,
                "Status": "tracked_order_without_position",
            }
        )
    return rows


def adopt_alpaca_position(position: dict, adopted_at: datetime | None = None) -> dict:
    symbol = str(position.get("Symbol", "")).strip().upper()
    quantity = _as_float(position.get("Quantity"))
    average_entry = _as_float(position.get("Average Entry"))
    if not symbol:
        raise ValueError("Cannot adopt an Alpaca position without a symbol.")
    if quantity <= 0:
        raise ValueError("Cannot adopt an Alpaca position without positive quantity.")
    if adopted_at is None:
        now_dt = datetime.now(PACIFIC_TIME)
    elif adopted_at.tzinfo:
        now_dt = adopted_at.astimezone(PACIFIC_TIME)
    else:
        now_dt = adopted_at.replace(tzinfo=PACIFIC_TIME)
    now = now_dt.isoformat()
    compact_time = now_dt.strftime("%Y%m%d%H%M%S")
    return {
        "broker_order_id": f"adopted-{symbol}-{compact_time}",
        "preview_hash": "",
        "symbol": symbol,
        "side": "buy",
        "quantity": _format_number(quantity),
        "status": "filled",
        "alpaca_status_raw": "adopted_position",
        "submitted_at": "",
        "filled_at": now,
        "filled_quantity": _format_number(quantity),
        "average_fill_price": _format_number(average_entry),
        "lifecycle_status": "adopted_alpaca_position",
        "last_synced_at": now,
        "source": "adopted_alpaca_position",
        "adopted": True,
        "broker_writes_submitted": 0,
    }


def build_exit_intent_from_position(position: dict) -> TradeIntent | None:
    symbol = str(position.get("Symbol", "")).strip().upper()
    qty = int(abs(_as_float(position.get("Quantity"))))
    if not symbol or qty <= 0:
        return None
    return TradeIntent(
        symbol=symbol,
        side="sell",
        quantity=qty,
        order_type="market",
        time_in_force="day",
        rationale="Exit preview generated from existing Alpaca paper position.",
        source_signals=["alpaca_position_exit_preview"],
    )


def build_exit_order_previews(position_records: list[dict], config: AlpacaConfig) -> list[AlpacaOrderPreview]:
    approved_risk = RiskCheckResult(approved=True, rejected_reasons=[], checks={"exit_preview": True})
    approved_decision = ExecutionDecision(
        mode="paper",
        approved_for_execution=True,
        requires_manual_approval=False,
        reason="Exit preview only; paper submission remains gated.",
        risk_check=approved_risk,
    )
    previews = []
    for position in position_records:
        intent = build_exit_intent_from_position(position)
        if intent is not None:
            previews.append(build_alpaca_order_preview(intent, approved_decision, config))
    return previews


def simulated_alpaca_fill_order(
    tracked_order: dict | None,
    fill_price: float | None = None,
    filled_at: datetime | None = None,
) -> dict:
    record = dict(tracked_order or {})
    quantity = _as_float(record.get("filled_quantity") or record.get("quantity"))
    price = fill_price if fill_price is not None else _as_float(record.get("average_fill_price") or record.get("entry_price"))
    now_dt = None
    if filled_at is not None:
        now_dt = filled_at.astimezone(PACIFIC_TIME) if filled_at.tzinfo else filled_at.replace(tzinfo=PACIFIC_TIME)
    now = (now_dt or datetime.now(PACIFIC_TIME)).isoformat()
    record.update(
        {
            "status": "filled",
            "alpaca_status_raw": "simulated_fill",
            "filled_at": now,
            "filled_quantity": _format_number(quantity),
            "average_fill_price": _format_number(price),
            "lifecycle_status": "filled_at_alpaca",
            "last_synced_at": now,
            "simulated": True,
        }
    )
    return record


def simulated_position_from_filled_order(filled_order: dict) -> dict:
    quantity = _as_float(filled_order.get("filled_quantity") or filled_order.get("quantity"))
    price = _as_float(filled_order.get("average_fill_price"))
    return {
        "Symbol": str(filled_order.get("symbol", "")).strip().upper(),
        "Quantity": _format_number(quantity),
        "Market Value": round(quantity * price, 2),
        "Average Entry": _format_number(price),
        "Source": "simulated_fill",
    }


def simulated_exit_preview_readiness_records(
    tracked_order: dict | None,
    config: AlpacaConfig,
) -> list[dict]:
    if not tracked_order:
        return [{"Check": "Filled order selected", "Passed": False, "Detail": "Select a saved order before practicing the exit check."}]
    filled_order = simulated_alpaca_fill_order(tracked_order)
    position = simulated_position_from_filled_order(filled_order)
    previews = build_exit_order_previews([position], config)
    preview = previews[0] if previews else None
    blockers = exit_position_reasons(preview, [position]) if preview else ["No simulated exit preview was generated."]
    return [
        {"Check": "Filled order selected", "Passed": True, "Detail": filled_order.get("broker_order_id", "")},
        {"Check": "Practice position created", "Passed": bool(position.get("Symbol")), "Detail": position.get("Symbol", "")},
        {"Check": "Exit order ready", "Passed": bool(preview and preview.valid), "Detail": "" if preview and preview.valid else "; ".join(preview.blocked_reasons if preview else blockers)},
        {"Check": "Position can be exited", "Passed": not blockers, "Detail": "Practice position can be exited." if not blockers else "; ".join(blockers)},
        {"Check": "Orders sent", "Passed": True, "Detail": "0"},
    ]


def exit_preview_records(previews: list[AlpacaOrderPreview]) -> list[dict]:
    return [
        {
            "Symbol": preview.order.get("symbol", ""),
            "Side": preview.order.get("side", ""),
            "Quantity": preview.order.get("quantity", 0),
            "Mode": preview.order.get("mode", ""),
            "Review ID": preview.preview_hash,
            "Valid": preview.valid,
            "Blocked Reasons": "; ".join(preview.blocked_reasons),
        }
        for preview in previews
    ]


def market_session_advisory(now: datetime | None = None) -> dict:
    eastern = ZoneInfo("America/New_York")
    current = (now or datetime.now(UTC)).astimezone(eastern)
    is_weekday = current.weekday() < 5
    regular_open = time(9, 30) <= current.time() <= time(16, 0)
    open_now = is_weekday and regular_open
    message = "Regular US equity session is open." if open_now else "Regular US equity session appears closed; paper market orders may queue."
    return {
        "Market Session": "open" if open_now else "closed_or_extended",
        "Open": open_now,
        "Timestamp": current.isoformat(),
        "Message": message,
    }


class BrokerStateStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else DEFAULT_BROKER_STATE_PATH

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []

    def upsert(self, record: dict) -> None:
        records = self.read()
        key = record.get("broker_order_id")
        records = [item for item in records if item.get("broker_order_id") != key]
        record = dict(record)
        record.setdefault("created_at", datetime.now(PACIFIC_TIME).isoformat())
        records.append(record)
        self.replace_all(records)

    def replace_all(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
