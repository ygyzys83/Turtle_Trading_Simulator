from __future__ import annotations

import json
import os
import time as time_module
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, time as datetime_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from agentloop_trader.assets import normalize_asset_class, normalize_symbol
from agentloop_trader.brokers import AlpacaConfig, AlpacaOrderPreview, build_alpaca_order_preview
from agentloop_trader.models import ExecutionDecision, PACIFIC_TIME, RiskCheckResult, TradeIntent


DEFAULT_BROKER_STATE_PATH = Path("broker_state") / "alpaca_paper_orders.json"


@dataclass(frozen=True)
class BrokerStateHealth:
    ready: bool
    stale: bool
    reasons: list[str]


OPEN_ORDER_STATUSES = {
    "accepted",
    "new",
    "pending_new",
    "partially_filled",
    "pending_cancel",
    "pending_replace",
    "held",
}
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
    # Position adds never permit a second waiting order for the same ticker.
    if intent is None or intent.side != "buy":
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
        source = str(record.get("source") or "").strip().lower()
        if source in {"position_plan", "position_observation"}:
            record["lifecycle_status"] = source
            refreshed.append(record)
            continue
        elif record.get("adopted") and source == "adopted_alpaca_position":
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
    refreshed = [
        order for order in refresh_tracked_alpaca_orders(tracked_orders, alpaca_orders)
        if str(order.get("source") or "").strip().lower() not in {"position_plan", "position_observation"}
    ]
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
    broker_orders = [
        order for order in tracked_orders
        if str(order.get("source") or "").strip().lower() not in {"position_plan", "position_observation"}
    ]
    statuses = [_enum_value(order.get("status", "")) for order in broker_orders]
    lifecycle_statuses = [str(order.get("lifecycle_status", "")) for order in broker_orders]
    return [
        {"Metric": "Saved Alpaca orders", "Value": len(broker_orders)},
        {"Metric": "Open orders at Alpaca", "Value": lifecycle_statuses.count("open_at_alpaca")},
        {"Metric": "Filled orders at Alpaca", "Value": statuses.count("filled")},
        {"Metric": "Positions manually added to app", "Value": lifecycle_statuses.count("adopted_alpaca_position")},
        {"Metric": "Canceled orders at Alpaca", "Value": statuses.count("canceled") + statuses.count("cancelled")},
        {"Metric": "Rejected orders at Alpaca", "Value": statuses.count("rejected")},
        {"Metric": "Saved orders missing at Alpaca", "Value": lifecycle_statuses.count("missing_from_alpaca_orders")},
    ]


def alpaca_position_lifecycle_records(position_records: list[dict], tracked_orders: list[dict]) -> list[dict]:
    current_records = [
        order for order in tracked_orders
        if str(order.get("source") or "").strip().lower() in {"position_observation", "position_plan"}
        and not str(order.get("status") or "").strip().lower().startswith("closed_")
    ]
    rows = []
    for position in position_records:
        symbol = str(position.get("Symbol", "")).strip().upper()
        if not symbol:
            continue
        matching = [row for row in current_records if str(row.get("symbol") or "").strip().upper() == symbol]
        plan = next((row for row in matching if str(row.get("source") or "").strip().lower() == "position_plan"), None)
        observation = next((row for row in matching if str(row.get("source") or "").strip().lower() == "position_observation"), None)
        current = plan or observation or {}
        exit_settings = plan.get("exit_settings") if plan else {}
        tracking_status = "managed_current_cycle" if plan else "position_cycle_observed" if observation else "untracked_alpaca_position"
        rows.append(
            {
                "Symbol": symbol,
                "Position Qty": position.get("Quantity", ""),
                "Position Value": position.get("Market Value", ""),
                "Avg Entry": position.get("Average Entry", ""),
                "Current Cycle": current.get("position_cycle_id", ""),
                "Basis BUY": current.get("position_basis_order_id", ""),
                "Exit Plan Saved": plan is not None,
                "Auto Exit On": bool(exit_settings.get("auto_exit_enabled", False)),
                "Tracking Status": _display_position_status(tracking_status),
            }
        )
    return rows


def alpaca_position_lifecycle_summary_records(position_lifecycle_rows: list[dict]) -> list[dict]:
    statuses = [str(row.get("Tracking Status", "")) for row in position_lifecycle_rows]
    return [
        {"Metric": "Open Alpaca positions", "Value": sum(bool(row.get("Position Qty")) for row in position_lifecycle_rows)},
        {"Metric": "Positions with saved exit plans", "Value": statuses.count("Managed current position")},
        {"Metric": "Positions found without an exit plan", "Value": statuses.count("Position found; exit plan not saved")},
        {"Metric": "Unmanaged positions", "Value": statuses.count("Unmanaged position")},
        {"Metric": "Positions with auto exit on", "Value": sum(bool(row.get("Auto Exit On")) for row in position_lifecycle_rows)},
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
        "managed_current_cycle": "Managed current position",
        "position_matched_to_filled_order": "Matched to app order",
        "position_cycle_observed": "Position found; exit plan not saved",
        "adopted_alpaca_position": "Tracked manually",
        "untracked_alpaca_position": "Unmanaged position",
        "filled_order_without_open_position": "No open Alpaca position",
    }.get(status, status or "")


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.8f}".rstrip("0").rstrip(".")


def reconcile_alpaca_positions(position_records: list[dict], tracked_orders: list[dict]) -> list[dict]:
    current_records = [
        order for order in tracked_orders
        if str(order.get("source") or "").strip().lower() in {"position_plan", "position_observation"}
        and not str(order.get("status") or "").strip().lower().startswith("closed_")
    ]
    by_symbol: dict[str, list[dict]] = {}
    for record in current_records:
        symbol = str(record.get("symbol") or "").strip().upper()
        if symbol:
            by_symbol.setdefault(symbol, []).append(record)
    rows = []
    for position in position_records:
        symbol = str(position.get("Symbol", "")).strip().upper()
        matching = by_symbol.get(symbol, [])
        plan = next((row for row in matching if str(row.get("source") or "").strip().lower() == "position_plan"), None)
        observation = next((row for row in matching if str(row.get("source") or "").strip().lower() == "position_observation"), None)
        current = plan or observation
        rows.append({
            "Symbol": symbol,
            "Alpaca Qty": position.get("Quantity", ""),
            "Market Value": position.get("Market Value", ""),
            "Current Cycle": current.get("position_cycle_id", "") if current else "",
            "Exit Plan Saved": plan is not None,
            "Status": "Managed current position" if plan else "Position found; exit plan not saved" if observation else "Unmanaged position",
        })
    return rows


def build_exit_intent_from_position(position: dict) -> TradeIntent | None:
    asset_class = normalize_asset_class(position.get("Asset Type"), position.get("Symbol", ""))
    symbol = normalize_symbol(position.get("Symbol", ""), asset_class)
    qty = abs(_as_float(position.get("Quantity")))
    if not symbol or qty <= 0:
        return None
    return TradeIntent(
        symbol=symbol,
        side="sell",
        quantity=qty,
        asset_class=asset_class,
        order_type="market",
        time_in_force="gtc" if asset_class == "crypto" else "day",
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
    regular_open = datetime_time(9, 30) <= current.time() <= datetime_time(16, 0)
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
        with self._exclusive_lock():
            return self._read_unlocked()

    def _read_unlocked(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []

    def upsert(self, record: dict) -> None:
        with self._exclusive_lock():
            records = self._read_unlocked()
            key = record.get("broker_order_id")
            prior = next((item for item in records if item.get("broker_order_id") == key), {})
            records = [item for item in records if item.get("broker_order_id") != key]
            merged = self._merge_record(prior, dict(record))
            for settings_key in ("strategy_settings", "exit_settings"):
                if not merged.get(settings_key) and prior.get(settings_key):
                    merged[settings_key] = prior[settings_key]
            merged.setdefault("created_at", prior.get("created_at") or datetime.now(PACIFIC_TIME).isoformat())
            records.append(merged)
            self._replace_all_unlocked(records)

    def replace_all(self, records: list[dict]) -> None:
        with self._exclusive_lock():
            self._replace_all_unlocked(records)

    def _replace_all_unlocked(self, records: list[dict]) -> None:
        current = self._read_unlocked()
        merged: dict[str, dict] = {}
        unkeyed: list[dict] = []
        for row in current + list(records):
            record = dict(row)
            key = str(record.get("broker_order_id") or record.get("Alpaca Order ID") or "").strip()
            if not key:
                unkeyed.append(record)
                continue
            prior = merged.get(key, {})
            merged_record = self._merge_record(prior, record)
            for settings_key in ("strategy_settings", "exit_settings"):
                if not record.get(settings_key) and prior.get(settings_key):
                    merged_record[settings_key] = prior[settings_key]
            merged[key] = merged_record
        payload = list(merged.values()) + unkeyed
        temporary = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        try:
            self._replace_with_retry(temporary)
        finally:
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass

    @staticmethod
    def _merge_record(prior: dict, incoming: dict) -> dict:
        """Merge one record without allowing a stale process to undo a newer user plan edit."""
        if not prior:
            return dict(incoming)
        prior_source = str(prior.get("source") or "").strip().lower()
        incoming_source = str(incoming.get("source") or "").strip().lower()
        if prior_source == incoming_source == "position_plan":
            prior_updated = BrokerStateStore._plan_user_updated_at(prior)
            incoming_updated = BrokerStateStore._plan_user_updated_at(incoming)
            if prior_updated is not None and (incoming_updated is None or prior_updated > incoming_updated):
                return dict(prior)
        return {**prior, **incoming}

    @staticmethod
    def _plan_user_updated_at(record: dict) -> datetime | None:
        value = (record.get("exit_settings") or {}).get("plan_user_updated_at")
        if value in (None, "", "None"):
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    @contextmanager
    def _exclusive_lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        lock_fd = None
        for _ in range(100):
            try:
                lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                try:
                    if time_module.time() - lock_path.stat().st_mtime > 30:
                        lock_path.unlink()
                        continue
                except OSError:
                    pass
                time_module.sleep(0.02)
        if lock_fd is None:
            raise RuntimeError("Alpaca tracking file is busy; try again.")
        try:
            os.close(lock_fd)
            yield
        finally:
            try:
                lock_path.unlink()
            except OSError:
                pass

    def _replace_with_retry(self, temporary: Path, attempts: int = 50, delay_seconds: float = 0.02) -> None:
        for attempt in range(attempts):
            try:
                temporary.replace(self.path)
                return
            except PermissionError:
                if attempt == attempts - 1:
                    raise
                time_module.sleep(delay_seconds)
