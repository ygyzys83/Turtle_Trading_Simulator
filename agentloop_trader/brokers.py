from __future__ import annotations

import os
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

from agentloop_trader.execution import PaperBroker, PaperOrder
from agentloop_trader.models import ExecutionDecision, TradeIntent

DEFAULT_ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets/v2"


@dataclass(frozen=True)
class BrokerStatus:
    name: str
    connected: bool
    mode: str
    can_submit_orders: bool
    message: str


@dataclass(frozen=True)
class AlpacaOrderPreview:
    valid: bool
    preview_hash: str
    blocked_reasons: list[str]
    order: dict


@dataclass(frozen=True)
class AlpacaCancelPreview:
    valid: bool
    preview_hash: str
    blocked_reasons: list[str]
    cancel: dict


@dataclass(frozen=True)
class AlpacaTrackedOrder:
    broker_order_id: str
    preview_hash: str
    symbol: str
    side: str
    quantity: str
    status: str
    submitted_at: str
    filled_at: str
    filled_quantity: str
    average_fill_price: str


class BrokerAdapter(Protocol):
    name: str

    def status(self) -> BrokerStatus:
        ...

    def submit_order(self, intent: TradeIntent, decision: ExecutionDecision):
        ...

    def order_records(self) -> list[dict]:
        ...

    def position_records(self) -> list[dict]:
        ...

    def account_records(self) -> list[dict]:
        ...


class PaperBrokerAdapter:
    name = "PaperBroker"

    def __init__(self, broker: PaperBroker):
        self.broker = broker

    def status(self) -> BrokerStatus:
        return BrokerStatus(
            name=self.name,
            connected=True,
            mode="paper",
            can_submit_orders=True,
            message="Local paper broker is active. No external broker API is used.",
        )

    def submit_order(self, intent: TradeIntent, decision: ExecutionDecision) -> PaperOrder:
        return self.broker.submit_order(intent, decision)

    def order_records(self) -> list[dict]:
        return self.broker.order_records()

    def position_records(self) -> list[dict]:
        return self.broker.position_records()

    def account_records(self) -> list[dict]:
        return [{"Field": "Cash", "Value": round(self.broker.cash, 2)}]


@dataclass(frozen=True)
class AlpacaConfig:
    api_key: str | None
    api_secret: str | None
    base_url: str = DEFAULT_ALPACA_PAPER_BASE_URL
    paper: bool = True

    @classmethod
    def from_env(cls) -> "AlpacaConfig":
        _load_dotenv_if_available()
        return cls(
            api_key=os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID"),
            api_secret=os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY"),
            base_url=os.getenv("APCA_API_BASE_URL") or os.getenv("ALPACA_API_BASE_URL") or DEFAULT_ALPACA_PAPER_BASE_URL,
            paper=(os.getenv("ALPACA_PAPER", "true").strip().lower() != "false"),
        )

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.api_secret)


def alpaca_config_validation_records(config: AlpacaConfig) -> list[dict]:
    endpoint = str(config.base_url or "")
    paper_endpoint = "paper-api.alpaca.markets" in endpoint
    live_endpoint = "api.alpaca.markets" in endpoint and not paper_endpoint
    return [
        {"Check": "API keys found", "Passed": config.has_credentials, "Detail": "API key and secret are set." if config.has_credentials else "API key or secret is missing."},
        {"Check": "Using paper account", "Passed": config.paper, "Detail": "Alpaca paper mode is on." if config.paper else "Live mode is configured, but live orders remain blocked."},
        {"Check": "Paper account URL", "Passed": paper_endpoint, "Detail": endpoint or "No Alpaca URL is set."},
        {"Check": "Live account URL blocked", "Passed": not live_endpoint, "Detail": "Live account URL is not configured." if not live_endpoint else "Live account URL detected; live orders are still blocked."},
        {"Check": "Alpaca URL includes /v2", "Passed": endpoint.endswith("/v2"), "Detail": endpoint or "No Alpaca URL is set."},
    ]


def _load_dotenv_if_available() -> None:
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except Exception:
        _load_local_dotenv_fallback()
        return
    load_dotenv(dotenv_path=env_path, override=False)


def _load_local_dotenv_fallback() -> None:
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip("'\"")
        if name and name not in os.environ:
            os.environ[name] = value


class AlpacaBrokerAdapterStub:
    name = "AlpacaPaperAdapter"

    def __init__(
        self,
        config: AlpacaConfig | None = None,
        trading_client=None,
        allow_order_submission: bool = False,
    ):
        self.config = config or AlpacaConfig.from_env()
        self._client = trading_client
        self._client_error: str | None = None
        self.allow_order_submission = allow_order_submission

    def status(self) -> BrokerStatus:
        if not self.config.has_credentials:
            return BrokerStatus(
                name=self.name,
                connected=False,
                mode="paper" if self.config.paper else "live",
                can_submit_orders=False,
                message="Alpaca credentials not configured. Set paper credentials before enabling this adapter.",
            )
        client = self._get_client()
        if client is None:
            return BrokerStatus(
                name=self.name,
                connected=False,
                mode="paper" if self.config.paper else "live",
                can_submit_orders=False,
                message=self._client_error or "Alpaca SDK client is unavailable.",
            )
        try:
            account = client.get_account()
        except Exception as exc:
            return BrokerStatus(
                name=self.name,
                connected=False,
                mode="paper" if self.config.paper else "live",
                can_submit_orders=False,
                message=f"Alpaca account read failed: {exc}",
            )
        return BrokerStatus(
            name=self.name,
            connected=True,
            mode="paper" if self.config.paper else "live",
            can_submit_orders=self.config.paper and self.allow_order_submission,
            message=(
                f"Alpaca account connected; status={getattr(account, 'status', 'unknown')}. "
                f"Order submission is {'enabled for paper trading' if self.config.paper and self.allow_order_submission else 'blocked'}."
            ),
        )

    def submit_order(self, intent: TradeIntent, decision: ExecutionDecision, expected_preview_hash: str | None = None):
        preview = build_alpaca_order_preview(intent, decision, self.config)
        if not preview.valid:
            raise RuntimeError("; ".join(preview.blocked_reasons))
        if expected_preview_hash is not None and expected_preview_hash != preview.preview_hash:
            raise RuntimeError("Alpaca paper order preview changed. Re-arm the order before submitting.")
        if not self.config.paper:
            raise RuntimeError("Alpaca live order submission is blocked. Use paper mode only.")
        if not self.allow_order_submission:
            raise RuntimeError("Paper orders are turned off in the sidebar.")
        if not decision.approved_for_execution:
            raise RuntimeError(f"Order blocked: {decision.reason}")
        client = self._get_client()
        if client is None:
            raise RuntimeError(self._client_error or "Alpaca client is unavailable.")
        request = self._build_market_order_request(intent)
        return client.submit_order(order_data=request)

    def cancel_order(self, broker_order_id: str, expected_cancel_hash: str | None = None):
        order = self._get_order_by_id(broker_order_id)
        if order is None:
            raise RuntimeError("Alpaca paper cancel blocked: order could not be refreshed.")
        order_record = alpaca_tracked_order_records([
            alpaca_tracked_order_from_broker_order(order, preview_hash="")
        ])[0]
        preview = build_alpaca_cancel_preview(order_record, self.config)
        if not preview.valid:
            raise RuntimeError("; ".join(preview.blocked_reasons))
        if expected_cancel_hash is not None and expected_cancel_hash != preview.preview_hash:
            raise RuntimeError("Alpaca paper cancel preview changed. Re-arm the cancel before submitting.")
        if not self.config.paper:
            raise RuntimeError("Alpaca live order cancellation is blocked. Use paper mode only.")
        if not self.allow_order_submission:
            raise RuntimeError("Paper cancels are turned off in the sidebar.")
        client = self._get_client()
        if client is None:
            raise RuntimeError(self._client_error or "Alpaca client is unavailable.")
        for method_name in ("cancel_order_by_id", "cancel_order"):
            method = getattr(client, method_name, None)
            if method is None:
                continue
            result = method(broker_order_id)
            return result or SimpleNamespace(id=broker_order_id, status="cancel_requested")
        raise RuntimeError("Alpaca client does not expose an order cancellation method.")

    def account_records(self) -> list[dict]:
        client = self._get_client()
        if client is None:
            return []
        try:
            account = client.get_account()
        except Exception:
            return []
        fields = ["status", "cash", "buying_power", "portfolio_value", "equity", "currency"]
        return [
            {"Field": field.replace("_", " ").title(), "Value": getattr(account, field, "")}
            for field in fields
        ]

    def order_records(self) -> list[dict]:
        client = self._get_client()
        if client is None:
            return []
        orders = self._get_orders(client)
        return alpaca_tracked_order_records([
            alpaca_tracked_order_from_broker_order(order, preview_hash="")
            for order in orders
        ])

    def refreshed_tracked_order_records(self, tracked_orders: list[dict]) -> list[dict]:
        rows = []
        for order in tracked_orders:
            broker_order_id = str(order.get("broker_order_id") or order.get("Broker Order ID") or "").strip()
            if not broker_order_id:
                continue
            rows.append(self.tracked_order_record(broker_order_id, order.get("preview_hash", "")))
        return rows

    def _get_orders(self, client) -> list:
        request = self._build_all_orders_request()
        if request is not None:
            for kwargs in ({"filter": request}, {"request_params": request}):
                try:
                    return client.get_orders(**kwargs)
                except TypeError:
                    continue
                except Exception:
                    return []
            try:
                return client.get_orders(request)
            except TypeError:
                pass
            except Exception:
                return []
        try:
            orders = client.get_orders()
        except TypeError:
            orders = client.get_orders(filter=None)
        except Exception:
            return []
        return orders

    def tracked_order_record(self, broker_order_id: str, preview_hash: str) -> dict:
        order = self._get_order_by_id(broker_order_id)
        if order is None:
            return {"Order ID": broker_order_id[:8], "Alpaca Order ID": broker_order_id, "Review ID": preview_hash, "Status": "unavailable"}
        return alpaca_tracked_order_records([
            alpaca_tracked_order_from_broker_order(order, preview_hash=preview_hash)
        ])[0]

    def position_records(self) -> list[dict]:
        client = self._get_client()
        if client is None:
            return []
        try:
            positions = client.get_all_positions()
        except Exception:
            return []
        return [
            {
                "Symbol": getattr(position, "symbol", ""),
                "Quantity": getattr(position, "qty", ""),
                "Market Value": getattr(position, "market_value", ""),
                "Average Entry": getattr(position, "avg_entry_price", ""),
                "Unrealized P&L": getattr(position, "unrealized_pl", ""),
            }
            for position in positions
        ]

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.config.has_credentials:
            return None
        try:
            from alpaca.trading.client import TradingClient
        except Exception as exc:
            self._client_error = f"alpaca-py is not installed or could not be imported: {exc}"
            return None
        try:
            self._client = TradingClient(
                api_key=self.config.api_key,
                secret_key=self.config.api_secret,
                paper=self.config.paper,
            )
        except Exception as exc:
            self._client_error = f"Alpaca client initialization failed: {exc}"
            return None
        return self._client

    def _build_market_order_request(self, intent: TradeIntent):
        try:
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import MarketOrderRequest
        except Exception:
            return {
                "symbol": intent.symbol_clean,
                "qty": intent.quantity,
                "side": intent.side,
                "type": "market",
                "time_in_force": intent.time_in_force,
            }
        side = OrderSide.BUY if intent.side == "buy" else OrderSide.SELL
        tif = TimeInForce.DAY if intent.time_in_force == "day" else TimeInForce.GTC
        return MarketOrderRequest(
            symbol=intent.symbol_clean,
            qty=intent.quantity,
            side=side,
            time_in_force=tif,
        )

    def _build_all_orders_request(self):
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest
        except Exception:
            return None
        return GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500)

    def _get_order_by_id(self, broker_order_id: str):
        client = self._get_client()
        if client is None or not broker_order_id:
            return None
        for method_name in ("get_order_by_id", "get_order"):
            method = getattr(client, method_name, None)
            if method is None:
                continue
            try:
                return method(broker_order_id)
            except Exception:
                continue
        return None


def build_alpaca_order_preview(
    intent: TradeIntent | None,
    decision: ExecutionDecision,
    config: AlpacaConfig,
) -> AlpacaOrderPreview:
    blocked: list[str] = []
    symbol = intent.symbol_clean if intent else ""
    order = {
        "broker": "alpaca",
        "mode": "paper" if config.paper else "live",
        "symbol": symbol,
        "side": intent.side if intent else "",
        "quantity": intent.quantity if intent else 0,
        "order_type": intent.order_type if intent else "",
        "time_in_force": intent.time_in_force if intent else "",
        "source": "adjusted_deterministic_trade_intent",
    }
    if intent is None:
        blocked.append("No trade intent is present.")
    if symbol == "SYNTH":
        blocked.append("Synthetic symbols cannot be sent to Alpaca.")
    if not config.paper:
        blocked.append("Alpaca live order submission is blocked. Use paper mode only.")
    if not decision.approved_for_execution:
        blocked.append(f"Execution decision blocked order: {decision.reason}")
    if intent is not None and intent.quantity <= 0:
        blocked.append("Quantity must be greater than zero.")

    payload = json.dumps(order, sort_keys=True, separators=(",", ":"))
    preview_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return AlpacaOrderPreview(
        valid=not blocked,
        preview_hash=preview_hash,
        blocked_reasons=list(dict.fromkeys(blocked)),
        order=order,
    )


def alpaca_preview_records(preview: AlpacaOrderPreview) -> list[dict]:
    rows = [{"Field": _display_order_field(key), "Value": value} for key, value in preview.order.items()]
    rows.append({"Field": "Review ID", "Value": preview.preview_hash})
    rows.append({"Field": "Ready To Send", "Value": preview.valid})
    return rows


def build_alpaca_cancel_preview(order_record: dict | None, config: AlpacaConfig) -> AlpacaCancelPreview:
    blocked: list[str] = []
    order_record = order_record or {}
    broker_order_id = str(order_record.get("Alpaca Order ID") or order_record.get("Broker Order ID") or order_record.get("Order ID") or "").strip()
    status = _enum_value(order_record.get("Status", ""))
    cancel = {
        "broker": "alpaca",
        "mode": "paper" if config.paper else "live",
        "action": "cancel_order",
        "broker_order_id": broker_order_id,
        "symbol": str(order_record.get("Symbol", "")).strip().upper(),
        "side": _enum_value(order_record.get("Side", "")),
        "quantity": _canonical_quantity(order_record.get("Quantity", "")),
        "status": status,
        "source": "alpaca_open_order_cancel",
    }
    if not broker_order_id:
        blocked.append("No Alpaca broker order ID is present.")
    if not config.paper:
        blocked.append("Alpaca live order cancellation is blocked. Use paper mode only.")
    open_statuses = {"accepted", "new", "pending_new", "partially_filled"}
    if status not in open_statuses:
        blocked.append(f"Only open Alpaca paper orders can be cancelled. Current status is {status or 'unknown'}.")

    payload = json.dumps(cancel, sort_keys=True, separators=(",", ":"))
    preview_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return AlpacaCancelPreview(
        valid=not blocked,
        preview_hash=preview_hash,
        blocked_reasons=list(dict.fromkeys(blocked)),
        cancel=cancel,
    )


def alpaca_cancel_preview_records(preview: AlpacaCancelPreview) -> list[dict]:
    rows = [{"Field": _display_order_field(key), "Value": value} for key, value in preview.cancel.items()]
    rows.append({"Field": "Review ID", "Value": preview.preview_hash})
    rows.append({"Field": "Ready To Cancel", "Value": preview.valid})
    return rows


def _enum_value(value) -> str:
    text = str(value or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.strip().lower()


def _canonical_quantity(value) -> str:
    text = str(value or "").strip()
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if number.is_integer():
        return str(int(number))
    return f"{number:.8f}".rstrip("0").rstrip(".")


def alpaca_tracked_order_from_broker_order(order, preview_hash: str) -> AlpacaTrackedOrder:
    return AlpacaTrackedOrder(
        broker_order_id=str(getattr(order, "id", "")),
        preview_hash=preview_hash,
        symbol=str(getattr(order, "symbol", "")),
        side=str(getattr(order, "side", "")).upper(),
        quantity=str(getattr(order, "qty", "")),
        status=str(getattr(order, "status", "")),
        submitted_at=str(getattr(order, "submitted_at", "")),
        filled_at=str(getattr(order, "filled_at", "")),
        filled_quantity=str(getattr(order, "filled_qty", "")),
        average_fill_price=str(getattr(order, "filled_avg_price", "")),
    )


def alpaca_tracked_order_records(orders: list[AlpacaTrackedOrder]) -> list[dict]:
    return [
        {
            "Order ID": order.broker_order_id[:8],
            "Alpaca Order ID": order.broker_order_id,
            "Review ID": order.preview_hash,
            "Symbol": order.symbol,
            "Side": order.side,
            "Quantity": order.quantity,
            "Status": order.status,
            "Submitted": order.submitted_at,
            "Filled": order.filled_at,
            "Filled Qty": order.filled_quantity,
            "Avg Fill": order.average_fill_price,
        }
        for order in orders
    ]


def _display_order_field(key: str) -> str:
    return {
        "broker": "Broker",
        "mode": "Account",
        "symbol": "Symbol",
        "side": "Side",
        "quantity": "Quantity",
        "order_type": "Order Type",
        "time_in_force": "Time In Force",
        "source": "Source",
        "action": "Action",
        "broker_order_id": "Alpaca Order ID",
        "status": "Alpaca Status",
    }.get(key, key.replace("_", " ").title())


def broker_status_records(statuses: list[BrokerStatus]) -> list[dict]:
    return [
        {
            "Broker": status.name,
            "Connected": status.connected,
            "Mode": status.mode,
            "Can Submit Orders": status.can_submit_orders,
            "Message": status.message,
        }
        for status in statuses
    ]
