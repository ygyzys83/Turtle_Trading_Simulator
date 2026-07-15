from __future__ import annotations

import os
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

from agentloop_trader.assets import normalize_asset_class, normalize_symbol
from agentloop_trader.execution import PaperBroker, PaperOrder
from agentloop_trader.fees import (
    ALPACA_CRYPTO_FEE_SCHEDULE_URL,
    ALPACA_EQUITY_FEE_SCHEDULE_EFFECTIVE,
    estimate_alpaca_order_fees,
)
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
    order_type: str
    limit_price: str
    status: str
    submitted_at: str
    filled_at: str
    filled_quantity: str
    average_fill_price: str
    asset_class: str = "equity"


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
    live_trading_enabled: bool = False
    live_confirmation: str = ""

    @classmethod
    def from_env(cls) -> "AlpacaConfig":
        _load_dotenv_if_available()
        return cls(
            api_key=os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID"),
            api_secret=os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY"),
            base_url=os.getenv("APCA_API_BASE_URL") or os.getenv("ALPACA_API_BASE_URL") or DEFAULT_ALPACA_PAPER_BASE_URL,
            paper=(os.getenv("ALPACA_PAPER", "true").strip().lower() != "false"),
            live_trading_enabled=(os.getenv("ALPACA_LIVE_TRADING_ENABLED", "false").strip().lower() == "true"),
            live_confirmation=os.getenv("ALPACA_LIVE_CONFIRMATION", "").strip(),
        )

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.api_secret)

    @property
    def live_order_enabled(self) -> bool:
        return (
            not self.paper
            and self.live_trading_enabled
            and self.live_confirmation == "I_UNDERSTAND_LIVE_TRADING"
        )

    @property
    def account_mode(self) -> str:
        return "paper" if self.paper else "live"


def alpaca_config_validation_records(config: AlpacaConfig) -> list[dict]:
    endpoint = str(config.base_url or "")
    paper_endpoint = "paper-api.alpaca.markets" in endpoint
    live_endpoint = "api.alpaca.markets" in endpoint and not paper_endpoint
    return [
        {"Check": "API keys found", "Passed": config.has_credentials, "Detail": "API key and secret are set." if config.has_credentials else "API key or secret is missing."},
        {"Check": "Using paper account", "Passed": config.paper, "Detail": "Alpaca paper mode is on." if config.paper else "Live mode is configured."},
        {"Check": "Paper account URL", "Passed": paper_endpoint if config.paper else True, "Detail": endpoint or "No Alpaca URL is set."},
        {"Check": "Live account URL", "Passed": live_endpoint if not config.paper else True, "Detail": endpoint or "No Alpaca URL is set."},
        {"Check": "Live env switch", "Passed": config.paper or config.live_trading_enabled, "Detail": "Set ALPACA_LIVE_TRADING_ENABLED=true to allow live order wiring." if not config.paper and not config.live_trading_enabled else "Live env switch is set." if not config.paper else "Not needed for paper."},
        {"Check": "Live confirmation", "Passed": config.paper or config.live_confirmation == "I_UNDERSTAND_LIVE_TRADING", "Detail": "Set ALPACA_LIVE_CONFIRMATION=I_UNDERSTAND_LIVE_TRADING to enable live order wiring." if not config.paper and config.live_confirmation != "I_UNDERSTAND_LIVE_TRADING" else "Live confirmation is set." if not config.paper else "Not needed for paper."},
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
    name = "AlpacaAdapter"

    def __init__(
        self,
        config: AlpacaConfig | None = None,
        trading_client=None,
        allow_order_submission: bool = False,
    ):
        self.config = config or AlpacaConfig.from_env()
        self._client = trading_client
        self._client_error: str | None = None
        self._read_errors: dict[str, str] = {}
        self.allow_order_submission = allow_order_submission

    @property
    def read_errors(self) -> dict[str, str]:
        return dict(self._read_errors)

    def status(self) -> BrokerStatus:
        if not self.config.has_credentials:
            return BrokerStatus(
                name=self.name,
                connected=False,
                mode=self.config.account_mode,
                can_submit_orders=False,
                message="Alpaca credentials not configured. Set paper credentials before enabling this adapter.",
            )
        client = self._get_client()
        if client is None:
            return BrokerStatus(
                name=self.name,
                connected=False,
                mode=self.config.account_mode,
                can_submit_orders=False,
                message=self._client_error or "Alpaca SDK client is unavailable.",
            )
        try:
            account = client.get_account()
        except Exception as exc:
            return BrokerStatus(
                name=self.name,
                connected=False,
                mode=self.config.account_mode,
                can_submit_orders=False,
                message=f"Alpaca account read failed: {exc}",
            )
        account_blocked = any(
            bool(getattr(account, field, False))
            for field in ("trading_blocked", "account_blocked", "trade_suspended_by_user")
        )
        can_submit = self.allow_order_submission and (self.config.paper or self.config.live_order_enabled) and not account_blocked
        submit_message = (
            "enabled for paper trading"
            if self.config.paper and can_submit
            else "enabled for live trading"
            if can_submit
            else "blocked by Alpaca account status"
            if account_blocked
            else "blocked"
        )
        return BrokerStatus(
            name=self.name,
            connected=True,
            mode=self.config.account_mode,
            can_submit_orders=can_submit,
            message=(
                f"Alpaca account connected; status={getattr(account, 'status', 'unknown')}. "
                f"Order submission is {submit_message}."
            ),
        )

    def submit_order(self, intent: TradeIntent, decision: ExecutionDecision, expected_preview_hash: str | None = None):
        preview = build_alpaca_order_preview(intent, decision, self.config)
        if not preview.valid:
            raise RuntimeError("; ".join(preview.blocked_reasons))
        if expected_preview_hash is not None and expected_preview_hash != preview.preview_hash:
            raise RuntimeError("Alpaca paper order preview changed. Re-arm the order before submitting.")
        if not self.config.paper and not self.config.live_order_enabled:
            raise RuntimeError("Alpaca live order submission is not enabled. Configure the live endpoint and confirmation first.")
        if not self.allow_order_submission:
            raise RuntimeError("Alpaca order submission is turned off in the sidebar.")
        if not decision.approved_for_execution:
            raise RuntimeError(f"Order blocked: {decision.reason}")
        client = self._get_client()
        if client is None:
            raise RuntimeError(self._client_error or "Alpaca client is unavailable.")
        request = self._build_order_request(intent, client_order_id=f"agentloop-{preview.preview_hash}"[:128])
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
        if not self.config.paper and not self.config.live_order_enabled:
            raise RuntimeError("Alpaca live order cancellation is not enabled. Configure the live endpoint and confirmation first.")
        if not self.allow_order_submission:
            raise RuntimeError("Alpaca cancels are turned off in the sidebar.")
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
        fields = [
            "status", "cash", "buying_power", "portfolio_value", "equity", "last_equity", "currency",
            "trading_blocked", "account_blocked", "trade_suspended_by_user",
        ]
        return [
            {"Field": field.replace("_", " ").title(), "Value": getattr(account, field, "")}
            for field in fields
        ]

    def market_is_open(self) -> bool | None:
        client = self._get_client()
        if client is None:
            return None
        try:
            clock = client.get_clock()
        except Exception:
            return None
        return bool(getattr(clock, "is_open", False))

    def order_records(self, strict: bool = False) -> list[dict]:
        client = self._get_client()
        if client is None:
            if strict:
                raise RuntimeError(self._client_error or "Alpaca client is unavailable.")
            return []
        orders = self._get_orders(client, strict=strict)
        self._read_errors.pop("orders", None)
        return alpaca_tracked_order_records([
            alpaca_tracked_order_from_broker_order(order, preview_hash="")
            for order in orders
        ])

    def refreshed_tracked_order_records(self, tracked_orders: list[dict]) -> list[dict]:
        rows = []
        for order in tracked_orders:
            if str(order.get("source") or "").strip().lower() in {
                "position_plan",
                "position_observation",
                "adopted_alpaca_position",
            }:
                continue
            broker_order_id = str(order.get("broker_order_id") or order.get("Broker Order ID") or "").strip()
            if not broker_order_id:
                continue
            rows.append(self.tracked_order_record(broker_order_id, order.get("preview_hash", "")))
        return rows

    def _get_orders(self, client, strict: bool = False) -> list:
        last_error: Exception | None = None
        request = self._build_all_orders_request()
        if request is not None:
            for kwargs in ({"filter": request}, {"request_params": request}):
                try:
                    return client.get_orders(**kwargs)
                except TypeError:
                    continue
                except Exception as exc:
                    last_error = exc
                    break
            try:
                return client.get_orders(request)
            except TypeError:
                pass
            except Exception as exc:
                last_error = exc
        try:
            orders = client.get_orders()
        except TypeError:
            try:
                orders = client.get_orders(filter=None)
            except Exception as exc:
                last_error = exc
                orders = None
        except Exception as exc:
            last_error = exc
            orders = None
        if orders is None:
            message = f"Alpaca orders read failed: {last_error or 'unknown error'}"
            self._read_errors["orders"] = message
            if strict:
                raise RuntimeError(message) from last_error
            return []
        return orders

    def tracked_order_record(self, broker_order_id: str, preview_hash: str) -> dict:
        order = self._get_order_by_id(broker_order_id)
        if order is None:
            return {"Order ID": broker_order_id[:8], "Alpaca Order ID": broker_order_id, "Review ID": preview_hash, "Status": "unavailable"}
        return alpaca_tracked_order_records([
            alpaca_tracked_order_from_broker_order(order, preview_hash=preview_hash)
        ])[0]

    def position_records(self, strict: bool = False) -> list[dict]:
        client = self._get_client()
        if client is None:
            if strict:
                raise RuntimeError(self._client_error or "Alpaca client is unavailable.")
            return []
        try:
            positions = client.get_all_positions()
        except Exception as exc:
            message = f"Alpaca positions read failed: {exc}"
            self._read_errors["positions"] = message
            if strict:
                raise RuntimeError(message) from exc
            return []
        self._read_errors.pop("positions", None)
        return [
            {
                "Symbol": normalize_symbol(
                    getattr(position, "symbol", ""),
                    _enum_value(getattr(position, "asset_class", "equity")),
                ),
                "Asset Type": normalize_asset_class(
                    _enum_value(getattr(position, "asset_class", "equity")),
                    getattr(position, "symbol", ""),
                ),
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

    def _build_order_request(self, intent: TradeIntent, client_order_id: str = ""):
        if intent.order_type == "limit" and (intent.limit_price is None or intent.limit_price <= 0):
            raise RuntimeError("Limit orders need a limit price.")
        try:
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
        except Exception:
            order = {
                "symbol": intent.symbol_clean,
                "qty": intent.quantity,
                "side": intent.side,
                "type": intent.order_type,
                "time_in_force": intent.time_in_force,
            }
            if client_order_id:
                order["client_order_id"] = client_order_id
            if intent.order_type == "limit":
                order["limit_price"] = intent.limit_price
            return order
        side = OrderSide.BUY if intent.side == "buy" else OrderSide.SELL
        tif = (
            TimeInForce.DAY
            if intent.time_in_force == "day"
            else TimeInForce.IOC
            if intent.time_in_force == "ioc"
            else TimeInForce.GTC
        )
        if intent.order_type == "limit":
            return LimitOrderRequest(
                symbol=intent.symbol_clean,
                qty=intent.quantity,
                side=side,
                time_in_force=tif,
                limit_price=float(intent.limit_price),
                client_order_id=client_order_id or None,
            )
        return MarketOrderRequest(
            symbol=intent.symbol_clean,
            qty=intent.quantity,
            side=side,
            time_in_force=tif,
            client_order_id=client_order_id or None,
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
    asset_class = intent.asset_class if intent else "equity"
    reference_price = 0.0
    if intent is not None:
        reference_price = float(intent.limit_price or intent.entry_price or 0.0)
    fee_estimate = (
        estimate_alpaca_order_fees(
            asset_class=asset_class,
            side=intent.side,
            quantity=intent.quantity,
            price=reference_price,
        )
        if intent is not None and intent.side in {"buy", "sell"} and intent.quantity > 0 and reference_price > 0
        else None
    )
    estimated_order_value = fee_estimate.trade_value if fee_estimate is not None else 0.0
    estimated_cash_change = (
        estimated_order_value + fee_estimate.total
        if fee_estimate is not None and intent is not None and intent.side == "buy"
        else estimated_order_value - fee_estimate.total
        if fee_estimate is not None
        else 0.0
    )
    maker_fee_estimate = (
        estimate_alpaca_order_fees(
            asset_class="crypto",
            side=intent.side,
            quantity=intent.quantity,
            price=reference_price,
            liquidity="maker",
        )
        if intent is not None and asset_class == "crypto" and intent.order_type == "limit" and reference_price > 0
        else None
    )
    order = {
        "broker": "alpaca",
        "mode": config.account_mode,
        "symbol": symbol,
        "asset_class": asset_class,
        "side": intent.side if intent else "",
        "quantity": intent.quantity if intent else 0,
        "order_type": intent.order_type if intent else "",
        "limit_price": intent.limit_price if intent and intent.order_type == "limit" else "",
        "time_in_force": intent.time_in_force if intent else "",
        "estimated_order_value": f"${estimated_order_value:,.2f}" if fee_estimate is not None else "Not available",
        "estimated_alpaca_fees": f"${fee_estimate.total:,.2f}" if fee_estimate is not None else "Not available",
        "possible_maker_fee": f"${maker_fee_estimate.total:,.2f}" if maker_fee_estimate is not None else "Not applicable",
        "estimated_cash_needed": (
            f"${estimated_cash_change:,.2f}"
            if fee_estimate is not None and intent is not None and intent.side == "buy"
            else "Not applicable"
        ),
        "estimated_net_proceeds": (
            f"${estimated_cash_change:,.2f}"
            if fee_estimate is not None and intent is not None and intent.side == "sell"
            else "Not applicable"
        ),
        "fee_estimate_note": (
            "Conservative Tier 1 taker estimate. A resting crypto limit order may receive the lower maker fee."
            if asset_class == "crypto"
            else "Live-equivalent estimate; Alpaca paper does not deduct regulatory fees."
            if config.paper
            else "Estimate only; Alpaca aggregates fee types daily and posts actual charges at day-end."
        ),
        "fee_schedule_effective": ALPACA_CRYPTO_FEE_SCHEDULE_URL if asset_class == "crypto" else ALPACA_EQUITY_FEE_SCHEDULE_EFFECTIVE,
        "source": (
            "manual_order"
            if intent is not None and intent.proposed_by_agent == "manual_order"
            else "adjusted_deterministic_trade_intent"
        ),
    }
    if intent is None:
        blocked.append("No trade intent is present.")
    if symbol == "SYNTH":
        blocked.append("Synthetic symbols cannot be sent to Alpaca.")
    if config.paper and decision.mode in {"live_with_approval", "automated_live"}:
        blocked.append("Live order mode cannot send to an Alpaca paper account.")
    if not config.paper and decision.mode not in {"live_with_approval", "automated_live"}:
        blocked.append("Paper order mode cannot send to an Alpaca live account.")
    if not config.paper and not config.live_order_enabled:
        blocked.append("Alpaca live order submission is not enabled. Configure the live endpoint and confirmation first.")
    if not decision.approved_for_execution:
        blocked.append(f"Execution decision blocked order: {decision.reason}")
    if intent is not None and intent.quantity <= 0:
        blocked.append("Quantity must be greater than zero.")
    if intent is not None and intent.order_type == "limit" and (intent.limit_price is None or intent.limit_price <= 0):
        blocked.append("Limit orders need a limit price.")
    if intent is not None and intent.asset_class == "crypto" and intent.time_in_force not in {"gtc", "ioc"}:
        blocked.append("Crypto orders require GTC or IOC time in force.")

    payload = json.dumps(order, sort_keys=True, separators=(",", ":"))
    preview_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return AlpacaOrderPreview(
        valid=not blocked,
        preview_hash=preview_hash,
        blocked_reasons=list(dict.fromkeys(blocked)),
        order=order,
    )


def alpaca_preview_records(preview: AlpacaOrderPreview) -> list[dict]:
    rows = [{"Field": _display_order_field(key), "Value": str(value)} for key, value in preview.order.items()]
    rows.append({"Field": "Review ID", "Value": preview.preview_hash})
    rows.append({"Field": "Ready To Send", "Value": "Yes" if preview.valid else "No"})
    return rows


def build_alpaca_cancel_preview(order_record: dict | None, config: AlpacaConfig) -> AlpacaCancelPreview:
    blocked: list[str] = []
    order_record = order_record or {}
    broker_order_id = str(order_record.get("Alpaca Order ID") or order_record.get("Broker Order ID") or order_record.get("Order ID") or "").strip()
    status = _enum_value(order_record.get("Status", ""))
    cancel = {
        "broker": "alpaca",
        "mode": config.account_mode,
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
    if not config.paper and not config.live_order_enabled:
        blocked.append("Alpaca live order cancellation is not enabled. Configure the live endpoint and confirmation first.")
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
    rows = [{"Field": _display_order_field(key), "Value": str(value)} for key, value in preview.cancel.items()]
    rows.append({"Field": "Review ID", "Value": preview.preview_hash})
    rows.append({"Field": "Ready To Cancel", "Value": "Yes" if preview.valid else "No"})
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
    asset_class = normalize_asset_class(
        _enum_value(getattr(order, "asset_class", "equity")),
        str(getattr(order, "symbol", "")),
    )
    return AlpacaTrackedOrder(
        broker_order_id=str(getattr(order, "id", "")),
        preview_hash=preview_hash,
        symbol=normalize_symbol(str(getattr(order, "symbol", "")), asset_class),
        side=str(getattr(order, "side", "")).upper(),
        quantity=str(getattr(order, "qty", "")),
        order_type=str(getattr(order, "type", "")),
        limit_price=str(getattr(order, "limit_price", "") or ""),
        status=str(getattr(order, "status", "")),
        submitted_at=str(getattr(order, "submitted_at", "")),
        filled_at=str(getattr(order, "filled_at", "")),
        filled_quantity=str(getattr(order, "filled_qty", "")),
        average_fill_price=str(getattr(order, "filled_avg_price", "")),
        asset_class=asset_class,
    )


def alpaca_tracked_order_records(orders: list[AlpacaTrackedOrder]) -> list[dict]:
    return [
        {
            "Order ID": order.broker_order_id[:8],
            "Alpaca Order ID": order.broker_order_id,
            "Review ID": order.preview_hash,
            "Symbol": order.symbol,
            "Asset Type": order.asset_class,
            "Side": order.side,
            "Quantity": order.quantity,
            "Order Type": order.order_type,
            "Limit Price": order.limit_price,
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
        "asset_class": "Asset Type",
        "side": "Side",
        "quantity": "Quantity",
        "order_type": "Order Type",
        "limit_price": "Limit Price",
        "time_in_force": "Time In Force",
        "source": "Source",
        "action": "Action",
        "broker_order_id": "Alpaca Order ID",
        "status": "Alpaca Status",
        "estimated_order_value": "Estimated Order Value",
        "estimated_alpaca_fees": "Estimated Alpaca Fees",
        "possible_maker_fee": "Possible Maker Fee",
        "estimated_cash_needed": "Estimated Cash Needed",
        "estimated_net_proceeds": "Estimated Net Proceeds",
        "fee_estimate_note": "Fee Estimate Note",
        "fee_schedule_effective": "Fee Schedule Effective",
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
