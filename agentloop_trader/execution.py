from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from agentloop_trader.models import ExecutionDecision, PACIFIC_TIME, TradeIntent


@dataclass
class PaperOrder:
    order_id: str
    symbol: str
    side: str
    quantity: int
    status: str
    submitted_at: datetime
    filled_price: float | None = None
    message: str = ""

    @property
    def notional(self) -> float:
        return (self.filled_price or 0.0) * self.quantity


@dataclass
class PaperPosition:
    symbol: str
    quantity: int = 0
    average_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.average_price


@dataclass
class PaperBroker:
    cash: float
    orders: list[PaperOrder] = field(default_factory=list)
    positions: dict[str, PaperPosition] = field(default_factory=dict)

    def submit_order(self, intent: TradeIntent, decision: ExecutionDecision) -> PaperOrder:
        if not decision.approved_for_execution:
            return self._record_order(intent, "blocked", None, decision.reason)

        fill_price = intent.entry_price or 0.0
        notional = fill_price * intent.quantity
        if fill_price <= 0:
            return self._record_order(intent, "rejected", None, "Missing valid fill price.")
        if intent.side == "buy" and notional > self.cash:
            return self._record_order(intent, "rejected", None, "Insufficient paper cash.")
        if intent.side == "sell" and self.positions.get(intent.symbol_clean, PaperPosition(intent.symbol_clean)).quantity < intent.quantity:
            return self._record_order(intent, "rejected", None, "Insufficient paper position.")

        if intent.side == "buy":
            self._fill_buy(intent.symbol_clean, intent.quantity, fill_price)
        else:
            self._fill_sell(intent.symbol_clean, intent.quantity, fill_price)

        return self._record_order(
            intent,
            "filled",
            fill_price,
            "Paper order filled at strategy reference price.",
        )

    def _fill_buy(self, symbol: str, quantity: int, price: float) -> None:
        position = self.positions.get(symbol, PaperPosition(symbol=symbol))
        existing_value = position.quantity * position.average_price
        new_value = quantity * price
        new_quantity = position.quantity + quantity
        position.quantity = new_quantity
        position.average_price = (existing_value + new_value) / new_quantity
        self.positions[symbol] = position
        self.cash -= new_value

    def _fill_sell(self, symbol: str, quantity: int, price: float) -> None:
        position = self.positions[symbol]
        position.quantity -= quantity
        self.cash += quantity * price
        if position.quantity <= 0:
            del self.positions[symbol]
        else:
            self.positions[symbol] = position

    def _record_order(
        self,
        intent: TradeIntent,
        status: str,
        filled_price: float | None,
        message: str,
    ) -> PaperOrder:
        order = PaperOrder(
            order_id=str(uuid4()),
            symbol=intent.symbol_clean,
            side=intent.side,
            quantity=intent.quantity,
            status=status,
            submitted_at=datetime.now(PACIFIC_TIME),
            filled_price=filled_price,
            message=message,
        )
        self.orders.append(order)
        return order

    def order_records(self) -> list[dict]:
        return [
            {
                "Submitted": order.submitted_at.astimezone(PACIFIC_TIME).strftime("%Y-%m-%d %H:%M:%S %Z"),
                "Order ID": order.order_id[:8],
                "Symbol": order.symbol,
                "Side": order.side.upper(),
                "Quantity": order.quantity,
                "Status": order.status,
                "Filled Price": order.filled_price,
                "Notional": round(order.notional, 2),
                "Message": order.message,
            }
            for order in self.orders
        ]

    def position_records(self) -> list[dict]:
        return [
            {
                "Symbol": position.symbol,
                "Quantity": position.quantity,
                "Average Price": round(position.average_price, 2),
                "Book Value": round(position.market_value, 2),
            }
            for position in self.positions.values()
        ]
