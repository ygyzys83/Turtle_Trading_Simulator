from __future__ import annotations

from agentloop_trader.assets import floor_quantity, normalize_asset_class, normalize_symbol
from agentloop_trader.models import TradeIntent


def _order_price(value: float) -> float:
    return round(float(value), 4 if float(value) < 1 else 2)


def build_manual_buy_intent(
    *,
    symbol: str,
    asset_class: str,
    current_price: float,
    atr: float,
    stop_atr_multiplier: float,
    order_type: str,
    requested_dollars: float | None = None,
    requested_quantity: float | None = None,
    limit_price: float | None = None,
) -> TradeIntent:
    """Build a discretionary buy without requiring a strategy entry signal."""
    kind = normalize_asset_class(asset_class, symbol)
    clean_symbol = normalize_symbol(symbol, kind)
    if not clean_symbol:
        raise ValueError("Enter a ticker or crypto pair.")
    if current_price <= 0:
        raise ValueError("A current price is required.")
    if atr <= 0:
        raise ValueError("A current ATR value is required to create the stop loss.")
    if stop_atr_multiplier <= 0:
        raise ValueError("The ATR stop multiplier must be greater than zero.")

    normalized_order_type = str(order_type).strip().lower()
    if normalized_order_type not in {"market", "limit"}:
        raise ValueError("Order type must be Market or Limit.")
    if normalized_order_type == "limit":
        if limit_price is None or limit_price <= 0:
            raise ValueError("Enter a valid limit price.")
        entry_price = _order_price(limit_price)
    else:
        entry_price = _order_price(current_price)

    if requested_quantity is not None:
        raw_quantity = float(requested_quantity)
    elif requested_dollars is not None:
        raw_quantity = float(requested_dollars) / entry_price
    else:
        raise ValueError("Enter an order amount or quantity.")
    quantity = floor_quantity(raw_quantity, kind)
    if quantity <= 0:
        raise ValueError("The requested order is too small for this asset.")

    stop_distance = float(atr) * float(stop_atr_multiplier)
    stop_loss = _order_price(entry_price - stop_distance)
    if stop_loss <= 0 or stop_loss >= entry_price:
        raise ValueError("The ATR stop must be below the buy price.")

    return TradeIntent(
        symbol=clean_symbol,
        side="buy",
        quantity=quantity,
        asset_class=kind,
        order_type=normalized_order_type,
        time_in_force="gtc" if kind == "crypto" else "day",
        limit_price=entry_price if normalized_order_type == "limit" else None,
        entry_price=entry_price,
        stop_loss=stop_loss,
        rationale=(
            "Manual order entered by the user. Quantity remains subject to the account risk limits, "
            "and the initial stop uses the selected ATR multiplier."
        ),
        proposed_by_agent="manual_order",
        source_signals=["manual_order", "atr_stop"],
    )
