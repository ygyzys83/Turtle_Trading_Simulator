from __future__ import annotations

from decimal import Decimal, ROUND_DOWN

from agentloop_trader.models import AssetClass


CRYPTO_QUANTITY_INCREMENT = Decimal("0.00000001")
EQUITY_QUANTITY_INCREMENT = Decimal("1")
_CRYPTO_QUOTES = ("USDT", "USDC", "USD", "BTC")


def normalize_asset_class(value: str | None = None, symbol: str = "") -> AssetClass:
    normalized = str(value or "").strip().lower()
    if normalized == "crypto" or "/" in str(symbol):
        return "crypto"
    return "equity"


def normalize_symbol(symbol: str, asset_class: str | None = None) -> str:
    clean = str(symbol or "").strip().upper().replace("-", "/")
    kind = normalize_asset_class(asset_class, clean)
    if kind != "crypto" or "/" in clean:
        return clean
    for quote in _CRYPTO_QUOTES:
        if clean.endswith(quote) and len(clean) > len(quote):
            return f"{clean[:-len(quote)]}/{quote}"
    return clean


def quantity_increment(asset_class: str | None = None) -> Decimal:
    return CRYPTO_QUANTITY_INCREMENT if normalize_asset_class(asset_class) == "crypto" else EQUITY_QUANTITY_INCREMENT


def floor_quantity(value: float | int, asset_class: str | None = None) -> float | int:
    increment = quantity_increment(asset_class)
    amount = max(Decimal("0"), Decimal(str(value)))
    units = (amount / increment).to_integral_value(rounding=ROUND_DOWN)
    result = units * increment
    if increment == EQUITY_QUANTITY_INCREMENT:
        return int(result)
    return float(result)


def format_quantity(value: float | int, asset_class: str | None = None) -> str:
    if normalize_asset_class(asset_class) == "crypto":
        return f"{float(value):,.8f}".rstrip("0").rstrip(".")
    return f"{int(float(value)):,}"
