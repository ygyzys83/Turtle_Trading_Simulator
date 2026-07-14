from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING


ALPACA_EQUITY_FEE_SCHEDULE_EFFECTIVE = "2026-07-01"
ALPACA_EQUITY_FEE_SCHEDULE_URL = "https://files.alpaca.markets/disclosures/library/BrokFeeSched.pdf"
ALPACA_CRYPTO_FEE_SCHEDULE_URL = "https://docs.alpaca.markets/us/docs/crypto-fees"

_SEC_SELL_RATE = Decimal("0.0000206")
_FINRA_TAF_PER_SHARE = Decimal("0.000195")
_FINRA_TAF_CAP = Decimal("9.79")
_FINRA_CAT_PER_SHARE = Decimal("0.000003")


@dataclass(frozen=True)
class EquityOrderFees:
    side: str
    quantity: float
    price: float
    trade_value: float
    commission: float
    sec_fee: float
    finra_taf: float
    finra_cat: float
    total: float


@dataclass(frozen=True)
class CryptoOrderFees:
    side: str
    quantity: float
    price: float
    trade_value: float
    liquidity: str
    trailing_30d_volume: float
    fee_rate_percent: float
    total: float


_CRYPTO_FEE_TIERS = (
    (Decimal("100000"), Decimal("0.15"), Decimal("0.25")),
    (Decimal("500000"), Decimal("0.12"), Decimal("0.22")),
    (Decimal("1000000"), Decimal("0.10"), Decimal("0.20")),
    (Decimal("10000000"), Decimal("0.08"), Decimal("0.18")),
    (Decimal("25000000"), Decimal("0.05"), Decimal("0.15")),
    (Decimal("50000000"), Decimal("0.02"), Decimal("0.13")),
    (Decimal("100000000"), Decimal("0.02"), Decimal("0.12")),
    (Decimal("Infinity"), Decimal("0.00"), Decimal("0.10")),
)


def _decimal(value: float | int) -> Decimal:
    return Decimal(str(max(0.0, float(value))))


def _round_up_cent(value: Decimal) -> Decimal:
    if value <= 0:
        return Decimal("0")
    return (value * 100).to_integral_value(rounding=ROUND_CEILING) / 100


def estimate_alpaca_equity_order_fees(
    *,
    side: str,
    quantity: float,
    price: float,
    commission_percent: float = 0.0,
    conservative_order_rounding: bool = True,
) -> EquityOrderFees:
    """Estimate current Alpaca U.S. equity fees for one executed order.

    Alpaca aggregates each fee type by account and trading day before rounding it
    up to a cent. A preview or backtest does not know the account's other daily
    activity, so conservative_order_rounding treats this order as the day's only
    activity and rounds each applicable component upward.
    """
    normalized_side = str(side).strip().lower()
    if normalized_side not in {"buy", "sell"}:
        raise ValueError("Equity fee side must be 'buy' or 'sell'.")

    qty = _decimal(quantity)
    execution_price = _decimal(price)
    trade_value = qty * execution_price
    commission = trade_value * (_decimal(commission_percent) / 100)
    sec_fee = trade_value * _SEC_SELL_RATE if normalized_side == "sell" else Decimal("0")
    finra_taf = min(qty * _FINRA_TAF_PER_SHARE, _FINRA_TAF_CAP) if normalized_side == "sell" else Decimal("0")
    finra_cat = qty * _FINRA_CAT_PER_SHARE

    components = [commission, sec_fee, finra_taf, finra_cat]
    if conservative_order_rounding:
        components = [_round_up_cent(value) for value in components]
    commission, sec_fee, finra_taf, finra_cat = components
    total = sum(components, Decimal("0"))

    return EquityOrderFees(
        side=normalized_side,
        quantity=float(qty),
        price=float(execution_price),
        trade_value=round(float(trade_value), 2),
        commission=round(float(commission), 2),
        sec_fee=round(float(sec_fee), 2),
        finra_taf=round(float(finra_taf), 2),
        finra_cat=round(float(finra_cat), 2),
        total=round(float(total), 2),
    )


def estimate_alpaca_equity_round_trip_fees(
    *,
    quantity: float,
    entry_price: float,
    exit_price: float,
    commission_percent: float = 0.0,
) -> tuple[EquityOrderFees, EquityOrderFees]:
    buy = estimate_alpaca_equity_order_fees(
        side="buy",
        quantity=quantity,
        price=entry_price,
        commission_percent=commission_percent,
    )
    sell = estimate_alpaca_equity_order_fees(
        side="sell",
        quantity=quantity,
        price=exit_price,
        commission_percent=commission_percent,
    )
    return buy, sell


def alpaca_crypto_fee_rate_percent(
    *, liquidity: str = "taker", trailing_30d_volume: float = 0.0,
) -> float:
    normalized = str(liquidity).strip().lower()
    if normalized not in {"maker", "taker"}:
        raise ValueError("Crypto liquidity must be 'maker' or 'taker'.")
    volume = _decimal(trailing_30d_volume)
    for upper_bound, maker_rate, taker_rate in _CRYPTO_FEE_TIERS:
        if volume < upper_bound:
            return float(maker_rate if normalized == "maker" else taker_rate)
    return 0.10


def estimate_alpaca_crypto_order_fees(
    *,
    side: str,
    quantity: float,
    price: float,
    liquidity: str = "taker",
    trailing_30d_volume: float = 0.0,
) -> CryptoOrderFees:
    normalized_side = str(side).strip().lower()
    if normalized_side not in {"buy", "sell"}:
        raise ValueError("Crypto fee side must be 'buy' or 'sell'.")
    normalized_liquidity = str(liquidity).strip().lower()
    rate = alpaca_crypto_fee_rate_percent(
        liquidity=normalized_liquidity,
        trailing_30d_volume=trailing_30d_volume,
    )
    qty = _decimal(quantity)
    execution_price = _decimal(price)
    trade_value = qty * execution_price
    total = trade_value * Decimal(str(rate)) / Decimal("100")
    return CryptoOrderFees(
        side=normalized_side,
        quantity=float(qty),
        price=float(execution_price),
        trade_value=round(float(trade_value), 8),
        liquidity=normalized_liquidity,
        trailing_30d_volume=float(_decimal(trailing_30d_volume)),
        fee_rate_percent=rate,
        total=round(float(total), 8),
    )


def estimate_alpaca_order_fees(
    *,
    asset_class: str,
    side: str,
    quantity: float,
    price: float,
    liquidity: str = "taker",
    trailing_30d_volume: float = 0.0,
) -> EquityOrderFees | CryptoOrderFees:
    if str(asset_class).strip().lower() == "crypto":
        return estimate_alpaca_crypto_order_fees(
            side=side,
            quantity=quantity,
            price=price,
            liquidity=liquidity,
            trailing_30d_volume=trailing_30d_volume,
        )
    return estimate_alpaca_equity_order_fees(side=side, quantity=quantity, price=price)


def estimate_alpaca_round_trip_fees(
    *,
    asset_class: str,
    quantity: float,
    entry_price: float,
    exit_price: float,
    liquidity: str = "taker",
    trailing_30d_volume: float = 0.0,
) -> tuple[EquityOrderFees | CryptoOrderFees, EquityOrderFees | CryptoOrderFees]:
    return (
        estimate_alpaca_order_fees(
            asset_class=asset_class,
            side="buy",
            quantity=quantity,
            price=entry_price,
            liquidity=liquidity,
            trailing_30d_volume=trailing_30d_volume,
        ),
        estimate_alpaca_order_fees(
            asset_class=asset_class,
            side="sell",
            quantity=quantity,
            price=exit_price,
            liquidity=liquidity,
            trailing_30d_volume=trailing_30d_volume,
        ),
    )


def fee_adjusted_break_even_price(
    *,
    asset_class: str,
    quantity: float,
    entry_price: float,
    liquidity: str = "taker",
    trailing_30d_volume: float = 0.0,
) -> float:
    """Return the estimated exit price where gross profit covers both order fees."""
    qty = max(0.0, float(quantity))
    entry = max(0.0, float(entry_price))
    if qty <= 0 or entry <= 0:
        return entry
    candidate = entry
    for _ in range(6):
        buy, sell = estimate_alpaca_round_trip_fees(
            asset_class=asset_class,
            quantity=qty,
            entry_price=entry,
            exit_price=candidate,
            liquidity=liquidity,
            trailing_30d_volume=trailing_30d_volume,
        )
        candidate = entry + (buy.total + sell.total) / qty
    return round(candidate, 8 if str(asset_class).lower() == "crypto" else 4)
