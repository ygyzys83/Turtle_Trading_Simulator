import pytest

from agentloop_trader.fees import (
    ALPACA_EQUITY_FEE_SCHEDULE_EFFECTIVE,
    estimate_alpaca_equity_order_fees,
    estimate_alpaca_equity_round_trip_fees,
    fee_adjusted_break_even_price,
)


def test_buy_fee_uses_cat_and_zero_default_commission():
    fees = estimate_alpaca_equity_order_fees(side="buy", quantity=100, price=50)

    assert fees.trade_value == 5_000
    assert fees.commission == 0
    assert fees.sec_fee == 0
    assert fees.finra_taf == 0
    assert fees.finra_cat == 0.01
    assert fees.total == 0.01


def test_sell_fee_uses_sec_taf_and_cat():
    fees = estimate_alpaca_equity_order_fees(side="sell", quantity=100, price=50)

    assert fees.sec_fee == 0.11
    assert fees.finra_taf == 0.02
    assert fees.finra_cat == 0.01
    assert fees.total == 0.14


def test_taf_is_capped_and_fractional_shares_are_supported():
    capped = estimate_alpaca_equity_order_fees(side="sell", quantity=100_000, price=10)
    fractional = estimate_alpaca_equity_order_fees(side="sell", quantity=0.5, price=100)

    assert capped.finra_taf == 9.79
    assert fractional.quantity == 0.5
    assert fractional.total == 0.03


def test_round_trip_includes_buy_and_sell_costs():
    buy, sell = estimate_alpaca_equity_round_trip_fees(
        quantity=100, entry_price=50, exit_price=55,
    )

    assert buy.total == 0.01
    assert sell.total == 0.15


def test_fee_adjusted_break_even_covers_equity_round_trip_fees():
    price = fee_adjusted_break_even_price(
        asset_class="equity", quantity=100, entry_price=50,
    )
    buy, sell = estimate_alpaca_equity_round_trip_fees(
        quantity=100, entry_price=50, exit_price=price,
    )

    assert price > 50
    assert (price - 50) * 100 >= buy.total + sell.total
    assert buy.total + sell.total == pytest.approx(0.15)


def test_invalid_side_is_rejected():
    with pytest.raises(ValueError, match="buy.*sell"):
        estimate_alpaca_equity_order_fees(side="hold", quantity=1, price=100)


def test_fee_schedule_is_dated():
    assert ALPACA_EQUITY_FEE_SCHEDULE_EFFECTIVE == "2026-07-01"
