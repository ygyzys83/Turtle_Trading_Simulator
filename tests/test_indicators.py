import pytest

from agentloop_trader.indicators import calc_atr, calc_rsi, calc_sma


def test_atr_uses_wilder_smoothing_after_initial_average():
    closes = [10, 11, 12, 11, 13]
    highs = [11, 12, 13, 12, 14]
    lows = [9, 10, 11, 10, 12]

    atr = calc_atr(closes, n=3, highs=highs, lows=lows)

    assert atr[:2] == [None, None]
    assert atr[2] == pytest.approx(2.0)
    assert atr[3] == pytest.approx(2.0)
    assert atr[4] == pytest.approx((2.0 * 2 + 3.0) / 3)


@pytest.mark.parametrize("calculator", [calc_sma, calc_rsi])
def test_indicators_reject_zero_length(calculator):
    with pytest.raises(ValueError):
        calculator([1, 2, 3], 0)
