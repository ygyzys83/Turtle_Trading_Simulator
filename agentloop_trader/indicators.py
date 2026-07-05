from __future__ import annotations

import numpy as np


def calc_atr(
    prices,
    n: int = 14,
    highs=None,
    lows=None,
) -> list[float | None]:
    prices = np.asarray(prices, dtype=float)
    atrs: list[float | None] = [None] * len(prices)
    if highs is not None and lows is not None:
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        tr = np.zeros(len(prices))
        tr[0] = highs[0] - lows[0]
        for i in range(1, len(prices)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - prices[i - 1]),
                abs(lows[i] - prices[i - 1]),
            )
        for i in range(n, len(prices)):
            atrs[i] = float(np.mean(tr[i - n + 1:i + 1]))
        return atrs

    for i in range(n, len(prices)):
        hi = prices[i - n + 1:i + 1] * 1.005
        lo = prices[i - n + 1:i + 1] * 0.995
        atrs[i] = float(np.mean(hi - lo))
    return atrs


def calc_sma(prices, n: int) -> list[float | None]:
    prices = np.asarray(prices, dtype=float)
    smas: list[float | None] = [None] * len(prices)
    for i in range(n - 1, len(prices)):
        smas[i] = float(np.mean(prices[i - n + 1:i + 1]))
    return smas

