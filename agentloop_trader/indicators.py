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


def calc_rsi(prices, n: int = 14) -> list[float | None]:
    prices = np.asarray(prices, dtype=float)
    rsis: list[float | None] = [None] * len(prices)
    if len(prices) <= n:
        return rsis

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:n]))
    avg_loss = float(np.mean(losses[:n]))

    for i in range(n, len(prices)):
        if i > n:
            avg_gain = (avg_gain * (n - 1) + gains[i - 1]) / n
            avg_loss = (avg_loss * (n - 1) + losses[i - 1]) / n
        if avg_loss == 0:
            rsis[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsis[i] = float(100 - (100 / (1 + rs)))
    return rsis
