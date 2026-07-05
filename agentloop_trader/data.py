from __future__ import annotations

import numpy as np


def generate_synthetic_prices(n: int = 400, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    prices = [100.0]
    vol = 0.015
    for i in range(1, n):
        drift = 0.0003 + (0.0005 if i > n * 0.4 else 0)
        shock = (rng.random() - 0.48) * vol * 2
        prices.append(max(10.0, prices[-1] * (1 + drift + shock)))
    return np.array(prices)

