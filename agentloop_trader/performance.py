from __future__ import annotations

from typing import Any

import pandas as pd

from agentloop_trader.models import RiskLimits


def ticker_allocation_percent(risk_limits: RiskLimits | None) -> float:
    if risk_limits is None:
        return 100.0
    return max(0.01, min(100.0, float(risk_limits.max_symbol_concentration_pct)))


def ticker_allocated_capital(account_equity: float, risk_limits: RiskLimits | None) -> float:
    return max(0.0, float(account_equity)) * ticker_allocation_percent(risk_limits) / 100.0


def elapsed_years(index: Any, *, start: int = 0, end: int | None = None) -> float | None:
    try:
        selected = index[start:end]
        if len(selected) < 2:
            return None
        first = pd.Timestamp(selected[0])
        last = pd.Timestamp(selected[-1])
        seconds = (last - first).total_seconds()
        return seconds / (365.2425 * 24 * 60 * 60) if seconds > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def annualized_return_percent(total_return_percent: float, years: float | None) -> float | None:
    if years is None or years <= 1.0:
        return None
    ending_multiple = 1.0 + float(total_return_percent) / 100.0
    if ending_multiple <= 0:
        return None
    return round((ending_multiple ** (1.0 / years) - 1.0) * 100.0, 2)


def allocation_metrics(
    *,
    account_equity: float,
    total_pnl: float,
    max_drawdown_dollars: float,
    risk_limits: RiskLimits | None,
    years: float | None,
) -> dict[str, float | bool | None]:
    account = max(0.0, float(account_equity))
    allocation = ticker_allocated_capital(account, risk_limits)
    account_return = float(total_pnl) / account * 100.0 if account > 0 else 0.0
    allocated_return = float(total_pnl) / allocation * 100.0 if allocation > 0 else 0.0
    allocated_drawdown = max(0.0, float(max_drawdown_dollars)) / allocation * 100.0 if allocation > 0 else 0.0
    exhausted = allocation > 0 and allocation + float(total_pnl) <= 0
    return {
        "allocated_capital": round(allocation, 2),
        "allocation_percent": ticker_allocation_percent(risk_limits),
        "account_return_pct": round(account_return, 2),
        "allocated_return_pct": round(allocated_return, 2),
        "allocated_max_drawdown_pct": round(allocated_drawdown, 2),
        "annualized_allocated_return_pct": (
            None if exhausted else annualized_return_percent(allocated_return, years)
        ),
        "period_years": round(years, 4) if years is not None else None,
        "allocated_capital_exhausted": exhausted,
    }
