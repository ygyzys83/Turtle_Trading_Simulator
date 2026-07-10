from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from agentloop_trader.models import PACIFIC_TIME, RiskLimits
from agentloop_trader.research_agent import build_research_agent_report
from agentloop_trader.strategy_runtime import run_strategy_suite, trade_intent_to_record


DEFAULT_SCAN_SYMBOLS = (
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "AMD",
    "TSLA", "NFLX", "JPM", "COST", "WMT", "XOM", "IBM",
)
DEFAULT_SCAN_PATH = Path("automation_logs") / "scanner_candidates.json"


@dataclass(frozen=True)
class ScanCandidate:
    symbol: str
    decision: str
    best_strategy: str
    selected_strategy: str
    fit_score: float
    last_price: float
    atr_percent: float
    liquidity: str
    backtest_return_percent: float
    win_rate_percent: float
    profit_factor: float
    max_drawdown_percent: float
    reason: str
    trade_intent: dict[str, Any] | None
    scanned_at: str


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _decision(live: dict[str, Any], fit_score: float) -> str:
    if str(live.get("signal", "")).lower() == "long" and live.get("trade_intent") is not None:
        return "TRADE"
    requirements = live.get("buy_requirements") or {}
    passed = sum(bool(value) for value in requirements.values())
    if requirements and passed >= max(1, len(requirements) - 1):
        return "WATCH"
    return "WAIT" if fit_score > 0 else "NO DATA"


def scan_symbol(
    symbol: str,
    market_data: pd.DataFrame,
    settings: dict[str, Any],
    account_equity: float,
    risk_limits: RiskLimits | None = None,
) -> ScanCandidate:
    clean = str(symbol).strip().upper()
    results = run_strategy_suite(market_data, settings, account_equity, risk_limits)
    selected_label = str(settings.get("strategy_label", "Trendline retest continuation"))
    selected = results.get(selected_label) or next(iter(results.values()))
    report = build_research_agent_report(
        ticker=clean,
        selected_strategy=selected_label,
        strategy_results=results,
        setup_rows=[],
        final_read="WAIT",
        decision_detail="Scanner comparison only; run the ticker on New Trade before sending an order.",
        next_action="Open the ticker in New Trade for full risk checks.",
    )
    best = results.get(report.best_strategy) or selected
    live = best.get("live") or {}
    stats = best.get("stats") or {}
    best_fit = next((fit for fit in report.strategy_fits if fit.strategy == report.best_strategy), None)
    price = _number(live.get("last_p"))
    atr = _number(live.get("last_atr"))
    decision = _decision(live, best_fit.score if best_fit else 0.0)
    reason = str(live.get("no_trade_reason") or (best_fit.reason if best_fit else "No strategy result."))
    if decision == "TRADE":
        reason = f"{report.best_strategy} has a current buy setup. Confirm it on New Trade before ordering."
    return ScanCandidate(
        symbol=clean,
        decision=decision,
        best_strategy=report.best_strategy,
        selected_strategy=selected_label,
        fit_score=best_fit.score if best_fit else 0.0,
        last_price=round(price, 4),
        atr_percent=round(atr / price * 100, 2) if price else 0.0,
        liquidity=str(live.get("liquidity_status", "Unknown")),
        backtest_return_percent=round(_number(stats.get("return_pct")), 2),
        win_rate_percent=round(_number(stats.get("win_rate")), 2),
        profit_factor=round(_number(stats.get("profit_factor")), 2),
        max_drawdown_percent=round(_number(stats.get("max_drawdown_pct")), 2),
        reason=reason,
        trade_intent=trade_intent_to_record(live.get("trade_intent")),
        scanned_at=datetime.now(PACIFIC_TIME).isoformat(),
    )


def scan_universe(
    symbols: list[str] | tuple[str, ...],
    fetch_bars: Callable[[str], pd.DataFrame],
    settings: dict[str, Any],
    account_equity: float,
    risk_limits: RiskLimits | None = None,
    *,
    max_symbols: int = 30,
) -> tuple[list[ScanCandidate], list[dict[str, str]]]:
    candidates: list[ScanCandidate] = []
    errors: list[dict[str, str]] = []
    unique = list(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()))[:max_symbols]
    for symbol in unique:
        try:
            candidates.append(scan_symbol(symbol, fetch_bars(symbol), settings, account_equity, risk_limits))
        except Exception as exc:
            errors.append({"Ticker": symbol, "Problem": str(exc)})
    decision_rank = {"TRADE": 3, "WATCH": 2, "WAIT": 1, "NO DATA": 0}
    candidates.sort(key=lambda row: (decision_rank.get(row.decision, 0), row.fit_score, row.profit_factor, row.backtest_return_percent), reverse=True)
    return candidates, errors


def scanner_records(candidates: list[ScanCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "Ticker": row.symbol,
            "Current Read": row.decision,
            "Best Strategy": row.best_strategy,
            "Fit": row.fit_score,
            "Price": row.last_price,
            "ATR Percent": row.atr_percent,
            "Liquidity": row.liquidity,
            "Backtest Return Percent": row.backtest_return_percent,
            "Win Rate Percent": row.win_rate_percent,
            "Profit Factor": row.profit_factor,
            "Worst Drop Percent": row.max_drawdown_percent,
            "Plain English": row.reason,
        }
        for row in candidates
    ]


class ScannerCandidateStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else DEFAULT_SCAN_PATH

    def save(self, candidates: list[ScanCandidate], errors: list[dict[str, str]] | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": datetime.now(PACIFIC_TIME).isoformat(),
            "candidates": [asdict(candidate) for candidate in candidates],
            "errors": list(errors or []),
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)

    def read(self) -> tuple[list[ScanCandidate], list[dict[str, str]]]:
        if not self.path.exists():
            return [], []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return [ScanCandidate(**row) for row in payload.get("candidates", [])], list(payload.get("errors", []))
        except (json.JSONDecodeError, TypeError):
            return [], []
