from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StrategyFit:
    strategy: str
    status: str
    score: float
    reason: str


@dataclass(frozen=True)
class ResearchAgentReport:
    ticker: str
    final_read: str
    best_strategy: str
    best_strategy_reason: str
    thesis: str
    next_action: str
    event_risk: str = "Not connected"
    event_risk_detail: str = "No earnings, news, or calendar feed is connected yet. This is informational only."
    strategy_fits: list[StrategyFit] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _setup_row(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    target = name.lower()
    return next((row for row in rows if str(row.get("Read", "")).lower() == target), {})


def _setup_status(rows: list[dict[str, Any]], name: str, default: str = "Unknown") -> str:
    return str(_setup_row(rows, name).get("Status", default))


def _setup_detail(rows: list[dict[str, Any]], name: str, default: str) -> str:
    return str(_setup_row(rows, name).get("Plain English", default))


def _requirements_pass_ratio(live: dict[str, Any]) -> tuple[int, int]:
    requirements = live.get("buy_requirements") or {}
    if not isinstance(requirements, dict) or not requirements:
        return 0, 0
    passed = sum(1 for value in requirements.values() if bool(value))
    return passed, len(requirements)


def _strategy_score(name: str, result: dict[str, Any]) -> StrategyFit:
    live = result.get("live") or {}
    stats = result.get("stats") or {}
    signal = str(live.get("signal", "flat")).lower()
    passed_rules, total_rules = _requirements_pass_ratio(live)
    rule_ratio = passed_rules / total_rules if total_rules else 0.0
    return_pct = _as_float(stats.get("return_pct"))
    win_rate = _as_float(stats.get("win_rate"))
    profit_factor = _as_float(stats.get("profit_factor"))
    max_drawdown = _as_float(stats.get("max_drawdown_pct"))
    trades = int(_as_float(stats.get("total_trades")))

    score = 0.0
    if signal == "long":
        score += 3.0
    score += rule_ratio * 2.0
    if trades >= 3:
        score += 0.5
    elif trades > 0:
        score -= 1.0
    score += max(-1.5, min(1.5, return_pct / 10.0))
    if profit_factor >= 1.2:
        score += 1.0
    elif trades > 0 and profit_factor < 1.0:
        score -= 1.0
    if win_rate >= 50:
        score += 0.5
    score -= min(2.0, max_drawdown / 20.0)

    status = "Buy setup" if signal == "long" else "Waiting"
    reason = (
        f"{passed_rules}/{total_rules} buy rules pass; "
        f"{trades} completed trades, return {return_pct:.2f}%, win rate {win_rate:.0f}%, "
        f"profit factor {profit_factor:.2f}, worst drop {max_drawdown:.2f}%."
    )
    return StrategyFit(strategy=name, status=status, score=round(score, 2), reason=reason)


def _risk_reward_read(stats: dict[str, Any]) -> tuple[str, str]:
    profit_factor = _as_float(stats.get("profit_factor"))
    win_rate = _as_float(stats.get("win_rate"))
    return_pct = _as_float(stats.get("return_pct"))
    max_drawdown = _as_float(stats.get("max_drawdown_pct"))
    trades = int(_as_float(stats.get("total_trades")))
    if trades <= 0:
        return "Not enough trades", "The selected settings have no completed historical trades to judge yet."
    read = f"PF {profit_factor:.2f} / Win {win_rate:.0f}%"
    detail = f"Backtest return is {return_pct:.2f}% with a worst drop of {max_drawdown:.2f}%."
    return read, detail


def build_research_agent_report(
    ticker: str,
    selected_strategy: str,
    strategy_results: dict[str, dict[str, Any]],
    setup_rows: list[dict[str, Any]],
    final_read: str,
    decision_detail: str,
    next_action: str,
) -> ResearchAgentReport:
    fits = [_strategy_score(name, result) for name, result in strategy_results.items()]
    fits.sort(key=lambda fit: fit.score, reverse=True)
    best = fits[0] if fits else StrategyFit(selected_strategy, "Unknown", 0.0, "No strategy results are available.")
    selected = strategy_results.get(selected_strategy, {})
    selected_stats = selected.get("stats") or {}
    risk_reward_read, risk_reward_detail = _risk_reward_read(selected_stats)

    trend = _setup_status(setup_rows, "Trend")
    trend_detail = _setup_detail(setup_rows, "Trend", "Trend check is not available.")
    overall = _setup_status(setup_rows, "Overall")
    setup_detail = _setup_detail(setup_rows, "Overall", "Setup grade is not available.")
    volatility = _setup_status(setup_rows, "Volatility")
    volatility_detail = _setup_detail(setup_rows, "Volatility", "Volatility check is not available.")
    liquidity = _setup_status(setup_rows, "Liquidity")
    liquidity_detail = _setup_detail(setup_rows, "Liquidity", "Liquidity check is not available.")

    best_reason = f"Best current fit: {best.strategy}. {best.reason}"
    thesis = (
        f"{ticker.upper()}: {final_read}. {decision_detail} "
        f"Best current fit is {best.strategy} based on current setup, backtest performance, and risk quality."
    )
    rows = [
        {"Area": "Final answer", "Read": final_read, "Plain English": decision_detail},
        {
            "Area": "Selected strategy",
            "Read": selected_strategy,
            "Plain English": "The exact strategy selected in the sidebar and used for the TRADE or WAIT decision.",
        },
        {
            "Area": "Best current fit across all strategies",
            "Read": best.strategy,
            "Plain English": f"The app compared all five strategies using the current inputs. {best.reason}",
        },
        {"Area": "Trend", "Read": trend, "Plain English": trend_detail},
        {"Area": "Setup", "Read": overall, "Plain English": setup_detail},
        {"Area": "Volatility", "Read": volatility, "Plain English": volatility_detail},
        {"Area": "Liquidity", "Read": liquidity, "Plain English": liquidity_detail},
        {"Area": "Risk / reward", "Read": risk_reward_read, "Plain English": risk_reward_detail},
        {"Area": "Event risk", "Read": "Not connected", "Plain English": "No earnings, news, or calendar feed is connected yet. This does not block trades."},
        {"Area": "Next action", "Read": next_action, "Plain English": "What the app expects you or automation to do next."},
    ]
    return ResearchAgentReport(
        ticker=ticker.upper(),
        final_read=final_read,
        best_strategy=best.strategy,
        best_strategy_reason=best_reason,
        thesis=thesis,
        next_action=next_action,
        strategy_fits=fits,
        rows=rows,
    )


def research_agent_records(report: ResearchAgentReport) -> list[dict[str, Any]]:
    return list(report.rows)


def strategy_fit_records(report: ResearchAgentReport) -> list[dict[str, Any]]:
    return [
        {
            "Strategy": fit.strategy,
            "Current Setup": fit.status,
            "Fit Score": fit.score,
            "Plain English": fit.reason,
            "Best": "Yes" if fit.strategy == report.best_strategy else "No",
        }
        for fit in report.strategy_fits
    ]
