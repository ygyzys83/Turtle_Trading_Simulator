from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import pandas as pd

from agentloop_trader.evaluation import WalkForwardResult, evaluate_walk_forward, synthetic_ohlc_frame
from agentloop_trader.models import RiskLimits, StrategyConfig
from agentloop_trader.strategy_runtime import STRATEGY_TYPES, _run_one


@dataclass(frozen=True)
class ParameterCandidate:
    config: StrategyConfig
    score: float
    status: str
    reason: str
    evaluation: WalkForwardResult | None = None


@dataclass(frozen=True)
class OptimizerCandidate:
    strategy_label: str
    strategy_type: str
    settings: dict[str, Any]
    score: float
    confidence: str
    reason: str
    concern: str
    test_return_percent: float
    test_trades: int
    test_win_rate_percent: float
    test_profit_factor: float
    test_max_drawdown_percent: float
    profitable_test_periods: int
    tested_periods: int
    train_return_percent: float
    train_trades: int
    stability: str
    recommended_risk_per_trade_percent: float


@dataclass(frozen=True)
class StrategyInputRecommendation:
    best: OptimizerCandidate | None
    candidates: list[OptimizerCandidate]
    summary: str
    tested_candidates: int
    rejected_candidates: int
    train_fraction: float


BOUNDED_ENTRY_WINDOWS = (15, 20, 25, 30)
BOUNDED_EXIT_WINDOWS = (5, 10, 15)
BOUNDED_ATR_MULTIPLIERS = (1.5, 2.0, 2.5, 3.0)
BOUNDED_MA_WINDOWS = (100, 150, 200, 250)
OPTIMIZER_ENTRY_WINDOWS = (10, 15, 20, 25, 30, 40, 50)
OPTIMIZER_EXIT_WINDOWS = (5, 10, 15, 20, 30)
OPTIMIZER_ATR_MULTIPLIERS = (1.0, 1.5, 2.0, 2.5, 3.0)
OPTIMIZER_TREND_WINDOWS = (50, 100, 150, 200)
OPTIMIZER_PULLBACK_WINDOWS = (10, 20, 30, 50, 100, 150, 200)
OPTIMIZER_MOMENTUM_WINDOWS = (3, 5, 10, 15, 20)


def generate_bounded_candidates(current: StrategyConfig, max_candidates: int = 12) -> list[StrategyConfig]:
    entry_options = _nearby(current.entry_window, BOUNDED_ENTRY_WINDOWS)
    exit_options = _nearby(current.exit_window, BOUNDED_EXIT_WINDOWS)
    atr_options = _nearby(current.atr_stop_multiplier, BOUNDED_ATR_MULTIPLIERS)
    ma_options = _nearby(current.moving_average_window, BOUNDED_MA_WINDOWS)

    configs = []
    seen = set()
    for entry_w, exit_w, atr_mult, ma_w in product(entry_options, exit_options, atr_options, ma_options):
        key = (entry_w, exit_w, atr_mult, ma_w)
        if key in seen:
            continue
        seen.add(key)
        configs.append(
            StrategyConfig(
                entry_window=entry_w,
                exit_window=exit_w,
                atr_stop_multiplier=atr_mult,
                risk_per_trade_pct=current.risk_per_trade_pct,
                moving_average_window=ma_w,
            )
        )

    configs.sort(key=lambda c: (
        abs(c.entry_window - current.entry_window),
        abs(c.exit_window - current.exit_window),
        abs(c.atr_stop_multiplier - current.atr_stop_multiplier),
        abs(c.moving_average_window - current.moving_average_window),
    ))
    return configs[:max_candidates]


def evaluate_parameter_candidates(
    current: StrategyConfig,
    account: float,
    risk_pct_dec: float,
    seed: int | None = None,
    market_data=None,
    train_fraction: float = 0.65,
    max_candidates: int = 12,
    risk_limits: RiskLimits | None = None,
) -> list[ParameterCandidate]:
    candidates = []
    for config in generate_bounded_candidates(current, max_candidates=max_candidates):
        try:
            evaluation = evaluate_walk_forward(
                account=account,
                entry_w=config.entry_window,
                exit_w=config.exit_window,
                atr_mult=config.atr_stop_multiplier,
                risk_pct_dec=risk_pct_dec,
                ma_w=config.moving_average_window,
                seed=seed,
                market_data=market_data,
                train_fraction=train_fraction,
                risk_limits=risk_limits,
            )
        except ValueError as exc:
            candidates.append(
                ParameterCandidate(
                    config=config,
                    score=-9999,
                    status="Rejected",
                    reason=str(exc),
                    evaluation=None,
                )
            )
            continue

        score = score_evaluation(evaluation)
        status = "Candidate"
        reason = "; ".join(evaluation.reasons)
        if evaluation.verdict == "Needs review":
            status = "Needs review"
        elif evaluation.verdict == "Inconclusive":
            status = "Inconclusive"
        elif evaluation.oos_stats["total_trades"] < 1:
            status = "Inconclusive"

        candidates.append(
            ParameterCandidate(
                config=config,
                score=score,
                status=status,
                reason=reason,
                evaluation=evaluation,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def score_evaluation(evaluation: WalkForwardResult) -> float:
    oos = evaluation.oos_stats
    train = evaluation.train_stats
    trade_bonus = min(oos["total_trades"], 5) * 0.5
    drawdown_penalty = oos["max_drawdown_pct"] * 0.35
    fragility_penalty = 0.0
    if train["profit_factor"] and oos["profit_factor"]:
        fragility_penalty = max(0.0, train["profit_factor"] - oos["profit_factor"]) * 1.5
    no_trade_penalty = 5.0 if oos["total_trades"] == 0 else 0.0
    verdict_bonus = {"Pass": 5.0, "Inconclusive": -1.0, "Needs review": -3.0}.get(evaluation.verdict, 0.0)
    return round(
        oos["return_pct"]
        + oos["profit_factor"]
        + trade_bonus
        + verdict_bonus
        - drawdown_penalty
        - fragility_penalty
        - no_trade_penalty,
        2,
    )


def recommend_candidate(candidates: list[ParameterCandidate]) -> ParameterCandidate | None:
    viable = [c for c in candidates if c.evaluation is not None and c.status in {"Candidate", "Needs review"}]
    return viable[0] if viable else None


def candidate_records(candidates: list[ParameterCandidate]) -> list[dict]:
    records = []
    for candidate in candidates:
        config = candidate.config
        evaluation = candidate.evaluation
        records.append({
            "Status": candidate.status,
            "Score": candidate.score,
            "Entry": config.entry_window,
            "Exit": config.exit_window,
            "ATR": config.atr_stop_multiplier,
            "Trend Filter": config.moving_average_window,
            "Newer Data Return %": "" if evaluation is None else evaluation.oos_stats["return_pct"],
            "Newer Data Trades": "" if evaluation is None else evaluation.oos_stats["total_trades"],
            "Profit Factor": "" if evaluation is None else evaluation.oos_stats["profit_factor"],
            "Worst Drop %": "" if evaluation is None else evaluation.oos_stats["max_drawdown_pct"],
            "Reason": candidate.reason,
        })
    return records


def recommendation_summary(candidate: ParameterCandidate | None) -> str:
    if candidate is None:
        return "No nearby setting looked good enough to suggest."
    c = candidate.config
    return (
        f"Try these settings next: entry {c.entry_window}, exit {c.exit_window}, "
        f"ATR {c.atr_stop_multiplier}, trend filter {c.moving_average_window}. "
        "This changes strategy settings only, not risk limits or order code."
    )


def optimize_strategy_inputs(
    market_data: pd.DataFrame | None,
    current_settings: dict[str, Any],
    account_equity: float,
    risk_limits: RiskLimits | None = None,
    *,
    train_fraction: float = 0.65,
    max_candidates_per_strategy: int = 18,
    min_test_trades: int = 2,
    target_max_drawdown_percent: float = 8.0,
) -> StrategyInputRecommendation:
    """Rank strategy settings using older data for fit and newer data for proof."""
    data = market_data.copy() if market_data is not None else synthetic_ohlc_frame(seed=42)
    candidates = generate_optimizer_settings(current_settings, max_candidates_per_strategy=max_candidates_per_strategy)
    ranked: list[OptimizerCandidate] = []
    rejected = 0
    for strategy_label, strategy_type, settings in candidates:
        try:
            candidate = _evaluate_optimizer_candidate(
                strategy_label=strategy_label,
                strategy_type=strategy_type,
                settings=settings,
                market_data=data,
                account_equity=account_equity,
                risk_limits=risk_limits,
                train_fraction=train_fraction,
                min_test_trades=min_test_trades,
                target_max_drawdown_percent=target_max_drawdown_percent,
            )
        except ValueError:
            rejected += 1
            continue
        ranked.append(candidate)

    ranked.sort(key=lambda row: row.score, reverse=True)
    best = ranked[0] if ranked else None
    summary = optimizer_summary(best)
    return StrategyInputRecommendation(
        best=best,
        candidates=ranked,
        summary=summary,
        tested_candidates=len(ranked),
        rejected_candidates=rejected,
        train_fraction=train_fraction,
    )


def generate_optimizer_settings(
    current_settings: dict[str, Any],
    *,
    max_candidates_per_strategy: int = 18,
) -> list[tuple[str, str, dict[str, Any]]]:
    base = _normal_settings(current_settings)
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for strategy_label, strategy_type in STRATEGY_TYPES.items():
        strategy_rows: list[dict[str, Any]] = []
        entry_options = _nearby(base["entry_window"], OPTIMIZER_ENTRY_WINDOWS)
        exit_options = _nearby(base["exit_window"], OPTIMIZER_EXIT_WINDOWS)
        atr_options = _nearby(base["atr_stop_multiplier"], OPTIMIZER_ATR_MULTIPLIERS)
        trend_options = _nearby(base["moving_average_window"], OPTIMIZER_TREND_WINDOWS)
        pullback_options = _nearby(base["pullback_average_length"], OPTIMIZER_PULLBACK_WINDOWS)
        momentum_options = _nearby(base["momentum_turn_length"], OPTIMIZER_MOMENTUM_WINDOWS)

        if strategy_type == "pullback":
            raw = product(exit_options, atr_options, trend_options, pullback_options, momentum_options)
            for exit_w, atr, trend_w, pullback_w, momentum_w in raw:
                row = base | {
                    "strategy_label": strategy_label,
                    "strategy_type": strategy_type,
                    "exit_window": exit_w,
                    "atr_stop_multiplier": atr,
                    "moving_average_window": trend_w,
                    "pullback_average_length": pullback_w,
                    "momentum_turn_length": momentum_w,
                }
                strategy_rows.append(row)
        elif strategy_type == "trendline_retest":
            raw = product(entry_options, exit_options, atr_options, trend_options, momentum_options)
            for entry_w, exit_w, atr, trend_w, momentum_w in raw:
                row = base | {
                    "strategy_label": strategy_label,
                    "strategy_type": strategy_type,
                    "entry_window": entry_w,
                    "exit_window": exit_w,
                    "atr_stop_multiplier": atr,
                    "moving_average_window": trend_w,
                    "momentum_turn_length": momentum_w,
                }
                strategy_rows.append(row)
        else:
            raw = product(entry_options, exit_options, atr_options, trend_options)
            for entry_w, exit_w, atr, trend_w in raw:
                row = base | {
                    "strategy_label": strategy_label,
                    "strategy_type": strategy_type,
                    "entry_window": entry_w,
                    "exit_window": exit_w,
                    "atr_stop_multiplier": atr,
                    "moving_average_window": trend_w,
                }
                strategy_rows.append(row)

        strategy_rows.sort(key=lambda row: _settings_distance(base, row))
        for row in strategy_rows[:max_candidates_per_strategy]:
            rows.append((strategy_label, strategy_type, row))
    return rows


def optimizer_summary(candidate: OptimizerCandidate | None) -> str:
    if candidate is None:
        return "No strategy settings had enough newer-data evidence to recommend."
    return (
        f"Best current fit: {candidate.strategy_label}. "
        f"Use buy lookback {candidate.settings['entry_window']}, sell exit {candidate.settings['exit_window']}, "
        f"stop {candidate.settings['atr_stop_multiplier']:.2f}x ATR, trend filter {candidate.settings['moving_average_window']}, "
        f"pullback average {candidate.settings['pullback_average_length']}, and momentum turn {candidate.settings['momentum_turn_length']}. "
        f"Newer-data return was {candidate.test_return_percent:.2f}% with a {candidate.test_max_drawdown_percent:.2f}% worst drop."
    )


def optimizer_recommendation_records(result: StrategyInputRecommendation) -> list[dict[str, Any]]:
    candidate = result.best
    if candidate is None:
        return [{"Item": "Recommendation", "Value": "No recommendation", "Plain English": result.summary}]
    settings = candidate.settings
    return [
        {"Item": "Best strategy", "Value": candidate.strategy_label, "Plain English": candidate.reason},
        {"Item": "Confidence", "Value": candidate.confidence, "Plain English": candidate.stability},
        {"Item": "Buy lookback", "Value": f"{settings['entry_window']} bars", "Plain English": "Bars used for breakout or trendline entry logic."},
        {"Item": "Sell exit", "Value": f"{settings['exit_window']} bars", "Plain English": "Bars used to calculate the strategy exit line."},
        {"Item": "Stop distance", "Value": f"{settings['atr_stop_multiplier']:.2f}x ATR", "Plain English": "Initial stop distance used for sizing and protection."},
        {"Item": "Trend filter", "Value": f"{settings['moving_average_window']} bars", "Plain English": "Trend average used before allowing long trades."},
        {"Item": "Pullback average", "Value": f"{settings['pullback_average_length']} bars", "Plain English": "Only affects Trend pullback continuation."},
        {"Item": "Momentum turn", "Value": f"{settings['momentum_turn_length']} bars", "Plain English": "Used by Trend pullback continuation and Trendline retest continuation."},
        {"Item": "Suggested strategy risk", "Value": f"{candidate.recommended_risk_per_trade_percent:.2f}%", "Plain English": "Risk size suggested by newer-data drawdown. Account risk limits still apply."},
        {"Item": "Main concern", "Value": candidate.concern, "Plain English": "What to watch before trusting this setting."},
    ]


def optimizer_candidate_records(candidates: list[OptimizerCandidate], limit: int = 12) -> list[dict[str, Any]]:
    return [
        {
            "Rank": index + 1,
            "Strategy": candidate.strategy_label,
            "Score": candidate.score,
            "Confidence": candidate.confidence,
            "Buy Lookback": candidate.settings["entry_window"],
            "Sell Exit": candidate.settings["exit_window"],
            "Stop ATR": candidate.settings["atr_stop_multiplier"],
            "Trend Filter": candidate.settings["moving_average_window"],
            "Pullback Average": candidate.settings["pullback_average_length"],
            "Momentum Turn": candidate.settings["momentum_turn_length"],
            "Newer Return %": candidate.test_return_percent,
            "Newer Trades": candidate.test_trades,
            "Profit Factor": candidate.test_profit_factor,
            "Worst Drop %": candidate.test_max_drawdown_percent,
            "Plain English": candidate.reason,
        }
        for index, candidate in enumerate(candidates[:limit])
    ]


def _evaluate_optimizer_candidate(
    *,
    strategy_label: str,
    strategy_type: str,
    settings: dict[str, Any],
    market_data: pd.DataFrame,
    account_equity: float,
    risk_limits: RiskLimits | None,
    train_fraction: float,
    min_test_trades: int,
    target_max_drawdown_percent: float,
) -> OptimizerCandidate:
    data = market_data.copy()
    total_bars = len(data)
    warmup_bars = _warmup_bars(settings)
    split_index = int(total_bars * train_fraction)
    if split_index < warmup_bars or total_bars - split_index < 30:
        raise ValueError("Not enough bars for strategy input search.")

    train_data = data.iloc[:split_index].copy()
    train_data.attrs["symbol"] = data.attrs.get("symbol", "MARKET")
    train_result = _run_one(strategy_type, train_data, settings, account_equity, risk_limits)
    train_stats = train_result["stats"]

    oos_start = max(0, split_index - warmup_bars)
    warmup_offset = split_index - oos_start
    oos_data = data.iloc[oos_start:].copy()
    oos_data.attrs["symbol"] = data.attrs.get("symbol", "MARKET")
    oos_result = _run_one(strategy_type, oos_data, settings, account_equity, risk_limits)
    oos_trades = [
        trade for trade in oos_result["trade_log"]
        if int(trade.get("entry_bar", 0)) >= warmup_offset
    ]
    oos_stats = _closed_trade_stats(account_equity, oos_trades, total_bars - split_index)

    test_trades = int(oos_stats["total_trades"])
    return_pct = float(oos_stats["return_pct"])
    profit_factor = float(oos_stats["profit_factor"])
    drawdown = float(oos_stats["max_drawdown_pct"])
    win_rate = float(oos_stats["win_rate"])
    train_return = float(train_stats["return_pct"])
    train_trades = int(train_stats["total_trades"])
    profitable_periods = 1 if return_pct > 0 else 0
    trade_penalty = 8.0 if test_trades < min_test_trades else 0.0
    drawdown_penalty = max(0.0, drawdown - target_max_drawdown_percent) * 0.9
    degradation_penalty = max(0.0, train_return - return_pct) * 0.15
    profit_factor_bonus = min(profit_factor, 3.0) * 2.0 if profit_factor > 0 else 0.0
    trade_bonus = min(test_trades, 8) * 0.4
    score = round(return_pct + profit_factor_bonus + trade_bonus - drawdown * 0.35 - drawdown_penalty - degradation_penalty - trade_penalty, 2)
    confidence = _optimizer_confidence(score, test_trades, drawdown, return_pct)
    concern = _optimizer_concern(test_trades, drawdown, return_pct, train_return, min_test_trades)
    reason = (
        f"Newer-data return {return_pct:.2f}%, {test_trades} trades, "
        f"profit factor {profit_factor:.2f}, worst drop {drawdown:.2f}%."
    )
    recommended_risk = _recommended_risk(settings, drawdown, target_max_drawdown_percent)
    return OptimizerCandidate(
        strategy_label=strategy_label,
        strategy_type=strategy_type,
        settings=settings,
        score=score,
        confidence=confidence,
        reason=reason,
        concern=concern,
        test_return_percent=round(return_pct, 2),
        test_trades=test_trades,
        test_win_rate_percent=round(win_rate, 2),
        test_profit_factor=round(profit_factor, 2),
        test_max_drawdown_percent=round(drawdown, 2),
        profitable_test_periods=profitable_periods,
        tested_periods=1,
        train_return_percent=round(train_return, 2),
        train_trades=train_trades,
        stability=_stability_read(score, test_trades, drawdown, train_return, return_pct),
        recommended_risk_per_trade_percent=recommended_risk,
    )


def _closed_trade_stats(account: float, trade_log: list[dict], eval_bars: int) -> dict[str, Any]:
    wins = [trade for trade in trade_log if float(trade.get("pnl", 0)) > 0]
    losses = [trade for trade in trade_log if float(trade.get("pnl", 0)) <= 0]
    total_pnl = round(sum(float(trade.get("pnl", 0)) for trade in trade_log), 2)
    gross_wins = sum(float(trade.get("pnl", 0)) for trade in wins)
    gross_losses = abs(sum(float(trade.get("pnl", 0)) for trade in losses))
    equity = account
    peak = account
    max_drawdown = 0.0
    for trade in trade_log:
        equity += float(trade.get("pnl", 0))
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = min(max_drawdown, (equity - peak) / peak)
    exposure_bars = sum(max(0, int(trade.get("exit_bar", 0)) - int(trade.get("entry_bar", 0))) for trade in trade_log)
    avg_loss = round(sum(float(trade.get("pnl", 0)) for trade in losses) / len(losses), 2) if losses else 0
    avg_win = round(gross_wins / len(wins), 2) if wins else 0
    return {
        "final_equity": round(account + total_pnl),
        "total_pnl": round(total_pnl),
        "return_pct": round(total_pnl / account * 100, 2) if account else 0,
        "win_rate": round(len(wins) / len(trade_log) * 100) if trade_log else 0,
        "wins": len(wins),
        "losses": len(losses),
        "total_trades": len(trade_log),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "rr_ratio": round(abs(avg_win / avg_loss), 2) if avg_loss else 0,
        "profit_factor": round(gross_wins / gross_losses, 2) if gross_losses else 0,
        "max_drawdown_pct": round(abs(max_drawdown) * 100, 2),
        "exposure_pct": round(exposure_bars / eval_bars * 100, 2) if eval_bars else 0,
    }


def _normal_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy_label": str(settings.get("strategy_label", "Trendline retest continuation")),
        "strategy_type": str(settings.get("strategy_type", "trendline_retest")),
        "entry_window": int(settings.get("entry_window", 20)),
        "exit_window": int(settings.get("exit_window", 10)),
        "atr_stop_multiplier": float(settings.get("atr_stop_multiplier", 2.0)),
        "risk_per_trade_pct": float(settings.get("risk_per_trade_pct", 1.0)),
        "moving_average_window": int(settings.get("moving_average_window", 50)),
        "pullback_average_length": int(settings.get("pullback_average_length", 20)),
        "momentum_turn_length": int(settings.get("momentum_turn_length", 10)),
    }


def _settings_distance(base: dict[str, Any], row: dict[str, Any]) -> tuple[float, ...]:
    return (
        abs(row["entry_window"] - base["entry_window"]) / 10,
        abs(row["exit_window"] - base["exit_window"]) / 5,
        abs(row["atr_stop_multiplier"] - base["atr_stop_multiplier"]),
        abs(row["moving_average_window"] - base["moving_average_window"]) / 50,
        abs(row["pullback_average_length"] - base["pullback_average_length"]) / 20,
        abs(row["momentum_turn_length"] - base["momentum_turn_length"]) / 5,
    )


def _warmup_bars(settings: dict[str, Any]) -> int:
    return max(
        int(settings.get("entry_window", 20)),
        int(settings.get("exit_window", 10)),
        int(settings.get("moving_average_window", 50)),
        int(settings.get("pullback_average_length", 20)),
        int(settings.get("momentum_turn_length", 10)),
        14,
    ) + 4


def _optimizer_confidence(score: float, trades: int, drawdown: float, return_pct: float) -> str:
    if trades <= 0:
        return "Low"
    if score >= 8 and trades >= 3 and return_pct > 0 and drawdown <= 10:
        return "High"
    if score >= 2 and return_pct > 0:
        return "Medium"
    return "Low"


def _optimizer_concern(trades: int, drawdown: float, return_pct: float, train_return: float, min_trades: int) -> str:
    if trades < min_trades:
        return "Too few newer-data trades to trust strongly."
    if return_pct <= 0:
        return "Newer data did not make money."
    if drawdown > 10:
        return "Worst drop may be too large for unattended trading."
    if train_return > return_pct * 2 and return_pct > 0:
        return "Older data was much stronger than newer data."
    return "No major issue from this bounded search."


def _stability_read(score: float, trades: int, drawdown: float, train_return: float, return_pct: float) -> str:
    if trades <= 0:
        return "Low confidence because the newer test period had no completed trades."
    if return_pct > 0 and drawdown <= 8 and train_return >= 0 and score >= 5:
        return "Stable enough for paper testing; newer data stayed profitable with controlled drawdown."
    if return_pct > 0:
        return "Usable for paper testing, but review trade count and drawdown."
    return "Weak stability; do not treat this as a strong recommendation."


def _recommended_risk(settings: dict[str, Any], drawdown: float, target_drawdown: float) -> float:
    current = float(settings.get("risk_per_trade_pct", 1.0))
    if drawdown <= 0:
        return round(min(current, 1.0), 2)
    scale = max(0.25, min(1.0, target_drawdown / max(drawdown, 0.01)))
    return round(max(0.25, min(current, current * scale)), 2)


def _nearby(current_value, allowed_values):
    values = sorted(allowed_values, key=lambda value: (abs(value - current_value), value))
    return values[:3]
