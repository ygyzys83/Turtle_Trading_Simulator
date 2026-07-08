from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from agentloop_trader.evaluation import WalkForwardResult, evaluate_walk_forward
from agentloop_trader.models import RiskLimits, StrategyConfig


@dataclass(frozen=True)
class ParameterCandidate:
    config: StrategyConfig
    score: float
    status: str
    reason: str
    evaluation: WalkForwardResult | None = None


BOUNDED_ENTRY_WINDOWS = (15, 20, 25, 30)
BOUNDED_EXIT_WINDOWS = (5, 10, 15)
BOUNDED_ATR_MULTIPLIERS = (1.5, 2.0, 2.5, 3.0)
BOUNDED_MA_WINDOWS = (100, 150, 200, 250)


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
            "SMA": config.moving_average_window,
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
        f"ATR {c.atr_stop_multiplier}, SMA {c.moving_average_window}. "
        "This changes strategy settings only, not risk limits or order code."
    )


def _nearby(current_value, allowed_values):
    values = sorted(allowed_values, key=lambda value: (abs(value - current_value), value))
    return values[:3]
