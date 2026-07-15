from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from itertools import product
from math import e, sqrt
from statistics import NormalDist, median
from typing import Any, Callable

import numpy as np
import pandas as pd

from agentloop_trader.evaluation import WalkForwardResult, evaluate_walk_forward, synthetic_ohlc_frame
from agentloop_trader.assets import normalize_asset_class
from agentloop_trader.fees import estimate_alpaca_round_trip_fees
from agentloop_trader.models import RiskLimits, StrategyConfig
from agentloop_trader.performance import (
    allocation_metrics,
    annualized_return_percent,
    elapsed_years,
    ticker_allocated_capital,
)
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
    train_benchmark_return_percent: float
    train_excess_return_percent: float
    stability: str
    recommended_risk_per_trade_percent: float
    plateau_neighbors: int = 1
    plateau_median_return_percent: float = 0.0
    plateau_profitable_percent: float = 0.0
    rolling_profitable_windows: int = 0
    rolling_windows: int = 0
    rolling_median_return_percent: float = 0.0
    rolling_worst_drawdown_percent: float = 0.0
    validation_trade_returns: tuple[float, ...] = ()
    benchmark_return_percent: float = 0.0
    benchmark_max_drawdown_percent: float = 0.0
    excess_return_percent: float = 0.0
    drawdown_advantage_percent: float = 0.0
    test_account_return_percent: float = 0.0
    test_annualized_return_percent: float | None = None
    benchmark_account_return_percent: float = 0.0
    benchmark_annualized_return_percent: float | None = None
    allocated_capital: float = 0.0


@dataclass(frozen=True)
class LockedTestResult:
    return_percent: float
    trades: int
    win_rate_percent: float
    profit_factor: float
    max_drawdown_percent: float
    passed: bool
    detail: str
    benchmark_return_percent: float = 0.0
    benchmark_max_drawdown_percent: float = 0.0
    excess_return_percent: float = 0.0
    account_return_percent: float = 0.0
    annualized_return_percent: float | None = None
    benchmark_annualized_return_percent: float | None = None


@dataclass(frozen=True)
class BootstrapResult:
    samples: int
    completed_trades: int
    median_return_percent: float
    fifth_percentile_return_percent: float
    loss_probability_percent: float
    ninety_fifth_percentile_drawdown_percent: float


@dataclass(frozen=True)
class TrialAdjustment:
    tested_candidates: int
    trade_sharpe: float
    expected_best_sharpe_from_search: float
    deflated_sharpe_probability_percent: float
    evidence: str


@dataclass(frozen=True)
class CandidateDiagnostics:
    after_cost_expectancy_percent: float
    best_trade_removed_return_percent: float
    best_trade_share_percent: float
    account_drawdown_percent: float
    slippage_survival_bps: int
    profitable_regime_percent: float | None
    return_to_drawdown: float


@dataclass(frozen=True)
class RobustnessEvidence:
    parameter_range: dict[str, tuple[float, float]]
    locked_test: LockedTestResult | None
    stress_rows: list[dict[str, Any]] = field(default_factory=list)
    regime_rows: list[dict[str, Any]] = field(default_factory=list)
    bootstrap: BootstrapResult | None = None
    trial_adjustment: TrialAdjustment | None = None
    diagnostics: CandidateDiagnostics | None = None


@dataclass(frozen=True)
class StrategyInputRecommendation:
    best: OptimizerCandidate | None
    candidates: list[OptimizerCandidate]
    summary: str
    tested_candidates: int
    rejected_candidates: int
    train_fraction: float
    locked_fraction: float = 0.20
    robustness: RobustnessEvidence | None = None


@dataclass(frozen=True)
class BuyAndHoldBenchmark:
    return_percent: float
    max_drawdown_percent: float
    final_equity: float
    start_price: float
    end_price: float
    bars: int
    account_return_percent: float = 0.0
    annualized_return_percent: float | None = None
    allocated_capital: float = 0.0
    period_years: float | None = None
    estimated_alpaca_fees: float = 0.0


@dataclass(frozen=True)
class IntervalOptimizationResult:
    interval: str
    history: str
    recommendation: StrategyInputRecommendation
    selection_score: float
    evidence_status: str
    comparison_history: str = "Latest 2 years"
    comparison_return_percent: float = 0.0
    comparison_benchmark_return_percent: float = 0.0
    comparison_excess_return_percent: float = 0.0
    comparison_max_drawdown_percent: float = 0.0
    comparison_trades: int = 0
    durability_return_percent: float = 0.0
    durability_benchmark_return_percent: float = 0.0
    durability_excess_return_percent: float = 0.0
    durability_max_drawdown_percent: float = 0.0
    durability_trades: int = 0
    durability_annualized_return_percent: float | None = None
    durability_benchmark_annualized_return_percent: float | None = None
    durability_annualized_excess_percent: float | None = None
    older_dates: str = "Older prices"
    newer_dates: str = "Newer prices"
    latest_dates: str = "Latest prices"


@dataclass(frozen=True)
class MultiIntervalRecommendation:
    best_interval: str
    best_history: str
    best_result: StrategyInputRecommendation
    interval_results: tuple[IntervalOptimizationResult, ...]
    summary: str


@dataclass(frozen=True)
class StrategySearchResult:
    strategy_label: str
    strategy_type: str
    best_interval: str
    best_history: str
    best_result: StrategyInputRecommendation
    interval_results: tuple[IntervalOptimizationResult, ...]


@dataclass(frozen=True)
class MultiStrategySearchResult:
    strategy_results: tuple[StrategySearchResult, ...]
    summary: str

    @property
    def best_strategy(self) -> StrategySearchResult | None:
        return self.strategy_results[0] if self.strategy_results else None


@dataclass(frozen=True)
class CrossTickerResult:
    tested_tickers: int
    profitable_tickers: int
    median_return_percent: float
    worst_drawdown_percent: float
    rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CandidateVerdict:
    tier: str
    summary: str
    passed: tuple[str, ...] = ()
    uncertain: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()


BOUNDED_ENTRY_WINDOWS = (15, 20, 25, 30)
BOUNDED_EXIT_WINDOWS = (5, 10, 15)
BOUNDED_ATR_MULTIPLIERS = (1.5, 2.0, 2.5, 3.0)
BOUNDED_MA_WINDOWS = (10, 20, 50, 100, 150, 200, 250)
OPTIMIZER_ENTRY_WINDOWS = (10, 15, 20, 25, 30, 40, 50)
OPTIMIZER_EXIT_WINDOWS = (5, 10, 15, 20, 30)
OPTIMIZER_ATR_MULTIPLIERS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
OPTIMIZER_TREND_WINDOWS = (10, 20, 50, 100, 150, 200)
OPTIMIZER_PULLBACK_WINDOWS = (10, 20, 30, 50, 100, 150, 200)
OPTIMIZER_MOMENTUM_WINDOWS = (3, 5, 10, 15, 20)
OPTIMIZER_STRATEGY_TYPES = {
    label: strategy_type
    for label, strategy_type in STRATEGY_TYPES.items()
    if strategy_type != "rsi_scalp"
}
OPTIMIZER_LOCAL_SETTINGS_PER_STRATEGY = 24


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
    drawdown_penalty = oos["allocated_max_drawdown_pct"] * 0.35
    fragility_penalty = 0.0
    if train["profit_factor"] and oos["profit_factor"]:
        fragility_penalty = max(0.0, train["profit_factor"] - oos["profit_factor"]) * 1.5
    insufficient_trade_penalty = 8.0 if oos["total_trades"] < 2 else 0.0
    verdict_bonus = {"Pass": 5.0, "Inconclusive": -1.0, "Needs review": -3.0}.get(evaluation.verdict, 0.0)
    return round(
        oos["allocated_return_pct"]
        + min(float(oos["profit_factor"]), 3.0)
        + trade_bonus
        + verdict_bonus
        - drawdown_penalty
        - fragility_penalty
        - insufficient_trade_penalty,
        2,
    )


def historical_trade_evidence(trades: int) -> str:
    if trades >= 35:
        return "Enough historical trades"
    if trades >= 15:
        return "Smaller historical sample"
    return "Very small historical sample"


def _trade_evidence_rank(trades: int) -> int:
    if trades >= 35:
        return 2
    if trades >= 15:
        return 1
    return 0


def _discovery_selection_key(candidate: OptimizerCandidate) -> tuple[float, ...]:
    """Rank the broad discovery search without using the later price sections."""
    return (
        float(_trade_evidence_rank(candidate.train_trades)),
        float(candidate.train_excess_return_percent),
        float(candidate.train_return_percent),
        float(candidate.train_trades),
    )


def _validation_selection_key(candidate: OptimizerCandidate) -> tuple[float, ...]:
    """Choose among older-price finalists using the unchanged newer-price result."""
    return (
        float(_trade_evidence_rank(candidate.train_trades)),
        float(candidate.excess_return_percent),
        float(candidate.test_return_percent),
        float(candidate.test_trades),
        float(candidate.train_excess_return_percent),
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
            "Newer Allocated Return %": "" if evaluation is None else evaluation.oos_stats["allocated_return_pct"],
            "Newer Data Trades": "" if evaluation is None else evaluation.oos_stats["total_trades"],
            "Profit Factor": "" if evaluation is None else evaluation.oos_stats["profit_factor"],
            "Allocated Worst Drop %": "" if evaluation is None else evaluation.oos_stats["allocated_max_drawdown_pct"],
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
    train_fraction: float = 0.55,
    max_candidates_per_strategy: int = 18,
    max_local_candidates_per_strategy: int = OPTIMIZER_LOCAL_SETTINGS_PER_STRATEGY,
    min_test_trades: int = 2,
    target_max_drawdown_percent: float = 8.0,
    locked_fraction: float = 0.20,
    rolling_windows: int = 4,
    bootstrap_samples: int = 1000,
    strategy_types: set[str] | None = None,
) -> StrategyInputRecommendation:
    """Search broadly, verify nearby settings, choose on 25%, and report the latest 20%."""
    data = market_data.copy() if market_data is not None else synthetic_ohlc_frame(seed=42)
    candidates = generate_optimizer_settings(
        current_settings,
        max_candidates_per_strategy=max_candidates_per_strategy,
        strategy_types=strategy_types,
    )
    broad_ranked: list[OptimizerCandidate] = []
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
                locked_fraction=locked_fraction,
                min_test_trades=min_test_trades,
                target_max_drawdown_percent=target_max_drawdown_percent,
            )
        except ValueError:
            rejected += 1
            continue
        broad_ranked.append(candidate)

    ranked = list(broad_ranked)
    tested_settings = {
        _optimizer_settings_identity(row.strategy_type, row.settings)
        for row in broad_ranked
    }
    for strategy_type in sorted({row.strategy_type for row in broad_ranked}):
        discovery_rows = [row for row in broad_ranked if row.strategy_type == strategy_type]
        discovery_rows.sort(key=_discovery_selection_key, reverse=True)
        local_settings = generate_local_optimizer_settings(
            [row.settings for row in discovery_rows[:2]],
            strategy_type,
            max_candidates=max_local_candidates_per_strategy,
            excluded_identities=tested_settings,
        )
        strategy_label = next(
            label for label, value in OPTIMIZER_STRATEGY_TYPES.items() if value == strategy_type
        )
        for settings in local_settings:
            try:
                candidate = _evaluate_optimizer_candidate(
                    strategy_label=strategy_label,
                    strategy_type=strategy_type,
                    settings=settings,
                    market_data=data,
                    account_equity=account_equity,
                    risk_limits=risk_limits,
                    train_fraction=train_fraction,
                    locked_fraction=locked_fraction,
                    min_test_trades=min_test_trades,
                    target_max_drawdown_percent=target_max_drawdown_percent,
                )
            except ValueError:
                rejected += 1
                continue
            ranked.append(candidate)
            tested_settings.add(_optimizer_settings_identity(strategy_type, settings))

    ranked = _attach_plateau_scores(ranked)
    finalists: list[OptimizerCandidate] = []
    remainder: list[OptimizerCandidate] = []
    for strategy_type in sorted({row.strategy_type for row in ranked}):
        strategy_rows = [row for row in ranked if row.strategy_type == strategy_type]
        strategy_rows.sort(key=_discovery_selection_key, reverse=True)
        shortlist_size = min(10, len(strategy_rows))
        for candidate in strategy_rows[:shortlist_size]:
            rolling = _rolling_evidence(
                candidate.strategy_type,
                candidate.settings,
                data,
                account_equity,
                risk_limits,
                locked_fraction=locked_fraction,
                windows=rolling_windows,
            )
            finalists.append(replace(
                candidate,
                rolling_profitable_windows=rolling["profitable_windows"],
                rolling_windows=rolling["windows"],
                rolling_median_return_percent=rolling["median_return_percent"],
                rolling_worst_drawdown_percent=rolling["worst_drawdown_percent"],
            ))
        remainder.extend(strategy_rows[shortlist_size:])
    finalists.sort(key=_validation_selection_key, reverse=True)
    remainder.sort(key=_discovery_selection_key, reverse=True)
    ranked = finalists + remainder
    best = ranked[0] if ranked else None
    robustness = None
    if best is not None:
        locked_test, locked_trades = _locked_test(
            best.strategy_type,
            best.settings,
            data,
            account_equity,
            risk_limits,
            locked_fraction=locked_fraction,
            min_trades=min_test_trades,
        )
        full_result = _run_one(best.strategy_type, data, best.settings, account_equity, risk_limits)
        full_trades = list(full_result["trade_log"])
        stress_rows = _execution_stress_rows(full_trades, account_equity, risk_limits)
        regime_rows = _regime_rows(data, full_trades, account_equity, best.settings, risk_limits)
        bootstrap = _bootstrap_trades(
            full_trades,
            ticker_allocated_capital(account_equity, risk_limits),
            samples=bootstrap_samples,
        )
        trial_adjustment = _trial_adjustment(best, ranked)
        parameter_range = _parameter_plateau_range(best, ranked)
        diagnostics = _candidate_diagnostics(
            best,
            full_trades,
            stress_rows,
            regime_rows,
            account_equity,
        )
        final_confidence, final_stability = _final_confidence(
            best,
            locked_test,
            stress_rows,
            trial_adjustment,
        )
        if not parameter_range:
            final_confidence = "Low"
            final_stability = (
                "No stable nearby numeric range was found. Treat this as an isolated historical result, not a durable setting recommendation."
            )
        updated_best = replace(best, confidence=final_confidence, stability=final_stability)
        ranked = [updated_best if row is best else row for row in ranked]
        best = updated_best
        robustness = RobustnessEvidence(
            parameter_range=parameter_range,
            locked_test=locked_test,
            stress_rows=stress_rows,
            regime_rows=regime_rows,
            bootstrap=bootstrap,
            trial_adjustment=trial_adjustment,
            diagnostics=diagnostics,
        )
    summary = optimizer_summary(best)
    return StrategyInputRecommendation(
        best=best,
        candidates=ranked,
        summary=summary,
        tested_candidates=len(ranked),
        rejected_candidates=rejected,
        train_fraction=train_fraction,
        locked_fraction=locked_fraction,
        robustness=robustness,
    )


def buy_and_hold_benchmark(
    market_data: pd.DataFrame,
    account_equity: float,
    *,
    start: int = 0,
    end: int | None = None,
    allocated_capital: float | None = None,
) -> BuyAndHoldBenchmark:
    """Return an adjusted-price buy-and-hold baseline for the requested bar range."""
    close = market_data["Close"].astype(float).iloc[start:end].dropna()
    allocation = float(account_equity) if allocated_capital is None else max(0.0, float(allocated_capital))
    years = elapsed_years(market_data.index, start=start, end=end)
    if len(close) < 2 or account_equity <= 0 or allocation <= 0:
        return BuyAndHoldBenchmark(
            0.0, 0.0, float(account_equity), 0.0, 0.0, len(close),
            allocated_capital=allocation, period_years=years,
        )
    start_price = float(close.iloc[0])
    end_price = float(close.iloc[-1])
    if start_price <= 0:
        return BuyAndHoldBenchmark(
            0.0, 0.0, float(account_equity), start_price, end_price, len(close),
            allocated_capital=allocation, period_years=years,
        )
    quantity = allocation / start_price
    buy_fees, sell_fees = estimate_alpaca_round_trip_fees(
        asset_class=normalize_asset_class(market_data.attrs.get("asset_class"), market_data.attrs.get("symbol", "")),
        quantity=quantity,
        entry_price=start_price,
        exit_price=end_price,
    )
    estimated_fees = buy_fees.total + sell_fees.total
    sleeve_curve = close.to_numpy(dtype=float) / start_price * allocation - buy_fees.total
    sleeve_curve[-1] -= sell_fees.total
    equity_curve = float(account_equity) - allocation + sleeve_curve
    sleeve_peaks = np.maximum.accumulate(sleeve_curve)
    max_drawdown_dollars = float(np.max(sleeve_peaks - sleeve_curve))
    raw_return = ((end_price - start_price) * quantity - estimated_fees) / allocation * 100
    account_return = (equity_curve[-1] / float(account_equity) - 1) * 100
    return BuyAndHoldBenchmark(
        return_percent=round(raw_return, 2),
        max_drawdown_percent=round(max_drawdown_dollars / allocation * 100, 2),
        final_equity=round(float(equity_curve[-1]), 2),
        start_price=round(start_price, 4),
        end_price=round(end_price, 4),
        bars=len(close),
        account_return_percent=round(account_return, 2),
        annualized_return_percent=annualized_return_percent(raw_return, years),
        allocated_capital=round(allocation, 2),
        period_years=round(years, 4) if years is not None else None,
        estimated_alpaca_fees=round(estimated_fees, 2),
    )


def recommendation_evidence_status(result: StrategyInputRecommendation) -> str:
    candidate = result.best
    evidence = result.robustness
    locked = evidence.locked_test if evidence else None
    if candidate is None or locked is None:
        return "Research only"
    rolling_ratio = candidate.rolling_profitable_windows / max(1, candidate.rolling_windows)
    stress_10 = next(
        (row for row in evidence.stress_rows if row["Round-trip slippage"] == "10 bps per side"),
        {},
    )
    credible = bool(
        candidate.confidence in {"Medium", "High"}
        and candidate.test_trades >= 2
        and candidate.excess_return_percent > 0
        and locked.passed
        and locked.excess_return_percent > 0
        and rolling_ratio >= 0.5
        and stress_10.get("Passed")
    )
    return "Ready for paper test" if credible else "Research only"


def candidate_verdict(
    result: StrategyInputRecommendation,
    interval: str = "1d",
) -> CandidateVerdict:
    """Classify a recommendation from unseen, cost, stability, and risk evidence."""
    candidate = result.best
    evidence = result.robustness
    if candidate is None or evidence is None or evidence.locked_test is None:
        return CandidateVerdict(
            tier="Research Only",
            summary="No complete candidate is available yet.",
            uncertain=("Run the strategy input search with enough price history.",),
        )

    locked = evidence.locked_test
    diagnostics = evidence.diagnostics
    bootstrap = evidence.bootstrap
    strong_trades, promising_trades = _trade_evidence_thresholds(interval)
    checks: list[tuple[str, str, str]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append((name, status, detail))

    validation_excess = candidate.excess_return_percent
    locked_excess = locked.excess_return_percent
    if validation_excess > 0 and locked_excess > 0:
        add("Unseen buy-and-hold comparison", "strong", f"Validation {validation_excess:+.2f}%; final {locked_excess:+.2f}%.")
    elif max(validation_excess, locked_excess) > 0 and min(validation_excess, locked_excess) >= -5:
        add("Unseen buy-and-hold comparison", "promising", f"One unseen period led; the other was no worse than -5% ({validation_excess:+.2f}%, {locked_excess:+.2f}%).")
    elif validation_excess <= -5 and locked_excess <= -5:
        add("Unseen buy-and-hold comparison", "failed", f"Both unseen periods materially trailed buy-and-hold ({validation_excess:+.2f}%, {locked_excess:+.2f}%).")
    else:
        add("Unseen buy-and-hold comparison", "uncertain", f"Results disagreed across unseen periods ({validation_excess:+.2f}%, {locked_excess:+.2f}%).")

    positive_probability = None if bootstrap is None else 100.0 - bootstrap.loss_probability_percent
    expectancy = None if diagnostics is None else diagnostics.after_cost_expectancy_percent
    if expectancy is None or positive_probability is None:
        add("After-cost expectancy", "uncertain", "Not enough completed trades for a reliable after-cost estimate.")
    elif expectancy <= 0 or positive_probability < 50:
        add("After-cost expectancy", "failed", f"Average trade after 10 bps per side was {expectancy:+.3f}%; resampled profit probability {positive_probability:.1f}%.")
    elif positive_probability >= 75:
        add("After-cost expectancy", "strong", f"Average trade after 10 bps per side was {expectancy:+.3f}%; resampled profit probability {positive_probability:.1f}%.")
    elif positive_probability >= 60:
        add("After-cost expectancy", "promising", f"Average trade after 10 bps per side was {expectancy:+.3f}%; resampled profit probability {positive_probability:.1f}%.")
    else:
        add("After-cost expectancy", "uncertain", f"Positive after costs, but resampled profit probability was only {positive_probability:.1f}%.")

    nearby = candidate.plateau_profitable_percent
    if nearby >= 70:
        add("Nearby settings", "strong", f"{nearby:.0f}% of nearby settings were profitable.")
    elif nearby >= 50:
        add("Nearby settings", "promising", f"{nearby:.0f}% of nearby settings were profitable.")
    elif nearby < 25 and candidate.plateau_neighbors >= 3:
        add("Nearby settings", "failed", f"Only {nearby:.0f}% of nearby settings were profitable.")
    else:
        add("Nearby settings", "uncertain", f"{nearby:.0f}% of nearby settings were profitable.")

    slippage_bps = 0 if diagnostics is None else diagnostics.slippage_survival_bps
    if slippage_bps >= 20:
        add("Trading-cost test", "strong", "Stayed profitable with 20 bps of slippage per side.")
    elif slippage_bps >= 10:
        add("Trading-cost test", "promising", "Stayed profitable with 10 bps of slippage per side.")
    elif slippage_bps >= 5:
        add("Trading-cost test", "uncertain", "Stayed profitable with only 5 bps of slippage per side.")
    else:
        add("Trading-cost test", "failed", "Did not remain profitable under the 5 bps slippage test.")

    if diagnostics is None:
        add("Best-trade dependence", "uncertain", "Not enough trade detail to remove the best trade.")
    elif diagnostics.best_trade_removed_return_percent > 0 and diagnostics.best_trade_share_percent < 50:
        add("Best-trade dependence", "strong", f"Return without the best trade was {diagnostics.best_trade_removed_return_percent:+.2f}%; best trade supplied {diagnostics.best_trade_share_percent:.0f}% of profit.")
    elif diagnostics.best_trade_removed_return_percent >= -1 and diagnostics.best_trade_share_percent < 70:
        add("Best-trade dependence", "promising", f"Return without the best trade was {diagnostics.best_trade_removed_return_percent:+.2f}%; best trade supplied {diagnostics.best_trade_share_percent:.0f}% of profit.")
    elif diagnostics.best_trade_removed_return_percent < 0 and diagnostics.best_trade_share_percent >= 70:
        add("Best-trade dependence", "failed", f"The best trade supplied {diagnostics.best_trade_share_percent:.0f}% of profit and removing it produced {diagnostics.best_trade_removed_return_percent:+.2f}%.")
    else:
        add("Best-trade dependence", "uncertain", f"Return without the best trade was {diagnostics.best_trade_removed_return_percent:+.2f}%.")

    account_drawdown = None if diagnostics is None else diagnostics.account_drawdown_percent
    if account_drawdown is None:
        add("Account drawdown", "uncertain", "Account-level drawdown could not be calculated.")
    elif account_drawdown <= 1.0:
        add("Account drawdown", "strong", f"Worst validation decline was {account_drawdown:.2f}% of the account.")
    elif account_drawdown <= 1.5:
        add("Account drawdown", "promising", f"Worst validation decline was {account_drawdown:.2f}% of the account.")
    elif account_drawdown > 2.5:
        add("Account drawdown", "failed", f"Worst validation decline was {account_drawdown:.2f}% of the account.")
    else:
        add("Account drawdown", "uncertain", f"Worst validation decline was {account_drawdown:.2f}% of the account.")

    return_to_drawdown = None if diagnostics is None else diagnostics.return_to_drawdown
    if return_to_drawdown is None:
        add("Return per unit of drawdown", "uncertain", "Return-to-drawdown could not be calculated.")
    elif return_to_drawdown >= 1.5:
        add("Return per unit of drawdown", "strong", f"Validation return was {return_to_drawdown:.2f}x its worst allocated decline.")
    elif return_to_drawdown >= 0.75:
        add("Return per unit of drawdown", "promising", f"Validation return was {return_to_drawdown:.2f}x its worst allocated decline.")
    elif return_to_drawdown <= 0:
        add("Return per unit of drawdown", "failed", f"Validation return-to-drawdown was {return_to_drawdown:.2f}x.")
    else:
        add("Return per unit of drawdown", "uncertain", f"Validation return was only {return_to_drawdown:.2f}x its worst allocated decline.")

    profitable_regimes = None if diagnostics is None else diagnostics.profitable_regime_percent
    if profitable_regimes is None:
        add("Market conditions", "uncertain", "No populated market-condition groups were available.")
    elif profitable_regimes >= 60:
        add("Market conditions", "strong", f"Profitable in {profitable_regimes:.0f}% of populated market-condition groups.")
    elif profitable_regimes >= 40:
        add("Market conditions", "promising", f"Profitable in {profitable_regimes:.0f}% of populated market-condition groups.")
    else:
        add("Market conditions", "uncertain", f"Profitable in only {profitable_regimes:.0f}% of populated market-condition groups.")

    trades = candidate.test_trades + locked.trades
    if trades >= strong_trades:
        add("Completed trades", "strong", f"{trades} unseen trades; strong threshold for {interval} is {strong_trades}.")
    elif trades >= promising_trades:
        add("Completed trades", "promising", f"{trades} unseen trades; promising threshold for {interval} is {promising_trades}.")
    else:
        add("Completed trades", "uncertain", f"Only {trades} unseen trades; {promising_trades} are preferred for a Promising Candidate on {interval}.")

    rolling_ratio = candidate.rolling_profitable_windows / max(1, candidate.rolling_windows)
    if rolling_ratio >= 0.70:
        add("Separate time periods", "strong", f"Profitable in {candidate.rolling_profitable_windows}/{candidate.rolling_windows} rolling periods.")
    elif rolling_ratio >= 0.50:
        add("Separate time periods", "promising", f"Profitable in {candidate.rolling_profitable_windows}/{candidate.rolling_windows} rolling periods.")
    elif candidate.rolling_windows and rolling_ratio < 0.25:
        add("Separate time periods", "failed", f"Profitable in only {candidate.rolling_profitable_windows}/{candidate.rolling_windows} rolling periods.")
    else:
        add("Separate time periods", "uncertain", f"Profitable in {candidate.rolling_profitable_windows}/{candidate.rolling_windows} rolling periods.")

    passed = tuple(f"{name}: {detail}" for name, status, detail in checks if status in {"strong", "promising"})
    uncertain = tuple(f"{name}: {detail}" for name, status, detail in checks if status == "uncertain")
    failed = tuple(f"{name}: {detail}" for name, status, detail in checks if status == "failed")
    status_by_name = {name: status for name, status, _ in checks}
    critical_failures = {
        "Unseen buy-and-hold comparison",
        "After-cost expectancy",
        "Best-trade dependence",
        "Nearby settings",
    }
    if any(status_by_name.get(name) == "failed" for name in critical_failures):
        tier = "Reject"
    else:
        strong_count = sum(status == "strong" for _, status, _ in checks)
        supportive_count = sum(status in {"strong", "promising"} for _, status, _ in checks)
        core_strong = all(status_by_name.get(name) == "strong" for name in (
            "Unseen buy-and-hold comparison", "After-cost expectancy", "Completed trades"
        ))
        if not failed and core_strong and strong_count >= 6:
            tier = "Strong Candidate"
        elif not failed and trades >= promising_trades and supportive_count >= 6:
            tier = "Promising Candidate"
        else:
            tier = "Research Only"

    next_step = {
        "Strong Candidate": "A high-priority paper-test candidate; still review the setup before buying.",
        "Promising Candidate": "Worth paper testing, with the uncertain evidence kept visible.",
        "Research Only": "Keep researching; the evidence is incomplete or inconsistent.",
        "Reject": "Do not use these settings without a materially better result.",
    }[tier]
    return CandidateVerdict(tier=tier, summary=next_step, passed=passed, uncertain=uncertain, failed=failed)


def candidate_verdict_records(verdict: CandidateVerdict) -> list[dict[str, str]]:
    return [
        {"Evidence": item.split(":", 1)[0], "Status": status, "Plain English": item.split(":", 1)[1].strip()}
        for status, items in (
            ("Supports candidate", verdict.passed),
            ("Needs more evidence", verdict.uncertain),
            ("Contradicts candidate", verdict.failed),
        )
        for item in items
    ]


def _trade_evidence_thresholds(interval: str) -> tuple[int, int]:
    normalized = str(interval).strip().lower()
    if normalized == "1h":
        return 40, 20
    if normalized == "4h":
        return 25, 12
    return 15, 8


def optimize_strategy_intervals(
    market_data_by_interval: dict[str, tuple[str, pd.DataFrame]],
    current_settings: dict[str, Any],
    account_equity: float,
    risk_limits: RiskLimits | None = None,
    *,
    train_fraction: float = 0.65,
    max_candidates_per_strategy: int = 12,
    max_local_candidates_per_strategy: int = OPTIMIZER_LOCAL_SETTINGS_PER_STRATEGY,
    bootstrap_samples: int = 1000,
    comparison_years: int = 2,
) -> MultiIntervalRecommendation:
    """Rank intervals on one calendar window, then check fixed settings on longer data."""
    interval_rows: list[IntervalOptimizationResult] = []
    shared_start, shared_end, shared_label = _shared_calendar_window(
        [market_data for _, market_data in market_data_by_interval.values()],
        comparison_years,
    )
    for interval, (history, market_data) in market_data_by_interval.items():
        settings = dict(current_settings)
        settings["interval"] = interval
        comparison_data = _calendar_slice(market_data, shared_start, shared_end)
        result = optimize_strategy_inputs(
            market_data=comparison_data,
            current_settings=settings,
            account_equity=account_equity,
            risk_limits=risk_limits,
            train_fraction=train_fraction,
            max_candidates_per_strategy=max_candidates_per_strategy,
            max_local_candidates_per_strategy=max_local_candidates_per_strategy,
            bootstrap_samples=bootstrap_samples,
        )
        candidate = result.best
        locked = result.robustness.locked_test if result.robustness else None
        confidence_bonus = {"High": 4.0, "Medium": 2.0, "Low": -2.0}.get(
            candidate.confidence if candidate else "Low", -2.0
        )
        selection_score = -9999.0 if candidate is None else (
            candidate.score
            + confidence_bonus
            + (3.0 if locked and locked.passed else -3.0)
            + max(-5.0, min(5.0, candidate.excess_return_percent * 0.25))
        )
        durability_return = 0.0
        durability_benchmark = 0.0
        durability_drawdown = 0.0
        durability_trades = 0
        comparison_return = 0.0
        comparison_benchmark = 0.0
        comparison_drawdown = 0.0
        comparison_trades = 0
        if candidate is not None:
            comparison_result = _run_one(
                candidate.strategy_type,
                comparison_data,
                candidate.settings,
                account_equity,
                risk_limits,
            )
            comparison_stats = _closed_trade_stats(
                account_equity,
                list(comparison_result["trade_log"]),
                len(comparison_data),
                risk_limits,
                elapsed_years(comparison_data.index),
            )
            comparison_hold = buy_and_hold_benchmark(
                comparison_data,
                account_equity,
                allocated_capital=ticker_allocated_capital(account_equity, risk_limits),
            )
            comparison_return = float(comparison_stats["allocated_return_pct"])
            comparison_benchmark = comparison_hold.return_percent
            comparison_drawdown = float(comparison_stats["allocated_max_drawdown_pct"])
            comparison_trades = int(comparison_stats["total_trades"])
            durability_result = _run_one(
                candidate.strategy_type,
                market_data,
                candidate.settings,
                account_equity,
                risk_limits,
            )
            durability_stats = _closed_trade_stats(
                account_equity,
                list(durability_result["trade_log"]),
                len(market_data),
                risk_limits,
                elapsed_years(market_data.index),
            )
            durability_hold = buy_and_hold_benchmark(
                market_data,
                account_equity,
                allocated_capital=ticker_allocated_capital(account_equity, risk_limits),
            )
            durability_return = float(durability_stats["allocated_return_pct"])
            durability_benchmark = durability_hold.return_percent
            durability_drawdown = float(durability_stats["allocated_max_drawdown_pct"])
            durability_trades = int(durability_stats["total_trades"])
        interval_rows.append(IntervalOptimizationResult(
            interval=interval,
            history=history,
            recommendation=result,
            selection_score=round(selection_score, 2),
            evidence_status=recommendation_evidence_status(result),
            comparison_history=shared_label,
            comparison_return_percent=round(comparison_return, 2),
            comparison_benchmark_return_percent=round(comparison_benchmark, 2),
            comparison_excess_return_percent=round(comparison_return - comparison_benchmark, 2),
            comparison_max_drawdown_percent=round(comparison_drawdown, 2),
            comparison_trades=comparison_trades,
            durability_return_percent=round(durability_return, 2),
            durability_benchmark_return_percent=round(durability_benchmark, 2),
            durability_excess_return_percent=round(durability_return - durability_benchmark, 2),
            durability_max_drawdown_percent=round(durability_drawdown, 2),
            durability_trades=durability_trades,
        ))
    if not interval_rows:
        raise ValueError("No interval data was available for the strategy input search.")
    interval_rows.sort(
        key=lambda row: (row.evidence_status == "Ready for paper test", row.selection_score),
        reverse=True,
    )
    best_row = interval_rows[0]
    best = best_row.recommendation.best
    locked = best_row.recommendation.robustness.locked_test if best_row.recommendation.robustness else None
    qualification_text = (
        "Qualified paper-test candidate."
        if best_row.evidence_status == "Ready for paper test"
        else "No interval qualified for paper testing; this is only the strongest research candidate."
    )
    summary = (
        f"{qualification_text} Candidate: {best_row.interval} / {best.strategy_label}. "
        f"Across the complete {shared_label.lower()} comparison, its unchanged settings returned "
        f"{best_row.comparison_return_percent:.2f}% versus {best_row.comparison_benchmark_return_percent:.2f}% "
        f"for buy-and-hold ({best_row.comparison_excess_return_percent:+.2f}% excess). "
        f"The middle validation period was {best.excess_return_percent:+.2f}% versus buy-and-hold; "
        f"the untouched final period was {locked.excess_return_percent:+.2f}% versus buy-and-hold. "
        f"Status: {best_row.evidence_status}."
        if best is not None
        else "No interval produced enough evidence to recommend settings."
    )
    return MultiIntervalRecommendation(
        best_interval=best_row.interval,
        best_history=best_row.history,
        best_result=best_row.recommendation,
        interval_results=tuple(interval_rows),
        summary=summary,
    )


def optimize_strategy_families(
    market_data_by_interval: dict[str, tuple[str, pd.DataFrame]],
    strategy_intervals: dict[str, tuple[str, ...]],
    current_settings: dict[str, Any],
    account_equity: float,
    risk_limits: RiskLimits | None = None,
    *,
    train_fraction: float = 0.55,
    max_candidates_per_strategy: int = 32,
    max_local_candidates_per_strategy: int = OPTIMIZER_LOCAL_SETTINGS_PER_STRATEGY,
    bootstrap_samples: int = 250,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> MultiStrategySearchResult:
    """Return one independently selected result for every strategy family."""
    tasks = [
        (strategy_label, strategy_type, interval)
        for strategy_label, strategy_type in OPTIMIZER_STRATEGY_TYPES.items()
        for interval in strategy_intervals.get(strategy_type, ())
        if interval in market_data_by_interval
    ]
    completed = 0
    strategy_results: list[StrategySearchResult] = []
    for strategy_label, strategy_type in OPTIMIZER_STRATEGY_TYPES.items():
        interval_rows: list[IntervalOptimizationResult] = []
        for interval in strategy_intervals.get(strategy_type, ()):
            payload = market_data_by_interval.get(interval)
            if payload is None:
                continue
            history, market_data = payload
            if progress_callback:
                progress_callback(completed, len(tasks), f"Testing {strategy_label} with {interval} prices")
            settings = dict(current_settings)
            settings.update({"strategy_label": strategy_label, "strategy_type": strategy_type, "interval": interval})
            recommendation = optimize_strategy_inputs(
                market_data=market_data,
                current_settings=settings,
                account_equity=account_equity,
                risk_limits=risk_limits,
                train_fraction=train_fraction,
                max_candidates_per_strategy=max_candidates_per_strategy,
                max_local_candidates_per_strategy=max_local_candidates_per_strategy,
                bootstrap_samples=bootstrap_samples,
                strategy_types={strategy_type},
            )
            completed += 1
            if progress_callback:
                progress_callback(completed, len(tasks), f"Finished {strategy_label} with {interval} prices")
            candidate = recommendation.best
            if candidate is None:
                continue
            full_result = _run_one(strategy_type, market_data, candidate.settings, account_equity, risk_limits)
            full_stats = _closed_trade_stats(
                account_equity,
                list(full_result["trade_log"]),
                len(market_data),
                risk_limits,
                elapsed_years(market_data.index),
            )
            full_benchmark = buy_and_hold_benchmark(
                market_data,
                account_equity,
                allocated_capital=ticker_allocated_capital(account_equity, risk_limits),
            )
            annualized_strategy = full_stats["annualized_allocated_return_pct"]
            annualized_benchmark = full_benchmark.annualized_return_percent
            annualized_excess = (
                None
                if annualized_strategy is None or annualized_benchmark is None
                else round(float(annualized_strategy) - float(annualized_benchmark), 2)
            )
            older_dates, newer_dates, latest_dates = _optimizer_date_ranges(
                market_data,
                train_fraction=train_fraction,
                locked_fraction=recommendation.locked_fraction,
            )
            interval_rows.append(IntervalOptimizationResult(
                interval=interval,
                history=history,
                recommendation=recommendation,
                selection_score=candidate.excess_return_percent,
                evidence_status=historical_trade_evidence(candidate.train_trades),
                comparison_history=history,
                durability_return_percent=float(full_stats["allocated_return_pct"]),
                durability_benchmark_return_percent=full_benchmark.return_percent,
                durability_excess_return_percent=round(
                    float(full_stats["allocated_return_pct"]) - full_benchmark.return_percent, 2
                ),
                durability_max_drawdown_percent=float(full_stats["allocated_max_drawdown_pct"]),
                durability_trades=int(full_stats["total_trades"]),
                durability_annualized_return_percent=annualized_strategy,
                durability_benchmark_annualized_return_percent=annualized_benchmark,
                durability_annualized_excess_percent=annualized_excess,
                older_dates=older_dates,
                newer_dates=newer_dates,
                latest_dates=latest_dates,
            ))
        if not interval_rows:
            continue
        interval_rows.sort(key=_strategy_interval_selection_key, reverse=True)
        best_row = interval_rows[0]
        strategy_results.append(StrategySearchResult(
            strategy_label=strategy_label,
            strategy_type=strategy_type,
            best_interval=best_row.interval,
            best_history=best_row.history,
            best_result=best_row.recommendation,
            interval_results=tuple(interval_rows),
        ))
    strategy_results.sort(key=_strategy_family_selection_key, reverse=True)
    if not strategy_results:
        raise ValueError("No strategy produced a historical result.")
    best = strategy_results[0]
    candidate = best.best_result.best
    older_percent, newer_percent, _ = _optimizer_split_percentages(best.best_result)
    summary = (
        f"Highest-ranked historical result: {best.strategy_label} using {best.best_interval} prices. "
        f"On the newer {newer_percent}% of prices it returned {candidate.excess_return_percent:+.2f}% versus buying and holding. "
        f"The older {older_percent}% produced {candidate.train_trades} completed trades "
        f"({historical_trade_evidence(candidate.train_trades).lower()})."
    )
    return MultiStrategySearchResult(tuple(strategy_results), summary)


def _strategy_interval_selection_key(row: IntervalOptimizationResult) -> tuple[float, ...]:
    candidate = row.recommendation.best
    if candidate is None:
        return (-1.0, -9999.0, -9999.0)
    return _validation_selection_key(candidate)


def _strategy_family_selection_key(row: StrategySearchResult) -> tuple[float, ...]:
    best_interval = row.interval_results[0]
    candidate = best_interval.recommendation.best
    if candidate is None:
        return (-1.0, -9999.0, -9999.0)
    return _validation_selection_key(candidate)


def _optimizer_date_ranges(
    market_data: pd.DataFrame,
    *,
    train_fraction: float,
    locked_fraction: float,
) -> tuple[str, str, str]:
    if market_data.empty or not isinstance(market_data.index, pd.DatetimeIndex):
        newer_fraction = max(0.0, 1.0 - train_fraction - locked_fraction)
        return (
            f"Older {train_fraction * 100:.0f}% of prices",
            f"Newer {newer_fraction * 100:.0f}% of prices",
            f"Latest {locked_fraction * 100:.0f}% of prices",
        )
    total = len(market_data)
    latest_start = int(total * (1.0 - locked_fraction))
    newer_start = min(int(total * train_fraction), latest_start)

    def label(start: int, end: int) -> str:
        if end <= start:
            return "No dates available"
        left = pd.Timestamp(market_data.index[start]).strftime("%b %Y")
        right = pd.Timestamp(market_data.index[end - 1]).strftime("%b %Y")
        return left if left == right else f"{left} to {right}"

    return label(0, newer_start), label(newer_start, latest_start), label(latest_start, total)


def _optimizer_split_percentages(
    recommendation: StrategyInputRecommendation,
) -> tuple[int, int, int]:
    older = int(round(recommendation.train_fraction * 100))
    latest = int(round(recommendation.locked_fraction * 100))
    newer = max(0, 100 - older - latest)
    return older, newer, latest


def _shared_calendar_window(
    frames: list[pd.DataFrame],
    years: int,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str]:
    """Return a common recent calendar range for every dated interval."""
    dated = [frame for frame in frames if not frame.empty and isinstance(frame.index, pd.DatetimeIndex)]
    if len(dated) != len(frames) or not dated:
        return None, None, "Available data"
    common_end = min(frame.index.max() for frame in dated)
    available_start = max(frame.index.min() for frame in dated)
    requested_start = common_end - pd.DateOffset(years=max(1, int(years)))
    common_start = max(available_start, requested_start)
    actual_years = max(0.0, (common_end - common_start).total_seconds() / (365.2425 * 24 * 3600))
    label = f"Latest {actual_years:.1f} years" if actual_years < years - 0.05 else f"Latest {years} years"
    return common_start, common_end, label


def _calendar_slice(
    market_data: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    """Apply a shared date range while preserving non-date test fixtures."""
    data = market_data.copy()
    if start is None or end is None or not isinstance(data.index, pd.DatetimeIndex):
        return data
    subset = data.loc[(data.index > start) & (data.index <= end)].copy()
    subset.attrs.update(data.attrs)
    return subset


def generate_optimizer_settings(
    current_settings: dict[str, Any],
    *,
    max_candidates_per_strategy: int = 18,
    strategy_types: set[str] | None = None,
) -> list[tuple[str, str, dict[str, Any]]]:
    base = _normal_settings(current_settings)
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for strategy_label, strategy_type in OPTIMIZER_STRATEGY_TYPES.items():
        if strategy_types is not None and strategy_type not in strategy_types:
            continue
        strategy_rows: list[dict[str, Any]] = []
        entry_options = list(OPTIMIZER_ENTRY_WINDOWS)
        exit_options = list(OPTIMIZER_EXIT_WINDOWS)
        atr_options = list(OPTIMIZER_ATR_MULTIPLIERS)
        trend_options = list(OPTIMIZER_TREND_WINDOWS)
        pullback_options = list(OPTIMIZER_PULLBACK_WINDOWS)
        momentum_options = list(OPTIMIZER_MOMENTUM_WINDOWS)

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

        unique_rows: list[dict[str, Any]] = []
        seen_rows: set[tuple[tuple[str, str], ...]] = set()
        for row in strategy_rows:
            identity = tuple(sorted((key, str(value)) for key, value in row.items()))
            if identity not in seen_rows:
                seen_rows.add(identity)
                unique_rows.append(row)
        strategy_rows = _broad_optimizer_sample(
            unique_rows,
            base,
            max_candidates=max_candidates_per_strategy,
        )
        for row in strategy_rows:
            rows.append((
                strategy_label,
                strategy_type,
                row | {"rsi_entry_filter_enabled": False},
            ))
    return rows


def _optimizer_numeric_keys(strategy_type: str) -> tuple[str, ...]:
    if strategy_type == "pullback":
        return (
            "exit_window", "atr_stop_multiplier", "moving_average_window",
            "pullback_average_length", "momentum_turn_length",
        )
    if strategy_type == "trendline_retest":
        return (
            "entry_window", "exit_window", "atr_stop_multiplier",
            "moving_average_window", "momentum_turn_length",
        )
    return (
        "entry_window", "exit_window", "atr_stop_multiplier", "moving_average_window",
    )


def _optimizer_options(key: str) -> tuple[float | int, ...]:
    return {
        "entry_window": OPTIMIZER_ENTRY_WINDOWS,
        "exit_window": OPTIMIZER_EXIT_WINDOWS,
        "atr_stop_multiplier": OPTIMIZER_ATR_MULTIPLIERS,
        "moving_average_window": OPTIMIZER_TREND_WINDOWS,
        "pullback_average_length": OPTIMIZER_PULLBACK_WINDOWS,
        "momentum_turn_length": OPTIMIZER_MOMENTUM_WINDOWS,
    }[key]


def _optimizer_settings_identity(
    strategy_type: str,
    settings: dict[str, Any],
) -> tuple[str, tuple[tuple[str, float | int], ...]]:
    return (
        strategy_type,
        tuple((key, settings[key]) for key in _optimizer_numeric_keys(strategy_type)),
    )


def _adjacent_optimizer_values(key: str, value: float | int) -> tuple[float | int, ...]:
    allowed = list(_optimizer_options(key))
    index = min(range(len(allowed)), key=lambda item: abs(float(allowed[item]) - float(value)))
    left = max(0, index - 1)
    right = min(len(allowed), index + 2)
    return tuple(allowed[left:right])


def generate_local_optimizer_settings(
    seed_settings: list[dict[str, Any]],
    strategy_type: str,
    *,
    max_candidates: int = OPTIMIZER_LOCAL_SETTINGS_PER_STRATEGY,
    excluded_identities: set[tuple[str, tuple[tuple[str, float | int], ...]]] | None = None,
) -> list[dict[str, Any]]:
    """Test deliberate numeric neighbors around the two strongest broad-search regions."""
    if strategy_type not in OPTIMIZER_STRATEGY_TYPES.values() or max_candidates <= 0:
        return []
    excluded = set(excluded_identities or set())
    keys = _optimizer_numeric_keys(strategy_type)
    per_seed: list[list[dict[str, Any]]] = []
    for seed in seed_settings[:2]:
        rows: list[dict[str, Any]] = []
        for values in product(*(_adjacent_optimizer_values(key, seed[key]) for key in keys)):
            row = dict(seed)
            row.update(dict(zip(keys, values)))
            row["rsi_entry_filter_enabled"] = False
            identity = _optimizer_settings_identity(strategy_type, row)
            if identity in excluded or identity == _optimizer_settings_identity(strategy_type, seed):
                continue
            rows.append(row)
        rows.sort(
            key=lambda row: (
                sum(row[key] != seed[key] for key in keys),
                _settings_distance(seed, row),
                sha256(repr(_optimizer_settings_identity(strategy_type, row)).encode("utf-8")).hexdigest(),
            )
        )
        per_seed.append(rows)

    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < max_candidates and any(per_seed):
        bucket_index = cursor % len(per_seed)
        cursor += 1
        bucket = per_seed[bucket_index]
        while bucket:
            row = bucket.pop(0)
            identity = _optimizer_settings_identity(strategy_type, row)
            if identity in excluded:
                continue
            selected.append(row)
            excluded.add(identity)
            break
        if all(not bucket for bucket in per_seed):
            break
    return selected


def _broad_optimizer_sample(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Return one current-like setup plus deterministic coverage of the full range."""
    if not rows or max_candidates <= 0:
        return []
    nearest = min(rows, key=lambda row: _settings_distance(base, row))
    if max_candidates == 1:
        return [nearest]
    remaining = [row for row in rows if row is not nearest]
    varying_keys = {
        key
        for key in nearest
        if len({str(row.get(key)) for row in rows}) > 1
    }
    covered = {(key, str(nearest.get(key))) for key in varying_keys}
    selected = [nearest]
    while remaining and len(selected) < max_candidates:
        best = max(
            remaining,
            key=lambda row: (
                sum((key, str(row.get(key))) not in covered for key in varying_keys),
                _settings_distance(base, row),
                sha256(
                    repr(tuple(sorted((key, str(value)) for key, value in row.items()))).encode("utf-8")
                ).hexdigest(),
            ),
        )
        selected.append(best)
        covered.update((key, str(best.get(key))) for key in varying_keys)
        remaining.remove(best)
    return selected


def optimizer_summary(candidate: OptimizerCandidate | None) -> str:
    if candidate is None:
        return "No strategy settings had enough validation evidence to recommend."
    return (
        f"Best current fit: {candidate.strategy_label}. "
        f"Suggested settings: {_settings_text(candidate.settings)}. "
        f"Middle validation-period allocated return was {candidate.test_return_percent:.2f}% with a "
        f"{candidate.test_max_drawdown_percent:.2f}% allocated-capital worst drop and "
        f"{candidate.excess_return_percent:+.2f}% versus buy-and-hold. Confidence: {candidate.confidence}."
    )


def optimizer_recommendation_records(
    result: StrategyInputRecommendation,
    interval: str = "1d",
) -> list[dict[str, Any]]:
    candidate = result.best
    if candidate is None:
        return [{"Item": "Recommendation", "Value": "No recommendation", "Plain English": result.summary}]
    settings = candidate.settings
    evidence = result.robustness
    locked = evidence.locked_test if evidence else None
    trial = evidence.trial_adjustment if evidence else None
    stress_10 = next(
        (row for row in evidence.stress_rows if row["Round-trip slippage"] == "10 bps per side"),
        None,
    ) if evidence else None
    verdict = candidate_verdict(result, interval)
    benchmark_read = (
        "Beat buy-and-hold"
        if candidate.excess_return_percent > 0 and locked and locked.excess_return_percent > 0
        else "Lower return, lower drawdown"
        if candidate.excess_return_percent <= 0 and candidate.drawdown_advantage_percent > 0
        else "Did not beat buy-and-hold"
    )
    best_condition = _best_market_condition(evidence.regime_rows if evidence else [])
    next_step = verdict.summary
    return [
        {"Item": "Recommendation status", "Value": verdict.tier, "Plain English": next_step},
        {"Item": "Best strategy", "Value": candidate.strategy_label, "Plain English": candidate.reason},
        {"Item": "Suggested settings", "Value": _settings_text(settings), "Plain English": "The single setting combination to paper test first."},
        {"Item": "Ticker allocation", "Value": f"${candidate.allocated_capital:,.2f}", "Plain English": "Stable capital budget set by Max symbol concentration."},
        {"Item": "Middle validation-period return", "Value": f"{candidate.test_return_percent:.2f}%", "Plain English": f"This is only the middle validation slice, not the complete backtest. Account impact was {candidate.test_account_return_percent:.2f}%."},
        {
            "Item": "Annualized allocated return",
            "Value": _annualized_text(candidate.test_annualized_return_percent),
            "Plain English": "Shown only when the middle validation period is longer than one year.",
        },
        {"Item": "Strong nearby range", "Value": _parameter_range_text(candidate.strategy_type, evidence.parameter_range if evidence else {}), "Plain English": "Nearby profitable settings. A useful result should not depend on one exact number."},
        {"Item": "Validation and final-period comparison", "Value": benchmark_read, "Plain English": f"Middle validation advantage {candidate.excess_return_percent:+.2f}%; untouched final-period advantage {locked.excess_return_percent:+.2f}%" if locked else "Untouched final-period comparison is unavailable."},
        {
            "Item": "Annualized buy-and-hold return",
            "Value": _annualized_text(candidate.benchmark_annualized_return_percent),
            "Plain English": "Equal-capital buy-and-hold over the middle validation period.",
        },
        {
            "Item": "Annualized excess return",
            "Value": (
                "Not shown (period is 1 year or less)"
                if candidate.test_annualized_return_percent is None or candidate.benchmark_annualized_return_percent is None
                else f"{candidate.test_annualized_return_percent - candidate.benchmark_annualized_return_percent:+.2f}%"
            ),
            "Plain English": "Annualized allocated return minus annualized equal-capital buy-and-hold.",
        },
        {"Item": "Best market conditions", "Value": best_condition, "Plain English": "The historical condition where these unchanged settings performed best."},
        {"Item": "Suggested strategy risk", "Value": f"{candidate.recommended_risk_per_trade_percent:.2f}%", "Plain English": "Risk size suggested by validation-period drawdown. Account risk limits still apply."},
        {"Item": "Rolling periods profitable", "Value": f"{candidate.rolling_profitable_windows}/{candidate.rolling_windows}", "Plain English": "How often the unchanged settings made money across separate chronological periods."},
        {"Item": "Locked final test", "Value": _locked_text(locked), "Plain English": "This final period was not used to choose the strategy or settings."},
        {"Item": "10 bps slippage test", "Value": _stress_text(stress_10), "Plain English": "Estimated result after adding 0.10% price friction to both entry and exit."},
        {"Item": "Trial-adjusted confidence", "Value": _trial_text(trial), "Plain English": "Reduces confidence when many setting combinations were searched."},
        {"Item": "Overall confidence", "Value": candidate.confidence, "Plain English": candidate.stability},
        {"Item": "Main concern", "Value": candidate.concern, "Plain English": "What to watch before trusting this setting."},
        {"Item": "Next step", "Value": next_step, "Plain English": "The appropriate action for the current evidence level."},
    ]


def optimizer_robustness_records(result: StrategyInputRecommendation) -> list[dict[str, Any]]:
    candidate = result.best
    evidence = result.robustness
    if candidate is None or evidence is None:
        return []
    locked = evidence.locked_test
    bootstrap = evidence.bootstrap
    trial = evidence.trial_adjustment
    return [
        {"Check": "Nearby settings", "Result": f"{candidate.plateau_profitable_percent:.0f}% profitable across {candidate.plateau_neighbors} nearby settings", "Why it matters": "Avoids choosing an isolated lucky setting."},
        {"Check": "Rolling periods", "Result": f"{candidate.rolling_profitable_windows}/{candidate.rolling_windows} profitable; median {candidate.rolling_median_return_percent:.2f}%", "Why it matters": "Checks different chronological periods."},
        {"Check": "Locked final period", "Result": _locked_text(locked), "Why it matters": locked.detail if locked else "Not available."},
        {"Check": "Resampled trade results", "Result": _bootstrap_text(bootstrap), "Why it matters": "Estimates how sensitive results are to a different ordering and mix of trades."},
        {"Check": "Many settings searched", "Result": _trial_text(trial), "Why it matters": trial.evidence if trial else "Not available."},
    ]


def optimizer_stress_records(result: StrategyInputRecommendation) -> list[dict[str, Any]]:
    return list(result.robustness.stress_rows) if result.robustness else []


def optimizer_regime_records(result: StrategyInputRecommendation) -> list[dict[str, Any]]:
    return list(result.robustness.regime_rows) if result.robustness else []


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
            "Middle Validation Return %": candidate.test_return_percent,
            "Middle Validation Account Return %": candidate.test_account_return_percent,
            "Annualized Allocated Return %": candidate.test_annualized_return_percent,
            "Equal-Capital Buy and Hold %": candidate.benchmark_return_percent,
            "Annualized Buy and Hold %": candidate.benchmark_annualized_return_percent,
            "Excess Return %": candidate.excess_return_percent,
            "Middle Validation Trades": candidate.test_trades,
            "Profit Factor": candidate.test_profit_factor,
            "Allocated Worst Drop %": candidate.test_max_drawdown_percent,
            "Profitable Periods": f"{candidate.profitable_test_periods}/{candidate.tested_periods}",
            "Nearby Settings": candidate.plateau_neighbors,
            "Nearby Profitable %": candidate.plateau_profitable_percent,
            "Rolling Periods": f"{candidate.rolling_profitable_windows}/{candidate.rolling_windows}",
            "Plain English": candidate.reason,
        }
        for index, candidate in enumerate(candidates[:limit])
    ]


def optimizer_interval_records(result: MultiIntervalRecommendation) -> list[dict[str, Any]]:
    rows = []
    for interval_result in result.interval_results:
        candidate = interval_result.recommendation.best
        locked = interval_result.recommendation.robustness.locked_test if interval_result.recommendation.robustness else None
        rows.append({
            "Interval": interval_result.interval,
            "Fair Comparison Period": interval_result.comparison_history,
            "Status": interval_result.evidence_status,
            "Strategy": candidate.strategy_label if candidate else "No candidate",
            "Comparable Newer Return %": candidate.test_return_percent if candidate else 0.0,
            "Comparable Buy and Hold %": candidate.benchmark_return_percent if candidate else 0.0,
            "Comparable Excess %": candidate.excess_return_percent if candidate else 0.0,
            "Locked Excess %": locked.excess_return_percent if locked else 0.0,
            "Comparable Worst Drop %": candidate.test_max_drawdown_percent if candidate else 0.0,
            "Confidence": candidate.confidence if candidate else "Low",
            "Complete Shared-Period Return %": interval_result.comparison_return_percent,
            "Complete Shared-Period Buy and Hold %": interval_result.comparison_benchmark_return_percent,
            "Complete Shared-Period Excess %": interval_result.comparison_excess_return_percent,
            "Long-History Check": interval_result.history,
            "Long-History Return %": interval_result.durability_return_percent,
            "Long-History Buy and Hold %": interval_result.durability_benchmark_return_percent,
            "Long-History Excess %": interval_result.durability_excess_return_percent,
            "Long-History Worst Drop %": interval_result.durability_max_drawdown_percent,
            "Long-History Trades": interval_result.durability_trades,
        })
    return rows


def strategy_search_ranking_records(result: MultiStrategySearchResult) -> list[dict[str, Any]]:
    rows = []
    for rank, strategy_result in enumerate(result.strategy_results, start=1):
        interval_result = strategy_result.interval_results[0]
        candidate = strategy_result.best_result.best
        locked = strategy_result.best_result.robustness.locked_test if strategy_result.best_result.robustness else None
        if candidate is None:
            continue
        older_percent, newer_percent, latest_percent = _optimizer_split_percentages(strategy_result.best_result)
        rows.append({
            "Rank": rank,
            "Strategy": strategy_result.strategy_label,
            "Best Interval": strategy_result.best_interval,
            f"Older {older_percent}% Trades": candidate.train_trades,
            "Historical Sample": historical_trade_evidence(candidate.train_trades),
            f"Newer {newer_percent}% vs Buy and Hold": f"{candidate.excess_return_percent:+.2f}%",
            f"Latest {latest_percent}% vs Buy and Hold": (
                f"{locked.excess_return_percent:+.2f}%" if locked is not None else "Not available"
            ),
            "Complete History Annualized vs Buy and Hold": (
                f"{interval_result.durability_annualized_excess_percent:+.2f}%"
                if interval_result.durability_annualized_excess_percent is not None
                else "Not shown"
            ),
            "Complete History Trades": interval_result.durability_trades,
            "Maximum Historical Decline": f"{interval_result.durability_max_drawdown_percent:.2f}%",
        })
    return rows


def strategy_search_detail_records(strategy_result: StrategySearchResult) -> list[dict[str, Any]]:
    interval_result = strategy_result.interval_results[0]
    recommendation = strategy_result.best_result
    candidate = recommendation.best
    if candidate is None:
        return []
    evidence = recommendation.robustness
    locked = evidence.locked_test if evidence else None
    diagnostics = evidence.diagnostics if evidence else None
    older_percent, newer_percent, latest_percent = _optimizer_split_percentages(recommendation)
    return [
        {
            "Price section": f"Older {older_percent}% ({interval_result.older_dates})",
            "Strategy return": f"{candidate.train_return_percent:.2f}%",
            "Buy and hold": f"{candidate.train_benchmark_return_percent:.2f}%",
            "Difference": f"{candidate.train_excess_return_percent:+.2f}%",
            "Completed trades": candidate.train_trades,
            "Plain English": historical_trade_evidence(candidate.train_trades),
        },
        {
            "Price section": f"Newer {newer_percent}% ({interval_result.newer_dates})",
            "Strategy return": f"{candidate.test_return_percent:.2f}%",
            "Buy and hold": f"{candidate.benchmark_return_percent:.2f}%",
            "Difference": f"{candidate.excess_return_percent:+.2f}%",
            "Completed trades": candidate.test_trades,
            "Plain English": "These unchanged settings were used to choose the result from the older-price finalists.",
        },
        {
            "Price section": f"Latest {latest_percent}% ({interval_result.latest_dates})",
            "Strategy return": f"{locked.return_percent:.2f}%" if locked else "Not available",
            "Buy and hold": f"{locked.benchmark_return_percent:.2f}%" if locked else "Not available",
            "Difference": f"{locked.excess_return_percent:+.2f}%" if locked else "Not available",
            "Completed trades": locked.trades if locked else 0,
            "Plain English": (
                "This latest section only reports what happened. It did not choose or replace the settings."
                if locked else "Latest-price results are unavailable."
            ),
        },
        {
            "Price section": f"Complete history ({strategy_result.best_history})",
            "Strategy return": f"{interval_result.durability_return_percent:.2f}%",
            "Buy and hold": f"{interval_result.durability_benchmark_return_percent:.2f}%",
            "Difference": f"{interval_result.durability_excess_return_percent:+.2f}%",
            "Completed trades": interval_result.durability_trades,
            "Plain English": (
                f"Maximum historical decline was {interval_result.durability_max_drawdown_percent:.2f}%. "
                + (
                    f"Return without the single best trade was {diagnostics.best_trade_removed_return_percent:+.2f}%."
                    if diagnostics else ""
                )
            ).strip(),
        },
    ]


def strategy_search_settings_records(strategy_result: StrategySearchResult) -> list[dict[str, Any]]:
    recommendation = strategy_result.best_result
    candidate = recommendation.best
    if candidate is None:
        return []
    evidence = recommendation.robustness
    settings_range = evidence.parameter_range if evidence else {}
    stable_range = bool(settings_range)
    _, newer_percent, _ = _optimizer_split_percentages(recommendation)
    return [
        {
            "Item": "Exact settings to test first",
            "Value": _settings_text(candidate.settings),
            "Plain English": f"These exact settings produced this strategy's best result in the newer {newer_percent}% of prices.",
        },
        {
            "Item": "Other settings with similar results",
            "Value": (
                _parameter_range_text(candidate.strategy_type, settings_range)
                if stable_range else "No stable nearby range found"
            ),
            "Plain English": (
                "Several distinct numeric combinations produced similar results across at least two inputs."
                if stable_range else
                "The surrounding numeric settings did not confirm this result. It may depend on one lucky combination."
            ),
        },
        {
            "Item": "Nearby settings that beat buy-and-hold",
            "Value": f"{candidate.plateau_profitable_percent:.0f}% of {candidate.plateau_neighbors}",
            "Plain English": (
                "A broader area that repeatedly beats buy-and-hold is more useful than one isolated winning setup."
            ),
        },
        {
            "Item": "Estimated trading friction",
            "Value": "Included in the detailed cost checks",
            "Plain English": "The strategy result includes Alpaca fees; the detailed test also shows added price slippage.",
        },
    ]


def strategy_search_interval_records(strategy_result: StrategySearchResult) -> list[dict[str, Any]]:
    rows = []
    older_percent, newer_percent, latest_percent = _optimizer_split_percentages(strategy_result.best_result)
    for interval_result in strategy_result.interval_results:
        candidate = interval_result.recommendation.best
        locked = interval_result.recommendation.robustness.locked_test if interval_result.recommendation.robustness else None
        if candidate is None:
            continue
        rows.append({
            "Interval": interval_result.interval,
            "History": interval_result.history,
            f"Older {older_percent}% Trades": candidate.train_trades,
            f"Older {older_percent}% vs Buy and Hold": f"{candidate.train_excess_return_percent:+.2f}%",
            f"Newer {newer_percent}% vs Buy and Hold": f"{candidate.excess_return_percent:+.2f}%",
            f"Latest {latest_percent}% vs Buy and Hold": f"{locked.excess_return_percent:+.2f}%" if locked else "Not available",
            "Complete History vs Buy and Hold": f"{interval_result.durability_excess_return_percent:+.2f}%",
            "Complete History Trades": interval_result.durability_trades,
        })
    return rows


def _settings_text(settings: dict[str, Any]) -> str:
    strategy_type = str(settings.get("strategy_type", ""))
    parts = []
    if strategy_type != "pullback":
        parts.append(f"buy lookback {int(settings['entry_window'])}")
    parts.extend([
        f"sell exit {int(settings['exit_window'])}",
        f"stop {float(settings['atr_stop_multiplier']):.2f}x ATR",
        f"trend filter {int(settings['moving_average_window'])}",
    ])
    if strategy_type == "pullback":
        parts.append(f"pullback average {int(settings['pullback_average_length'])}")
    if strategy_type in {"pullback", "trendline_retest"}:
        parts.append(f"momentum turn {int(settings['momentum_turn_length'])}")
    return "; ".join(parts)


def _parameter_range_text(strategy_type: str, ranges: dict[str, tuple[float, float]]) -> str:
    if not ranges:
        return "Not available"
    labels = {
        "entry_window": "buy lookback",
        "exit_window": "sell exit",
        "atr_stop_multiplier": "stop ATR",
        "moving_average_window": "trend filter",
        "pullback_average_length": "pullback average",
        "momentum_turn_length": "momentum turn",
        "rsi_length": "RSI length",
        "rsi_oversold": "oversold level",
        "rsi_overbought": "overbought cap",
        "rsi_decline_points": "arming drop",
        "rsi_rebound_points": "buy rebound",
        "rsi_sell_recovery_points": "sell recovery",
        "rsi_swing_lookback": "RSI lookback",
        "rsi_max_holding_bars": "maximum hold",
    }
    if strategy_type == "rsi_scalp":
        keys = [
            "rsi_length", "rsi_oversold", "rsi_decline_points", "rsi_rebound_points",
            "rsi_sell_recovery_points", "rsi_overbought", "rsi_swing_lookback",
            "rsi_max_holding_bars", "atr_stop_multiplier",
        ]
        parts = []
        for key in keys:
            if key not in ranges:
                continue
            low, high = ranges[key]
            number = lambda value: f"{value:.2f}" if key == "atr_stop_multiplier" else f"{value:.0f}"
            parts.append(f"{labels[key]} {number(low)}-{number(high)}")
        return "; ".join(parts) if parts else "Not available"
    keys = ["exit_window", "atr_stop_multiplier", "moving_average_window"]
    if strategy_type != "pullback":
        keys.insert(0, "entry_window")
    if strategy_type == "pullback":
        keys.append("pullback_average_length")
    if strategy_type in {"pullback", "trendline_retest"}:
        keys.append("momentum_turn_length")
    parts = []
    for key in keys:
        low, high = ranges[key]
        number = lambda value: f"{value:.2f}" if key == "atr_stop_multiplier" else f"{value:.0f}"
        parts.append(f"{labels[key]} {number(low)}-{number(high)}")
    return "; ".join(parts)


def _locked_text(locked: LockedTestResult | None) -> str:
    if locked is None:
        return "Not available"
    status = "Passed" if locked.passed else "Did not pass"
    return (
        f"{status}: {locked.return_percent:.2f}% allocated return, {locked.trades} trades, "
        f"{locked.max_drawdown_percent:.2f}% allocated worst drop, {locked.excess_return_percent:+.2f}% vs buy-and-hold"
    )


def _annualized_text(value: float | None) -> str:
    return "Not shown (period is 1 year or less)" if value is None else f"{value:.2f}%"


def _best_market_condition(rows: list[dict[str, Any]]) -> str:
    eligible = [row for row in rows if int(row.get("Trades", 0)) > 0]
    if not eligible:
        return "Not enough trades"
    best = max(eligible, key=lambda row: float(row.get("Return Percent", 0)))
    return f"{best['Market condition']} ({float(best['Return Percent']):+.2f}%)"


def _stress_text(row: dict[str, Any] | None) -> str:
    if not row:
        return "Not available"
    status = "Passed" if row["Passed"] else "Did not pass"
    return f"{status}: {float(row['Return Percent']):.2f}% allocated return"


def _trial_text(trial: TrialAdjustment | None) -> str:
    if trial is None:
        return "Not available"
    if trial.evidence.startswith("Insufficient"):
        return "Not enough completed trades"
    return f"{trial.deflated_sharpe_probability_percent:.1f}% after {trial.tested_candidates} settings"


def _bootstrap_text(bootstrap: BootstrapResult | None) -> str:
    if bootstrap is None:
        return "Not enough completed trades"
    return (
        f"{bootstrap.fifth_percentile_return_percent:.2f}% fifth-percentile allocated return; "
        f"{bootstrap.loss_probability_percent:.1f}% chance of loss"
    )


def _evaluate_optimizer_candidate(
    *,
    strategy_label: str,
    strategy_type: str,
    settings: dict[str, Any],
    market_data: pd.DataFrame,
    account_equity: float,
    risk_limits: RiskLimits | None,
    train_fraction: float,
    locked_fraction: float,
    min_test_trades: int,
    target_max_drawdown_percent: float,
) -> OptimizerCandidate:
    data = market_data.copy()
    total_bars = len(data)
    warmup_bars = _warmup_bars(settings)
    locked_start = int(total_bars * (1.0 - locked_fraction))
    split_index = min(int(total_bars * train_fraction), locked_start - 30)
    if split_index < warmup_bars or locked_start - split_index < 30 or total_bars - locked_start < 30:
        raise ValueError("Not enough bars for strategy input search.")

    train_data = data.iloc[:split_index].copy()
    train_data.attrs["symbol"] = data.attrs.get("symbol", "MARKET")
    train_result = _run_one(strategy_type, train_data, settings, account_equity, risk_limits)
    train_stats = _closed_trade_stats(
        account_equity,
        train_result["trade_log"],
        split_index,
        risk_limits,
        elapsed_years(train_data.index),
    )
    allocation = ticker_allocated_capital(account_equity, risk_limits)
    train_benchmark = buy_and_hold_benchmark(
        data,
        account_equity,
        start=0,
        end=split_index,
        allocated_capital=allocation,
    )

    oos_start = max(0, split_index - warmup_bars)
    warmup_offset = split_index - oos_start
    oos_data = data.iloc[oos_start:locked_start].copy()
    oos_data.attrs["symbol"] = data.attrs.get("symbol", "MARKET")
    oos_result = _run_one(strategy_type, oos_data, settings, account_equity, risk_limits)
    oos_trades = [
        trade for trade in oos_result["trade_log"]
        if int(trade.get("entry_bar", 0)) >= warmup_offset
    ]
    validation_bars = locked_start - split_index
    oos_stats = _closed_trade_stats(
        account_equity,
        oos_trades,
        validation_bars,
        risk_limits,
        elapsed_years(data.index, start=split_index, end=locked_start),
    )
    benchmark = buy_and_hold_benchmark(
        data,
        account_equity,
        start=split_index,
        end=locked_start,
        allocated_capital=allocation,
    )

    tested_periods = min(3, max(1, validation_bars // 30))
    period_pnl = [0.0] * tested_periods
    test_span = max(1, validation_bars)
    for trade in oos_trades:
        global_entry = oos_start + int(trade.get("entry_bar", 0))
        relative_entry = max(0, global_entry - split_index)
        bucket = min(tested_periods - 1, int(relative_entry * tested_periods / test_span))
        period_pnl[bucket] += float(trade.get("pnl", 0))

    test_trades = int(oos_stats["total_trades"])
    return_pct = float(oos_stats["allocated_return_pct"])
    profit_factor = float(oos_stats["profit_factor"])
    drawdown = float(oos_stats["allocated_max_drawdown_pct"])
    win_rate = float(oos_stats["win_rate"])
    train_return = float(train_stats["allocated_return_pct"])
    train_trades = int(train_stats["total_trades"])
    train_excess_return = train_return - train_benchmark.return_percent
    profitable_periods = sum(value > 0 for value in period_pnl)
    excess_return = return_pct - benchmark.return_percent
    drawdown_advantage = benchmark.max_drawdown_percent - drawdown
    score = round(excess_return, 2)
    confidence = _optimizer_confidence(score, test_trades, drawdown, return_pct, profitable_periods, tested_periods)
    concern = _optimizer_concern(test_trades, drawdown, return_pct, train_return, min_test_trades)
    reason = (
        f"Middle validation-period allocated return {return_pct:.2f}%, {test_trades} trades, "
        f"profit factor {profit_factor:.2f}, allocated worst drop {drawdown:.2f}%, "
        f"profitable in {profitable_periods}/{tested_periods} newer periods, "
        f"{excess_return:+.2f}% versus buy-and-hold."
    )
    recommended_risk = _recommended_risk(settings, drawdown, target_max_drawdown_percent)
    trade_returns = tuple(
        float(trade.get("pnl", 0)) / float(trade.get("notional", 0))
        for trade in oos_trades
        if float(trade.get("notional", 0)) > 0
    )
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
        tested_periods=tested_periods,
        train_return_percent=round(train_return, 2),
        train_trades=train_trades,
        train_benchmark_return_percent=train_benchmark.return_percent,
        train_excess_return_percent=round(train_excess_return, 2),
        stability=_stability_read(score, test_trades, drawdown, train_return, return_pct, profitable_periods, tested_periods),
        recommended_risk_per_trade_percent=recommended_risk,
        validation_trade_returns=trade_returns,
        benchmark_return_percent=benchmark.return_percent,
        benchmark_max_drawdown_percent=benchmark.max_drawdown_percent,
        excess_return_percent=round(excess_return, 2),
        drawdown_advantage_percent=round(drawdown_advantage, 2),
        test_account_return_percent=float(oos_stats["return_pct"]),
        test_annualized_return_percent=oos_stats["annualized_allocated_return_pct"],
        benchmark_account_return_percent=benchmark.account_return_percent,
        benchmark_annualized_return_percent=benchmark.annualized_return_percent,
        allocated_capital=allocation,
    )


def _evaluate_range(
    strategy_type: str,
    settings: dict[str, Any],
    data: pd.DataFrame,
    start: int,
    end: int,
    account_equity: float,
    risk_limits: RiskLimits | None,
) -> tuple[dict[str, Any], list[dict]]:
    warmup = _warmup_bars(settings)
    warm_start = max(0, start - warmup)
    segment = data.iloc[warm_start:end].copy()
    segment.attrs["symbol"] = data.attrs.get("symbol", "MARKET")
    result = _run_one(strategy_type, segment, settings, account_equity, risk_limits)
    offset = start - warm_start
    trades = [trade for trade in result["trade_log"] if int(trade.get("entry_bar", 0)) >= offset]
    return _closed_trade_stats(
        account_equity,
        trades,
        max(1, end - start),
        risk_limits,
        elapsed_years(data.index, start=start, end=end),
    ), trades


def _attach_plateau_scores(candidates: list[OptimizerCandidate]) -> list[OptimizerCandidate]:
    updated = []
    for candidate in candidates:
        neighbors = _distinct_numeric_neighbors(candidate, candidates)
        returns = [row.excess_return_percent for row in neighbors]
        median_return = float(median(returns)) if returns else candidate.excess_return_percent
        profitable_percent = sum(value > 0 for value in returns) / len(returns) * 100 if returns else 0.0
        updated.append(replace(
            candidate,
            plateau_neighbors=len(neighbors),
            plateau_median_return_percent=round(median_return, 2),
            plateau_profitable_percent=round(profitable_percent, 1),
        ))
    return updated


def _distinct_numeric_neighbors(
    candidate: OptimizerCandidate,
    candidates: list[OptimizerCandidate],
) -> list[OptimizerCandidate]:
    neighbors: dict[tuple[str, tuple[tuple[str, float | int], ...]], OptimizerCandidate] = {}
    for row in candidates:
        if row.strategy_type != candidate.strategy_type:
            continue
        if _settings_distance(candidate.settings, row.settings) > 3.5:
            continue
        identity = _optimizer_settings_identity(row.strategy_type, row.settings)
        existing = neighbors.get(identity)
        if existing is None or row.excess_return_percent > existing.excess_return_percent:
            neighbors[identity] = row
    return list(neighbors.values())


def _similar_profitable_neighbors(
    best: OptimizerCandidate,
    candidates: list[OptimizerCandidate],
) -> list[OptimizerCandidate]:
    if best.excess_return_percent <= 0:
        return []
    tolerance = max(1.0, abs(best.excess_return_percent) * 0.20)
    return [
        row for row in _distinct_numeric_neighbors(best, candidates)
        if row.excess_return_percent > 0
        and row.excess_return_percent >= best.excess_return_percent - tolerance
    ]


def _rolling_evidence(
    strategy_type: str,
    settings: dict[str, Any],
    data: pd.DataFrame,
    account_equity: float,
    risk_limits: RiskLimits | None,
    *,
    locked_fraction: float,
    windows: int,
) -> dict[str, Any]:
    end = int(len(data) * (1.0 - locked_fraction))
    start = _warmup_bars(settings)
    usable = max(0, end - start)
    window_count = min(max(1, windows), max(1, usable // 30))
    boundaries = np.linspace(start, end, window_count + 1, dtype=int)
    rows = []
    for index in range(window_count):
        left, right = int(boundaries[index]), int(boundaries[index + 1])
        if right - left < 10:
            continue
        try:
            stats, _ = _evaluate_range(
                strategy_type, settings, data, left, right, account_equity, risk_limits
            )
        except ValueError:
            continue
        rows.append(stats)
    returns = [float(row["allocated_return_pct"]) for row in rows]
    return {
        "windows": len(rows),
        "profitable_windows": sum(value > 0 for value in returns),
        "median_return_percent": round(float(median(returns)), 2) if returns else 0.0,
        "worst_drawdown_percent": round(max((float(row["allocated_max_drawdown_pct"]) for row in rows), default=0.0), 2),
    }


def _parameter_plateau_range(
    best: OptimizerCandidate,
    candidates: list[OptimizerCandidate],
) -> dict[str, tuple[float, float]]:
    neighbors = _similar_profitable_neighbors(best, candidates)
    keys = _optimizer_numeric_keys(best.strategy_type)
    varied_keys = sum(
        len({row.settings[key] for row in neighbors}) > 1
        for key in keys
    )
    if len(neighbors) < 4 or varied_keys < 2:
        return {}
    return {
        key: (
            min(float(row.settings[key]) for row in neighbors),
            max(float(row.settings[key]) for row in neighbors),
        )
        for key in keys
    }


def _locked_test(
    strategy_type: str,
    settings: dict[str, Any],
    data: pd.DataFrame,
    account_equity: float,
    risk_limits: RiskLimits | None,
    *,
    locked_fraction: float,
    min_trades: int,
) -> tuple[LockedTestResult, list[dict]]:
    start = int(len(data) * (1.0 - locked_fraction))
    stats, trades = _evaluate_range(
        strategy_type, settings, data, start, len(data), account_equity, risk_limits
    )
    benchmark = buy_and_hold_benchmark(
        data,
        account_equity,
        start=start,
        end=len(data),
        allocated_capital=ticker_allocated_capital(account_equity, risk_limits),
    )
    excess_return = float(stats["allocated_return_pct"]) - benchmark.return_percent
    passed = bool(
        stats["total_trades"] >= min_trades
        and stats["allocated_return_pct"] > 0
        and stats["profit_factor"] >= 1.0
        and stats["allocated_max_drawdown_pct"] <= 10.0
    )
    if stats["total_trades"] < min_trades:
        detail = "Not enough completed trades in the locked period."
    elif passed:
        detail = "The untouched final period stayed profitable with controlled drawdown."
    else:
        detail = "The untouched final period did not confirm every profitability and drawdown requirement."
    return LockedTestResult(
        return_percent=float(stats["allocated_return_pct"]),
        trades=int(stats["total_trades"]),
        win_rate_percent=float(stats["win_rate"]),
        profit_factor=float(stats["profit_factor"]),
        max_drawdown_percent=float(stats["allocated_max_drawdown_pct"]),
        passed=passed,
        detail=detail,
        benchmark_return_percent=benchmark.return_percent,
        benchmark_max_drawdown_percent=benchmark.max_drawdown_percent,
        excess_return_percent=round(excess_return, 2),
        account_return_percent=float(stats["return_pct"]),
        annualized_return_percent=stats["annualized_allocated_return_pct"],
        benchmark_annualized_return_percent=benchmark.annualized_return_percent,
    ), trades


def _execution_stress_rows(
    trades: list[dict],
    account_equity: float,
    risk_limits: RiskLimits | None,
) -> list[dict[str, Any]]:
    rows = []
    for basis_points in (0, 5, 10, 20):
        adjusted = []
        for trade in trades:
            entry = float(trade.get("entry", 0))
            exit_price = float(trade.get("exit", 0))
            shares = float(trade.get("shares", 0))
            execution_cost = (entry + exit_price) * shares * basis_points / 10_000
            record = dict(trade)
            record["pnl"] = float(trade.get("pnl", 0)) - execution_cost
            record["max_adverse_pnl"] = float(trade.get("max_adverse_pnl", 0)) - execution_cost
            adjusted.append(record)
        stats = _closed_trade_stats(account_equity, adjusted, max(1, len(adjusted)), risk_limits)
        rows.append({
            "Round-trip slippage": f"{basis_points} bps per side",
            "Return Percent": stats["allocated_return_pct"],
            "Profit Factor": stats["profit_factor"],
            "Worst Drop Percent": stats["allocated_max_drawdown_pct"],
            "Passed": bool(stats["total_trades"] > 0 and stats["allocated_return_pct"] > 0),
        })
    return rows


def _candidate_diagnostics(
    candidate: OptimizerCandidate,
    trades: list[dict],
    stress_rows: list[dict[str, Any]],
    regime_rows: list[dict[str, Any]],
    account_equity: float,
) -> CandidateDiagnostics:
    adjusted_trade_returns = []
    pnl_values = []
    for trade in trades:
        entry = float(trade.get("entry", 0))
        exit_price = float(trade.get("exit", 0))
        shares = float(trade.get("shares", 0))
        notional = float(trade.get("notional", 0)) or entry * shares
        execution_cost = (entry + exit_price) * shares * 10 / 10_000
        adjusted_pnl = float(trade.get("pnl", 0)) - execution_cost
        pnl_values.append(adjusted_pnl)
        if notional > 0:
            adjusted_trade_returns.append(adjusted_pnl / notional * 100)

    allocation = max(0.0, float(candidate.allocated_capital))
    total_pnl = sum(pnl_values)
    best_pnl = max(pnl_values, default=0.0)
    positive_pnl = sum(value for value in pnl_values if value > 0)
    best_removed_return = (total_pnl - best_pnl) / allocation * 100 if allocation > 0 else 0.0
    best_share = best_pnl / positive_pnl * 100 if best_pnl > 0 and positive_pnl > 0 else 0.0
    slippage_survival = max(
        (
            int(str(row.get("Round-trip slippage", "0")).split()[0])
            for row in stress_rows
            if bool(row.get("Passed"))
        ),
        default=0,
    )
    populated_regimes = [row for row in regime_rows if int(row.get("Trades", 0)) > 0]
    profitable_regime_percent = (
        sum(float(row.get("Return Percent", 0)) > 0 for row in populated_regimes) / len(populated_regimes) * 100
        if populated_regimes
        else None
    )
    account_drawdown = (
        candidate.test_max_drawdown_percent * allocation / account_equity
        if account_equity > 0
        else 0.0
    )
    return_to_drawdown = (
        candidate.test_return_percent / candidate.test_max_drawdown_percent
        if candidate.test_max_drawdown_percent > 0
        else 0.0
    )
    return CandidateDiagnostics(
        after_cost_expectancy_percent=round(float(np.mean(adjusted_trade_returns)), 3) if adjusted_trade_returns else 0.0,
        best_trade_removed_return_percent=round(best_removed_return, 2),
        best_trade_share_percent=round(best_share, 1),
        account_drawdown_percent=round(account_drawdown, 2),
        slippage_survival_bps=slippage_survival,
        profitable_regime_percent=round(profitable_regime_percent, 1) if profitable_regime_percent is not None else None,
        return_to_drawdown=round(return_to_drawdown, 2),
    )


def _regime_rows(
    data: pd.DataFrame,
    trades: list[dict],
    account_equity: float,
    settings: dict[str, Any],
    risk_limits: RiskLimits | None,
) -> list[dict[str, Any]]:
    if data.empty or not trades:
        return []
    close = data["Close"].astype(float)
    trend_window = max(10, int(settings.get("moving_average_window", 50)))
    trend_average = close.rolling(trend_window).mean()
    trend_prior = trend_average.shift(max(1, trend_window // 10))
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            data["High"].astype(float) - data["Low"].astype(float),
            (data["High"].astype(float) - previous_close).abs(),
            (data["Low"].astype(float) - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_percent = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / close * 100
    volatility_cutoff = float(atr_percent.dropna().median()) if not atr_percent.dropna().empty else 0.0
    groups: dict[str, list[dict]] = {
        "Rising trend": [], "Sideways trend": [], "Falling trend": [],
        "Lower volatility": [], "Higher volatility": [],
    }
    for trade in trades:
        index = int(trade.get("entry_bar", -1))
        if index < 0 or index >= len(data):
            continue
        average = trend_average.iloc[index]
        prior = trend_prior.iloc[index]
        price = close.iloc[index]
        if pd.notna(average) and pd.notna(prior) and price > average and average > prior:
            groups["Rising trend"].append(trade)
        elif pd.notna(average) and pd.notna(prior) and price < average and average < prior:
            groups["Falling trend"].append(trade)
        else:
            groups["Sideways trend"].append(trade)
        volatility = atr_percent.iloc[index]
        groups["Higher volatility" if pd.notna(volatility) and volatility > volatility_cutoff else "Lower volatility"].append(trade)
    rows = []
    for name, group in groups.items():
        stats = _closed_trade_stats(account_equity, group, max(1, len(group)), risk_limits)
        rows.append({
            "Market condition": name,
            "Trades": stats["total_trades"],
            "Return Percent": stats["allocated_return_pct"],
            "Win Rate Percent": stats["win_rate"],
            "Profit Factor": stats["profit_factor"],
            "Worst Drop Percent": stats["allocated_max_drawdown_pct"],
        })
    return rows


def _bootstrap_trades(
    trades: list[dict],
    account_equity: float,
    *,
    samples: int,
    seed: int = 42,
) -> BootstrapResult | None:
    pnl = np.asarray([float(trade.get("pnl", 0)) for trade in trades], dtype=float)
    if len(pnl) < 2 or samples <= 0:
        return None
    rng = np.random.default_rng(seed)
    returns = []
    drawdowns = []
    for _ in range(samples):
        sequence = rng.choice(pnl, size=len(pnl), replace=True)
        equity = account_equity + np.cumsum(sequence)
        curve = np.concatenate(([account_equity], equity))
        peaks = np.maximum.accumulate(curve)
        drawdown = np.max(np.where(peaks > 0, (peaks - curve) / peaks * 100, 0))
        returns.append(float(sequence.sum() / account_equity * 100))
        drawdowns.append(float(drawdown))
    return BootstrapResult(
        samples=samples,
        completed_trades=len(trades),
        median_return_percent=round(float(np.median(returns)), 2),
        fifth_percentile_return_percent=round(float(np.percentile(returns, 5)), 2),
        loss_probability_percent=round(float(np.mean(np.asarray(returns) < 0) * 100), 1),
        ninety_fifth_percentile_drawdown_percent=round(float(np.percentile(drawdowns, 95)), 2),
    )


def _trade_sharpe(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    array = np.asarray(values, dtype=float)
    deviation = float(np.std(array, ddof=1))
    return float(np.mean(array) / deviation) if deviation > 0 else 0.0


def _trial_adjustment(
    best: OptimizerCandidate,
    candidates: list[OptimizerCandidate],
) -> TrialAdjustment | None:
    values = np.asarray(best.validation_trade_returns, dtype=float)
    candidate_sharpes = [
        _trade_sharpe(row.validation_trade_returns)
        for row in candidates
        if len(row.validation_trade_returns) >= 2
    ]
    if len(values) < 3 or len(candidate_sharpes) < 2:
        return TrialAdjustment(
            tested_candidates=len(candidates),
            trade_sharpe=round(_trade_sharpe(best.validation_trade_returns), 3),
            expected_best_sharpe_from_search=0.0,
            deflated_sharpe_probability_percent=0.0,
            evidence="Insufficient completed validation trades for a trial-adjusted probability.",
        )
    sharpe = _trade_sharpe(best.validation_trade_returns)
    sharpe_std = float(np.std(candidate_sharpes, ddof=1))
    trials = max(2, len(candidates))
    normal = NormalDist()
    gamma = 0.5772156649
    expected_best = sharpe_std * (
        (1 - gamma) * normal.inv_cdf(1 - 1 / trials)
        + gamma * normal.inv_cdf(1 - 1 / (trials * e))
    )
    centered = values - float(np.mean(values))
    sigma = float(np.std(values, ddof=0))
    skew = float(np.mean((centered / sigma) ** 3)) if sigma > 0 else 0.0
    kurtosis = float(np.mean((centered / sigma) ** 4)) if sigma > 0 else 3.0
    denominator = max(1e-9, 1 - skew * sharpe + ((kurtosis - 1) / 4) * sharpe ** 2)
    statistic = (sharpe - expected_best) * sqrt(len(values) - 1) / sqrt(denominator)
    probability = normal.cdf(statistic) * 100
    evidence = (
        "Stronger after accounting for the number of settings tested."
        if probability >= 80
        else "Moderate after accounting for the number of settings tested."
        if probability >= 60
        else "Weak after accounting for the number of settings tested."
    )
    return TrialAdjustment(
        tested_candidates=len(candidates),
        trade_sharpe=round(sharpe, 3),
        expected_best_sharpe_from_search=round(expected_best, 3),
        deflated_sharpe_probability_percent=round(probability, 1),
        evidence=evidence,
    )


def _final_confidence(
    best: OptimizerCandidate,
    locked: LockedTestResult,
    stress_rows: list[dict[str, Any]],
    trial: TrialAdjustment | None,
) -> tuple[str, str]:
    rolling_ratio = best.rolling_profitable_windows / max(1, best.rolling_windows)
    stress_10 = next((row for row in stress_rows if row["Round-trip slippage"] == "10 bps per side"), {})
    trial_probability = trial.deflated_sharpe_probability_percent if trial else 0.0
    if locked.passed and rolling_ratio >= 0.75 and bool(stress_10.get("Passed")) and trial_probability >= 80:
        return "High", "Strong enough for paper testing across rolling windows, locked data, execution stress, and trial adjustment."
    if locked.passed and rolling_ratio >= 0.5 and bool(stress_10.get("Passed")):
        return "Medium", "Reasonable for paper testing, but the statistical or rolling evidence is not uniformly strong."
    return "Low", "Historical evidence is fragile or incomplete; treat this as a research candidate, not a preferred setup."


def validate_settings_across_tickers(
    settings: dict[str, Any],
    market_data_by_symbol: dict[str, pd.DataFrame],
    account_equity: float,
    risk_limits: RiskLimits | None = None,
) -> CrossTickerResult:
    strategy_type = str(settings.get("strategy_type", "trendline_retest"))
    rows = []
    for symbol, data in market_data_by_symbol.items():
        try:
            result = _run_one(strategy_type, data, settings, account_equity, risk_limits)
            stats = _closed_trade_stats(
                account_equity,
                result["trade_log"],
                len(data),
                risk_limits,
                elapsed_years(data.index),
            )
            rows.append({
                "Ticker": str(symbol).upper(),
                "Trades": stats["total_trades"],
                "Return Percent": stats["allocated_return_pct"],
                "Profit Factor": stats["profit_factor"],
                "Worst Drop Percent": stats["allocated_max_drawdown_pct"],
                "Profitable": bool(stats["total_trades"] >= 2 and stats["allocated_return_pct"] > 0),
            })
        except ValueError as exc:
            rows.append({
                "Ticker": str(symbol).upper(), "Trades": 0, "Return Percent": 0.0,
                "Profit Factor": 0.0, "Worst Drop Percent": 0.0,
                "Profitable": False, "Problem": str(exc),
            })
    returns = [float(row["Return Percent"]) for row in rows]
    return CrossTickerResult(
        tested_tickers=len(rows),
        profitable_tickers=sum(bool(row["Profitable"]) for row in rows),
        median_return_percent=round(float(median(returns)), 2) if returns else 0.0,
        worst_drawdown_percent=round(max((float(row["Worst Drop Percent"]) for row in rows), default=0.0), 2),
        rows=rows,
    )


def _closed_trade_stats(
    account: float,
    trade_log: list[dict],
    eval_bars: int,
    risk_limits: RiskLimits | None = None,
    years: float | None = None,
) -> dict[str, Any]:
    wins = [trade for trade in trade_log if float(trade.get("pnl", 0)) > 0]
    losses = [trade for trade in trade_log if float(trade.get("pnl", 0)) <= 0]
    total_pnl = round(sum(float(trade.get("pnl", 0)) for trade in trade_log), 2)
    gross_wins = sum(float(trade.get("pnl", 0)) for trade in wins)
    gross_losses = abs(sum(float(trade.get("pnl", 0)) for trade in losses))
    equity = account
    peak = account
    max_drawdown_dollars = 0.0
    for trade in trade_log:
        adverse_equity = equity + float(trade.get("max_adverse_pnl", 0))
        max_drawdown_dollars = max(max_drawdown_dollars, peak - adverse_equity)
        equity += float(trade.get("pnl", 0))
        peak = max(peak, equity)
        max_drawdown_dollars = max(max_drawdown_dollars, peak - equity)
    exposure_bars = sum(max(0, int(trade.get("exit_bar", 0)) - int(trade.get("entry_bar", 0))) for trade in trade_log)
    avg_loss = round(sum(float(trade.get("pnl", 0)) for trade in losses) / len(losses), 2) if losses else 0
    avg_win = round(gross_wins / len(wins), 2) if wins else 0
    capital = allocation_metrics(
        account_equity=account,
        total_pnl=total_pnl,
        max_drawdown_dollars=max_drawdown_dollars,
        risk_limits=risk_limits,
        years=years,
    )
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
        "rr_ratio": round(abs(avg_win / avg_loss), 2) if avg_loss else (99.0 if avg_win > 0 else 0),
        "profit_factor": round(gross_wins / gross_losses, 2) if gross_losses else (99.0 if gross_wins > 0 else 0),
        "max_drawdown_pct": round(max_drawdown_dollars / account * 100, 2) if account else 0,
        "exposure_pct": round(exposure_bars / eval_bars * 100, 2) if eval_bars else 0,
        **capital,
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
        "rsi_entry_filter_enabled": bool(settings.get("rsi_entry_filter_enabled", False)),
        "rsi_length": int(settings.get("rsi_length", 14)),
        "rsi_oversold": float(settings.get("rsi_oversold", 30.0)),
        "rsi_overbought": float(settings.get("rsi_overbought", 70.0)),
        "rsi_decline_points": float(settings.get("rsi_decline_points", 40.0)),
        "rsi_rebound_points": float(settings.get("rsi_rebound_points", 3.0)),
        "rsi_sell_recovery_points": float(settings.get("rsi_sell_recovery_points", 35.0)),
        "rsi_swing_lookback": int(settings.get("rsi_swing_lookback", 24)),
        "rsi_stop_mode": str(settings.get("rsi_stop_mode", "standard_atr")),
        "rsi_emergency_atr_multiplier": float(settings.get("rsi_emergency_atr_multiplier", 5.0)),
        "rsi_max_holding_enabled": bool(settings.get("rsi_max_holding_enabled", True)),
        "rsi_max_holding_bars": int(settings.get("rsi_max_holding_bars", 100)),
        "rsi_profit_only_exit": bool(settings.get("rsi_profit_only_exit", False)),
    }


def _settings_distance(base: dict[str, Any], row: dict[str, Any]) -> float:
    distances = (
        abs(row["entry_window"] - base["entry_window"]) / 10,
        abs(row["exit_window"] - base["exit_window"]) / 5,
        abs(row["atr_stop_multiplier"] - base["atr_stop_multiplier"]),
        abs(row["moving_average_window"] - base["moving_average_window"]) / 50,
        abs(row["pullback_average_length"] - base["pullback_average_length"]) / 20,
        abs(row["momentum_turn_length"] - base["momentum_turn_length"]) / 5,
        1.0 if row.get("rsi_entry_filter_enabled", False) != base.get("rsi_entry_filter_enabled", False) else 0.0,
        abs(row.get("rsi_length", 14) - base.get("rsi_length", 14)) / 7,
        abs(row.get("rsi_oversold", 30.0) - base.get("rsi_oversold", 30.0)) / 5,
        abs(row.get("rsi_overbought", 70.0) - base.get("rsi_overbought", 70.0)) / 5,
        abs(row.get("rsi_decline_points", 40.0) - base.get("rsi_decline_points", 40.0)) / 10,
        abs(row.get("rsi_rebound_points", 3.0) - base.get("rsi_rebound_points", 3.0)) / 2,
        abs(row.get("rsi_sell_recovery_points", 35.0) - base.get("rsi_sell_recovery_points", 35.0)) / 5,
        abs(row.get("rsi_swing_lookback", 24) - base.get("rsi_swing_lookback", 24)) / 12,
        1.0 if row.get("rsi_stop_mode", "standard_atr") != base.get("rsi_stop_mode", "standard_atr") else 0.0,
        abs(row.get("rsi_emergency_atr_multiplier", 5.0) - base.get("rsi_emergency_atr_multiplier", 5.0)),
        1.0 if row.get("rsi_max_holding_enabled", True) != base.get("rsi_max_holding_enabled", True) else 0.0,
        abs(row.get("rsi_max_holding_bars", 100) - base.get("rsi_max_holding_bars", 100)) / 50,
    )
    return float(sum(distances))


def _warmup_bars(settings: dict[str, Any]) -> int:
    return max(
        int(settings.get("entry_window", 20)),
        int(settings.get("exit_window", 10)),
        int(settings.get("moving_average_window", 50)),
        int(settings.get("pullback_average_length", 20)),
        int(settings.get("momentum_turn_length", 10)),
        int(settings.get("rsi_length", 14)),
        int(settings.get("rsi_swing_lookback", 24)),
        14,
    ) + 4


def _optimizer_confidence(
    score: float,
    trades: int,
    drawdown: float,
    return_pct: float,
    profitable_periods: int,
    tested_periods: int,
) -> str:
    if trades <= 0:
        return "Low"
    if score >= 8 and trades >= 4 and return_pct > 0 and drawdown <= 10 and profitable_periods >= max(2, tested_periods - 1):
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


def _stability_read(
    score: float,
    trades: int,
    drawdown: float,
    train_return: float,
    return_pct: float,
    profitable_periods: int,
    tested_periods: int,
) -> str:
    if trades <= 0:
        return "Low confidence because the newer test period had no completed trades."
    if tested_periods > 1 and profitable_periods <= 1:
        return "Low confidence because most newer periods were not profitable."
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
