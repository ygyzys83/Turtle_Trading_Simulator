from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo


Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop"]
TimeInForce = Literal["day", "gtc"]
ExecutionMode = Literal[
    "backtest_only",
    "paper",
    "shadow",
    "live_with_approval",
    "automated_live",
]

PACIFIC_TIME = ZoneInfo("America/Los_Angeles")


@dataclass(frozen=True)
class StrategyConfig:
    name: str = "Turtle Trend Following"
    entry_window: int = 20
    exit_window: int = 10
    atr_window: int = 14
    atr_stop_multiplier: float = 2.0
    risk_per_trade_pct: float = 1.0
    moving_average_window: int = 200


@dataclass(frozen=True)
class TradeIntent:
    symbol: str
    side: Side
    quantity: int
    order_type: OrderType = "market"
    time_in_force: TimeInForce = "day"
    limit_price: float | None = None
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    max_holding_bars: int | None = None
    rationale: str = ""
    proposed_by_agent: str = "strategy_engine"
    confidence: float | None = None
    source_signals: list[str] = field(default_factory=list)

    @property
    def symbol_clean(self) -> str:
        return self.symbol.strip().upper()

    @property
    def estimated_notional(self) -> float:
        if self.entry_price is None:
            return 0.0
        return max(0.0, self.quantity * self.entry_price)

    @property
    def estimated_risk_dollars(self) -> float:
        if self.entry_price is None or self.stop_loss is None:
            return 0.0
        if self.side == "buy":
            return max(0.0, self.entry_price - self.stop_loss) * self.quantity
        return max(0.0, self.stop_loss - self.entry_price) * self.quantity


@dataclass(frozen=True)
class TradeThesis:
    symbol: str
    thesis: str
    data_basis: list[str]
    invalidation: str
    generated_by: str = "human"
    created_at: datetime = field(default_factory=lambda: datetime.now(PACIFIC_TIME))


@dataclass(frozen=True)
class BacktestResult:
    starting_equity: float
    final_equity: float
    total_pnl: float
    return_pct: float
    total_trades: int
    win_rate: float
    wins: int
    losses: int
    avg_win: float
    avg_loss: float
    reward_to_risk: float
    profit_factor: float
    max_drawdown_pct: float
    exposure_pct: float


@dataclass(frozen=True)
class RiskLimits:
    allowed_symbols: tuple[str, ...] = ()
    max_risk_per_trade_pct: float = 1.0
    max_position_notional_pct: float = 25.0
    max_portfolio_exposure_pct: float = 75.0
    max_symbol_concentration_pct: float = 35.0
    max_session_loss_pct: float = 2.0
    max_quantity: int = 100_000
    max_open_positions: int = 5
    allow_add_to_existing_position: bool = False
    require_stop_loss: bool = True
    kill_switch_enabled: bool = False


@dataclass(frozen=True)
class RiskCheckResult:
    approved: bool
    rejected_reasons: list[str]
    checks: dict[str, bool]
    risk_dollars: float = 0.0
    notional_dollars: float = 0.0


@dataclass(frozen=True)
class ExecutionDecision:
    mode: ExecutionMode
    approved_for_execution: bool
    requires_manual_approval: bool
    reason: str
    risk_check: RiskCheckResult


@dataclass(frozen=True)
class PreflightCheckResult:
    ready: bool
    checks: dict[str, bool]
    blocked_reasons: list[str]


@dataclass(frozen=True)
class TradeProposal:
    thesis: TradeThesis
    trade_intent: TradeIntent | None
    risk_check: RiskCheckResult
    execution_decision: ExecutionDecision
    loop_stage: str = "propose_evaluate_decide"


@dataclass(frozen=True)
class TradingSessionConfig:
    mode: ExecutionMode = "backtest_only"
    account_equity: float = 50_000.0
    allowed_symbols: tuple[str, ...] = ()
    max_daily_loss_pct: float = 2.0
    kill_switch_enabled: bool = False


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(PACIFIC_TIME))
