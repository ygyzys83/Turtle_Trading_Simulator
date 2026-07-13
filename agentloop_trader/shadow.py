from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from agentloop_trader.models import ExecutionDecision, PACIFIC_TIME, PreflightCheckResult, RiskCheckResult, TradeIntent


@dataclass(frozen=True)
class ShadowDecision:
    shadow_id: str
    created_at: datetime
    symbol: str
    intended_action: str
    quantity: float
    reference_price: float | None
    risk_approved: bool
    preflight_ready: bool
    execution_reason: str
    blocked_reasons: list[str]


def record_shadow_decision(
    intent: TradeIntent | None,
    risk_check: RiskCheckResult,
    execution_decision: ExecutionDecision,
    preflight: PreflightCheckResult,
) -> ShadowDecision:
    return ShadowDecision(
        shadow_id=str(uuid4()),
        created_at=datetime.now(PACIFIC_TIME),
        symbol=intent.symbol_clean if intent else "NONE",
        intended_action=intent.side if intent else "none",
        quantity=intent.quantity if intent else 0,
        reference_price=intent.entry_price if intent else None,
        risk_approved=risk_check.approved,
        preflight_ready=preflight.ready,
        execution_reason=execution_decision.reason,
        blocked_reasons=preflight.blocked_reasons,
    )


def shadow_records(decisions: list[ShadowDecision]) -> list[dict]:
    return [
        {
            "Time": decision.created_at.astimezone(PACIFIC_TIME).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "Practice ID": decision.shadow_id[:8],
            "Symbol": decision.symbol,
            "Action": decision.intended_action.upper(),
            "Quantity": decision.quantity,
            "Reference Price": decision.reference_price,
            "Trade Allowed": decision.risk_approved,
            "Ready To Send": decision.preflight_ready,
            "Reason": decision.execution_reason,
            "Blocked Reasons": "; ".join(dict.fromkeys(decision.blocked_reasons)),
        }
        for decision in decisions
    ]
