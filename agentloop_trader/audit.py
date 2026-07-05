from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from agentloop_trader.models import AuditEvent, ExecutionDecision, RiskCheckResult, TradeIntent, TradeProposal


def _clean_payload(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _clean_payload(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _clean_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_payload(v) for v in value]
    return value


def build_audit_events(
    mode_label: str,
    source_caption: str,
    trade_intent: TradeIntent | None,
    risk_check: RiskCheckResult,
    execution_decision: ExecutionDecision,
    stats: dict,
    trade_proposal: TradeProposal | None = None,
) -> list[AuditEvent]:
    events = [
        AuditEvent(
            event_type="backtest_completed",
            message="Backtest harness completed for the selected strategy settings.",
            payload={
                "source": source_caption,
                "final_equity": stats.get("final_equity"),
                "total_trades": stats.get("total_trades"),
                "max_drawdown_pct": stats.get("max_drawdown_pct"),
            },
        )
    ]

    if trade_proposal is not None:
        events.append(
            AuditEvent(
                event_type="research_proposal_generated",
                message=f"Rules research agent generated a proposal for {trade_proposal.thesis.symbol}.",
                payload={
                    "symbol": trade_proposal.thesis.symbol,
                    "generated_by": trade_proposal.thesis.generated_by,
                    "loop_stage": trade_proposal.loop_stage,
                    "has_trade_intent": trade_proposal.trade_intent is not None,
                },
            )
        )

    if trade_intent is None:
        events.append(
            AuditEvent(
                event_type="trade_intent_absent",
                message="No current trade intent was generated.",
                payload={"signal_state": "flat_or_no_entry"},
            )
        )
    else:
        events.append(
            AuditEvent(
                event_type="trade_intent_generated",
                message=f"Trade intent generated for {trade_intent.symbol_clean}.",
                payload=trade_intent,
            )
        )

    events.append(
        AuditEvent(
            event_type="risk_check_completed",
            message="Deterministic risk gate evaluated the trade intent.",
            payload=risk_check,
        )
    )
    events.append(
        AuditEvent(
            event_type="execution_decision_recorded",
            message=f"Execution decision recorded in {mode_label} mode.",
            payload=execution_decision,
        )
    )
    return events


def events_to_records(events: list[AuditEvent]) -> list[dict[str, Any]]:
    records = []
    for event in events:
        records.append({
            "Time": event.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "Type": event.event_type,
            "Message": event.message,
            "Payload": _clean_payload(event.payload),
        })
    return records
