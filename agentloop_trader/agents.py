from __future__ import annotations

from agentloop_trader.models import (
    ExecutionDecision,
    RiskCheckResult,
    TradeIntent,
    TradeProposal,
    TradeThesis,
)


def generate_research_thesis(
    symbol: str,
    live: dict,
    stats: dict,
    trade_intent: TradeIntent | None,
    risk_check: RiskCheckResult,
) -> TradeThesis:
    symbol_clean = symbol.strip().upper() or "SYNTH"
    last_price = live.get("last_p")
    breakout_level = live.get("don_high")
    exit_level = live.get("don_low")
    atr = live.get("last_atr")
    sma_up = live.get("sma_up")
    signal = live.get("signal")

    data_basis = [
        f"Latest price: ${last_price:,.2f}" if last_price else "Latest price unavailable",
        f"Breakout level: ${breakout_level:,.2f}" if breakout_level else "Breakout level unavailable",
        f"Exit level: ${exit_level:,.2f}" if exit_level else "Exit level unavailable",
        f"ATR: ${atr:,.2f}" if atr else "ATR unavailable",
        f"SMA trend filter: {'upward' if sma_up else 'not upward'}",
        f"Backtest return: {stats.get('return_pct')}%",
        f"Win rate: {stats.get('win_rate')}%",
        f"Max drawdown: {stats.get('max_drawdown_pct')}%",
        f"Profit factor: {stats.get('profit_factor')}",
    ]

    if trade_intent is not None:
        thesis = (
            f"{symbol_clean} has generated a governed turtle-strategy entry proposal: "
            f"price is above the breakout level while the trend filter is supportive. "
            "The proposal is only actionable if deterministic risk and execution gates approve it."
        )
        invalidation = (
            f"Invalidate the long thesis if price falls to the stop near "
            f"${trade_intent.stop_loss:,.2f}, loses the exit channel, or a risk rule rejects the order."
        )
    elif signal == "exit":
        thesis = (
            f"{symbol_clean} is not a new-entry candidate because the current state is an exit signal. "
            "The workflow should prioritize risk reduction over new exposure."
        )
        invalidation = "Reconsider only after price stabilizes, clears the entry channel again, and risk checks pass."
    else:
        rejected = "; ".join(risk_check.rejected_reasons) if risk_check.rejected_reasons else "no current entry signal"
        thesis = (
            f"{symbol_clean} is not currently actionable. The strategy remains flat because {rejected}. "
            "Discipline means waiting for a valid breakout rather than forcing a trade."
        )
        invalidation = "Reconsider when price breaks above the entry channel with an upward trend filter and valid stop."

    return TradeThesis(
        symbol=symbol_clean,
        thesis=thesis,
        data_basis=data_basis,
        invalidation=invalidation,
        generated_by="rules_research_agent",
    )


def build_trade_proposal(
    symbol: str,
    live: dict,
    stats: dict,
    trade_intent: TradeIntent | None,
    risk_check: RiskCheckResult,
    execution_decision: ExecutionDecision,
) -> TradeProposal:
    thesis = generate_research_thesis(
        symbol=symbol,
        live=live,
        stats=stats,
        trade_intent=trade_intent,
        risk_check=risk_check,
    )
    return TradeProposal(
        thesis=thesis,
        trade_intent=trade_intent,
        risk_check=risk_check,
        execution_decision=execution_decision,
    )


def proposal_records(proposal: TradeProposal) -> list[dict]:
    intent = proposal.trade_intent
    return [
        {"Field": "Agent", "Value": proposal.thesis.generated_by},
        {"Field": "Loop", "Value": proposal.loop_stage},
        {"Field": "Symbol", "Value": proposal.thesis.symbol},
        {"Field": "Intent", "Value": "None" if intent is None else f"{intent.side.upper()} {intent.quantity}"},
        {"Field": "Risk decision", "Value": "Approved" if proposal.risk_check.approved else "Rejected"},
        {"Field": "Execution decision", "Value": proposal.execution_decision.reason},
    ]

