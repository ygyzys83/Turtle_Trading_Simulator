from __future__ import annotations

from dataclasses import replace

from agentloop_trader.models import (
    ExecutionDecision,
    ExecutionMode,
    PreflightCheckResult,
    RiskCheckResult,
    RiskLimits,
    TradeIntent,
)


def constrain_trade_intent_to_limits(
    intent: TradeIntent | None,
    account_equity: float,
    limits: RiskLimits,
    current_portfolio_notional: float = 0.0,
    symbol_current_notional: float = 0.0,
    available_cash: float | None = None,
) -> TradeIntent | None:
    if intent is None or intent.entry_price is None or intent.quantity <= 0:
        return intent

    entry_price = float(intent.entry_price)
    if entry_price <= 0:
        return replace(intent, quantity=0)

    max_quantities = [intent.quantity, limits.max_quantity]
    if intent.stop_loss is not None:
        risk_per_share = abs(entry_price - float(intent.stop_loss))
        max_risk_dollars = account_equity * limits.max_risk_per_trade_pct / 100
        max_quantities.append(int(max_risk_dollars // risk_per_share) if risk_per_share > 0 else 0)

    max_position_notional = account_equity * limits.max_position_notional_pct / 100
    max_quantities.append(int(max_position_notional // entry_price))

    remaining_portfolio_notional = account_equity * limits.max_portfolio_exposure_pct / 100 - current_portfolio_notional
    max_quantities.append(int(max(0.0, remaining_portfolio_notional) // entry_price))

    remaining_symbol_notional = account_equity * limits.max_symbol_concentration_pct / 100 - symbol_current_notional
    max_quantities.append(int(max(0.0, remaining_symbol_notional) // entry_price))

    if available_cash is not None and intent.side == "buy":
        max_quantities.append(int(max(0.0, available_cash) // entry_price))

    adjusted_quantity = max(0, min(max_quantities))
    if adjusted_quantity == intent.quantity:
        return intent

    rationale = intent.rationale
    if adjusted_quantity > 0:
        rationale = f"{rationale} Quantity reduced from {intent.quantity:,} to {adjusted_quantity:,} by deterministic risk sizing."
    else:
        rationale = f"{rationale} Deterministic risk sizing found no allowable quantity under current limits."

    source_signals = list(intent.source_signals)
    if "deterministic_risk_sizing" not in source_signals:
        source_signals.append("deterministic_risk_sizing")

    return replace(intent, quantity=adjusted_quantity, rationale=rationale.strip(), source_signals=source_signals)


def check_trade_intent(
    intent: TradeIntent | None,
    account_equity: float,
    limits: RiskLimits,
    open_positions: set[str] | None = None,
    open_position_count: int | None = None,
    current_portfolio_notional: float = 0.0,
    symbol_current_notional: float = 0.0,
    session_pnl: float = 0.0,
    available_cash: float | None = None,
) -> RiskCheckResult:
    open_positions = open_positions or set()
    open_position_count = len(open_positions) if open_position_count is None else open_position_count
    checks: dict[str, bool] = {}
    rejected: list[str] = []

    if intent is None:
        return RiskCheckResult(
            approved=False,
            rejected_reasons=["No trade intent was generated."],
            checks={"intent_present": False},
        )

    symbol = intent.symbol_clean
    risk_dollars = intent.estimated_risk_dollars
    notional_dollars = intent.estimated_notional

    checks["kill_switch_off"] = not limits.kill_switch_enabled
    if limits.kill_switch_enabled:
        rejected.append("Kill Switch is on.")

    checks["symbol_allowed"] = not limits.allowed_symbols or symbol in limits.allowed_symbols
    if not checks["symbol_allowed"]:
        rejected.append(f"{symbol} is not in the allowed symbol list.")

    checks["quantity_positive"] = intent.quantity > 0
    if not checks["quantity_positive"]:
        rejected.append("Quantity must be greater than zero.")

    checks["quantity_within_limit"] = intent.quantity <= limits.max_quantity
    if not checks["quantity_within_limit"]:
        rejected.append(f"Quantity exceeds max quantity of {limits.max_quantity:,}.")

    checks["stop_loss_present"] = (not limits.require_stop_loss) or intent.stop_loss is not None
    if not checks["stop_loss_present"]:
        rejected.append("Stop loss is required.")

    max_risk_dollars = account_equity * limits.max_risk_per_trade_pct / 100
    checks["risk_within_limit"] = risk_dollars <= max_risk_dollars
    if not checks["risk_within_limit"]:
        rejected.append(
            f"Estimated trade risk ${risk_dollars:,.2f} exceeds max "
            f"${max_risk_dollars:,.2f}."
        )

    max_notional = account_equity * limits.max_position_notional_pct / 100
    checks["notional_within_limit"] = notional_dollars <= max_notional
    if not checks["notional_within_limit"]:
        rejected.append(
            f"Estimated notional ${notional_dollars:,.2f} exceeds max "
            f"${max_notional:,.2f}."
        )

    checks["no_duplicate_position"] = limits.allow_add_to_existing_position or symbol not in {s.upper() for s in open_positions}
    if not checks["no_duplicate_position"]:
        rejected.append(f"{symbol} already has an open position.")

    checks["open_positions_within_limit"] = open_position_count < limits.max_open_positions or symbol in {s.upper() for s in open_positions}
    if not checks["open_positions_within_limit"]:
        rejected.append(f"Open positions would exceed max of {limits.max_open_positions}.")

    max_portfolio_notional = account_equity * limits.max_portfolio_exposure_pct / 100
    projected_portfolio_notional = current_portfolio_notional + notional_dollars
    checks["portfolio_exposure_within_limit"] = projected_portfolio_notional <= max_portfolio_notional
    if not checks["portfolio_exposure_within_limit"]:
        rejected.append(
            f"Projected portfolio exposure ${projected_portfolio_notional:,.2f} exceeds max "
            f"${max_portfolio_notional:,.2f}."
        )

    max_symbol_notional = account_equity * limits.max_symbol_concentration_pct / 100
    projected_symbol_notional = symbol_current_notional + notional_dollars
    checks["symbol_concentration_within_limit"] = projected_symbol_notional <= max_symbol_notional
    if not checks["symbol_concentration_within_limit"]:
        rejected.append(
            f"Projected {symbol} exposure ${projected_symbol_notional:,.2f} exceeds max "
            f"${max_symbol_notional:,.2f}."
        )

    max_session_loss = account_equity * limits.max_session_loss_pct / 100
    checks["session_loss_within_limit"] = session_pnl >= -max_session_loss
    if not checks["session_loss_within_limit"]:
        rejected.append(
            f"Session loss ${abs(session_pnl):,.2f} exceeds max ${max_session_loss:,.2f}."
        )

    if available_cash is not None:
        checks["cash_available"] = intent.side != "buy" or notional_dollars <= available_cash
        if not checks["cash_available"]:
            rejected.append(
                f"Estimated notional ${notional_dollars:,.2f} exceeds available cash ${available_cash:,.2f}."
            )

    return RiskCheckResult(
        approved=not rejected,
        rejected_reasons=rejected,
        checks=checks,
        risk_dollars=round(risk_dollars, 2),
        notional_dollars=round(notional_dollars, 2),
    )


def build_preflight_check(
    intent: TradeIntent | None,
    risk_check: RiskCheckResult,
    execution_decision: ExecutionDecision,
    broker_connected: bool,
    audit_logging_enabled: bool,
) -> PreflightCheckResult:
    checks = {
        "trade_intent_present": intent is not None,
        "risk_gate_approved": risk_check.approved,
        "execution_mode_allows_order": execution_decision.approved_for_execution,
        "manual_approval_not_required": not execution_decision.requires_manual_approval,
        "broker_state_present": broker_connected,
        "audit_logging_enabled": audit_logging_enabled,
    }
    blocked_reasons = []
    if intent is None:
        blocked_reasons.append("No trade intent is present.")
    if not risk_check.approved:
        blocked_reasons.extend(risk_check.rejected_reasons or ["Risk check did not approve the trade."])
    if not execution_decision.approved_for_execution and risk_check.approved:
        blocked_reasons.append(execution_decision.reason)
    if execution_decision.requires_manual_approval:
        blocked_reasons.append("Manual approval is required.")
    if not broker_connected:
        blocked_reasons.append("Broker state is unavailable.")
    if not audit_logging_enabled:
        blocked_reasons.append("Audit logging is unavailable.")
    blocked_reasons = list(dict.fromkeys(blocked_reasons))

    return PreflightCheckResult(
        ready=all(checks.values()),
        checks=checks,
        blocked_reasons=blocked_reasons,
    )


def preflight_records(preflight: PreflightCheckResult) -> list[dict]:
    return [
        {
            "Check": _display_check_name(name),
            "Passed": passed,
        }
        for name, passed in preflight.checks.items()
    ]


def risk_policy_records(limits: RiskLimits) -> list[dict]:
    return [
        {"Policy": "Allowed symbols", "Value": ", ".join(limits.allowed_symbols) if limits.allowed_symbols else "Any"},
        {"Policy": "Max risk per trade", "Value": f"{limits.max_risk_per_trade_pct}%"},
        {"Policy": "Max position notional", "Value": f"{limits.max_position_notional_pct}%"},
        {"Policy": "Max portfolio exposure", "Value": f"{limits.max_portfolio_exposure_pct}%"},
        {"Policy": "Max symbol concentration", "Value": f"{limits.max_symbol_concentration_pct}%"},
        {"Policy": "Max session loss", "Value": f"{limits.max_session_loss_pct}%"},
        {"Policy": "Max open positions", "Value": limits.max_open_positions},
        {"Policy": "Add to existing position", "Value": limits.allow_add_to_existing_position},
        {"Policy": "Stop loss required", "Value": limits.require_stop_loss},
        {"Policy": "Kill Switch on", "Value": limits.kill_switch_enabled},
        {"Policy": "Broker target", "Value": "Alpaca paper account"},
    ]


def decide_execution(mode: ExecutionMode, risk_check: RiskCheckResult) -> ExecutionDecision:
    if not risk_check.approved:
        detail = "; ".join(dict.fromkeys(risk_check.rejected_reasons))
        reason = f"Risk rules blocked this trade: {detail}" if detail else "Risk rules blocked this trade."
        return ExecutionDecision(
            mode=mode,
            approved_for_execution=False,
            requires_manual_approval=False,
            reason=reason,
            risk_check=risk_check,
        )

    if mode == "backtest_only":
        return ExecutionDecision(
            mode=mode,
            approved_for_execution=False,
            requires_manual_approval=False,
            reason="Backtest only does not send orders.",
            risk_check=risk_check,
        )

    if mode in {"shadow", "live_with_approval"}:
        return ExecutionDecision(
            mode=mode,
            approved_for_execution=False,
            requires_manual_approval=True,
            reason="Risk rules passed, but this mode needs your approval.",
            risk_check=risk_check,
        )

    return ExecutionDecision(
        mode=mode,
        approved_for_execution=True,
        requires_manual_approval=False,
        reason="Risk rules passed for the selected order mode.",
        risk_check=risk_check,
    )


def _display_check_name(name: str) -> str:
    return {
        "risk_approved": "Risk rules passed",
        "execution_approved": "Order mode allows sending",
        "broker_connected": "Alpaca connected",
        "audit_logging_enabled": "Activity log enabled",
        "preflight": "Ready to send",
    }.get(name, name.replace("_", " ").title())
