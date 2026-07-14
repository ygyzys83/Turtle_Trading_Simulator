from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agentloop_trader.broker_governance import BrokerStateHealth
from agentloop_trader.models import PACIFIC_TIME, PreflightCheckResult, RiskCheckResult, TradeIntent


DEFAULT_AUTOMATION_DRY_RUN_PATH = Path("automation_logs") / "paper_automation_dry_runs.jsonl"


@dataclass(frozen=True)
class AutomationDecision:
    action: str
    ready: bool
    reasons: list[str]


@dataclass(frozen=True)
class AutomationCandidate:
    action: str
    ready: bool
    broker_write_required: bool
    dry_run_only: bool
    symbol: str
    side: str
    quantity: str
    source: str
    reasons: list[str]


@dataclass(frozen=True)
class AutomationDryRunSnapshot:
    session_id: str
    candidates: list[dict]
    readiness: list[dict]
    created_at: str


@dataclass(frozen=True)
class AutoExitDecision:
    status: str
    ready: bool
    preview_hash: str
    symbol: str
    quantity: str
    reasons: list[str]


@dataclass(frozen=True)
class AutoEntryDecision:
    status: str
    ready: bool
    preview_hash: str
    symbol: str
    quantity: str
    reasons: list[str]


@dataclass(frozen=True)
class AutomationRuntimeState:
    mode: str
    status: str
    last_checked_at: str
    last_action: str
    blocked_reason: str


def strategy_settings_match(current: dict | None, entry: dict | None) -> bool:
    if not current or not entry:
        return False
    keys = [
        "symbol",
        "interval",
        "history",
        "strategy_type",
        "entry_window",
        "exit_window",
        "atr_stop_multiplier",
        "risk_per_trade_pct",
        "moving_average_window",
        "pullback_average_length",
        "momentum_turn_length",
        "rsi_length",
        "rsi_oversold",
        "rsi_overbought",
        "rsi_decline_points",
        "rsi_rebound_points",
        "rsi_sell_recovery_points",
        "rsi_swing_lookback",
        "rsi_stop_mode",
        "rsi_emergency_atr_multiplier",
        "rsi_max_holding_enabled",
        "rsi_max_holding_bars",
    ]
    return all(str(current.get(key, "")).strip().lower() == str(entry.get(key, "")).strip().lower() for key in keys)


def strategy_settings_match_reason(current: dict | None, entry: dict | None) -> str:
    if strategy_settings_match(current, entry):
        return "Exit settings match the buy settings."
    if not entry:
        return "No saved buy settings for this position; auto exit is paused."
    if not current:
        return "Current settings are unavailable; auto exit is paused."
    return "Settings changed since entry; auto exit is paused."


def active_automation_level(selected_level: str, full_automation_enabled: bool, kill_switch_enabled: bool) -> str:
    if selected_level == "Auto entries and exits" and kill_switch_enabled:
        return "Manual review only"
    if selected_level == "Auto entries and exits" and not full_automation_enabled:
        return "Auto exits only"
    return selected_level


class AutomationDryRunStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else DEFAULT_AUTOMATION_DRY_RUN_PATH

    def append(self, snapshot: AutomationDryRunSnapshot) -> dict:
        record = automation_snapshot_record(snapshot)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def read_recent(self, limit: int = 100) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        records = []
        for line in lines[-limit:]:
            if line.strip():
                records.append(json.loads(line))
        return records


def paper_automation_dry_run(
    intent: TradeIntent | None,
    risk_check: RiskCheckResult,
    preflight: PreflightCheckResult,
    broker_health: BrokerStateHealth,
    duplicate_reasons: list[str],
    idempotency_blocked: bool,
) -> AutomationDecision:
    reasons: list[str] = []
    if intent is None:
        reasons.append("No trade intent is present.")
    if not risk_check.approved:
        reasons.extend(risk_check.rejected_reasons)
    if not preflight.ready:
        reasons.extend(preflight.blocked_reasons)
    if broker_health.stale:
        reasons.extend(broker_health.reasons)
    reasons.extend(duplicate_reasons)
    if idempotency_blocked:
        reasons.append("This signal preview was already submitted or tracked.")

    deduped = list(dict.fromkeys(reasons))
    if deduped:
        return AutomationDecision(action="hold", ready=False, reasons=deduped)
    return AutomationDecision(action="paper_order_candidate", ready=True, reasons=["Check only; no order submitted."])


def automation_decision_records(decision: AutomationDecision) -> list[dict]:
    return [
        {"Field": "Action", "Value": decision.action},
        {"Field": "Ready", "Value": decision.ready},
        {"Field": "Reasons", "Value": "; ".join(decision.reasons)},
    ]


def auto_exit_decision(
    automation_level: str,
    execution_mode: str,
    broker_connected: bool,
    broker_can_submit: bool,
    paper_orders_enabled: bool,
    kill_switch_enabled: bool,
    broker_state_stale: bool,
    market_open: bool,
    strategy_exit_ready: bool,
    strategy_exit_reason: str,
    exit_preview_records: list[dict],
    exit_blockers: dict[str, list[str]],
    entry_settings_match: bool = True,
    entry_settings_reason: str = "",
    already_sent_hashes: set[str] | None = None,
) -> AutoExitDecision:
    reasons: list[str] = []
    already_sent_hashes = already_sent_hashes or set()
    if automation_level == "Manual review only":
        reasons.append("Automation level is Manual review only.")
    if execution_mode != "paper":
        reasons.append("How orders are handled must be Paper trading.")
    if not broker_connected:
        reasons.append("Alpaca paper is not connected.")
    if not broker_can_submit:
        reasons.append("Alpaca paper order submission is not available.")
    if not paper_orders_enabled:
        reasons.append("Use Alpaca paper account is off.")
    if kill_switch_enabled:
        reasons.append("Kill Switch is on.")
    if broker_state_stale:
        reasons.append("Refresh Alpaca positions and orders.")
    if not market_open:
        reasons.append("Market is closed; automatic exits wait for market hours.")
    if not strategy_exit_ready:
        reasons.append(strategy_exit_reason or "Strategy exit rule is not triggered.")
    if not entry_settings_match:
        reasons.append(entry_settings_reason or "Settings changed since entry; auto exit is paused.")
    if not exit_preview_records:
        reasons.append("No paper exit is available.")

    first_preview = next((row for row in exit_preview_records if row.get("Valid")), None)
    if first_preview is None:
        preview_hash = ""
        symbol = ""
        quantity = ""
    else:
        preview_hash = str(first_preview.get("Review ID", first_preview.get("Preview Hash", "")))
        symbol = str(first_preview.get("Symbol", ""))
        quantity = str(first_preview.get("Quantity", ""))
        blockers = exit_blockers.get(preview_hash, [])
        if blockers:
            reasons.extend(blockers)
        if preview_hash in already_sent_hashes:
            reasons.append("This exact auto exit was already sent.")

    reasons = list(dict.fromkeys(reason for reason in reasons if reason))
    if reasons:
        status = "Off" if automation_level == "Manual review only" else "Blocked"
        strategy_wait_reason = strategy_exit_reason or "Strategy exit rule is not triggered."
        if (
            automation_level != "Manual review only"
            and exit_preview_records
            and reasons == [strategy_wait_reason]
        ):
            status = "Waiting for exit signal"
        elif exit_preview_records and not reasons[:1] == ["Automation level is Manual review only."]:
            status = "Exit blocked"
        elif not exit_preview_records and automation_level != "Manual review only":
            status = "Watching position"
        return AutoExitDecision(
            status=status,
            ready=False,
            preview_hash=preview_hash,
            symbol=symbol,
            quantity=quantity,
            reasons=reasons,
        )
    return AutoExitDecision(
        status="Exit ready",
        ready=True,
        preview_hash=preview_hash,
        symbol=symbol,
        quantity=quantity,
        reasons=["Automatic paper exit can be sent now."],
    )


def auto_entry_decision(
    automation_level: str,
    execution_mode: str,
    broker_connected: bool,
    broker_can_submit: bool,
    paper_orders_enabled: bool,
    kill_switch_enabled: bool,
    broker_state_stale: bool,
    market_open: bool,
    intent_present: bool,
    risk_approved: bool,
    preflight_ready: bool,
    preview_valid: bool,
    preview_hash: str,
    symbol: str,
    quantity: int | str,
    blocked_reasons: list[str],
    already_sent_hashes: set[str] | None = None,
) -> AutoEntryDecision:
    reasons: list[str] = []
    already_sent_hashes = already_sent_hashes or set()
    if automation_level == "Manual review only":
        reasons.append("Automation level is Manual review only.")
    if automation_level == "Auto exits only":
        reasons.append("Auto entries are off.")
    if execution_mode != "paper":
        reasons.append("How orders are handled must be Paper trading.")
    if not broker_connected:
        reasons.append("Alpaca paper is not connected.")
    if not broker_can_submit:
        reasons.append("Alpaca paper order submission is not available.")
    if not paper_orders_enabled:
        reasons.append("Use Alpaca paper account is off.")
    if kill_switch_enabled:
        reasons.append("Kill Switch is on.")
    if broker_state_stale:
        reasons.append("Refresh Alpaca positions and orders.")
    if not market_open:
        reasons.append("Market is closed; automatic buys wait for market hours.")
    if not intent_present:
        reasons.append("No buy setup right now.")
    if not risk_approved:
        reasons.append("Risk check did not pass.")
    if not preflight_ready:
        reasons.append("Trade is blocked before broker submission.")
    if not preview_valid:
        reasons.append("Alpaca paper buy is not ready.")
    reasons.extend(blocked_reasons)
    if preview_hash and preview_hash in already_sent_hashes:
        reasons.append("This exact auto buy was already sent.")

    reasons = list(dict.fromkeys(reason for reason in reasons if reason))
    if reasons:
        status = "Off" if automation_level == "Manual review only" else "Buy blocked"
        if not intent_present and automation_level == "Auto entries and exits":
            status = "Waiting for buy setup"
        return AutoEntryDecision(
            status=status,
            ready=False,
            preview_hash=preview_hash,
            symbol=symbol,
            quantity=str(quantity),
            reasons=reasons,
        )
    return AutoEntryDecision(
        status="Buy ready",
        ready=True,
        preview_hash=preview_hash,
        symbol=symbol,
        quantity=str(quantity),
        reasons=["Automatic paper buy can be sent now."],
    )


def auto_entry_decision_records(decision: AutoEntryDecision) -> list[dict]:
    return [
        {"Field": "Status", "Value": decision.status},
        {"Field": "Ready To Send", "Value": "Yes" if decision.ready else "No"},
        {"Field": "Symbol", "Value": decision.symbol},
        {"Field": "Quantity", "Value": str(decision.quantity)},
        {"Field": "Review ID", "Value": decision.preview_hash},
        {"Field": "Reason", "Value": "; ".join(decision.reasons)},
    ]


def auto_exit_decision_records(decision: AutoExitDecision) -> list[dict]:
    return [
        {"Field": "Status", "Value": decision.status},
        {"Field": "Ready To Send", "Value": "Yes" if decision.ready else "No"},
        {"Field": "Symbol", "Value": decision.symbol},
        {"Field": "Quantity", "Value": str(decision.quantity)},
        {"Field": "Review ID", "Value": decision.preview_hash},
        {"Field": "Reason", "Value": "; ".join(decision.reasons)},
    ]


def automation_runtime_records(state: AutomationRuntimeState) -> list[dict]:
    return [
        {"Field": "Mode", "Value": state.mode},
        {"Field": "Status", "Value": state.status},
        {"Field": "Last Checked", "Value": state.last_checked_at},
        {"Field": "Last Action", "Value": state.last_action},
        {"Field": "Blocked Reason", "Value": state.blocked_reason},
    ]


def paper_automation_candidate_records(
    entry_decision: AutomationDecision,
    entry_symbol: str = "",
    entry_side: str = "",
    entry_quantity: int | str = "",
    exit_previews: list[dict] | None = None,
    cancelable_orders: list[dict] | None = None,
    exit_blockers: dict[str, list[str]] | None = None,
) -> list[dict]:
    candidates = build_paper_automation_candidates(
        entry_decision=entry_decision,
        entry_symbol=entry_symbol,
        entry_side=entry_side,
        entry_quantity=entry_quantity,
        exit_previews=exit_previews or [],
        cancelable_orders=cancelable_orders or [],
        exit_blockers=exit_blockers or {},
    )
    return [
        {
            "Action": _display_candidate_action(candidate.action),
            "Ready": candidate.ready,
            "No Orders Sent": candidate.dry_run_only,
            "Would Need Approval": candidate.broker_write_required,
            "Symbol": candidate.symbol,
            "Side": candidate.side,
            "Quantity": candidate.quantity,
            "Source": candidate.source,
            "Reasons": "; ".join(candidate.reasons),
        }
        for candidate in candidates
    ]


def build_paper_automation_candidates(
    entry_decision: AutomationDecision,
    entry_symbol: str = "",
    entry_side: str = "",
    entry_quantity: int | str = "",
    exit_previews: list[dict] | None = None,
    cancelable_orders: list[dict] | None = None,
    exit_blockers: dict[str, list[str]] | None = None,
) -> list[AutomationCandidate]:
    candidates = [
        AutomationCandidate(
            action="entry_candidate" if entry_decision.ready else "entry_hold",
            ready=entry_decision.ready,
            broker_write_required=entry_decision.ready,
            dry_run_only=True,
            symbol=str(entry_symbol or ""),
            side=str(entry_side or ""),
            quantity=str(entry_quantity or ""),
            source="strategy_entry_signal",
            reasons=entry_decision.reasons,
        )
    ]
    for preview in exit_previews or []:
        preview_hash = str(preview.get("Review ID", preview.get("Preview Hash", "")))
        blockers = (exit_blockers or {}).get(preview_hash, [])
        valid = bool(preview.get("Valid", False))
        ready = valid and not blockers
        hold_reason = preview.get("Blocked Reasons", "") or "Exit preview is not ready."
        candidates.append(
            AutomationCandidate(
                action="exit_candidate" if ready else "exit_hold",
                ready=ready,
                broker_write_required=ready,
                dry_run_only=True,
                symbol=str(preview.get("Symbol", "")),
                side=str(preview.get("Side", "")),
                quantity=str(preview.get("Quantity", "")),
                source="alpaca_position_exit_preview",
                reasons=(["Check only; no exit order submitted."] if ready else list(dict.fromkeys(blockers or [hold_reason])))
            )
        )
    for order in cancelable_orders or []:
        candidates.append(
            AutomationCandidate(
                action="cancel_candidate",
                ready=True,
                broker_write_required=True,
                dry_run_only=True,
                symbol=str(order.get("Symbol", "")),
                side=str(order.get("Side", "")),
                quantity=str(order.get("Quantity", "")),
                source="alpaca_open_order_cancel_preview",
                reasons=["Check only; no cancel request submitted."],
            )
        )
    return candidates


def automation_readiness_records(
    broker_connected: bool,
    broker_state_stale: bool,
    manual_order_gate_enabled: bool,
    kill_switch_enabled: bool,
    candidates: list[dict],
) -> list[dict]:
    ready_candidates = sum(bool(row.get("Ready")) for row in candidates)
    broker_write_candidates = sum(bool(row.get("Would Need Approval", row.get("Broker Write Required"))) for row in candidates)
    return [
        {"Check": "No orders sent", "Passed": True, "Detail": "Automation check never submits broker orders."},
        {"Check": "Alpaca connected", "Passed": broker_connected, "Detail": "Alpaca is connected." if broker_connected else "Alpaca account is not connected."},
        {"Check": "Alpaca data current", "Passed": not broker_state_stale, "Detail": "Positions and orders are refreshed." if not broker_state_stale else "Refresh Alpaca before automation checks."},
        {"Check": "Paper orders allowed", "Passed": manual_order_gate_enabled, "Detail": "Paper orders are allowed." if manual_order_gate_enabled else "Paper orders are not allowed right now."},
        {"Check": "Kill Switch off", "Passed": not kill_switch_enabled, "Detail": "Kill Switch is off." if not kill_switch_enabled else "Kill Switch is on."},
        {"Check": "Paper actions ready", "Passed": ready_candidates > 0, "Detail": f"{ready_candidates} paper action(s) ready; {broker_write_candidates} could contact Alpaca paper."},
    ]


def build_automation_snapshot(session_id: str, candidates: list[dict], readiness: list[dict]) -> AutomationDryRunSnapshot:
    return AutomationDryRunSnapshot(
        session_id=session_id,
        candidates=candidates,
        readiness=readiness,
        created_at=datetime.now(PACIFIC_TIME).isoformat(),
    )


def automation_snapshot_record(snapshot: AutomationDryRunSnapshot) -> dict:
    return {
        "created_at": snapshot.created_at,
        "session_id": snapshot.session_id,
        "candidate_count": len(snapshot.candidates),
        "ready_candidate_count": sum(bool(row.get("Ready")) for row in snapshot.candidates),
        "broker_write_candidate_count": sum(bool(row.get("Would Need Approval", row.get("Broker Write Required"))) for row in snapshot.candidates),
        "hold_count": sum(_is_blocked_action(str(row.get("Action", ""))) for row in snapshot.candidates),
        "candidates": snapshot.candidates,
        "readiness": snapshot.readiness,
    }


def automation_evidence_records(snapshot_records: list[dict]) -> list[dict]:
    candidate_count = sum(int(record.get("candidate_count", 0)) for record in snapshot_records)
    ready_count = sum(int(record.get("ready_candidate_count", 0)) for record in snapshot_records)
    write_count = sum(int(record.get("broker_write_candidate_count", 0)) for record in snapshot_records)
    hold_count = sum(int(record.get("hold_count", 0)) for record in snapshot_records)
    action_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for record in snapshot_records:
        for candidate in record.get("candidates", []):
            action = str(candidate.get("Action", ""))
            action_counts[action] = action_counts.get(action, 0) + 1
            if not candidate.get("Ready"):
                reasons = str(candidate.get("Reasons", "")).split(";")
                for reason in reasons:
                    reason = reason.strip()
                    if reason:
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    rows = [
        {"Metric": "Saved checks", "Value": len(snapshot_records)},
        {"Metric": "Actions checked", "Value": candidate_count},
        {"Metric": "Paper actions ready", "Value": ready_count},
        {"Metric": "Actions that would need approval", "Value": write_count},
        {"Metric": "Actions blocked", "Value": hold_count},
    ]
    for action, count in sorted(action_counts.items()):
        rows.append({"Metric": f"Action type: {action}", "Value": count})
    for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:8]:
        rows.append({"Metric": f"Blocked reason: {reason}", "Value": count})
    return rows


def automation_supervisor_dry_run_records(
    candidates: list[dict],
    readiness_rows: list[dict],
    halt_rows: list[dict],
) -> list[dict]:
    ready_candidates = [row for row in candidates if row.get("Ready")]
    active_halts = [row for row in halt_rows if row.get("Active")]
    readiness_blockers = [row for row in readiness_rows if not row.get("Passed")]
    if active_halts:
        decision = "blocked"
        detail = "; ".join(row.get("Detail", "") for row in active_halts if row.get("Detail"))
    elif readiness_blockers:
        decision = "blocked"
        detail = "; ".join(row.get("Detail", "") for row in readiness_blockers if row.get("Detail"))
    elif ready_candidates:
        decision = "ready"
        detail = f"{len(ready_candidates)} paper action(s) ready."
    else:
        decision = "blocked"
        detail = "No automation actions are ready."
    return [
        {"Field": "Mode", "Value": "check only"},
        {"Field": "Decision", "Value": decision},
        {"Field": "Paper actions ready", "Value": len(ready_candidates)},
        {"Field": "Active blocks", "Value": len(active_halts)},
        {"Field": "Orders sent", "Value": 0},
        {"Field": "Detail", "Value": detail},
    ]


def evidence_dashboard_records(audit_records: list[dict], tracked_orders: list[dict]) -> list[dict]:
    event_types = [record.get("event_type", "") for record in audit_records]
    tracked_statuses = [_enum_value(order.get("status", "")) for order in tracked_orders]
    lifecycle_statuses = [str(order.get("lifecycle_status", "")) for order in tracked_orders]
    filled_quantity = sum(
        _as_float(order.get("filled_quantity") or order.get("quantity"))
        for order in tracked_orders
        if _enum_value(order.get("status", "")) == "filled"
    )
    return [
        {"Metric": "Activity records loaded", "Value": len(audit_records)},
        {"Metric": "Paper buys sent", "Value": event_types.count("alpaca_paper_order_submitted") + event_types.count("auto_paper_entry_submitted")},
        {"Metric": "Paper buys blocked", "Value": event_types.count("alpaca_paper_order_blocked") + event_types.count("auto_paper_entry_blocked")},
        {"Metric": "Paper cancels sent", "Value": event_types.count("alpaca_paper_cancel_submitted")},
        {"Metric": "Paper cancels blocked", "Value": event_types.count("alpaca_paper_cancel_blocked")},
        {"Metric": "Paper exits sent", "Value": event_types.count("alpaca_paper_exit_submitted") + event_types.count("auto_paper_exit_submitted")},
        {"Metric": "Paper exits blocked", "Value": event_types.count("alpaca_paper_exit_blocked") + event_types.count("auto_paper_exit_blocked")},
        {"Metric": "Saved Alpaca orders", "Value": len(tracked_orders)},
        {"Metric": "Open Alpaca orders", "Value": lifecycle_statuses.count("open_at_alpaca")},
        {"Metric": "Filled Alpaca orders", "Value": tracked_statuses.count("filled")},
        {"Metric": "Filled Alpaca shares", "Value": _format_number(filled_quantity)},
        {"Metric": "Canceled Alpaca orders", "Value": tracked_statuses.count("canceled") + tracked_statuses.count("cancelled")},
        {"Metric": "Saved orders missing at Alpaca", "Value": lifecycle_statuses.count("missing_from_alpaca_orders")},
        {"Metric": "Practice decisions", "Value": event_types.count("shadow_decision_recorded")},
        {"Metric": "App paper orders", "Value": len([event for event in event_types if event.startswith("paper_order_")])},
    ]


def _display_candidate_action(action: str) -> str:
    return {
        "entry_candidate": "Buy ready to send",
        "entry_hold": "Buy blocked",
        "exit_candidate": "Exit ready to send",
        "exit_hold": "Exit blocked",
        "cancel_candidate": "Cancel ready to send",
    }.get(action, action)


def _is_blocked_action(action: str) -> bool:
    return action.endswith("_hold") or "blocked" in action.lower()


def _enum_value(value) -> str:
    text = str(value or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.strip().lower()


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.8f}".rstrip("0").rstrip(".")
