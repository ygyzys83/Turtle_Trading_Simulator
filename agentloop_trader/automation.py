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
    return AutomationDecision(action="paper_order_candidate", ready=True, reasons=["Dry-run only; no order submitted."])


def automation_decision_records(decision: AutomationDecision) -> list[dict]:
    return [
        {"Field": "Action", "Value": decision.action},
        {"Field": "Ready", "Value": decision.ready},
        {"Field": "Reasons", "Value": "; ".join(decision.reasons)},
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
            "Action": candidate.action,
            "Ready": candidate.ready,
            "Dry Run Only": candidate.dry_run_only,
            "Broker Write Required": candidate.broker_write_required,
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
        preview_hash = str(preview.get("Preview Hash", ""))
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
                reasons=(["Dry-run only; no exit order submitted."] if ready else list(dict.fromkeys(blockers or [hold_reason])))
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
                reasons=["Dry-run only; no cancel request submitted."],
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
    broker_write_candidates = sum(bool(row.get("Broker Write Required")) for row in candidates)
    return [
        {"Check": "Dry Run Only", "Passed": True, "Detail": "Automation queue never submits broker orders."},
        {"Check": "Alpaca Connected", "Passed": broker_connected, "Detail": "Broker reads are available." if broker_connected else "Alpaca account is not connected."},
        {"Check": "Broker State Fresh", "Passed": not broker_state_stale, "Detail": "Positions/orders refreshed." if not broker_state_stale else "Refresh broker state before automation."},
        {"Check": "Manual Order Gate", "Passed": manual_order_gate_enabled, "Detail": "Manual gate is enabled." if manual_order_gate_enabled else "Manual gate is off; broker writes remain disabled."},
        {"Check": "Kill Switch Off", "Passed": not kill_switch_enabled, "Detail": "Session kill switch is off." if not kill_switch_enabled else "Kill switch is active."},
        {"Check": "Ready Candidates", "Passed": ready_candidates > 0, "Detail": f"{ready_candidates} ready candidate(s); {broker_write_candidates} would require broker write approval."},
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
        "broker_write_candidate_count": sum(bool(row.get("Broker Write Required")) for row in snapshot.candidates),
        "hold_count": sum(str(row.get("Action", "")).endswith("_hold") for row in snapshot.candidates),
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
        {"Metric": "Dry Run Snapshots", "Value": len(snapshot_records)},
        {"Metric": "Dry Run Candidates", "Value": candidate_count},
        {"Metric": "Ready Candidates", "Value": ready_count},
        {"Metric": "Broker Write Candidates", "Value": write_count},
        {"Metric": "Hold Candidates", "Value": hold_count},
    ]
    for action, count in sorted(action_counts.items()):
        rows.append({"Metric": f"Action: {action}", "Value": count})
    for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:8]:
        rows.append({"Metric": f"Hold Reason: {reason}", "Value": count})
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
        decision = "halt"
        detail = "; ".join(row.get("Detail", "") for row in active_halts if row.get("Detail"))
    elif readiness_blockers:
        decision = "hold"
        detail = "; ".join(row.get("Detail", "") for row in readiness_blockers if row.get("Detail"))
    elif ready_candidates:
        decision = "would_queue_manual_review"
        detail = f"{len(ready_candidates)} candidate(s) would be queued for manual approval."
    else:
        decision = "hold"
        detail = "No ready automation candidates."
    return [
        {"Field": "Supervisor Mode", "Value": "dry_run_only"},
        {"Field": "Decision", "Value": decision},
        {"Field": "Ready Candidates", "Value": len(ready_candidates)},
        {"Field": "Active Halt Reasons", "Value": len(active_halts)},
        {"Field": "Broker Writes Submitted", "Value": 0},
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
        {"Metric": "Audit Events Loaded", "Value": len(audit_records)},
        {"Metric": "Alpaca Paper Orders Submitted", "Value": event_types.count("alpaca_paper_order_submitted")},
        {"Metric": "Alpaca Paper Orders Armed", "Value": event_types.count("alpaca_paper_order_armed")},
        {"Metric": "Alpaca Paper Orders Blocked", "Value": event_types.count("alpaca_paper_order_blocked")},
        {"Metric": "Alpaca Paper Cancels Submitted", "Value": event_types.count("alpaca_paper_cancel_submitted")},
        {"Metric": "Alpaca Paper Cancels Armed", "Value": event_types.count("alpaca_paper_cancel_armed")},
        {"Metric": "Alpaca Paper Cancels Blocked", "Value": event_types.count("alpaca_paper_cancel_blocked")},
        {"Metric": "Alpaca Paper Exits Submitted", "Value": event_types.count("alpaca_paper_exit_submitted")},
        {"Metric": "Alpaca Paper Exits Armed", "Value": event_types.count("alpaca_paper_exit_armed")},
        {"Metric": "Alpaca Paper Exits Blocked", "Value": event_types.count("alpaca_paper_exit_blocked")},
        {"Metric": "Tracked Alpaca Orders", "Value": len(tracked_orders)},
        {"Metric": "Tracked Alpaca Open Orders", "Value": lifecycle_statuses.count("open_at_alpaca")},
        {"Metric": "Tracked Alpaca Filled Orders", "Value": tracked_statuses.count("filled")},
        {"Metric": "Tracked Alpaca Filled Quantity", "Value": _format_number(filled_quantity)},
        {"Metric": "Tracked Alpaca Canceled Orders", "Value": tracked_statuses.count("canceled") + tracked_statuses.count("cancelled")},
        {"Metric": "Tracked Alpaca Missing Orders", "Value": lifecycle_statuses.count("missing_from_alpaca_orders")},
        {"Metric": "Shadow Decisions", "Value": event_types.count("shadow_decision_recorded")},
        {"Metric": "Local Paper Orders", "Value": len([event for event in event_types if event.startswith("paper_order_")])},
    ]


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
