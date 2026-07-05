from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_EVIDENCE_EXPORT_PATH = Path("audit_logs") / "latest_evidence_package.json"


def approval_ledger_records(audit_records: list[dict]) -> list[dict]:
    action_names = {
        "alpaca_paper_order_armed": "order armed",
        "alpaca_paper_order_submitted": "order submitted",
        "alpaca_paper_order_blocked": "order blocked",
        "alpaca_paper_cancel_armed": "cancel armed",
        "alpaca_paper_cancel_submitted": "cancel submitted",
        "alpaca_paper_cancel_blocked": "cancel blocked",
        "alpaca_paper_exit_armed": "exit armed",
        "alpaca_paper_exit_submitted": "exit submitted",
        "alpaca_paper_exit_blocked": "exit blocked",
    }
    rows = []
    for record in audit_records:
        event_type = record.get("event_type", "")
        if event_type not in action_names:
            continue
        payload = record.get("payload", {}) or {}
        preview_hash = payload.get("preview_hash") or payload.get("cancel_preview_hash") or ""
        rows.append(
            {
                "Created": record.get("created_at", ""),
                "Action": action_names[event_type],
                "Symbol": payload.get("symbol", ""),
                "Side": payload.get("side", ""),
                "Quantity": payload.get("quantity", ""),
                "Preview Hash": preview_hash,
                "Broker Order ID": payload.get("broker_order_id", ""),
                "Manual Gate Required": _manual_gate_required(event_type),
                "Broker Write": event_type.endswith("_submitted"),
            }
        )
    return rows


def approval_ledger_summary_records(ledger_rows: list[dict]) -> list[dict]:
    return [
        {"Metric": "Approval Ledger Rows", "Value": len(ledger_rows)},
        {"Metric": "Armed Actions", "Value": sum("armed" in row.get("Action", "") for row in ledger_rows)},
        {"Metric": "Submitted Actions", "Value": sum(bool(row.get("Broker Write")) for row in ledger_rows)},
        {
            "Metric": "Rows With Preview Hash",
            "Value": sum(bool(row.get("Preview Hash")) for row in ledger_rows),
        },
    ]


def build_evidence_package(
    session_id: str,
    manifest: dict,
    audit_records: list[dict],
    approval_ledger: list[dict],
    tracked_orders: list[dict],
    automation_snapshots: list[dict],
    readiness_rows: list[dict],
    risk_halts: list[dict],
) -> dict:
    return {
        "exported_at": datetime.now(UTC).isoformat(),
        "session_id": session_id,
        "manifest": manifest,
        "audit_records": audit_records,
        "approval_ledger": approval_ledger,
        "tracked_alpaca_orders": tracked_orders,
        "automation_snapshots": automation_snapshots,
        "pre_live_readiness": readiness_rows,
        "risk_halts": risk_halts,
    }


def write_evidence_package(package: dict, path: str | Path | None = None) -> Path:
    output_path = Path(path) if path is not None else DEFAULT_EVIDENCE_EXPORT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(package, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def evidence_package_records(package: dict) -> list[dict]:
    return [
        {"Field": "Exported At", "Value": package.get("exported_at", "")},
        {"Field": "Session", "Value": package.get("session_id", "")},
        {"Field": "Audit Records", "Value": len(package.get("audit_records", []))},
        {"Field": "Approval Ledger Rows", "Value": len(package.get("approval_ledger", []))},
        {"Field": "Tracked Alpaca Orders", "Value": len(package.get("tracked_alpaca_orders", []))},
        {"Field": "Automation Snapshots", "Value": len(package.get("automation_snapshots", []))},
        {"Field": "Risk Halt Rows", "Value": len(package.get("risk_halts", []))},
    ]


def _manual_gate_required(event_type: str) -> bool:
    return event_type.startswith("alpaca_paper_")
