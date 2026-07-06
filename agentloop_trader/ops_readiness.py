from __future__ import annotations

from pathlib import Path

from agentloop_trader.models import PreflightCheckResult, RiskCheckResult, RiskLimits, StrategyConfig, TradeIntent
from agentloop_trader.monitoring import MonitoringResult


def market_data_freshness_records(
    data_source: str,
    source_caption: str,
    row_count: int,
    latest_label: str,
    minimum_rows: int,
) -> list[dict]:
    enough_rows = row_count >= minimum_rows
    has_latest = bool(str(latest_label).strip())
    synthetic = data_source.strip().lower() == "synthetic"
    source_ok = synthetic or "yfinance" in source_caption.lower()
    return [
        {
            "Check": "Data Source Identified",
            "Passed": source_ok,
            "Detail": source_caption or "No source caption is available.",
        },
        {
            "Check": "Enough Bars Loaded",
            "Passed": enough_rows,
            "Detail": f"{row_count} bar(s) loaded; minimum for current rules is {minimum_rows}.",
        },
        {
            "Check": "Latest Bar Present",
            "Passed": has_latest,
            "Detail": f"Latest bar: {latest_label}" if has_latest else "No latest bar label is available.",
        },
        {
            "Check": "Freshness Policy",
            "Passed": True,
            "Detail": (
                "Synthetic data is deterministic test data."
                if synthetic
                else "Yahoo Finance data is treated as research/paper data and may be delayed."
            ),
        },
    ]


def strategy_state_snapshot_records(
    config: StrategyConfig,
    live: dict,
    intent: TradeIntent | None,
    risk_check: RiskCheckResult,
    preflight: PreflightCheckResult,
) -> list[dict]:
    return [
        {"Field": "Strategy", "Value": config.name},
        {"Field": "Entry Window", "Value": config.entry_window},
        {"Field": "Exit Window", "Value": config.exit_window},
        {"Field": "ATR Stop Multiple", "Value": config.atr_stop_multiplier},
        {"Field": "MA Trend Filter", "Value": config.moving_average_window},
        {"Field": "Signal", "Value": str(live.get("signal", "")).upper()},
        {"Field": "Reference Price", "Value": _money(live.get("last_p"))},
        {"Field": "Proposed Symbol", "Value": intent.symbol_clean if intent else ""},
        {"Field": "Proposed Quantity", "Value": intent.quantity if intent else 0},
        {"Field": "Risk Approved", "Value": risk_check.approved},
        {"Field": "Preflight Ready", "Value": preflight.ready},
    ]


def restart_recovery_records(
    audit_log_path: str,
    broker_state_path: str,
    automation_dry_run_path: str,
    run_manifest_path: str,
    audit_records_loaded: int,
    tracked_orders_loaded: int,
    automation_snapshots_loaded: int,
) -> list[dict]:
    configured_paths = {
        "Audit Log Path": audit_log_path,
        "Broker State Path": broker_state_path,
        "Automation Dry Run Path": automation_dry_run_path,
        "Run Manifest Path": run_manifest_path,
    }
    rows = []
    for check, raw_path in configured_paths.items():
        path = Path(raw_path)
        rows.append(
            {
                "Check": check,
                "Passed": bool(str(raw_path).strip()),
                "Detail": str(path) if str(raw_path).strip() else "Path is not configured.",
            }
        )
        rows.append(
            {
                "Check": f"{check} Parent Available",
                "Passed": bool(str(raw_path).strip()) and (path.parent == Path(".") or path.parent.exists()),
                "Detail": str(path.parent),
            }
        )
    rows.extend(
        [
            {"Check": "Audit Events Recoverable", "Passed": audit_records_loaded >= 0, "Detail": f"{audit_records_loaded} event(s) loaded."},
            {"Check": "Tracked Orders Recoverable", "Passed": tracked_orders_loaded >= 0, "Detail": f"{tracked_orders_loaded} tracked order(s) loaded."},
            {"Check": "Automation Snapshots Recoverable", "Passed": automation_snapshots_loaded >= 0, "Detail": f"{automation_snapshots_loaded} snapshot(s) loaded."},
        ]
    )
    return rows


def paper_account_health_records(
    paper_cash: float,
    paper_equity: float,
    starting_cash: float,
    local_open_positions: int,
    tracked_alpaca_orders: list[dict],
    monitoring_result: MonitoringResult,
    limits: RiskLimits,
) -> list[dict]:
    active_tracked_orders = [
        order
        for order in tracked_alpaca_orders
        if _enum_value(order.get("status", "")) in {"accepted", "new", "pending_new", "partially_filled", "filled"}
    ]
    return [
        {"Check": "Paper Cash Nonnegative", "Passed": paper_cash >= 0, "Detail": _money(paper_cash)},
        {"Check": "Paper Equity Positive", "Passed": paper_equity > 0, "Detail": _money(paper_equity)},
        {"Check": "Session P&L Available", "Passed": True, "Detail": _money(paper_equity - starting_cash)},
        {
            "Check": "Open Positions Within Limit",
            "Passed": local_open_positions <= limits.max_open_positions,
            "Detail": f"{local_open_positions} local open position(s); max {limits.max_open_positions}.",
        },
        {
            "Check": "Tracked Alpaca Activity Visible",
            "Passed": True,
            "Detail": f"{len(tracked_alpaca_orders)} tracked order(s); {len(active_tracked_orders)} active/filled.",
        },
        {
            "Check": "Monitoring Status",
            "Passed": monitoring_result.status != "BREACH",
            "Detail": f"{monitoring_result.status}: {'; '.join(monitoring_result.alerts)}",
        },
    ]


def scheduler_preview_records(
    interval_minutes: int,
    market_open: bool,
    kill_switch_enabled: bool,
    ready_candidate_count: int,
    halt_count: int,
) -> list[dict]:
    would_evaluate = not kill_switch_enabled and halt_count == 0
    would_queue_review = would_evaluate and ready_candidate_count > 0
    return [
        {"Field": "Scheduler Mode", "Value": "preview_only"},
        {"Field": "Interval Minutes", "Value": interval_minutes},
        {"Field": "Market Open", "Value": market_open},
        {"Field": "Would Evaluate Strategy", "Value": would_evaluate},
        {"Field": "Would Queue Manual Review", "Value": would_queue_review},
        {"Field": "Broker Writes Submitted", "Value": 0},
        {
            "Field": "Detail",
            "Value": (
                "A future scheduler would only evaluate and queue review in this build."
                if would_evaluate
                else "Scheduler preview is halted by the current session state."
            ),
        },
    ]


def paper_automation_gate_records(
    broker_connected: bool,
    broker_state_stale: bool,
    market_data_rows: list[dict],
    account_health_rows: list[dict],
    restart_rows: list[dict],
    readiness_rows: list[dict],
    halt_rows: list[dict],
    dry_run_snapshots_loaded: int,
) -> list[dict]:
    blockers: list[str] = []
    _collect_failed(blockers, market_data_rows, "Check")
    _collect_failed(blockers, account_health_rows, "Check")
    _collect_failed(blockers, restart_rows, "Check", required_checks={"Audit Log Path", "Broker State Path", "Automation Dry Run Path", "Run Manifest Path"})
    _collect_failed(blockers, readiness_rows, "Check")
    active_halts = [row for row in halt_rows if row.get("Active")]
    blockers.extend(str(row.get("Halt Reason", "")) for row in active_halts if row.get("Halt Reason"))
    if dry_run_snapshots_loaded <= 0:
        blockers.append("Record Paper Automation Dry Run before considering unattended paper automation.")
    if not broker_connected:
        blockers.append("Alpaca account is not connected.")
    if broker_state_stale:
        blockers.append("Alpaca broker state is stale.")
    blockers = list(dict.fromkeys(reason for reason in blockers if reason))
    return [
        {"Check": "Broker Connected", "Passed": broker_connected, "Detail": "Alpaca reads are connected." if broker_connected else "Connect Alpaca paper credentials."},
        {"Check": "Broker State Fresh", "Passed": not broker_state_stale, "Detail": "Current read is fresh." if not broker_state_stale else "Refresh Alpaca positions/orders."},
        {"Check": "Dry Run Evidence Present", "Passed": dry_run_snapshots_loaded > 0, "Detail": f"{dry_run_snapshots_loaded} dry-run snapshot(s) loaded."},
        {"Check": "Ready For Paper Automation", "Passed": not blockers, "Detail": "All local gates passed." if not blockers else "; ".join(blockers)},
    ]


def _collect_failed(blockers: list[str], rows: list[dict], label_key: str, required_checks: set[str] | None = None) -> None:
    for row in rows:
        label = str(row.get(label_key, ""))
        if required_checks is not None and label not in required_checks:
            continue
        if not row.get("Passed", True):
            detail = str(row.get("Detail", "")).strip()
            blockers.append(f"{label}: {detail}" if detail else label)


def _enum_value(value) -> str:
    text = str(value or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.strip().lower()


def _money(value) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return ""
