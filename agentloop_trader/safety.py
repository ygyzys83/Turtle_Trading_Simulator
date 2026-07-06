from __future__ import annotations

from pathlib import Path


DEFAULT_LIVE_MODE_LOCKFILE_PATH = Path("live_mode") / "LIVE_TRADING_LOCKED.txt"


IMMUTABLE_AGENT_BOUNDARIES = (
    "risk_limits",
    "broker_credentials",
    "order_submission_code",
    "execution_mode",
    "kill_switch",
)


def production_readiness_checks() -> list[dict]:
    return [
        {"Check": "Paper trading tested", "Required Before Live": True},
        {"Check": "Paper exit path tested", "Required Before Live": True},
        {"Check": "Paper automation dry-run reviewed", "Required Before Live": True},
        {"Check": "Paper performance dashboard reviewed", "Required Before Live": True},
        {"Check": "Shadow mode reviewed", "Required Before Live": True},
        {"Check": "Manual live approval tested", "Required Before Unattended": True},
        {"Check": "Broker reconciliation implemented", "Required Before Unattended": True},
        {"Check": "Persistent audit logs implemented", "Required Before Unattended": True},
        {"Check": "Emergency disable tested", "Required Before Unattended": True},
        {"Check": "Agent cannot modify risk/execution boundaries", "Required Before Live": True},
    ]


def immutable_boundary_records() -> list[dict]:
    return [
        {"Boundary": boundary.replace("_", " ").title(), "Agent Modifiable": False}
        for boundary in IMMUTABLE_AGENT_BOUNDARIES
    ]


def pre_live_readiness_report(
    paper_order_submitted: bool,
    paper_cancel_submitted: bool,
    paper_exit_tested: bool,
    paper_fill_reconciled: bool,
    automation_dry_run_recorded: bool,
    performance_reviewed: bool,
    emergency_disable_tested: bool,
    live_mode_blocked: bool = True,
) -> list[dict]:
    checks = [
        ("Paper order submitted", paper_order_submitted, "Submit Alpaca Paper Order has reached Alpaca paper."),
        ("Paper cancel submitted", paper_cancel_submitted, "Cancel Alpaca Paper Order has reached Alpaca paper."),
        ("Paper exit tested", paper_exit_tested, "Submit Alpaca Paper Exit has been inspected and tested in paper."),
        ("Paper fill reconciled", paper_fill_reconciled, "Filled paper order matched to Alpaca position lifecycle."),
        ("Automation dry-run recorded", automation_dry_run_recorded, "Paper automation candidate queue has durable snapshots."),
        ("Performance dashboard reviewed", performance_reviewed, "Paper performance dashboard has been reviewed."),
        ("Emergency disable tested", emergency_disable_tested, "Emergency disable session behavior has been verified."),
        ("Live broker writes blocked", live_mode_blocked, "Live broker writes remain unavailable."),
    ]
    return [
        {
            "Check": name,
            "Passed": passed,
            "Status": "complete" if passed else "blocked",
            "Detail": detail,
        }
        for name, passed, detail in checks
    ]


def live_mode_lockfile_records(path: str | Path | None = None) -> list[dict]:
    lock_path = Path(path) if path is not None else DEFAULT_LIVE_MODE_LOCKFILE_PATH
    exists = lock_path.exists()
    return [
        {"Check": "Live Lockfile Path", "Passed": True, "Detail": str(lock_path)},
        {
            "Check": "Live Mode Locked",
            "Passed": exists,
            "Detail": "Lockfile exists; future live enablement remains administratively locked." if exists else "Lockfile is missing; live readiness remains blocked.",
        },
        {"Check": "Broker Writes Blocked", "Passed": True, "Detail": "Current code still blocks Alpaca live broker writes."},
    ]


def write_live_mode_lockfile(path: str | Path | None = None) -> Path:
    lock_path = Path(path) if path is not None else DEFAULT_LIVE_MODE_LOCKFILE_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        "\n".join(
            [
                "LIVE TRADING IS LOCKED",
                "Do not enable Alpaca live broker writes without manual code review, paper evidence, and user approval.",
                "This file is a local operational guardrail, not permission to trade live.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return lock_path


def deployment_readiness_records(
    env_example_present: bool,
    dotenv_ignored: bool,
    audit_path_configured: bool,
    broker_state_path_configured: bool,
    evidence_export_path_configured: bool,
    live_lockfile_present: bool,
    tests_passing: bool | None = None,
) -> list[dict]:
    rows = [
        ("Environment template present", env_example_present, ".env.example is available for non-secret configuration shape."),
        ("Secrets ignored by git", dotenv_ignored, ".env is ignored and must not be committed."),
        ("Audit path configured", audit_path_configured, "Durable audit JSONL path is configured."),
        ("Broker state path configured", broker_state_path_configured, "Tracked Alpaca paper order state path is configured."),
        ("Evidence export path configured", evidence_export_path_configured, "Evidence export path is configured."),
        ("Live lockfile present", live_lockfile_present, "Local live-mode lockfile exists."),
    ]
    if tests_passing is not None:
        rows.append(("Tests passing", tests_passing, "Latest local test checkpoint passed." if tests_passing else "Run tests before deployment."))
    return [
        {
            "Check": name,
            "Passed": passed,
            "Status": "ready" if passed else "blocked",
            "Detail": detail,
        }
        for name, passed, detail in rows
    ]


def broker_state_simulation_records() -> list[dict]:
    return [
        {
            "Scenario": "Alpaca disconnected",
            "Expected App Behavior": "Broker adapters show disconnected; Alpaca paper order/exit/cancel buttons remain disabled.",
        },
        {
            "Scenario": "Stale positions",
            "Expected App Behavior": "Previews warn that broker state is stale; submission remains blocked.",
        },
        {
            "Scenario": "Stale orders",
            "Expected App Behavior": "Duplicate-order and lifecycle checks require refresh before broker writes.",
        },
        {
            "Scenario": "Missing tracked order",
            "Expected App Behavior": "Lifecycle shows missing_from_alpaca_orders until per-order refresh resolves it.",
        },
        {
            "Scenario": "Duplicate open order",
            "Expected App Behavior": "Entry or exit submission is blocked by duplicate open order checks.",
        },
        {
            "Scenario": "Kill switch enabled",
            "Expected App Behavior": "Preflight and automation readiness show blocked state.",
        },
    ]
