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
        {"Check": "Paper exit tested", "Required Before Live": True},
        {"Check": "Automation check reviewed", "Required Before Live": True},
        {"Check": "Paper performance reviewed", "Required Before Live": True},
        {"Check": "Practice mode reviewed", "Required Before Live": True},
        {"Check": "Manual live approval tested", "Required Before Unattended": True},
        {"Check": "Alpaca records refresh correctly", "Required Before Unattended": True},
        {"Check": "Activity logs are saved", "Required Before Unattended": True},
        {"Check": "Stop trading button tested", "Required Before Unattended": True},
        {"Check": "Agent cannot change risk or order code", "Required Before Live": True},
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
        ("Paper buy sent", paper_order_submitted, "A paper buy order reached Alpaca paper."),
        ("Paper cancel sent", paper_cancel_submitted, "A paper cancel request reached Alpaca paper."),
        ("Paper exit sent", paper_exit_tested, "A paper exit order reached Alpaca paper."),
        ("Paper fill matched", paper_fill_reconciled, "A filled paper order matched an Alpaca position."),
        ("Automation check saved", automation_dry_run_recorded, "An automation check was saved locally."),
        ("Paper performance reviewed", performance_reviewed, "Paper account performance was reviewed."),
        ("Stop trading tested", emergency_disable_tested, "The stop trading button was tested."),
        ("Live orders blocked", live_mode_blocked, "Live orders are still unavailable."),
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
        {"Check": "Live trading lock file", "Passed": True, "Detail": str(lock_path)},
        {
            "Check": "Live trading locked",
            "Passed": exists,
            "Detail": "Lock file exists; live trading stays locked." if exists else "Lock file is missing; live trading stays blocked.",
        },
        {"Check": "Live orders blocked", "Passed": True, "Detail": "Current code still blocks Alpaca live orders."},
    ]


def write_live_mode_lockfile(path: str | Path | None = None) -> Path:
    lock_path = Path(path) if path is not None else DEFAULT_LIVE_MODE_LOCKFILE_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        "\n".join(
            [
                "LIVE TRADING IS LOCKED",
                "Do not enable Alpaca live orders without reviewing the code, paper-trading results, and account settings.",
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
        ("Example settings file exists", env_example_present, ".env.example is available for non-secret setup shape."),
        ("Secrets ignored by git", dotenv_ignored, ".env is ignored and must not be committed."),
        ("Activity log file set", audit_path_configured, "Activity log file is set."),
        ("Alpaca order file set", broker_state_path_configured, "Saved Alpaca paper order file is set."),
        ("Records export file set", evidence_export_path_configured, "Records export file is set."),
        ("Live trading lock exists", live_lockfile_present, "Local live trading lock file exists."),
    ]
    if tests_passing is not None:
        rows.append(("Tests passing", tests_passing, "Latest local test checkpoint passed." if tests_passing else "Run tests before using live trading."))
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
            "Expected App Behavior": "The app shows Alpaca disconnected; paper buy, exit, and cancel buttons stay disabled.",
        },
        {
            "Scenario": "Old position data",
            "Expected App Behavior": "The app asks you to refresh Alpaca before sending an order.",
        },
        {
            "Scenario": "Old order data",
            "Expected App Behavior": "The app asks you to refresh Alpaca before sending an order.",
        },
        {
            "Scenario": "Saved order not found at Alpaca",
            "Expected App Behavior": "The app marks the saved order as missing until refresh finds it or you review it.",
        },
        {
            "Scenario": "Duplicate open order",
            "Expected App Behavior": "Buy or exit is blocked so you do not double-order the same symbol.",
        },
        {
            "Scenario": "Stop trading is on",
            "Expected App Behavior": "New orders are blocked.",
        },
    ]
