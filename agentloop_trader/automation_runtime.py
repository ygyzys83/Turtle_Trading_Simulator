from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from agentloop_trader.models import PACIFIC_TIME


DEFAULT_CONTROL_PATH = Path("automation_logs") / "automation_control.json"
DEFAULT_STATUS_PATH = Path("automation_logs") / "automation_worker_status.json"
DEFAULT_LOCK_PATH = Path("automation_logs") / "automation_worker.lock"


@dataclass(frozen=True)
class AutomationControl:
    enabled: bool = False
    stop_requested: bool = False
    mode: str = "Manual review only"
    paper_orders_enabled: bool = False
    kill_switch_enabled: bool = False
    full_automation_enabled: bool = False
    allow_duplicate_positions: bool = False
    allow_limit_buys_outside_market_hours: bool = False
    auto_cancel_limit_buys: bool = True
    stale_limit_order_minutes: int = 60
    refresh_seconds: int = 15
    symbol: str = "AAPL"
    price_data_source: str = "Ticker (Alpaca)"
    history: str = "1y"
    interval: str = "1h"
    strategy_settings: dict[str, Any] = field(default_factory=dict)
    risk_limits: dict[str, Any] = field(default_factory=dict)
    order_style: str = "Market"
    limit_adjustment_pct: float = 0.0
    custom_limit_price: float = 0.0
    account_size: float = 100000.0
    broker_state_path: str = "broker_state/alpaca_paper_orders.json"
    audit_log_path: str = "audit_logs/session_audit.jsonl"
    created_at: str = ""


@dataclass(frozen=True)
class WorkerStatus:
    running: bool = False
    pid: int = 0
    state: str = "Not running"
    last_checked_at: str = ""
    last_action: str = "None"
    last_error: str = ""
    loop_count: int = 0
    orders_sent: int = 0
    cancels_sent: int = 0
    exits_sent: int = 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


class AutomationControlStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else DEFAULT_CONTROL_PATH

    def read(self) -> AutomationControl:
        payload = _read_json(self.path)
        if not payload:
            return AutomationControl()
        allowed = {field.name for field in AutomationControl.__dataclass_fields__.values()}
        return AutomationControl(**{key: value for key, value in payload.items() if key in allowed})

    def write(self, control: AutomationControl) -> None:
        payload = asdict(control)
        if not payload.get("created_at"):
            payload["created_at"] = datetime.now(PACIFIC_TIME).isoformat()
        _write_json(self.path, payload)


class WorkerStatusStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else DEFAULT_STATUS_PATH

    def read(self) -> WorkerStatus:
        payload = _read_json(self.path)
        if not payload:
            return WorkerStatus()
        allowed = {field.name for field in WorkerStatus.__dataclass_fields__.values()}
        return WorkerStatus(**{key: value for key, value in payload.items() if key in allowed})

    def write(self, status: WorkerStatus) -> None:
        _write_json(self.path, asdict(status))


def worker_status_is_active(status: WorkerStatus, max_age_seconds: int = 120) -> bool:
    if not status.running:
        return False
    if not status.last_checked_at:
        return True
    try:
        checked = datetime.fromisoformat(status.last_checked_at)
    except ValueError:
        return True
    if checked.tzinfo is None:
        checked = checked.replace(tzinfo=PACIFIC_TIME)
    return datetime.now(PACIFIC_TIME) - checked.astimezone(PACIFIC_TIME) <= timedelta(seconds=max_age_seconds)


def start_worker_process(cwd: str | Path | None = None, python_executable: str | None = None) -> int:
    command = [python_executable or sys.executable, "-m", "agentloop_trader.worker"]
    kwargs: dict[str, Any] = {
        "cwd": str(cwd or Path.cwd()),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    process = subprocess.Popen(command, **kwargs)
    return int(process.pid)


class WorkerLock:
    def __init__(self, path: str | Path | None = None, stale_minutes: int = 10):
        self.path = Path(path) if path is not None else DEFAULT_LOCK_PATH
        self.stale_minutes = stale_minutes
        self._owned = False

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self._stale():
            try:
                self.path.unlink()
            except OSError:
                return False
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "created_at": datetime.now(PACIFIC_TIME).isoformat()}))
            self._owned = True
            return True
        except FileExistsError:
            return False

    def release(self) -> None:
        if not self._owned:
            return
        try:
            self.path.unlink()
        except OSError:
            pass
        self._owned = False

    def _stale(self) -> bool:
        try:
            age = datetime.now(PACIFIC_TIME) - datetime.fromtimestamp(self.path.stat().st_mtime, tz=PACIFIC_TIME)
        except OSError:
            return False
        return age > timedelta(minutes=self.stale_minutes)

    def __enter__(self) -> "WorkerLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def worker_status_records(status: WorkerStatus) -> list[dict[str, Any]]:
    return [
        {"Item": "Background worker", "Value": "Running" if status.running else "Not running"},
        {"Item": "State", "Value": str(status.state)},
        {"Item": "Last check", "Value": status.last_checked_at or "Not checked yet"},
        {"Item": "Last action", "Value": status.last_action or "None"},
        {"Item": "Orders sent", "Value": str(status.orders_sent)},
        {"Item": "Cancels sent", "Value": str(status.cancels_sent)},
        {"Item": "Exits sent", "Value": str(status.exits_sent)},
        {"Item": "Last error", "Value": status.last_error or ""},
    ]
