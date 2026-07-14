import json

from agentloop_trader.automation_runtime import (
    AutomationControl,
    AutomationControlStore,
    WorkerLock,
    WorkerStatus,
    WorkerStatusStore,
    automation_mode_for_new_ui_session,
    request_worker_stop,
    start_worker_process,
    worker_status_is_active,
    worker_status_records,
)


def test_control_store_roundtrips_settings(tmp_path):
    path = tmp_path / "control.json"
    store = AutomationControlStore(path)
    control = AutomationControl(
        enabled=True,
        stop_requested=True,
        mode="Auto exits only",
        symbol="AAPL",
        refresh_seconds=30,
        buy_watchlist_path="custom-watchlist.json",
    )

    store.write(control)
    loaded = store.read()

    assert loaded.enabled is True
    assert loaded.mode == "Auto exits only"
    assert loaded.stop_requested is True
    assert loaded.symbol == "AAPL"
    assert loaded.refresh_seconds == 30
    assert loaded.buy_watchlist_path == "custom-watchlist.json"


def test_status_store_roundtrips_heartbeat(tmp_path):
    path = tmp_path / "status.json"
    store = WorkerStatusStore(path)

    store.write(WorkerStatus(running=True, state="Watching", loop_count=2))

    assert store.read().running is True
    assert store.read().loop_count == 2
    assert worker_status_records(store.read())[0]["Value"] == "Running"


def test_worker_status_active_uses_recent_heartbeat():
    from datetime import datetime, timedelta
    from agentloop_trader.models import PACIFIC_TIME

    recent = WorkerStatus(running=True, last_checked_at=datetime.now(PACIFIC_TIME).isoformat())
    stale = WorkerStatus(running=True, last_checked_at=(datetime.now(PACIFIC_TIME) - timedelta(minutes=10)).isoformat())

    assert worker_status_is_active(recent, max_age_seconds=120) is True
    assert worker_status_is_active(stale, max_age_seconds=120) is False


def test_fresh_ui_restores_mode_only_when_a_worker_record_is_present():
    saved = AutomationControl(mode="Auto entries and exits", full_automation_enabled=True)

    assert automation_mode_for_new_ui_session(saved, worker_present=True) == "Auto entries and exits"
    assert automation_mode_for_new_ui_session(saved, worker_present=False) == "Manual review only"
    assert automation_mode_for_new_ui_session(AutomationControl(mode="unknown"), worker_present=True) == "Manual review only"


def test_start_worker_process_launches_worker_module(monkeypatch, tmp_path):
    calls = {}

    class FakeProcess:
        pid = 1234

    def fake_popen(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr("agentloop_trader.automation_runtime.subprocess.Popen", fake_popen)

    pid = start_worker_process(cwd=tmp_path, python_executable="python-test")

    assert pid == 1234
    assert calls["command"] == ["python-test", "-m", "agentloop_trader.worker"]
    assert calls["kwargs"]["cwd"] == str(tmp_path)


def test_worker_lock_allows_one_owner(tmp_path):
    path = tmp_path / "worker.lock"
    first = WorkerLock(path)
    second = WorkerLock(path)

    assert first.acquire() is True
    assert second.acquire() is False
    first.release()
    assert second.acquire() is True
    second.release()


def test_request_worker_stop_persists_stop_when_worker_is_already_stopped(tmp_path):
    control_store = AutomationControlStore(tmp_path / "control.json")
    status_store = WorkerStatusStore(tmp_path / "status.json")
    control_store.write(AutomationControl(enabled=True, stop_requested=False))
    status_store.write(WorkerStatus(running=False, state="Stopped"))

    status = request_worker_stop(
        control_store,
        status_store,
        lock_path=tmp_path / "worker.lock",
        timeout_seconds=0,
    )

    assert control_store.read().enabled is False
    assert control_store.read().stop_requested is True
    assert status.running is False
    assert status.state == "Stopped"


def test_request_worker_stop_force_stops_only_verified_worker_pid(monkeypatch, tmp_path):
    control_store = AutomationControlStore(tmp_path / "control.json")
    status_store = WorkerStatusStore(tmp_path / "status.json")
    lock_path = tmp_path / "worker.lock"
    lock_path.write_text(json.dumps({"pid": 1234}), encoding="utf-8")
    control_store.write(AutomationControl(enabled=True))
    status_store.write(WorkerStatus(running=True, pid=1234, state="Watching"))
    alive = {"value": True}
    terminated = []

    monkeypatch.setattr("agentloop_trader.automation_runtime._process_exists", lambda pid: alive["value"])

    def terminate(pid):
        terminated.append(pid)
        alive["value"] = False

    monkeypatch.setattr("agentloop_trader.automation_runtime._terminate_worker_process", terminate)

    status = request_worker_stop(
        control_store,
        status_store,
        lock_path=lock_path,
        timeout_seconds=0,
    )

    assert terminated == [1234]
    assert status.running is False
    assert status.state == "Stopped"
    assert not lock_path.exists()
