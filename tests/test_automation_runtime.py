from agentloop_trader.automation_runtime import (
    AutomationControl,
    AutomationControlStore,
    WorkerLock,
    WorkerStatus,
    WorkerStatusStore,
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
