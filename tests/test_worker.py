from types import SimpleNamespace

from agentloop_trader.automation_runtime import AutomationControl, WorkerStatus
from agentloop_trader.worker import _send_exits, _stop_requested_during_wait, run_once


def test_worker_disabled_only_records_heartbeat():
    status = run_once(AutomationControl(enabled=False), WorkerStatus(loop_count=4))

    assert status.running is True
    assert status.state == "Watching only"
    assert status.loop_count == 5
    assert status.orders_sent == 0


class FakeLiveAdapter:
    config = SimpleNamespace(paper=False)


def test_worker_refuses_live_adapter():
    control = AutomationControl(enabled=True, paper_orders_enabled=True, mode="Auto exits only")

    status = run_once(control, adapter=FakeLiveAdapter())

    assert status.state == "Blocked"
    assert "paper-only" in status.last_error


def test_worker_recognizes_auto_exits_only_mode(monkeypatch, tmp_path):
    control = AutomationControl(
        enabled=True,
        paper_orders_enabled=True,
        mode="Auto exits only",
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )
    adapter = SimpleNamespace(config=SimpleNamespace(paper=True))
    monkeypatch.setattr("agentloop_trader.worker.market_session_advisory", lambda: {"Open": True})

    sent, tracked, message = _send_exits(
        control,
        adapter,
        positions=[],
        orders=[],
        tracked_orders=[],
        fetch_bars=lambda symbol, history, interval, source: None,
        audit_store=SimpleNamespace(append=lambda event: None),
    )

    assert sent == 0
    assert tracked == []
    assert message == "No auto exits were ready."


def test_worker_stop_wait_checks_control_each_second(monkeypatch):
    checks = []

    class Store:
        def read(self):
            checks.append(1)
            return SimpleNamespace(stop_requested=len(checks) >= 2)

    monkeypatch.setattr("agentloop_trader.worker.time.sleep", lambda seconds: None)

    assert _stop_requested_during_wait(Store(), 15) is True
    assert len(checks) == 2
