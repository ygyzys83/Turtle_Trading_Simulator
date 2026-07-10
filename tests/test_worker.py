from types import SimpleNamespace

from agentloop_trader.automation_runtime import AutomationControl, WorkerStatus
from agentloop_trader.worker import run_once


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
