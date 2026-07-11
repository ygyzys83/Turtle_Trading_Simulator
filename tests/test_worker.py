from types import SimpleNamespace

from agentloop_trader.automation_runtime import AutomationControl, WorkerStatus
from agentloop_trader.brokers import AlpacaConfig
from agentloop_trader.models import TradeIntent
from agentloop_trader.worker import _open_buy_order_notional, _send_entry, _send_exits, _stop_requested_during_wait, run_once


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


def test_worker_sizes_strategy_from_alpaca_equity_not_saved_simulator_value(monkeypatch):
    captured = {}
    control = AutomationControl(
        mode="Auto entries and exits", full_automation_enabled=True,
        paper_orders_enabled=True, account_size=50_000, symbol="AAPL",
    )
    adapter = SimpleNamespace(
        config=SimpleNamespace(paper=True),
        account_records=lambda: [
            {"Field": "Portfolio Value", "Value": "100000"},
            {"Field": "Cash", "Value": "90000"},
            {"Field": "Last Equity", "Value": "99500"},
        ],
    )
    monkeypatch.setattr("agentloop_trader.worker.market_session_advisory", lambda: {"Open": True})
    def selected_result(data, settings, account, limits):
        captured["account"] = account
        return {"live": {"trade_intent": None}}

    monkeypatch.setattr("agentloop_trader.worker.selected_strategy_result", selected_result)

    _send_entry(
        control, adapter, [], [], [], lambda *_: object(),
        SimpleNamespace(append=lambda event: None),
    )

    assert captured["account"] == 100_000


def test_worker_blocks_new_entry_after_alpaca_daily_loss_limit(monkeypatch):
    control = AutomationControl(
        mode="Auto entries and exits", full_automation_enabled=True,
        paper_orders_enabled=True, symbol="AAPL",
        risk_limits={"max_session_loss_pct": 2.0},
    )
    adapter = SimpleNamespace(
        config=AlpacaConfig(api_key="key", api_secret="secret", paper=True),
        account_records=lambda: [
            {"Field": "Portfolio Value", "Value": "97000"},
            {"Field": "Cash", "Value": "90000"},
            {"Field": "Last Equity", "Value": "100000"},
        ],
        submit_order=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Order should not be sent")),
    )
    monkeypatch.setattr("agentloop_trader.worker.market_session_advisory", lambda: {"Open": True})
    monkeypatch.setattr(
        "agentloop_trader.worker.selected_strategy_result",
        lambda *args, **kwargs: {
            "live": {"trade_intent": TradeIntent(symbol="AAPL", side="buy", quantity=10, entry_price=100, stop_loss=95)}
        },
    )

    sent, _, message = _send_entry(
        control, adapter, [], [], [], lambda *_: object(),
        SimpleNamespace(append=lambda event: None),
    )

    assert sent == 0
    assert "Daily loss" in message


def test_worker_counts_only_unfilled_open_buy_order_notional():
    orders = [
        {"Symbol": "AAPL", "Side": "buy", "Status": "partially_filled", "Quantity": "10", "Filled Qty": "4", "Limit Price": "100"},
        {"Symbol": "MSFT", "Side": "buy", "Status": "accepted", "Quantity": "5", "Filled Qty": "0", "Limit Price": "200"},
        {"Symbol": "AAPL", "Side": "buy", "Status": "canceled", "Quantity": "20", "Limit Price": "100"},
    ]

    assert _open_buy_order_notional(orders) == 1600
    assert _open_buy_order_notional(orders, "AAPL") == 600
