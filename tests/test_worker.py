from types import SimpleNamespace

from agentloop_trader.automation_runtime import AutomationControl, WorkerStatus
from agentloop_trader.buy_watchlist import BuyWatchPlan, BuyWatchlistStore, buy_watch_plan_id
from agentloop_trader.brokers import AlpacaConfig
from agentloop_trader.models import TradeIntent
from agentloop_trader.worker import _BAR_CACHE, _fetcher, _open_buy_order_notional, _send_entry, _send_exits, _send_watchlist_entries, _stop_requested_during_wait, run_once


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


def test_worker_never_falls_back_to_displayed_ticker_when_buy_watchlist_is_empty(monkeypatch, tmp_path):
    control = AutomationControl(
        enabled=True,
        mode="Auto entries and exits",
        full_automation_enabled=True,
        paper_orders_enabled=True,
        symbol="TSLA",
        audit_log_path=str(tmp_path / "audit.jsonl"),
        broker_state_path=str(tmp_path / "broker.json"),
        buy_watchlist_path=str(tmp_path / "watchlist.json"),
    )
    adapter = SimpleNamespace(
        config=AlpacaConfig(api_key="key", api_secret="secret", paper=True),
        position_records=lambda **kwargs: [],
        order_records=lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "agentloop_trader.worker._send_entry",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Displayed ticker must never be auto-bought")),
    )

    status = run_once(control, adapter=adapter)

    assert status.state == "Watching"
    assert status.orders_sent == 0
    assert "Buy watchlist is empty" in status.last_action


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
        config=AlpacaConfig(api_key="key", api_secret="secret", paper=True),
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


def test_worker_updates_watchlist_status_when_buy_rules_are_waiting(monkeypatch, tmp_path):
    store = BuyWatchlistStore(tmp_path / "watchlist.json")
    plan = BuyWatchPlan(
        plan_id=buy_watch_plan_id("TSLA", "4h", "Breakout continuation"),
        symbol="TSLA",
        interval="4h",
        history="5y",
        price_data_source="Ticker (Alpaca)",
        strategy_label="Breakout continuation",
    )
    store.upsert(plan)
    monkeypatch.setattr(
        "agentloop_trader.worker._send_entry",
        lambda *args, **kwargs: (0, args[4], "No BUY setup right now."),
    )

    sent, _, _ = _send_watchlist_entries(
        AutomationControl(mode="Auto entries and exits", full_automation_enabled=True),
        SimpleNamespace(),
        [],
        [],
        [],
        lambda *_: None,
        SimpleNamespace(append=lambda event: None),
        store,
        latest_prices={"TSLA": 250.0},
        max_to_send=1,
    )

    updated = store.read()[0]
    assert sent == 0
    assert updated.enabled is True
    assert updated.status == "Waiting for BUY"


def test_worker_disables_watchlist_setup_after_order_is_sent(monkeypatch, tmp_path):
    store = BuyWatchlistStore(tmp_path / "watchlist.json")
    plan = BuyWatchPlan(
        plan_id=buy_watch_plan_id("TSLA", "4h", "Trend pullback continuation"),
        symbol="TSLA",
        interval="4h",
        history="5y",
        price_data_source="Ticker (Alpaca)",
        strategy_label="Trend pullback continuation",
    )
    store.upsert(plan)
    monkeypatch.setattr(
        "agentloop_trader.worker._send_entry",
        lambda *args, **kwargs: (1, args[4], "Sent paper buy for 5 TSLA."),
    )
    adapter = SimpleNamespace(position_records=lambda **kwargs: [], order_records=lambda **kwargs: [])

    sent, _, _ = _send_watchlist_entries(
        AutomationControl(mode="Auto entries and exits", full_automation_enabled=True),
        adapter,
        [],
        [],
        [],
        lambda *_: None,
        SimpleNamespace(append=lambda event: None),
        store,
        latest_prices={"TSLA": 250.0},
        max_to_send=1,
    )

    updated = store.read()[0]
    assert sent == 1
    assert updated.enabled is False
    assert updated.status == "Order sent"


def test_worker_keeps_repeating_setup_enabled_after_order_is_sent(monkeypatch, tmp_path):
    store = BuyWatchlistStore(tmp_path / "watchlist.json")
    plan = BuyWatchPlan(
        plan_id=buy_watch_plan_id("TSLA", "4h", "Trend pullback continuation"),
        symbol="TSLA",
        interval="4h",
        history="5y",
        price_data_source="Ticker (Alpaca)",
        strategy_label="Trend pullback continuation",
        repeat_after_exit=True,
    )
    store.upsert(plan)
    monkeypatch.setattr(
        "agentloop_trader.worker._send_entry",
        lambda *args, **kwargs: (1, args[4], "Sent paper buy for 5 TSLA."),
    )
    adapter = SimpleNamespace(position_records=lambda **kwargs: [], order_records=lambda **kwargs: [])

    sent, _, _ = _send_watchlist_entries(
        AutomationControl(mode="Auto entries and exits", full_automation_enabled=True),
        adapter,
        [],
        [],
        [],
        lambda *_: None,
        SimpleNamespace(append=lambda event: None),
        store,
        latest_prices={"TSLA": 250.0},
        max_to_send=1,
    )

    updated = store.read()[0]
    assert sent == 1
    assert updated.enabled is True
    assert updated.repeat_after_exit is True
    assert updated.cycle_state == "order_pending"
    assert updated.status == "Buy order sent"


def test_repeating_setup_waits_for_prior_buy_signal_to_clear(monkeypatch, tmp_path):
    store = BuyWatchlistStore(tmp_path / "watchlist.json")
    plan = BuyWatchPlan(
        plan_id=buy_watch_plan_id("TSLA", "4h", "Trend pullback continuation"),
        symbol="TSLA",
        interval="4h",
        history="5y",
        price_data_source="Ticker (Alpaca)",
        strategy_label="Trend pullback continuation",
        repeat_after_exit=True,
        cycle_state="position_open",
    )
    store.upsert(plan)
    control = AutomationControl(mode="Auto entries and exits", full_automation_enabled=True)
    adapter = SimpleNamespace()
    common_args = (
        control,
        adapter,
        [],
        [],
        [],
        lambda *_: None,
        SimpleNamespace(append=lambda event: None),
        store,
    )

    _send_watchlist_entries(*common_args, latest_prices={"TSLA": 250.0}, max_to_send=1)
    assert store.read()[0].cycle_state == "waiting_for_signal_reset"

    monkeypatch.setattr("agentloop_trader.worker._repeat_signal_state", lambda *args: (True, ""))
    _send_watchlist_entries(*common_args, latest_prices={"TSLA": 250.0}, max_to_send=1)
    assert store.read()[0].cycle_state == "waiting_for_signal_reset"
    assert store.read()[0].status == "Waiting for a new BUY"

    monkeypatch.setattr("agentloop_trader.worker._repeat_signal_state", lambda *args: (False, ""))
    _send_watchlist_entries(*common_args, latest_prices={"TSLA": 250.0}, max_to_send=1)
    assert store.read()[0].cycle_state == "waiting_for_buy"
    assert store.read()[0].enabled is True


def test_worker_reuses_recent_price_history_for_queued_setups(monkeypatch):
    calls = []
    data = SimpleNamespace(copy=lambda: "copied-bars")
    monkeypatch.setattr(
        "agentloop_trader.worker.fetch_price_bars",
        lambda *args: calls.append(args) or data,
    )
    _BAR_CACHE.clear()
    fetch = _fetcher(AutomationControl(), AlpacaConfig(api_key="key", api_secret="secret", paper=True))

    first = fetch("TSLA", "2y", "1h", "Ticker (Alpaca)")
    second = fetch("TSLA", "2y", "1h", "Ticker (Alpaca)")

    assert first == "copied-bars"
    assert second == "copied-bars"
    assert len(calls) == 1


def test_worker_reprices_queued_buy_with_latest_trade(monkeypatch):
    captured = {}
    control = AutomationControl(
        mode="Auto entries and exits",
        full_automation_enabled=True,
        paper_orders_enabled=True,
        symbol="TSLA",
    )
    adapter = SimpleNamespace(
        config=AlpacaConfig(api_key="key", api_secret="secret", paper=True),
        account_records=lambda: [
            {"Field": "Portfolio Value", "Value": "100000"},
            {"Field": "Cash", "Value": "100000"},
            {"Field": "Last Equity", "Value": "100000"},
        ],
    )
    intent = TradeIntent(symbol="TSLA", side="buy", quantity=1, entry_price=200, stop_loss=190)
    monkeypatch.setattr("agentloop_trader.worker.market_session_advisory", lambda: {"Open": True})
    monkeypatch.setattr(
        "agentloop_trader.worker.selected_strategy_result",
        lambda *args, **kwargs: {"live": {"trade_intent": intent}},
    )
    monkeypatch.setattr(
        "agentloop_trader.worker.reprice_trade_intent",
        lambda trade_intent, price: captured.update(price=price) or trade_intent,
    )
    monkeypatch.setattr("agentloop_trader.worker.constrain_trade_intent_to_limits", lambda *args, **kwargs: None)

    _send_entry(
        control,
        adapter,
        [],
        [],
        [],
        lambda *_: SimpleNamespace(attrs={"latest_price": 201}),
        SimpleNamespace(append=lambda event: None),
        latest_price=225.50,
        require_latest_price=True,
    )

    assert captured["price"] == 225.50
