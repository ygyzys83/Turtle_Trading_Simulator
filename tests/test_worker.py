from types import SimpleNamespace

from agentloop_trader.automation_runtime import AutomationControl, WorkerStatus
from agentloop_trader.buy_watchlist import BuyWatchPlan, BuyWatchlistStore, buy_watch_plan_id
from agentloop_trader.brokers import AlpacaConfig
from agentloop_trader.models import TradeIntent
from agentloop_trader.position_lifecycle import resolve_position_plan
from agentloop_trader.worker import _BAR_CACHE, _cancel_late_rsi_limit_buys, _fetcher, _open_buy_order_notional, _send_entry, _send_exits, _send_watchlist_entries, _stop_requested_during_wait, run_once, sleep_resume_detected


def test_worker_disabled_only_records_heartbeat():
    status = run_once(AutomationControl(enabled=False), WorkerStatus(loop_count=4))

    assert status.running is True
    assert status.state == "Watching only"
    assert status.loop_count == 5
    assert status.orders_sent == 0


def test_worker_detects_resume_after_system_sleep():
    assert sleep_resume_detected(15.0, 15.0) is False
    assert sleep_resume_detected(24.9, 15.0) is False
    assert sleep_resume_detected(25.1, 15.0) is True


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


def test_worker_does_not_sell_adopted_manual_position_without_saved_exit_plan(monkeypatch, tmp_path):
    control = AutomationControl(
        enabled=True,
        paper_orders_enabled=True,
        mode="Auto entries and exits",
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )
    adapter = SimpleNamespace(
        config=SimpleNamespace(paper=True),
        submit_order=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("A manual position without saved exit settings must not be sold")
        ),
    )
    tracked = [
        {
            "symbol": "CRWV",
            "side": "buy",
            "status": "filled",
            "exit_settings": {"auto_exit_enabled": True, "interval": "4h"},
        },
        {
            "symbol": "CRWV",
            "side": "buy",
            "status": "filled",
            "source": "adopted_alpaca_position",
            "adopted": True,
        },
    ]
    monkeypatch.setattr("agentloop_trader.worker._market_is_open", lambda adapter: True)

    sent, updated, message = _send_exits(
        control,
        adapter,
        positions=[{"Symbol": "CRWV", "Quantity": "64", "Average Entry": "77.08"}],
        orders=[],
        tracked_orders=tracked,
        fetch_bars=lambda *args: (_ for _ in ()).throw(AssertionError("No bars should be loaded")),
        audit_store=SimpleNamespace(append=lambda event: None),
    )

    assert sent == 0
    assert updated == tracked
    assert message == "No auto exits were ready."


def test_worker_exit_uses_current_reentry_order_not_prior_symbol_cycle(monkeypatch, tmp_path):
    captured = {}
    control = AutomationControl(enabled=True, paper_orders_enabled=True, mode="Auto exits only")
    adapter = SimpleNamespace(config=SimpleNamespace(paper=True))
    orders = [
        {"Alpaca Order ID": "old-buy", "Symbol": "NVDA", "Side": "buy", "Status": "filled", "Filled Qty": 24, "Avg Fill": 202.48, "Filled": "2026-07-09T16:33:04+00:00"},
        {"Alpaca Order ID": "old-sell", "Symbol": "NVDA", "Side": "sell", "Status": "filled", "Filled Qty": 24, "Avg Fill": 210.15, "Filled": "2026-07-10T19:00:59+00:00"},
        {"Alpaca Order ID": "new-buy", "Symbol": "NVDA", "Side": "buy", "Status": "filled", "Filled Qty": 23, "Avg Fill": 210.758696, "Filled": "2026-07-15T19:20:40+00:00"},
    ]
    tracked = [
        {"broker_order_id": "old-buy", "symbol": "NVDA", "side": "buy", "status": "filled", "exit_settings": {"entry_stop_distance": 2.0, "highest_high_since_entry": 213.775}},
        {"broker_order_id": "new-buy", "symbol": "NVDA", "side": "buy", "status": "new", "exit_settings": {"entry_stop_distance": 6.78, "auto_exit_enabled": True}},
    ]

    def evaluate(settings, position, fetch_bars):
        captured.update(settings)
        return {"ready": False, "state_changed": False}

    monkeypatch.setattr("agentloop_trader.worker._market_is_open", lambda adapter: True)
    monkeypatch.setattr("agentloop_trader.worker.evaluate_exit_settings", evaluate)

    sent, _, _ = _send_exits(
        control,
        adapter,
        [{"Symbol": "NVDA", "Quantity": 23, "Average Entry": 210.758696}],
        orders,
        tracked,
        lambda *_: None,
        SimpleNamespace(append=lambda event: None),
    )

    assert sent == 0
    assert captured["entry_broker_order_id"] == "new-buy"
    assert captured["entry_reference_price"] == 210.758696
    assert captured["entry_stop_distance"] == 6.78
    assert captured["highest_high_since_entry"] == 210.758696


def test_worker_persists_latest_exit_calculation_for_fast_ui_startup(monkeypatch):
    control = AutomationControl(enabled=True, paper_orders_enabled=True, mode="Auto exits only")
    adapter = SimpleNamespace(config=SimpleNamespace(paper=True))
    orders = [{
        "Alpaca Order ID": "buy-1",
        "Symbol": "IBM",
        "Side": "buy",
        "Status": "filled",
        "Filled Qty": 10,
        "Avg Fill": 218.50,
        "Filled": "2026-07-16T15:00:00+00:00",
    }]
    tracked = [{
        "broker_order_id": "buy-1",
        "symbol": "IBM",
        "side": "buy",
        "status": "filled",
        "exit_settings": {
            "entry_stop_distance": 5.0,
            "auto_exit_enabled": True,
        },
    }]
    monkeypatch.setattr("agentloop_trader.worker._market_is_open", lambda adapter: True)
    monkeypatch.setattr(
        "agentloop_trader.worker.evaluate_exit_settings",
        lambda *args: {
            "ready": False,
            "state_changed": False,
            "reason": "Hold.",
            "current_price": 219.25,
            "current_price_source": "Alpaca position market value",
            "trigger_price": 213.50,
            "trigger_source": "fill-adjusted initial stop",
            "original_stop_price": 213.50,
            "profit_r": 0.15,
            "checked_at": "2026-07-16T12:00:00-07:00",
        },
    )

    sent, updated, _ = _send_exits(
        control,
        adapter,
        [{"Symbol": "IBM", "Quantity": 10, "Market Value": 2192.50, "Average Entry": 218.50}],
        orders,
        tracked,
        lambda *_: None,
        SimpleNamespace(append=lambda event: None),
    )

    resolution = resolve_position_plan(
        {"Symbol": "IBM", "Quantity": 10, "Market Value": 2192.50, "Average Entry": 218.50},
        orders,
        updated,
    )
    snapshot = resolution.exit_settings["last_exit_snapshot"]
    assert sent == 0
    assert snapshot["current_price"] == 219.25
    assert snapshot["trigger_price"] == 213.50
    assert snapshot["checked_at"] == "2026-07-16T12:00:00-07:00"


def test_worker_does_not_attach_old_settings_to_new_manual_reentry(monkeypatch):
    control = AutomationControl(enabled=True, paper_orders_enabled=True, mode="Auto exits only")
    adapter = SimpleNamespace(
        config=SimpleNamespace(paper=True),
        submit_order=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Manual reentry must remain unmanaged")),
    )
    orders = [
        {"Alpaca Order ID": "old-buy", "Symbol": "IBM", "Side": "buy", "Status": "filled", "Filled Qty": 10, "Filled": "2026-07-01T15:00:00+00:00"},
        {"Alpaca Order ID": "old-sell", "Symbol": "IBM", "Side": "sell", "Status": "filled", "Filled Qty": 10, "Filled": "2026-07-02T15:00:00+00:00"},
        {"Alpaca Order ID": "manual-buy", "Symbol": "IBM", "Side": "buy", "Status": "filled", "Filled Qty": 5, "Filled": "2026-07-15T15:00:00+00:00"},
    ]
    tracked = [{
        "broker_order_id": "old-buy",
        "symbol": "IBM",
        "side": "buy",
        "status": "filled",
        "exit_settings": {"entry_stop_distance": 4.0, "auto_exit_enabled": True},
    }]
    monkeypatch.setattr("agentloop_trader.worker._market_is_open", lambda adapter: True)

    sent, _, _ = _send_exits(
        control,
        adapter,
        [{"Symbol": "IBM", "Quantity": 5, "Average Entry": 220.0}],
        orders,
        tracked,
        lambda *_: (_ for _ in ()).throw(AssertionError("Unmanaged position must not fetch bars")),
        SimpleNamespace(append=lambda event: None),
    )

    assert sent == 0


def test_worker_exit_audit_identifies_exact_position_cycle(monkeypatch):
    events = []
    control = AutomationControl(enabled=True, paper_orders_enabled=True, mode="Auto exits only")
    adapter = SimpleNamespace(
        config=SimpleNamespace(paper=True),
        submit_order=lambda *args, **kwargs: SimpleNamespace(id="sell-1"),
    )
    orders = [{
        "Alpaca Order ID": "buy-1",
        "Symbol": "NVDA",
        "Side": "buy",
        "Status": "filled",
        "Filled Qty": 23,
        "Avg Fill": 210.758696,
        "Filled": "2026-07-15T19:20:40+00:00",
    }]
    tracked = [{
        "broker_order_id": "buy-1",
        "symbol": "NVDA",
        "side": "buy",
        "status": "filled",
        "exit_settings": {"entry_stop_distance": 6.78, "auto_exit_enabled": True},
    }]
    monkeypatch.setattr("agentloop_trader.worker._market_is_open", lambda adapter: True)
    monkeypatch.setattr(
        "agentloop_trader.worker.evaluate_exit_settings",
        lambda *args: {
            "ready": True,
            "state_changed": False,
            "reason": "Exit trigger reached.",
            "current_price": 203.90,
            "trigger_price": 203.98,
            "trigger_source": "fill-adjusted initial stop",
            "profit_r": -1.01,
            "highest_profit_r": 0.09,
        },
    )
    monkeypatch.setattr(
        "agentloop_trader.worker.build_alpaca_order_preview",
        lambda *args: SimpleNamespace(valid=True, preview_hash="exit-preview"),
    )
    monkeypatch.setattr("agentloop_trader.worker.open_exit_order_reasons", lambda *args: [])
    monkeypatch.setattr(
        "agentloop_trader.worker._track_broker_order",
        lambda *args: {"broker_order_id": "sell-1", "symbol": "NVDA", "side": "sell"},
    )

    sent, updated, _ = _send_exits(
        control,
        adapter,
        [{"Symbol": "NVDA", "Quantity": 23, "Average Entry": 210.758696}],
        orders,
        tracked,
        lambda *_: None,
        SimpleNamespace(append=events.append),
    )

    assert sent == 1
    assert updated[-1]["parent_position_cycle_id"] == "buy-1"
    assert events[0].payload["position_cycle_id"] == "buy-1"
    assert events[0].payload["position_average_entry"] == 210.758696
    assert events[0].payload["entry_stop_distance"] == 6.78
    assert events[0].payload["exit_details"]["highest_profit_r"] == 0.09


def test_worker_reports_triggered_exit_that_is_blocked_by_existing_sell_order(monkeypatch):
    events = []
    control = AutomationControl(enabled=True, paper_orders_enabled=True, mode="Auto exits only")
    adapter = SimpleNamespace(
        config=SimpleNamespace(paper=True),
        submit_order=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("A blocked exit must not be submitted")
        ),
    )
    orders = [{
        "Alpaca Order ID": "buy-1",
        "Symbol": "CRWV",
        "Side": "buy",
        "Status": "filled",
        "Filled Qty": 64,
        "Avg Fill": 77.08,
        "Filled": "2026-07-15T19:05:42+00:00",
    }]
    tracked = [{
        "broker_order_id": "buy-1",
        "symbol": "CRWV",
        "side": "buy",
        "status": "filled",
        "exit_settings": {"entry_stop_distance": 1.84, "auto_exit_enabled": True},
    }]
    monkeypatch.setattr("agentloop_trader.worker._market_is_open", lambda adapter: True)
    monkeypatch.setattr(
        "agentloop_trader.worker.evaluate_exit_settings",
        lambda *args: {
            "ready": True,
            "state_changed": False,
            "reason": "CRWV reached its stop.",
            "current_price": 73.20,
            "current_price_source": "Alpaca position market value",
            "trigger_price": 75.24,
            "trigger_source": "fill-adjusted initial stop",
        },
    )
    monkeypatch.setattr(
        "agentloop_trader.worker.build_alpaca_order_preview",
        lambda *args: SimpleNamespace(valid=True, preview_hash="exit-preview", blocked_reasons=[]),
    )
    monkeypatch.setattr(
        "agentloop_trader.worker.open_exit_order_reasons",
        lambda *args: ["Alpaca Orders already has an open CRWV sell order with status accepted."],
    )

    sent, _, message = _send_exits(
        control,
        adapter,
        [{"Symbol": "CRWV", "Quantity": 64, "Average Entry": 77.08}],
        orders,
        tracked,
        lambda *_: None,
        SimpleNamespace(append=events.append),
    )

    assert sent == 0
    assert "CRWV exit triggered but was not sent" in message
    assert "open CRWV sell order" in message
    assert events[0].event_type == "worker_paper_exit_blocked"
    assert events[0].payload["current_price_source"] == "Alpaca position market value"


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
            {"Field": "Portfolio Value", "Value": "98000"},
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


def test_worker_cancels_unfilled_rsi_buy_after_maximum_rebound(monkeypatch):
    canceled = []
    events = []
    adapter = SimpleNamespace(
        config=AlpacaConfig(api_key="key", api_secret="secret", paper=True),
        cancel_order=lambda order_id, expected_cancel_hash: canceled.append((order_id, expected_cancel_hash)),
    )
    orders = [{
        "Alpaca Order ID": "rsi-order-1",
        "Symbol": "CAG",
        "Side": "buy",
        "Status": "accepted",
        "Order Type": "limit",
    }]
    tracked = [{
        "broker_order_id": "rsi-order-1",
        "strategy_settings": {
            "symbol": "CAG",
            "strategy_type": "rsi_scalp",
            "entry_rsi_setup_low": 13.5,
            "rsi_max_rebound_points": 12.0,
            "account_size": 100_000,
        },
    }]
    monkeypatch.setattr(
        "agentloop_trader.worker.selected_strategy_result",
        lambda *args, **kwargs: {"live": {"rsi": 26.0}},
    )
    monkeypatch.setattr(
        "agentloop_trader.worker.build_alpaca_cancel_preview",
        lambda *args, **kwargs: SimpleNamespace(valid=True, preview_hash="cancel-hash"),
    )

    count, message = _cancel_late_rsi_limit_buys(
        AutomationControl(
            mode="Auto entries and exits",
            full_automation_enabled=True,
            paper_orders_enabled=True,
        ),
        adapter,
        orders,
        tracked,
        lambda *_: object(),
        SimpleNamespace(append=events.append),
    )

    assert count == 1
    assert "late RSI" in message
    assert canceled == [("rsi-order-1", "cancel-hash")]
    assert events[0].event_type == "worker_rsi_late_buy_cancelled"
    assert events[0].payload["rebound_points"] == 12.5


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
    )

    updated = store.read()[0]
    assert sent == 1
    assert updated.enabled is True
    assert updated.repeat_after_exit is True
    assert updated.cycle_state == "order_pending"
    assert updated.status == "Buy order sent"


def test_worker_reconciles_filled_watchlist_order_with_open_position(monkeypatch, tmp_path):
    store = BuyWatchlistStore(tmp_path / "watchlist.json")
    plan = BuyWatchPlan(
        plan_id=buy_watch_plan_id("HOOD", "4h", "Trend pullback continuation"),
        symbol="HOOD",
        interval="4h",
        history="5y",
        price_data_source="Ticker (Alpaca)",
        strategy_label="Trend pullback continuation",
        repeat_after_exit=True,
        cycle_state="order_pending",
        active_order_id="queued-order-filled",
        status="Buy order sent",
    )
    store.upsert(plan)
    filled_order = {
        "Alpaca Order ID": "queued-order-filled",
        "Symbol": "HOOD",
        "Side": "buy",
        "Status": "filled",
        "Filled Qty": 43,
    }
    position = {"Symbol": "HOOD", "Quantity": 43}
    monkeypatch.setattr(
        "agentloop_trader.worker._send_entry",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Open position must block a new submission")),
    )

    sent, _, message = _send_watchlist_entries(
        AutomationControl(mode="Auto entries and exits", full_automation_enabled=True),
        SimpleNamespace(),
        [position],
        [filled_order],
        [],
        lambda *_: None,
        SimpleNamespace(append=lambda event: None),
        store,
        latest_prices={"HOOD": 114.75},
    )

    updated = store.read()[0]
    assert sent == 0
    assert updated.cycle_state == "position_open"
    assert updated.cycle_had_filled_position is True
    assert updated.active_order_id == "queued-order-filled"
    assert updated.status == "Position open"
    assert "queued order filled" in updated.detail.lower()
    assert "still being refreshed" not in message


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

    _send_watchlist_entries(*common_args, latest_prices={"TSLA": 250.0})
    assert store.read()[0].cycle_state == "waiting_for_signal_reset"

    monkeypatch.setattr("agentloop_trader.worker._repeat_signal_state", lambda *args: (True, ""))
    _send_watchlist_entries(*common_args, latest_prices={"TSLA": 250.0})
    assert store.read()[0].cycle_state == "waiting_for_signal_reset"
    assert store.read()[0].status == "Waiting for a new BUY"

    monkeypatch.setattr("agentloop_trader.worker._repeat_signal_state", lambda *args: (False, ""))
    _send_watchlist_entries(*common_args, latest_prices={"TSLA": 250.0})
    assert store.read()[0].cycle_state == "waiting_for_buy"
    assert store.read()[0].enabled is True


def test_repeating_setup_retries_an_unfilled_canceled_order_without_signal_reset(monkeypatch, tmp_path):
    store = BuyWatchlistStore(tmp_path / "watchlist.json")
    plan = BuyWatchPlan(
        plan_id=buy_watch_plan_id("HOOD", "4h", "Trend pullback continuation"),
        symbol="HOOD",
        interval="4h",
        history="5y",
        price_data_source="Ticker (Alpaca)",
        strategy_label="Trend pullback continuation",
        strategy_settings={"reentry_cooldown_minutes": 0},
        repeat_after_exit=True,
        cycle_state="order_pending",
        active_order_id="queued-order-1",
    )
    store.upsert(plan)
    canceled_order = {
        "Alpaca Order ID": "queued-order-1",
        "Symbol": "HOOD",
        "Side": "buy",
        "Status": "canceled",
        "Filled Qty": 0,
    }
    monkeypatch.setattr(
        "agentloop_trader.worker._send_entry",
        lambda *args, **kwargs: (
            1,
            [*args[4], {"broker_order_id": "queued-order-2"}],
            "Sent paper buy for 5 HOOD.",
        ),
    )
    control = AutomationControl(mode="Auto entries and exits", full_automation_enabled=True)
    adapter = SimpleNamespace(position_records=lambda **kwargs: [], order_records=lambda **kwargs: [])
    common = (
        control,
        adapter,
        [],
        [],
        [],
        lambda *_: None,
        SimpleNamespace(append=lambda event: None),
        store,
    )

    _send_watchlist_entries(
        control, adapter, [], [canceled_order], [], lambda *_: None,
        SimpleNamespace(append=lambda event: None), store,
        latest_prices={"HOOD": 115.0},
    )
    assert store.read()[0].cycle_state == "waiting_for_retry"

    _send_watchlist_entries(*common, latest_prices={"HOOD": 115.0})
    assert store.read()[0].cycle_state == "waiting_for_buy"

    sent, _, _ = _send_watchlist_entries(*common, latest_prices={"HOOD": 115.0})
    updated = store.read()[0]
    assert sent == 1
    assert updated.cycle_state == "order_pending"
    assert updated.active_order_id == "queued-order-2"


def test_manual_buy_order_does_not_become_the_queued_setups_cycle(monkeypatch, tmp_path):
    store = BuyWatchlistStore(tmp_path / "watchlist.json")
    plan = BuyWatchPlan(
        plan_id=buy_watch_plan_id("HOOD", "4h", "Trend pullback continuation"),
        symbol="HOOD",
        interval="4h",
        history="5y",
        price_data_source="Ticker (Alpaca)",
        strategy_label="Trend pullback continuation",
        repeat_after_exit=True,
    )
    store.upsert(plan)
    manual_order = {
        "Alpaca Order ID": "manual-order-1",
        "Symbol": "HOOD",
        "Side": "buy",
        "Status": "accepted",
        "Filled Qty": 0,
    }
    monkeypatch.setattr(
        "agentloop_trader.worker._send_entry",
        lambda *args, **kwargs: (0, args[4], "No BUY setup right now."),
    )
    common = (
        AutomationControl(mode="Auto entries and exits", full_automation_enabled=True),
        SimpleNamespace(),
        [],
        [],
        [],
        lambda *_: None,
        SimpleNamespace(append=lambda event: None),
        store,
    )

    _send_watchlist_entries(
        common[0], common[1], [], [manual_order], [], common[5], common[6], store,
        latest_prices={"HOOD": 115.0},
    )
    blocked = store.read()[0]
    assert blocked.cycle_state == "blocked_by_order"
    assert blocked.active_order_id == ""

    _send_watchlist_entries(*common, latest_prices={"HOOD": 115.0})
    resumed = store.read()[0]
    assert resumed.cycle_state == "waiting_for_buy"
    assert resumed.cycle_had_filled_position is False


def test_legacy_canceled_order_is_migrated_out_of_signal_reset(monkeypatch, tmp_path):
    store = BuyWatchlistStore(tmp_path / "watchlist.json")
    sent_at = "2026-07-14T13:13:11-07:00"
    plan = BuyWatchPlan(
        plan_id=buy_watch_plan_id("HOOD", "4h", "Trend pullback continuation"),
        symbol="HOOD",
        interval="4h",
        history="5y",
        price_data_source="Ticker (Alpaca)",
        strategy_label="Trend pullback continuation",
        repeat_after_exit=True,
        cycle_state="waiting_for_signal_reset",
        order_sent_at=sent_at,
        last_cycle_completed_at="2026-07-15T00:03:03-07:00",
    )
    store.replace_all([plan])
    tracked = [{
        "broker_order_id": "legacy-queued-order",
        "symbol": "HOOD",
        "side": "buy",
        "status": "canceled",
        "filled_quantity": 0,
        "submitted_at": "2026-07-14T20:13:12+00:00",
    }]

    _send_watchlist_entries(
        AutomationControl(mode="Auto entries and exits", full_automation_enabled=True),
        SimpleNamespace(), [], [], tracked, lambda *_: None,
        SimpleNamespace(append=lambda event: None), store,
        latest_prices={"HOOD": 115.0},
    )

    updated = store.read()[0]
    assert updated.cycle_state == "waiting_for_retry"
    assert updated.status == "Waiting to retry"
    assert updated.cycle_had_filled_position is False


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
