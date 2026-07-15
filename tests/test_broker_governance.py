import tempfile
from datetime import UTC, datetime
from pathlib import Path

from agentloop_trader.broker_governance import (
    BrokerStateStore,
    alpaca_order_lifecycle_records,
    alpaca_order_lifecycle_summary_records,
    alpaca_position_lifecycle_records,
    alpaca_position_lifecycle_summary_records,
    broker_state_health,
    build_exit_intent_from_position,
    build_exit_order_previews,
    cancelable_alpaca_order_records,
    duplicate_exposure_reasons,
    exit_position_reasons,
    market_session_advisory,
    open_exit_order_reasons,
    open_order_exposure_reasons,
    preview_already_tracked,
    reconcile_alpaca_positions,
    refresh_tracked_alpaca_orders,
)
from agentloop_trader.brokers import AlpacaConfig, build_alpaca_order_preview
from agentloop_trader.models import ExecutionDecision, RiskCheckResult
from agentloop_trader.models import TradeIntent


def test_duplicate_exposure_blocks_existing_alpaca_buy_symbol():
    intent = TradeIntent(symbol="AAPL", side="buy", quantity=1)
    positions = [{"Symbol": "AAPL", "Quantity": "10"}]

    reasons = duplicate_exposure_reasons(intent, positions)

    assert reasons == ["Alpaca paper already has an open AAPL position."]


def test_open_order_exposure_blocks_existing_accepted_buy_order():
    intent = TradeIntent(symbol="AAPL", side="buy", quantity=1)
    orders = [{"Symbol": "AAPL", "Side": "BUY", "Status": "accepted"}]

    reasons = open_order_exposure_reasons(intent, orders)

    assert reasons == ["Alpaca Orders already has an open AAPL buy order with status accepted."]


def test_open_order_exposure_normalizes_alpaca_enum_values():
    intent = TradeIntent(symbol="AAPL", side="buy", quantity=1)
    orders = [{"Symbol": "AAPL", "Side": "OrderSide.BUY", "Status": "OrderStatus.ACCEPTED"}]

    reasons = open_order_exposure_reasons(intent, orders)

    assert reasons == ["Alpaca Orders already has an open AAPL buy order with status accepted."]


def test_open_order_exposure_stays_blocked_when_position_adds_are_allowed():
    intent = TradeIntent(symbol="AAPL", side="buy", quantity=10, entry_price=100, stop_loss=95)
    orders = [{"Symbol": "AAPL", "Side": "buy", "Status": "pending_cancel"}]

    reasons = open_order_exposure_reasons(intent, orders, allow_duplicate=True)

    assert reasons
    assert "open AAPL buy order" in reasons[0]


def test_exit_position_reasons_require_existing_position():
    preview = build_exit_order_previews(
        [{"Symbol": "AAPL", "Quantity": "12"}],
        AlpacaConfig(api_key="key", api_secret="secret", paper=True),
    )[0]

    reasons = exit_position_reasons(preview, [])

    assert reasons == ["Alpaca paper has no open AAPL position to exit."]


def test_exit_position_reasons_block_quantity_above_position():
    approved = RiskCheckResult(approved=True, rejected_reasons=[], checks={"exit_preview": True})
    decision = ExecutionDecision(
        mode="paper",
        approved_for_execution=True,
        requires_manual_approval=False,
        reason="test",
        risk_check=approved,
    )
    preview = build_alpaca_order_preview(
        TradeIntent(symbol="AAPL", side="sell", quantity=20),
        decision,
        AlpacaConfig(api_key="key", api_secret="secret", paper=True),
    )

    reasons = exit_position_reasons(preview, [{"Symbol": "AAPL", "Quantity": "12"}])

    assert reasons == ["Exit quantity 20 exceeds Alpaca paper AAPL position quantity 12."]


def test_open_exit_order_reasons_block_duplicate_open_sell_order():
    preview = build_exit_order_previews(
        [{"Symbol": "AAPL", "Quantity": "12"}],
        AlpacaConfig(api_key="key", api_secret="secret", paper=True),
    )[0]
    orders = [{"Symbol": "AAPL", "Side": "OrderSide.SELL", "Status": "OrderStatus.ACCEPTED"}]

    reasons = open_exit_order_reasons(preview, orders)

    assert reasons == ["Alpaca Orders already has an open AAPL sell order with status accepted."]


def test_cancelable_alpaca_order_records_require_open_status_and_full_order_id():
    orders = [
        {
            "Order ID": "abcdef12",
            "Broker Order ID": "abcdef123456",
            "Symbol": "AAPL",
            "Side": "OrderSide.BUY",
            "Quantity": "40",
            "Status": "OrderStatus.ACCEPTED",
            "Submitted": "2026-07-05T16:37:41Z",
        },
        {
            "Order ID": "filled12",
            "Broker Order ID": "filled123456",
            "Symbol": "AAPL",
            "Side": "buy",
            "Quantity": "40",
            "Status": "filled",
        },
        {
            "Order ID": "missing1",
            "Symbol": "MSFT",
            "Side": "buy",
            "Quantity": "1",
            "Status": "accepted",
        },
    ]

    rows = cancelable_alpaca_order_records(orders)

    assert len(rows) == 1
    assert rows[0]["Alpaca Order ID"] == "abcdef123456"
    assert rows[0]["Status"] == "accepted"


def test_reconcile_alpaca_positions_marks_matched_and_unmatched_rows():
    positions = [{"Symbol": "AAPL", "Quantity": "10", "Market Value": "1000"}]
    tracked = [{
        "symbol": "AAPL",
        "source": "position_plan",
        "position_cycle_id": "buy-1",
        "status": "managed_exit_settings",
    }]

    rows = reconcile_alpaca_positions(positions, tracked)

    assert rows[0]["Status"] == "Managed current position"
    assert rows[0]["Current Cycle"] == "buy-1"
    assert rows[0]["Exit Plan Saved"]


def test_refresh_tracked_alpaca_orders_updates_lifecycle_from_alpaca_orders():
    tracked = [
        {
            "broker_order_id": "abcdef123456",
            "preview_hash": "hash1",
            "symbol": "AAPL",
            "side": "buy",
            "quantity": "40",
            "status": "accepted",
        }
    ]
    alpaca_orders = [
        {
            "Broker Order ID": "abcdef123456",
            "Symbol": "AAPL",
            "Side": "OrderSide.BUY",
            "Quantity": "40.0",
            "Status": "OrderStatus.CANCELED",
            "Submitted": "2026-07-05T16:37:41Z",
            "Filled": "",
            "Filled Qty": "0",
            "Avg Fill": "",
        }
    ]

    refreshed = refresh_tracked_alpaca_orders(tracked, alpaca_orders)

    assert refreshed[0]["status"] == "canceled"
    assert refreshed[0]["lifecycle_status"] == "canceled_at_alpaca"
    assert refreshed[0]["filled_quantity"] == "0"
    assert refreshed[0]["preview_hash"] == "hash1"


def test_refresh_tracked_alpaca_orders_marks_missing_broker_rows():
    tracked = [{"broker_order_id": "missing-order", "symbol": "AAPL", "status": "accepted"}]

    refreshed = refresh_tracked_alpaca_orders(tracked, [])

    assert refreshed[0]["lifecycle_status"] == "missing_from_alpaca_orders"


def test_alpaca_order_lifecycle_records_and_summary_are_display_ready():
    tracked = [{"broker_order_id": "abcdef123456", "symbol": "AAPL", "side": "buy", "quantity": "40"}]
    alpaca_orders = [
        {
            "Broker Order ID": "abcdef123456",
            "Symbol": "AAPL",
            "Side": "buy",
            "Quantity": "40",
            "Status": "filled",
            "Filled Qty": "40",
            "Avg Fill": "212.34",
        }
    ]

    rows = alpaca_order_lifecycle_records(tracked, alpaca_orders)
    summary = alpaca_order_lifecycle_summary_records(refresh_tracked_alpaca_orders(tracked, alpaca_orders))

    assert rows[0]["Tracking Status"] == "Filled at Alpaca"
    assert rows[0]["Filled Qty"] == "40"
    metrics = {row["Metric"]: row["Value"] for row in summary}
    assert metrics["Filled orders at Alpaca"] == 1


def test_alpaca_position_lifecycle_matches_filled_order_to_position():
    positions = [
        {
            "Symbol": "AAPL",
            "Quantity": "40",
            "Market Value": "8500",
            "Average Entry": "212.34",
        }
    ]
    tracked = [
        {
            "broker_order_id": "position-plan-abcdef123456",
            "symbol": "AAPL",
            "source": "position_plan",
            "status": "managed_exit_settings",
            "position_cycle_id": "abcdef123456",
            "position_basis_order_id": "abcdef123456",
            "exit_settings": {"auto_exit_enabled": True},
        }
    ]

    rows = alpaca_position_lifecycle_records(positions, tracked)
    summary = alpaca_position_lifecycle_summary_records(rows)

    assert rows[0]["Tracking Status"] == "Managed current position"
    assert rows[0]["Auto Exit On"]
    assert rows[0]["Current Cycle"] == "abcdef123456"
    metrics = {row["Metric"]: row["Value"] for row in summary}
    assert metrics["Positions with saved exit plans"] == 1
    assert metrics["Positions with auto exit on"] == 1


def test_alpaca_position_lifecycle_marks_untracked_position():
    rows = alpaca_position_lifecycle_records(
        [{"Symbol": "MSFT", "Quantity": "5", "Market Value": "2000", "Average Entry": "400"}],
        [],
    )

    assert rows[0]["Tracking Status"] == "Unmanaged position"
    assert not rows[0]["Auto Exit On"]


def test_old_filled_order_does_not_make_a_new_position_look_managed():
    rows = alpaca_position_lifecycle_records(
        [{"Symbol": "NVDA", "Quantity": "23", "Market Value": "4850", "Average Entry": "210.75"}],
        [{
            "broker_order_id": "old-buy",
            "symbol": "NVDA",
            "side": "buy",
            "status": "filled",
            "filled_quantity": "24",
            "average_fill_price": "202.48",
        }],
    )

    assert rows[0]["Tracking Status"] == "Unmanaged position"
    assert not rows[0]["Exit Plan Saved"]
    assert not rows[0]["Auto Exit On"]


def test_broker_state_health_flags_stale_refreshes():
    health = broker_state_health(alpaca_connected=True, position_records=None, order_records=[])

    assert not health.ready
    assert health.stale
    assert "positions" in health.reasons[0]


def test_exit_preview_uses_sell_side_for_existing_position():
    intent = build_exit_intent_from_position({"Symbol": "AAPL", "Quantity": "12"})
    previews = build_exit_order_previews(
        [{"Symbol": "AAPL", "Quantity": "12"}],
        AlpacaConfig(api_key="key", api_secret="secret", paper=True),
    )

    assert intent.side == "sell"
    assert intent.quantity == 12
    assert previews[0].order["side"] == "sell"
    assert previews[0].valid


def test_preview_already_tracked_detects_duplicate_open_signal():
    tracked = [{"preview_hash": "abc", "status": "filled"}]

    assert preview_already_tracked("abc", tracked)
    assert not preview_already_tracked("def", tracked)


def test_broker_state_store_upserts_records():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = BrokerStateStore(f"{tmp_dir}/broker_state.json")
        store.upsert({"broker_order_id": "1", "symbol": "AAPL", "status": "accepted"})
        store.upsert({"broker_order_id": "1", "symbol": "AAPL", "status": "filled"})

        records = store.read()

    assert len(records) == 1
    assert records[0]["status"] == "filled"


def test_broker_state_store_replace_all_merges_existing_history():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = BrokerStateStore(f"{tmp_dir}/broker_state.json")
        store.upsert({"broker_order_id": "1", "symbol": "AAPL"})
        store.replace_all([{"broker_order_id": "2", "symbol": "MSFT"}])

        records = store.read()

    assert {record["broker_order_id"] for record in records} == {"1", "2"}


def test_broker_state_store_preserves_exit_settings_during_refresh():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = BrokerStateStore(f"{tmp_dir}/broker_state.json")
        store.replace_all([
            {"broker_order_id": "1", "status": "accepted", "exit_settings": {"interval": "1h"}}
        ])
        store.replace_all([{"broker_order_id": "1", "status": "filled"}])

        record = store.read()[0]

    assert record["status"] == "filled"
    assert record["exit_settings"] == {"interval": "1h"}


def test_broker_state_store_rejects_stale_position_plan_overwrite():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = BrokerStateStore(f"{tmp_dir}/broker_state.json")
        store.upsert({
            "broker_order_id": "position-plan-buy-1",
            "source": "position_plan",
            "exit_settings": {
                "interval": "4h",
                "plan_user_updated_at": "2026-07-15T12:01:00-07:00",
            },
        })
        store.replace_all([{
            "broker_order_id": "position-plan-buy-1",
            "source": "position_plan",
            "exit_settings": {
                "interval": "1h",
                "plan_user_updated_at": "2026-07-15T12:00:00-07:00",
                "highest_high_since_entry": 220.0,
            },
        }])

        record = store.read()[0]

    assert record["exit_settings"]["interval"] == "4h"
    assert "highest_high_since_entry" not in record["exit_settings"]


def test_broker_state_store_upsert_rejects_stale_position_plan_overwrite():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = BrokerStateStore(f"{tmp_dir}/broker_state.json")
        store.upsert({
            "broker_order_id": "position-plan-buy-1",
            "source": "position_plan",
            "exit_settings": {
                "interval": "4h",
                "plan_user_updated_at": "2026-07-15T12:01:00-07:00",
            },
        })
        store.upsert({
            "broker_order_id": "position-plan-buy-1",
            "source": "position_plan",
            "exit_settings": {
                "interval": "1h",
                "plan_user_updated_at": "2026-07-15T12:00:00-07:00",
            },
        })

        record = store.read()[0]

    assert record["exit_settings"]["interval"] == "4h"


def test_broker_state_store_accepts_worker_state_for_current_user_plan_revision():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = BrokerStateStore(f"{tmp_dir}/broker_state.json")
        user_updated_at = "2026-07-15T12:01:00-07:00"
        store.upsert({
            "broker_order_id": "position-plan-buy-1",
            "source": "position_plan",
            "exit_settings": {"interval": "4h", "plan_user_updated_at": user_updated_at},
        })
        store.replace_all([{
            "broker_order_id": "position-plan-buy-1",
            "source": "position_plan",
            "exit_settings": {
                "interval": "4h",
                "plan_user_updated_at": user_updated_at,
                "highest_high_since_entry": 220.0,
            },
        }])

        record = store.read()[0]

    assert record["exit_settings"]["interval"] == "4h"
    assert record["exit_settings"]["highest_high_since_entry"] == 220.0


def test_broker_state_store_waits_for_short_lived_writer_lock(monkeypatch, tmp_path):
    state_path = tmp_path / "broker_state.json"
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_path.write_text("held by another writer", encoding="utf-8")
    waits = []

    def release_lock_after_wait(seconds):
        waits.append(seconds)
        lock_path.unlink()

    monkeypatch.setattr(
        "agentloop_trader.broker_governance.time_module.sleep",
        release_lock_after_wait,
    )
    store = BrokerStateStore(state_path)

    store.replace_all([{"broker_order_id": "1", "symbol": "WYFI"}])

    assert waits == [0.02]
    assert store.read()[0]["symbol"] == "WYFI"
    assert not lock_path.exists()


def test_broker_state_store_read_waits_for_active_writer(monkeypatch, tmp_path):
    state_path = tmp_path / "broker_state.json"
    state_path.write_text('[{"broker_order_id": "1"}]', encoding="utf-8")
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_path.write_text("held by another writer", encoding="utf-8")
    waits = []

    def release_lock_after_wait(seconds):
        waits.append(seconds)
        lock_path.unlink()

    monkeypatch.setattr(
        "agentloop_trader.broker_governance.time_module.sleep",
        release_lock_after_wait,
    )

    records = BrokerStateStore(state_path).read()

    assert waits == [0.02]
    assert records == [{"broker_order_id": "1"}]


def test_broker_state_store_retries_transient_windows_replace_lock(monkeypatch, tmp_path):
    state_path = tmp_path / "broker_state.json"
    store = BrokerStateStore(state_path)
    real_replace = Path.replace
    attempts = []
    waits = []

    def replace_once_unlocked(path, target):
        attempts.append((path, target))
        if len(attempts) == 1:
            raise PermissionError(5, "Access is denied", str(target))
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace_once_unlocked)
    monkeypatch.setattr(
        "agentloop_trader.broker_governance.time_module.sleep",
        waits.append,
    )

    store.replace_all([{"broker_order_id": "1", "symbol": "AAPL"}])

    assert len(attempts) == 2
    assert waits == [0.02]
    assert store.read()[0]["symbol"] == "AAPL"
    assert not list(tmp_path.glob("*.tmp"))


def test_market_session_advisory_identifies_weekend_closed():
    advisory = market_session_advisory(datetime(2026, 7, 5, 16, 0, tzinfo=UTC))

    assert advisory["Market Session"] == "closed_or_extended"
    assert advisory["Open"] is False


def test_market_session_advisory_exposes_regular_open_flag():
    advisory = market_session_advisory(datetime(2026, 7, 7, 18, 0, tzinfo=UTC))

    assert advisory["Market Session"] == "open"
    assert advisory["Open"] is True
