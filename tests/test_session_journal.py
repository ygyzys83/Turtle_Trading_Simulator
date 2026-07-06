from agentloop_trader.session_journal import (
    PaperSessionSnapshot,
    alpaca_paper_activity_records,
    new_session_id,
    paper_performance_records,
    session_summary_records,
    session_timeline_records,
)


def _snapshot():
    return PaperSessionSnapshot(
        session_id="paper-test",
        started_at="2026-07-05T16:00:00Z",
        mode="Paper trading",
        paper_cash=49_000.0,
        paper_equity=50_500.0,
        session_pnl=500.0,
        local_orders=[
            {"Status": "filled", "Notional": 1000.0},
            {"Status": "rejected", "Notional": 0.0},
        ],
        local_positions=[
            {"Symbol": "AAPL", "Book Value": 1000.0},
        ],
        tracked_alpaca_orders=[
            {"status": "filled", "filled_quantity": "40"},
            {"status": "canceled", "quantity": "10"},
            {"status": "accepted", "quantity": "5"},
        ],
        audit_records=[
            {
                "created_at": "2026-07-05T16:01:00Z",
                "event_type": "alpaca_paper_order_submitted",
                "message": "submitted",
                "payload": {"symbol": "AAPL", "side": "buy", "quantity": 40, "preview_hash": "abc"},
            },
            {
                "created_at": "2026-07-05T16:02:00Z",
                "event_type": "alpaca_paper_cancel_submitted",
                "message": "canceled",
                "payload": {"symbol": "AAPL", "cancel_status": "canceled", "cancel_preview_hash": "def"},
            },
            {
                "created_at": "2026-07-05T16:03:00Z",
                "event_type": "alpaca_paper_exit_submitted",
                "message": "exit submitted",
                "payload": {"symbol": "AAPL", "side": "sell", "quantity": 40, "preview_hash": "ghi"},
            },
        ],
    )


def test_new_session_id_is_display_ready():
    session_id = new_session_id()

    assert session_id.startswith("paper-")


def test_new_session_id_uses_pacific_wall_clock_prefix():
    from datetime import datetime
    from agentloop_trader.models import PACIFIC_TIME

    session_id = new_session_id()
    prefix = datetime.now(PACIFIC_TIME).strftime("paper-%Y%m%d-")

    assert session_id.startswith(prefix)


def test_session_summary_records_count_local_and_alpaca_activity():
    metrics = {row["Metric"]: row["Value"] for row in session_summary_records(_snapshot())}

    assert metrics["Session ID"] == "paper-test"
    assert metrics["Filled app paper orders"] == 1
    assert metrics["Filled Alpaca orders"] == 1
    assert metrics["Canceled Alpaca orders"] == 1
    assert metrics["Paper buys sent"] == 1
    assert metrics["Paper cancels sent"] == 1
    assert metrics["Paper exits sent"] == 1


def test_session_timeline_records_extract_key_payload_fields():
    rows = session_timeline_records(_snapshot().audit_records)

    assert rows[0]["Record Type"] == "alpaca_paper_order_submitted"
    assert rows[0]["Symbol"] == "AAPL"
    assert rows[0]["Review ID"] == "abc"
    assert rows[1]["Status"] == "canceled"


def test_paper_performance_records_report_only_local_simulator_metrics():
    metrics = {row["Metric"]: row["Value"] for row in paper_performance_records(_snapshot())}

    assert metrics["Simulator cash"] == "$49,000.00"
    assert metrics["Session P&L"] == "$500.00"
    assert metrics["Filled app order value"] == "$1,000.00"
    assert metrics["Filled simulator orders"] == 1
    assert "Filled Alpaca shares" not in metrics


def test_alpaca_paper_activity_records_report_saved_broker_history():
    metrics = {row["Metric"]: row["Value"] for row in alpaca_paper_activity_records(_snapshot())}

    assert metrics["Saved Alpaca orders"] == 3
    assert metrics["Filled Alpaca shares"] == "40"
    assert metrics["Alpaca orders waiting to fill"] == 1
