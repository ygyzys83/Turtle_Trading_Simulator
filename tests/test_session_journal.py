from agentloop_trader.session_journal import (
    PaperSessionSnapshot,
    alpaca_paper_activity_records,
    automatic_exit_records,
    new_session_id,
    paper_performance_records,
    paper_testing_progress_records,
    paper_trading_review_records,
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
            {
                "status": "filled", "side": "buy", "filled_quantity": "40",
                "average_fill_price": "300.00",
            },
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


def test_automatic_exit_records_explain_worker_trigger_and_fill():
    audit_records = [{
        "created_at": "2026-07-14T11:15:38-07:00",
        "event_type": "worker_paper_exit_sent",
        "message": "Background worker sent an Alpaca paper exit.",
        "payload": {
            "symbol": "WYFI",
            "quantity": 142,
            "broker_order_id": "exit-123",
            "reason": "Exit now because WYFI is at or below the break-even stop at $32.09.",
            "exit_details": {
                "current_price": 32.01,
                "trigger_price": 32.09,
                "trigger_source": "break-even stop",
            },
        },
    }]
    tracked_orders = [{"broker_order_id": "exit-123", "average_fill_price": "32.00"}]

    rows = automatic_exit_records(audit_records, tracked_orders)

    assert rows == [{
        "Time": "2026-07-14T11:15:38-07:00",
        "Ticker": "WYFI",
        "Shares": "142",
        "Decision Price": "$32.01",
        "Sell Trigger": "$32.09",
        "Trigger Rule": "break-even stop",
        "Alpaca Fill": "$32.00",
        "Reason": "Exit now because WYFI is at or below the break-even stop at $32.09.",
        "Alpaca Order ID": "exit-123",
    }]


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
    assert metrics["Estimated live fees on filled orders"] == "$0.01"


def test_paper_trading_review_records_surface_next_step_and_account_value():
    rows = paper_trading_review_records(_snapshot(), alpaca_position_count=1, alpaca_account_value=100_250.50)
    reads = {row["Area"]: row["Read"] for row in rows}

    assert reads["Account"] == "$100,250.50"
    assert reads["Open positions"] == 1
    assert reads["Waiting orders"] == 1
    assert reads["Paper buys sent"] == 1
    assert reads["Paper exits sent"] == 1
    assert reads["Estimated live fees"] == "$0.01"
    assert "Watch or cancel" in reads["Next step"]


def test_paper_testing_progress_records_track_days_and_status():
    audit_records = [
        {
            "created_at": f"2026-07-{day:02d}T16:00:00-07:00",
            "event_type": "alpaca_paper_order_submitted" if day == 1 else "alpaca_paper_exit_submitted",
            "message": "paper event",
            "payload": {},
        }
        for day in range(1, 11)
    ]
    tracked_orders = [{"status": "filled"}]

    rows = paper_testing_progress_records(audit_records, tracked_orders, target_days=10)
    reads = {row["Check"]: row["Read"] for row in rows}

    assert reads["Paper trading days"] == "10/10"
    assert reads["Paper buys tested"] == 1
    assert reads["Paper exits tested"] == 9
    assert reads["Testing status"] == "Ready to review live setup"
