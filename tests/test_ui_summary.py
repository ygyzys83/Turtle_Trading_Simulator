from agentloop_trader.ui_summary import (
    agent_decision_summary,
    agent_loop_stage_records,
    compact_status_records,
    operator_state_record,
    portfolio_story_records,
    strategy_context_records,
)


def test_compact_status_records_surface_daily_operator_state():
    rows = compact_status_records(
        mode_label="Paper trading",
        risk_approved=True,
        broker_connected=False,
        broker_state_stale=True,
        kill_switch_enabled=False,
    )
    statuses = {row["Status"]: row for row in rows}

    assert statuses["Mode"]["Value"] == "Paper trading"
    assert statuses["Trade Check"]["State"] == "ok"
    assert statuses["Alpaca"]["State"] == "block"
    assert statuses["Alpaca Data"]["Value"] == "needs refresh"
    assert statuses["Live Orders"]["Value"] == "blocked"


def test_agent_loop_stage_records_show_human_gate():
    rows = agent_loop_stage_records(
        intent_present=True,
        risk_approved=True,
        preflight_ready=True,
        human_gate_required=True,
        broker_connected=True,
    )
    stages = {row["Stage"]: row for row in rows}

    assert stages["Find Trade"]["Ready"]
    assert stages["Your Review"]["Ready"]
    assert "review" in stages["Your Review"]["Detail"].lower()


def test_portfolio_story_records_explain_agentic_loop():
    rows = portfolio_story_records()
    text = " ".join(row["Portfolio Signal"] for row in rows)

    assert "agent" in text
    assert "operator reviews" in text
    assert "Closed trades are reviewed" in text


def test_operator_state_record_prioritizes_refresh_before_orders():
    row = operator_state_record(
        intent_present=True,
        risk_approved=True,
        preflight_ready=True,
        execution_mode="paper",
        broker_connected=True,
        broker_state_stale=True,
        alpaca_enabled=True,
        buy_preview_ready=True,
        buy_preview_armed=False,
        exit_preview_count=0,
        cancelable_order_count=0,
        open_position_count=0,
    )

    assert row["State"] == "Refresh Alpaca"
    assert "Refresh Alpaca" in row["Next Action"]


def test_operator_state_record_surfaces_ready_buy_review():
    row = operator_state_record(
        intent_present=True,
        risk_approved=True,
        preflight_ready=True,
        execution_mode="paper",
        broker_connected=True,
        broker_state_stale=False,
        alpaca_enabled=True,
        buy_preview_ready=True,
        buy_preview_armed=False,
        exit_preview_count=0,
        cancelable_order_count=0,
        open_position_count=0,
    )

    assert row["State"] == "Ready to review"
    assert "Review" in row["Next Action"]


def test_strategy_context_records_explain_signal_and_price_levels():
    rows = strategy_context_records(
        {"signal": "long", "last_p": 105, "don_high": 100, "don_low": 95, "last_atr": 2.5, "sma_up": True},
        entry_window=20,
        exit_window=10,
        moving_average_window=200,
    )
    topics = {row["Topic"]: row for row in rows}

    assert topics["Signal"]["Read"] == "Buy setup"
    assert "above" in topics["Breakout"]["Read"]
    assert topics["Volatility"]["Read"] == "2.38% ATR"


def test_agent_decision_summary_names_next_action():
    summary = agent_decision_summary(
        intent_present=False,
        thesis="",
        blocked_reasons=["No entry signal."],
        next_action="Wait for a valid strategy signal.",
    )

    assert "No entry signal" in summary
    assert "Wait" in summary
