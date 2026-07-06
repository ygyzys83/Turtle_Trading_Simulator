from agentloop_trader.ui_summary import (
    agent_decision_summary,
    agent_loop_stage_records,
    compact_status_records,
    operator_state_record,
    portfolio_story_records,
    setup_scorecard_records,
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


def test_setup_scorecard_records_show_clean_setup():
    rows = setup_scorecard_records(
        {
            "signal": "long",
            "last_p": 105,
            "don_high": 100,
            "don_low": 95,
            "last_atr": 2.5,
            "stop_from_entry": 5,
            "sma_up": True,
        },
        risk_approved=True,
        blocked_reasons=[],
    )
    reads = {row["Read"]: row for row in rows}

    assert reads["Overall"]["Status"] == "A"
    assert reads["Breakout"]["Status"] == "Triggered"
    assert reads["Room above exit"]["Status"] == "2.0x stop"


def test_setup_scorecard_records_show_first_blocker():
    rows = setup_scorecard_records(
        {
            "signal": "long",
            "last_p": 105,
            "don_high": 100,
            "don_low": 95,
            "last_atr": 2.5,
            "stop_from_entry": 5,
            "sma_up": True,
        },
        risk_approved=False,
        blocked_reasons=["Position would be too large."],
    )
    reads = {row["Read"]: row for row in rows}

    assert reads["Overall"]["Status"] == "C"
    assert reads["Overall"]["Plain English"] == "Position would be too large."


def test_setup_scorecard_records_follow_enabled_inputs():
    rows = setup_scorecard_records(
        {
            "signal": "long",
            "last_p": 105,
            "don_high": 100,
            "don_low": 95,
            "last_atr": 2.5,
            "stop_from_entry": 5,
            "sma_up": True,
            "volume_status": "Strong",
            "rsi_status": "Good",
        },
        risk_approved=True,
        blocked_reasons=[],
        enabled_inputs={"volume": True, "rsi": True, "volatility": False, "room_above_exit": False},
    )
    reads = {row["Read"]: row for row in rows}

    assert "Volume" in reads
    assert "RSI condition" in reads
    assert "Volatility" not in reads


def test_setup_scorecard_records_show_pullback_core_inputs():
    rows = setup_scorecard_records(
        {
            "setup_type": "pullback",
            "signal": "long",
            "last_p": 105,
            "don_high": 100,
            "don_low": 95,
            "last_atr": 2.5,
            "stop_from_entry": 5,
            "sma_up": True,
            "pullback_ready": True,
            "pullback_depth_pct": 4.2,
            "momentum_turn": True,
        },
        risk_approved=True,
        blocked_reasons=[],
        strategy_type="pullback",
    )
    reads = {row["Read"]: row for row in rows}

    assert reads["Overall"]["Status"] == "A"
    assert reads["Pullback"]["Status"] == "Controlled"
    assert reads["Momentum turn"]["Status"] == "Yes"
