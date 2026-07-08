from agentloop_trader.ui_summary import (
    agent_decision_summary,
    agent_loop_stage_records,
    buy_requirement_records,
    compact_status_records,
    managed_position_records,
    no_buy_reason,
    operator_state_record,
    optional_quality_input_records,
    optional_sell_quality_records,
    position_exit_plan_records,
    sell_requirement_records,
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
    assert stages["Your Action"]["Ready"]
    assert "click" in stages["Your Action"]["Detail"].lower()


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

    assert row["State"] == "Ready to send"
    assert "Send" in row["Next Action"]


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


def test_buy_requirement_records_separate_hard_buy_rules():
    rows = buy_requirement_records(
        {
            "buy_requirements": {
                "Price above entry level": False,
                "Trend filter rising": True,
            }
        }
    )
    rules = {row["Required BUY Rule"]: row for row in rows}

    assert rules["Trend filter rising"]["Status"] == "Pass"
    assert rules["Price above entry level"]["Status"] == "Not met"
    assert "must pass" in rules["Price above entry level"]["Plain English"]


def test_optional_quality_input_records_do_not_claim_to_create_buy():
    rows = optional_quality_input_records({"volume": True, "rsi": True, "market_condition": False})
    inputs = {row["Quality Input"]: row for row in rows}

    assert inputs["Volume"]["Creates BUY?"] == "No"
    assert inputs["RSI"]["Creates BUY?"] == "No"
    assert "Market condition" not in inputs


def test_no_buy_reason_prefers_strategy_reason():
    assert no_buy_reason({"no_trade_reason": "No BUY because momentum has not turned up."}) == (
        "No BUY because momentum has not turned up."
    )


def test_sell_requirement_records_separate_hard_sell_rules():
    rows = sell_requirement_records(
        {"sell_requirements": {"Price below pullback average": False}},
        exit_preview_count=1,
        exit_settings_saved=True,
    )
    rules = {row["Required SELL Rule"]: row for row in rows}

    assert rules["Price below pullback average"]["Status"] == "Not met"
    assert rules["Alpaca paper position can be sold"]["Status"] == "Pass"
    assert rules["Saved exit settings are available"]["Status"] == "Pass"


def test_optional_sell_quality_records_do_not_claim_to_create_sell():
    rows = optional_sell_quality_records({"volume": True, "rsi": True})
    inputs = {row["Quality Input"]: row for row in rows}

    assert inputs["Volume"]["Creates SELL?"] == "No"
    assert inputs["RSI"]["Creates SELL?"] == "No"


def test_managed_position_records_show_daily_management_state():
    rows = managed_position_records(
        [{"Symbol": "IBM", "Quantity": "62", "Market Value": "19200", "Average Entry": "310.5"}],
        {"IBM": {"auto_exit_enabled": True, "strategy_label": "Trend pullback continuation"}},
    )

    assert rows[0]["Symbol"] == "IBM"
    assert rows[0]["Auto Exit"] == "On"
    assert rows[0]["Management Status"] == "Managed"
    assert rows[0]["Market Value"] == "$19,200.00"


def test_position_exit_plan_records_summarize_saved_exit_settings():
    rows = position_exit_plan_records(
        {
            "strategy_label": "Trend pullback continuation",
            "auto_exit_enabled": True,
            "exit_window": 10,
            "atr_stop_multiplier": 2.0,
            "moving_average_window": 200,
            "pullback_average_length": 20,
            "momentum_turn_length": 5,
        },
        exit_ready=False,
        exit_reason="Hold because price is above the pullback average.",
    )
    records = {row["Field"]: row["Value"] for row in rows}

    assert records["Exit Strategy"] == "Trend pullback continuation"
    assert records["Auto Exit"] == "On"
    assert records["Current Exit Check"] == "Hold because price is above the pullback average."
