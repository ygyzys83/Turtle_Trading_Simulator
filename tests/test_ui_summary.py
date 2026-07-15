from agentloop_trader.ui_summary import (
    agent_decision_summary,
    agent_loop_stage_records,
    alpaca_evidence_summary_records,
    buy_requirement_records,
    compact_status_records,
    managed_position_records,
    no_buy_reason,
    operator_state_record,
    optional_quality_input_records,
    optional_sell_quality_records,
    position_exit_plan_records,
    saved_records_overview_records,
    sell_requirement_records,
    setup_scorecard_records,
    strategy_context_records,
    trade_evidence_summary_records,
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
    assert "ATR is $2.50" in reads["Volatility"]["Plain English"]
    assert "Stop distance is $5.00" in reads["Volatility"]["Plain English"]
    assert "Approx stop is $100.00" in reads["Volatility"]["Plain English"]


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


def test_breakout_buy_requirements_show_exact_price_and_distance():
    rows = buy_requirement_records(
        {
            "setup_type": "breakout",
            "last_p": 100.0,
            "entry_level": 105.0,
            "last_sma": 95.0,
            "trend_window": 50,
            "pos_size": 10,
            "buy_requirements": {
                "Price above 20-bar high": False,
                "Price above rising 50-bar trend filter": True,
                "Position size above zero": True,
            },
        },
        interval="4h",
    )
    by_rule = {row["Required BUY Rule"]: row for row in rows}

    assert by_rule["Breakout price"]["Required"] == "Completed 4h bar closes above $105.00"
    assert by_rule["Breakout price"]["Distance"] == "+5.00%"
    assert by_rule["Position size"]["Status"] == "Pass"


def test_optional_quality_input_records_do_not_claim_to_create_buy():
    rows = optional_quality_input_records({"volume": True, "rsi": True, "market_condition": False})
    inputs = {row["Quality Input"]: row for row in rows}

    assert inputs["Volume"]["Creates BUY?"] == "No"
    assert inputs["RSI"]["Creates BUY?"] == "Optional rule"
    assert "Require RSI 50-70 for BUY" in inputs["RSI"]["Plain English"]
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
    assert records["Sell Exit Length"] == "10 bars"
    assert records["Trend Filter Length"] == "200 bars"
    assert records["Pullback Average Length"] == "20 bars"
    assert records["Momentum Turn Length"] == "5 bars"
    assert records["Current Exit Check"] == "Hold because price is above the pullback average."


def test_trade_evidence_summary_records_put_plain_answer_before_details():
    rows = trade_evidence_summary_records(
        intent_present=True,
        risk_approved=False,
        preflight_ready=False,
        setup_rows=[{"Read": "Overall", "Status": "C"}],
        blocked_reasons=["Position would be too large."],
    )
    records = {row["Evidence Area"]: row for row in rows}

    assert records["Setup quality"]["Current Read"] == "C"
    assert records["Trade idea"]["Current Read"] == "Present"
    assert records["Risk check"]["Current Read"] == "Blocked"
    assert records["Main blocker"]["Current Read"] == "Position would be too large."


def test_alpaca_evidence_summary_records_count_operating_state():
    rows = alpaca_evidence_summary_records(
        alpaca_connected=True,
        alpaca_state_stale=False,
        paper_orders_enabled=True,
        alpaca_positions=[{"Symbol": "IBM"}],
        alpaca_orders=[{"Status": "accepted"}, {"Status": "filled"}],
        tracked_orders=[{"broker_order_id": "1"}, {"broker_order_id": "2"}],
        automation_status="Auto exits only",
    )
    records = {row["Evidence Area"]: row for row in rows}

    assert records["Broker connection"]["Current Read"] == "Connected"
    assert records["Open positions"]["Current Read"] == 1
    assert records["Waiting orders"]["Current Read"] == 1
    assert records["Saved order records"]["Current Read"] == 2
    assert records["Automation"]["Current Read"] == "Auto exits only"


def test_saved_records_overview_records_summarize_evidence_without_raw_log():
    rows = saved_records_overview_records(
        audit_records=[
            {"event_type": "alpaca_paper_order_submitted"},
            {"event_type": "auto_paper_entry_submitted"},
            {"event_type": "auto_paper_exit_submitted"},
        ],
        tracked_orders=[{"broker_order_id": "1"}],
        automation_snapshots=[{"session_id": "s1"}, {"session_id": "s2"}],
    )
    records = {row["Record Set"]: row for row in rows}

    assert records["Activity log"]["Count"] == 3
    assert records["Alpaca paper orders"]["Count"] == 1
    assert records["Automation checks"]["Count"] == 2
    assert records["Paper buys sent"]["Count"] == 2
    assert records["Paper exits sent"]["Count"] == 1


def test_saved_records_overview_does_not_count_internal_position_plans_as_orders():
    rows = saved_records_overview_records(
        tracked_orders=[
            {"broker_order_id": "real", "source": "buy_watchlist"},
            {"broker_order_id": "plan", "source": "position_plan"},
            {"broker_order_id": "observation", "source": "position_observation"},
        ]
    )

    records = {row["Record Set"]: row["Count"] for row in rows}
    assert records["Alpaca paper orders"] == 1
