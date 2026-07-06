from agentloop_trader.ui_summary import (
    agent_loop_stage_records,
    compact_status_records,
    portfolio_story_records,
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

    assert statuses["Execution Mode"]["Value"] == "Paper trading"
    assert statuses["Risk Gate"]["State"] == "ok"
    assert statuses["Alpaca"]["State"] == "block"
    assert statuses["Broker State"]["Value"] == "stale"
    assert statuses["Live Writes"]["Value"] == "blocked"


def test_agent_loop_stage_records_show_human_gate():
    rows = agent_loop_stage_records(
        intent_present=True,
        risk_approved=True,
        preflight_ready=True,
        human_gate_required=True,
        broker_connected=True,
    )
    stages = {row["Stage"]: row for row in rows}

    assert stages["Propose"]["Ready"]
    assert stages["Human Gate"]["Ready"]
    assert "Manual approval" in stages["Human Gate"]["Detail"]


def test_portfolio_story_records_explain_agentic_loop():
    rows = portfolio_story_records()
    text = " ".join(row["Portfolio Signal"] for row in rows)

    assert "agent" in text
    assert "Human-in-the-loop" in text
    assert "Post-trade review" in text
