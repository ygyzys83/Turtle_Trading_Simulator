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
