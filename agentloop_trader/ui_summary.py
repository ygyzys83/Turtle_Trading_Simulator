from __future__ import annotations


def compact_status_records(
    mode_label: str,
    risk_approved: bool,
    broker_connected: bool,
    broker_state_stale: bool,
    kill_switch_enabled: bool,
    live_writes_blocked: bool = True,
) -> list[dict]:
    return [
        {"Status": "Execution Mode", "Value": mode_label, "State": "info"},
        {"Status": "Risk Gate", "Value": "approved" if risk_approved else "blocked", "State": "ok" if risk_approved else "block"},
        {"Status": "Alpaca", "Value": "connected" if broker_connected else "disconnected", "State": "ok" if broker_connected else "block"},
        {"Status": "Broker State", "Value": "stale" if broker_state_stale else "fresh", "State": "block" if broker_state_stale else "ok"},
        {"Status": "Kill Switch", "Value": "on" if kill_switch_enabled else "off", "State": "block" if kill_switch_enabled else "ok"},
        {"Status": "Live Writes", "Value": "blocked" if live_writes_blocked else "available", "State": "ok" if live_writes_blocked else "block"},
    ]


def agent_loop_stage_records(
    intent_present: bool,
    risk_approved: bool,
    preflight_ready: bool,
    human_gate_required: bool,
    broker_connected: bool,
) -> list[dict]:
    stages = [
        ("Observe", True, "Market data loaded and strategy rules evaluated."),
        ("Propose", intent_present, "Trade intent exists." if intent_present else "No current trade intent."),
        ("Risk Gate", risk_approved, "Deterministic risk policy approved." if risk_approved else "Risk policy blocked or no intent."),
        ("Preflight", preflight_ready, "Execution preflight is ready." if preflight_ready else "Execution preflight is blocked."),
        ("Human Gate", human_gate_required, "Manual approval required before broker write." if human_gate_required else "No manual broker-write approval is currently armed."),
        ("Broker State", broker_connected, "Broker read path is connected." if broker_connected else "Broker read path is unavailable."),
        ("Review", True, "Audit, reconciliation, and post-trade review close the loop."),
    ]
    return [
        {
            "Stage": name,
            "Ready": ready,
            "Detail": detail,
        }
        for name, ready, detail in stages
    ]


def portfolio_story_records() -> list[dict]:
    return [
        {"Step": "1. Observe", "Portfolio Signal": "Market data and strategy rules produce a current state."},
        {"Step": "2. Propose", "Portfolio Signal": "The agent creates a structured trade thesis and intent."},
        {"Step": "3. Gate", "Portfolio Signal": "Deterministic risk policy can block the agent."},
        {"Step": "4. Approve", "Portfolio Signal": "Human-in-the-loop controls arm one exact preview hash."},
        {"Step": "5. Execute", "Portfolio Signal": "Only gated paper broker actions can reach Alpaca paper."},
        {"Step": "6. Reconcile", "Portfolio Signal": "Broker state refresh confirms what actually happened."},
        {"Step": "7. Learn", "Portfolio Signal": "Post-trade review and audit evidence close the loop."},
    ]
