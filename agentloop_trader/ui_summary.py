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
        {"Status": "Mode", "Value": mode_label, "State": "info"},
        {"Status": "Trade Check", "Value": "passed" if risk_approved else "blocked", "State": "ok" if risk_approved else "block"},
        {"Status": "Alpaca", "Value": "connected" if broker_connected else "disconnected", "State": "ok" if broker_connected else "block"},
        {"Status": "Alpaca Data", "Value": "needs refresh" if broker_state_stale else "current", "State": "block" if broker_state_stale else "ok"},
        {"Status": "Stop Trading", "Value": "on" if kill_switch_enabled else "off", "State": "block" if kill_switch_enabled else "ok"},
        {"Status": "Live Orders", "Value": "blocked" if live_writes_blocked else "available", "State": "ok" if live_writes_blocked else "block"},
    ]


def agent_loop_stage_records(
    intent_present: bool,
    risk_approved: bool,
    preflight_ready: bool,
    human_gate_required: bool,
    broker_connected: bool,
) -> list[dict]:
    stages = [
        ("Read Prices", True, "Market data loaded and strategy rules checked."),
        ("Find Trade", intent_present, "There is a trade to review." if intent_present else "No trade to review right now."),
        ("Check Risk", risk_approved, "Risk checks passed." if risk_approved else "Risk checks blocked the trade or there is no trade."),
        ("Ready To Send", preflight_ready, "The trade can be reviewed for sending." if preflight_ready else "The trade cannot be sent yet."),
        ("Your Review", human_gate_required, "You must review before any broker action." if human_gate_required else "No broker action is ready."),
        ("Alpaca", broker_connected, "Alpaca is connected." if broker_connected else "Alpaca is not connected."),
        ("Learn", True, "The app records actions and reviews closed trades."),
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
        {"Step": "1. Read", "Portfolio Signal": "The app reads prices and checks the strategy rules."},
        {"Step": "2. Suggest", "Portfolio Signal": "The agent explains a trade idea and a proposed order."},
        {"Step": "3. Check", "Portfolio Signal": "Risk rules can block the trade."},
        {"Step": "4. Review", "Portfolio Signal": "The operator reviews one exact paper order before sending."},
        {"Step": "5. Send", "Portfolio Signal": "Only reviewed paper orders can reach Alpaca paper."},
        {"Step": "6. Refresh", "Portfolio Signal": "The app refreshes Alpaca to see what happened."},
        {"Step": "7. Learn", "Portfolio Signal": "Closed trades are reviewed so the system can improve."},
    ]
