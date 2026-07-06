from __future__ import annotations


def operator_state_record(
    intent_present: bool,
    risk_approved: bool,
    preflight_ready: bool,
    execution_mode: str,
    broker_connected: bool,
    broker_state_stale: bool,
    alpaca_enabled: bool,
    buy_preview_ready: bool,
    buy_preview_armed: bool,
    exit_preview_count: int,
    cancelable_order_count: int,
    open_position_count: int,
) -> dict:
    if broker_state_stale:
        return {
            "State": "Refresh Alpaca",
            "Next Action": "Refresh Alpaca positions and orders.",
            "Detail": "Broker data is stale, so paper actions stay blocked until refreshed.",
        }
    if exit_preview_count > 0 and open_position_count > 0:
        return {
            "State": "Exit ready",
            "Next Action": "Review the paper exit order.",
            "Detail": f"{exit_preview_count} Alpaca paper position(s) can be reviewed for exit.",
        }
    if cancelable_order_count > 0:
        return {
            "State": "Cancel available",
            "Next Action": "Review the waiting paper order if you want to cancel it.",
            "Detail": f"{cancelable_order_count} Alpaca paper order(s) are waiting to fill.",
        }
    if not intent_present:
        return {
            "State": "No trade",
            "Next Action": "Wait for a valid strategy signal.",
            "Detail": "The strategy does not have a buy setup on the latest bar.",
        }
    if not risk_approved or not preflight_ready:
        return {
            "State": "Trade blocked",
            "Next Action": "Review the blocked reasons before doing anything.",
            "Detail": "The strategy found a trade idea, but risk or execution checks blocked it.",
        }
    if execution_mode != "paper":
        return {
            "State": "Review only",
            "Next Action": "Switch to Paper trading if you want to send a paper order.",
            "Detail": "The trade passed checks, but the selected mode does not send paper orders.",
        }
    if not broker_connected:
        return {
            "State": "Connect Alpaca",
            "Next Action": "Connect Alpaca paper before sending orders.",
            "Detail": "The trade passed checks, but broker account data is unavailable.",
        }
    if not alpaca_enabled:
        return {
            "State": "Paper orders off",
            "Next Action": "Turn on Allow Alpaca paper orders when ready.",
            "Detail": "Paper order submission is intentionally disabled in the sidebar.",
        }
    if buy_preview_ready and buy_preview_armed:
        return {
            "State": "Ready to send",
            "Next Action": "Confirm and send the reviewed paper buy order.",
            "Detail": "The exact paper order has been reviewed and still matches.",
        }
    if buy_preview_ready:
        return {
            "State": "Ready to review",
            "Next Action": "Review the paper buy order.",
            "Detail": "The trade passed checks and can be reviewed before sending.",
        }
    return {
        "State": "Trade blocked",
        "Next Action": "Review the order blockers.",
        "Detail": "The trade passed core checks, but the Alpaca paper order preview is blocked.",
    }


def strategy_context_records(live: dict, entry_window: int, exit_window: int, moving_average_window: int) -> list[dict]:
    last_price = _as_float(live.get("last_p"))
    entry_level = _as_float(live.get("don_high"))
    exit_level = _as_float(live.get("don_low"))
    atr = _as_float(live.get("last_atr"))
    signal = str(live.get("signal", "flat")).lower()
    breakout_gap = last_price - entry_level if last_price and entry_level else 0.0
    exit_gap = last_price - exit_level if last_price and exit_level else 0.0
    atr_pct = atr / last_price * 100 if last_price else 0.0
    return [
        {
            "Topic": "Signal",
            "Read": _display_signal(signal),
            "What It Means": _signal_detail(signal),
        },
        {
            "Topic": "Trend filter",
            "Read": "Uptrend" if live.get("sma_up") else "Not an uptrend",
            "What It Means": f"Uses the {moving_average_window}-bar moving average to avoid weak trend entries.",
        },
        {
            "Topic": "Breakout",
            "Read": _money_gap(breakout_gap),
            "What It Means": f"Price versus the {entry_window}-bar entry level.",
        },
        {
            "Topic": "Exit line",
            "Read": _money_gap(exit_gap),
            "What It Means": f"Price versus the {exit_window}-bar exit level.",
        },
        {
            "Topic": "Volatility",
            "Read": f"{atr_pct:.2f}% ATR",
            "What It Means": "Higher volatility increases stop distance and affects size.",
        },
    ]


def agent_decision_summary(
    intent_present: bool,
    thesis: str,
    blocked_reasons: list[str],
    next_action: str,
) -> str:
    if intent_present:
        return f"The agent found a trade idea. {thesis} Next: {next_action}"
    if blocked_reasons:
        first_reason = blocked_reasons[0]
        return f"The agent is waiting. Main reason: {first_reason} Next: {next_action}"
    return f"The agent is waiting for a clean setup. Next: {next_action}"


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


def _display_signal(signal: str) -> str:
    if signal == "long":
        return "Buy setup"
    if signal == "exit":
        return "Exit setup"
    return "No trade"


def _signal_detail(signal: str) -> str:
    if signal == "long":
        return "Price has cleared the entry rule and can move to risk checks."
    if signal == "exit":
        return "The strategy is focused on reducing exposure, not opening a new trade."
    return "The latest bar does not meet the entry rule."


def _money_gap(value: float) -> str:
    if value > 0:
        return f"${value:,.2f} above"
    if value < 0:
        return f"${abs(value):,.2f} below"
    return "$0.00"


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
