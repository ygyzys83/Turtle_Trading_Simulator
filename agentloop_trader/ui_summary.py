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
            "Next Action": "Send the paper exit or let Auto exits handle it.",
            "Detail": f"{exit_preview_count} Alpaca paper position(s) can be sold by the app.",
        }
    if cancelable_order_count > 0:
        return {
            "State": "Cancel available",
            "Next Action": "Cancel the waiting paper order if you want to.",
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
            "Next Action": "Fix the blocked reasons before sending anything.",
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
            "Next Action": "Send the paper buy order to Alpaca.",
            "Detail": "The strategy, risk checks, and Alpaca paper order check all passed.",
        }
    if buy_preview_ready:
        return {
            "State": "Ready to send",
            "Next Action": "Send the paper buy order to Alpaca.",
            "Detail": "The strategy, risk checks, and Alpaca paper order check all passed.",
        }
    return {
        "State": "Trade blocked",
        "Next Action": "Fix the order blockers.",
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


def setup_scorecard_records(
    live: dict,
    risk_approved: bool,
    blocked_reasons: list[str] | None = None,
    enabled_inputs: dict[str, bool] | None = None,
    strategy_type: str | None = None,
) -> list[dict]:
    last_price = _as_float(live.get("last_p"))
    entry_level = _as_float(live.get("don_high"))
    exit_level = _as_float(live.get("don_low"))
    atr = _as_float(live.get("last_atr"))
    signal = str(live.get("signal", "flat")).lower()
    setup_type = str(live.get("setup_type") or strategy_type or "breakout").lower()
    trend_ok = bool(live.get("sma_up"))
    breakout_gap = last_price - entry_level if last_price and entry_level else 0.0
    stop_distance = _as_float(live.get("stop_from_entry"))
    atr_pct = atr / last_price * 100 if last_price else 0.0
    reward_proxy = (last_price - exit_level) / stop_distance if stop_distance > 0 and exit_level else 0.0
    blocks = [reason for reason in (blocked_reasons or []) if reason]
    enabled = enabled_inputs or {}

    if setup_type == "pullback":
        setup_ready = bool(live.get("pullback_ready")) or signal == "long"
        core_rows = [
            {
                "Read": "Trend",
                "Status": "Good" if trend_ok else "Weak",
                "Plain English": "Stock is above a rising trend filter." if trend_ok else "Trend filter does not support a buy.",
            },
            {
                "Read": "Pullback",
                "Status": _pullback_status(live.get("pullback_depth_pct")),
                "Plain English": _pullback_detail(live.get("pullback_depth_pct")),
            },
            {
                "Read": "Momentum turn",
                "Status": "Yes" if live.get("momentum_turn") else "No",
                "Plain English": "Price is turning back up after the pullback." if live.get("momentum_turn") else "No clear turn back up yet.",
            },
            {
                "Read": "Risk approval",
                "Status": "Passed" if risk_approved and not blocks else "Blocked",
                "Plain English": "Risk checks allow this idea." if risk_approved and not blocks else (blocks[0] if blocks else "Risk checks do not allow this idea."),
            },
        ]
    else:
        setup_ready = signal == "long" and breakout_gap > 0
        core_rows = [
            {
                "Read": "Trend",
                "Status": "Good" if trend_ok else "Weak",
                "Plain English": "Trend filter supports a buy." if trend_ok else "Trend filter does not support a buy.",
            },
            {
                "Read": "Breakout",
                "Status": "Triggered" if signal == "long" and breakout_gap > 0 else "Not triggered",
                "Plain English": _breakout_detail(breakout_gap),
            },
            {
                "Read": "Risk approval",
                "Status": "Passed" if risk_approved and not blocks else "Blocked",
                "Plain English": "Risk checks allow this idea." if risk_approved and not blocks else (blocks[0] if blocks else "Risk checks do not allow this idea."),
            },
        ]

    grade = _setup_grade(setup_ready=setup_ready, trend_ok=trend_ok, risk_ok=risk_approved and not blocks)
    rows = [
        {
            "Read": "Overall",
            "Status": grade,
            "Plain English": _grade_detail(grade, setup_type, blocks),
        },
    ]
    rows.extend(core_rows)

    optional_rows = {
        "breakout_strength": {
            "Read": "Breakout strength",
            "Status": _breakout_strength_status(breakout_gap, last_price),
            "Plain English": _breakout_detail(breakout_gap),
        },
        "volume": {
            "Read": "Volume",
            "Status": str(live.get("volume_status") or "Unknown"),
            "Plain English": _volume_detail(live.get("volume_status")),
        },
        "volatility": {
            "Read": "Volatility",
            "Status": f"{atr_pct:.2f}% ATR",
            "Plain English": _volatility_detail(atr_pct),
        },
        "room_above_exit": {
            "Read": "Room above exit",
            "Status": f"{reward_proxy:.1f}x stop" if reward_proxy > 0 else "n/a",
            "Plain English": "Price has room above the exit line." if reward_proxy > 1 else "Price is close to the exit line or no trade is active.",
        },
        "relative_strength": {
            "Read": "Relative strength",
            "Status": str(live.get("relative_strength") or "Unknown"),
            "Plain English": "Compares this ticker against the market benchmark.",
        },
        "market_condition": {
            "Read": "Market condition",
            "Status": str(live.get("market_condition") or "Unknown"),
            "Plain English": "Checks whether the broad market supports new long trades.",
        },
        "liquidity": {
            "Read": "Liquidity",
            "Status": str(live.get("liquidity_status") or "Unknown"),
            "Plain English": "Checks whether average dollar volume is large enough for clean fills.",
        },
        "event_risk": {
            "Read": "Event risk",
            "Status": str(live.get("event_risk") or "Unknown"),
            "Plain English": "Earnings/news calendar is not connected yet, so this stays informational.",
        },
        "rsi": {
            "Read": "RSI condition",
            "Status": str(live.get("rsi_status") or "Unknown"),
            "Plain English": _rsi_detail(live.get("rsi_status")),
        },
    }
    for key, row in optional_rows.items():
        if enabled.get(key, key in {"volatility", "room_above_exit"}):
            rows.append(row)
    return rows


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
        ("Find Trade", intent_present, "There is a trade idea." if intent_present else "No trade right now."),
        ("Check Risk", risk_approved, "Risk checks passed." if risk_approved else "Risk checks blocked the trade or there is no trade."),
        ("Ready To Send", preflight_ready, "The trade can be sent in paper mode." if preflight_ready else "The trade cannot be sent yet."),
        ("Your Action", human_gate_required, "Manual mode means you click the paper order button." if human_gate_required else "No paper action is waiting."),
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


def _overall_setup_detail(overall: str, blocks: list[str]) -> str:
    if overall == "Clean setup":
        return "Trade idea, trend, breakout, and risk check line up."
    if blocks:
        return blocks[0]
    if overall == "Needs review":
        return "The chart setup exists, but one check still needs attention."
    return "Wait. The current bar does not show a clean buy setup."


def _setup_grade(setup_ready: bool, trend_ok: bool, risk_ok: bool) -> str:
    if not setup_ready:
        return "No trade"
    if trend_ok and risk_ok:
        return "A"
    if risk_ok:
        return "B"
    return "C"


def _grade_detail(grade: str, setup_type: str, blocks: list[str]) -> str:
    if blocks:
        return blocks[0]
    if grade == "A":
        return "The selected strategy has a clean setup and risk checks pass."
    if grade == "B":
        return "The selected strategy has a setup, but one quality input is only okay."
    if grade == "C":
        return "The setup exists, but risk or quality checks need review."
    strategy = "trend pullback" if setup_type == "pullback" else "breakout"
    return f"No clean {strategy} setup right now."


def _breakout_detail(value: float) -> str:
    if value > 0:
        return f"Price is ${value:,.2f} above the breakout level."
    if value < 0:
        return f"Price is ${abs(value):,.2f} below the breakout level."
    return "Price is sitting on the breakout level."


def _breakout_strength_status(value: float, price: float) -> str:
    if price <= 0 or value <= 0:
        return "Not active"
    pct = value / price * 100
    if pct >= 2:
        return "Strong"
    if pct >= 0.5:
        return "Good"
    return "Thin"


def _pullback_status(value) -> str:
    depth = _as_float(value)
    if depth <= 0:
        return "Unknown"
    if depth <= 3:
        return "Shallow"
    if depth <= 8:
        return "Controlled"
    return "Deep"


def _pullback_detail(value) -> str:
    depth = _as_float(value)
    if depth <= 0:
        return "Pullback depth is not available."
    return f"Recent pullback is about {depth:.1f}% from the latest swing area."


def _volume_detail(status) -> str:
    if status == "Strong":
        return "Volume is above recent average."
    if status == "Normal":
        return "Volume is near recent average."
    if status == "Light":
        return "Volume is below recent average."
    return "Volume is unavailable for this data source."


def _volatility_detail(atr_pct: float) -> str:
    if atr_pct <= 0:
        return "Volatility is not available."
    if atr_pct < 1:
        return "Quiet movement; stops may be tighter."
    if atr_pct <= 3:
        return "Normal movement for this setup."
    return "Wide movement; position size should be smaller."


def _rsi_detail(status) -> str:
    if status == "Good":
        return "Momentum supports the setup without being extreme."
    if status == "Strong":
        return "Momentum is strong and close to extended."
    if status == "Extended":
        return "Momentum may be stretched; avoid chasing."
    if status == "Weak":
        return "Momentum does not support a long setup."
    return "RSI is unavailable."


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
