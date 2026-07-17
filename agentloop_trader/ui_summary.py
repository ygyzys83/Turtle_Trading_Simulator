from __future__ import annotations

from agentloop_trader.strategy_levels import build_buy_level_snapshot
from agentloop_trader.strategy_runtime import exit_mode_for_settings


def _order_status(value) -> str:
    text = str(value or "").strip().lower()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


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
            "What It Means": f"Uses the {moving_average_window}-bar trend filter length to avoid weak trend entries.",
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

    if setup_type == "rsi_scalp":
        setup_ready = signal == "long"
        setup_armed = bool(live.get("rsi_setup_armed")) or setup_ready
        rebound = _as_float(live.get("rsi_rebound_points"))
        required_rebound = _as_float(live.get("required_rsi_rebound_points")) or 3.0
        price_turn = bool(live.get("last_p") and live.get("prior_p") and _as_float(live.get("last_p")) > _as_float(live.get("prior_p")))
        requirements = list((live.get("buy_requirements") or {}).values())
        if len(requirements) >= 3:
            setup_armed, rebound_ok, price_turn = bool(requirements[0]), bool(requirements[1]), bool(requirements[2])
        else:
            rebound_ok = rebound >= required_rebound
        core_rows = [
            {
                "Read": "RSI setup",
                "Status": "Armed" if setup_armed else "Waiting",
                "Plain English": (
                    f"The setup low is RSI {float(live.get('rsi_setup_low')):.1f}."
                    if setup_armed and live.get("rsi_setup_low") is not None
                    else "RSI has not reached the arm level or fallen far enough from its recent high."
                ),
            },
            {
                "Read": "RSI rebound",
                "Status": "Yes" if rebound_ok else "No",
                "Plain English": f"RSI has rebounded {rebound:.1f} points from the setup low; {required_rebound:g} are required.",
            },
            {
                "Read": "Price turn",
                "Status": "Yes" if price_turn else "No",
                "Plain English": "Price closed above the prior completed bar." if price_turn else "Price has not closed above the prior completed bar.",
            },
            {
                "Read": "Risk approval",
                "Status": "Passed" if risk_approved and not blocks else "Blocked",
                "Plain English": "Risk checks allow this idea." if risk_approved and not blocks else (blocks[0] if blocks else "Risk checks do not allow this idea."),
            },
        ]
    elif setup_type in {"trendline", "trendline_retest"}:
        trendline_level = _as_float(live.get("trendline_level"))
        breakout_level = _as_float(live.get("trendline_breakout_level"))
        trendline_break = bool(live.get("trendline_break"))
        retest_ready = bool(live.get("retest_ready"))
        quality = str(live.get("trendline_quality") or "No valid line")
        setup_ready = signal == "long"
        core_rows = [
            {
                "Read": "Trend",
                "Status": "Good" if trend_ok else "Weak",
                "Plain English": "Stock is above a rising trend filter." if trend_ok else "Trend filter does not support a buy.",
            },
            {
                "Read": "Trendline",
                "Status": "Found" if trendline_level > 0 else "Not found",
                "Plain English": (
                    f"The selected descending trendline is near ${trendline_level:,.2f}; quality: {quality.lower()}."
                    if trendline_level > 0
                    else "No valid touch-scored descending trendline was found."
                ),
            },
            {
                "Read": "Completed-close breakout",
                "Status": "Yes" if trendline_break else "No",
                "Plain English": (
                    f"A completed bar crossed above the required ${breakout_level:,.2f} level."
                    if trendline_break and breakout_level > 0
                    else (
                        f"Waiting for a completed bar to cross above ${breakout_level:,.2f}; this includes the 0.10 ATR confirmation buffer."
                        if breakout_level > 0
                        else "A valid line is required before an exact breakout price can be calculated."
                    )
                ),
            },
        ]
        if setup_type == "trendline_retest":
            core_rows.append(
                {
                    "Read": "Retest",
                    "Status": "Yes" if retest_ready else "No",
                    "Plain English": "Price retested the broken trendline and turned up." if retest_ready else "Waiting for a retest and turn back up.",
                }
            )
        core_rows.append(
            {
                "Read": "Risk approval",
                "Status": "Passed" if risk_approved and not blocks else "Blocked",
                "Plain English": "Risk checks allow this idea." if risk_approved and not blocks else (blocks[0] if blocks else "Risk checks do not allow this idea."),
            }
        )
    elif setup_type == "pullback":
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

    grade = _setup_grade(
        setup_ready=setup_ready,
        trend_ok=True if setup_type == "rsi_scalp" else trend_ok,
        risk_ok=risk_approved and not blocks,
    )
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
            "Plain English": _volatility_detail(
                atr_pct,
                atr_dollars=atr,
                stop_distance=stop_distance,
                stop_price=last_price - stop_distance if last_price and stop_distance else None,
            ),
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


def buy_requirement_records(live: dict, interval: str = "", latest_price: float | None = None) -> list[dict]:
    return build_buy_level_snapshot(live, interval=interval, latest_price=latest_price)["records"]


def optional_quality_input_records(enabled_inputs: dict[str, bool] | None = None) -> list[dict]:
    enabled = enabled_inputs or {}
    descriptions = {
        "breakout_strength": "Shows whether the breakout has enough distance above the entry level.",
        "volume": "Shows whether the move is supported by normal or strong volume.",
        "volatility": "Shows whether movement is quiet, normal, or wide.",
        "room_above_exit": "Shows whether the setup has enough room above the exit line.",
        "relative_strength": "Compares this ticker against the market benchmark when connected.",
        "market_condition": "Checks whether the broad market supports new long trades when connected.",
        "liquidity": "Checks whether the ticker trades enough dollar volume for clean fills.",
        "event_risk": "Flags earnings or news risk when an event calendar is connected.",
        "rsi": "Always shows current RSI. It becomes a required BUY rule only when Require RSI 50-70 for BUY is on.",
    }
    labels = {
        "breakout_strength": "Breakout strength",
        "volume": "Volume",
        "volatility": "Volatility",
        "room_above_exit": "Room above exit",
        "relative_strength": "Relative strength",
        "market_condition": "Market condition",
        "liquidity": "Liquidity",
        "event_risk": "Event risk",
        "rsi": "RSI",
    }

    rows = []
    for key, description in descriptions.items():
        if enabled.get(key, key in {"volatility", "room_above_exit"}):
            rows.append(
                {
                    "Quality Input": labels[key],
                    "Creates BUY?": "Optional rule" if key == "rsi" else "No",
                    "Plain English": description,
                }
            )
    return rows


def sell_requirement_records(live: dict, exit_preview_count: int = 0, exit_settings_saved: bool | None = None) -> list[dict]:
    requirements = dict(live.get("sell_requirements") or {})
    requirements["Alpaca paper position can be sold"] = exit_preview_count > 0
    if exit_settings_saved is not None:
        requirements["Saved exit settings are available"] = exit_settings_saved

    rows = []
    for rule, passed in requirements.items():
        rows.append(
            {
                "Required SELL Rule": str(rule),
                "Status": "Pass" if passed else "Not met",
                "Plain English": "This must pass before the app can send an automatic paper sell.",
            }
        )
    return rows


def optional_sell_quality_records(enabled_inputs: dict[str, bool] | None = None) -> list[dict]:
    rows = optional_quality_input_records(enabled_inputs)
    return [
        {
            "Quality Input": row["Quality Input"],
            "Creates SELL?": "No",
            "Plain English": row["Plain English"],
        }
        for row in rows
    ]


def managed_position_records(position_records: list[dict], exit_settings_by_symbol: dict[str, dict] | None = None) -> list[dict]:
    exit_settings_by_symbol = exit_settings_by_symbol or {}
    rows = []
    for position in position_records:
        symbol = str(position.get("Symbol", "")).strip().upper()
        settings = exit_settings_by_symbol.get(symbol, {})
        auto_exit_enabled = bool(settings.get("auto_exit_enabled", False)) if settings else False
        exit_mode = exit_mode_for_settings(settings)
        strategy = str(settings.get("strategy_label") or settings.get("strategy_type") or "Not saved")
        exit_method = "ATR protection only" if exit_mode == "atr_only" else f"{strategy} + ATR protection"
        rows.append(
            {
                "Symbol": symbol,
                "Quantity": position.get("Quantity", ""),
                "Market Value": _money(_as_float(position.get("Market Value"))),
                "Avg Entry": _money(_as_float(position.get("Average Entry"))),
                "Auto Exit": "On" if auto_exit_enabled else "Off",
                "Exit Method": exit_method,
                "Management Status": "Managed" if settings else "Needs exit settings",
            }
        )
    return rows


def _bars(value: object) -> str:
    try:
        if value in (None, ""):
            return "Not saved"
        return f"{int(value)} bars"
    except (TypeError, ValueError):
        return "Not saved"


def position_exit_plan_records(
    settings: dict | None,
    exit_ready: bool = False,
    exit_reason: str = "",
    exit_trigger_price: float | None = None,
) -> list[dict]:
    if not settings:
        return [
            {"Field": "Exit Settings", "Value": "Not saved"},
            {"Field": "Auto Exit", "Value": "Off"},
            {"Field": "Current Exit Check", "Value": exit_reason or "No saved exit settings for this position."},
        ]
    exit_mode = exit_mode_for_settings(settings)
    strategy = str(settings.get("strategy_label") or settings.get("strategy_type") or "Unknown")
    strategy_type = str(settings.get("strategy_type", ""))
    rows = [
        {"Field": "Exit Method", "Value": "ATR protection only" if exit_mode == "atr_only" else "Strategy exit + ATR protection"},
        {"Field": "Auto Exit", "Value": "On" if settings.get("auto_exit_enabled", False) else "Off"},
        {
            "Field": "Auto Exit Trigger Price",
            "Value": _money(exit_trigger_price) if exit_trigger_price else "Not available",
        },
        {"Field": "Price Interval", "Value": str(settings.get("interval", "Not saved"))},
        {"Field": "ATR Stop Distance", "Value": str(settings.get("atr_stop_multiplier", "Not saved"))},
    ]
    if exit_mode == "atr_only":
        pass
    elif strategy_type == "rsi_scalp":
        rows.append({"Field": "Exit Strategy", "Value": strategy})
        rows.extend([
            {"Field": "RSI Length", "Value": _bars(settings.get("rsi_length"))},
            {"Field": "RSI Setup Low At Entry", "Value": str(settings.get("entry_rsi_setup_low", "Not recorded"))},
            {"Field": "Maximum RSI Rebound Allowed For Buy", "Value": f"{settings.get('rsi_max_rebound_points', 12)} points"},
            {"Field": "Sell After RSI Recovery", "Value": f"{settings.get('rsi_sell_recovery_points', 35)} points"},
            {"Field": "RSI Sell Cap", "Value": str(settings.get("rsi_overbought", 70))},
            {"Field": "Stop Protection", "Value": str(settings.get("rsi_stop_mode", "standard_atr")).replace("_", " ").title()},
            {
                "Field": "Emergency Stop Distance",
                "Value": (
                    f"{settings.get('rsi_emergency_atr_multiplier', 5.0)} ATR"
                    if settings.get("rsi_stop_mode") == "emergency_atr"
                    else "Not used"
                ),
            },
            {
                "Field": "Maximum Holding Period",
                "Value": (
                    _bars(settings.get("rsi_max_holding_bars", 100))
                    if settings.get("rsi_max_holding_enabled", True)
                    else "Off"
                ),
            },
        ])
    else:
        rows.append({"Field": "Exit Strategy", "Value": strategy})
        rows.extend([
            {"Field": "Sell Exit Length", "Value": _bars(settings.get("exit_window"))},
            {"Field": "Trend Filter Length", "Value": _bars(settings.get("moving_average_window"))},
            {"Field": "Pullback Average Length", "Value": _bars(settings.get("pullback_average_length"))},
            {"Field": "Momentum Turn Length", "Value": _bars(settings.get("momentum_turn_length"))},
        ])
    rows.extend([
        {"Field": "Move Stop To Break-Even After", "Value": f"+{settings.get('breakeven_after_r', 1.0)}R"},
        {"Field": "Start ATR Trail After", "Value": f"+{settings.get('trail_after_r', 2.0)}R"},
        {"Field": "Trailing ATR Distance", "Value": str(settings.get("trailing_atr_multiplier", "Not saved"))},
        {"Field": "Current Exit Check", "Value": "Exit now" if exit_ready else (exit_reason or "Hold")},
    ])
    return rows


def trade_evidence_summary_records(
    intent_present: bool,
    risk_approved: bool,
    preflight_ready: bool,
    setup_rows: list[dict] | None = None,
    blocked_reasons: list[str] | None = None,
) -> list[dict]:
    setup_rows = setup_rows or []
    blocked_reasons = [reason for reason in (blocked_reasons or []) if reason]
    setup_grade = next((str(row.get("Status", "")) for row in setup_rows if row.get("Read") == "Overall"), "Unknown")
    return [
        {
            "Evidence Area": "Setup quality",
            "Current Read": setup_grade,
            "What To Use It For": "Quickly judge whether the selected ticker has a clean setup.",
        },
        {
            "Evidence Area": "Trade idea",
            "Current Read": "Present" if intent_present else "None",
            "What To Use It For": "Shows whether the strategy generated a possible BUY.",
        },
        {
            "Evidence Area": "Risk check",
            "Current Read": "Passed" if risk_approved else "Blocked",
            "What To Use It For": "Confirms whether sizing and risk limits allow the idea.",
        },
        {
            "Evidence Area": "Ready to send",
            "Current Read": "Yes" if preflight_ready else "No",
            "What To Use It For": "Confirms whether the trade can move to paper execution.",
        },
        {
            "Evidence Area": "Main blocker",
            "Current Read": blocked_reasons[0] if blocked_reasons else "None",
            "What To Use It For": "First issue to fix or understand before taking action.",
        },
    ]


def alpaca_evidence_summary_records(
    alpaca_connected: bool,
    alpaca_state_stale: bool,
    paper_orders_enabled: bool,
    alpaca_positions: list[dict] | None = None,
    alpaca_orders: list[dict] | None = None,
    tracked_orders: list[dict] | None = None,
    automation_status: str = "",
) -> list[dict]:
    positions = alpaca_positions or []
    orders = alpaca_orders or []
    tracked = tracked_orders or []
    waiting_orders = [
        order for order in orders
        if _order_status(order.get("Status", order.get("status", ""))) in {"accepted", "new", "pending_new", "partially_filled"}
    ]
    return [
        {
            "Evidence Area": "Broker connection",
            "Current Read": "Connected" if alpaca_connected else "Disconnected",
            "What To Use It For": "Confirms whether the app can read Alpaca paper account data.",
        },
        {
            "Evidence Area": "Broker data",
            "Current Read": "Needs refresh" if alpaca_state_stale else "Current",
            "What To Use It For": "Tells you whether position/order reads are fresh enough to trust.",
        },
        {
            "Evidence Area": "Paper order switch",
            "Current Read": "On" if paper_orders_enabled else "Off",
            "What To Use It For": "Confirms whether Alpaca paper writes are allowed.",
        },
        {
            "Evidence Area": "Open positions",
            "Current Read": len(positions),
            "What To Use It For": "Shows how many Alpaca paper positions need management.",
        },
        {
            "Evidence Area": "Waiting orders",
            "Current Read": len(waiting_orders),
            "What To Use It For": "Shows how many Alpaca paper orders are still waiting to fill.",
        },
        {
            "Evidence Area": "Saved order records",
            "Current Read": len(tracked),
            "What To Use It For": "Shows how many Alpaca paper orders the app is tracking locally.",
        },
        {
            "Evidence Area": "Automation",
            "Current Read": automation_status or "Not checked",
            "What To Use It For": "Shows the current automation state without reading every detail table.",
        },
    ]


def saved_records_overview_records(
    audit_records: list[dict] | None = None,
    tracked_orders: list[dict] | None = None,
    automation_snapshots: list[dict] | None = None,
) -> list[dict]:
    audit_records = audit_records or []
    tracked_orders = tracked_orders or []
    automation_snapshots = automation_snapshots or []
    broker_orders = [
        order for order in tracked_orders
        if str(order.get("source") or "").strip().lower()
        not in {"position_plan", "position_observation"}
    ]
    event_types = [str(record.get("event_type", "")) for record in audit_records]
    return [
        {
            "Record Set": "Activity log",
            "Count": len(audit_records),
            "Why It Matters": "Timeline of what the app decided, blocked, or sent.",
        },
        {
            "Record Set": "Alpaca paper orders",
            "Count": len(broker_orders),
            "Why It Matters": "Local record of paper orders submitted, canceled, filled, or reconciled.",
        },
        {
            "Record Set": "Automation checks",
            "Count": len(automation_snapshots),
            "Why It Matters": "Saved snapshots of what automation saw before taking or skipping action.",
        },
        {
            "Record Set": "Paper buys sent",
            "Count": event_types.count("alpaca_paper_order_submitted") + event_types.count("auto_paper_entry_submitted") + event_types.count("worker_paper_buy_sent"),
            "Why It Matters": "Confirms how many paper buys reached Alpaca.",
        },
        {
            "Record Set": "Paper exits sent",
            "Count": event_types.count("alpaca_paper_exit_submitted") + event_types.count("auto_paper_exit_submitted") + event_types.count("worker_paper_exit_sent"),
            "Why It Matters": "Confirms how many paper exits reached Alpaca.",
        },
    ]


def no_buy_reason(live: dict) -> str:
    reason = str(live.get("no_trade_reason") or "").strip()
    if reason:
        return reason
    if live.get("trade_intent") is not None:
        return "BUY intent is present."
    return "No BUY because the selected strategy rules are not fully met."


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
    if setup_type == "pullback":
        strategy = "trend pullback"
    elif setup_type == "trendline_retest":
        strategy = "trendline retest"
    elif setup_type == "trendline":
        strategy = "trendline breakout"
    elif setup_type == "rsi_scalp":
        strategy = "RSI mean-reversion scalp"
    else:
        strategy = "breakout"
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


def _volatility_detail(
    atr_pct: float,
    atr_dollars: float | None = None,
    stop_distance: float | None = None,
    stop_price: float | None = None,
) -> str:
    if atr_pct <= 0:
        return "Volatility is not available."
    atr_text = f"ATR is ${atr_dollars:,.2f}." if atr_dollars and atr_dollars > 0 else ""
    stop_text = ""
    if stop_distance and stop_distance > 0:
        stop_text = f" Stop distance is ${stop_distance:,.2f}."
    if stop_price and stop_price > 0:
        stop_text = f"{stop_text} Approx stop is ${stop_price:,.2f}."
    if atr_pct < 1:
        return f"Quiet movement; stops may be tighter. {atr_text}{stop_text}".strip()
    if atr_pct <= 3:
        return f"Normal movement for this setup. {atr_text}{stop_text}".strip()
    return f"Wide movement; position size should be smaller. {atr_text}{stop_text}".strip()


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


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
