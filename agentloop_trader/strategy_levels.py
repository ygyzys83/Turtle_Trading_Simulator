from __future__ import annotations

from typing import Any


def _number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _money(value: Any) -> str:
    number = _number(value)
    return f"${number:,.2f}" if number is not None else "Not available"


def _distance(current: float | None, target: float | None) -> str:
    if current is None or target is None or current == 0:
        return "Depends on rule"
    return f"{(target - current) / current * 100:+.2f}%"


def _status(requirements: dict[str, Any], text: str) -> str:
    match = next((bool(value) for key, value in requirements.items() if text.lower() in str(key).lower()), None)
    return "Pass" if match else "Not met" if match is not None else "Unknown"


def build_buy_level_snapshot(
    live: dict[str, Any] | None,
    *,
    interval: str = "",
    latest_price: float | None = None,
) -> dict[str, Any]:
    """Explain a strategy's live BUY requirements without inventing a price for stateful rules."""
    live = dict(live or {})
    requirements = dict(live.get("buy_requirements") or {})
    current = _number(latest_price)
    if current is None:
        current = _number(live.get("last_p"))
    setup_type = str(live.get("setup_type") or "").strip().lower()
    timeframe = interval or str(live.get("interval") or "the saved interval")
    rows: list[dict[str, str]] = []
    actionable_target: float | None = None

    def add(rule: str, current_value: str, required: str, status: str, plain: str, target: float | None = None) -> None:
        nonlocal actionable_target
        rows.append({
            "Required BUY Rule": rule,
            "Current": current_value,
            "Required": required,
            "Distance": _distance(current, target),
            "Status": status,
            "Plain English": plain,
        })
        if target is not None and status != "Pass" and actionable_target is None:
            actionable_target = target

    trend_level = _number(live.get("trend_filter_level"))
    if trend_level is None:
        trend_level = _number(live.get("last_sma"))

    if setup_type == "breakout":
        entry = _number(live.get("entry_level"))
        add(
            "Breakout price",
            _money(current),
            f"Completed {timeframe} bar closes above {_money(entry)}",
            _status(requirements, "bar high"),
            "This is the exact prior-high level the completed bar must clear.",
            entry,
        )
        add(
            "Trend filter",
            f"Price {_money(current)}; filter {_money(trend_level)}",
            f"Price above the rising {live.get('trend_window', '')}-bar filter".replace("  ", " "),
            _status(requirements, "trend filter"),
            "Price must be above the filter and the filter must be rising.",
            trend_level,
        )
    elif setup_type == "pullback":
        pullback = _number(live.get("pullback_average_level"))
        zone_high = _number(live.get("pullback_zone_high"))
        recent_low = _number(live.get("recent_pullback_low"))
        momentum = _number(live.get("momentum_average_level"))
        prior = _number(live.get("prior_p"))
        momentum_target = max([value for value in (momentum, prior) if value is not None], default=None)
        add("Trend filter", f"Price {_money(current)}; filter {_money(trend_level)}", "Price above a rising trend filter", _status(requirements, "trend filter"), "The broader trend must still be up.", trend_level)
        add("Pullback zone", f"Recent low {_money(recent_low)}", f"Recent low at or below {_money(zone_high or pullback)}", _status(requirements, "pullback touched"), "The recent pullback must reach the saved moving-average zone. This is path-dependent, not a single current-price trigger.")
        add("Momentum turn", f"Price {_money(current)}; short average {_money(momentum)}", f"Completed {timeframe} bar above {_money(momentum_target)} while the short average is rising", _status(requirements, "momentum turned"), "After the pullback, price and the short average must turn back up.", momentum_target)
    elif setup_type in {"trendline", "trendline_retest"}:
        line = _number(live.get("trendline_level"))
        breakout = _number(live.get("trendline_breakout_level"))
        quality = str(live.get("trendline_quality") or "No valid line")
        add(
            "Descending trendline",
            quality,
            "Two meaningful descending swing-high anchors with no completed-close violation",
            _status(requirements, "touch-scored descending trendline"),
            "The app compares recent descending lines and selects the one with the best confirmed touches, span, and recency.",
        )
        add(
            "Completed-close breakout",
            _money(current),
            f"Completed {timeframe} bar closes above {_money(breakout)}",
            _status(requirements, "buffered trendline") if setup_type == "trendline" else _status(requirements, "prior completed bar"),
            f"The trendline itself is {_money(line)}. The required close adds a 0.10 ATR buffer so a one-cent move above the line does not count.",
            breakout,
        )
        if setup_type == "trendline_retest":
            add("Retest and turn up", str(live.get("retest_description") or "Current pattern"), "Broken line is retested and momentum turns up", _status(requirements, "retest held"), "This requires a sequence of bars, so one exact BUY price would be misleading.")
        add("Trend filter", f"Price {_money(current)}; filter {_money(trend_level)}", "Price above a flat or rising trend filter", "Pass" if bool(live.get("trend_ok")) else "Not met", "The broader trend must support the setup.", trend_level)
    elif setup_type == "rsi_scalp":
        rsi = _number(live.get("rsi"))
        low = _number(live.get("rsi_setup_low"))
        rebound = _number(live.get("rsi_rebound_points"))
        required_rebound = _number(live.get("required_rsi_rebound_points"))
        maximum_rebound = _number(live.get("rsi_max_rebound_points"))
        prior = _number(live.get("prior_p"))
        add("Arm RSI setup", f"RSI {rsi:.1f}" if rsi is not None else "RSI unavailable", f"RSI reaches {live.get('rsi_oversold', 30)} or falls {live.get('required_rsi_decline_points', 40)} points", _status(requirements, "RSI("), "This setup is driven by RSI, not one fixed stock price.")
        add("RSI rebound", f"{rebound:.1f} points from {low:.1f}" if rebound is not None and low is not None else "Not armed", f"At least {required_rebound:g} points" if required_rebound is not None else "Saved rebound", _status(requirements, "RSI rebounded"), "RSI must recover from the setup low.")
        add(
            "Maximum RSI rebound",
            f"{rebound:.1f} points" if rebound is not None else "Waiting for setup",
            f"No more than {maximum_rebound:g} points" if maximum_rebound is not None else "Saved maximum",
            _status(requirements, "stayed at or below"),
            "If completed-bar RSI rebounds farther than this before the buy, the setup expires so the app does not chase a late entry.",
        )
        add("Price confirmation", _money(current), f"Completed {timeframe} bar above prior close {_money(prior)}", _status(requirements, "prior completed bar"), "Price must also turn upward.", prior)

    generic_requirements = not rows and bool(requirements)
    if generic_requirements:
        for rule, passed in requirements.items():
            add(str(rule), "See live strategy", "Rule must pass", "Pass" if passed else "Not met", "This must pass before the app can create a BUY intent.")

    for rule, passed in requirements.items() if not generic_requirements else []:
        if "position size" in str(rule).lower():
            add("Position size", str(live.get("pos_size", "Not available")), "Greater than zero after risk limits", "Pass" if passed else "Not met", "The calculated order must remain large enough to send.")

    if live.get("rsi_entry_filter_enabled"):
        rsi_rule = next(((key, value) for key, value in requirements.items() if "rsi" in str(key).lower()), None)
        if rsi_rule:
            add("Optional RSI entry rule", f"RSI {_number(live.get('rsi')):.1f}" if _number(live.get("rsi")) is not None else "RSI unavailable", str(rsi_rule[0]), "Pass" if rsi_rule[1] else "Not met", "This rule is required only because it was enabled for this setup.")

    passed = sum(row["Status"] == "Pass" for row in rows)
    total = len(rows)
    if setup_type in {"trendline", "trendline_retest"} and _number(live.get("trendline_level")) is None:
        # A moving-average level is not a substitute for a trendline that does not exist.
        actionable_target = None
    return {
        "records": rows or [{
            "Required BUY Rule": "Selected strategy rules",
            "Current": "Unknown",
            "Required": "Rule details unavailable",
            "Distance": "Unknown",
            "Status": "Unknown",
            "Plain English": "The strategy did not provide a rule-by-rule BUY explanation.",
        }],
        "next_buy_level": actionable_target,
        "distance_to_buy_pct": ((actionable_target - current) / current * 100) if actionable_target is not None and current else None,
        "rules_passed": passed,
        "rules_total": total,
    }
