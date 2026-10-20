"""Review-only rectangle drafts and frozen, Daily execution rules.

This module never sends orders. A draft is not a durable Buy queue record.
"""
from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np
import pandas as pd

from agentloop_trader.assets import floor_quantity
from agentloop_trader.buy_watchlist import BuyWatchPlan, buy_watch_plan_id
from agentloop_trader.indicators import calc_atr
from agentloop_trader.models import RiskLimits, TradeIntent
from agentloop_trader.rectangle_scanner import refresh_saved_rectangle
from agentloop_trader.rectangle_timeframes import completed_daily_bars, latest_completed_date, session_schedule, utc_now
from agentloop_trader.risk import constrain_trade_intent_to_limits


RECTANGLE_STRATEGY = "Rectangle breakout"
BUFFER_ATR = .10
MAX_CHASE_ATR = .50
PLAN_DAYS = 30


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return value


def _recent_complete(frame, now):
    schedule = session_schedule(frame.index[0].year,max(frame.index[-1].year,now.year))
    expected = schedule.index[schedule["close"] <= now][-16:]
    return frame.index[-16:].equals(expected)


def prepare_rectangle_trade(rectangle, anchor_close, research_source, research_history,
                            execution_history, *, defaults=None, risk_limits=None, now=None):
    """Freeze a reviewed box and verify it against fresh Alpaca daily history."""
    now = utc_now(now)
    frame = completed_daily_bars(execution_history, now)
    if frame.index[-1] != latest_completed_date("1d", now) or not _recent_complete(frame,now):
        raise ValueError("Alpaca daily history is stale. Refresh the trade preview before saving.")
    current, _ = refresh_saved_rectangle(rectangle, anchor_close, research_history, now=now)
    anchor_date = pd.Timestamp(rectangle.end)
    if anchor_date not in frame.index:
        raise ValueError("The original rectangle's anchor is missing from Alpaca history. Refresh research first.")
    alpaca_anchor = _positive(frame.loc[anchor_date, "Close"], "Alpaca anchor")
    # Reconcile the research chart's current price basis before freezing it. Do
    # not silently translate a materially different Yahoo/Alpaca level.
    reference_bars = completed_daily_bars(research_history, now)
    research_anchor = float(reference_bars.loc[anchor_date, "Close"])
    if abs(alpaca_anchor/research_anchor-1) > .01:
        raise ValueError("Research and Alpaca price bases differ by more than 1%. Refresh the research scan before preparing this trade.")
    atrs = calc_atr(frame.Close.to_numpy(), 14, frame.High.to_numpy(), frame.Low.to_numpy())
    atr = _positive(atrs[-1], "Daily ATR")
    trigger = math.ceil((current.resistance+BUFFER_ATR*atr)*100-1e-9)/100
    maximum = math.floor((trigger+MAX_CHASE_ATR*atr)*100+1e-9)/100
    if maximum <= trigger:
        raise ValueError("Daily ATR is too small for a one-cent entry band at this trigger.")
    context = {
        "version": 1, "symbol": rectangle.symbol, "discovery_interval": rectangle.interval,
        "research_source": research_source, "start": rectangle.start, "end": rectangle.end,
        "support": current.support, "resistance": current.resistance,
        "anchor_date": rectangle.end, "anchor_close": alpaca_anchor,
        "review_atr": atr, "trigger": trigger, "maximum_entry": maximum,
        "reviewed_at": now.isoformat(), "review_data_date": frame.index[-1].strftime("%Y-%m-%d"),
    }
    settings = dict(defaults or {})
    settings.update(symbol=rectangle.symbol, asset_class="equity", interval="1d", history="5y",
                    price_data_source="Ticker (Alpaca)", rectangle_context=context,
                    rectangle_armed_at="", strategy_type="rectangle", strategy_label=RECTANGLE_STRATEGY)
    for name, value in {"account_size":100_000., "risk_per_trade_pct":.5, "atr_stop_multiplier":2.,
                        "exit_window":10, "entry_window":20, "moving_average_window":50,
                        "profit_protection_enabled":True, "breakeven_after_r":1., "trail_after_r":2.,
                        "trailing_atr_multiplier":3., "confirm_exit_on_bar_close":True}.items():
        settings.setdefault(name, value)
    limits = risk_limits if isinstance(risk_limits, RiskLimits) else RiskLimits(**(risk_limits or {}))
    return {"settings":settings, "risk_limits":asdict(limits), "frame":frame,
            "research_rectangle":current.record()}


def preview_rectangle_size(settings, limits):
    context = settings["rectangle_context"]
    entry = _positive(context["trigger"], "Breakout trigger")
    distance = _positive(context["review_atr"], "Review ATR") * _positive(settings["atr_stop_multiplier"], "Stop multiplier")
    equity = _positive(settings["account_size"], "Review account size")
    risk = _positive(settings["risk_per_trade_pct"], "Risk per trade")
    if entry-distance <= 0:
        raise ValueError("The ATR stop would be zero or negative. Reduce the stop multiplier.")
    intent = TradeIntent(symbol=context["symbol"], side="buy", quantity=floor_quantity(equity*risk/100/distance,"equity"),
                         entry_price=entry, stop_loss=round(entry-distance,2), order_type="limit", limit_price=entry)
    return constrain_trade_intent_to_limits(intent, equity, limits)


def build_rectangle_buy_plan(draft, *, strategy_label=RECTANGLE_STRATEGY, strategy_type="rectangle", now=None):
    """Build a plan only on the explicit Save action; this function does not persist it."""
    now = utc_now(now)
    from agentloop_trader.strategy_runtime import STRATEGY_TYPES
    if {RECTANGLE_STRATEGY:"rectangle",**STRATEGY_TYPES}.get(strategy_label) != strategy_type:
        raise ValueError("Choose one of the supported entry strategies.")
    settings = dict(draft["settings"])
    context = dict(settings["rectangle_context"])
    symbol = context["symbol"]
    _positive(settings["account_size"], "Review account size")
    risk = _positive(settings["risk_per_trade_pct"], "Risk per trade")
    _positive(settings["atr_stop_multiplier"], "Stop multiplier")
    limits = RiskLimits(**draft["risk_limits"])
    if risk > limits.max_risk_per_trade_pct:
        raise ValueError("Risk per trade exceeds the saved risk limit.")
    if limits.allowed_symbols and symbol not in limits.allowed_symbols:
        raise ValueError("This ticker is not in the saved risk policy's allowed-symbol list.")
    if strategy_type == "rsi_scalp" and settings.get("rsi_stop_mode") == "no_price_stop":
        raise ValueError("The RSI no-price-stop mode is backtest-only and cannot be queued.")
    frame = draft["frame"]
    if now-pd.Timestamp(context["reviewed_at"]) > pd.Timedelta(minutes=15):
        raise ValueError("Trade preview is over 15 minutes old. Create a fresh trade plan before saving.")
    if frame.index[-1] != latest_completed_date("1d", now):
        raise ValueError("A new daily candle has completed. Create a fresh trade plan before saving.")
    if strategy_type == "rectangle":
        if not context["support"] <= float(frame.Close.iloc[-1]) <= context["trigger"]:
            raise ValueError("Price has already left this saved range. Refresh research or choose another existing strategy.")
        estimate = preview_rectangle_size(settings, limits)
        if estimate is None or estimate.quantity <= 0:
            raise ValueError("The saved risk limits allow no position at the reviewed trigger.")
    settings.update(strategy_label=strategy_label, strategy_type=strategy_type,
                    rectangle_armed_at=now.isoformat(), rectangle_context=context,
                    auto_exit_enabled=True, exit_mode="strategy_and_atr")
    if strategy_type == "rectangle":
        settings["rsi_entry_filter_enabled"] = False
    return BuyWatchPlan(plan_id=buy_watch_plan_id(symbol,"1d",strategy_label), symbol=symbol,
                        interval="1d", history="5y", price_data_source="Ticker (Alpaca)",
                        strategy_label=strategy_label, strategy_settings=settings,
                        risk_limits=dict(draft["risk_limits"]), order_style="Limit at current price",
                        repeat_after_exit=False, enabled=True,
                        detail="Armed for a new Daily breakout after saving." if strategy_type == "rectangle"
                               else "Waiting for the selected existing strategy's BUY rules.")


def rectangle_order_price_reason(settings, price):
    """Apply again to the actual quote/order price after worker repricing."""
    if settings.get("strategy_type") != "rectangle":
        return ""
    try:
        context = settings["rectangle_context"]
        price = _positive(price, "Entry price")
        trigger = _positive(context["trigger"], "Breakout trigger")
        maximum = _positive(context["maximum_entry"], "Maximum entry")
    except (KeyError, TypeError, ValueError):
        return "Rectangle entry is blocked: frozen trigger or current price is invalid."
    if price <= trigger:
        return "Rectangle entry is blocked: current price has fallen back to or below the breakout trigger."
    if price > maximum+1e-9:
        return f"Rectangle entry is too extended: current/order price exceeds the saved ${maximum:.2f} entry cap."
    return ""


def evaluate_rectangle_strategy(data, settings, account_equity, risk_limits=None, *, now=None):
    """Current-plan evaluation only; no historical trades using hindsight geometry.

    Entry failures leave ATR and strategy exit data available to held positions.
    A new session close must occur AFTER explicit arming and cross frozen trigger.
    """
    from agentloop_trader.backtest import _base_live_fields
    from agentloop_trader.indicators import calc_rsi, calc_sma

    now = utc_now(now)
    frame = completed_daily_bars(data, now)
    exit_window = max(1,int(settings.get("exit_window",10)))
    if len(frame) < max(16,exit_window+1):
        raise ValueError("Not enough completed Daily candles for rectangle ATR/exit rules.")
    prices, highs, lows = (frame[c].to_numpy(dtype=float) for c in ["Close","High","Low"])
    atrs = calc_atr(prices,14,highs,lows)
    atr = _positive(atrs[-1], "Daily ATR")
    stop_distance = atr*_positive(settings.get("atr_stop_multiplier",2), "Stop multiplier")
    last = float(prices[-1])
    exit_level = float(np.min(lows[-1-exit_window:-1]))
    requirements = {}
    reason = ""
    trigger = None
    context = dict(settings.get("rectangle_context") or {})
    try:
        trigger = _positive(context["trigger"], "Frozen breakout trigger")
        resistance = _positive(context["resistance"], "Frozen resistance")
        support = _positive(context["support"], "Frozen support")
        anchor = _positive(context["anchor_close"], "Anchor close")
        maximum = _positive(context["maximum_entry"], "Maximum entry")
        armed = pd.Timestamp(settings["rectangle_armed_at"])
        if armed.tzinfo is None or pd.isna(armed):
            raise ValueError("Arming timestamp is invalid.")
        if not support < resistance < trigger < maximum or context.get("version") != 1:
            raise ValueError("Frozen rectangle levels are invalid.")
        schedule = session_schedule(frame.index[0].year,max(frame.index[-1].year,now.year))
        close_times = schedule["close"].reindex(frame.index)
        after_arming = close_times > armed
        anchor_date = pd.Timestamp(context["anchor_date"])
        basis_ok = anchor_date in frame.index and abs(float(frame.loc[anchor_date,"Close"])/anchor-1) <= .001
        expected_since_arming = schedule.index[(schedule["close"] > armed) & (schedule["close"] <= now)]
        requirements = {
            "Daily execution interval": settings.get("interval") == "1d",
            "Frozen ticker matches": str(data.attrs.get("symbol",context["symbol"])).upper() == context["symbol"] == settings.get("symbol",context["symbol"]),
            "Current completed Daily data": bool(frame.index[-1] == latest_completed_date("1d",now) and _recent_complete(frame,now) and expected_since_arming.isin(frame.index).all()),
            "Frozen price basis unchanged": bool(basis_ok),
            "Plan is within 30 days of arming": bool(armed <= now <= armed+pd.Timedelta(days=PLAN_DAYS)),
            "New completed close after arming": bool(after_arming.iloc[-1]),
            "Range support has held since arming": not bool((frame.Close.loc[after_arming] < support).any()),
            "Fresh cross above frozen trigger": bool(prices[-2] <= trigger < prices[-1]),
            "Completed close within entry cap": last <= maximum,
        }
        reason = next((f"Waiting / blocked: {name}." for name,passed in requirements.items() if not passed),"")
    except (KeyError, TypeError, ValueError) as exc:
        reason = f"Rectangle plan is not armed or valid: {exc}"
        requirements = {"Frozen rectangle plan is valid and armed":False}
    risk_pct = _positive(settings.get("risk_per_trade_pct",.5),"Risk per trade")
    quantity = floor_quantity(max(0,account_equity)*risk_pct/100/stop_distance,"equity")
    intent = None
    if not reason and quantity > 0 and last-stop_distance > 0:
        intent = TradeIntent(symbol=context["symbol"],side="buy",quantity=quantity,
                             entry_price=last,stop_loss=round(last-stop_distance,2),max_holding_bars=exit_window,
                             rationale="New completed Daily close above the frozen rectangle breakout trigger.",
                             source_signals=["frozen_rectangle_breakout","completed_daily_cross_after_arming","atr_position_sizing"])
        if risk_limits is not None:
            intent = constrain_trade_intent_to_limits(intent,account_equity,risk_limits)
        quantity = intent.quantity
        if intent.quantity <= 0:
            intent = None
            reason = "No allowable position under the saved risk limits."
    requirements["Position size above zero"] = bool(quantity > 0 and last-stop_distance > 0)
    if not reason and intent is None:
        reason = "No allowable position under the saved risk limits."
    smas = calc_sma(prices,int(settings.get("moving_average_window",50)))
    live = _base_live_fields(prices=prices,smas=smas,atrs=atrs,rsis=calc_rsi(prices,14),
                            volumes=frame.Volume.to_numpy() if "Volume" in frame else None,
                            index=len(frame)-1,entry_level=trigger,exit_level=exit_level,
                            stop_distance=stop_distance,balance=account_equity,pos_size=quantity,
                            signal="long" if intent is not None else "flat",trade_intent=intent,
                            strategy_name=RECTANGLE_STRATEGY,setup_type="rectangle")
    live.update(buy_requirements=requirements,no_trade_reason=reason or "BUY intent is present.",
                rectangle_context=context,rectangle_armed_at=settings.get("rectangle_armed_at",""),
                rectangle_maximum_entry=context.get("maximum_entry"),
                exit_ready=last <= exit_level,exit_reason=f"Daily close at or below prior {exit_window}-bar low." if last <= exit_level else "Hold above the Daily exit level.",
                sell_requirements={f"Price at or below {exit_window}-bar exit level":last <= exit_level},
                interval="1d",last_completed_bar_at=frame.index[-1].isoformat(),
                in_simulated_trade=False,backtest_available=False)
    return dict(prices=prices,smas=smas,atrs=atrs,trade_log=[],live=live,stats=None,
                labels=frame.index.strftime("%Y-%m-%d").tolist())
