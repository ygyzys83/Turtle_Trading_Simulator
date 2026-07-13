import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dataclasses import asdict, replace
from datetime import datetime
from html import escape
from pathlib import Path
import hashlib
import json
import re

from agentloop_trader.audit import build_audit_events, events_to_records
from agentloop_trader.audit_store import JsonlAuditStore
from agentloop_trader.agents import build_trade_proposal, proposal_records
from agentloop_trader.automation import (
    AutomationDryRunStore,
    AutomationRuntimeState,
    active_automation_level as resolve_active_automation_level,
    auto_entry_decision,
    auto_entry_decision_records,
    auto_exit_decision,
    auto_exit_decision_records,
    automation_decision_records,
    automation_evidence_records,
    automation_readiness_records,
    automation_runtime_records,
    automation_supervisor_dry_run_records,
    build_automation_snapshot,
    paper_automation_candidate_records,
    evidence_dashboard_records,
    paper_automation_dry_run,
)
from agentloop_trader.automation_runtime import (
    AutomationControl,
    AutomationControlStore,
    DEFAULT_LOCK_PATH,
    WorkerStatusStore,
    request_worker_stop,
    start_worker_process,
    worker_status_is_active,
    worker_status_records,
)
from agentloop_trader.backtest import (
    simulate_trendline_breakout_strategy,
    simulate_trendline_retest_strategy,
    simulate_trend_pullback_strategy,
    simulate_turtle_strategy,
    strategy_comparison_records,
)
from agentloop_trader.brokers import (
    AlpacaConfig,
    AlpacaBrokerAdapterStub,
    PaperBrokerAdapter,
    alpaca_config_validation_records,
    alpaca_cancel_preview_records,
    alpaca_preview_records,
    build_alpaca_cancel_preview,
    build_alpaca_order_preview,
    broker_status_records,
)
from agentloop_trader.buy_watchlist import (
    MAX_BUY_WATCHLIST_ITEMS,
    BuyWatchPlan,
    BuyWatchlistStore,
    buy_watch_plan_id,
    buy_watch_plan_detail_records,
    buy_watchlist_records,
)
from agentloop_trader.display import dataframe_for_streamlit
from agentloop_trader.broker_governance import (
    BrokerStateStore,
    adopt_alpaca_position,
    alpaca_order_lifecycle_records,
    alpaca_order_lifecycle_summary_records,
    alpaca_position_lifecycle_records,
    alpaca_position_lifecycle_summary_records,
    broker_state_health,
    build_exit_order_previews,
    cancelable_alpaca_order_records,
    duplicate_exposure_reasons,
    exit_preview_records,
    market_session_advisory,
    open_exit_order_reasons,
    open_order_exposure_reasons,
    preview_already_tracked,
    reconcile_alpaca_positions,
    exit_position_reasons,
    refresh_tracked_alpaca_orders,
    simulated_alpaca_fill_order,
    simulated_exit_preview_readiness_records,
    simulated_position_from_filled_order,
)
from agentloop_trader.evaluation import evaluate_walk_forward, walk_forward_records
from agentloop_trader.evidence import (
    approval_ledger_records,
    approval_ledger_summary_records,
    build_evidence_package,
    evidence_package_records,
    write_evidence_package,
)
from agentloop_trader.fees import (
    ALPACA_EQUITY_FEE_SCHEDULE_EFFECTIVE,
    estimate_alpaca_equity_order_fees,
    estimate_alpaca_equity_round_trip_fees,
)
from agentloop_trader.execution import PaperBroker
from agentloop_trader.llm_research import LLMResearchConfig, LLMResearchResult, analyze_candidate, llm_research_records
from agentloop_trader.market_data import build_company_research_context, fetch_alpaca_bars, fetch_yfinance_bars
from agentloop_trader.models import AuditEvent, ExecutionDecision, RiskCheckResult, RiskLimits, StrategyConfig, TradeIntent
from agentloop_trader.strategy_runtime import adjust_initial_stop_settings, reprice_trade_intent
from agentloop_trader.monitoring import (
    broker_heartbeat_records,
    daily_risk_records,
    monitor_paper_session,
    monitoring_records,
    risk_halt_records,
)
from agentloop_trader.ops_readiness import (
    market_data_freshness_records,
    paper_account_health_records,
    paper_automation_gate_records,
    restart_recovery_records,
    scheduler_preview_records,
    strategy_state_snapshot_records,
)
from agentloop_trader.parameter_loop import (
    buy_and_hold_benchmark,
    candidate_verdict,
    candidate_verdict_records,
    optimize_strategy_intervals,
    optimize_strategy_inputs,
    optimizer_candidate_records,
    optimizer_interval_records,
    optimizer_regime_records,
    optimizer_recommendation_records,
    optimizer_robustness_records,
    optimizer_stress_records,
    validate_settings_across_tickers,
)
from agentloop_trader.performance import ticker_allocated_capital
from agentloop_trader.research_agent import (
    build_research_agent_report,
    research_agent_records,
    strategy_fit_records,
)
from agentloop_trader.research_memory import (
    ResearchSnapshotStore,
    build_research_snapshot,
    compare_research_snapshots,
    research_snapshot_records,
)
from agentloop_trader.risk import (
    build_preflight_check,
    check_trade_intent,
    constrain_trade_intent_to_limits,
    decide_execution,
    preflight_records,
    risk_policy_records,
)
from agentloop_trader.reviews import review_closed_trade, review_records
from agentloop_trader.run_manifest import (
    RunManifestStore,
    build_run_manifest,
    run_manifest_record,
    run_manifest_records,
)
from agentloop_trader.safety import (
    broker_state_simulation_records,
    deployment_readiness_records,
    immutable_boundary_records,
    live_mode_lockfile_records,
    pre_live_readiness_report,
    production_readiness_checks,
    write_live_mode_lockfile,
)
from agentloop_trader.scanner import DEFAULT_SCAN_SYMBOLS, ScannerCandidateStore, scan_universe, scanner_records
from agentloop_trader.session_journal import (
    PaperSessionSnapshot,
    alpaca_paper_activity_records,
    new_session_id,
    paper_performance_records,
    paper_testing_progress_records,
    paper_trading_review_records,
    session_summary_records,
    session_timeline_records,
)
from agentloop_trader.shadow import record_shadow_decision, shadow_records
from agentloop_trader.ui_summary import (
    agent_loop_stage_records,
    alpaca_evidence_summary_records,
    buy_requirement_records,
    compact_status_records,
    managed_position_records,
    no_buy_reason,
    operator_state_record,
    optional_quality_input_records,
    position_exit_plan_records,
    saved_records_overview_records,
    sell_requirement_records,
    setup_scorecard_records,
    strategy_context_records,
    trade_evidence_summary_records,
)
from agentloop_trader.ui_theme import CHART_COLORS, TRADING_CONSOLE_CSS

st.set_page_config(page_title="AgentLoop Trader", layout="wide", initial_sidebar_state="expanded")

if not hasattr(st, "_agentloop_original_dataframe"):
    st._agentloop_original_dataframe = st.dataframe


def safe_streamlit_dataframe(data=None, *args, **kwargs):
    if isinstance(data, pd.DataFrame):
        data = dataframe_for_streamlit(data)
    return st._agentloop_original_dataframe(data, *args, **kwargs)


st.dataframe = safe_streamlit_dataframe

st.markdown(TRADING_CONSOLE_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker, period, interval):
    return fetch_yfinance_bars(ticker, period, interval)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_alpaca_stock_data(ticker: str, period: str, interval: str, api_key: str | None, api_secret: str | None) -> pd.DataFrame:
    return fetch_alpaca_bars(ticker, period, interval, api_key, api_secret)


def fetch_price_data_for_source(symbol: str, history: str, interval_value: str, price_source: str) -> pd.DataFrame:
    if price_source == "Ticker (Alpaca)":
        market_data_config = AlpacaConfig.from_env()
        return fetch_alpaca_stock_data(symbol, history, interval_value, market_data_config.api_key, market_data_config.api_secret)
    return fetch_stock_data(symbol, history, interval_value)


def metric_card(col, label, value, sub, color_class=""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def status_strip(rows: list[dict]) -> None:
    items = []
    for row in rows:
        state = str(row.get("State", "")).strip().lower()
        value = str(row.get("Value", "")).strip().lower()
        semantic_class = (
            "status-negative"
            if state in {"block", "blocked", "error", "failed"}
            or value in {"block", "blocked", "rejected", "failed"}
            else "status-warning"
            if state in {"warning", "stale", "review"}
            else "status-positive"
            if state in {"ok", "ready", "pass", "approved"}
            else ""
        )
        items.append(
            "<div class='status-strip-item {semantic}'>"
            "<div class='status-strip-label'>{label}</div>"
            "<div class='status-strip-value'>{value}</div>"
            "<div class='status-strip-state'>{state}</div>"
            "</div>".format(
                semantic=semantic_class,
                label=escape(str(row.get("Status", ""))),
                value=escape(str(row.get("Value", ""))),
                state=escape(str(row.get("State", ""))),
            )
        )
    st.markdown("<div class='status-strip'>" + "".join(items) + "</div>", unsafe_allow_html=True)


def page_section(title: str, caption: str | None = None) -> None:
    clean_title = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", title).strip()
    st.markdown(f"## {clean_title}")
    if caption:
        st.markdown(f"<div class='ui-section-caption'>{caption}</div>", unsafe_allow_html=True)


def sub_section(title: str, caption: str | None = None) -> None:
    clean_title = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", title).strip()
    st.markdown(f"### {clean_title}")
    if caption:
        st.markdown(f"<div class='ui-section-caption'>{caption}</div>", unsafe_allow_html=True)


def plain_yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def normalized_order_status(value) -> str:
    text = str(value or "").strip().lower()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def count_waiting_alpaca_orders(order_rows: list[dict]) -> int:
    waiting_statuses = {"accepted", "new", "pending_new", "partially_filled"}
    return sum(normalized_order_status(row.get("Status", row.get("status", ""))) in waiting_statuses for row in order_rows)


def open_buy_order_notional(order_rows: list[dict], symbol: str = "") -> float:
    clean_symbol = str(symbol).strip().upper()
    total = 0.0
    for row in order_rows:
        if normalized_order_status(row.get("Status", "")) not in ACTIVE_ALPACA_ORDER_STATUSES:
            continue
        if normalized_order_status(row.get("Side", "")) != "buy":
            continue
        row_symbol = str(row.get("Symbol", "")).strip().upper()
        if clean_symbol and row_symbol != clean_symbol:
            continue
        quantity = optional_float(row.get("Quantity")) or 0.0
        filled_quantity = optional_float(row.get("Filled Qty")) or 0.0
        price = first_available_number(row.get("Limit Price"), row.get("Avg Fill")) or 0.0
        total += max(0.0, quantity - filled_quantity) * max(0.0, price)
    return total


def open_buy_order_symbols(order_rows: list[dict]) -> set[str]:
    return {
        str(row.get("Symbol", "")).strip().upper()
        for row in order_rows
        if normalized_order_status(row.get("Side", "")) == "buy"
        and normalized_order_status(row.get("Status", "")) in ACTIVE_ALPACA_ORDER_STATUSES
    }


ACTIVE_ALPACA_ORDER_STATUSES = {
    "accepted",
    "new",
    "pending_new",
    "partially_filled",
    "pending_cancel",
    "pending_replace",
    "held",
}


STALE_LIMIT_ORDER_OPTIONS = {
    "5 minutes": 5,
    "10 minutes": 10,
    "15 minutes": 15,
    "30 minutes": 30,
    "1 hour": 60,
    "2 hours": 120,
    "4 hours": 240,
    "8 hours": 480,
}


def local_open_buy_order_reasons(symbol: str, tracked_orders: list[dict]) -> list[str]:
    clean_symbol = str(symbol).strip().upper()
    if not clean_symbol:
        return []
    for order in tracked_orders:
        order_symbol = str(order.get("symbol", order.get("Symbol", ""))).strip().upper()
        order_side = str(order.get("side", order.get("Side", ""))).strip().lower()
        order_status = str(order.get("status", order.get("Status", ""))).strip().lower()
        if order_symbol == clean_symbol and order_side == "buy" and order_status in ACTIVE_ALPACA_ORDER_STATUSES:
            return [f"The app already tracks an open {clean_symbol} buy order."]
    return []


def parse_order_timestamp(value) -> pd.Timestamp | None:
    try:
        if value is None or value == "":
            return None
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return None
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.tz_convert("America/Los_Angeles")
    except Exception:
        return None


def order_age_minutes(order: dict, now: pd.Timestamp | None = None) -> float | None:
    submitted_at = parse_order_timestamp(order.get("Submitted") or order.get("submitted_at"))
    if submitted_at is None:
        return None
    current_time = now or pd.Timestamp.now(tz="America/Los_Angeles")
    return max(0.0, (current_time - submitted_at).total_seconds() / 60)


def order_age_label(minutes: float | None) -> str:
    if minutes is None:
        return "Unknown"
    if minutes < 60:
        return f"{minutes:.0f} minutes"
    return f"{minutes / 60:.1f} hours"


def tracked_order_lookup(tracked_orders: list[dict]) -> dict[str, dict]:
    return {
        str(order.get("broker_order_id") or order.get("Alpaca Order ID") or order.get("Broker Order ID") or "").strip(): order
        for order in tracked_orders
        if order.get("broker_order_id") or order.get("Alpaca Order ID") or order.get("Broker Order ID")
    }


def enriched_cancelable_order_records(cancelable_orders: list[dict], tracked_orders: list[dict], alpaca_order_rows: list[dict] | None = None) -> list[dict]:
    tracked_by_id = tracked_order_lookup(tracked_orders)
    alpaca_by_id = {alpaca_order_row_id(order): order for order in (alpaca_order_rows or []) if alpaca_order_row_id(order)}
    enriched = []
    for order in cancelable_orders:
        record = dict(order)
        raw_order = alpaca_by_id.get(alpaca_order_row_id(order), {})
        tracked = tracked_by_id.get(alpaca_order_row_id(order), {})
        record["Order Type"] = str(
            tracked.get("order_type")
            or tracked.get("Order Type")
            or raw_order.get("Order Type")
            or raw_order.get("order_type")
            or order.get("Order Type")
            or order.get("order_type")
            or ""
        ).strip().lower()
        record["Limit Price"] = (
            tracked.get("limit_price")
            or tracked.get("Limit Price")
            or raw_order.get("Limit Price")
            or raw_order.get("limit_price")
            or order.get("Limit Price")
            or order.get("limit_price")
            or ""
        )
        record["Review ID"] = tracked.get("preview_hash") or tracked.get("Review ID") or ""
        record["Source"] = tracked.get("source") or tracked.get("Source") or ""
        enriched.append(record)
    return enriched


def is_waiting_limit_buy_order(order: dict) -> bool:
    side = normalized_order_status(order.get("Side", order.get("side", "")))
    status = normalized_order_status(order.get("Status", order.get("status", "")))
    order_type = normalized_order_status(order.get("Order Type", order.get("order_type", "")))
    limit_price = optional_float(order.get("Limit Price") or order.get("limit_price"))
    return side == "buy" and status in ACTIVE_ALPACA_ORDER_STATUSES and (order_type == "limit" or limit_price is not None)


def waiting_limit_buy_order_records(
    orders: list[dict],
    current_symbol: str,
    current_price: float | None,
    auto_cancel_enabled: bool,
    cancel_after_minutes: int,
) -> list[dict]:
    rows = []
    now = pd.Timestamp.now(tz="America/Los_Angeles")
    current_symbol_clean = str(current_symbol).strip().upper()
    for order in orders:
        if not is_waiting_limit_buy_order(order):
            continue
        symbol = str(order.get("Symbol", "")).strip().upper()
        limit_price = optional_float(order.get("Limit Price"))
        age_minutes = order_age_minutes(order, now)
        loaded_price = current_price if symbol == current_symbol_clean else None
        price_distance = ((loaded_price - limit_price) / limit_price * 100) if loaded_price and limit_price else None
        action = "Waiting"
        if auto_cancel_enabled and age_minutes is not None and age_minutes >= cancel_after_minutes:
            action = "Will auto-cancel"
        rows.append(
            {
                "Ticker": symbol,
                "Shares": order.get("Quantity", ""),
                "Limit Price": money_or_missing(limit_price),
                "Current Price": money_or_missing(loaded_price),
                "Current vs Limit": pct_or_missing(price_distance) if price_distance is not None else "Load ticker to compare",
                "Waiting For": order_age_label(age_minutes),
                "Status": order.get("Status", ""),
                "Next Action": action,
                "Order ID": order.get("Order ID", ""),
            }
        )
    return rows


def stale_limit_buy_orders(orders: list[dict], max_age_minutes: int) -> list[dict]:
    now = pd.Timestamp.now(tz="America/Los_Angeles")
    stale = []
    for order in orders:
        if not is_waiting_limit_buy_order(order):
            continue
        age_minutes = order_age_minutes(order, now)
        if age_minutes is not None and age_minutes >= max_age_minutes:
            stale.append((age_minutes, order))
    return [order for _, order in sorted(stale, key=lambda item: item[0], reverse=True)]


def stale_limit_cancel_blockers(orders: list[dict], max_age_minutes: int) -> list[str]:
    blockers = []
    waiting_limit_orders = [order for order in orders if is_waiting_limit_buy_order(order)]
    stale_orders = stale_limit_buy_orders(orders, max_age_minutes)
    if not auto_cancel_stale_limit_orders:
        blockers.append("Auto-cancel old limit buys is off.")
    if active_automation_level == "Manual review only":
        blockers.append("Automation is set to Manual.")
    if execution_mode != "paper":
        blockers.append("Order mode is not Paper trading.")
    if not enable_alpaca_paper_orders:
        blockers.append("Alpaca is not configured for paper trading.")
    if not alpaca_status.connected:
        blockers.append("Alpaca is not connected.")
    if not alpaca_status.can_submit_orders:
        blockers.append("Alpaca paper order submission is off.")
    if alpaca_state_health.stale:
        blockers.append("Alpaca data needs refresh.")
    if effective_kill_switch:
        blockers.append("Kill Switch is on.")
    if not waiting_limit_orders:
        blockers.append("No waiting BUY limit orders were found.")
    elif not stale_orders:
        blockers.append(f"Waiting BUY limit order has not reached {order_age_label(max_age_minutes)} yet.")
    if any(is_waiting_limit_buy_order(order) and order_age_minutes(order) is None for order in orders):
        blockers.append("At least one waiting BUY limit order is missing its submitted time.")
    return blockers


def stale_limit_cancel_status_records(orders: list[dict], max_age_minutes: int) -> list[dict]:
    waiting_limit_orders = [order for order in orders if is_waiting_limit_buy_order(order)]
    stale_orders = stale_limit_buy_orders(orders, max_age_minutes)
    oldest_age = max((order_age_minutes(order) or 0 for order in waiting_limit_orders), default=None)
    blockers = stale_limit_cancel_blockers(orders, max_age_minutes)
    return [
        {"Item": "Auto-cancel old limit buys", "Value": plain_yes_no(auto_cancel_stale_limit_orders)},
        {"Item": "Automation running", "Value": plain_yes_no(active_automation_level != "Manual review only")},
        {"Item": "Waiting BUY limit orders", "Value": str(len(waiting_limit_orders))},
        {"Item": "Old enough to cancel", "Value": str(len(stale_orders))},
        {"Item": "Oldest order age", "Value": order_age_label(oldest_age)},
        {"Item": "Cancel check", "Value": "Ready" if not blockers else "Blocked"},
        {"Item": "Reason", "Value": "Ready to cancel the oldest stale limit buy." if not blockers else " ".join(blockers)},
    ]


def cancelable_order_debug_records(orders: list[dict]) -> list[dict]:
    return [
        {
            "Ticker": str(order.get("Symbol", "")),
            "Side": str(order.get("Side", order.get("side", ""))),
            "Status": str(order.get("Status", order.get("status", ""))),
            "Order Type": str(order.get("Order Type", order.get("order_type", ""))),
            "Limit Price": str(order.get("Limit Price", order.get("limit_price", ""))),
            "Submitted": str(order.get("Submitted", order.get("submitted_at", ""))),
            "Recognized As BUY Limit": plain_yes_no(is_waiting_limit_buy_order(order)),
        }
        for order in orders
    ]


def round_alpaca_price(price: float) -> float:
    return round(float(price), 4 if float(price) < 1 else 2)


def optional_float(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "").strip()
        return float(value)
    except (TypeError, ValueError):
        return None


def account_record_value(records: list[dict], field_name: str) -> float | None:
    target = field_name.strip().lower()
    for record in records:
        if str(record.get("Field", "")).strip().lower() == target:
            return optional_float(record.get("Value"))
    return None


def first_available_number(*values) -> float | None:
    for value in values:
        number = optional_float(value)
        if number is not None:
            return number
    return None


def parse_saved_time(value) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("America/Los_Angeles")
        return timestamp
    except Exception:
        return None


def resize_trade_intent_for_account(intent: TradeIntent | None, account_equity: float, risk_pct_dec: float) -> TradeIntent | None:
    if intent is None or intent.entry_price is None or intent.stop_loss is None or intent.side != "buy":
        return intent
    risk_per_share = abs(float(intent.entry_price) - float(intent.stop_loss))
    if risk_per_share <= 0:
        return intent
    account_risk_dollars = max(0.0, float(account_equity)) * float(risk_pct_dec)
    resized_quantity = int(account_risk_dollars // risk_per_share)
    if resized_quantity <= 0 or resized_quantity == intent.quantity:
        return intent
    return replace(intent, quantity=resized_quantity)


def money_or_missing(value) -> str:
    number = optional_float(value)
    return f"${number:,.2f}" if number is not None else "Not recorded"


def pct_or_missing(value) -> str:
    number = optional_float(value)
    return f"{number:.2f}%" if number is not None else "Not recorded"


def bars_or_missing(value) -> str:
    try:
        if value is None or value == "":
            return "Not recorded"
        return f"{int(value)} bars"
    except (TypeError, ValueError):
        return "Not recorded"


def plain_setting_name(key: str) -> str:
    replacements = {
        "atr": "ATR",
        "pct": "percent",
        "pnl": "P&L",
        "sma": "SMA",
        "id": "ID",
    }
    words = [
        replacements.get(part.lower(), part.lower())
        for part in str(key).replace("-", "_").split("_")
        if part
    ]
    if not words:
        return ""
    words[0] = words[0][:1].upper() + words[0][1:]
    return " ".join(words)


def entry_snapshot_records(settings: dict | None) -> list[dict]:
    settings = settings or {}
    return [
        {"Field": "Ticker", "Value": str(settings.get("symbol", "Not recorded"))},
        {"Field": "Strategy", "Value": str(settings.get("strategy_label", settings.get("strategy_type", "Not recorded")))},
        {"Field": "Price data source", "Value": str(settings.get("price_data_source", "Not recorded"))},
        {"Field": "Price interval", "Value": str(settings.get("interval", "Not recorded"))},
        {"Field": "History used", "Value": str(settings.get("history", "Not recorded"))},
        {"Field": "Buy breakout length", "Value": bars_or_missing(settings.get("entry_window"))},
        {"Field": "Sell exit length", "Value": bars_or_missing(settings.get("exit_window"))},
        {"Field": "Trend filter length", "Value": bars_or_missing(settings.get("moving_average_window"))},
        {"Field": "Pullback average length", "Value": bars_or_missing(settings.get("pullback_average_length"))},
        {"Field": "Momentum turn length", "Value": bars_or_missing(settings.get("momentum_turn_length"))},
        {"Field": "RSI entry rule", "Value": "Require RSI 50-70" if settings.get("rsi_entry_filter_enabled", False) else "Off"},
        {"Field": "Sizing account", "Value": str(settings.get("sizing_account_source", "Not recorded"))},
        {"Field": "Sizing account value", "Value": money_or_missing(settings.get("sizing_account_equity"))},
        {"Field": "Sizing cash available", "Value": money_or_missing(settings.get("sizing_available_cash"))},
        {"Field": "Reference price at entry", "Value": money_or_missing(settings.get("entry_reference_price"))},
        {"Field": "ATR at entry", "Value": money_or_missing(settings.get("entry_atr"))},
        {"Field": "ATR percent at entry", "Value": pct_or_missing(settings.get("entry_atr_pct"))},
        {"Field": "Stop distance ATR multiplier", "Value": str(settings.get("entry_stop_atr_multiplier", settings.get("atr_stop_multiplier", "Not recorded")))},
        {"Field": "Stop distance at entry", "Value": money_or_missing(settings.get("entry_stop_distance"))},
        {"Field": "Planned stop before fill", "Value": money_or_missing(settings.get("entry_stop_loss"))},
        {"Field": "Entry rule level", "Value": money_or_missing(settings.get("entry_rule_level"))},
        {"Field": "Exit rule level at entry", "Value": money_or_missing(settings.get("exit_rule_level_at_entry"))},
        {"Field": "Profit protection", "Value": "On" if settings.get("profit_protection_enabled", True) else "Off"},
        {"Field": "Move stop to break-even after", "Value": f"+{settings.get('breakeven_after_r', 1.0)}R"},
        {"Field": "Start ATR trail after", "Value": f"+{settings.get('trail_after_r', 2.0)}R"},
        {"Field": "Trailing ATR multiplier", "Value": str(settings.get("trailing_atr_multiplier", 3.0))},
        {"Field": "Planned order type", "Value": str(settings.get("planned_order_type", "Not recorded"))},
        {"Field": "Planned limit price", "Value": money_or_missing(settings.get("planned_limit_price"))},
        {"Field": "Planned quantity", "Value": str(settings.get("planned_quantity", "Not recorded"))},
    ]


def combined_position_risk_records(position: dict, exit_trigger_price: float | None) -> list[dict]:
    symbol = str(position.get("Symbol", "")).strip().upper()
    quantity = optional_float(position.get("Quantity"))
    avg_entry = optional_float(position.get("Average Entry"))
    market_value = optional_float(position.get("Market Value"))
    current_price = market_value / quantity if market_value is not None and quantity else None
    trigger = optional_float(exit_trigger_price)
    dollars_at_risk = (avg_entry - trigger) * quantity if avg_entry is not None and trigger is not None and quantity else None
    pct_at_risk = ((avg_entry - trigger) / avg_entry * 100) if avg_entry and trigger is not None else None
    distance_to_exit = ((current_price - trigger) / current_price * 100) if current_price and trigger is not None else None
    return [
        {"Field": "Ticker", "Value": symbol or "Not available"},
        {"Field": "Total shares", "Value": f"{quantity:g}" if quantity is not None else "Not available"},
        {"Field": "Alpaca average entry", "Value": money_or_missing(avg_entry)},
        {"Field": "Current price", "Value": money_or_missing(current_price)},
        {"Field": "Auto exit trigger", "Value": money_or_missing(trigger)},
        {"Field": "Dollars at risk if exit triggers", "Value": money_or_missing(max(0.0, dollars_at_risk)) if dollars_at_risk is not None else "Not available"},
        {"Field": "Percent at risk if exit triggers", "Value": pct_or_missing(max(0.0, pct_at_risk)) if pct_at_risk is not None else "Not available"},
        {"Field": "Distance from current price to exit", "Value": pct_or_missing(distance_to_exit) if distance_to_exit is not None else "Not available"},
    ]


def position_current_price(position: dict) -> float | None:
    quantity = optional_float(position.get("Quantity"))
    market_value = optional_float(position.get("Market Value"))
    if quantity and market_value is not None:
        return market_value / quantity
    return None


def position_unrealized_pct(position: dict) -> float | None:
    unrealized = optional_float(position.get("Unrealized P&L"))
    avg_entry = optional_float(position.get("Average Entry"))
    quantity = optional_float(position.get("Quantity"))
    entry_value = avg_entry * quantity if avg_entry is not None and quantity else None
    if entry_value and unrealized is not None:
        return unrealized / entry_value * 100
    return None


def position_next_action(position: dict, settings: dict, exit_details: dict) -> str:
    symbol = str(position.get("Symbol", "")).strip().upper()
    if not settings:
        return f"Save exit settings for {symbol}."
    if not bool(settings.get("auto_exit_enabled", True)):
        return f"Auto exit is off for {symbol}."
    if bool(exit_details.get("ready")):
        return f"Exit is triggered for {symbol}."
    trigger = optional_float(exit_details.get("trigger_price"))
    if trigger is not None:
        return f"Hold. Auto exit sells at ${trigger:,.2f} or lower."
    return f"Hold. Exit rule is being watched for {symbol}."


def daily_position_rows(positions: list[dict], settings_by_symbol: dict[str, dict]) -> list[dict]:
    rows = []
    for position in positions:
        symbol = str(position.get("Symbol", "")).strip().upper()
        settings = settings_by_symbol.get(symbol, {})
        exit_details = evaluate_exit_rule_details_from_settings(settings) if settings else {}
        current_price = position_current_price(position)
        trigger = optional_float(exit_details.get("trigger_price"))
        distance_to_exit = ((current_price - trigger) / current_price * 100) if current_price and trigger is not None else None
        rows.append(
            {
                "Ticker": symbol,
                "Shares": position.get("Quantity", ""),
                "Current Price": money_or_missing(current_price),
                "Average Entry": money_or_missing(position.get("Average Entry")),
                "Unrealized P&L": money_or_missing(position.get("Unrealized P&L")),
                "Auto Exit": "On" if settings.get("auto_exit_enabled", False) else "Off",
                "Exit Price": money_or_missing(trigger),
                "Exit Rule": str(exit_details.get("trigger_source", "Not set")).title(),
                "Room Before Exit": pct_or_missing(distance_to_exit) if distance_to_exit is not None else "Not set",
                "Next Action": position_next_action(position, settings, exit_details),
            }
        )
    return rows


def position_management_summary_records(position: dict, settings: dict, exit_details: dict) -> list[dict]:
    current_price = position_current_price(position)
    trigger = optional_float(exit_details.get("trigger_price"))
    distance_to_exit = ((current_price - trigger) / current_price * 100) if current_price and trigger is not None else None
    profit_r = optional_float(exit_details.get("profit_r"))
    highest_profit_r = optional_float(exit_details.get("highest_profit_r"))
    return [
        {"Area": "Position", "Item": "Ticker", "Status / Value": str(position.get("Symbol", "")).strip().upper(), "Plain English": "Open Alpaca paper position."},
        {"Area": "Position", "Item": "Shares", "Status / Value": str(position.get("Quantity", "")), "Plain English": "Current shares held at Alpaca."},
        {"Area": "Position", "Item": "Current price", "Status / Value": money_or_missing(current_price), "Plain English": "Estimated from Alpaca market value."},
        {"Area": "Position", "Item": "Average entry", "Status / Value": money_or_missing(position.get("Average Entry")), "Plain English": "Average entry price reported by Alpaca."},
        {"Area": "Exit plan", "Item": "Auto exit", "Status / Value": "On" if settings.get("auto_exit_enabled", True) else "Off", "Plain English": "Whether the app may sell this position automatically."},
        {"Area": "Exit plan", "Item": "Price interval", "Status / Value": str(settings.get("interval", "Not saved")), "Plain English": "Bars used to calculate ATR and exit levels."},
        {"Area": "Exit plan", "Item": "Sell exit length", "Status / Value": bars_or_missing(settings.get("exit_window")), "Plain English": "Number of saved price bars used to calculate the strategy exit line."},
        {"Area": "Exit plan", "Item": "Trend filter length", "Status / Value": bars_or_missing(settings.get("moving_average_window")), "Plain English": "Saved trend filter length used to confirm the broader uptrend."},
        {"Area": "Exit plan", "Item": "Pullback average length", "Status / Value": bars_or_missing(settings.get("pullback_average_length")), "Plain English": "Saved pullback-zone average used by Trend pullback continuation."},
        {"Area": "Exit plan", "Item": "Momentum turn length", "Status / Value": bars_or_missing(settings.get("momentum_turn_length")), "Plain English": "Saved short average used by Trend pullback continuation and Trendline retest continuation to confirm price turned back up."},
        {"Area": "Exit plan", "Item": "Strategy exit", "Status / Value": money_or_missing(exit_details.get("strategy_exit_price")), "Plain English": "Exit line from the saved strategy settings."},
        {"Area": "Exit plan", "Item": "Fill-adjusted initial stop", "Status / Value": money_or_missing(exit_details.get("original_stop_price")), "Plain English": "Actual Alpaca average fill minus the saved initial ATR stop distance."},
        {"Area": "Exit plan", "Item": "Initial stop ATR multiplier", "Status / Value": str(settings.get("entry_stop_atr_multiplier", settings.get("atr_stop_multiplier", "Not recorded"))), "Plain English": "Saved entry ATR multiplier used to calculate the fill-adjusted initial stop."},
        {"Area": "Profit protection", "Item": "Current profit in R", "Status / Value": f"{profit_r:.2f}R" if profit_r is not None else "Not available", "Plain English": "Current profit compared with original trade risk."},
        {"Area": "Profit protection", "Item": "Highest profit reached in R", "Status / Value": f"{highest_profit_r:.2f}R" if highest_profit_r is not None else "Not available", "Plain English": "Highest price reached since entry compared with original trade risk. This turns profit protection on permanently."},
        {"Area": "Profit protection", "Item": "ATR trail", "Status / Value": money_or_missing(exit_details.get("trailing_stop_price")), "Plain English": "Highest high since entry minus the trailing ATR distance."},
        {"Area": "Profit protection", "Item": "Break-even stop", "Status / Value": money_or_missing(exit_details.get("breakeven_stop_price")), "Plain English": "Turns on after the saved profit threshold."},
        {"Area": "Automation", "Item": "Active sell trigger", "Status / Value": money_or_missing(trigger), "Plain English": f"Sell if price reaches this level or lower. Source: {exit_details.get('trigger_source', 'exit rule')}."},
        {"Area": "Automation", "Item": "Room before exit", "Status / Value": pct_or_missing(distance_to_exit) if distance_to_exit is not None else "Not available", "Plain English": "How far current price is above the active sell trigger."},
        {"Area": "Automation", "Item": "Current action", "Status / Value": "Exit now" if exit_details.get("ready") else "Hold", "Plain English": str(exit_details.get("reason", ""))},
    ]


def alpaca_daily_summary_records() -> list[dict]:
    status = next((str(row.get("Value", "")) for row in alpaca_account_records if str(row.get("Field", "")).lower() == "status"), "")
    return [
        {"Item": "Connection", "Value": "Connected" if alpaca_status.connected else "Not connected", "Plain English": f"Account status: {status or 'unknown'}."},
        {"Item": f"{alpaca_account_label.title()} value", "Value": money_or_missing(account_record_value(alpaca_account_records, "Portfolio Value")), "Plain English": f"Value reported by {alpaca_account_label}."},
        {"Item": "Buying power", "Value": money_or_missing(account_record_value(alpaca_account_records, "Buying Power")), "Plain English": f"Buying power reported by {alpaca_account_label}."},
        {"Item": "Open positions", "Value": len(alpaca_positions), "Plain English": f"Current {alpaca_account_label} positions."},
        {"Item": "Orders waiting to fill", "Value": count_waiting_alpaca_orders(alpaca_orders), "Plain English": "Open Alpaca orders not filled or canceled yet."},
        {"Item": "Automation", "Value": active_automation_level, "Plain English": "Current app automation setting."},
    ]


def automation_status_text() -> tuple[str, str, str]:
    if active_automation_level == "Manual review only":
        return "Manual", "You click paper orders manually.", "info"
    if auto_exit_status.ready:
        return "Auto exit ready", f"The app can sell {auto_exit_status.quantity} {auto_exit_status.symbol}.", "warn"
    if auto_entry_status.ready:
        return "Auto buy ready", f"The app can buy {auto_entry_status.quantity} {auto_entry_status.symbol}.", "ok"
    visible_status = auto_entry_status.status if active_automation_level == "Auto entries and exits" else auto_exit_status.status
    visible_reasons = auto_entry_status.reasons if active_automation_level == "Auto entries and exits" else auto_exit_status.reasons
    detail = "; ".join(visible_reasons) if visible_reasons else "No automation action is ready."
    return visible_status, detail, "warn"


def daily_automation_readiness_records(context: str) -> list[dict]:
    queued_plans = buy_watchlist_store.read() if context == "New trade" and "buy_watchlist_store" in globals() else []
    if queued_plans:
        enabled_plans = [plan for plan in queued_plans if plan.enabled]
        status_counts: dict[str, int] = {}
        for plan in enabled_plans:
            status_counts[plan.status] = status_counts.get(plan.status, 0) + 1
        if effective_kill_switch:
            read = "Blocked"
            detail = "Kill Switch is on. No queued paper orders can be sent."
        elif active_automation_level != "Auto entries and exits" or not full_automation_enabled:
            read = "Paused"
            detail = "Select Auto entries and exits and check Allow automatic paper buys."
        elif execution_mode != "paper" or not enable_alpaca_paper_orders:
            read = "Paused"
            detail = "Select Paper trading. That order mode automatically uses Alpaca paper."
        elif not sidebar_worker_active:
            read = "Worker stopped"
            detail = "Click Start Worker. The Streamlit page timer does not monitor queued setups."
        elif not enabled_plans:
            read = "No enabled setups"
            detail = "Resume a saved setup or add a new one."
        elif status_counts.get("Blocked"):
            read = "Blocked"
            detail = f"{status_counts['Blocked']} queued setup(s) are blocked. Review the Buy watchlist status and saved details."
        else:
            read = "Watching queue"
            detail = f"The worker is monitoring {len(enabled_plans)} enabled setup(s) for completed BUY rules."
        status_summary = ", ".join(f"{count} {status.lower()}" for status, count in status_counts.items()) or "No enabled setups"
        return [
            {"Area": "Buy watchlist", "Read": read, "Plain English": detail},
            {"Area": "Queued setups", "Read": f"{len(enabled_plans)} enabled / {len(queued_plans)} saved", "Plain English": status_summary},
            {"Area": "Automation mode", "Read": automation_level_label, "Plain English": "Queued entries require Auto entries and exits with Allow automatic paper buys checked."},
            {"Area": "Background worker", "Read": "Running" if sidebar_worker_active else "Stopped", "Plain English": "Only the background worker monitors the durable Buy watchlist."},
            {"Area": "Last worker check", "Read": automation_worker_status.last_checked_at or "Not checked yet", "Plain English": automation_worker_status.last_action or "No worker action yet."},
        ]
    if effective_kill_switch:
        read = "Blocked"
        detail = "Kill Switch is on. No paper orders can be sent."
    elif auto_exit_status.ready:
        read = "Ready to sell"
        detail = f"The app can sell {auto_exit_status.quantity} {auto_exit_status.symbol} if automation is enabled."
    elif auto_entry_status.ready:
        read = "Ready to buy"
        detail = f"The app can buy {auto_entry_status.quantity} {auto_entry_status.symbol} if automation is enabled."
    elif active_automation_level == "Manual review only":
        read = "Watching only"
        detail = "Automation is off. The app will show ideas, but you click orders manually."
    else:
        visible_reasons = auto_entry_status.reasons if active_automation_level == "Auto entries and exits" else auto_exit_status.reasons
        read = "Blocked" if visible_reasons else "Watching only"
        detail = visible_reasons[0] if visible_reasons else "Automation is on, but no buy or sell action is ready."
    watched = "Open positions" if active_automation_level == "Auto exits only" else f"{ticker} buys and open positions" if active_automation_level == "Auto entries and exits" else "No automatic orders"
    return [
        {"Area": context, "Read": read, "Plain English": detail},
        {"Area": "Automation mode", "Read": automation_level_label, "Plain English": f"Currently watching: {watched}."},
        {"Area": "Last automation check", "Read": st.session_state.get("last_automation_checked_at", "Not checked yet"), "Plain English": f"Checks every {automation_refresh_seconds} seconds while automation is on."},
        {"Area": "Last automation action", "Read": st.session_state.get("last_automation_action", "None"), "Plain English": st.session_state.get("last_automation_blocked_reason", "") or "No recent action."},
    ]


def open_positions_next_step(position_settings_by_symbol: dict[str, dict]) -> str:
    if not alpaca_positions:
        return "No open Alpaca positions. Use New Trade when you want to research the next setup."
    unmanaged = [
        str(position.get("Symbol", "")).strip().upper()
        for position in alpaca_positions
        if not position_settings_by_symbol.get(str(position.get("Symbol", "")).strip().upper())
    ]
    if auto_exit_status.ready:
        return f"Ready to sell {auto_exit_status.quantity} {auto_exit_status.symbol} if automation is enabled."
    if unmanaged:
        return f"Save exit settings for {', '.join(unmanaged)}."
    if count_waiting_alpaca_orders(alpaca_orders):
        return "Review waiting Alpaca orders before adding new exposure."
    if active_automation_level == "Auto exits only":
        return "Auto exits are watching saved position exit rules."
    return "Positions have saved exit settings. Turn on Auto exits if you want the app to manage sells."


def live_trading_setup_records() -> list[dict]:
    config = alpaca_adapter.config
    live_mode_selected = execution_mode in {"live_with_approval", "automated_live"}
    return [
        {
            "Check": "Order mode",
            "Read": mode_label,
            "Plain English": "Choose Live with approval before sending any real order.",
        },
        {
            "Check": "Alpaca account",
            "Read": config.account_mode.title(),
            "Plain English": "Live orders require ALPACA_PAPER=false and the live Alpaca endpoint.",
        },
        {
            "Check": "Live env switch",
            "Read": "On" if config.live_trading_enabled else "Off",
            "Plain English": "Set ALPACA_LIVE_TRADING_ENABLED=true when you are ready to test manual live orders.",
        },
        {
            "Check": "Live confirmation",
            "Read": "Set" if config.live_confirmation == "I_UNDERSTAND_LIVE_TRADING" else "Missing",
            "Plain English": "Set ALPACA_LIVE_CONFIRMATION=I_UNDERSTAND_LIVE_TRADING before live order buttons can send.",
        },
        {
            "Check": "Sidebar live orders",
            "Read": "On" if enable_alpaca_live_orders else "Off",
            "Plain English": "This checkbox appears only when a live order mode is selected.",
        },
        {
            "Check": "Ready for manual live order",
            "Read": "Yes" if live_mode_selected and config.live_order_enabled and enable_alpaca_live_orders and alpaca_status.can_submit_orders else "No",
            "Plain English": "All live switches must be on, Alpaca must be connected, and the Kill Switch must be off.",
        },
    ]


def apply_paper_buy_order_settings(
    intent: TradeIntent | None,
    order_style: str,
    limit_adjustment_pct: float,
    custom_limit_price: float,
    reference_price: float,
) -> TradeIntent | None:
    if intent is None or intent.side != "buy":
        return intent
    if order_style == "Market":
        return replace(intent, order_type="market", limit_price=None)
    base_price = reference_price or intent.entry_price
    if base_price is None or base_price <= 0:
        return replace(intent, order_type="market", limit_price=None)
    source_signals = list(intent.source_signals)
    if "paper_limit_order" not in source_signals:
        source_signals.append("paper_limit_order")
    if order_style == "Custom limit price":
        custom_price = optional_float(custom_limit_price)
        limit_price = round_alpaca_price(custom_price) if custom_price and custom_price > 0 else None
    else:
        adjustment = max(0.0, float(limit_adjustment_pct)) / 100
        if order_style == "Limit below current price":
            adjustment *= -1
        else:
            adjustment = adjustment if order_style == "Limit above current price" else 0.0
        limit_price = round_alpaca_price(base_price * (1 + adjustment))
    return replace(
        intent,
        order_type="limit",
        limit_price=limit_price,
        entry_price=limit_price or base_price,
        source_signals=source_signals,
    )


def paper_buy_price_records(
    intent: TradeIntent | None,
    order_style: str,
    limit_adjustment_pct: float,
    custom_limit_price: float,
    reference_price: float,
) -> list[dict]:
    if intent is None:
        return [{"Field": "Paper buy price", "Value": "No BUY idea right now"}]
    order_type = "Market" if intent.order_type == "market" else "Limit"
    max_price = "Market order" if intent.order_type == "market" else money_or_missing(intent.limit_price)
    adjustment_label = "None"
    if order_style == "Limit below current price":
        adjustment_label = f"{limit_adjustment_pct:.2f}% below reference"
    elif order_style == "Limit above current price":
        adjustment_label = f"{limit_adjustment_pct:.2f}% above reference"
    elif order_style == "Custom limit price":
        adjustment_label = f"Exact price: {money_or_missing(custom_limit_price)}"
    limit_rule = (
        "Buy at the next available market price."
        if intent.order_type == "market"
        else "Enter a custom limit price before sending this order."
        if intent.limit_price is None
        else "Do not buy above the listed limit price."
    )
    fee_price = float(intent.limit_price or intent.entry_price or reference_price or 0.0)
    buy_fee = estimate_alpaca_equity_order_fees(
        side="buy", quantity=intent.quantity, price=fee_price,
    ) if fee_price > 0 else None
    return [
        {"Field": "Order sent to Alpaca", "Value": order_type},
        {"Field": "Shares", "Value": f"{intent.quantity:,}"},
        {"Field": "Reference price", "Value": money_or_missing(reference_price)},
        {"Field": "Price instruction", "Value": order_style},
        {"Field": "Price adjustment", "Value": adjustment_label},
        {"Field": "Highest buy price", "Value": max_price},
        {
            "Field": "Estimated live buy fee",
            "Value": money_or_missing(buy_fee.total if buy_fee is not None else None),
        },
        {"Field": "Plain English", "Value": limit_rule},
    ]


def add_automation_history(action: str, detail: str = "", checked_at: str | None = None) -> None:
    timestamp = checked_at or pd.Timestamp.now(tz="America/Los_Angeles").isoformat()
    record = {
        "Time": timestamp,
        "Action": str(action or "None"),
        "Detail": str(detail or ""),
    }
    history = list(st.session_state.get("automation_event_history", []))
    if not history or (history[-1].get("Action"), history[-1].get("Detail")) != (record["Action"], record["Detail"]):
        history.append(record)
    st.session_state["automation_event_history"] = history[-5:]


def set_automation_action(action: str, detail: str = "", checked_at: str | None = None) -> None:
    st.session_state["last_automation_action"] = action
    st.session_state["last_automation_blocked_reason"] = detail
    add_automation_history(action, detail, checked_at)


def strategy_use_case_records(selected_strategy: str) -> list[dict]:
    descriptions = {
        "Breakout continuation": "Best when price is clearing a recent range high and the broad trend is already up.",
        "Trend pullback continuation": "Best when an uptrend pauses, pulls toward a moving average, then turns back up.",
        "Trendline breakout": "Best when price breaks a descending resistance line after a controlled pullback.",
        "Trendline retest continuation": "Best when price breaks resistance, retests that line, then resumes higher.",
    }
    cautions = {
        "Breakout continuation": "Can chase extended moves if volume and volatility are poor.",
        "Trend pullback continuation": "Can enter too early if the pullback keeps falling.",
        "Trendline breakout": "Trendlines are estimated, so false breaks can happen.",
        "Trendline retest continuation": "Fewer signals; the retest may never happen.",
    }
    return [
        {"Strategy": name, "Best Use": descriptions[name], "Main Caution": cautions[name], "Selected": plain_yes_no(name == selected_strategy)}
        for name in descriptions
    ]


def exit_model_records() -> list[dict]:
    return [
        {
            "Exit Rule": "Initial stop",
            "Current Setting": f"{atr_mult:.2f} ATR",
            "Plain English": "The first stop is based on the selected ATR stop distance.",
        },
        {
            "Exit Rule": "Strategy exit",
            "Current Setting": f"{exit_w} bars",
            "Plain English": "The selected strategy also watches its saved sell line.",
        },
        {
            "Exit Rule": "Break-even protection",
            "Current Setting": "+1R",
            "Plain English": "After the trade gains one original-risk unit, the stop can move to the entry price.",
        },
        {
            "Exit Rule": "ATR trailing stop",
            "Current Setting": "+2R then 3.0 ATR",
            "Plain English": "After the trade gains two original-risk units, the stop trails from the highest price since entry.",
        },
        {
            "Exit Rule": "Active sell trigger",
            "Current Setting": "Highest protection level",
            "Plain English": "The app uses the highest valid stop so protection can tighten but not loosen.",
        },
    ]


def strategy_decision_detail() -> str:
    if intent is not None and preflight_check.ready:
        return f"{strategy_label} produced a BUY and risk checks passed."
    blockers = [reason for reason in (preflight_check.blocked_reasons or risk_check.rejected_reasons) if reason]
    if blockers:
        return f"Blocked before order: {blockers[0]}"
    requirements = live.get("buy_requirements") or {}
    missing = [str(rule) for rule, passed in requirements.items() if not passed]
    if missing:
        return f"No BUY because this rule is not met: {missing[0]}."
    if bool(live.get("in_simulated_trade")):
        return "No BUY because the historical simulation is already in a trade, but Alpaca reality is checked separately."
    return no_buy_reason(live)


def decision_ticket_records() -> list[dict]:
    if intent is None:
        return [
            {"Item": "Decision", "Value": final_answer},
            {"Item": "Reason", "Value": final_detail},
            {"Item": "Next action", "Value": operator_state["Next Action"]},
            {"Item": "Ticker", "Value": ticker},
            {"Item": "Strategy", "Value": strategy_label},
        ]
    dollars_at_risk = estimated_intent_risk_with_fees(intent)
    risk_pct_account = (dollars_at_risk / paper_order_risk_equity * 100) if dollars_at_risk is not None and paper_order_risk_equity else None
    fee_exit_price = intent.stop_loss or intent.entry_price
    buy_fees, sell_fees = estimate_alpaca_equity_round_trip_fees(
        quantity=intent.quantity,
        entry_price=intent.entry_price,
        exit_price=fee_exit_price,
    )
    return [
        {"Item": "Decision", "Value": final_answer},
        {"Item": "Reason", "Value": final_detail},
        {"Item": "Ticker", "Value": intent.symbol_clean},
        {"Item": "Strategy", "Value": strategy_label},
        {"Item": "Shares", "Value": f"{intent.quantity:,}"},
        {"Item": "Reference price", "Value": money_or_missing(live.get("last_p"))},
        {"Item": "Highest buy price", "Value": "Market order" if intent.order_type == "market" else money_or_missing(intent.limit_price)},
        {"Item": "Stop loss", "Value": money_or_missing(intent.stop_loss)},
        {"Item": "Dollars at risk", "Value": money_or_missing(dollars_at_risk)},
        {"Item": "Account risk", "Value": pct_or_missing(risk_pct_account)},
        {
            "Item": "Estimated round-trip Alpaca fees",
            "Value": money_or_missing(buy_fees.total + sell_fees.total),
        },
        {"Item": "Next action", "Value": operator_state["Next Action"]},
    ]


def estimated_intent_risk_with_fees(trade_intent: TradeIntent | None) -> float | None:
    if trade_intent is None or trade_intent.entry_price is None or trade_intent.stop_loss is None:
        return None
    price_risk = abs(float(trade_intent.entry_price) - float(trade_intent.stop_loss)) * trade_intent.quantity
    buy_fees, sell_fees = estimate_alpaca_equity_round_trip_fees(
        quantity=trade_intent.quantity,
        entry_price=trade_intent.entry_price,
        exit_price=trade_intent.stop_loss,
    )
    return round(price_risk + buy_fees.total + sell_fees.total, 2)


def required_setup_reads(selected_strategy_type: str, require_rsi: bool = False) -> set[str]:
    base = {"Trend", "Risk approval"}
    if require_rsi:
        base.add("RSI condition")
    if selected_strategy_type == "pullback":
        return base | {"Pullback", "Momentum turn"}
    if selected_strategy_type == "trendline":
        return base | {"Trendline", "Break above line"}
    if selected_strategy_type == "trendline_retest":
        return base | {"Trendline", "Break above line", "Retest"}
    return base | {"Breakout"}


def trade_read_records() -> list[dict]:
    rows = [
        {"Section": "Decision", "Item": "Final answer", "Status / Value": final_answer, "Plain English": final_detail},
        {"Section": "Decision", "Item": "Next action", "Status / Value": operator_state["Next Action"], "Plain English": "What to do from here."},
        {"Section": "Decision", "Item": "Ticker", "Status / Value": ticker, "Plain English": "Ticker being researched."},
        {"Section": "Decision", "Item": "Strategy", "Status / Value": strategy_label, "Plain English": "Trading rule selected in the sidebar."},
    ]
    if intent is not None:
        dollars_at_risk = estimated_intent_risk_with_fees(intent)
        risk_pct_account = (dollars_at_risk / paper_order_risk_equity * 100) if dollars_at_risk is not None and paper_order_risk_equity else None
        rows.extend(
            [
                {"Section": "Decision", "Item": "Shares", "Status / Value": f"{intent.quantity:,}", "Plain English": "Order size after risk sizing."},
                {
                    "Section": "Decision",
                    "Item": "Highest buy price",
                    "Status / Value": "Market order" if intent.order_type == "market" else money_or_missing(intent.limit_price),
                    "Plain English": "Highest price the app plans to pay.",
                },
                {"Section": "Decision", "Item": "Stop loss", "Status / Value": money_or_missing(intent.stop_loss), "Plain English": "Current protective stop level."},
                {"Section": "Decision", "Item": "Dollars at risk", "Status / Value": money_or_missing(dollars_at_risk), "Plain English": "Estimated loss if the stop is hit."},
                {"Section": "Decision", "Item": "Account risk", "Status / Value": pct_or_missing(risk_pct_account), "Plain English": "Estimated risk as a percent of the account used for sizing."},
            ]
        )

    required_reads = required_setup_reads(strategy_type, rsi_entry_filter_enabled)
    for source in setup_scorecard_rows:
        read = str(source.get("Read", ""))
        if read == "Overall":
            continue
        section = "Required for BUY" if read in required_reads else "Quality check"
        rows.append(
            {
                "Section": section,
                "Item": read,
                "Status / Value": source.get("Status", ""),
                "Plain English": source.get("Plain English", ""),
            }
        )
    return rows


def automation_watch_records() -> list[dict]:
    watched = []
    if active_automation_level == "Auto exits only":
        watched.append("open paper positions")
    elif active_automation_level == "Auto entries and exits":
        watched.append(f"{ticker} buy setup")
        watched.append("open paper positions")
    if auto_cancel_stale_limit_orders:
        watched.append("old BUY limit orders")
    return [
        {"Item": "Automation session started", "Value": st.session_state.get("automation_session_started_at", "Not started")},
        {"Item": "Watching", "Value": ", ".join(watched) if watched else "Nothing automatic right now"},
        {"Item": "Check interval", "Value": f"{automation_refresh_seconds} seconds"},
        {"Item": "Last check", "Value": st.session_state.get("last_automation_checked_at", "Not checked yet")},
        {"Item": "Last action", "Value": st.session_state.get("last_automation_action", "None")},
    ]


def auto_entry_session_safety_blockers(symbol: str) -> list[str]:
    events = st.session_state.get("session_audit_events", [])
    auto_buys = [event for event in events if getattr(event, "event_type", "") == "auto_paper_entry_submitted"]
    if len(auto_buys) >= max_auto_buys_per_session:
        return [f"Automatic buys reached the session limit of {max_auto_buys_per_session}."]
    if reentry_cooldown_minutes <= 0 or not symbol:
        return []
    latest_exit_time = None
    for event in events:
        if getattr(event, "event_type", "") not in {"auto_paper_exit_submitted", "alpaca_paper_exit_submitted"}:
            continue
        payload = getattr(event, "payload", {}) or {}
        if str(payload.get("symbol", "")).strip().upper() != symbol:
            continue
        try:
            event_time = pd.Timestamp(payload.get("checked_at") or getattr(event, "created_at", None))
            if event_time.tzinfo is None:
                event_time = event_time.tz_localize("America/Los_Angeles")
            latest_exit_time = max(latest_exit_time, event_time) if latest_exit_time is not None else event_time
        except Exception:
            continue
    if latest_exit_time is None:
        return []
    elapsed_minutes = (pd.Timestamp.now(tz="America/Los_Angeles") - latest_exit_time.tz_convert("America/Los_Angeles")).total_seconds() / 60
    if elapsed_minutes < reentry_cooldown_minutes:
        remaining = reentry_cooldown_minutes - elapsed_minutes
        return [f"Waiting {remaining:.0f} more minutes before another automatic buy in {symbol}."]
    return []


def saved_buy_settings_for_symbol(symbol: str, tracked_orders: list[dict]) -> dict | None:
    clean_symbol = str(symbol).strip().upper()
    matching_buys = [
        order
        for order in tracked_orders
        if str(order.get("symbol", "")).strip().upper() == clean_symbol
        and str(order.get("side", "")).strip().lower() == "buy"
        and order.get("strategy_settings")
    ]
    if not matching_buys:
        return None
    latest = matching_buys[-1]
    settings = dict(latest.get("strategy_settings", {}))
    settings.setdefault("entry_submitted_at", latest.get("submitted_at", ""))
    settings.setdefault("entry_broker_order_id", latest.get("broker_order_id", ""))
    return settings


def saved_exit_settings_for_symbol(symbol: str, tracked_orders: list[dict]) -> dict | None:
    clean_symbol = str(symbol).strip().upper()
    matching_buys = [
        order
        for order in tracked_orders
        if str(order.get("symbol", "")).strip().upper() == clean_symbol
        and str(order.get("side", "")).strip().lower() == "buy"
        and (order.get("exit_settings") or order.get("strategy_settings"))
    ]
    if not matching_buys:
        return None
    def priority(order: dict) -> int:
        status = normalized_order_status(order.get("status", order.get("Status", "")))
        source = str(order.get("source", "")).strip().lower()
        if source == "position_exit_settings" or status == "managed_exit_settings":
            return 3
        if status in {"filled", "partially_filled"}:
            return 2
        if source == "adopted_alpaca_position":
            return 1
        return 0

    latest = max(enumerate(matching_buys), key=lambda item: (priority(item[1]), item[0]))[1]
    settings = dict(latest.get("exit_settings") or latest.get("strategy_settings") or {})
    settings.setdefault("entry_submitted_at", latest.get("submitted_at", ""))
    settings.setdefault("entry_broker_order_id", latest.get("broker_order_id", ""))
    return settings


def update_exit_settings_for_symbol(symbol: str, tracked_orders: list[dict], exit_settings: dict) -> list[dict]:
    clean_symbol = str(symbol).strip().upper()
    updated = []
    matched = False
    for order in tracked_orders:
        record = dict(order)
        if str(record.get("symbol", "")).strip().upper() == clean_symbol and str(record.get("side", "")).strip().lower() == "buy":
            record["exit_settings"] = dict(exit_settings)
            matched = True
        updated.append(record)
    if not matched:
        placeholder = {
            "broker_order_id": f"managed-exit-{clean_symbol}",
            "symbol": clean_symbol,
            "side": "buy",
            "quantity": "",
            "status": "managed_exit_settings",
            "source": "position_exit_settings",
            "exit_settings": dict(exit_settings),
        }
        updated.append(placeholder)
    return updated


def alpaca_order_row_id(row: dict) -> str:
    return str(row.get("Alpaca Order ID") or row.get("Broker Order ID") or row.get("Order ID") or "").strip()


def tracked_order_state_signature(orders: list[dict]) -> tuple[tuple[str, str, str, str, str], ...]:
    return tuple(
        (
            str(order.get("broker_order_id", order.get("Broker Order ID", ""))).strip(),
            str(order.get("status", order.get("Status", ""))).strip().lower(),
            str(order.get("lifecycle_status", "")).strip().lower(),
            str(order.get("filled_quantity", order.get("Filled Qty", ""))).strip(),
            str(order.get("average_fill_price", order.get("Avg Fill", ""))).strip(),
        )
        for order in orders
    )


def active_tracked_preview_hashes(orders: list[dict]) -> set[str]:
    return {
        str(order.get("preview_hash", "")).strip()
        for order in orders
        if str(order.get("preview_hash", "")).strip()
        and str(order.get("status", order.get("Status", ""))).strip().lower() in ACTIVE_ALPACA_ORDER_STATUSES
    }


def sync_auto_entry_sent_hashes_with_open_orders() -> None:
    active_hashes = active_tracked_preview_hashes(st.session_state.get("tracked_alpaca_orders", []))
    st.session_state["auto_entry_sent_hashes"] = [
        preview_hash
        for preview_hash in st.session_state.get("auto_entry_sent_hashes", [])
        if preview_hash in active_hashes
    ]


def refresh_tracked_alpaca_orders_from_broker() -> tuple[list[dict], bool]:
    tracked_orders = st.session_state.get("tracked_alpaca_orders", [])
    if not alpaca_status.connected or not tracked_orders:
        return tracked_orders, False
    tracked_refresh_rows = alpaca_adapter.refreshed_tracked_order_records(tracked_orders)
    lifecycle_alpaca_orders = {alpaca_order_row_id(row): row for row in alpaca_orders if alpaca_order_row_id(row)}
    for row in tracked_refresh_rows:
        row_id = alpaca_order_row_id(row)
        if row_id:
            lifecycle_alpaca_orders[row_id] = row
    lifecycle_order_rows = list(lifecycle_alpaca_orders.values())
    if not lifecycle_order_rows:
        return tracked_orders, False
    before = tracked_order_state_signature(tracked_orders)
    refreshed_orders = refresh_tracked_alpaca_orders(tracked_orders, lifecycle_order_rows)
    changed = tracked_order_state_signature(refreshed_orders) != before
    if changed:
        broker_state_store.replace_all(refreshed_orders)
        st.session_state["tracked_alpaca_orders"] = refreshed_orders
        sync_auto_entry_sent_hashes_with_open_orders()
    return refreshed_orders, changed


def show_blockers(title: str, blockers: list[str]) -> None:
    cleaned = [reason for reason in dict.fromkeys(blockers) if reason]
    if cleaned:
        st.warning(f"{title}: " + " ".join(cleaned))


def build_chart(
    prices,
    smas,
    atrs,
    entry_w,
    exit_w,
    ma_w,
    labels,
    trade_log,
    selected_trade=None,
    *,
    strategy_type="breakout",
    pullback_w=20,
    market_data=None,
):
    x = labels
    highs = market_data["High"].to_numpy(dtype=float) if market_data is not None and "High" in market_data else prices
    lows = market_data["Low"].to_numpy(dtype=float) if market_data is not None and "Low" in market_data else prices
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=prices.tolist(), name="Price", mode="lines",
        line=dict(color=CHART_COLORS["price"], width=1.6),
        hovertemplate="%{x}<br>Price: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=smas, name=f"{ma_w}-bar trend filter", mode="lines",
        line=dict(color=CHART_COLORS["trend"], width=1.2, dash="dot"),
        hovertemplate="%{x}<br>Trend filter: $%{y:.2f}<extra></extra>",
    ))
    if strategy_type == "breakout":
        entry_line = [float(np.max(highs[i - entry_w:i])) if i >= entry_w else None for i in range(len(prices))]
        exit_line = [float(np.min(lows[i - exit_w:i])) if i >= exit_w else None for i in range(len(prices))]
        fig.add_trace(go.Scatter(
            x=x, y=entry_line, name=f"{entry_w}-bar entry high", mode="lines",
            line=dict(color=CHART_COLORS["entry"], width=1.1, dash="dash"),
            hovertemplate="%{x}<br>Entry high: $%{y:.2f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=exit_line, name=f"{exit_w}-bar exit low", mode="lines",
            line=dict(color=CHART_COLORS["exit"], width=1.1, dash="dash"),
            hovertemplate="%{x}<br>Exit low: $%{y:.2f}<extra></extra>",
        ))
    elif strategy_type == "pullback":
        pullback_line = [float(np.mean(prices[i - pullback_w + 1:i + 1])) if i + 1 >= pullback_w else None for i in range(len(prices))]
        exit_line = [float(np.mean(prices[i - exit_w + 1:i + 1])) if i + 1 >= exit_w else None for i in range(len(prices))]
        fig.add_trace(go.Scatter(
            x=x, y=pullback_line, name=f"{pullback_w}-bar pullback average", mode="lines",
            line=dict(color=CHART_COLORS["entry"], width=1.1, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=x, y=exit_line, name=f"{exit_w}-bar exit average", mode="lines",
            line=dict(color=CHART_COLORS["exit"], width=1.1, dash="dash"),
        ))
    else:
        exit_line = [float(np.min(lows[i - exit_w:i])) if i >= exit_w else None for i in range(len(prices))]
        fig.add_trace(go.Scatter(
            x=x, y=exit_line, name=f"{exit_w}-bar exit low", mode="lines",
            line=dict(color=CHART_COLORS["exit"], width=1.1, dash="dash"),
            hovertemplate="%{x}<br>Exit low: $%{y:.2f}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=x, y=atrs, name="ATR (14 bars)", mode="lines",
        line=dict(color=CHART_COLORS["atr"], width=1, dash="dot"),
        yaxis="y2",
        hovertemplate="%{x}<br>ATR: $%{y:.2f}<extra></extra>",
    ))

    if trade_log:
        fig.add_trace(go.Scatter(
            x=[t["entry_date"] for t in trade_log],
            y=[t["entry"] for t in trade_log],
            name="Entry", mode="markers",
            marker=dict(symbol="triangle-up", size=9, color=CHART_COLORS["entry"], opacity=0.75),
            customdata=[t["trade"] for t in trade_log],
            hovertemplate="Entry #%{customdata}<br>%{x}: $%{y:.2f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[t["exit_date"] for t in trade_log],
            y=[t["exit"] for t in trade_log],
            name="Exit", mode="markers",
            marker=dict(symbol="triangle-down", size=9, color=CHART_COLORS["sell"], opacity=0.75),
            customdata=[t["trade"] for t in trade_log],
            hovertemplate="Exit #%{customdata}<br>%{x}: $%{y:.2f}<extra></extra>",
        ))

    if selected_trade is not None:
        t = selected_trade
        ei = t["entry_bar"]
        xi = t["exit_bar"]
        band_x = x[ei:xi + 1]
        band_y = prices[ei:xi + 1].tolist()
        fill = "rgba(93,191,138,0.10)" if t["pnl"] >= 0 else "rgba(232,100,90,0.10)"
        fig.add_trace(go.Scatter(
            x=band_x + band_x[::-1],
            y=band_y + [t["entry"]] * len(band_y),
            fill="toself", fillcolor=fill, line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[t["entry_date"]], y=[t["entry"]], mode="markers+text",
            marker=dict(symbol="triangle-up", size=16, color=CHART_COLORS["entry"]),
            text=[f" BUY #{t['trade']}"], textposition="middle right",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[t["exit_date"]], y=[t["exit"]], mode="markers+text",
            marker=dict(symbol="triangle-down", size=16, color=CHART_COLORS["sell"]),
            text=[f" SELL #{t['trade']}"], textposition="middle right",
            showlegend=False,
        ))

    fig.update_layout(
        height=410,
        margin=dict(l=18, r=18, t=42, b=38),
        font=dict(color=CHART_COLORS["text"], family="Inter, Segoe UI, sans-serif", size=11),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=CHART_COLORS["text"]),
        ),
        xaxis=dict(
            title=None, showgrid=False, type="category", nticks=10,
            linecolor=CHART_COLORS["border"], tickfont=dict(color=CHART_COLORS["text"]),
        ),
        yaxis=dict(
            tickprefix="$", gridcolor=CHART_COLORS["grid"], zeroline=False,
            tickfont=dict(color=CHART_COLORS["text"]),
        ),
        yaxis2=dict(
            tickprefix="$", overlaying="y", side="right", showgrid=False, zeroline=False,
            tickfont=dict(color=CHART_COLORS["text"]),
        ),
        plot_bgcolor=CHART_COLORS["surface"],
        paper_bgcolor=CHART_COLORS["surface"],
        hoverlabel=dict(bgcolor="#202730", bordercolor=CHART_COLORS["border"], font_color="#E8EDF3"),
        hovermode="x unified",
    )
    return fig


alpaca_config = AlpacaConfig.from_env()

st.sidebar.markdown('<div class="sidebar-kill-title">Kill Switch</div>', unsafe_allow_html=True)
kill_switch = st.sidebar.checkbox(
    "Enabled",
    value=False,
    help="Immediate hard stop. When this is on, the app blocks new paper orders and automation actions.",
)
automation_level_options = {
    "Manual - I click paper orders": "Manual review only",
    "Auto exits - app sells paper positions": "Auto exits only",
    "Auto entries and exits - app trades paper": "Auto entries and exits",
}
st.sidebar.markdown("### :material/smart_toy: Background Automation")
automation_level_label = st.sidebar.selectbox(
    "Automation",
    list(automation_level_options.keys()),
    index=0,
    help=(
        "Manual means the app never sends an automatic order. Auto exits lets the open Streamlit page or the Background Worker sell paper positions from their saved exit plans. "
        "Auto entries and exits also permits automatic paper buys for the loaded ticker; the Buy watchlist is monitored only by the Background Worker."
    ),
)
automation_level = automation_level_options[automation_level_label]
if kill_switch:
    st.session_state["enable_full_paper_automation"] = False
full_automation_enabled = False
if automation_level == "Auto entries and exits":
    full_automation_enabled = st.sidebar.checkbox(
        "Allow automatic paper buys",
        value=bool(st.session_state.get("enable_full_paper_automation", False)),
        key="enable_full_paper_automation",
        disabled=kill_switch,
        help=(
            "Required before the app may send an automatic BUY to Alpaca paper. It does not control automatic exits. "
            "The Kill Switch turns this permission off."
        ),
    )
    full_automation_enabled = bool(full_automation_enabled and not kill_switch)
    if not full_automation_enabled:
        st.sidebar.caption("Automatic paper buys are not allowed.")

active_automation_level = resolve_active_automation_level(
    automation_level,
    full_automation_enabled=full_automation_enabled,
    kill_switch_enabled=kill_switch,
)

sidebar_worker_status_store = WorkerStatusStore()
sidebar_worker_status = sidebar_worker_status_store.read()
sidebar_worker_active = worker_status_is_active(sidebar_worker_status)
sidebar_worker_present = bool(
    sidebar_worker_active
    or sidebar_worker_status.running
    or DEFAULT_LOCK_PATH.exists()
)
sidebar_control_store = AutomationControlStore()
if "background_worker_enabled" not in st.session_state:
    st.session_state["background_worker_enabled"] = sidebar_worker_active
if "worker_stop_pending" not in st.session_state:
    st.session_state["worker_stop_pending"] = False
if not sidebar_worker_active:
    st.session_state["worker_stop_pending"] = False
if kill_switch:
    st.session_state["background_worker_enabled"] = False
worker_status_text = (
    "Running"
    if sidebar_worker_active
    else "Needs attention"
    if sidebar_worker_present
    else "Stopped"
)
worker_behavior_text = (
    "Monitoring continues if Streamlit closes; the Buy watchlist is active."
    if sidebar_worker_active
    else "The heartbeat is stale. Stop Worker will terminate the verified worker process."
    if sidebar_worker_present
    else "Only the open Streamlit page can check the loaded ticker and exits; the Buy watchlist is paused."
)
st.sidebar.caption(f"Worker: {worker_status_text}. {worker_behavior_text}")
worker_control_cols = st.sidebar.columns(2)
if worker_control_cols[0].button(
    "Start Worker",
    disabled=kill_switch or sidebar_worker_active,
    help="Start background monitoring so saved exits and the Buy watchlist continue when Streamlit is closed.",
):
    st.session_state["background_worker_enabled"] = True
    st.session_state["worker_stop_pending"] = False
    st.session_state["start_background_worker_requested"] = True
    st.session_state["stop_background_worker_requested"] = False
if worker_control_cols[1].button(
    "Stop Worker",
    disabled=not sidebar_worker_present,
    help="Stop background monitoring. The open Streamlit page can still check the loaded ticker and saved exits.",
):
    st.session_state["background_worker_enabled"] = False
    st.session_state["worker_stop_pending"] = True
    st.session_state["stop_background_worker_requested"] = True
    st.session_state["start_background_worker_requested"] = False
    existing_control = sidebar_control_store.read()
    sidebar_control_store.write(replace(existing_control, enabled=False, stop_requested=True))
    sidebar_worker_status_store.write(
        replace(
            sidebar_worker_status_store.read(),
            state="Stopping",
            last_checked_at=datetime.now().astimezone().isoformat(),
            last_action="Stop requested from the Streamlit sidebar.",
            last_error="",
        )
    )

st.sidebar.markdown("### :material/space_dashboard: Navigation")
workspace_mode = st.sidebar.radio(
    "Workspace",
    ["Daily Trading Screen", "Full Records and Evidence *"],
    index=0,
    help="Daily Trading Screen shows the controls you use day to day. Full Records and Evidence adds logs, setup records, and extra proof tables. An asterisk marks sections that only appear in Full Records and Evidence.",
)
show_portfolio_evidence = workspace_mode == "Full Records and Evidence *"

st.sidebar.markdown("### :material/account_balance_wallet: Paper trading")
mode_options = {
    "Backtest only - no orders are sent": "backtest_only",
    "Paper trading - send orders to Alpaca paper": "paper",
    "Practice mode - record decisions only": "shadow",
    "Live with approval - real orders": "live_with_approval",
    "Automated live - setup only": "automated_live",
}
mode_label = st.sidebar.selectbox(
    "Order mode",
    list(mode_options.keys()),
    index=0,
    help=(
        "Backtest only uses the chart and simulator. Paper trading can send orders to Alpaca paper. "
        "Practice mode records decisions without sending broker orders. Live with approval can send real orders only after live credentials and confirmation are configured. Automated live is visible for setup but does not auto-submit yet."
    ),
)
execution_mode = mode_options[mode_label]
enable_alpaca_paper_orders = bool(execution_mode == "paper" and alpaca_config.paper)
if execution_mode == "paper":
    st.sidebar.caption("Orders use the connected Alpaca paper account.")
enable_alpaca_live_orders = False
if execution_mode in {"live_with_approval", "automated_live"}:
    enable_alpaca_live_orders = st.sidebar.checkbox(
        "Enable Live Orders",
        value=False,
        disabled=kill_switch,
        help="Allows live Alpaca order buttons only when the live endpoint, live env switch, and live confirmation are configured. The Kill Switch turns this off.",
    )
    if execution_mode == "automated_live":
        st.sidebar.caption("Automated live is configured here, but automatic live submission is still off. Use live with approval first.")
alpaca_order_submission_enabled = (
    bool(enable_alpaca_paper_orders)
    if alpaca_config.paper
    else bool(enable_alpaca_live_orders and execution_mode == "live_with_approval" and not kill_switch)
)
automation_refresh_seconds = st.sidebar.selectbox(
    "Check automation every",
    [5, 15, 30, 60],
    index=1,
    format_func=lambda seconds: f"{seconds} seconds",
    help="How often the app checks for automatic paper buys or sells while automation is on.",
)
background_worker_enabled = bool(st.session_state.get("background_worker_enabled", False))
paper_buy_order_style = st.sidebar.selectbox(
    "Paper buy price",
    ["Limit below current price", "Limit at current price", "Limit above current price", "Custom limit price", "Market"],
    index=1,
    help=(
        "Limit below current price waits for a pullback. Limit at current price uses the current reference price. "
        "Limit above current price adds a small cushion. Custom limit price lets you type the exact max buy price. "
        "Market buys immediately at the available market price."
    ),
)
auto_cancel_stale_limit_orders = st.sidebar.checkbox(
    "Auto-cancel old limit buys",
    value=False,
    help="When this is on, the app cancels unfilled paper BUY limit orders after the waiting time you choose below.",
)
stale_limit_order_label = st.sidebar.selectbox(
    "Cancel unfilled limit buy after",
    list(STALE_LIMIT_ORDER_OPTIONS.keys()),
    index=4,
    disabled=not auto_cancel_stale_limit_orders,
    help="Default is 1 hour. Shorter times keep stale limit orders from lingering; longer times give the order more time to fill.",
)
stale_limit_order_minutes = STALE_LIMIT_ORDER_OPTIONS[stale_limit_order_label]
allow_limit_buys_outside_market_hours = st.sidebar.checkbox(
    "Allow limit buys outside market hours",
    value=False,
    disabled=paper_buy_order_style == "Market",
    help=(
        "Allows automatic paper BUY limit orders while the regular market is closed. "
        "Market orders still wait for regular market hours."
    ),
)
with st.sidebar.expander("Advanced safety", expanded=False):
    allowed_symbols_text = st.text_input(
        "Allowed symbols",
        value="",
        help=(
            "Optional whitelist for paper orders. Leave blank to allow the ticker you typed and every queued setup. "
            "Use commas to restrict both manual and queued paper orders to specific tickers, such as AAPL, MSFT, NVDA."
        ),
    )
    allow_add_to_existing_position = st.checkbox(
        "Allow adding to an existing paper position",
        value=False,
        help=(
            "When this is on, a new BUY can add shares to an Alpaca paper position you already hold. "
            "Risk limits, cash, concentration, and open-order checks still apply."
        ),
    )
    max_auto_buys_per_session = st.number_input(
        "Max automatic buys this session",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
        help="Caps app-triggered automatic paper buys until you reset the paper session or restart the app.",
    )
    reentry_cooldown_minutes = st.selectbox(
        "Wait after an exit before re-buying",
        [0, 15, 30, 60, 120, 240],
        index=3,
        format_func=lambda minutes: "No wait" if minutes == 0 else f"{minutes} minutes",
        help="Prevents the app from automatically re-buying the same ticker right after an app-recorded paper exit.",
    )
allowed_symbols = tuple(s.strip().upper() for s in allowed_symbols_text.split(",") if s.strip())
paper_buy_limit_adjustment_pct = 0.0
paper_buy_custom_limit_price = 0.0
if paper_buy_order_style == "Limit below current price":
    paper_buy_limit_adjustment_pct = st.sidebar.number_input(
        "Buy limit discount (%)",
        min_value=0.00,
        max_value=10.00,
        value=0.25,
        step=0.01,
        format="%.2f",
        help="Subtracts this percent from the current reference price. Example: 0.50% on $100 sets a $99.50 buy limit.",
    )
elif paper_buy_order_style == "Limit above current price":
    paper_buy_limit_adjustment_pct = st.sidebar.number_input(
        "Buy limit cushion (%)",
        min_value=0.00,
        max_value=2.00,
        value=0.10,
        step=0.01,
        format="%.2f",
        help="Adds this percent above the current reference price. Example: 0.10% on $100 sets a $100.10 buy limit.",
    )
elif paper_buy_order_style == "Custom limit price":
    paper_buy_custom_limit_price = st.sidebar.number_input(
        "Exact buy limit price ($)",
        min_value=0.00,
        value=0.00,
        step=0.01,
        format="%.2f",
        help="The order will not buy above this exact price. Leave at 0.00 to block sending until you type a price.",
)
reset_paper_broker = False

st.sidebar.markdown("### :material/candlestick_chart: Ticker and price data")
data_source = st.sidebar.radio(
    "Prices to use",
    ["Synthetic", "Ticker (Alpaca)", "Ticker (yfinance)"],
    horizontal=True,
)
market_data = None
source_caption = "synthetic price data"
ticker = "SYNTH"
interval = "1d"
period = "synthetic"

if data_source in ("Ticker (Alpaca)", "Ticker (yfinance)"):
    ticker = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()
    interval_options = ["1d", "4h", "1h", "30m", "15m", "5m", "1m"]
    optimizer_interval_to_apply = st.session_state.pop("optimizer_apply_interval", None)
    if optimizer_interval_to_apply in interval_options:
        st.session_state["price_interval_input"] = optimizer_interval_to_apply
    interval = st.sidebar.selectbox("Interval", interval_options, index=2, key="price_interval_input")
    if interval == "1d":
        period_options, period_index = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"], 5
    elif interval == "4h" and data_source == "Ticker (Alpaca)":
        period_options, period_index = ["1mo", "3mo", "6mo", "1y", "2y", "5y"], 3
    elif interval == "1h" and data_source == "Ticker (Alpaca)":
        period_options, period_index = ["1mo", "3mo", "6mo", "1y", "2y"], 3
    elif interval in ("4h", "1h"):
        period_options, period_index = ["1mo", "3mo", "6mo", "1y"], 3
    elif interval in ("30m", "15m"):
        period_options, period_index = ["1mo", "3mo", "6mo"], 2
    else:
        period_options, period_index = ["1d", "5d", "1mo"], 2
    optimizer_history_to_apply = st.session_state.pop("optimizer_apply_history", None)
    if optimizer_history_to_apply in period_options:
        st.session_state["history_period_input"] = optimizer_history_to_apply
    elif st.session_state.get("history_period_input") not in period_options:
        st.session_state["history_period_input"] = period_options[period_index]
    period = st.sidebar.selectbox("History period", period_options, index=period_index, key="history_period_input")
    if st.sidebar.button("Refresh stock data", type="primary"):
        fetch_alpaca_stock_data.clear()
        fetch_stock_data.clear()
    try:
        with st.spinner(f"Fetching {ticker}..."):
            if data_source == "Ticker (Alpaca)":
                market_data_config = AlpacaConfig.from_env()
                market_data = fetch_alpaca_stock_data(ticker, period, interval, market_data_config.api_key, market_data_config.api_secret)
                source_caption = f"{ticker} via Alpaca IEX ({period}, {interval}); latest completed bar {market_data.index[-1]}"
                st.sidebar.caption(f"Loaded {len(market_data):,} completed Alpaca bars. Free IEX data can be delayed.")
            else:
                market_data = fetch_stock_data(ticker, period, interval)
                source_caption = f"{ticker} via yfinance ({period}, {interval}); latest completed bar {market_data.index[-1]}"
                st.sidebar.caption(f"Loaded {len(market_data):,} completed yfinance bars. Yahoo intraday data may be delayed or limited.")
    except Exception as exc:
        st.error(f"Could not load {data_source} price data: {exc}")
        st.stop()
st.session_state["last_loaded_symbol"] = ticker

optimizer_apply_settings = st.session_state.pop("optimizer_apply_settings", None)
if optimizer_apply_settings:
    st.session_state["strategy_label_input"] = optimizer_apply_settings.get("strategy_label", "Trendline retest continuation")
    st.session_state["entry_window_input"] = int(optimizer_apply_settings.get("entry_window", 20))
    st.session_state["exit_window_input"] = int(optimizer_apply_settings.get("exit_window", 10))
    st.session_state["atr_stop_multiplier_input"] = float(optimizer_apply_settings.get("atr_stop_multiplier", 2.0))
    optimizer_risk_pct = float(optimizer_apply_settings.get("risk_per_trade_pct", 1.0))
    st.session_state["risk_pct_input"] = max(0.5, min(3.0, round(optimizer_risk_pct * 2) / 2))
    st.session_state["moving_average_window_input"] = int(optimizer_apply_settings.get("moving_average_window", 50))
    st.session_state["pullback_average_length_input"] = int(optimizer_apply_settings.get("pullback_average_length", 20))
    st.session_state["momentum_turn_length_input"] = int(optimizer_apply_settings.get("momentum_turn_length", 10))
    st.session_state["rsi_entry_filter_input"] = bool(optimizer_apply_settings.get("rsi_entry_filter_enabled", False))

st.sidebar.markdown("### :material/tune: Strategy settings")
account = st.sidebar.number_input(
    "Simulator account size ($)",
    min_value=1000,
    max_value=10_000_000,
    value=100000,
    step=1000,
    help="Used by the local simulator and strategy sizing. Alpaca account balance is read separately from Alpaca.",
)
strategy_options = {
    "Breakout continuation": "breakout",
    "Trend pullback continuation": "pullback",
    "Trendline breakout": "trendline",
    "Trendline retest continuation": "trendline_retest",
}
strategy_label = st.sidebar.selectbox(
    "Strategy type",
    list(strategy_options.keys()),
    index=3,
    key="strategy_label_input",
    help=(
        "Choose the rule set that creates trade ideas. Breakout continuation uses prior highs. "
        "Trend pullback continuation uses a trend filter, a pullback average, and a momentum turn. "
        "Trendline breakout and Trendline retest continuation use descending resistance lines from swing highs."
    ),
)
strategy_type = strategy_options[strategy_label]
entry_w = st.sidebar.slider(
    "Buy breakout / trendline lookback (bars)",
    10,
    55,
    20,
    step=5,
    key="entry_window_input",
    help=(
        "Breakout continuation uses this many bars to find the prior high. "
        "Trendline breakout and Trendline retest continuation use this many bars to find descending swing-high resistance."
    ),
)
exit_w = st.sidebar.slider(
    "Sell exit length (bars)",
    5,
    30,
    10,
    step=5,
    key="exit_window_input",
    help=(
        "Controls the saved strategy exit line. Breakout continuation, Trendline breakout, and Trendline retest continuation "
        "use this many bars to find the recent low exit level. Trend pullback continuation uses this many bars to calculate "
        "the exit average. Higher gives trades more room; lower exits faster."
    ),
)
atr_mult = st.sidebar.number_input(
    "Stop distance (ATR multiplier)",
    min_value=0.50,
    max_value=5.00,
    value=2.00,
    step=0.01,
    format="%.2f",
    key="atr_stop_multiplier_input",
    help=(
        "Sets stop distance as a multiple of ATR. Example: 1.28 means 1.28x ATR, not 1.28%. "
        "Higher means a wider stop and smaller share size. Lower means a tighter stop and larger share size."
    ),
)
risk_pct = st.sidebar.slider(
    "Strategy risk per trade (%)",
    0.5,
    3.0,
    1.0,
    step=0.5,
    key="risk_pct_input",
    help="How much of the simulator account the strategy is allowed to risk on one trade before separate risk limits are applied.",
)
ma_w = st.sidebar.slider(
    "Trend filter length (bars)",
    50,
    300,
    50,
    step=50,
    key="moving_average_window_input",
    help="Calculates the trend filter. The app treats the ticker as healthier when price is above this average and the average is rising.",
)
pullback_w = st.sidebar.slider(
    "Pullback average length (bars)",
    10,
    200,
    20,
    step=5,
    key="pullback_average_length_input",
    help=(
        "Used by Trend pullback continuation. This calculates the moving average used as the pullback zone. "
        "The strategy looks for the recent low to touch or come near this average before buying. "
        "Shorter values create shallower pullback zones; longer values create deeper, slower zones."
    ),
)
momentum_w = st.sidebar.slider(
    "Momentum turn length (bars)",
    3,
    20,
    10,
    step=1,
    key="momentum_turn_length_input",
    help=(
        "Used by Trend pullback continuation and Trendline retest continuation. This calculates the short average used to confirm "
        "price has turned back up before buying. Shorter values react faster; longer values wait for more confirmation."
    ),
)
rsi_entry_filter_enabled = st.sidebar.checkbox(
    "Require RSI 50-70 for BUY",
    value=False,
    key="rsi_entry_filter_input",
    help=(
        "When on, every historical and current BUY must also have 14-bar RSI between 50 and 70. "
        "This changes simulated trades, backtest results, recommendations, and automated entries. "
        "It never creates a BUY by itself and does not control exits."
    ),
)

st.sidebar.markdown("### :material/shield: Risk limits")
max_risk_limit = st.sidebar.slider(
    "Max risk per trade (%)",
    0.25,
    5.0,
    1.0,
    step=0.25,
    help="Hard cap on dollars at risk for one trade. If the stop loss would risk more than this, the app reduces size or blocks the trade.",
)
max_notional_limit = st.sidebar.slider(
    "Max new order size (%)",
    5.0,
    100.0,
    5.0,
    step=5.0,
    help="Hard cap on each new buy order as a percent of account value. Total exposure to one ticker is controlled separately by Max symbol concentration.",
)
max_portfolio_exposure = st.sidebar.slider(
    "Max portfolio exposure (%)",
    10.0,
    100.0,
    80.0,
    step=5.0,
    help="Hard cap on total open exposure across all tracked positions and orders. This keeps the app from putting too much of the account to work at once.",
)
max_symbol_concentration = st.sidebar.slider(
    "Max symbol concentration (%)",
    5.0,
    100.0,
    5.0,
    step=5.0,
    help="Hard cap on exposure to one ticker. This prevents one symbol from becoming too large relative to the account.",
)
max_session_loss = st.sidebar.slider(
    "Max daily loss (%)",
    0.5,
    10.0,
    2.0,
    step=0.5,
    help="Hard cap on today's account loss. For Alpaca this compares current equity with Alpaca's prior-day equity; new buys are blocked after the limit is reached.",
)
max_open_positions = st.sidebar.slider(
    "Max open positions",
    1,
    20,
    20,
    step=1,
    help="Maximum number of positions the app can have open or tracked at the same time.",
)
st.sidebar.markdown("### :material/query_stats: Research options")
run_walk_forward = st.sidebar.checkbox(
    "Test on newer price data",
    value=True,
    help="Splits the price history into older data and newer data. The app checks whether the selected strategy still works on the newer bars instead of only fitting the older bars.",
)
train_fraction = st.sidebar.slider("Older data used first (%)", 55, 80, 65, step=5) / 100
max_parameter_candidates = st.sidebar.slider(
    "Settings to compare per strategy",
    4,
    24,
    12,
    step=4,
    help=(
        "How many nearby strategy-setting combinations to inspect. Each combination is tested twice: "
        "once without the RSI entry rule and once requiring RSI 50-70."
    ),
)
run_strategy_input_search = st.sidebar.button(
    "Run Strategy Input Search",
    help=(
        "Runs the four-strategy input search once and saves the result. For real tickers it compares daily, 4-hour, "
        "and 1-hour data against buy-and-hold. Ordinary page refreshes do not rerun it."
    ),
)
optimizer_sidebar_status = st.sidebar.empty()

st.sidebar.subheader(
    "Setup quality checks",
    help=(
        "These switches decide which quality checks appear in the research table. "
        "Trend, the selected strategy rule, and risk approval are always checked."
    ),
)
setup_inputs = {
    "breakout_strength": st.sidebar.checkbox(
        "Breakout strength",
        value=True,
        help="Shows whether price has moved clearly above the entry level. This helps grade the setup; it does not create a BUY by itself.",
    ),
    "volume": st.sidebar.checkbox(
        "Volume confirmation",
        value=True,
        help="Shows whether volume supports the move. Stronger volume can improve confidence; weak volume is a caution flag.",
    ),
    "volatility": st.sidebar.checkbox(
        "Volatility",
        value=True,
        help="Shows whether recent price movement is quiet, normal, or wide using ATR. Very wide movement can mean the setup is unstable.",
    ),
    "room_above_exit": st.sidebar.checkbox(
        "Room above exit",
        value=True,
        help="Checks whether price has enough room above the sell line compared with the stop distance.",
    ),
    "relative_strength": st.sidebar.checkbox(
        "Relative strength",
        value=False,
        help="Compares this ticker against a market benchmark when benchmark data is available.",
    ),
    "market_condition": st.sidebar.checkbox(
        "Market condition",
        value=False,
        help="Checks whether the broad market supports new long trades when market data is available.",
    ),
    "liquidity": st.sidebar.checkbox(
        "Liquidity",
        value=True,
        help="Checks whether average dollar volume is high enough for cleaner fills and less slippage.",
    ),
    "event_risk": st.sidebar.checkbox(
        "Event risk",
        value=False,
        help="Flags upcoming earnings or major news risk when an event calendar is connected.",
    ),
    "rsi": True,
}

with st.sidebar.expander("Files, records, and simulator reset", expanded=False):
    reset_paper_broker = st.button(
        "Reset local simulator account",
        help="Resets only the in-app simulator cash, simulated positions, and local session records. It does not touch Alpaca.",
    )
    persist_audit_log = st.checkbox("Save activity log", value=True)
    audit_log_path = st.text_input("Activity log file", value="audit_logs/agentloop_audit.jsonl")
    broker_state_path = st.text_input("Alpaca order file", value="broker_state/alpaca_paper_orders.json")
    automation_dry_run_path = st.text_input("Automation check file", value="automation_logs/paper_automation_dry_runs.jsonl")
    research_snapshot_path = st.text_input("Research loop file", value="automation_logs/research_snapshots.jsonl")
    run_manifest_path = st.text_input("Run summary file", value="audit_logs/run_manifests.jsonl")
    evidence_export_path = st.text_input("Records export file", value="audit_logs/latest_evidence_package.json")
    live_lockfile_path = st.text_input("Live trading lock file", value="live_mode/LIVE_TRADING_LOCKED.txt")
    automation_preview_interval = (
        st.slider("Automation check interval (min)", 5, 60, 15, step=5)
        if show_portfolio_evidence
        else 15
    )

if data_source == "Synthetic" and st.sidebar.button("Simulate new run", type="primary"):
    st.session_state["seed"] = np.random.randint(0, 100_000)
    st.session_state["selected_trade_idx"] = None

seed = st.session_state.get("seed", 42)
risk_dec = risk_pct / 100
if "paper_broker" not in st.session_state or st.session_state.get("paper_starting_cash") != account:
    st.session_state["paper_broker"] = PaperBroker(cash=float(account))
    st.session_state["paper_starting_cash"] = account
    st.session_state["paper_session_id"] = new_session_id()
    st.session_state["paper_session_started_at"] = pd.Timestamp.now(tz="America/Los_Angeles").isoformat()
    st.session_state["session_audit_events"] = []
    st.session_state["shadow_decisions"] = []
    st.session_state["last_audit_key"] = None
    st.session_state["auto_exit_sent_hashes"] = []
    st.session_state["auto_entry_sent_hashes"] = []
    st.session_state["last_auto_exit_decision_key"] = None
    st.session_state["last_auto_entry_decision_key"] = None
    st.session_state["last_automation_action"] = "None"
    st.session_state["last_automation_blocked_reason"] = ""
    st.session_state["automation_event_history"] = []
    st.session_state["automation_session_started_at"] = pd.Timestamp.now(tz="America/Los_Angeles").isoformat()
    st.session_state["tracked_alpaca_orders"] = []
    st.session_state["simulated_alpaca_positions"] = []
if reset_paper_broker:
    st.session_state["paper_broker"] = PaperBroker(cash=float(account))
    st.session_state["paper_starting_cash"] = account
    st.session_state["paper_session_id"] = new_session_id()
    st.session_state["paper_session_started_at"] = pd.Timestamp.now(tz="America/Los_Angeles").isoformat()
    st.session_state["session_audit_events"] = []
    st.session_state["shadow_decisions"] = []
    st.session_state["last_audit_key"] = None
    st.session_state["auto_exit_sent_hashes"] = []
    st.session_state["auto_entry_sent_hashes"] = []
    st.session_state["last_auto_exit_decision_key"] = None
    st.session_state["last_auto_entry_decision_key"] = None
    st.session_state["last_automation_action"] = "None"
    st.session_state["last_automation_blocked_reason"] = ""
    st.session_state["automation_event_history"] = []
    st.session_state["automation_session_started_at"] = pd.Timestamp.now(tz="America/Los_Angeles").isoformat()
    st.session_state["tracked_alpaca_orders"] = []
    st.session_state["simulated_alpaca_positions"] = []
st.session_state.setdefault("shadow_decisions", [])
st.session_state.setdefault("paper_session_id", new_session_id())
st.session_state.setdefault("paper_session_started_at", pd.Timestamp.now(tz="America/Los_Angeles").isoformat())
st.session_state.setdefault("auto_exit_sent_hashes", [])
st.session_state.setdefault("auto_entry_sent_hashes", [])
st.session_state.setdefault("last_auto_exit_decision_key", None)
st.session_state.setdefault("last_auto_entry_decision_key", None)
st.session_state.setdefault("last_automation_action", "None")
st.session_state.setdefault("last_automation_blocked_reason", "")
st.session_state.setdefault("automation_event_history", [])
st.session_state.setdefault("automation_session_started_at", pd.Timestamp.now(tz="America/Los_Angeles").isoformat())
st.session_state.setdefault("tracked_alpaca_orders", [])
st.session_state.setdefault("simulated_alpaca_positions", [])
audit_store = JsonlAuditStore(audit_log_path)
automation_store = AutomationDryRunStore(automation_dry_run_path)
research_snapshot_store = ResearchSnapshotStore(research_snapshot_path)
manifest_store = RunManifestStore(run_manifest_path)
broker_state_store = BrokerStateStore(broker_state_path)
if not st.session_state["tracked_alpaca_orders"]:
    st.session_state["tracked_alpaca_orders"] = broker_state_store.read()
paper_broker: PaperBroker = st.session_state["paper_broker"]
paper_adapter = PaperBrokerAdapter(paper_broker)
alpaca_adapter = AlpacaBrokerAdapterStub(
    config=alpaca_config,
    allow_order_submission=alpaca_order_submission_enabled,
)
broker_statuses = [paper_adapter.status(), alpaca_adapter.status()]
alpaca_status = broker_statuses[1]
alpaca_market_open = alpaca_adapter.market_is_open() if alpaca_status.connected else None
if alpaca_market_open is not None:
    current_market_advisory = {
        "Market Session": "open" if alpaca_market_open else "closed_or_extended",
        "Open": alpaca_market_open,
        "Timestamp": pd.Timestamp.now(tz="America/Los_Angeles").isoformat(),
        "Message": "Alpaca reports that the regular US equity session is open." if alpaca_market_open else "Alpaca reports that the regular US equity session is closed.",
    }
alpaca_account_mode = alpaca_adapter.config.account_mode
alpaca_account_label = "Alpaca paper" if alpaca_adapter.config.paper else "Alpaca live"
alpaca_order_noun = "paper" if alpaca_adapter.config.paper else "live"
alpaca_mode_matches_order_mode = (
    (execution_mode == "paper" and alpaca_adapter.config.paper)
    or (execution_mode == "live_with_approval" and not alpaca_adapter.config.paper)
)
alpaca_manual_order_mode = execution_mode in {"paper", "live_with_approval"}
alpaca_orders_enabled_for_mode = (
    enable_alpaca_paper_orders if alpaca_adapter.config.paper else enable_alpaca_live_orders
)
alpaca_positions = alpaca_adapter.position_records() if alpaca_status.connected else []
alpaca_orders = alpaca_adapter.order_records() if alpaca_status.connected else []
alpaca_account_records = alpaca_adapter.account_records() if alpaca_status.connected else []
alpaca_read_errors = getattr(alpaca_adapter, "read_errors", {})
alpaca_state_health = broker_state_health(
    alpaca_status.connected,
    None if "positions" in alpaca_read_errors else alpaca_positions,
    None if "orders" in alpaca_read_errors else alpaca_orders,
)
paper_positions_notional = sum(position.market_value for position in paper_broker.positions.values())
alpaca_positions_notional = sum(float(row.get("Market Value") or 0) for row in alpaca_positions)
alpaca_open_buy_notional = open_buy_order_notional(alpaca_orders)
paper_equity = paper_broker.cash + paper_positions_notional
session_pnl = paper_equity - st.session_state["paper_starting_cash"]
alpaca_account_equity = first_available_number(
    account_record_value(alpaca_account_records, "Portfolio Value"),
    account_record_value(alpaca_account_records, "Equity"),
)
alpaca_account_cash = first_available_number(
    account_record_value(alpaca_account_records, "Cash"),
    account_record_value(alpaca_account_records, "Buying Power"),
)
alpaca_last_equity = first_available_number(
    account_record_value(alpaca_account_records, "Last Equity"),
    alpaca_account_equity,
)
use_alpaca_account_for_paper_risk = bool(enable_alpaca_paper_orders and alpaca_status.connected and alpaca_account_equity)
paper_order_risk_equity = float(alpaca_account_equity) if use_alpaca_account_for_paper_risk else float(account)
paper_order_available_cash = (
    float(alpaca_account_cash)
    if use_alpaca_account_for_paper_risk and alpaca_account_cash is not None
    else float(paper_broker.cash)
)
paper_order_portfolio_notional = (
    alpaca_positions_notional + alpaca_open_buy_notional
    if use_alpaca_account_for_paper_risk
    else paper_positions_notional + alpaca_positions_notional + alpaca_open_buy_notional
)
paper_order_session_pnl = (
    float(alpaca_account_equity) - float(alpaca_last_equity)
    if use_alpaca_account_for_paper_risk and alpaca_last_equity is not None
    else session_pnl
)
paper_order_account_source = "Alpaca paper account" if use_alpaca_account_for_paper_risk else "Simulator account"
effective_kill_switch = kill_switch
risk_limits = RiskLimits(
    allowed_symbols=allowed_symbols,
    max_risk_per_trade_pct=max_risk_limit,
    max_position_notional_pct=max_notional_limit,
    max_portfolio_exposure_pct=max_portfolio_exposure,
    max_symbol_concentration_pct=max_symbol_concentration,
    max_session_loss_pct=max_session_loss,
    max_open_positions=max_open_positions,
    allow_add_to_existing_position=allow_add_to_existing_position,
    require_stop_loss=True,
    kill_switch_enabled=effective_kill_switch,
)

try:
    breakout_prices, breakout_smas, breakout_atrs, breakout_trade_log, breakout_live, breakout_stats, breakout_labels = simulate_turtle_strategy(
        account, entry_w, exit_w, atr_mult, risk_dec, ma_w, seed, market_data, risk_limits,
        rsi_entry_filter_enabled=rsi_entry_filter_enabled,
    )
    pullback_prices, pullback_smas, pullback_atrs, pullback_trade_log, pullback_live, pullback_stats, pullback_labels = simulate_trend_pullback_strategy(
        account=account,
        pullback_w=pullback_w,
        exit_w=exit_w,
        atr_mult=atr_mult,
        risk_pct_dec=risk_dec,
        trend_w=ma_w,
        momentum_w=momentum_w,
        seed=seed,
        market_data=market_data,
        risk_limits=risk_limits,
        rsi_entry_filter_enabled=rsi_entry_filter_enabled,
    )
    trendline_prices, trendline_smas, trendline_atrs, trendline_trade_log, trendline_live, trendline_stats, trendline_labels = simulate_trendline_breakout_strategy(
        account=account,
        trendline_w=entry_w,
        exit_w=exit_w,
        atr_mult=atr_mult,
        risk_pct_dec=risk_dec,
        ma_w=ma_w,
        seed=seed,
        market_data=market_data,
        risk_limits=risk_limits,
        rsi_entry_filter_enabled=rsi_entry_filter_enabled,
    )
    retest_prices, retest_smas, retest_atrs, retest_trade_log, retest_live, retest_stats, retest_labels = simulate_trendline_retest_strategy(
        account=account,
        trendline_w=entry_w,
        exit_w=exit_w,
        atr_mult=atr_mult,
        risk_pct_dec=risk_dec,
        ma_w=ma_w,
        momentum_w=momentum_w,
        seed=seed,
        market_data=market_data,
        risk_limits=risk_limits,
        rsi_entry_filter_enabled=rsi_entry_filter_enabled,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

strategy_results = {
    "Breakout continuation": {
        "prices": breakout_prices,
        "smas": breakout_smas,
        "atrs": breakout_atrs,
        "trade_log": breakout_trade_log,
        "live": breakout_live,
        "stats": breakout_stats,
        "labels": breakout_labels,
    },
    "Trend pullback continuation": {
        "prices": pullback_prices,
        "smas": pullback_smas,
        "atrs": pullback_atrs,
        "trade_log": pullback_trade_log,
        "live": pullback_live,
        "stats": pullback_stats,
        "labels": pullback_labels,
    },
    "Trendline breakout": {
        "prices": trendline_prices,
        "smas": trendline_smas,
        "atrs": trendline_atrs,
        "trade_log": trendline_trade_log,
        "live": trendline_live,
        "stats": trendline_stats,
        "labels": trendline_labels,
    },
    "Trendline retest continuation": {
        "prices": retest_prices,
        "smas": retest_smas,
        "atrs": retest_atrs,
        "trade_log": retest_trade_log,
        "live": retest_live,
        "stats": retest_stats,
        "labels": retest_labels,
    },
}
selected_strategy_result = strategy_results[strategy_label]
prices = selected_strategy_result["prices"]
smas = selected_strategy_result["smas"]
atrs = selected_strategy_result["atrs"]
trade_log = selected_strategy_result["trade_log"]
live = selected_strategy_result["live"]
latest_market_price = optional_float(market_data.attrs.get("latest_price")) if market_data is not None else None
if latest_market_price is not None:
    live["signal_bar_price"] = live.get("last_p")
    live["latest_price"] = latest_market_price
    live["trade_intent"] = reprice_trade_intent(live.get("trade_intent"), latest_market_price)
    live["last_p"] = latest_market_price
stats = selected_strategy_result["stats"]
labels = selected_strategy_result["labels"]
comparison_rows = strategy_comparison_records({
    "Breakout continuation": breakout_stats,
    "Trend pullback continuation": pullback_stats,
    "Trendline breakout": trendline_stats,
    "Trendline retest continuation": retest_stats,
})
for row in comparison_rows:
    row["Exit Style"] = "Strategy exit + break-even + ATR trail"
benchmark = None
if market_data is not None:
    benchmark = buy_and_hold_benchmark(
        market_data,
        float(account),
        allocated_capital=ticker_allocated_capital(float(account), risk_limits),
    )
    comparison_rows.append({
        "Strategy": "Buy and hold benchmark",
        "Allocated Return": f"{benchmark.return_percent:.2f}%",
        "Annualized Return": (
            "Not shown (period is 1 year or less)"
            if benchmark.annualized_return_percent is None
            else f"{benchmark.annualized_return_percent:.2f}%"
        ),
        "Account Return": f"{benchmark.account_return_percent:.2f}%",
        "Trades": "1 holding",
        "Win Rate": "Not applicable",
        "Allocated Worst Drop": f"{benchmark.max_drawdown_percent:.2f}%",
        "Profit Factor": "Not applicable",
        "Exit Style": "Held from first adjusted close to last adjusted close",
    })

walk_forward_result = None
walk_forward_error = None
if run_walk_forward:
    try:
        walk_forward_result = evaluate_walk_forward(
            account=account,
            entry_w=entry_w,
            exit_w=exit_w,
            atr_mult=atr_mult,
            risk_pct_dec=risk_dec,
            ma_w=ma_w,
            seed=seed,
            market_data=market_data,
            train_fraction=train_fraction,
            risk_limits=risk_limits,
            strategy_type=strategy_type,
            pullback_w=pullback_w,
            momentum_w=momentum_w,
            rsi_entry_filter_enabled=rsi_entry_filter_enabled,
        )
    except ValueError as exc:
        walk_forward_error = str(exc)

current_strategy_config = StrategyConfig(
    name=strategy_label,
    entry_window=entry_w,
    exit_window=exit_w,
    atr_stop_multiplier=atr_mult,
    risk_per_trade_pct=risk_pct,
    moving_average_window=ma_w,
)
current_strategy_settings = {
    "symbol": ticker,
    "interval": interval,
    "history": period,
    "price_data_source": data_source,
    "strategy_type": strategy_type,
    "strategy_label": strategy_label,
    "account_size": account,
    "entry_window": entry_w,
    "exit_window": exit_w,
    "atr_stop_multiplier": atr_mult,
    "risk_per_trade_pct": risk_pct,
    "moving_average_window": ma_w,
    "pullback_average_length": pullback_w,
    "momentum_turn_length": momentum_w,
    "rsi_entry_filter_enabled": rsi_entry_filter_enabled,
    "paper_buy_order_type": paper_buy_order_style,
    "paper_buy_limit_adjustment_pct": paper_buy_limit_adjustment_pct,
    "paper_buy_custom_limit_price": paper_buy_custom_limit_price,
    "auto_cancel_stale_limit_orders": auto_cancel_stale_limit_orders,
    "stale_limit_order_minutes": stale_limit_order_minutes,
    "allow_limit_buys_outside_market_hours": allow_limit_buys_outside_market_hours,
    "automation_refresh_seconds": automation_refresh_seconds,
    "allow_add_to_existing_position": allow_add_to_existing_position,
    "max_auto_buys_per_session": max_auto_buys_per_session,
    "reentry_cooldown_minutes": reentry_cooldown_minutes,
    "profit_protection_enabled": True,
    "breakeven_after_r": 1.0,
    "trail_after_r": 2.0,
    "trailing_atr_multiplier": 3.0,
    "confirm_exit_on_bar_close": True,
}
current_exit_settings = dict(current_strategy_settings)
current_exit_settings["auto_exit_enabled"] = True

automation_control_store = AutomationControlStore()
automation_worker_status_store = WorkerStatusStore()
buy_watchlist_store = BuyWatchlistStore()
previous_automation_control = automation_control_store.read()
start_worker_requested = bool(st.session_state.pop("start_background_worker_requested", False))
stop_worker_requested = bool(st.session_state.pop("stop_background_worker_requested", False))
if start_worker_requested:
    background_worker_enabled = True
if stop_worker_requested:
    background_worker_enabled = False
worker_stop_requested = bool(
    st.session_state.get("worker_stop_pending", False)
    or stop_worker_requested
    or (previous_automation_control.stop_requested and not start_worker_requested)
)
automation_control = AutomationControl(
    enabled=bool(background_worker_enabled and active_automation_level != "Manual review only"),
    stop_requested=worker_stop_requested,
    mode=active_automation_level,
    paper_orders_enabled=bool(enable_alpaca_paper_orders and execution_mode == "paper"),
    kill_switch_enabled=bool(effective_kill_switch),
    full_automation_enabled=bool(full_automation_enabled),
    allow_duplicate_positions=bool(allow_add_to_existing_position),
    allow_limit_buys_outside_market_hours=bool(allow_limit_buys_outside_market_hours),
    auto_cancel_limit_buys=bool(auto_cancel_stale_limit_orders),
    stale_limit_order_minutes=int(stale_limit_order_minutes),
    refresh_seconds=int(automation_refresh_seconds),
    symbol=ticker,
    price_data_source=data_source,
    history=period,
    interval=interval,
    strategy_settings=current_strategy_settings,
    risk_limits=asdict(risk_limits),
    order_style=paper_buy_order_style,
    limit_adjustment_pct=float(paper_buy_limit_adjustment_pct),
    custom_limit_price=float(paper_buy_custom_limit_price),
    account_size=float(paper_order_risk_equity),
    broker_state_path=broker_state_path,
    audit_log_path=audit_log_path,
    buy_watchlist_path=str(buy_watchlist_store.path),
)
automation_control_store.write(automation_control)
if start_worker_requested:
    try:
        worker_pid = start_worker_process(Path.cwd())
        automation_worker_status_store.write(
            replace(
                automation_worker_status_store.read(),
                running=True,
                pid=worker_pid,
                state="Starting",
                last_checked_at=datetime.now().astimezone().isoformat(),
                last_action="Started from the Streamlit sidebar.",
                last_error="",
            )
        )
    except Exception as exc:
        automation_worker_status_store.write(
            replace(
                automation_worker_status_store.read(),
                running=False,
                state="Start failed",
                last_checked_at=datetime.now().astimezone().isoformat(),
                last_action="Could not start background worker.",
                last_error=str(exc),
            )
        )
    st.rerun()
elif stop_worker_requested:
    request_worker_stop(
        automation_control_store,
        automation_worker_status_store,
        timeout_seconds=5.0,
    )
    st.session_state["worker_stop_pending"] = False
    st.rerun()
automation_worker_status = automation_worker_status_store.read()


def evaluate_exit_rule_details_from_settings(settings: dict | None) -> dict:
    if not settings:
        return {
            "ready": False,
            "reason": "No saved exit settings for this position; auto exit is paused.",
            "trigger_price": None,
        }
    if not bool(settings.get("auto_exit_enabled", True)):
        return {
            "ready": False,
            "reason": "Auto exit is off for this position.",
            "trigger_price": None,
        }
    try:
        symbol = str(settings.get("symbol", "")).strip().upper()
        history = str(settings.get("history", "1y"))
        saved_interval = str(settings.get("interval", "1d"))
        saved_strategy_type = str(settings.get("strategy_type", "breakout"))
        saved_account = float(settings.get("account_size") or account)
        saved_exit_w = int(settings.get("exit_window") or exit_w)
        saved_atr_mult = float(settings.get("atr_stop_multiplier") or atr_mult)
        saved_risk_dec = float(settings.get("risk_per_trade_pct") or risk_pct) / 100
        saved_ma_w = int(settings.get("moving_average_window") or ma_w)
        saved_pullback_w = int(settings.get("pullback_average_length") or pullback_w)
        saved_momentum_w = int(settings.get("momentum_turn_length") or momentum_w)
        saved_entry_w = int(settings.get("entry_window") or entry_w)
        if not symbol:
            return {"ready": False, "reason": "Saved exit settings do not include a ticker.", "trigger_price": None}
        if history == "synthetic":
            return {"ready": False, "reason": "Saved exit settings use synthetic data; auto exit needs real ticker data.", "trigger_price": None}
        saved_price_source = str(settings.get("price_data_source", "Ticker (yfinance)"))
        data = fetch_price_data_for_source(symbol, history, saved_interval, saved_price_source)
        if saved_strategy_type == "pullback":
            _, _, _, _, saved_live, _, _ = simulate_trend_pullback_strategy(
                account=saved_account,
                pullback_w=saved_pullback_w,
                exit_w=saved_exit_w,
                atr_mult=saved_atr_mult,
                risk_pct_dec=saved_risk_dec,
                trend_w=saved_ma_w,
                momentum_w=saved_momentum_w,
                seed=None,
                market_data=data,
            )
        elif saved_strategy_type == "trendline":
            _, _, _, _, saved_live, _, _ = simulate_trendline_breakout_strategy(
                account=saved_account,
                trendline_w=saved_entry_w,
                exit_w=saved_exit_w,
                atr_mult=saved_atr_mult,
                risk_pct_dec=saved_risk_dec,
                ma_w=saved_ma_w,
                seed=None,
                market_data=data,
            )
        elif saved_strategy_type == "trendline_retest":
            _, _, _, _, saved_live, _, _ = simulate_trendline_retest_strategy(
                account=saved_account,
                trendline_w=saved_entry_w,
                exit_w=saved_exit_w,
                atr_mult=saved_atr_mult,
                risk_pct_dec=saved_risk_dec,
                ma_w=saved_ma_w,
                momentum_w=saved_momentum_w,
                seed=None,
                market_data=data,
            )
        else:
            _, _, _, _, saved_live, _, _ = simulate_turtle_strategy(
                account=saved_account,
                entry_w=saved_entry_w,
                exit_w=saved_exit_w,
                atr_mult=saved_atr_mult,
                risk_pct_dec=saved_risk_dec,
                ma_w=saved_ma_w,
                seed=None,
                market_data=data,
            )
        current_price = first_available_number(data.attrs.get("latest_price"), saved_live.get("last_p"))
        strategy_exit_price = optional_float(saved_live.get("exit_level"))
        current_atr = optional_float(saved_live.get("last_atr"))
        profit_protection_enabled = bool(settings.get("profit_protection_enabled", True))
        breakeven_after_r = float(settings.get("breakeven_after_r", 1.0))
        trail_after_r = float(settings.get("trail_after_r", 2.0))
        trailing_atr_multiplier = float(settings.get("trailing_atr_multiplier", 3.0))
        matching_position = next(
            (position for position in alpaca_positions if str(position.get("Symbol", "")).strip().upper() == symbol),
            {},
        )
        position_avg_entry = optional_float(matching_position.get("Average Entry"))
        entry_reference_price = first_available_number(
            position_avg_entry,
            settings.get("planned_entry_price"),
            settings.get("entry_reference_price"),
            current_price,
        )
        saved_entry_reference = first_available_number(settings.get("planned_entry_price"), settings.get("entry_reference_price"))
        saved_entry_stop = optional_float(settings.get("entry_stop_loss"))
        initial_risk = first_available_number(
            settings.get("entry_stop_distance"),
            saved_entry_reference - saved_entry_stop
            if saved_entry_reference is not None and saved_entry_stop is not None
            else None,
        )
        if initial_risk is not None and initial_risk <= 0:
            initial_risk = None
        original_stop_price = first_available_number(
            entry_reference_price - initial_risk
            if entry_reference_price is not None and initial_risk is not None
            else None,
            saved_entry_stop,
            entry_reference_price - saved_atr_mult * current_atr
            if entry_reference_price is not None and current_atr is not None
            else None,
        )
        profit_r = (
            (current_price - entry_reference_price) / initial_risk
            if current_price is not None and entry_reference_price is not None and initial_risk
            else None
        )
        entry_time = parse_saved_time(settings.get("entry_filled_at") or settings.get("entry_submitted_at"))
        high_data = data.tail(1)
        if entry_time is not None and not data.empty:
            index = data.index
            if getattr(index, "tz", None) is None:
                entry_compare = entry_time.tz_convert("America/Los_Angeles").tz_localize(None)
            else:
                entry_compare = entry_time.tz_convert(index.tz)
            since_entry = data.loc[index >= entry_compare]
            if not since_entry.empty:
                high_data = since_entry
        current_high_since_entry = optional_float(high_data["High"].max()) if "High" in high_data.columns and not high_data.empty else None
        current_high_since_entry = max(
            [value for value in (current_high_since_entry, optional_float(data.attrs.get("latest_high"))) if value is not None],
            default=None,
        )
        saved_high_since_entry = optional_float(settings.get("highest_high_since_entry"))
        highest_high_since_entry = max(
            [value for value in [current_high_since_entry, saved_high_since_entry, entry_reference_price] if value is not None],
            default=None,
        )
        highest_profit_r = (
            (highest_high_since_entry - entry_reference_price) / initial_risk
            if highest_high_since_entry is not None and entry_reference_price is not None and initial_risk
            else None
        )
        breakeven_stop_price = (
            entry_reference_price
            if profit_protection_enabled
            and highest_profit_r is not None
            and highest_profit_r >= breakeven_after_r
            and entry_reference_price is not None
            else None
        )
        trailing_stop_price = (
            highest_high_since_entry - trailing_atr_multiplier * current_atr
            if profit_protection_enabled
            and highest_profit_r is not None
            and highest_profit_r >= trail_after_r
            and highest_high_since_entry is not None
            and current_atr is not None
            else None
        )
        saved_trigger_price = optional_float(settings.get("last_exit_trigger_price"))
        trigger_candidates = [
            ("strategy exit", strategy_exit_price),
            ("fill-adjusted initial stop", original_stop_price),
            ("break-even stop", breakeven_stop_price),
            ("ATR trail", trailing_stop_price),
            ("saved trigger", saved_trigger_price),
        ]
        usable_triggers = [(label, price) for label, price in trigger_candidates if price is not None]
        trigger_source, trigger_price = max(usable_triggers, key=lambda item: item[1]) if usable_triggers else ("exit rule", None)
        ready = bool(current_price is not None and trigger_price is not None and current_price <= trigger_price)
        reason = (
            f"Exit now because {symbol} is at or below the {trigger_source} at ${trigger_price:,.2f}."
            if ready and trigger_price
            else f"Hold. Auto exit will trigger if {symbol} falls to ${trigger_price:,.2f} or lower. This uses the highest active protection level."
            if trigger_price
            else str(saved_live.get("exit_reason", "Strategy exit rule is not triggered."))
        )
        state_changed = False
        if highest_high_since_entry is not None and highest_high_since_entry > (saved_high_since_entry or 0):
            state_changed = True
        if trigger_price is not None and trigger_price > (saved_trigger_price or 0):
            state_changed = True
        return {
            "ready": ready,
            "reason": reason,
            "trigger_price": trigger_price,
            "strategy_exit_price": strategy_exit_price,
            "atr_stop_price": original_stop_price,
            "original_stop_price": original_stop_price,
            "breakeven_stop_price": breakeven_stop_price,
            "trailing_stop_price": trailing_stop_price,
            "highest_high_since_entry": highest_high_since_entry,
            "profit_r": profit_r,
            "highest_profit_r": highest_profit_r,
            "initial_risk": initial_risk,
            "current_atr": current_atr,
            "state_changed": state_changed,
            "trigger_source": trigger_source,
            "interval": saved_interval,
            "exit_window": saved_exit_w,
            "atr_multiplier": saved_atr_mult,
            "trailing_atr_multiplier": trailing_atr_multiplier,
            "breakeven_after_r": breakeven_after_r,
            "trail_after_r": trail_after_r,
        }
    except Exception as exc:
        return {"ready": False, "reason": f"Could not check saved exit rule: {exc}", "trigger_price": None}


def evaluate_exit_rule_from_settings(settings: dict | None) -> tuple[bool, str]:
    details = evaluate_exit_rule_details_from_settings(settings)
    return bool(details["ready"]), str(details["reason"])


def refresh_trailing_state_for_open_positions() -> bool:
    updated_orders = st.session_state.get("tracked_alpaca_orders", [])
    changed = False
    for position in alpaca_positions:
        symbol = str(position.get("Symbol", "")).strip().upper()
        settings = saved_exit_settings_for_symbol(symbol, updated_orders)
        if not settings:
            continue
        details = evaluate_exit_rule_details_from_settings(settings)
        if not details.get("state_changed"):
            continue
        refreshed_settings = dict(settings)
        high = optional_float(details.get("highest_high_since_entry"))
        trigger = optional_float(details.get("trigger_price"))
        if high is not None:
            refreshed_settings["highest_high_since_entry"] = high
        if trigger is not None:
            refreshed_settings["last_exit_trigger_price"] = trigger
            refreshed_settings["last_exit_trigger_source"] = details.get("trigger_source", "")
        refreshed_settings["last_exit_checked_at"] = pd.Timestamp.now(tz="America/Los_Angeles").isoformat()
        updated_orders = update_exit_settings_for_symbol(symbol, updated_orders, refreshed_settings)
        changed = True
    if changed:
        broker_state_store.replace_all(updated_orders)
        st.session_state["tracked_alpaca_orders"] = updated_orders
    return changed


optimizer_setting_keys = (
    "strategy_type",
    "strategy_label",
    "entry_window",
    "exit_window",
    "atr_stop_multiplier",
    "risk_per_trade_pct",
    "moving_average_window",
    "pullback_average_length",
    "momentum_turn_length",
    "rsi_entry_filter_enabled",
)
optimizer_market_fingerprint = "synthetic-default"
if market_data is not None:
    fingerprint_columns = [
        column for column in ("Open", "High", "Low", "Close", "Volume")
        if column in market_data.columns
    ]
    hashed_market_data = pd.util.hash_pandas_object(
        market_data[fingerprint_columns], index=True
    ).values.tobytes()
    optimizer_market_fingerprint = hashlib.sha256(hashed_market_data).hexdigest()
optimizer_equity = float(paper_order_risk_equity)
optimizer_equity_rounding_digits = max(
    0,
    len(str(max(1, int(abs(optimizer_equity))))) - 3,
)
optimizer_equity_bucket = round(optimizer_equity, -optimizer_equity_rounding_digits)
optimizer_signature_payload = {
    "ticker": ticker,
    "source": data_source,
    "interval": interval,
    "history": period,
    "market_data": optimizer_market_fingerprint,
    "strategy_settings": {
        key: current_strategy_settings[key] for key in optimizer_setting_keys
    },
    "account_equity_bucket": optimizer_equity_bucket,
    "risk_limits": asdict(risk_limits),
    "older_data_fraction": train_fraction,
    "settings_per_strategy": max_parameter_candidates,
}
optimizer_signature = hashlib.sha256(
    json.dumps(optimizer_signature_payload, sort_keys=True, default=str).encode("utf-8")
).hexdigest()
optimizer_search_completed = False
if run_strategy_input_search:
    try:
        with st.spinner("Searching strategy inputs..."):
            interval_result = None
            interval_errors = []
            if data_source == "Synthetic":
                fresh_optimizer_result = optimize_strategy_inputs(
                    market_data=market_data,
                    current_settings=current_strategy_settings,
                    account_equity=float(paper_order_risk_equity),
                    risk_limits=risk_limits,
                    train_fraction=train_fraction,
                    max_candidates_per_strategy=max_parameter_candidates,
                )
            else:
                interval_histories = (
                    {"1d": "10y", "4h": "5y", "1h": "2y"}
                    if data_source == "Ticker (Alpaca)"
                    else {"1d": "10y", "4h": "1y", "1h": "1y"}
                )
                interval_market_data = {}
                for search_interval, search_history in interval_histories.items():
                    try:
                        if search_interval == interval and search_history == period and market_data is not None:
                            search_data = market_data
                        else:
                            search_data = fetch_price_data_for_source(
                                ticker,
                                search_history,
                                search_interval,
                                data_source,
                            )
                        interval_market_data[search_interval] = (search_history, search_data)
                    except Exception as exc:
                        interval_errors.append(f"{search_interval}: {exc}")
                interval_result = optimize_strategy_intervals(
                    market_data_by_interval=interval_market_data,
                    current_settings=current_strategy_settings,
                    account_equity=float(paper_order_risk_equity),
                    risk_limits=risk_limits,
                    train_fraction=train_fraction,
                    max_candidates_per_strategy=max_parameter_candidates,
                )
                fresh_optimizer_result = interval_result.best_result
        st.session_state["strategy_optimizer_search"] = {
            "signature": optimizer_signature,
            "result": fresh_optimizer_result,
            "interval_result": interval_result,
            "interval_errors": interval_errors,
            "error": None,
            "completed_at": pd.Timestamp.now(tz="America/Los_Angeles").isoformat(),
        }
        optimizer_search_completed = True
    except ValueError as exc:
        st.session_state["strategy_optimizer_search"] = {
            "signature": optimizer_signature,
            "result": None,
            "interval_result": None,
            "interval_errors": [],
            "error": str(exc),
            "completed_at": pd.Timestamp.now(tz="America/Los_Angeles").isoformat(),
        }

optimizer_search_state = st.session_state.get("strategy_optimizer_search")
strategy_optimizer_result = (
    optimizer_search_state.get("result") if optimizer_search_state else None
)
strategy_optimizer_interval_result = (
    optimizer_search_state.get("interval_result") if optimizer_search_state else None
)
strategy_optimizer_interval_errors = (
    optimizer_search_state.get("interval_errors", []) if optimizer_search_state else []
)
parameter_loop_error = (
    optimizer_search_state.get("error") if optimizer_search_state else None
)
optimizer_result_stale = bool(
    optimizer_search_state
    and optimizer_search_state.get("signature") != optimizer_signature
)
if optimizer_search_state is None:
    optimizer_sidebar_status.caption("No saved strategy input search.")
elif optimizer_result_stale:
    optimizer_sidebar_status.warning("Inputs changed. Run the search again.")
elif parameter_loop_error:
    optimizer_sidebar_status.error("Strategy input search could not finish.")
else:
    optimizer_sidebar_status.success("Strategy input search is ready.")

monitoring_result = monitor_paper_session(
    broker=paper_broker,
    starting_cash=st.session_state["paper_starting_cash"],
    account_equity=account,
    limits=risk_limits,
)
current_run_manifest = build_run_manifest(
    session_id=st.session_state["paper_session_id"],
    mode_label=mode_label,
    data_source=source_caption,
    strategy_config=current_strategy_config,
    risk_limits=risk_limits,
    alpaca_config=alpaca_adapter.config,
    account_equity=paper_order_risk_equity,
    paper_cash=paper_order_available_cash,
)
current_manifest_record = run_manifest_record(current_run_manifest)
intent = apply_paper_buy_order_settings(
    live.get("trade_intent"),
    paper_buy_order_style,
    paper_buy_limit_adjustment_pct,
    paper_buy_custom_limit_price,
    float(live.get("last_p", 0) or 0),
)
intent = resize_trade_intent_for_account(intent, paper_order_risk_equity, risk_dec)
intent_symbol = intent.symbol_clean if intent else ""
alpaca_position_symbols = {str(row.get("Symbol", "")).strip().upper() for row in alpaca_positions}
symbol_current_notional = 0.0
if not use_alpaca_account_for_paper_risk and intent_symbol in paper_broker.positions:
    symbol_current_notional += paper_broker.positions[intent_symbol].market_value
symbol_current_notional += sum(float(row.get("Market Value") or 0) for row in alpaca_positions if str(row.get("Symbol", "")).strip().upper() == intent_symbol)
symbol_current_notional += open_buy_order_notional(alpaca_orders, intent_symbol)
raw_intent_quantity = intent.quantity if intent else None
intent = constrain_trade_intent_to_limits(
    intent,
    account_equity=paper_order_risk_equity,
    limits=risk_limits,
    current_portfolio_notional=paper_order_portfolio_notional,
    symbol_current_notional=symbol_current_notional,
    available_cash=paper_order_available_cash,
)
live["trade_intent"] = intent
if intent is not None:
    live["raw_pos_size"] = raw_intent_quantity
    live["pos_size"] = intent.quantity
    entry_atr = optional_float(live.get("last_atr"))
    entry_reference_price = optional_float(live.get("last_p"))
    entry_atr_pct = (entry_atr / entry_reference_price * 100) if entry_atr and entry_reference_price else None
    current_strategy_settings.update(
        {
            "entry_reference_price": entry_reference_price,
            "entry_atr": entry_atr,
            "entry_atr_pct": entry_atr_pct,
            "entry_stop_atr_multiplier": atr_mult,
            "entry_stop_distance": optional_float(live.get("stop_from_entry")),
            "entry_stop_loss": intent.stop_loss,
            "entry_rule_level": optional_float(live.get("entry_level")),
            "exit_rule_level_at_entry": optional_float(live.get("exit_level")),
            "planned_order_type": intent.order_type,
            "planned_limit_price": intent.limit_price,
            "planned_quantity": intent.quantity,
            "planned_entry_price": intent.entry_price,
            "highest_high_since_entry": entry_reference_price,
            "last_exit_trigger_price": intent.stop_loss,
            "last_exit_trigger_source": "fill-adjusted initial stop",
            "sizing_account_source": paper_order_account_source,
            "sizing_account_equity": paper_order_risk_equity,
            "sizing_available_cash": paper_order_available_cash,
        }
    )
    current_exit_settings.update(current_strategy_settings)
    current_exit_settings["auto_exit_enabled"] = True
trade_open_position_symbols = (
    alpaca_position_symbols
    if use_alpaca_account_for_paper_risk
    else set(paper_broker.positions.keys()) | alpaca_position_symbols
)
risk_check = check_trade_intent(
    intent,
    account_equity=paper_order_risk_equity,
    limits=risk_limits,
    open_positions=trade_open_position_symbols,
    open_position_count=len(trade_open_position_symbols | open_buy_order_symbols(alpaca_orders)),
    current_portfolio_notional=paper_order_portfolio_notional,
    symbol_current_notional=symbol_current_notional,
    session_pnl=paper_order_session_pnl,
    available_cash=paper_order_available_cash,
)
execution_decision = decide_execution(execution_mode, risk_check)
preflight_check = build_preflight_check(
    intent=intent,
    risk_check=risk_check,
    execution_decision=execution_decision,
    broker_connected=paper_broker is not None,
    audit_logging_enabled="session_audit_events" in st.session_state,
)
trade_proposal = build_trade_proposal(
    symbol=ticker,
    live=live,
    stats=stats,
    trade_intent=intent,
    risk_check=risk_check,
    execution_decision=execution_decision,
)
audit_events = build_audit_events(
    mode_label=mode_label,
    source_caption=source_caption,
    trade_intent=intent,
    risk_check=risk_check,
    execution_decision=execution_decision,
    stats=stats,
    trade_proposal=trade_proposal,
)
audit_key = (
    seed,
    mode_label,
    source_caption,
    live["signal"],
    strategy_label,
    intent.symbol_clean if intent else None,
    intent.quantity if intent else None,
    risk_check.approved,
    execution_decision.reason,
    stats["total_trades"],
    max_portfolio_exposure,
    max_symbol_concentration,
    max_session_loss,
    max_open_positions,
    preflight_check.ready,
)
if st.session_state.get("last_audit_key") != audit_key:
    st.session_state["session_audit_events"].extend(audit_events)
    if persist_audit_log:
        audit_store.append_many(audit_events)
    st.session_state["last_audit_key"] = audit_key
if optimizer_search_completed and strategy_optimizer_result is not None:
    optimizer_best = strategy_optimizer_result.best
    parameter_event = AuditEvent(
        event_type="strategy_input_search_completed",
        message=strategy_optimizer_result.summary,
        payload={
            "tested_candidates": strategy_optimizer_result.tested_candidates,
            "best_strategy": None if optimizer_best is None else optimizer_best.strategy_label,
            "best_score": None if optimizer_best is None else optimizer_best.score,
            "confidence": None if optimizer_best is None else optimizer_best.confidence,
            "search_signature": optimizer_signature,
        },
    )
    st.session_state["session_audit_events"].append(parameter_event)
    if persist_audit_log:
        audit_store.append(parameter_event)

with st.container(key="top_navigation"):
    st.title("AgentLoop Trader")
    command_center_view = st.radio(
        "Command center page",
        ["Open Positions", "Ideas", "New Trade", "Alpaca", "Paper Review"],
        horizontal=True,
        label_visibility="collapsed",
    )

status_rows = compact_status_records(
    mode_label=mode_label,
    risk_approved=risk_check.approved,
    broker_connected=alpaca_status.connected,
    broker_state_stale=alpaca_state_health.stale,
    kill_switch_enabled=effective_kill_switch,
    live_writes_blocked=not (not alpaca_adapter.config.paper and alpaca_status.can_submit_orders),
)
status_strip(status_rows)

tracked_alpaca_orders = st.session_state.get("tracked_alpaca_orders", [])
alpaca_preview = build_alpaca_order_preview(intent, execution_decision, alpaca_adapter.config)
duplicate_alpaca_reasons = duplicate_exposure_reasons(
    intent,
    alpaca_positions,
    allow_duplicate=allow_add_to_existing_position,
)
open_order_reasons = open_order_exposure_reasons(intent, alpaca_orders)
duplicate_preview_submitted = preview_already_tracked(alpaca_preview.preview_hash, tracked_alpaca_orders)
exit_previews = build_exit_order_previews(alpaca_positions, alpaca_adapter.config)
if alpaca_positions and tracked_alpaca_orders:
    if refresh_trailing_state_for_open_positions():
        tracked_alpaca_orders = st.session_state.get("tracked_alpaca_orders", [])
cancelable_alpaca_orders = cancelable_alpaca_order_records(alpaca_orders)
managed_cancelable_alpaca_orders = enriched_cancelable_order_records(cancelable_alpaca_orders, tracked_alpaca_orders, alpaca_orders)
waiting_limit_buy_rows = waiting_limit_buy_order_records(
    managed_cancelable_alpaca_orders,
    ticker,
    live.get("last_p"),
    auto_cancel_stale_limit_orders,
    stale_limit_order_minutes,
)
current_market_advisory = market_session_advisory()
regular_market_open = bool(current_market_advisory.get("Open", False))
limit_buy_allowed_outside_market = (
    allow_limit_buys_outside_market_hours
    and paper_buy_order_style != "Market"
)
auto_buy_session_allows_order = regular_market_open or limit_buy_allowed_outside_market
first_exit_preview = next((preview for preview in exit_previews if preview.valid), None)
first_exit_symbol = str(first_exit_preview.order.get("symbol", "")).strip().upper() if first_exit_preview else ""
saved_exit_settings = saved_exit_settings_for_symbol(first_exit_symbol, tracked_alpaca_orders) if first_exit_symbol else None
auto_exit_enabled_for_position = bool(saved_exit_settings.get("auto_exit_enabled", True)) if saved_exit_settings else False
exit_settings_available = True if not first_exit_symbol else bool(saved_exit_settings and auto_exit_enabled_for_position)
managed_exit_ready, managed_exit_reason = evaluate_exit_rule_from_settings(saved_exit_settings) if first_exit_symbol else (False, "No Alpaca paper position is waiting for auto-exit.")
exit_settings_reason = (
    "No Alpaca paper position is waiting for auto-exit."
    if not first_exit_symbol
    else "Auto exit is off for this position."
    if saved_exit_settings and not auto_exit_enabled_for_position
    else "Saved exit settings are loaded for this position."
    if exit_settings_available
    else "No saved exit settings for this position; auto exit is paused."
)
buy_preview_ready = (
    intent is not None
    and preflight_check.ready
    and alpaca_preview.valid
    and not duplicate_alpaca_reasons
    and not open_order_reasons
    and not duplicate_preview_submitted
)
operator_state = operator_state_record(
    intent_present=intent is not None,
    risk_approved=risk_check.approved,
    preflight_ready=preflight_check.ready,
    execution_mode=execution_mode,
    broker_connected=alpaca_status.connected,
    broker_state_stale=alpaca_state_health.stale,
    alpaca_enabled=enable_alpaca_paper_orders,
    buy_preview_ready=buy_preview_ready,
    buy_preview_armed=buy_preview_ready,
    exit_preview_count=len(exit_previews),
    cancelable_order_count=len(managed_cancelable_alpaca_orders),
    open_position_count=len(alpaca_positions),
)
setup_scorecard_rows = setup_scorecard_records(
    live,
    risk_approved=risk_check.approved,
    blocked_reasons=preflight_check.blocked_reasons or risk_check.rejected_reasons,
    enabled_inputs=setup_inputs,
    strategy_type=strategy_type,
)
if intent is None:
    final_answer = "WAIT"
    final_detail = strategy_decision_detail()
elif preflight_check.blocked_reasons or risk_check.rejected_reasons:
    final_answer = "BLOCK"
    final_detail = strategy_decision_detail()
elif preflight_check.ready:
    final_answer = "TRADE"
    final_detail = strategy_decision_detail()
else:
    final_answer = "WAIT"
    final_detail = strategy_decision_detail()
answer_color = {"TRADE": "#3B6D11", "WAIT": "#8A6D1D", "BLOCK": "#A32D2D"}[final_answer]
if final_answer == "TRADE":
    new_trade_next_action = (
        "Send the paper buy order to Alpaca."
        if execution_mode == "paper"
        else "Switch Order mode to Paper trading when you want to send this BUY."
    )
elif final_answer == "BLOCK":
    first_trade_blocker = next(iter(preflight_check.blocked_reasons or risk_check.rejected_reasons), "A trade check is blocking this BUY.")
    new_trade_next_action = f"Resolve this BUY blocker: {first_trade_blocker}"
else:
    new_trade_next_action = "Wait for the selected strategy's required BUY rules to pass."
research_agent_report = build_research_agent_report(
    ticker=ticker,
    selected_strategy=strategy_label,
    strategy_results=strategy_results,
    setup_rows=setup_scorecard_rows,
    final_read=final_answer,
    decision_detail=final_detail,
    next_action=new_trade_next_action,
)
research_snapshot_key = hashlib.sha256(
    json.dumps(
        {
            "ticker": ticker,
            "strategy": strategy_label,
            "settings": current_strategy_settings,
            "final_read": final_answer,
            "best_strategy": research_agent_report.best_strategy,
            "next_action": new_trade_next_action,
        },
        sort_keys=True,
        default=str,
    ).encode("utf-8")
).hexdigest()[:16]
previous_research_snapshot = research_snapshot_store.latest_for_ticker(ticker, exclude_settings_key=research_snapshot_key)
current_research_snapshot = build_research_snapshot(
    research_agent_report,
    selected_strategy=strategy_label,
    settings_key=research_snapshot_key,
)
if st.session_state.get("last_research_snapshot_key") != research_snapshot_key:
    research_snapshot_store.append(current_research_snapshot)
    st.session_state["last_research_snapshot_key"] = research_snapshot_key
research_loop_rows = compare_research_snapshots(previous_research_snapshot, current_research_snapshot)

exit_preview_rows_for_auto = exit_preview_records(exit_previews)
exit_blockers_by_hash_for_auto = {
    preview.preview_hash: (
        preview.blocked_reasons
        + exit_position_reasons(preview, alpaca_positions)
        + open_exit_order_reasons(preview, alpaca_orders)
    )
    for preview in exit_previews
}
auto_entry_blockers = (
    list(alpaca_preview.blocked_reasons)
    + duplicate_alpaca_reasons
    + open_order_reasons
    + (["This paper order is already tracked in the app."] if duplicate_preview_submitted else [])
    + auto_entry_session_safety_blockers(intent.symbol_clean if intent else "")
)
auto_entry_status = auto_entry_decision(
    automation_level=active_automation_level,
    execution_mode=execution_mode,
    broker_connected=alpaca_status.connected,
    broker_can_submit=alpaca_status.can_submit_orders,
    paper_orders_enabled=enable_alpaca_paper_orders,
    kill_switch_enabled=effective_kill_switch,
    broker_state_stale=alpaca_state_health.stale,
    market_open=auto_buy_session_allows_order,
    intent_present=intent is not None,
    risk_approved=risk_check.approved,
    preflight_ready=preflight_check.ready,
    preview_valid=alpaca_preview.valid,
    preview_hash=alpaca_preview.preview_hash,
    symbol=intent.symbol_clean if intent else "",
    quantity=intent.quantity if intent else "",
    blocked_reasons=auto_entry_blockers,
    already_sent_hashes=set(st.session_state.get("auto_entry_sent_hashes", [])),
)
auto_exit_status = auto_exit_decision(
    automation_level=active_automation_level,
    execution_mode=execution_mode,
    broker_connected=alpaca_status.connected,
    broker_can_submit=alpaca_status.can_submit_orders,
    paper_orders_enabled=enable_alpaca_paper_orders,
    kill_switch_enabled=effective_kill_switch,
    broker_state_stale=alpaca_state_health.stale,
    market_open=regular_market_open,
    strategy_exit_ready=managed_exit_ready,
    strategy_exit_reason=managed_exit_reason,
    exit_preview_records=exit_preview_rows_for_auto,
    exit_blockers=exit_blockers_by_hash_for_auto,
    entry_settings_match=exit_settings_available,
    entry_settings_reason=exit_settings_reason,
    already_sent_hashes=set(st.session_state.get("auto_exit_sent_hashes", [])),
)


def record_automation_decisions(checked_at: str) -> None:
    auto_entry_key = (
        active_automation_level,
        full_automation_enabled,
        auto_entry_status.status,
        auto_entry_status.preview_hash,
        tuple(auto_entry_status.reasons),
    )
    if active_automation_level == "Auto entries and exits" and st.session_state.get("last_auto_entry_decision_key") != auto_entry_key:
        auto_entry_event = AuditEvent(
            event_type="auto_entry_decision_recorded",
            message=f"Automatic paper buy decision: {auto_entry_status.status}.",
            payload={
                "automation_level": automation_level,
                "active_automation_level": active_automation_level,
                "full_automation_enabled": full_automation_enabled,
                "status": auto_entry_status.status,
                "ready": auto_entry_status.ready,
                "symbol": auto_entry_status.symbol,
                "quantity": auto_entry_status.quantity,
                "preview_hash": auto_entry_status.preview_hash,
                "reasons": auto_entry_status.reasons,
                "checked_at": checked_at,
                "broker_writes_submitted": 0,
            },
        )
        st.session_state["session_audit_events"].append(auto_entry_event)
        if persist_audit_log:
            audit_store.append(auto_entry_event)
        st.session_state["last_auto_entry_decision_key"] = auto_entry_key

    auto_exit_key = (
        active_automation_level,
        auto_exit_status.status,
        auto_exit_status.preview_hash,
        tuple(auto_exit_status.reasons),
    )
    if active_automation_level != "Manual review only" and st.session_state.get("last_auto_exit_decision_key") != auto_exit_key:
        auto_exit_event = AuditEvent(
            event_type="auto_exit_decision_recorded",
            message=f"Automatic paper exit decision: {auto_exit_status.status}.",
            payload={
                "automation_level": automation_level,
                "active_automation_level": active_automation_level,
                "status": auto_exit_status.status,
                "ready": auto_exit_status.ready,
                "symbol": auto_exit_status.symbol,
                "quantity": auto_exit_status.quantity,
                "preview_hash": auto_exit_status.preview_hash,
                "reasons": auto_exit_status.reasons,
                "broker_writes_submitted": 0,
            },
        )
        st.session_state["session_audit_events"].append(auto_exit_event)
        if persist_audit_log:
            audit_store.append(auto_exit_event)
        st.session_state["last_auto_exit_decision_key"] = auto_exit_key


def auto_cancel_stale_limit_buy_once(automation_checked_at: str) -> bool:
    if not auto_cancel_stale_limit_orders:
        return False
    blockers = stale_limit_cancel_blockers(managed_cancelable_alpaca_orders, stale_limit_order_minutes)
    if blockers:
        set_automation_action("Limit buy cancel waiting", " ".join(blockers), automation_checked_at)
        return False
    stale_orders = stale_limit_buy_orders(managed_cancelable_alpaca_orders, stale_limit_order_minutes)
    if not stale_orders:
        return False
    selected_order = stale_orders[0]
    selected_order_id = selected_order.get("Alpaca Order ID") or selected_order.get("Broker Order ID", "")
    age_minutes = order_age_minutes(selected_order)
    cancel_preview = build_alpaca_cancel_preview(selected_order, alpaca_adapter.config)
    if not cancel_preview.valid:
        set_automation_action("Limit buy cancel blocked", " ".join(cancel_preview.blocked_reasons), automation_checked_at)
        return False
    try:
        cancel_result = alpaca_adapter.cancel_order(
            selected_order_id,
            expected_cancel_hash=cancel_preview.preview_hash,
        )
        broker_state_store.upsert(
            {
                "broker_order_id": selected_order_id,
                "preview_hash": selected_order.get("Review ID", ""),
                "symbol": selected_order.get("Symbol", ""),
                "side": selected_order.get("Side", ""),
                "quantity": selected_order.get("Quantity", ""),
                "order_type": selected_order.get("Order Type", ""),
                "limit_price": selected_order.get("Limit Price", ""),
                "status": "cancel_requested",
                "submitted_at": selected_order.get("Submitted", ""),
                "source": "auto_cancel_old_limit_buy",
            }
        )
        st.session_state["tracked_alpaca_orders"] = broker_state_store.read()
        sync_auto_entry_sent_hashes_with_open_orders()
        set_automation_action(
            f"Canceled old limit buy: {selected_order.get('Symbol', '')}",
            f"Order waited {order_age_label(age_minutes)}.",
            automation_checked_at,
        )
        cancel_event = AuditEvent(
            event_type="auto_old_limit_buy_cancel_submitted",
            message="Automatic stale paper limit buy cancel sent to Alpaca.",
            payload={
                "broker_order_id": selected_order_id,
                "symbol": selected_order.get("Symbol", ""),
                "side": selected_order.get("Side", ""),
                "quantity": selected_order.get("Quantity", ""),
                "limit_price": selected_order.get("Limit Price", ""),
                "age_minutes": age_minutes,
                "cancel_after_minutes": stale_limit_order_minutes,
                "cancel_preview_hash": cancel_preview.preview_hash,
                "cancel_status": str(getattr(cancel_result, "status", "cancel_requested")),
                "checked_at": automation_checked_at,
                "broker_writes_submitted": 1,
            },
        )
        st.session_state["session_audit_events"].append(cancel_event)
        if persist_audit_log:
            audit_store.append(cancel_event)
        return True
    except Exception as exc:
        set_automation_action("Limit buy cancel blocked", str(exc), automation_checked_at)
        cancel_event = AuditEvent(
            event_type="auto_old_limit_buy_cancel_blocked",
            message=str(exc),
            payload={
                "broker_order_id": selected_order_id,
                "symbol": selected_order.get("Symbol", ""),
                "cancel_after_minutes": stale_limit_order_minutes,
                "checked_at": automation_checked_at,
                "broker_writes_submitted": 0,
            },
        )
        st.session_state["session_audit_events"].append(cancel_event)
        if persist_audit_log:
            audit_store.append(cancel_event)
        return False


def run_paper_automation_once() -> None:
    automation_checked_at = pd.Timestamp.now(tz="America/Los_Angeles").isoformat()
    st.session_state["last_automation_checked_at"] = automation_checked_at
    record_automation_decisions(automation_checked_at)
    _, broker_order_state_changed = refresh_tracked_alpaca_orders_from_broker()
    if broker_order_state_changed:
        set_automation_action(
            "Alpaca orders refreshed",
            "Order status changed at Alpaca. Automation will re-check with fresh order state.",
            automation_checked_at,
        )
        st.rerun()

    if auto_exit_status.ready:
        auto_exit_preview = next((preview for preview in exit_previews if preview.preview_hash == auto_exit_status.preview_hash), None)
        auto_exit_symbol = str(auto_exit_status.symbol).strip().upper()
        auto_exit_position = next(
            (position for position in alpaca_positions if str(position.get("Symbol", "")).strip().upper() == auto_exit_symbol),
            None,
        )
        auto_exit_intent = (
            TradeIntent(
                symbol=auto_exit_symbol,
                side="sell",
                quantity=int(float(auto_exit_status.quantity)),
                order_type="market",
                time_in_force="day",
                rationale="Automatic paper exit generated from an existing Alpaca paper position.",
                source_signals=["auto_exit_only", "alpaca_position_exit_preview"],
            )
            if auto_exit_preview is not None and auto_exit_position is not None
            else None
        )
        auto_exit_execution_decision = ExecutionDecision(
            mode="paper",
            approved_for_execution=True,
            requires_manual_approval=False,
            reason="Paper exit passed all automation checks.",
            risk_check=RiskCheckResult(approved=True, rejected_reasons=[], checks={"auto_exit_position": True}),
        )
        try:
            if auto_exit_intent is None:
                raise ValueError("Automatic exit could not match an open Alpaca paper position.")
            alpaca_auto_exit_order = alpaca_adapter.submit_order(
                auto_exit_intent,
                auto_exit_execution_decision,
                expected_preview_hash=auto_exit_status.preview_hash,
            )
            broker_order_id = str(getattr(alpaca_auto_exit_order, "id", ""))
            if broker_order_id:
                tracked_record = {
                    "broker_order_id": broker_order_id,
                    "preview_hash": auto_exit_status.preview_hash,
                    "symbol": auto_exit_symbol,
                    "side": "sell",
                    "quantity": auto_exit_intent.quantity,
                    "status": str(getattr(alpaca_auto_exit_order, "status", "")),
                    "submitted_at": str(getattr(alpaca_auto_exit_order, "submitted_at", "")),
                    "source": "auto_exit_only",
                }
                st.session_state["tracked_alpaca_orders"].append(tracked_record)
                broker_state_store.upsert(tracked_record)
            st.session_state["auto_exit_sent_hashes"].append(auto_exit_status.preview_hash)
            set_automation_action(f"Paper exit sent: {auto_exit_intent.quantity} {auto_exit_symbol}", "", automation_checked_at)
            auto_submit_event = AuditEvent(
                event_type="auto_paper_exit_submitted",
                message="Automatic paper exit sent to Alpaca.",
                payload={
                    "symbol": auto_exit_symbol,
                    "side": "sell",
                    "quantity": auto_exit_intent.quantity,
                    "preview_hash": auto_exit_status.preview_hash,
                    "broker_order_id": broker_order_id,
                    "automation_level": automation_level,
                    "checked_at": automation_checked_at,
                    "broker_writes_submitted": 1,
                },
            )
            st.session_state["session_audit_events"].append(auto_submit_event)
            if persist_audit_log:
                audit_store.append(auto_submit_event)
        except Exception as exc:
            st.session_state["auto_exit_sent_hashes"].append(auto_exit_status.preview_hash)
            set_automation_action("Paper exit blocked", str(exc), automation_checked_at)
            auto_block_event = AuditEvent(
                event_type="auto_paper_exit_blocked",
                message=str(exc),
                payload={
                    "symbol": auto_exit_symbol,
                    "preview_hash": auto_exit_status.preview_hash,
                    "automation_level": automation_level,
                    "checked_at": automation_checked_at,
                    "broker_writes_submitted": 0,
                },
            )
            st.session_state["session_audit_events"].append(auto_block_event)
            if persist_audit_log:
                audit_store.append(auto_block_event)

    elif auto_cancel_stale_limit_buy_once(automation_checked_at):
        st.rerun()

    elif auto_entry_status.ready:
        try:
            if intent is None:
                raise ValueError("Automatic buy could not find a current trade idea.")
            live_sent_hashes = active_tracked_preview_hashes(st.session_state.get("tracked_alpaca_orders", []))
            if auto_entry_status.preview_hash in live_sent_hashes:
                set_automation_action("Paper buy skipped", "This exact paper buy is already open at Alpaca.", automation_checked_at)
                return
            local_buy_reasons = local_open_buy_order_reasons(
                intent.symbol_clean,
                st.session_state.get("tracked_alpaca_orders", []),
            )
            if local_buy_reasons:
                set_automation_action("Paper buy skipped", "; ".join(local_buy_reasons), automation_checked_at)
                return
            st.session_state["auto_entry_sent_hashes"].append(auto_entry_status.preview_hash)
            alpaca_auto_entry_order = alpaca_adapter.submit_order(
                intent,
                execution_decision,
                expected_preview_hash=auto_entry_status.preview_hash,
            )
            broker_order_id = str(getattr(alpaca_auto_entry_order, "id", ""))
            if broker_order_id:
                tracked_record = {
                    "broker_order_id": broker_order_id,
                    "preview_hash": auto_entry_status.preview_hash,
                    "symbol": intent.symbol_clean,
                    "side": intent.side,
                    "quantity": intent.quantity,
                    "order_type": intent.order_type,
                    "limit_price": intent.limit_price,
                    "status": str(getattr(alpaca_auto_entry_order, "status", "")),
                    "submitted_at": str(getattr(alpaca_auto_entry_order, "submitted_at", "")),
                    "source": "auto_entries_and_exits",
                    "strategy_settings": current_strategy_settings,
                    "exit_settings": current_exit_settings,
                }
                st.session_state["tracked_alpaca_orders"].append(tracked_record)
                broker_state_store.upsert(tracked_record)
                updated_orders = update_exit_settings_for_symbol(
                    intent.symbol_clean,
                    st.session_state["tracked_alpaca_orders"],
                    current_exit_settings,
                )
                broker_state_store.replace_all(updated_orders)
                st.session_state["tracked_alpaca_orders"] = updated_orders
            order_type = "limit" if intent.order_type == "limit" else "market"
            set_automation_action(f"Paper buy {order_type} sent: {intent.quantity} {intent.symbol_clean}", "", automation_checked_at)
            auto_entry_submit_event = AuditEvent(
                event_type="auto_paper_entry_submitted",
                message="Automatic paper buy sent to Alpaca.",
                payload={
                    "symbol": intent.symbol_clean,
                    "side": intent.side,
                    "quantity": intent.quantity,
                    "order_type": intent.order_type,
                    "limit_price": intent.limit_price,
                    "preview_hash": auto_entry_status.preview_hash,
                    "broker_order_id": broker_order_id,
                    "automation_level": automation_level,
                    "checked_at": automation_checked_at,
                    "broker_writes_submitted": 1,
                    "strategy_settings": current_strategy_settings,
                    "exit_settings": current_exit_settings,
                },
            )
            st.session_state["session_audit_events"].append(auto_entry_submit_event)
            if persist_audit_log:
                audit_store.append(auto_entry_submit_event)
        except Exception as exc:
            if auto_entry_status.preview_hash not in st.session_state["auto_entry_sent_hashes"]:
                st.session_state["auto_entry_sent_hashes"].append(auto_entry_status.preview_hash)
            set_automation_action("Paper buy blocked", str(exc), automation_checked_at)
            auto_entry_block_event = AuditEvent(
                event_type="auto_paper_entry_blocked",
                message=str(exc),
                payload={
                    "symbol": auto_entry_status.symbol,
                    "preview_hash": auto_entry_status.preview_hash,
                    "automation_level": automation_level,
                    "checked_at": automation_checked_at,
                    "broker_writes_submitted": 0,
                },
            )
            st.session_state["session_audit_events"].append(auto_entry_block_event)
            if persist_audit_log:
                audit_store.append(auto_entry_block_event)


automation_timer_enabled = active_automation_level != "Manual review only" and not background_worker_enabled
if automation_timer_enabled:
    @st.fragment(run_every=f"{automation_refresh_seconds}s")
    def automation_timer_tick() -> None:
        run_paper_automation_once()
        st.caption(
            f"Automation checked: {st.session_state.get('last_automation_checked_at', 'Not checked yet')}. "
            f"Last action: {st.session_state.get('last_automation_action', 'None')}."
        )

    automation_timer_tick()


position_records = paper_broker.position_records()
order_records = paper_broker.order_records()
session_audit_records = [JsonlAuditStore.event_to_record(event) for event in st.session_state["session_audit_events"]]
current_evidence_records = session_audit_records
if persist_audit_log:
    current_evidence_records = audit_store.read_recent(limit=500) or session_audit_records
session_snapshot = PaperSessionSnapshot(
    session_id=st.session_state["paper_session_id"],
    started_at=st.session_state["paper_session_started_at"],
    mode=mode_label,
    paper_cash=paper_broker.cash,
    paper_equity=paper_equity,
    session_pnl=session_pnl,
    local_orders=order_records,
    local_positions=position_records,
    tracked_alpaca_orders=st.session_state["tracked_alpaca_orders"],
    audit_records=session_audit_records,
)


def render_automation_status() -> None:
    sub_section("Automation status")
    status_label, status_detail, status_kind = automation_status_text()
    if status_kind == "ok":
        st.success(f"{status_label}: {status_detail}")
    elif status_kind == "warn":
        st.warning(f"{status_label}: {status_detail}")
    else:
        st.info(f"{status_label}: {status_detail}")
    if auto_entry_status.ready and limit_buy_allowed_outside_market and not regular_market_open:
        st.info("Market is closed. The app can send a paper limit buy because outside-hours limit buys are enabled. The order may wait at Alpaca before it fills.")
    if automation_level == "Auto entries and exits":
        if full_automation_enabled:
            st.caption("Automatic buys are enabled for paper trading. Risk checks, account sizing, and duplicate-order checks still apply.")
        else:
            st.caption("Automatic paper buys are not allowed. Check Allow automatic paper buys in Background Automation to permit them.")
    elif automation_level == "Auto exits only":
        st.caption("Auto exits can sell Alpaca paper positions. You still approve new buys manually.")
    runtime_state = AutomationRuntimeState(
        mode=automation_level_label,
        status=(
            "Exit ready"
            if auto_exit_status.ready
            else "Buy ready"
            if auto_entry_status.ready
            else "Manual"
            if active_automation_level == "Manual review only"
            else auto_entry_status.status
            if active_automation_level == "Auto entries and exits"
            else auto_exit_status.status
        ),
        last_checked_at=st.session_state.get("last_automation_checked_at", "Not checked yet"),
        last_action=st.session_state.get("last_automation_action", "None"),
        blocked_reason=st.session_state.get("last_automation_blocked_reason", ""),
    )
    st.caption(
        f"Last checked: {runtime_state.last_checked_at}. "
        f"Next check: every {automation_refresh_seconds} seconds while automation is on. "
        f"Last action: {runtime_state.last_action}."
    )
    if background_worker_enabled:
        st.info(
            "Background worker mode is on. Use Start Worker and Stop Worker under the Kill Switch. "
            "The in-page timer is paused while the worker is enabled."
        )
        st.dataframe(pd.DataFrame(worker_status_records(automation_worker_status)), width="stretch", hide_index=True)
    if show_portfolio_evidence:
        st.markdown("#### Automation runtime *")
        st.dataframe(pd.DataFrame(automation_runtime_records(runtime_state)), width="stretch", hide_index=True)
        st.markdown("#### Automatic buy check *")
        st.dataframe(pd.DataFrame(auto_entry_decision_records(auto_entry_status)), width="stretch", hide_index=True)
        st.markdown("#### Automatic exit check *")
        st.dataframe(pd.DataFrame(auto_exit_decision_records(auto_exit_status)), width="stretch", hide_index=True)


def render_open_positions_panel() -> None:
    position_settings_by_symbol = {
        str(position.get("Symbol", "")).strip().upper(): (
            saved_exit_settings_for_symbol(str(position.get("Symbol", "")).strip().upper(), st.session_state["tracked_alpaca_orders"]) or {}
        )
        for position in alpaca_positions
    }
    managed_count = sum(1 for settings in position_settings_by_symbol.values() if settings)
    auto_exit_on_count = sum(1 for settings in position_settings_by_symbol.values() if settings.get("auto_exit_enabled", False))
    position_summary_cols = st.columns(4)
    metric_card(position_summary_cols[0], "Open Positions", len(alpaca_positions), "Alpaca paper")
    metric_card(position_summary_cols[1], "Managed", managed_count, "Have saved exit settings")
    metric_card(position_summary_cols[2], "Auto Exit On", auto_exit_on_count, "Position-level setting")
    metric_card(position_summary_cols[3], "Waiting Orders", count_waiting_alpaca_orders(alpaca_orders), "Alpaca paper")
    st.info(open_positions_next_step(position_settings_by_symbol))
    st.markdown("#### Automation readiness")
    st.dataframe(pd.DataFrame(daily_automation_readiness_records("Open positions")), width="stretch", hide_index=True)

    if not alpaca_positions:
        st.info("No Alpaca paper positions are open. When a position exists, this tab becomes the daily management panel for its exit settings and automation status.")
        return

    st.markdown("#### Position manager")
    st.dataframe(pd.DataFrame(daily_position_rows(alpaca_positions, position_settings_by_symbol)), width="stretch", hide_index=True)
    if show_portfolio_evidence:
        with st.expander("Position management details *", expanded=False):
            st.dataframe(
                pd.DataFrame(managed_position_records(alpaca_positions, position_settings_by_symbol)),
                width="stretch",
                hide_index=True,
            )
    position_options = [
        f"{str(position.get('Symbol', '')).strip().upper()} qty {position.get('Quantity', '')}"
        for position in alpaca_positions
    ]
    selected_position_idx = st.selectbox(
        "Alpaca paper position",
        range(len(position_options)),
        format_func=lambda idx: position_options[idx],
    )
    selected_position = alpaca_positions[selected_position_idx]
    selected_position_symbol = str(selected_position.get("Symbol", "")).strip().upper()
    selected_entry_settings = saved_buy_settings_for_symbol(selected_position_symbol, st.session_state["tracked_alpaca_orders"])
    selected_exit_settings = (
        saved_exit_settings_for_symbol(selected_position_symbol, st.session_state["tracked_alpaca_orders"])
        or selected_entry_settings
        or {**current_exit_settings, "symbol": selected_position_symbol}
    )
    selected_exit_details = evaluate_exit_rule_details_from_settings(selected_exit_settings)
    selected_exit_ready = bool(selected_exit_details["ready"])
    selected_exit_reason = str(selected_exit_details["reason"])
    selected_exit_trigger_price = selected_exit_details.get("trigger_price")
    selected_strategy_exit_price = selected_exit_details.get("strategy_exit_price")
    selected_atr_stop_price = selected_exit_details.get("atr_stop_price")
    selected_trailing_stop_price = selected_exit_details.get("trailing_stop_price")
    selected_exit_interval = str(selected_exit_details.get("interval") or selected_exit_settings.get("interval", interval))
    selected_exit_window = int(selected_exit_details.get("exit_window") or selected_exit_settings.get("exit_window", exit_w))

    try:
        selected_avg_entry = float(selected_position.get("Average Entry") or 0)
    except (TypeError, ValueError):
        selected_avg_entry = 0.0
    selected_current_price = position_current_price(selected_position)
    selected_unrealized = optional_float(selected_position.get("Unrealized P&L"))
    selected_unrealized_pct = position_unrealized_pct(selected_position)
    position_cols = st.columns(5)
    metric_card(position_cols[0], "Symbol", selected_position_symbol, "Alpaca paper")
    metric_card(position_cols[1], "Quantity", selected_position.get("Quantity", ""), "Current position")
    metric_card(position_cols[2], "Current Price", money_or_missing(selected_current_price), "From Alpaca value")
    metric_card(position_cols[3], "Average Entry", f"${selected_avg_entry:,.2f}", "From Alpaca")
    metric_card(
        position_cols[4],
        "Open P&L",
        money_or_missing(selected_unrealized),
        pct_or_missing(selected_unrealized_pct) if selected_unrealized_pct is not None else "Alpaca paper",
        "pos" if (selected_unrealized or 0) >= 0 else "neg",
    )
    if selected_exit_trigger_price:
        st.info(f"Auto exit: sell {selected_position_symbol} if price is at or below ${float(selected_exit_trigger_price):,.2f}.")
    if selected_exit_ready:
        st.warning(selected_exit_reason)
    st.markdown("#### Position plan")
    st.dataframe(
        pd.DataFrame(
            position_management_summary_records(
                selected_position,
                selected_exit_settings,
                selected_exit_details,
            )
        ),
        width="stretch",
        hide_index=True,
    )
    with st.expander("Quick exit changes", expanded=False):
        quick_cols = st.columns(3)
        if quick_cols[0].button("Tighten ATR Stop", key=f"tighten_exit_{selected_position_symbol}"):
            quick_settings = adjust_initial_stop_settings(
                selected_exit_settings,
                max(0.5, float(selected_exit_settings.get("entry_stop_atr_multiplier", selected_exit_settings.get("atr_stop_multiplier", atr_mult))) - 0.25),
            )
            quick_settings["symbol"] = selected_position_symbol
            updated_orders = update_exit_settings_for_symbol(selected_position_symbol, st.session_state["tracked_alpaca_orders"], quick_settings)
            broker_state_store.replace_all(updated_orders)
            st.session_state["tracked_alpaca_orders"] = updated_orders
            st.session_state["session_audit_events"].append(AuditEvent(event_type="position_exit_settings_tightened", message="ATR stop tightened for an Alpaca paper position.", payload={"symbol": selected_position_symbol, "exit_settings": quick_settings, "broker_writes_submitted": 0}))
            st.rerun()
        if quick_cols[1].button("Loosen ATR Stop", key=f"loosen_exit_{selected_position_symbol}"):
            quick_settings = adjust_initial_stop_settings(
                selected_exit_settings,
                min(5.0, float(selected_exit_settings.get("entry_stop_atr_multiplier", selected_exit_settings.get("atr_stop_multiplier", atr_mult))) + 0.25),
            )
            quick_settings["symbol"] = selected_position_symbol
            updated_orders = update_exit_settings_for_symbol(selected_position_symbol, st.session_state["tracked_alpaca_orders"], quick_settings)
            broker_state_store.replace_all(updated_orders)
            st.session_state["tracked_alpaca_orders"] = updated_orders
            st.session_state["session_audit_events"].append(AuditEvent(event_type="position_exit_settings_loosened", message="ATR stop loosened for an Alpaca paper position.", payload={"symbol": selected_position_symbol, "exit_settings": quick_settings, "broker_writes_submitted": 0}))
            st.rerun()
        if quick_cols[2].button("Use Current Sidebar Exit Settings", key=f"use_current_exit_{selected_position_symbol}"):
            quick_settings = {**current_exit_settings, "symbol": selected_position_symbol, "auto_exit_enabled": bool(selected_exit_settings.get("auto_exit_enabled", True))}
            updated_orders = update_exit_settings_for_symbol(selected_position_symbol, st.session_state["tracked_alpaca_orders"], quick_settings)
            broker_state_store.replace_all(updated_orders)
            st.session_state["tracked_alpaca_orders"] = updated_orders
            st.session_state["session_audit_events"].append(AuditEvent(event_type="position_exit_settings_replaced", message="Exit settings replaced with current sidebar settings.", payload={"symbol": selected_position_symbol, "exit_settings": quick_settings, "broker_writes_submitted": 0}))
            st.rerun()
    if show_portfolio_evidence:
        with st.expander("Saved exit rule details *", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    position_exit_plan_records(
                        selected_exit_settings,
                        selected_exit_ready,
                        selected_exit_reason,
                        selected_exit_trigger_price,
                    )
                ),
                width="stretch",
                hide_index=True,
            )
            st.markdown("#### Exit risk *")
            st.dataframe(
                pd.DataFrame(combined_position_risk_records(selected_position, selected_exit_trigger_price)),
                width="stretch",
                hide_index=True,
            )

    with st.form(f"exit_settings_{selected_position_symbol}"):
        st.markdown("#### Edit exit settings")
        st.caption(
            f"Exit interval: {selected_exit_interval}. "
            f"Sell exit length uses {selected_exit_interval} bars, so {selected_exit_window} bars means {selected_exit_window} bars on the {selected_exit_interval} chart."
        )
        exit_auto_enabled = st.checkbox(
            "Auto exit this position",
            value=bool(selected_exit_settings.get("auto_exit_enabled", True)),
            key=f"auto_exit_enabled_{selected_position_symbol}",
        )
        exit_strategy_options = {
            "Breakout continuation": "breakout",
            "Trend pullback continuation": "pullback",
            "Trendline breakout": "trendline",
            "Trendline retest continuation": "trendline_retest",
        }
        default_exit_strategy = str(selected_exit_settings.get("strategy_type", strategy_type))
        default_exit_strategy_label = next(
            (label for label, value in exit_strategy_options.items() if value == default_exit_strategy),
            "Breakout continuation",
        )
        edited_exit_strategy_label = st.selectbox(
            "Exit strategy",
            list(exit_strategy_options.keys()),
            index=list(exit_strategy_options.keys()).index(default_exit_strategy_label),
            key=f"exit_strategy_{selected_position_symbol}",
        )
        edited_exit_strategy_type = exit_strategy_options[edited_exit_strategy_label]
        edit_cols = st.columns(3)
        edited_exit_window = edit_cols[0].slider(
            "Sell exit length",
            5,
            30,
            int(selected_exit_settings.get("exit_window", exit_w)),
            step=5,
            key=f"exit_window_{selected_position_symbol}",
        )
        edited_atr_mult = edit_cols[1].slider(
            "Initial stop ATR multiplier",
            0.5,
            5.0,
            float(selected_exit_settings.get("entry_stop_atr_multiplier", selected_exit_settings.get("atr_stop_multiplier", atr_mult))),
            step=0.1,
            key=f"exit_atr_mult_{selected_position_symbol}",
            help=(
                "Recalculates this position's initial stop from its saved ATR at entry and actual Alpaca average fill. "
                "Higher numbers give the position more room and increase its R distance; lower numbers tighten the stop."
            ),
        )
        edited_trend_filter = edit_cols[2].slider(
            "Trend filter length",
            50,
            300,
            int(selected_exit_settings.get("moving_average_window", ma_w)),
            step=50,
            key=f"exit_trend_filter_{selected_position_symbol}",
            help="Calculates the trend filter for this position's saved exit plan.",
        )
        saved_entry_atr = optional_float(selected_exit_settings.get("entry_atr"))
        projected_fill_adjusted_stop = (
            selected_avg_entry - saved_entry_atr * edited_atr_mult
            if selected_avg_entry and saved_entry_atr is not None
            else None
        )
        st.caption(
            f"Projected fill-adjusted initial stop: {money_or_missing(projected_fill_adjusted_stop)}. "
            "Changing this multiplier also changes the position's 1R distance and can loosen or tighten its active stop."
        )
        st.markdown("#### Profit protection")
        profit_protection_enabled = st.checkbox(
            "Protect profits with ATR trail",
            value=bool(selected_exit_settings.get("profit_protection_enabled", True)),
            key=f"profit_protection_enabled_{selected_position_symbol}",
            help="After the position has enough profit, the app can trail from the highest price since entry.",
        )
        protection_cols = st.columns(3)
        edited_breakeven_after_r = protection_cols[0].slider(
            "Move stop to break-even after",
            0.5,
            3.0,
            float(selected_exit_settings.get("breakeven_after_r", 1.0)),
            step=0.5,
            key=f"breakeven_after_r_{selected_position_symbol}",
            help="1R means the trade is up by the amount originally risked.",
        )
        edited_trail_after_r = protection_cols[1].slider(
            "Start ATR trail after",
            1.0,
            5.0,
            float(selected_exit_settings.get("trail_after_r", 2.0)),
            step=0.5,
            key=f"trail_after_r_{selected_position_symbol}",
            help="The ATR trail starts only after this much profit.",
        )
        edited_trailing_atr_multiplier = protection_cols[2].slider(
            "Trailing ATR distance",
            1.0,
            5.0,
            float(selected_exit_settings.get("trailing_atr_multiplier", 3.0)),
            step=0.1,
            key=f"trailing_atr_multiplier_{selected_position_symbol}",
            help="Higher numbers give profitable trades more room before the trail sells.",
        )
        pullback_cols = st.columns(2)
        edited_pullback_length = pullback_cols[0].slider(
            "Pullback average length",
            10,
            200,
            int(selected_exit_settings.get("pullback_average_length", pullback_w)),
            step=5,
            key=f"exit_pullback_length_{selected_position_symbol}",
            help="Used by Trend pullback continuation as the pullback-zone average for this position.",
        )
        edited_momentum_length = pullback_cols[1].slider(
            "Momentum turn length",
            3,
            20,
            int(selected_exit_settings.get("momentum_turn_length", momentum_w)),
            step=1,
            key=f"exit_momentum_length_{selected_position_symbol}",
            help="Used by Trend pullback continuation and Trendline retest continuation to confirm price turned back up.",
        )
        save_exit_settings = st.form_submit_button("Save Exit Settings For This Position")

    if save_exit_settings:
        edited_exit_settings = adjust_initial_stop_settings({
            **selected_exit_settings,
            "symbol": selected_position_symbol,
            "strategy_type": edited_exit_strategy_type,
            "strategy_label": edited_exit_strategy_label,
            "exit_window": edited_exit_window,
            "atr_stop_multiplier": edited_atr_mult,
            "moving_average_window": edited_trend_filter,
            "pullback_average_length": edited_pullback_length,
            "momentum_turn_length": edited_momentum_length,
            "auto_exit_enabled": exit_auto_enabled,
            "profit_protection_enabled": profit_protection_enabled,
            "breakeven_after_r": edited_breakeven_after_r,
            "trail_after_r": edited_trail_after_r,
            "trailing_atr_multiplier": edited_trailing_atr_multiplier,
        }, edited_atr_mult)
        updated_orders = update_exit_settings_for_symbol(
            selected_position_symbol,
            st.session_state["tracked_alpaca_orders"],
            edited_exit_settings,
        )
        broker_state_store.replace_all(updated_orders)
        st.session_state["tracked_alpaca_orders"] = updated_orders
        settings_event = AuditEvent(
            event_type="position_exit_settings_saved",
            message="Exit settings saved for an Alpaca paper position.",
            payload={
                "symbol": selected_position_symbol,
                "exit_settings": edited_exit_settings,
                "broker_writes_submitted": 0,
            },
        )
        st.session_state["session_audit_events"].append(settings_event)
        if persist_audit_log:
            audit_store.append(settings_event)
        st.rerun()

    with st.expander("Saved entry plan", expanded=False):
        if selected_entry_settings:
            st.dataframe(
                pd.DataFrame(entry_snapshot_records(selected_entry_settings)),
                width="stretch",
                hide_index=True,
            )
            if show_portfolio_evidence:
                with st.expander("Raw saved entry settings *", expanded=False):
                    st.dataframe(
                        pd.DataFrame([{"Setting": plain_setting_name(key), "Value": str(value)} for key, value in selected_entry_settings.items()]),
                        width="stretch",
                        hide_index=True,
                    )
        else:
            st.caption("No saved entry settings found for this position.")


if command_center_view == "Open Positions":
    sub_section("Open positions", "Manage each Alpaca paper position and its own exit settings.")
    render_open_positions_panel()
elif command_center_view == "Ideas":
    sub_section("Ideas", "Scan tickers, compare strategy fit, and create a simple research read.")
    scanner_store = ScannerCandidateStore()
    default_scan_text = ", ".join(DEFAULT_SCAN_SYMBOLS)
    scan_symbols_text = st.text_input("Tickers to scan", value=default_scan_text)
    scan_source = data_source if data_source in {"Ticker (Alpaca)", "Ticker (yfinance)"} else "Ticker (Alpaca)"
    st.caption(f"Scanner uses {scan_source}, {period if period != 'synthetic' else '1y'}, {interval if interval != '1d' or data_source != 'Synthetic' else '1h'}.")

    if st.button("Scan Tickers", type="primary"):
        scan_symbols = [item.strip().upper() for item in scan_symbols_text.split(",") if item.strip()]
        scan_period = period if period != "synthetic" else "1y"
        scan_interval = interval if data_source != "Synthetic" else "1h"

        def scan_fetch(symbol: str) -> pd.DataFrame:
            return fetch_price_data_for_source(symbol, scan_period, scan_interval, scan_source)

        with st.spinner("Scanning tickers..."):
            candidates, scan_errors = scan_universe(
                scan_symbols,
                scan_fetch,
                current_strategy_settings | {"history": scan_period, "interval": scan_interval, "price_data_source": scan_source},
                float(paper_order_risk_equity),
                risk_limits,
                max_symbols=30,
            )
        scanner_store.save(candidates, scan_errors)
        st.session_state["selected_scan_symbol"] = candidates[0].symbol if candidates else ""
        st.rerun()

    saved_candidates, saved_scan_errors = scanner_store.read()
    if saved_candidates:
        st.markdown("#### Current ideas")
        st.dataframe(pd.DataFrame(scanner_records(saved_candidates)), width="stretch", hide_index=True)
        selected_scan_symbol = st.selectbox(
            "Research one ticker",
            [candidate.symbol for candidate in saved_candidates],
            index=max(0, [candidate.symbol for candidate in saved_candidates].index(st.session_state.get("selected_scan_symbol", saved_candidates[0].symbol))) if st.session_state.get("selected_scan_symbol") in [candidate.symbol for candidate in saved_candidates] else 0,
        )
        selected_candidate = next(candidate for candidate in saved_candidates if candidate.symbol == selected_scan_symbol)
        research_provider = st.selectbox("Research writer", ["Built-in", "Ollama", "Gemini"], index=0)
        if st.button("Analyze Selected Ticker"):
            config = LLMResearchConfig.from_env(research_provider.lower().replace("built-in", "deterministic"))
            with st.spinner("Building research read..."):
                context = build_company_research_context(selected_candidate.symbol, alpaca_config.api_key, alpaca_config.api_secret)
                result = analyze_candidate(selected_candidate, context, config)
            st.session_state["latest_llm_research"] = asdict(result)
            st.session_state["latest_company_context"] = {
                "Ticker": context.symbol,
                "Event risk": context.event_risk,
                "Event detail": context.event_detail,
                "News": context.news_status,
                "Fundamentals": context.fundamentals_status,
                "Headlines": " | ".join(item.headline for item in context.headlines[:5]) or "None loaded",
            }
            st.rerun()
        if st.session_state.get("latest_llm_research"):
            st.markdown(
                "#### Research read",
                help=(
                    "Summarizes the selected ticker from the scanner, deterministic research, and the optional research writer. "
                    "Use it to understand the idea before opening the ticker in New Trade. It cannot send an order."
                ),
            )
            latest_result = LLMResearchResult(**st.session_state["latest_llm_research"])
            st.dataframe(pd.DataFrame(llm_research_records(latest_result)), width="stretch", hide_index=True)
            if st.session_state.get("latest_company_context"):
                st.markdown("#### Company context")
                st.dataframe(pd.DataFrame([{"Item": key, "Value": value} for key, value in st.session_state["latest_company_context"].items()]), width="stretch", hide_index=True)
            st.caption("This research read cannot send orders. Open the ticker in New Trade before placing a paper buy.")
    else:
        st.info("No ideas scanned yet. Enter tickers above and click Scan Tickers.")
    if saved_scan_errors and show_portfolio_evidence:
        with st.expander("Scanner errors *", expanded=False):
            st.dataframe(pd.DataFrame(saved_scan_errors), width="stretch", hide_index=True)
elif command_center_view == "New Trade":
    sub_section("New trade", "Research the ticker, review the setup, and decide whether to send a paper buy.")
    desk_cols = st.columns(4)
    metric_card(desk_cols[0], "Final Answer", final_answer, final_detail)
    metric_card(desk_cols[1], "Reference Price", f"${float(live['last_p']):,.2f}", ticker)
    metric_card(desk_cols[2], "Strategy", strategy_label, "Selected in the sidebar")
    metric_card(desk_cols[3], "Account P&L today", f"${paper_order_session_pnl:,.2f}", paper_order_account_source, "pos" if paper_order_session_pnl >= 0 else "neg")
    st.info(f"Next action: {new_trade_next_action}")
    st.markdown(
        "#### Research read",
        help=(
            "Summarizes the selected ticker's current setup. Selected strategy is the exact strategy chosen in the sidebar and used for "
            "the TRADE or WAIT decision. Best current fit across all strategies separately compares Breakout continuation, "
            "Trend pullback continuation, Trendline breakout, and Trendline retest continuation using today's BUY-rule progress, "
            "the backtest from the current sidebar settings, trade count, return, win rate, profit factor, and worst drop. "
            "It answers which of those four exact strategies fits the ticker now; it does not search for better settings."
        ),
    )
    st.dataframe(pd.DataFrame(research_agent_records(research_agent_report)), width="stretch", hide_index=True)
    st.markdown(
        "#### Buy watchlist",
        help=(
            "Saves the current ticker, interval, strategy, strategy inputs, risk limits, and paper-order instructions. "
            f"The background worker checks each enabled setup independently. The queue is capped at {MAX_BUY_WATCHLIST_ITEMS} setups. "
            "Allowed symbols is only a whitelist; it does not add rows here."
        ),
    )
    watchlist_plans = buy_watchlist_store.read()
    watchlist_cols = st.columns([1, 1, 2])
    add_watch_disabled = data_source != "Ticker (Alpaca)" or ticker.strip().upper() == "SYNTH"
    if watchlist_cols[0].button(
        "Add or Update Current Setup",
        disabled=add_watch_disabled,
        help="Adds this exact ticker, interval, and strategy as one monitored setup. Adding it again updates its saved inputs.",
        key="add_current_buy_watch_plan",
    ):
        plan = BuyWatchPlan(
            plan_id=buy_watch_plan_id(ticker, interval, strategy_label),
            symbol=ticker,
            interval=interval,
            history=period,
            price_data_source=data_source,
            strategy_label=strategy_label,
            strategy_settings=dict(current_strategy_settings),
            risk_limits=asdict(risk_limits),
            order_style=paper_buy_order_style,
            limit_adjustment_pct=float(paper_buy_limit_adjustment_pct),
            custom_limit_price=float(paper_buy_custom_limit_price),
            enabled=True,
            status="Waiting for BUY",
            detail="Waiting for the saved strategy's required BUY rules.",
        )
        try:
            buy_watchlist_store.upsert(plan)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    watchlist_cols[1].markdown(f"**Queued setups:** {len(watchlist_plans)}/{MAX_BUY_WATCHLIST_ITEMS}")
    watchlist_cols[2].caption(
        "Signals use cached completed bars. Order pricing uses a lightweight latest Alpaca trade every worker check."
    )
    if add_watch_disabled:
        st.caption("Select Ticker (Alpaca) before adding an automatic BUY setup.")
    if watchlist_plans:
        st.dataframe(pd.DataFrame(buy_watchlist_records(watchlist_plans)), width="stretch", hide_index=True)
        watch_labels = {
            f"{plan.symbol} | {plan.interval} | {plan.strategy_label}": plan
            for plan in watchlist_plans
        }
        selected_watch_label = st.selectbox(
            "Manage queued setup",
            list(watch_labels),
            key="selected_buy_watch_plan",
        )
        selected_watch_plan = watch_labels[selected_watch_label]
        with st.expander("Saved setup details", expanded=True):
            st.dataframe(
                pd.DataFrame(buy_watch_plan_detail_records(selected_watch_plan)),
                width="stretch",
                hide_index=True,
            )
        manage_cols = st.columns(2)
        toggle_label = "Pause Selected Setup" if selected_watch_plan.enabled else "Resume Selected Setup"
        if manage_cols[0].button(toggle_label, key="toggle_selected_buy_watch_plan"):
            buy_watchlist_store.update(
                selected_watch_plan.plan_id,
                enabled=not selected_watch_plan.enabled,
                status="Paused" if selected_watch_plan.enabled else "Waiting for BUY",
                detail=(
                    "Paused manually."
                    if selected_watch_plan.enabled
                    else "Waiting for the saved strategy's required BUY rules."
                ),
            )
            st.rerun()
        if manage_cols[1].button("Remove Selected Setup", key="remove_selected_buy_watch_plan"):
            buy_watchlist_store.remove(selected_watch_plan.plan_id)
            st.rerun()
    else:
        st.caption("No queued setups. Configure a real ticker and strategy, then add the current setup.")
    st.markdown("#### Automation readiness")
    st.dataframe(pd.DataFrame(daily_automation_readiness_records("New trade")), width="stretch", hide_index=True)
    with st.expander("Compare all four current strategy fits", expanded=False):
        st.dataframe(pd.DataFrame(strategy_fit_records(research_agent_report)), width="stretch", hide_index=True)
    if show_portfolio_evidence:
        with st.expander("Detailed research records *", expanded=False):
            st.markdown("#### Research loop *")
            st.dataframe(pd.DataFrame(research_loop_rows), width="stretch", hide_index=True)
            st.markdown("#### Setup details *")
            st.dataframe(pd.DataFrame(setup_scorecard_rows), width="stretch", hide_index=True)
            st.markdown("#### Required BUY rules *")
            st.dataframe(pd.DataFrame(buy_requirement_records(live)), width="stretch", hide_index=True)
            st.markdown("#### Strategy use cases *")
            st.dataframe(pd.DataFrame(strategy_use_case_records(strategy_label)), width="stretch", hide_index=True)
            st.markdown("#### Saved research reads *")
            st.dataframe(pd.DataFrame(research_snapshot_records(research_snapshot_store.read_recent(limit=20))), width="stretch", hide_index=True)
    if intent is None:
        st.info("No trade right now. The strategy is waiting.")
    else:
        proposal_cols = st.columns(4)
        metric_card(proposal_cols[0], "Symbol", intent.symbol_clean, intent.side.upper())
        metric_card(proposal_cols[1], "Quantity", f"{intent.quantity:,}", intent.order_type)
        metric_card(proposal_cols[2], "Buy near", f"${intent.entry_price:,.2f}", "Strategy price")
        metric_card(proposal_cols[3], "Stop loss", f"${intent.stop_loss:,.2f}", "Risk rule")
        st.caption(
            f"Order sizing uses {paper_order_account_source}: "
            f"${paper_order_risk_equity:,.2f} account value and ${paper_order_available_cash:,.2f} available cash."
        )
        st.markdown("#### Paper buy price")
        st.dataframe(
            pd.DataFrame(
                paper_buy_price_records(
                    intent,
                    paper_buy_order_style,
                    paper_buy_limit_adjustment_pct,
                    paper_buy_custom_limit_price,
                    float(live.get("last_p", 0) or 0),
                )
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(intent.rationale)
        with st.expander("Trade idea details", expanded=False):
            st.dataframe(pd.DataFrame(proposal_records(trade_proposal)), width="stretch", hide_index=True)
    if show_portfolio_evidence:
        st.markdown("#### New trade evidence *")
        st.dataframe(
            pd.DataFrame(
                agent_loop_stage_records(
                    intent_present=intent is not None,
                    risk_approved=risk_check.approved,
                    preflight_ready=preflight_check.ready,
                    human_gate_required=execution_decision.requires_manual_approval or execution_mode == "paper",
                    broker_connected=alpaca_status.connected,
                )
            ),
            width="stretch",
            hide_index=True,
        )
        with st.expander("Research and strategy context *", expanded=False):
            st.dataframe(
                pd.DataFrame(strategy_context_records(live, entry_w, exit_w, ma_w)),
                width="stretch",
                hide_index=True,
            )
elif command_center_view == "Alpaca":
    sub_section("Alpaca account", "Check broker connection, paper account status, and current Alpaca counts.")
    if show_portfolio_evidence:
        st.markdown("#### Broker details *")
        st.dataframe(pd.DataFrame(broker_status_records(broker_statuses)), width="stretch", hide_index=True)
    if alpaca_state_health.reasons:
        st.warning(" ".join(alpaca_state_health.reasons))

    if show_portfolio_evidence:
        with st.expander("Safety summary *", expanded=False):
            st.markdown("- Alpaca is the target broker. Paper remains the default workflow.")
            st.markdown("- Paper orders require paper mode, passed risk checks, a connected paper account, and the paper account switch.")
            st.markdown("- Live orders require live mode, live Alpaca configuration, the live sidebar switch, passed risk checks, and the Kill Switch off.")
            st.markdown("- The app cannot let the agent change risk rules, credentials, order code, or the Kill Switch.")
            st.markdown("- Automated live submission stays off until manual live order testing is complete.")
            st.markdown("#### Live setup *")
            st.dataframe(pd.DataFrame(live_trading_setup_records()), width="stretch", hide_index=True)
            st.dataframe(pd.DataFrame(production_readiness_checks()), width="stretch", hide_index=True)
            st.dataframe(pd.DataFrame(immutable_boundary_records()), width="stretch", hide_index=True)
            st.markdown("#### Broker failure examples")
            st.dataframe(pd.DataFrame(broker_state_simulation_records()), width="stretch", hide_index=True)
    
    if show_portfolio_evidence:
        with st.expander("Session settings record *", expanded=False):
            st.dataframe(pd.DataFrame(run_manifest_records([current_manifest_record])), width="stretch", hide_index=True)
            if st.button("Save Run Summary"):
                manifest_store.append(current_run_manifest)
                manifest_event = AuditEvent(
                    event_type="run_manifest_recorded",
                message="Run summary recorded locally for this session.",
                    payload={
                        "session_id": current_run_manifest.session_id,
                        "mode_label": current_run_manifest.mode_label,
                        "data_source": current_run_manifest.data_source,
                    },
                )
                st.session_state["session_audit_events"].append(manifest_event)
                if persist_audit_log:
                    audit_store.append(manifest_event)
                st.rerun()
            recent_manifests = manifest_store.read_recent(limit=10)
            if recent_manifests:
                st.markdown("#### Recent run summaries")
                st.dataframe(pd.DataFrame(run_manifest_records(recent_manifests)), width="stretch", hide_index=True)
            st.caption("This saves the app settings used for a session. It does not contact Alpaca.")
    
        with st.expander("Live trading setup records *", expanded=False):
            gitignore_text = Path(".gitignore").read_text(encoding="utf-8") if Path(".gitignore").exists() else ""
            live_lock_rows = live_mode_lockfile_records(live_lockfile_path)
            live_lockfile_present = any(row["Check"] == "Live trading locked" and row["Passed"] for row in live_lock_rows)
            st.markdown("#### Live trading lock")
            st.dataframe(pd.DataFrame(live_lock_rows), width="stretch", hide_index=True)
            if st.button("Create Live Trading Lock"):
                lock_path = write_live_mode_lockfile(live_lockfile_path)
                lock_event = AuditEvent(
                    event_type="live_mode_lockfile_created",
                    message="Live mode lockfile created locally.",
                    payload={"path": str(lock_path), "broker_writes_submitted": 0},
                )
                st.session_state["session_audit_events"].append(lock_event)
                if persist_audit_log:
                    audit_store.append(lock_event)
                st.rerun()
            st.markdown("#### Setup checklist")
            st.dataframe(
                pd.DataFrame(
                    deployment_readiness_records(
                        env_example_present=Path(".env.example").exists(),
                        dotenv_ignored=(".env" in gitignore_text and "!.env.example" in gitignore_text),
                        audit_path_configured=bool(audit_log_path.strip()),
                        broker_state_path_configured=bool(broker_state_path.strip()),
                        evidence_export_path_configured=bool(evidence_export_path.strip()),
                        live_lockfile_present=live_lockfile_present,
                    )
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption("These checks are local only. The lock file does not enable live trading.")
elif command_center_view == "Paper Review":
    sub_section("Paper review", "Review paper trading progress, account activity, and what needs attention.")
    st.markdown("#### Daily paper review")
    st.dataframe(
        pd.DataFrame(
            paper_trading_review_records(
                session_snapshot,
                alpaca_position_count=len(alpaca_positions),
                alpaca_account_value=alpaca_account_equity,
            )
        ),
        width="stretch",
        hide_index=True,
    )
    st.markdown("#### Paper testing progress")
    st.dataframe(
        pd.DataFrame(
            paper_testing_progress_records(
                current_evidence_records,
                st.session_state["tracked_alpaca_orders"],
                target_days=10,
            )
        ),
        width="stretch",
        hide_index=True,
    )
    if st.button("Save Paper Performance Review", key="save_daily_paper_review"):
        performance_event = AuditEvent(
            event_type="paper_performance_reviewed",
            message="Paper trading review was saved by the user.",
            payload={"session_id": st.session_state["paper_session_id"]},
        )
        st.session_state["session_audit_events"].append(performance_event)
        if persist_audit_log:
            audit_store.append(performance_event)
        st.rerun()
    with st.expander("Recent paper activity", expanded=False):
        timeline_rows = session_timeline_records(session_audit_records, limit=25)
        if timeline_rows:
            st.dataframe(pd.DataFrame(timeline_rows), width="stretch", hide_index=True)
        else:
            st.caption("No paper activity recorded in this app session yet.")
    
if command_center_view == "New Trade":
    page_section("Chart and past trades", "Review the selected strategy on historical prices. Click a trade below to highlight it on the chart.")
    selected_idx = st.session_state.get("selected_trade_idx", None)
    selected_trade = trade_log[selected_idx] if selected_idx is not None and 0 <= selected_idx < len(trade_log) else None
    st.plotly_chart(
        build_chart(
            prices,
            smas,
            atrs,
            entry_w,
            exit_w,
            ma_w,
            labels,
            trade_log,
            selected_trade,
            strategy_type=strategy_type,
            pullback_w=pullback_w,
            market_data=market_data,
        ),
        width="stretch",
        config={"scrollZoom": True},
    )
    
    st.markdown(f"#### Past simulated trades ({len(trade_log)} total)")
    if trade_log:
        display_df = pd.DataFrame([{
            "#": t["trade"],
            "Entry Date": t["entry_date"],
            "Exit Date": t["exit_date"],
            "Entry $": t["entry"],
            "Exit $": t["exit"],
            "Shares": t["shares"],
            "Position $": t.get("notional", round(t["entry"] * t["shares"], 2)),
            "Risk $": t.get("risk_dollars", ""),
            "Risk %": t.get("risk_pct", ""),
            "Worst Intratrade P&L $": t.get("max_adverse_pnl", ""),
            "Stop $": t["stop"],
            "Exit Rule": str(t.get("exit_rule", "Exit rule")).title(),
            "Gross P&L $": t.get("gross_pnl", t["pnl"]),
            "Estimated Alpaca Fees $": t.get("estimated_alpaca_fees", 0.0),
            "Net P&L $": t["pnl"],
            "% Account": t["pct_acct"],
        } for t in trade_log]).set_index("#")
    
        def color_pnl(val):
            return "color: #35C46A" if val > 0 else "color: #FF6262"
    
        event = st.dataframe(
            display_df.style.map(color_pnl, subset=["Net P&L $", "% Account"]),
            width="stretch",
            height=min(400, 38 + 35 * len(display_df)),
            on_select="rerun",
            selection_mode="single-row",
        )
        rows = event.selection.rows if event and hasattr(event, "selection") and event.selection else []
        new_idx = rows[0] if rows else None
        if new_idx != st.session_state.get("selected_trade_idx"):
            st.session_state["selected_trade_idx"] = new_idx
            st.rerun()
    
        if selected_trade is not None:
            st.markdown("#### Trade review")
            review = review_closed_trade(selected_trade, trade_proposal.thesis)
            st.dataframe(pd.DataFrame(review_records(review)), width="stretch", hide_index=True)
            with st.expander("Lessons from this trade", expanded=False):
                for lesson in review.lessons:
                    st.markdown(f"- {lesson}")
    else:
        st.caption("No trades happened in this simulation run.")
    
    sub_section("Backtest results")
    st.info(f"Current trading rule: {strategy_label}. Only the selected strategy can create the trade idea shown in this run.")
    c1, c2, c3, c4 = st.columns(4)
    pnl_color = "pos" if stats["total_pnl"] >= 0 else "neg"
    metric_card(
        c1,
        "Final equity",
        f"${stats['final_equity']:,}",
        f"Net P&L ${stats['total_pnl']:,}; estimated Alpaca fees ${stats.get('estimated_alpaca_fees', 0):,.2f}",
        pnl_color,
    )
    metric_card(c2, "Account return", f"{stats['return_pct']}%", "Impact on the complete account", pnl_color)
    annualized_allocated = stats.get("annualized_allocated_return_pct")
    allocated_sub = f"${stats.get('allocated_capital', account):,.0f} ticker allocation"
    if annualized_allocated is not None:
        allocated_sub += f"; {annualized_allocated:.2f}% annualized"
    metric_card(c3, "Allocated return", f"{stats.get('allocated_return_pct', stats['return_pct'])}%", allocated_sub, pnl_color)
    if benchmark is not None:
        excess = stats.get("allocated_return_pct", stats["return_pct"]) - benchmark.return_percent
        excess_sub = f"Buy and hold {benchmark.return_percent:.2f}%"
        if annualized_allocated is not None and benchmark.annualized_return_percent is not None:
            excess_sub += f"; {annualized_allocated - benchmark.annualized_return_percent:+.2f}% annualized excess"
        metric_card(c4, "Excess vs buy and hold", f"{excess:+.2f}%", excess_sub, "pos" if excess >= 0 else "neg")
    else:
        metric_card(c4, "Annualized return", "Not shown" if annualized_allocated is None else f"{annualized_allocated:.2f}%", "Shown for periods longer than one year")
    
    c5, c6, c7, c8 = st.columns(4)
    metric_card(c5, "Win rate", f"{stats['win_rate']}%", f"{stats['wins']}W / {stats['losses']}L of {stats['total_trades']} trades")
    metric_card(c6, "Allocated worst drop", f"{stats.get('allocated_max_drawdown_pct', stats['max_drawdown_pct'])}%", "Largest drop versus ticker allocation")
    metric_card(c7, "Win/loss dollars", f"{stats['profit_factor']}x", "Total wins vs total losses")
    metric_card(c8, "Time in trade", f"{stats['exposure_pct']}%", "Share of bars spent in a trade")
    with st.expander("Backtest assumptions and exit model", expanded=False):
        st.markdown(
            "Historical signals use completed bars. Entries use the signal-bar close. "
            "Protective stops fill at the stop price, or at the bar open after a gap below the stop. "
            f"Results include estimated Alpaca U.S. equity regulatory fees using the fee schedule effective "
            f"{ALPACA_EQUITY_FEE_SCHEDULE_EFFECTIVE} and a 0% direct-account commission assumption. "
            "Paper trading does not deduct these fees, so they are included here as live-equivalent costs. "
            "The estimate conservatively rounds each order's applicable fee components upward; Alpaca live "
            "aggregates each fee type by account and day, so actual day-end charges may differ by a few cents. "
            "Spread, slippage, market impact, taxes, and idle-cash interest are not included."
        )
        st.dataframe(pd.DataFrame(exit_model_records()), width="stretch", hide_index=True)
    
    with st.expander("Optional strategy tests" + (" *" if show_portfolio_evidence else ""), expanded=False):
        st.markdown(
            "#### Strategy comparison",
            help=(
                "Runs all four strategies on the same ticker, interval, history, account size, risk limits, and current sidebar settings. "
                "Compare return, completed trades, win rate, worst drop, and profit factor. This table does not search for better settings "
                "and does not change the selected strategy."
            ),
        )
        st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)
        st.caption(
            "This compares all four strategies and buy-and-hold using the same ticker allocation set by Max symbol concentration. "
            "Account return remains visible separately. Buy and hold uses adjusted closing prices; taxes, idle-cash interest, "
            "spread, slippage, and market impact are not included. Strategy results are net of estimated Alpaca regulatory fees; "
            "buy and hold includes one estimated buy and one estimated sell."
        )
        
        st.markdown("#### Test on newer price data")
        if walk_forward_result is None:
            if walk_forward_error:
                st.warning(walk_forward_error)
            else:
                st.caption("Testing on newer data is turned off.")
        else:
            verdict_color = {
                "Pass": "#3B6D11",
                "Inconclusive": "#8A6D1D",
                "Needs review": "#A32D2D",
            }.get(walk_forward_result.verdict, "inherit")
            st.markdown(
                f"**Result:** "
                f"<span style='color:{verdict_color};font-weight:600'>{walk_forward_result.verdict}</span>",
                unsafe_allow_html=True,
            )
            if show_portfolio_evidence:
                st.dataframe(
                    pd.DataFrame(walk_forward_records(walk_forward_result)),
                    width="stretch",
                    hide_index=True,
                )
            with st.expander("What this test used", expanded=False):
                st.markdown(
                    f"Older bars used first: **{walk_forward_result.train_bars}**. "
                    f"Newer bars used for the final test: **{walk_forward_result.oos_bars}**. "
                    f"Extra bars needed for indicators: **{walk_forward_result.warmup_bars}**."
                )
                for reason in walk_forward_result.reasons:
                    st.markdown(f"- {reason}")
        
        st.markdown(
            "#### Strategy input search result",
            help=(
                "Searches nearby input combinations for all four strategies, then favors settings that also hold up on newer data, "
                "nearby settings, separate time periods, an untouched final period, and simulated trading friction. "
                "Every setting combination is tested with the RSI 50-70 BUY rule off and on. "
                "Use this as the strongest candidate for paper testing, not as a promise of future profit."
            ),
        )
        if optimizer_search_state is None:
            st.caption("Click Run Strategy Input Search in the sidebar when you want a recommendation.")
        elif parameter_loop_error:
            st.warning(parameter_loop_error)
        elif strategy_optimizer_result is None:
            st.caption("No recommendation is available yet.")
        else:
            if optimizer_result_stale:
                st.warning("Inputs changed since this result was created. Run Strategy Input Search again before using the recommendation.")
            recommendation_summary_text = (
                strategy_optimizer_interval_result.summary
                if strategy_optimizer_interval_result is not None
                else strategy_optimizer_result.summary
            )
            st.info(recommendation_summary_text)
            verdict_interval = (
                strategy_optimizer_interval_result.best_interval
                if strategy_optimizer_interval_result is not None
                else interval
            )
            setup_verdict = candidate_verdict(strategy_optimizer_result, verdict_interval)
            st.markdown(
                "#### Recommended setup verdict",
                help=(
                    "A deterministic classification of the recommended strategy and inputs. Strong Candidate has broad "
                    "support across unseen periods, costs, nearby settings, trade count, and drawdown. Promising Candidate "
                    "allows modest or incomplete evidence but no major contradiction. Research Only needs more evidence. "
                    "Reject means a core test clearly failed."
                ),
            )
            verdict_message = (
                f"**{setup_verdict.tier}** - {strategy_optimizer_result.best.strategy_label if strategy_optimizer_result.best else 'No strategy'} "
                f"on {verdict_interval}. {setup_verdict.summary}"
            )
            if setup_verdict.tier == "Strong Candidate":
                st.success(verdict_message)
            elif setup_verdict.tier == "Promising Candidate":
                st.info(verdict_message)
            elif setup_verdict.tier == "Reject":
                st.error(verdict_message)
            else:
                st.warning(verdict_message)
            recommendation_rows = optimizer_recommendation_records(strategy_optimizer_result, verdict_interval)
            if strategy_optimizer_interval_result is not None:
                best_interval_evidence = next(
                    row for row in strategy_optimizer_interval_result.interval_results
                    if row.interval == strategy_optimizer_interval_result.best_interval
                )
                recommendation_rows.insert(0, {
                    "Item": "Recommended interval",
                    "Value": strategy_optimizer_interval_result.best_interval,
                    "Plain English": (
                        "Best result when daily, 4-hour, and 1-hour data were compared over the same "
                        f"{strategy_optimizer_interval_result.interval_results[0].comparison_history.lower()}. "
                        f"The winning settings were then checked without changes on "
                        f"{strategy_optimizer_interval_result.best_history} of available history."
                    ),
                })
                recommendation_rows.insert(1, {
                    "Item": "Complete-period comparison",
                    "Value": f"{best_interval_evidence.comparison_excess_return_percent:+.2f}% vs buy-and-hold",
                    "Plain English": (
                        f"Across the complete {best_interval_evidence.comparison_history.lower()}, the strategy returned "
                        f"{best_interval_evidence.comparison_return_percent:.2f}% and equal-capital buy-and-hold returned "
                        f"{best_interval_evidence.comparison_benchmark_return_percent:.2f}%. This is the comparison that "
                        "matches the full Backtest results after these inputs are applied."
                    ),
                })
            st.dataframe(
                pd.DataFrame(recommendation_rows),
                width="stretch",
                hide_index=True,
            )
            recommendation_status = next(
                (row["Value"] for row in recommendation_rows if row["Item"] == "Recommendation status"),
                "Research only",
            )
            apply_button_label = (
                "Use Recommended Inputs"
                if recommendation_status in {"Strong Candidate", "Promising Candidate"}
                else "Use Research Candidate Inputs"
            )
            if strategy_optimizer_result.best is not None and st.button(
                apply_button_label,
                disabled=optimizer_result_stale,
            ):
                apply_settings = dict(strategy_optimizer_result.best.settings)
                apply_settings["risk_per_trade_pct"] = strategy_optimizer_result.best.recommended_risk_per_trade_percent
                st.session_state["optimizer_apply_settings"] = apply_settings
                if strategy_optimizer_interval_result is not None:
                    st.session_state["optimizer_apply_interval"] = strategy_optimizer_interval_result.best_interval
                    st.session_state["optimizer_apply_history"] = strategy_optimizer_interval_result.best_history
                st.rerun()
            if strategy_optimizer_interval_result is not None:
                with st.expander("Compare daily, 4-hour, and 1-hour results", expanded=False):
                    st.dataframe(
                        pd.DataFrame(optimizer_interval_records(strategy_optimizer_interval_result)),
                        width="stretch",
                        hide_index=True,
                    )
                    st.caption(
                        "The interval ranking uses the same calendar period for a fair comparison. Long-history columns "
                        "then show how each interval's unchanged winning settings behaved over all available data. "
                        "The app also considers newer data, the untouched final period, nearby settings, and trading costs."
                    )
                    if strategy_optimizer_interval_errors:
                        st.warning("Some intervals could not be tested: " + "; ".join(strategy_optimizer_interval_errors))
            with st.expander("Why this recommendation", expanded=False):
                verdict_rows = candidate_verdict_records(setup_verdict)
                if verdict_rows:
                    st.markdown("##### Candidate verdict evidence")
                    st.dataframe(
                        pd.DataFrame(verdict_rows),
                        width="stretch",
                        hide_index=True,
                    )
                st.dataframe(
                    pd.DataFrame(optimizer_robustness_records(strategy_optimizer_result)),
                    width="stretch",
                    hide_index=True,
                )
                st.markdown("##### Trading-cost test")
                st.dataframe(
                    pd.DataFrame(optimizer_stress_records(strategy_optimizer_result)),
                    width="stretch",
                    hide_index=True,
                )
                st.markdown("##### Results by market condition")
                st.dataframe(
                    pd.DataFrame(optimizer_regime_records(strategy_optimizer_result)),
                    width="stretch",
                    hide_index=True,
                )
            with st.expander("Test these settings on other tickers", expanded=False):
                cross_ticker_context = (ticker, data_source, interval, period)
                if optimizer_result_stale:
                    st.caption("Run Strategy Input Search again before testing this recommendation on other tickers.")
                elif data_source == "Synthetic":
                    st.caption("Choose Ticker (Alpaca) or Ticker (yfinance) first. This check needs real price history.")
                elif strategy_optimizer_result.best is None:
                    st.caption("No recommended settings are available to test.")
                else:
                    comparison_symbols = st.text_input(
                        "Other tickers",
                        value="MSFT, NVDA, AMZN, META, GOOGL",
                        help="The app applies the recommended settings without changing them. Use liquid stocks that are not the ticker used to choose the settings.",
                    )
                    if st.button("Run Other-Ticker Test"):
                        symbols = []
                        for value in comparison_symbols.split(","):
                            candidate_symbol = value.strip().upper()
                            if candidate_symbol and candidate_symbol != ticker and candidate_symbol not in symbols:
                                symbols.append(candidate_symbol)
                        symbols = symbols[:8]
                        loaded = {}
                        problems = []
                        with st.spinner("Testing unchanged settings on other tickers..."):
                            for comparison_symbol in symbols:
                                try:
                                    loaded[comparison_symbol] = fetch_price_data_for_source(
                                        comparison_symbol, period, interval, data_source
                                    )
                                except Exception as exc:
                                    problems.append(f"{comparison_symbol}: {exc}")
                        cross_result = validate_settings_across_tickers(
                            strategy_optimizer_result.best.settings,
                            loaded,
                            float(paper_order_risk_equity),
                            risk_limits,
                        )
                        st.session_state["optimizer_cross_ticker"] = {
                            "settings": dict(strategy_optimizer_result.best.settings),
                            "context": cross_ticker_context,
                            "result": cross_result,
                            "problems": problems,
                        }
                    cross_state = st.session_state.get("optimizer_cross_ticker")
                    if (
                        cross_state
                        and cross_state.get("settings") == strategy_optimizer_result.best.settings
                        and cross_state.get("context") == cross_ticker_context
                    ):
                        cross_result = cross_state["result"]
                        st.markdown(
                            f"**Profitable on {cross_result.profitable_tickers} of {cross_result.tested_tickers} other tickers.** "
                            f"Median return: **{cross_result.median_return_percent:.2f}%**; "
                            f"worst drop: **{cross_result.worst_drawdown_percent:.2f}%**."
                        )
                        st.dataframe(pd.DataFrame(cross_result.rows), width="stretch", hide_index=True)
                        for problem in cross_state.get("problems", []):
                            st.warning(problem)
            if show_portfolio_evidence:
                st.markdown("#### Top tested settings *")
                st.dataframe(
                    pd.DataFrame(optimizer_candidate_records(strategy_optimizer_result.candidates)),
                    width="stretch",
                    hide_index=True,
                )
            st.caption(
                "This searches bounded settings and favors stable ranges, rolling results, an untouched final period, and realistic trading costs. It is not proof of future profit. "
                "It does not change account risk limits, broker access, order mode, credentials, or the Kill Switch."
            )
    
    if show_portfolio_evidence:
        page_section("Trade details *", "Full Records view only: detailed trade rules, agent notes, and risk records.")
        detail_tabs = st.tabs(["Rules *", "Agent notes *", "Risk records *"])
        with detail_tabs[0]:
            sig = live["signal"]
            if sig == "long":
                st.markdown('<span class="signal-long">BUY SIGNAL</span>', unsafe_allow_html=True)
            elif sig == "exit":
                st.markdown('<span class="signal-exit">SELL SIGNAL</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="signal-flat">NO TRADE RIGHT NOW</span>', unsafe_allow_html=True)
            st.markdown("#### Required for BUY")
            st.dataframe(pd.DataFrame(buy_requirement_records(live)), width="stretch", hide_index=True)
            st.markdown("#### Required for automatic SELL")
            st.dataframe(
                pd.DataFrame(
                    sell_requirement_records(
                        live,
                        exit_preview_count=len(exit_previews),
                        exit_settings_saved=exit_settings_available if first_exit_symbol else None,
                    )
                ),
                width="stretch",
                hide_index=True,
            )
            st.markdown("#### Quality checks")
            st.dataframe(pd.DataFrame(optional_quality_input_records(setup_inputs)), width="stretch", hide_index=True)
        with detail_tabs[1]:
            st.dataframe(pd.DataFrame(proposal_records(trade_proposal)), width="stretch", hide_index=True)
            st.markdown(f"**Trade idea:** {trade_proposal.thesis.thesis}")
            st.markdown(f"**What would make this wrong:** {trade_proposal.thesis.invalidation}")
            st.markdown("#### What the agent used")
            for item in trade_proposal.thesis.data_basis:
                st.markdown(f"- {item}")
        with detail_tabs[2]:
            st.markdown(
                f"**Final answer:** "
                f"<span style='color:{answer_color};font-weight:700'>{final_answer}</span> - {final_detail}",
                unsafe_allow_html=True,
            )
            st.dataframe(
                pd.DataFrame(
                    trade_evidence_summary_records(
                        intent_present=intent is not None,
                        risk_approved=risk_check.approved,
                        preflight_ready=preflight_check.ready,
                        setup_rows=setup_scorecard_rows,
                        blocked_reasons=preflight_check.blocked_reasons or risk_check.rejected_reasons,
                    )
                ),
                width="stretch",
                hide_index=True,
            )
            st.dataframe(pd.DataFrame(risk_policy_records(risk_limits)), width="stretch", hide_index=True)
            st.dataframe(pd.DataFrame(preflight_records(preflight_check)), width="stretch", hide_index=True)
            checks_df = pd.DataFrame([{"Check": name.replace("_", " ").title(), "Passed": passed} for name, passed in risk_check.checks.items()])
            if not checks_df.empty:
                st.dataframe(checks_df, width="stretch", hide_index=True)
if command_center_view == "Alpaca":
    st.info(f"{operator_state['State']}: {operator_state['Next Action']}")
    if show_portfolio_evidence:
        st.markdown("#### Alpaca evidence summary *")
        st.dataframe(
            pd.DataFrame(
                alpaca_evidence_summary_records(
                    alpaca_connected=alpaca_status.connected,
                    alpaca_state_stale=alpaca_state_health.stale,
                    paper_orders_enabled=enable_alpaca_paper_orders,
                    alpaca_positions=alpaca_positions,
                    alpaca_orders=alpaca_orders,
                    tracked_orders=st.session_state["tracked_alpaca_orders"],
                    automation_status=active_automation_level,
                )
            ),
            width="stretch",
            hide_index=True,
        )
    st.markdown("#### Account overview")
    st.dataframe(pd.DataFrame(alpaca_daily_summary_records()), width="stretch", hide_index=True)
    with st.expander("Account, positions, and orders" + (" *" if show_portfolio_evidence else ""), expanded=False):
        if show_portfolio_evidence:
            st.markdown("#### Broker details *")
            st.dataframe(
                pd.DataFrame(broker_status_records(broker_statuses)),
                width="stretch",
                hide_index=True,
            )
            st.markdown("#### Alpaca setup checks *")
            st.dataframe(pd.DataFrame(alpaca_config_validation_records(alpaca_adapter.config)), width="stretch", hide_index=True)
        alpaca_tabs = st.tabs(["Alpaca account", "Alpaca positions", "Alpaca orders"])
        with alpaca_tabs[0]:
            account_records = alpaca_account_records
            if account_records:
                st.dataframe(pd.DataFrame(account_records), width="stretch", hide_index=True)
            else:
                st.caption("No Alpaca account data available. Configure paper credentials and install alpaca-py.")
        with alpaca_tabs[1]:
            if alpaca_positions:
                st.dataframe(pd.DataFrame(alpaca_positions), width="stretch", hide_index=True)
            else:
                st.caption("No Alpaca positions available.")
        with alpaca_tabs[2]:
            if alpaca_orders:
                st.dataframe(pd.DataFrame(alpaca_orders), width="stretch", hide_index=True)
            else:
                st.caption("No Alpaca orders available.")
        reconcile_rows = reconcile_alpaca_positions(alpaca_positions, st.session_state["tracked_alpaca_orders"])
        if show_portfolio_evidence and reconcile_rows:
            st.markdown("#### Position tracking details *")
            st.dataframe(pd.DataFrame(reconcile_rows), width="stretch", hide_index=True)
        if show_portfolio_evidence and exit_previews:
            st.markdown("#### Exit order details *")
            st.dataframe(pd.DataFrame(exit_preview_records(exit_previews)), width="stretch", hide_index=True)
        if alpaca_state_health.reasons:
            st.warning(" ".join(alpaca_state_health.reasons))
    
    if show_portfolio_evidence:
        with st.expander("Local simulator details *", expanded=False):
            broker_cols = st.columns(4)
            metric_card(broker_cols[0], "Simulator cash", f"${paper_broker.cash:,.2f}", "Resets with the app session")
            metric_card(broker_cols[1], "Simulator positions", f"{len(paper_broker.positions)}", "Local only")
            metric_card(broker_cols[2], "Simulator orders", f"{len(paper_broker.orders)}", "Submitted this session")
            metric_card(
                broker_cols[3],
                "Order status",
                "Ready" if preflight_check.ready else "Blocked",
                "Checks passed" if preflight_check.ready else "Checks blocked",
            )
            portfolio_cols = st.columns(3)
            session_color = "pos" if session_pnl >= 0 else "neg"
            metric_card(portfolio_cols[0], "Simulator equity", f"${paper_equity:,.2f}", "Cash plus local positions")
            metric_card(portfolio_cols[1], "Session P&L", f"${session_pnl:,.2f}", "Since last reset", session_color)
            metric_card(portfolio_cols[2], "Simulator exposure", f"${paper_positions_notional:,.2f}", "Book value")
    
    st.markdown("#### Trading status")
    monitor_color = {"OK": "#3B6D11", "WARN": "#8A6D1D", "BREACH": "#A32D2D"}.get(monitoring_result.status, "inherit")
    monitor_label = {"OK": "OK", "WARN": "Needs attention", "BREACH": "Blocked"}.get(monitoring_result.status, monitoring_result.status)
    st.markdown(
        f"**Trading status:** "
        f"<span style='color:{monitor_color};font-weight:600'>{monitor_label}</span>",
        unsafe_allow_html=True,
    )
    st.caption(current_market_advisory.get("Message", ""))
    if show_portfolio_evidence:
        st.dataframe(pd.DataFrame(monitoring_records(monitoring_result)), width="stretch", hide_index=True)
        st.dataframe(pd.DataFrame([current_market_advisory]), width="stretch", hide_index=True)
        st.markdown("#### Broker health details *")
        st.dataframe(
            pd.DataFrame(
                broker_heartbeat_records(
                    broker_connected=alpaca_status.connected,
                    broker_state_stale=alpaca_state_health.stale,
                    broker_reasons=alpaca_state_health.reasons,
                    market_advisory=current_market_advisory,
                )
            ),
            width="stretch",
            hide_index=True,
        )
    for alert in monitoring_result.alerts:
        if monitoring_result.status == "BREACH":
            st.error(alert)
        elif monitoring_result.status == "WARN":
            st.warning(alert)
        else:
            st.caption(alert)
    
    render_automation_status()

    st.markdown("#### Order actions")
    st.caption("Send a buy, sell an open position, or cancel a waiting order for the configured Alpaca account.")
    if waiting_limit_buy_rows:
        st.markdown("#### Waiting limit buys")
        st.dataframe(pd.DataFrame(waiting_limit_buy_rows), width="stretch", hide_index=True)
        if auto_cancel_stale_limit_orders:
            st.caption(f"Auto-cancel is on: an unfilled BUY limit order is canceled after {stale_limit_order_label}.")
        else:
            st.caption("Auto-cancel is off: waiting limit buys stay open until you cancel them or Alpaca expires them.")
    if show_portfolio_evidence and (auto_cancel_stale_limit_orders or waiting_limit_buy_rows):
        with st.expander("Limit buy auto-cancel details *", expanded=False):
            st.dataframe(
                pd.DataFrame(stale_limit_cancel_status_records(managed_cancelable_alpaca_orders, stale_limit_order_minutes)),
                width="stretch",
                hide_index=True,
            )
            if managed_cancelable_alpaca_orders:
                st.dataframe(
                    pd.DataFrame(cancelable_order_debug_records(managed_cancelable_alpaca_orders)),
                    width="stretch",
                    hide_index=True,
                )
    can_submit = intent is not None and alpaca_manual_order_mode
    submit_disabled = intent is None or not preflight_check.ready or not alpaca_manual_order_mode
    if show_portfolio_evidence:
        with st.expander("Local simulator order *", expanded=False):
            if st.button("Submit Local Simulator Order", disabled=submit_disabled):
                order = paper_adapter.submit_order(intent, execution_decision)
                paper_event = AuditEvent(
                    event_type=f"paper_order_{order.status}",
                    message=order.message,
                    payload={
                        "order_id": order.order_id,
                        "symbol": order.symbol,
                        "side": order.side,
                        "quantity": order.quantity,
                        "status": order.status,
                        "filled_price": order.filled_price,
                        "paper_cash": paper_broker.cash,
                    },
                )
                st.session_state["session_audit_events"].append(paper_event)
                if persist_audit_log:
                    audit_store.append(paper_event)
                st.rerun()
            st.caption("This updates only the in-app simulator. It does not contact Alpaca.")
    
    alpaca_base_disabled = (
        intent is None
        or not preflight_check.ready
        or not alpaca_manual_order_mode
        or not alpaca_mode_matches_order_mode
        or not alpaca_status.connected
        or not alpaca_orders_enabled_for_mode
        or not alpaca_preview.valid
        or alpaca_state_health.stale
        or bool(duplicate_alpaca_reasons)
        or bool(open_order_reasons)
        or duplicate_preview_submitted
    )
    if intent is not None:
        st.markdown(f"#### Manual {alpaca_order_noun} buy")
        st.dataframe(
            pd.DataFrame(
                paper_buy_price_records(
                    intent,
                    paper_buy_order_style,
                    paper_buy_limit_adjustment_pct,
                    paper_buy_custom_limit_price,
                    float(live.get("last_p", 0) or 0),
                )
            ),
            width="stretch",
            hide_index=True,
        )
        if show_portfolio_evidence:
            st.dataframe(pd.DataFrame(alpaca_preview_records(alpaca_preview)), width="stretch", hide_index=True)
        elif alpaca_preview.valid:
            limit_note = (
                f" limit ${float(alpaca_preview.order.get('limit_price')):,.2f}"
                if alpaca_preview.order.get("limit_price")
                else " market"
            )
            st.info(f"Ready: buy {alpaca_preview.order.get('quantity', '')} {alpaca_preview.order.get('symbol', '')} with a{limit_note} order in {alpaca_account_label}.")
        if alpaca_preview.blocked_reasons:
            show_blockers("Order blocked", alpaca_preview.blocked_reasons)
        if not alpaca_mode_matches_order_mode:
            st.warning(f"Order mode and Alpaca account do not match. Current Alpaca account is {alpaca_account_mode}.")
        if duplicate_alpaca_reasons:
            show_blockers("Order blocked", duplicate_alpaca_reasons)
        if open_order_reasons:
            show_blockers("Order blocked", open_order_reasons)
        if duplicate_preview_submitted:
            st.warning("This order is already tracked in the app.")
        if alpaca_state_health.stale:
            st.warning("Refresh Alpaca positions and orders before sending this.")
        elif not duplicate_alpaca_reasons and not open_order_reasons and not duplicate_preview_submitted:
            st.success(f"{alpaca_account_label.title()} buy is ready to send.")
    
    alpaca_submit_disabled = (
        alpaca_base_disabled
        or not alpaca_status.can_submit_orders
    )
    buy_button_label = "Send Paper Buy to Alpaca" if alpaca_adapter.config.paper else "Send Live Buy to Alpaca"
    if st.button(buy_button_label, disabled=alpaca_submit_disabled):
        try:
            alpaca_order = alpaca_adapter.submit_order(
                intent,
                execution_decision,
                expected_preview_hash=alpaca_preview.preview_hash,
            )
            alpaca_event = AuditEvent(
                event_type=f"alpaca_{alpaca_order_noun}_order_submitted",
                message=f"Alpaca {alpaca_order_noun} order submitted through the gated adapter.",
                payload={
                    "symbol": intent.symbol_clean,
                    "side": intent.side,
                    "quantity": intent.quantity,
                    "order_type": intent.order_type,
                    "limit_price": intent.limit_price,
                    "preview_hash": alpaca_preview.preview_hash,
                    "broker_order_id": str(getattr(alpaca_order, "id", "")),
                    "strategy_settings": current_strategy_settings,
                    "exit_settings": current_exit_settings,
                },
            )
            broker_order_id = str(getattr(alpaca_order, "id", ""))
            if broker_order_id:
                tracked_record = {
                    "broker_order_id": broker_order_id,
                    "preview_hash": alpaca_preview.preview_hash,
                    "symbol": intent.symbol_clean,
                    "side": intent.side,
                    "quantity": intent.quantity,
                    "order_type": intent.order_type,
                    "limit_price": intent.limit_price,
                    "status": str(getattr(alpaca_order, "status", "")),
                    "submitted_at": str(getattr(alpaca_order, "submitted_at", "")),
                    "strategy_settings": current_strategy_settings,
                    "exit_settings": current_exit_settings,
                }
                st.session_state["tracked_alpaca_orders"].append(tracked_record)
                broker_state_store.upsert(tracked_record)
                updated_orders = update_exit_settings_for_symbol(
                    intent.symbol_clean,
                    st.session_state["tracked_alpaca_orders"],
                    current_exit_settings,
                )
                broker_state_store.replace_all(updated_orders)
                st.session_state["tracked_alpaca_orders"] = updated_orders
            st.session_state["session_audit_events"].append(alpaca_event)
            if persist_audit_log:
                audit_store.append(alpaca_event)
            st.success(f"{alpaca_account_label.title()} buy order sent to Alpaca.")
        except Exception as exc:
            alpaca_event = AuditEvent(
                event_type=f"alpaca_{alpaca_order_noun}_order_blocked",
                message=str(exc),
                payload={"symbol": intent.symbol_clean if intent else None, "preview_hash": alpaca_preview.preview_hash},
            )
            st.session_state["session_audit_events"].append(alpaca_event)
            if persist_audit_log:
                audit_store.append(alpaca_event)
            st.error(f"{alpaca_account_label.title()} buy order blocked: {exc}")
    
    if show_portfolio_evidence and tracked_alpaca_orders:
        st.markdown("#### Alpaca orders saved in the app *")
        tracked_rows = [
            alpaca_adapter.tracked_order_record(
                item.get("broker_order_id", ""),
                item.get("preview_hash", ""),
            )
            for item in tracked_alpaca_orders
        ]
        st.dataframe(pd.DataFrame(tracked_rows), width="stretch", hide_index=True)
    
    if st.session_state.get("tracked_alpaca_orders") or alpaca_positions:
        tracked_refresh_rows = alpaca_adapter.refreshed_tracked_order_records(st.session_state["tracked_alpaca_orders"])
        lifecycle_alpaca_orders = {alpaca_order_row_id(row): row for row in alpaca_orders if alpaca_order_row_id(row)}
        for row in tracked_refresh_rows:
            row_id = alpaca_order_row_id(row)
            if row_id:
                lifecycle_alpaca_orders[row_id] = row
        lifecycle_order_rows = list(lifecycle_alpaca_orders.values())
        lifecycle_rows = alpaca_order_lifecycle_records(st.session_state["tracked_alpaca_orders"], lifecycle_order_rows)
        refreshed_order_state = refresh_tracked_alpaca_orders(st.session_state["tracked_alpaca_orders"], lifecycle_order_rows)
        lifecycle_summary = alpaca_order_lifecycle_summary_records(
            refreshed_order_state
        )
        if show_portfolio_evidence:
            st.markdown("#### Alpaca order history *")
            st.dataframe(pd.DataFrame(lifecycle_summary), width="stretch", hide_index=True)
            st.dataframe(pd.DataFrame(lifecycle_rows), width="stretch", hide_index=True)
    
        position_lifecycle_rows = alpaca_position_lifecycle_records(alpaca_positions, refreshed_order_state)
        if position_lifecycle_rows:
            if show_portfolio_evidence:
                st.markdown("#### Paper position history *")
                st.dataframe(
                    pd.DataFrame(alpaca_position_lifecycle_summary_records(position_lifecycle_rows)),
                    width="stretch",
                    hide_index=True,
                )
                st.dataframe(pd.DataFrame(position_lifecycle_rows), width="stretch", hide_index=True)
                st.caption("These rows explain how Alpaca positions connect to local app records.")
            untracked_position_rows = [row for row in position_lifecycle_rows if row.get("Tracking Status") == "Needs app tracking"]
            if untracked_position_rows:
                st.markdown("#### Track Alpaca position in this app")
                st.warning("Alpaca has a position that is not linked to an app record yet.")
                adopt_options = [
                    f"{row.get('Symbol', '')} qty {row.get('Position Qty', '')}"
                    for row in untracked_position_rows
                ]
                selected_adopt_idx = st.selectbox(
                    "Position to track",
                    range(len(adopt_options)),
                    format_func=lambda idx: adopt_options[idx],
                )
                selected_adopt_symbol = str(untracked_position_rows[selected_adopt_idx].get("Symbol", "")).strip().upper()
                selected_adopt_position = next(
                    (position for position in alpaca_positions if str(position.get("Symbol", "")).strip().upper() == selected_adopt_symbol),
                    None,
                )
                adopt_disabled = selected_adopt_position is None or not alpaca_status.connected or alpaca_state_health.stale
                if st.button("Track This Alpaca Position in the App", disabled=adopt_disabled):
                    adopted_record = adopt_alpaca_position(selected_adopt_position)
                    broker_state_store.upsert(adopted_record)
                    st.session_state["tracked_alpaca_orders"] = broker_state_store.read()
                    adoption_event = AuditEvent(
                        event_type="alpaca_paper_position_adopted",
                        message="Alpaca paper position is now tracked by the app.",
                        payload={
                            "broker_order_id": adopted_record.get("broker_order_id", ""),
                            "symbol": adopted_record.get("symbol", ""),
                            "quantity": adopted_record.get("filled_quantity", ""),
                            "average_fill_price": adopted_record.get("average_fill_price", ""),
                            "source": adopted_record.get("source", ""),
                            "broker_writes_submitted": 0,
                        },
                    )
                    st.session_state["session_audit_events"].append(adoption_event)
                    if persist_audit_log:
                        audit_store.append(adoption_event)
                    st.rerun()
                st.caption("This only updates the app's local record. It does not send anything to Alpaca.")
    
        refresh_state_disabled = not alpaca_status.connected or alpaca_state_health.stale
        if st.button("Refresh Alpaca Orders From Alpaca", disabled=refresh_state_disabled):
            refreshed_orders = refreshed_order_state
            broker_state_store.replace_all(refreshed_orders)
            st.session_state["tracked_alpaca_orders"] = refreshed_orders
            refresh_event = AuditEvent(
                event_type="alpaca_paper_order_state_refreshed",
                message="Saved Alpaca paper orders refreshed from Alpaca.",
                payload={
                    "tracked_orders": len(refreshed_orders),
                    "lifecycle_summary": lifecycle_summary,
                },
            )
            st.session_state["session_audit_events"].append(refresh_event)
            if persist_audit_log:
                audit_store.append(refresh_event)
            st.rerun()
        st.caption("Refresh reads Alpaca Orders and updates only the app's local tracking file.")
    
    if show_portfolio_evidence and st.session_state.get("tracked_alpaca_orders"):
        with st.expander("Practice fill and exit records *", expanded=False):
            simulator_options = [
                f"{order.get('symbol', '')} {order.get('side', '')} {order.get('quantity', '')} ({str(order.get('broker_order_id', ''))[:8]})"
                for order in st.session_state["tracked_alpaca_orders"]
            ]
            selected_sim_idx = st.selectbox(
                "Tracked order to simulate",
                range(len(simulator_options)),
                format_func=lambda idx: simulator_options[idx],
            )
            selected_sim_order = st.session_state["tracked_alpaca_orders"][selected_sim_idx]
            default_fill_price = float(live["last_p"]) if live.get("last_p") else 0.0
            simulated_fill_price = st.number_input(
                "Simulated fill price",
                min_value=0.0,
                value=round(default_fill_price, 2),
                step=0.01,
            )
            if st.button("Simulate Alpaca Paper Fill"):
                filled_record = simulated_alpaca_fill_order(selected_sim_order, fill_price=simulated_fill_price)
                simulated_position = simulated_position_from_filled_order(filled_record)
                refreshed_orders = [
                    filled_record if item.get("broker_order_id") == filled_record.get("broker_order_id") else item
                    for item in st.session_state["tracked_alpaca_orders"]
                ]
                broker_state_store.replace_all(refreshed_orders)
                st.session_state["tracked_alpaca_orders"] = refreshed_orders
                st.session_state["simulated_alpaca_positions"] = [simulated_position]
                fill_event = AuditEvent(
                    event_type="simulated_alpaca_paper_fill_recorded",
                    message="Local Alpaca paper fill simulation recorded without contacting Alpaca.",
                    payload={
                        "broker_order_id": filled_record.get("broker_order_id", ""),
                        "symbol": filled_record.get("symbol", ""),
                        "quantity": filled_record.get("filled_quantity", ""),
                        "average_fill_price": filled_record.get("average_fill_price", ""),
                        "broker_writes_submitted": 0,
                    },
                )
                st.session_state["session_audit_events"].append(fill_event)
                if persist_audit_log:
                    audit_store.append(fill_event)
                st.rerun()
    
            simulated_positions = st.session_state.get("simulated_alpaca_positions", [])
            if simulated_positions:
                st.markdown("#### Practice position record")
                simulated_lifecycle_rows = alpaca_position_lifecycle_records(
                    simulated_positions,
                    st.session_state["tracked_alpaca_orders"],
                )
                st.dataframe(
                    pd.DataFrame(alpaca_position_lifecycle_summary_records(simulated_lifecycle_rows)),
                    width="stretch",
                    hide_index=True,
                )
                st.dataframe(pd.DataFrame(simulated_lifecycle_rows), width="stretch", hide_index=True)
    
            st.markdown("#### Practice exit check")
            exit_readiness_rows = simulated_exit_preview_readiness_records(
                selected_sim_order,
                alpaca_adapter.config,
            )
            st.dataframe(pd.DataFrame(exit_readiness_rows), width="stretch", hide_index=True)
            if st.button("Save Practice Exit Check"):
                exit_sim_event = AuditEvent(
                    event_type="simulated_exit_readiness_recorded",
                    message="Local exit-path readiness simulation recorded without contacting Alpaca.",
                    payload={
                        "broker_order_id": selected_sim_order.get("broker_order_id", ""),
                        "symbol": selected_sim_order.get("symbol", ""),
                        "checks": exit_readiness_rows,
                        "broker_writes_submitted": 0,
                    },
                )
                st.session_state["session_audit_events"].append(exit_sim_event)
                if persist_audit_log:
                    audit_store.append(exit_sim_event)
                st.rerun()
            st.caption("These practice tools update local records only. They never submit, cancel, or exit Alpaca orders.")
    if exit_previews:
        st.markdown("#### Sell paper position")
        exit_options = [
            f"{preview.order.get('symbol', '')} {preview.order.get('side', '')} {preview.order.get('quantity', 0)} ({preview.preview_hash})"
            for preview in exit_previews
        ]
        selected_exit_idx = st.selectbox(
            "Alpaca paper position to exit",
            range(len(exit_options)),
            format_func=lambda idx: exit_options[idx],
        )
        alpaca_exit_preview = exit_previews[selected_exit_idx]
        if show_portfolio_evidence:
            st.dataframe(pd.DataFrame(alpaca_preview_records(alpaca_exit_preview)), width="stretch", hide_index=True)
        elif alpaca_exit_preview.valid:
            st.info(f"Ready: sell {alpaca_exit_preview.order.get('quantity', '')} {alpaca_exit_preview.order.get('symbol', '')} in Alpaca paper.")
    
        exit_symbol = str(alpaca_exit_preview.order.get("symbol", "")).strip().upper()
        selected_exit_position = next(
            (position for position in alpaca_positions if str(position.get("Symbol", "")).strip().upper() == exit_symbol),
            None,
        )
        selected_exit_intent = (
            TradeIntent(
                symbol=exit_symbol,
                side="sell",
                quantity=int(float(alpaca_exit_preview.order.get("quantity", 0))),
                order_type="market",
                time_in_force="day",
                rationale=f"Exit order generated from existing {alpaca_account_label} position.",
                source_signals=["alpaca_position_exit_preview"],
            )
            if selected_exit_position is not None
            else None
        )
        exit_decision = ExecutionDecision(
            mode="paper" if alpaca_adapter.config.paper else "live_with_approval",
            approved_for_execution=True,
            requires_manual_approval=not alpaca_adapter.config.paper,
            reason=f"{alpaca_account_label.title()} exit passed the exit checks.",
            risk_check=RiskCheckResult(approved=True, rejected_reasons=[], checks={"exit_position": True}),
        )
        exit_position_blockers = exit_position_reasons(alpaca_exit_preview, alpaca_positions)
        duplicate_exit_reasons = open_exit_order_reasons(alpaca_exit_preview, alpaca_orders)
        if alpaca_exit_preview.blocked_reasons:
            show_blockers("Exit blocked", alpaca_exit_preview.blocked_reasons)
        if exit_position_blockers:
            show_blockers("Exit blocked", exit_position_blockers)
        if duplicate_exit_reasons:
            show_blockers("Exit blocked", duplicate_exit_reasons)
        if alpaca_state_health.stale:
            st.warning("Refresh Alpaca positions and orders before sending this exit.")
        elif not exit_position_blockers and not duplicate_exit_reasons and alpaca_exit_preview.valid:
            st.success(f"{alpaca_account_label.title()} exit is ready to send.")
    
        exit_base_disabled = (
            selected_exit_intent is None
            or not alpaca_manual_order_mode
            or not alpaca_mode_matches_order_mode
            or not alpaca_status.connected
            or not alpaca_orders_enabled_for_mode
            or alpaca_state_health.stale
            or not alpaca_exit_preview.valid
            or bool(exit_position_blockers)
            or bool(duplicate_exit_reasons)
        )
        alpaca_exit_submit_disabled = (
            exit_base_disabled
            or not alpaca_status.can_submit_orders
        )
        exit_button_label = "Send Paper Exit to Alpaca" if alpaca_adapter.config.paper else "Send Live Exit to Alpaca"
        if st.button(exit_button_label, disabled=alpaca_exit_submit_disabled):
            try:
                alpaca_exit_order = alpaca_adapter.submit_order(
                    selected_exit_intent,
                    exit_decision,
                    expected_preview_hash=alpaca_exit_preview.preview_hash,
                )
                exit_event = AuditEvent(
                    event_type=f"alpaca_{alpaca_order_noun}_exit_submitted",
                    message=f"Alpaca {alpaca_order_noun} exit submitted through the gated adapter.",
                    payload={
                        "symbol": exit_symbol,
                        "side": "sell",
                        "quantity": selected_exit_intent.quantity,
                        "preview_hash": alpaca_exit_preview.preview_hash,
                        "broker_order_id": str(getattr(alpaca_exit_order, "id", "")),
                    },
                )
                broker_order_id = str(getattr(alpaca_exit_order, "id", ""))
                if broker_order_id:
                    tracked_record = {
                        "broker_order_id": broker_order_id,
                        "preview_hash": alpaca_exit_preview.preview_hash,
                        "symbol": exit_symbol,
                        "side": "sell",
                        "quantity": selected_exit_intent.quantity,
                        "status": str(getattr(alpaca_exit_order, "status", "")),
                        "submitted_at": str(getattr(alpaca_exit_order, "submitted_at", "")),
                    }
                    st.session_state["tracked_alpaca_orders"].append(tracked_record)
                    broker_state_store.upsert(tracked_record)
                st.session_state["session_audit_events"].append(exit_event)
                if persist_audit_log:
                    audit_store.append(exit_event)
                st.success(f"{alpaca_account_label.title()} exit sent to Alpaca.")
            except Exception as exc:
                exit_event = AuditEvent(
                    event_type=f"alpaca_{alpaca_order_noun}_exit_blocked",
                    message=str(exc),
                    payload={"symbol": exit_symbol, "preview_hash": alpaca_exit_preview.preview_hash},
                )
                st.session_state["session_audit_events"].append(exit_event)
                if persist_audit_log:
                    audit_store.append(exit_event)
                st.error(f"{alpaca_account_label.title()} exit blocked: {exc}")
        st.caption(f"This contacts {alpaca_account_label} only.")
    
    if managed_cancelable_alpaca_orders:
        st.markdown(f"#### Cancel {alpaca_order_noun} order")
        cancel_options = [
            f"{row['Symbol']} {row['Side']} {row['Quantity']} {row['Status']} ({row['Order ID']})"
            for row in managed_cancelable_alpaca_orders
        ]
        selected_cancel_idx = st.selectbox(
            "Alpaca paper order to cancel",
            range(len(cancel_options)),
            format_func=lambda idx: cancel_options[idx],
        )
        selected_cancel_order = managed_cancelable_alpaca_orders[selected_cancel_idx]
        alpaca_cancel_preview = build_alpaca_cancel_preview(selected_cancel_order, alpaca_adapter.config)
        if show_portfolio_evidence:
            st.dataframe(pd.DataFrame(alpaca_cancel_preview_records(alpaca_cancel_preview)), width="stretch", hide_index=True)
        if alpaca_cancel_preview.blocked_reasons:
            show_blockers("Cancel blocked", alpaca_cancel_preview.blocked_reasons)
        if alpaca_state_health.stale:
            st.warning("Refresh Alpaca positions and orders before canceling.")
    
        selected_cancel_order_id = selected_cancel_order.get("Alpaca Order ID") or selected_cancel_order.get("Broker Order ID", "")
        if alpaca_cancel_preview.valid and not alpaca_state_health.stale:
            st.success(f"{alpaca_account_label.title()} cancel is ready to send.")
    
        cancel_base_disabled = (
            not alpaca_manual_order_mode
            or not alpaca_mode_matches_order_mode
            or not alpaca_status.connected
            or not alpaca_orders_enabled_for_mode
            or alpaca_state_health.stale
            or not alpaca_cancel_preview.valid
        )
        alpaca_cancel_submit_disabled = (
            cancel_base_disabled
            or not alpaca_status.can_submit_orders
        )
        cancel_button_label = "Send Paper Cancel to Alpaca" if alpaca_adapter.config.paper else "Send Live Cancel to Alpaca"
        if st.button(cancel_button_label, disabled=alpaca_cancel_submit_disabled):
            try:
                cancel_result = alpaca_adapter.cancel_order(
                    selected_cancel_order_id,
                    expected_cancel_hash=alpaca_cancel_preview.preview_hash,
                )
                cancel_event = AuditEvent(
                    event_type=f"alpaca_{alpaca_order_noun}_cancel_submitted",
                    message=f"Alpaca {alpaca_order_noun} order cancel submitted through the gated adapter.",
                    payload={
                        "broker_order_id": selected_cancel_order_id,
                        "cancel_preview_hash": alpaca_cancel_preview.preview_hash,
                        "cancel_status": str(getattr(cancel_result, "status", "cancel_requested")),
                        **alpaca_cancel_preview.cancel,
                    },
                )
                broker_state_store.upsert({
                    "broker_order_id": selected_cancel_order_id,
                    "preview_hash": "",
                    "symbol": selected_cancel_order.get("Symbol", ""),
                    "side": selected_cancel_order.get("Side", ""),
                    "quantity": selected_cancel_order.get("Quantity", ""),
                    "status": "cancel_requested",
                    "submitted_at": selected_cancel_order.get("Submitted", ""),
                })
                st.session_state["tracked_alpaca_orders"] = broker_state_store.read()
                st.session_state["session_audit_events"].append(cancel_event)
                if persist_audit_log:
                    audit_store.append(cancel_event)
                st.success(f"{alpaca_account_label.title()} cancel sent to Alpaca.")
            except Exception as exc:
                cancel_event = AuditEvent(
                    event_type=f"alpaca_{alpaca_order_noun}_cancel_blocked",
                    message=str(exc),
                    payload={
                        "broker_order_id": selected_cancel_order_id,
                        "cancel_preview_hash": alpaca_cancel_preview.preview_hash,
                    },
                )
                st.session_state["session_audit_events"].append(cancel_event)
                if persist_audit_log:
                    audit_store.append(cancel_event)
                st.error(f"{alpaca_account_label.title()} cancel blocked: {exc}")
        st.caption("This contacts Alpaca paper only.")
    
    if intent is not None and execution_mode != "paper":
        st.caption("Switch execution mode to Paper trading to enable paper order submission.")
    elif intent is not None and not preflight_check.ready:
        st.caption("Paper order sending is disabled until the risk checks pass.")
    elif can_submit:
        st.caption("Paper orders use the strategy reference price and never contact a live broker.")
    if intent is not None:
        st.caption("Alpaca paper orders require paper mode, a real ticker, connected Alpaca paper credentials, and the paper account switch.")
    
    with st.expander("Automation check", expanded=False):
        automation_decision = paper_automation_dry_run(
            intent=intent,
            risk_check=risk_check,
            preflight=preflight_check,
            broker_health=alpaca_state_health,
            duplicate_reasons=duplicate_alpaca_reasons + open_order_reasons,
            idempotency_blocked=duplicate_preview_submitted,
        )
        if automation_decision.ready:
            st.success("The app found a paper action that is ready.")
        else:
            st.info("No automation action is ready right now.")
            show_blockers("Reason", automation_decision.reasons)
        if show_portfolio_evidence:
            st.dataframe(pd.DataFrame(automation_decision_records(automation_decision)), width="stretch", hide_index=True)
        exit_preview_rows = exit_preview_records(exit_previews)
        exit_blockers_by_hash = {
            preview.preview_hash: (
                preview.blocked_reasons
                + exit_position_reasons(preview, alpaca_positions)
                + open_exit_order_reasons(preview, alpaca_orders)
            )
            for preview in exit_previews
        }
        automation_candidates = paper_automation_candidate_records(
            entry_decision=automation_decision,
            entry_symbol=intent.symbol_clean if intent else "",
            entry_side=intent.side if intent else "",
            entry_quantity=intent.quantity if intent else "",
            exit_previews=exit_preview_rows,
            cancelable_orders=cancelable_alpaca_orders,
            exit_blockers=exit_blockers_by_hash,
        )
        if show_portfolio_evidence:
            st.markdown("#### Automation checks *")
        readiness_rows = automation_readiness_records(
            broker_connected=alpaca_status.connected,
            broker_state_stale=alpaca_state_health.stale,
            manual_order_gate_enabled=enable_alpaca_paper_orders,
            kill_switch_enabled=effective_kill_switch,
            candidates=automation_candidates,
        )
        if show_portfolio_evidence:
            st.dataframe(
                pd.DataFrame(readiness_rows),
                width="stretch",
                hide_index=True,
            )
            st.markdown("#### Paper actions ready")
            st.dataframe(pd.DataFrame(automation_candidates), width="stretch", hide_index=True)
        automation_halt_rows = risk_halt_records(
            monitoring_result=monitoring_result,
            broker_connected=alpaca_status.connected,
            broker_state_stale=alpaca_state_health.stale,
            automation_ready_rows=readiness_rows,
        )
        recent_automation_snapshots = automation_store.read_recent(limit=100)
        market_row_count = len(market_data) if market_data is not None else len(prices)
        latest_market_label = str(market_data.index[-1]) if market_data is not None else str(labels[-1] if labels else "")
        market_freshness_rows = market_data_freshness_records(
            data_source=data_source,
            source_caption=source_caption,
            row_count=market_row_count,
            latest_label=latest_market_label,
            minimum_rows=max(entry_w, exit_w, ma_w, 30),
        )
        account_health_rows = paper_account_health_records(
            paper_cash=paper_broker.cash,
            paper_equity=paper_equity,
            starting_cash=st.session_state["paper_starting_cash"],
            local_open_positions=len(paper_broker.positions),
            tracked_alpaca_orders=st.session_state["tracked_alpaca_orders"],
            monitoring_result=monitoring_result,
            limits=risk_limits,
        )
        restart_rows = restart_recovery_records(
            audit_log_path=audit_log_path,
            broker_state_path=broker_state_path,
            automation_dry_run_path=automation_dry_run_path,
            run_manifest_path=run_manifest_path,
            audit_records_loaded=len(audit_store.read_recent(limit=500)) if persist_audit_log else len(st.session_state["session_audit_events"]),
            tracked_orders_loaded=len(st.session_state["tracked_alpaca_orders"]),
            automation_snapshots_loaded=len(recent_automation_snapshots),
        )
        scheduler_rows = scheduler_preview_records(
            interval_minutes=automation_preview_interval,
            market_open=bool(market_session_advisory().get("Open", False)),
            kill_switch_enabled=effective_kill_switch,
            ready_candidate_count=sum(bool(row.get("Ready")) for row in automation_candidates),
            halt_count=sum(bool(row.get("Active")) for row in automation_halt_rows),
        )
        paper_automation_gate_rows = paper_automation_gate_records(
            broker_connected=alpaca_status.connected,
            broker_state_stale=alpaca_state_health.stale,
            market_data_rows=market_freshness_rows,
            account_health_rows=account_health_rows,
            restart_rows=restart_rows,
            readiness_rows=readiness_rows,
            halt_rows=automation_halt_rows,
            dry_run_snapshots_loaded=len(recent_automation_snapshots),
        )
        gate_summary = {row["Check"]: row for row in paper_automation_gate_rows}
        final_gate = gate_summary.get("Paper automation can run", {})
        if final_gate.get("Passed"):
            st.success("No hard automation blocks found.")
        elif final_gate.get("Detail"):
            st.warning(str(final_gate.get("Detail")))
        if show_portfolio_evidence:
            st.markdown("#### Market data *")
            st.dataframe(pd.DataFrame(market_freshness_rows), width="stretch", hide_index=True)
            st.markdown("#### Strategy state *")
            st.dataframe(
                pd.DataFrame(
                    strategy_state_snapshot_records(
                        config=current_strategy_config,
                        live=live,
                        intent=intent,
                        risk_check=risk_check,
                        preflight=preflight_check,
                    )
                ),
                width="stretch",
                hide_index=True,
            )
            st.markdown("#### Paper account *")
            st.dataframe(pd.DataFrame(account_health_rows), width="stretch", hide_index=True)
            st.markdown("#### Restart check *")
            st.dataframe(pd.DataFrame(restart_rows), width="stretch", hide_index=True)
            st.markdown("#### Timer check *")
            st.dataframe(pd.DataFrame(scheduler_rows), width="stretch", hide_index=True)
            st.markdown("#### Safety blocks *")
            st.dataframe(pd.DataFrame(paper_automation_gate_rows), width="stretch", hide_index=True)
            st.markdown("#### Automation decision *")
            st.dataframe(
                pd.DataFrame(
                    automation_supervisor_dry_run_records(
                        candidates=automation_candidates,
                        readiness_rows=readiness_rows,
                        halt_rows=automation_halt_rows,
                    )
                ),
                width="stretch",
                hide_index=True,
            )
        if st.button("Record Automation Check"):
            snapshot = build_automation_snapshot(
                session_id=st.session_state["paper_session_id"],
                candidates=automation_candidates,
                readiness=readiness_rows,
            )
            snapshot_record = automation_store.append(snapshot)
            automation_event = AuditEvent(
                event_type="paper_automation_dry_run_recorded",
                message="Automation check saved locally.",
                payload={
                    "session_id": snapshot.session_id,
                    "candidate_count": snapshot_record["candidate_count"],
                    "ready_candidate_count": snapshot_record["ready_candidate_count"],
                    "broker_write_candidate_count": snapshot_record["broker_write_candidate_count"],
                },
            )
            st.session_state["session_audit_events"].append(automation_event)
            if persist_audit_log:
                audit_store.append(automation_event)
            st.rerun()
        if show_portfolio_evidence:
            st.markdown("#### Saved automation checks *")
            st.dataframe(pd.DataFrame(automation_evidence_records(recent_automation_snapshots)), width="stretch", hide_index=True)
        st.caption("This check records what automation sees. It does not submit, exit, or cancel broker orders.")
    
    if show_portfolio_evidence:
        st.markdown("#### Practice decision log *")
        shadow_disabled = intent is None or execution_mode != "shadow"
        if st.button("Save Practice Decision", disabled=shadow_disabled):
            shadow_decision = record_shadow_decision(
                intent=intent,
                risk_check=risk_check,
                execution_decision=execution_decision,
                preflight=preflight_check,
            )
            st.session_state["shadow_decisions"].append(shadow_decision)
            shadow_event = AuditEvent(
                event_type="shadow_decision_recorded",
                message="Shadow decision recorded without submitting an order.",
                payload=shadow_decision,
            )
            st.session_state["session_audit_events"].append(shadow_event)
            if persist_audit_log:
                audit_store.append(shadow_event)
            st.rerun()
        if execution_mode != "shadow":
            st.caption("Switch mode to Practice mode to save practice decisions.")
        elif intent is None:
            st.caption("No trade is available to save as a practice decision.")
        shadow_rows = shadow_records(st.session_state["shadow_decisions"])
        if shadow_rows:
            st.dataframe(pd.DataFrame(shadow_rows), width="stretch", hide_index=True)
        else:
            st.caption("No practice decisions saved this session.")
    
    with st.expander("Local simulator positions and orders", expanded=False):
        exec_tabs = st.tabs(["Positions", "Orders"])
        with exec_tabs[0]:
            if position_records:
                st.dataframe(pd.DataFrame(position_records), width="stretch", hide_index=True)
            else:
                st.caption("No open paper positions.")
        with exec_tabs[1]:
            if order_records:
                st.dataframe(pd.DataFrame(order_records), width="stretch", hide_index=True)
            else:
                st.caption("No paper orders submitted.")
    current_risk_halt_rows = risk_halt_records(
        monitoring_result=monitoring_result,
        broker_connected=alpaca_status.connected,
        broker_state_stale=alpaca_state_health.stale,
        automation_ready_rows=readiness_rows,
    )
    recent_automation_records = automation_store.read_recent(limit=100)
    if show_portfolio_evidence:
        page_section("Saved records *", "Detailed session records, local simulator results, Alpaca paper history, and export tools.")
        sub_section("Records overview *")
        st.dataframe(
            pd.DataFrame(
                saved_records_overview_records(
                    audit_records=current_evidence_records,
                    tracked_orders=st.session_state["tracked_alpaca_orders"],
                    automation_snapshots=recent_automation_records,
                )
            ),
            width="stretch",
            hide_index=True,
        )
        sub_section("This session *")
        st.dataframe(pd.DataFrame(session_summary_records(session_snapshot)), width="stretch", hide_index=True)
        timeline_rows = session_timeline_records(session_audit_records)
        if timeline_rows:
            st.dataframe(pd.DataFrame(timeline_rows), width="stretch", hide_index=True)
        else:
            st.caption("No session events recorded yet.")
    
        sub_section("Local simulator results *")
        st.dataframe(pd.DataFrame(paper_performance_records(session_snapshot)), width="stretch", hide_index=True)
        if st.button("Save Paper Performance Review", key="save_records_paper_review"):
            performance_event = AuditEvent(
                event_type="paper_performance_reviewed",
                message="Local simulator performance was reviewed by the user.",
                payload={"session_id": st.session_state["paper_session_id"]},
            )
            st.session_state["session_audit_events"].append(performance_event)
            if persist_audit_log:
                audit_store.append(performance_event)
            st.rerun()
        st.caption("This uses the sidebar account size and local simulator records. Alpaca account balance is shown above under Alpaca Account.")
    
        sub_section("Alpaca paper order history *")
        st.dataframe(pd.DataFrame(alpaca_paper_activity_records(session_snapshot)), width="stretch", hide_index=True)
        st.caption("This is saved Alpaca paper order history. It does not affect simulator cash or simulator equity.")
    
        sub_section("Risk details *")
        st.dataframe(
            pd.DataFrame(
                daily_risk_records(
                    local_order_records=order_records,
                    tracked_alpaca_orders=st.session_state["tracked_alpaca_orders"],
                    account_equity=paper_order_risk_equity,
                    session_pnl=paper_order_session_pnl,
                    portfolio_exposure=paper_order_portfolio_notional,
                    limits=risk_limits,
                )
            ),
            width="stretch",
            hide_index=True,
        )
        sub_section("Current blocks *")
        st.dataframe(pd.DataFrame(current_risk_halt_rows), width="stretch", hide_index=True)
    
    readiness_event_types = {record.get("event_type", "") for record in current_evidence_records}
    current_pre_live_readiness_rows = pre_live_readiness_report(
        paper_order_submitted="alpaca_paper_order_submitted" in readiness_event_types,
        paper_cancel_submitted="alpaca_paper_cancel_submitted" in readiness_event_types,
        paper_exit_tested="alpaca_paper_exit_submitted" in readiness_event_types,
        paper_fill_reconciled=any(
            str(order.get("lifecycle_status", "")) == "filled_at_alpaca"
            and not bool(order.get("simulated"))
            for order in st.session_state["tracked_alpaca_orders"]
        ),
        automation_dry_run_recorded=(
            "paper_automation_dry_run_recorded" in readiness_event_types
            or bool(recent_automation_records)
        ),
        performance_reviewed="paper_performance_reviewed" in readiness_event_types,
        emergency_disable_tested="session_disabled" in readiness_event_types,
        live_mode_blocked=not (not alpaca_adapter.config.paper and alpaca_status.can_submit_orders),
    )
    current_approval_ledger_rows = approval_ledger_records(current_evidence_records)
    
    if show_portfolio_evidence:
        sub_section("Activity log *")
        st.dataframe(
            pd.DataFrame(events_to_records(st.session_state["session_audit_events"])),
            width="stretch",
            hide_index=True,
        )
        with st.expander("Saved activity log *", expanded=False):
            st.caption(f"Path: {audit_store.path}")
            durable_records = audit_store.read_recent(limit=50) if persist_audit_log else []
            if durable_records:
                st.dataframe(pd.DataFrame(durable_records), width="stretch", hide_index=True)
            else:
                st.caption("No saved activity records found, or saving is turned off.")
        with st.expander("Detailed records *", expanded=False):
            st.dataframe(
                pd.DataFrame(evidence_dashboard_records(current_evidence_records, st.session_state["tracked_alpaca_orders"])),
                width="stretch",
                hide_index=True,
            )
            st.markdown("#### Review history *")
            st.dataframe(pd.DataFrame(approval_ledger_summary_records(current_approval_ledger_rows)), width="stretch", hide_index=True)
            if current_approval_ledger_rows:
                st.dataframe(pd.DataFrame(current_approval_ledger_rows), width="stretch", hide_index=True)
            if st.button("Export Records"):
                evidence_package = build_evidence_package(
                    session_id=st.session_state["paper_session_id"],
                    manifest=current_manifest_record,
                    audit_records=current_evidence_records,
                    approval_ledger=current_approval_ledger_rows,
                    tracked_orders=st.session_state["tracked_alpaca_orders"],
                    automation_snapshots=recent_automation_records,
                    readiness_rows=current_pre_live_readiness_rows,
                    risk_halts=current_risk_halt_rows,
                )
                output_path = write_evidence_package(evidence_package, evidence_export_path)
                export_event = AuditEvent(
                    event_type="evidence_package_exported",
                    message="Evidence package exported locally.",
                    payload={"path": str(output_path), "session_id": st.session_state["paper_session_id"]},
                )
                st.session_state["session_audit_events"].append(export_event)
                if persist_audit_log:
                    audit_store.append(export_event)
                st.dataframe(pd.DataFrame(evidence_package_records(evidence_package)), width="stretch", hide_index=True)
                st.success(f"Records exported to {output_path}")
    
        with st.expander("Live trading checklist *", expanded=False):
            st.dataframe(pd.DataFrame(current_pre_live_readiness_rows), width="stretch", hide_index=True)
            st.caption("This checklist is based on saved records. Some items stay blocked until the matching action is recorded.")




