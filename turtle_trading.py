import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from agentloop_trader.audit import build_audit_events, events_to_records
from agentloop_trader.audit_store import JsonlAuditStore
from agentloop_trader.agents import build_trade_proposal, proposal_records
from agentloop_trader.automation import (
    AutomationDryRunStore,
    AutomationRuntimeState,
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
    strategy_settings_match,
    strategy_settings_match_reason,
)
from agentloop_trader.backtest import (
    simulate_trend_pullback_strategy,
    simulate_turtle_strategy,
    strategy_comparison_records,
)
from agentloop_trader.brokers import (
    AlpacaBrokerAdapterStub,
    PaperBrokerAdapter,
    alpaca_config_validation_records,
    alpaca_cancel_preview_records,
    alpaca_preview_records,
    build_alpaca_cancel_preview,
    build_alpaca_order_preview,
    broker_status_records,
)
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
from agentloop_trader.execution import PaperBroker
from agentloop_trader.models import AuditEvent, ExecutionDecision, RiskCheckResult, RiskLimits, StrategyConfig, TradeIntent
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
    candidate_records,
    evaluate_parameter_candidates,
    recommend_candidate,
    recommendation_summary,
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
from agentloop_trader.session_journal import (
    PaperSessionSnapshot,
    alpaca_paper_activity_records,
    new_session_id,
    paper_performance_records,
    session_summary_records,
    session_timeline_records,
)
from agentloop_trader.shadow import record_shadow_decision, shadow_records
from agentloop_trader.ui_summary import (
    agent_loop_stage_records,
    agent_decision_summary,
    buy_requirement_records,
    compact_status_records,
    no_buy_reason,
    operator_state_record,
    optional_quality_input_records,
    optional_sell_quality_records,
    sell_requirement_records,
    setup_scorecard_records,
    strategy_context_records,
)

try:
    import yfinance as yf
except ImportError:
    yf = None


st.set_page_config(page_title="AgentLoop Trader", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background-color: rgba(128,128,128,0.08);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }
    .metric-label { font-size: 12px; color: rgba(128,128,128,0.85); margin-bottom: 4px; }
    .metric-value { font-size: 22px; font-weight: 500; color: inherit; }
    .metric-sub { font-size: 11px; color: rgba(128,128,128,0.7); margin-top: 2px; }
    .pos { color: #3B6D11; }
    .neg { color: #A32D2D; }
    .rule-box {
        border-left: 3px solid rgba(128,128,128,0.25);
        padding: 8px 14px;
        margin-bottom: 8px;
        font-size: 14px;
        line-height: 1.6;
    }
    .signal-long {
        background: #EAF3DE; color: #3B6D11;
        padding: 8px 16px; border-radius: 8px;
        font-weight: 500; display: inline-block;
    }
    .signal-exit {
        background: #FCEBEB; color: #A32D2D;
        padding: 8px 16px; border-radius: 8px;
        font-weight: 500; display: inline-block;
    }
    .signal-flat {
        background: rgba(128,128,128,0.12); color: rgba(128,128,128,0.9);
        padding: 8px 16px; border-radius: 8px;
        font-weight: 500; display: inline-block;
    }
    .ui-section-caption {
        color: rgba(128,128,128,0.85);
        font-size: 0.95rem;
        margin-top: -0.6rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker, period, interval):
    if yf is None:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")
    clean = ticker.strip().upper()
    if not clean:
        raise ValueError("Enter a ticker symbol.")

    fetch_interval = "1h" if interval == "4h" else interval
    kwargs = dict(
        tickers=clean,
        period=period,
        interval=fetch_interval,
        auto_adjust=True,
        progress=False,
        threads=False,
        prepost=False,
    )
    try:
        data = yf.download(**kwargs, multi_level_index=False)
    except TypeError:
        data = yf.download(**kwargs)

    if data is None or data.empty:
        raise ValueError(f"No price data returned for {clean}.")
    if isinstance(data.columns, pd.MultiIndex):
        levels = [list(map(str, data.columns.get_level_values(i))) for i in range(data.columns.nlevels)]
        if clean in levels[-1]:
            data = data.xs(clean, axis=1, level=-1)
        elif clean in levels[0]:
            data = data.xs(clean, axis=1, level=0)
        else:
            data.columns = data.columns.get_level_values(0)

    required = ["Close", "High", "Low"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    available_columns = required + (["Volume"] if "Volume" in data.columns else [])
    data = data[available_columns].dropna(subset=required)
    if data.empty:
        raise ValueError(f"No usable rows for {clean}.")

    if interval == "4h":
        aggregations = {
            "Close": "last",
            "High": "max",
            "Low": "min",
        }
        if "Volume" in data.columns:
            aggregations["Volume"] = "sum"
        data = data.resample("4h").agg(aggregations).dropna(subset=required)

    data.attrs["symbol"] = clean
    return data


def metric_card(col, label, value, sub, color_class=""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def page_section(title: str, caption: str | None = None) -> None:
    st.markdown(f"## {title}")
    if caption:
        st.markdown(f"<div class='ui-section-caption'>{caption}</div>", unsafe_allow_html=True)


def sub_section(title: str, caption: str | None = None) -> None:
    st.markdown(f"### {title}")
    if caption:
        st.markdown(f"<div class='ui-section-caption'>{caption}</div>", unsafe_allow_html=True)


def plain_yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def count_waiting_alpaca_orders(order_rows: list[dict]) -> int:
    waiting_statuses = {"accepted", "new", "pending_new", "partially_filled"}
    return sum(str(row.get("Status", "")).strip().lower() in waiting_statuses for row in order_rows)


def saved_buy_settings_for_symbol(symbol: str, tracked_orders: list[dict]) -> dict | None:
    clean_symbol = str(symbol).strip().upper()
    matching_buys = [
        order
        for order in tracked_orders
        if str(order.get("symbol", "")).strip().upper() == clean_symbol
        and str(order.get("side", "")).strip().lower() == "buy"
        and order.get("strategy_settings")
    ]
    return dict(matching_buys[-1].get("strategy_settings", {})) if matching_buys else None


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
    latest = matching_buys[-1]
    return dict(latest.get("exit_settings") or latest.get("strategy_settings") or {})


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


def show_blockers(title: str, blockers: list[str]) -> None:
    cleaned = [reason for reason in dict.fromkeys(blockers) if reason]
    if cleaned:
        st.warning(f"{title}: " + " ".join(cleaned))


def build_chart(prices, smas, atrs, entry_w, exit_w, ma_w, labels, trade_log, selected_trade=None):
    x = labels
    dh = [float(np.max(prices[i - entry_w:i])) if i >= entry_w else None for i in range(len(prices))]
    dl = [float(np.min(prices[i - exit_w:i])) if i >= exit_w else None for i in range(len(prices))]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=prices.tolist(), name="Price", mode="lines",
        line=dict(color="#4C9BE8", width=1.5),
        hovertemplate="%{x}<br>Price: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=smas, name=f"{ma_w}d SMA", mode="lines",
        line=dict(color="#F0A830", width=1, dash="dot"),
        hovertemplate="%{x}<br>SMA: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=dh, name=f"{entry_w}d High", mode="lines",
        line=dict(color="#5DBF8A", width=1, dash="dash"),
        hovertemplate="%{x}<br>" + str(entry_w) + "d High: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=dl, name=f"{exit_w}d Low", mode="lines",
        line=dict(color="#E8645A", width=1, dash="dash"),
        hovertemplate="%{x}<br>" + str(exit_w) + "d Low: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=atrs, name="ATR (14d)", mode="lines",
        line=dict(color="#B07FE8", width=1, dash="dot"),
        yaxis="y2",
        hovertemplate="%{x}<br>ATR: $%{y:.2f}<extra></extra>",
    ))

    if trade_log:
        fig.add_trace(go.Scatter(
            x=[t["entry_date"] for t in trade_log],
            y=[t["entry"] for t in trade_log],
            name="Entry", mode="markers",
            marker=dict(symbol="triangle-up", size=10, color="#5DBF8A", opacity=0.55),
            customdata=[t["trade"] for t in trade_log],
            hovertemplate="Entry #%{customdata}<br>%{x}: $%{y:.2f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[t["exit_date"] for t in trade_log],
            y=[t["exit"] for t in trade_log],
            name="Exit", mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="#E8645A", opacity=0.55),
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
            marker=dict(symbol="triangle-up", size=18, color="#5DBF8A"),
            text=[f" BUY #{t['trade']}"], textposition="middle right",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[t["exit_date"]], y=[t["exit"]], mode="markers+text",
            marker=dict(symbol="triangle-down", size=18, color="#E8645A"),
            text=[f" SELL #{t['trade']}"], textposition="middle right",
            showlegend=False,
        ))

    fig.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Time", showgrid=False, type="category", nticks=10),
        yaxis=dict(tickprefix="$", gridcolor="rgba(128,128,128,0.15)"),
        yaxis2=dict(tickprefix="$", overlaying="y", side="right", showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    return fig


st.sidebar.title("Trading Simulator")
st.sidebar.markdown("### Workspace")
workspace_mode = st.sidebar.radio(
    "What do you want to do?",
    ["Daily Operator", "Portfolio Evidence"],
    index=0,
    help="Daily Operator keeps the screen simple. Portfolio Evidence shows detailed records.",
)
show_portfolio_evidence = workspace_mode == "Portfolio Evidence"

st.sidebar.markdown("### Automation")
automation_level_options = {
    "Manual - I click paper orders": "Manual review only",
    "Auto exits - app sells paper positions": "Auto exits only",
    "Auto entries and exits - app trades paper": "Auto entries and exits",
}
automation_level_label = st.sidebar.selectbox("Automation level", list(automation_level_options.keys()), index=0)
automation_level = automation_level_options[automation_level_label]
st.sidebar.caption(
    "Auto exits manages open positions. Auto entries and exits is experimental until multi-ticker research is added."
)
if automation_level == "Auto entries and exits":
    st.sidebar.caption("Auto entries can buy the ticker currently loaded in the research screen.")

st.sidebar.markdown("### 1. Ticker and Price Data")
data_source = st.sidebar.radio("Prices to use", ["Synthetic", "Stock via yfinance"], horizontal=True)
market_data = None
source_caption = "synthetic price data"
ticker = "SYNTH"
interval = "1d"
period = "synthetic"

if data_source == "Stock via yfinance":
    ticker = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()
    interval = st.sidebar.selectbox("Interval", ["1d", "4h", "1h", "30m", "15m", "5m", "1m"], index=0)
    if interval == "1d":
        period_options, period_index = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"], 5
    elif interval in ("4h", "1h"):
        period_options, period_index = ["1mo", "3mo", "6mo", "1y"], 3
    elif interval in ("30m", "15m"):
        period_options, period_index = ["1mo", "3mo", "6mo"], 2
    else:
        period_options, period_index = ["1d", "5d", "1mo"], 2
    period = st.sidebar.selectbox("History period", period_options, index=period_index)
    if st.sidebar.button("Refresh stock data", type="primary"):
        fetch_stock_data.clear()
    try:
        with st.spinner(f"Fetching {ticker}..."):
            market_data = fetch_stock_data(ticker, period, interval)
        source_caption = f"{ticker} via yfinance ({period}, {interval}); latest bar {market_data.index[-1]}"
        st.sidebar.caption(f"Loaded {len(market_data):,} bars. Yahoo intraday data may be delayed.")
    except Exception as exc:
        st.error(f"Could not load yfinance data: {exc}")
        st.stop()

st.sidebar.markdown("### 2. Strategy and Backtest")
account = st.sidebar.number_input(
    "Simulator account size ($)",
    min_value=1000,
    max_value=10_000_000,
    value=50000,
    step=1000,
    help="Used by the local simulator and strategy sizing. Alpaca account balance is read separately from Alpaca.",
)
strategy_options = {
    "Breakout continuation": "breakout",
    "Trend pullback continuation": "pullback",
}
strategy_label = st.sidebar.selectbox(
    "Strategy type",
    list(strategy_options.keys()),
    index=0,
    help="Choose the rule set that creates trade ideas. Breakout buys strength through prior highs. Pullback buys a dip inside an existing uptrend.",
)
strategy_type = strategy_options[strategy_label]
entry_w = st.sidebar.slider(
    "Buy breakout length (bars)",
    10,
    55,
    20,
    step=5,
    help="Breakout strategy only. The app looks back this many bars and buys when price breaks above that prior high. Higher means fewer, stronger breakouts.",
)
exit_w = st.sidebar.slider(
    "Sell exit length (bars)",
    5,
    30,
    10,
    step=5,
    help="Breakout strategy exit. The app sells when price falls to the lowest price from this many bars. Higher gives trades more room; lower exits faster.",
)
atr_mult = st.sidebar.slider(
    "Stop distance (ATR multiplier)",
    1.0,
    4.0,
    2.0,
    step=0.5,
    help="Sets stop distance using volatility. Higher means a wider stop and smaller share size. Lower means a tighter stop and larger share size.",
)
risk_pct = st.sidebar.slider(
    "Strategy risk per trade (%)",
    0.5,
    3.0,
    1.0,
    step=0.5,
    help="How much of the simulator account the strategy is allowed to risk on one trade before separate risk limits are applied.",
)
ma_w = st.sidebar.slider(
    "Trend filter length (bars)",
    50,
    300,
    200,
    step=50,
    help="The moving average used to decide whether the ticker is in an uptrend. Higher is slower and stricter; lower reacts faster.",
)
pullback_w = st.sidebar.slider(
    "Pullback average length (bars)",
    10,
    50,
    20,
    step=5,
    help="Pullback strategy only. The app uses this average as the area where a dip may be buyable inside an uptrend.",
)
momentum_w = st.sidebar.slider(
    "Momentum turn length (bars)",
    3,
    20,
    10,
    step=1,
    help="Pullback strategy only. The app looks for price to turn back up above this shorter average before buying.",
)

st.sidebar.markdown("### 3. Paper Trading Mode")
mode_options = {
    "Backtest only": "backtest_only",
    "Paper trading": "paper",
    "Practice mode": "shadow",
    "Live with approval": "live_with_approval",
    "Automated live": "automated_live",
}
mode_label = st.sidebar.selectbox("How orders are handled", list(mode_options.keys()), index=0)
execution_mode = mode_options[mode_label]
allowed_symbols_text = st.sidebar.text_input("Allowed symbols", value=ticker)

st.sidebar.markdown("### 4. Risk Limits")
max_risk_limit = st.sidebar.slider(
    "Max risk per trade (%)",
    0.25,
    5.0,
    1.0,
    step=0.25,
    help="Hard cap on dollars at risk for one trade. If the stop loss would risk more than this, the app reduces size or blocks the trade.",
)
max_notional_limit = st.sidebar.slider(
    "Max position notional (%)",
    5.0,
    100.0,
    25.0,
    step=5.0,
    help="Hard cap on position size as a percent of account value. Example: 25% on a $100,000 account allows up to $25,000 in one order.",
)
max_portfolio_exposure = st.sidebar.slider(
    "Max portfolio exposure (%)",
    10.0,
    100.0,
    75.0,
    step=5.0,
    help="Hard cap on total open exposure across all tracked positions and orders. This keeps the app from putting too much of the account to work at once.",
)
max_symbol_concentration = st.sidebar.slider(
    "Max symbol concentration (%)",
    5.0,
    100.0,
    35.0,
    step=5.0,
    help="Hard cap on exposure to one ticker. This prevents one symbol from becoming too large relative to the account.",
)
max_session_loss = st.sidebar.slider(
    "Max session loss (%)",
    0.5,
    10.0,
    2.0,
    step=0.5,
    help="Hard cap on loss for the current app session. If session loss reaches this level, new orders are blocked.",
)
max_open_positions = st.sidebar.slider(
    "Max open positions",
    1,
    20,
    5,
    step=1,
    help="Maximum number of positions the app can have open or tracked at the same time.",
)
kill_switch = st.sidebar.checkbox(
    "Kill Switch",
    value=False,
    help="Immediate hard stop. When this is on, the app blocks new paper orders and automation actions.",
)
allowed_symbols = tuple(s.strip().upper() for s in allowed_symbols_text.split(",") if s.strip())

st.sidebar.markdown("### 5. Paper Account")
reset_paper_broker = st.sidebar.button("Reset paper account")
emergency_disable_session = st.sidebar.button("Stop trading now")
enable_alpaca_paper_orders = st.sidebar.checkbox("Use Alpaca paper account", value=False)

st.sidebar.markdown("### Research Loops")
run_walk_forward = st.sidebar.checkbox("Test on newer price data", value=True)
train_fraction = st.sidebar.slider("Older data used first (%)", 55, 80, 65, step=5) / 100
run_parameter_loop = st.sidebar.checkbox("Compare nearby strategy settings", value=False)
max_parameter_candidates = st.sidebar.slider("Settings to compare", 4, 16, 8, step=4)

st.sidebar.markdown("### Research Inputs")
st.sidebar.caption("Trend and risk approval are always included. The selected strategy adds either breakout or pullback inputs.")
setup_inputs = {
    "breakout_strength": st.sidebar.checkbox("Breakout strength", value=True),
    "volume": st.sidebar.checkbox("Volume confirmation", value=True),
    "volatility": st.sidebar.checkbox("Volatility", value=True),
    "room_above_exit": st.sidebar.checkbox("Room above exit", value=True),
    "relative_strength": st.sidebar.checkbox("Relative strength", value=False),
    "market_condition": st.sidebar.checkbox("Market condition", value=False),
    "liquidity": st.sidebar.checkbox("Liquidity", value=True),
    "event_risk": st.sidebar.checkbox("Event risk", value=False),
    "rsi": st.sidebar.checkbox("RSI condition", value=True),
}

with st.sidebar.expander("Files and saved records", expanded=False):
    persist_audit_log = st.checkbox("Save activity log", value=True)
    audit_log_path = st.text_input("Activity log file", value="audit_logs/agentloop_audit.jsonl")
    broker_state_path = st.text_input("Alpaca order file", value="broker_state/alpaca_paper_orders.json")
    automation_dry_run_path = st.text_input("Automation check file", value="automation_logs/paper_automation_dry_runs.jsonl")
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
    st.session_state["tracked_alpaca_orders"] = []
    st.session_state["simulated_alpaca_positions"] = []
    st.session_state["session_disabled"] = False
st.session_state.setdefault("shadow_decisions", [])
st.session_state.setdefault("paper_session_id", new_session_id())
st.session_state.setdefault("paper_session_started_at", pd.Timestamp.now(tz="America/Los_Angeles").isoformat())
st.session_state.setdefault("auto_exit_sent_hashes", [])
st.session_state.setdefault("auto_entry_sent_hashes", [])
st.session_state.setdefault("last_auto_exit_decision_key", None)
st.session_state.setdefault("last_auto_entry_decision_key", None)
st.session_state.setdefault("last_automation_action", "None")
st.session_state.setdefault("last_automation_blocked_reason", "")
st.session_state.setdefault("tracked_alpaca_orders", [])
st.session_state.setdefault("simulated_alpaca_positions", [])
audit_store = JsonlAuditStore(audit_log_path)
automation_store = AutomationDryRunStore(automation_dry_run_path)
manifest_store = RunManifestStore(run_manifest_path)
broker_state_store = BrokerStateStore(broker_state_path)
if not st.session_state["tracked_alpaca_orders"]:
    st.session_state["tracked_alpaca_orders"] = broker_state_store.read()
if emergency_disable_session:
    st.session_state["session_disabled"] = True
    disable_event = AuditEvent(
        event_type="session_disabled",
        message="Emergency disable session was tested and activated by the user.",
        payload={
            "source": "sidebar",
            "evidence_check": "emergency_disable_tested",
            "kill_switch_effective": True,
            "broker_writes_submitted": 0,
        },
    )
    st.session_state.setdefault("session_audit_events", []).append(disable_event)
    if persist_audit_log:
        audit_store.append(disable_event)
paper_broker: PaperBroker = st.session_state["paper_broker"]
paper_adapter = PaperBrokerAdapter(paper_broker)
alpaca_adapter = AlpacaBrokerAdapterStub(allow_order_submission=enable_alpaca_paper_orders)
broker_statuses = [paper_adapter.status(), alpaca_adapter.status()]
alpaca_status = broker_statuses[1]
alpaca_positions = alpaca_adapter.position_records() if alpaca_status.connected else []
alpaca_orders = alpaca_adapter.order_records() if alpaca_status.connected else []
alpaca_state_health = broker_state_health(alpaca_status.connected, alpaca_positions, alpaca_orders)
paper_positions_notional = sum(position.market_value for position in paper_broker.positions.values())
alpaca_positions_notional = sum(float(row.get("Market Value") or 0) for row in alpaca_positions)
paper_equity = paper_broker.cash + paper_positions_notional
session_pnl = paper_equity - st.session_state["paper_starting_cash"]
effective_kill_switch = kill_switch or st.session_state.get("session_disabled", False)

try:
    breakout_prices, breakout_smas, breakout_atrs, breakout_trade_log, breakout_live, breakout_stats, breakout_labels = simulate_turtle_strategy(
        account, entry_w, exit_w, atr_mult, risk_dec, ma_w, seed, market_data
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
}
selected_strategy_result = strategy_results[strategy_label]
prices = selected_strategy_result["prices"]
smas = selected_strategy_result["smas"]
atrs = selected_strategy_result["atrs"]
trade_log = selected_strategy_result["trade_log"]
live = selected_strategy_result["live"]
stats = selected_strategy_result["stats"]
labels = selected_strategy_result["labels"]
comparison_rows = strategy_comparison_records({
    "Breakout continuation": breakout_stats,
    "Trend pullback continuation": pullback_stats,
})

walk_forward_result = None
walk_forward_error = None
if run_walk_forward and strategy_type == "breakout":
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
        )
    except ValueError as exc:
        walk_forward_error = str(exc)
elif run_walk_forward and strategy_type == "pullback":
    walk_forward_error = "Newer-data test for trend pullback will be added after the pullback backtest has been reviewed."

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
}
current_exit_settings = dict(current_strategy_settings)
current_exit_settings["auto_exit_enabled"] = True


def evaluate_exit_rule_from_settings(settings: dict | None) -> tuple[bool, str]:
    if not settings:
        return False, "No saved exit settings for this position; auto exit is paused."
    if not bool(settings.get("auto_exit_enabled", True)):
        return False, "Auto exit is off for this position."
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
            return False, "Saved exit settings do not include a ticker."
        if history == "synthetic":
            return False, "Saved exit settings use synthetic data; auto exit needs real ticker data."
        data = fetch_stock_data(symbol, history, saved_interval)
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
        return bool(saved_live.get("exit_ready", False)), str(saved_live.get("exit_reason", "Strategy exit rule is not triggered."))
    except Exception as exc:
        return False, f"Could not check saved exit rule: {exc}"


parameter_candidates = []
recommended_candidate = None
parameter_loop_error = None
if run_parameter_loop and strategy_type == "breakout":
    try:
        parameter_candidates = evaluate_parameter_candidates(
            current=current_strategy_config,
            account=account,
            risk_pct_dec=risk_dec,
            seed=seed,
            market_data=market_data,
            train_fraction=train_fraction,
            max_candidates=max_parameter_candidates,
        )
        recommended_candidate = recommend_candidate(parameter_candidates)
    except ValueError as exc:
        parameter_loop_error = str(exc)
elif run_parameter_loop and strategy_type == "pullback":
    parameter_loop_error = "Nearby-setting comparison is currently available for breakout continuation only."

risk_limits = RiskLimits(
    allowed_symbols=allowed_symbols,
    max_risk_per_trade_pct=max_risk_limit,
    max_position_notional_pct=max_notional_limit,
    max_portfolio_exposure_pct=max_portfolio_exposure,
    max_symbol_concentration_pct=max_symbol_concentration,
    max_session_loss_pct=max_session_loss,
    max_open_positions=max_open_positions,
    require_stop_loss=True,
    kill_switch_enabled=effective_kill_switch,
)
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
    account_equity=account,
    paper_cash=paper_broker.cash,
)
current_manifest_record = run_manifest_record(current_run_manifest)
intent = live.get("trade_intent")
intent_symbol = intent.symbol_clean if intent else ""
alpaca_position_symbols = {str(row.get("Symbol", "")).strip().upper() for row in alpaca_positions}
symbol_current_notional = (
    paper_broker.positions[intent_symbol].market_value
    if intent_symbol in paper_broker.positions
    else 0.0
)
symbol_current_notional += sum(float(row.get("Market Value") or 0) for row in alpaca_positions if str(row.get("Symbol", "")).strip().upper() == intent_symbol)
raw_intent_quantity = intent.quantity if intent else None
intent = constrain_trade_intent_to_limits(
    intent,
    account_equity=account,
    limits=risk_limits,
    current_portfolio_notional=paper_positions_notional + alpaca_positions_notional,
    symbol_current_notional=symbol_current_notional,
    available_cash=paper_broker.cash,
)
live["trade_intent"] = intent
if intent is not None:
    live["raw_pos_size"] = raw_intent_quantity
    live["pos_size"] = intent.quantity
risk_check = check_trade_intent(
    intent,
    account_equity=account,
    limits=risk_limits,
    open_positions=set(paper_broker.positions.keys()) | alpaca_position_symbols,
    open_position_count=len(set(paper_broker.positions.keys()) | alpaca_position_symbols),
    current_portfolio_notional=paper_positions_notional + alpaca_positions_notional,
    symbol_current_notional=symbol_current_notional,
    session_pnl=session_pnl,
    available_cash=paper_broker.cash,
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
    run_parameter_loop,
    max_parameter_candidates,
    recommended_candidate.score if recommended_candidate else None,
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
    if run_parameter_loop and parameter_loop_error is None:
        parameter_event = AuditEvent(
            event_type="bounded_parameter_loop_completed",
            message=recommendation_summary(recommended_candidate),
            payload={
                "candidate_count": len(parameter_candidates),
                "recommended_score": None if recommended_candidate is None else recommended_candidate.score,
                "recommended_status": None if recommended_candidate is None else recommended_candidate.status,
            },
        )
        st.session_state["session_audit_events"].append(parameter_event)
        if persist_audit_log:
            audit_store.append(parameter_event)
    st.session_state["last_audit_key"] = audit_key

st.title("AgentLoop Trader")
st.caption(f"Research, risk checks, and paper trading. Prices: {source_caption}")

page_section("1. Daily command center", "Start here. This shows the current trading state and the next action.")
status_rows = compact_status_records(
    mode_label=mode_label,
    risk_approved=risk_check.approved,
    broker_connected=alpaca_status.connected,
    broker_state_stale=alpaca_state_health.stale,
    kill_switch_enabled=effective_kill_switch,
    live_writes_blocked=True,
)
status_cols = st.columns(len(status_rows))
for col, row in zip(status_cols, status_rows):
    state_label = str(row["Value"]).upper()
    metric_card(col, row["Status"], state_label, row["State"])

tracked_alpaca_orders = st.session_state.get("tracked_alpaca_orders", [])
alpaca_preview = build_alpaca_order_preview(intent, execution_decision, alpaca_adapter.config)
duplicate_alpaca_reasons = duplicate_exposure_reasons(intent, alpaca_positions)
open_order_reasons = open_order_exposure_reasons(intent, alpaca_orders)
duplicate_preview_submitted = preview_already_tracked(alpaca_preview.preview_hash, tracked_alpaca_orders)
exit_previews = build_exit_order_previews(alpaca_positions, alpaca_adapter.config)
cancelable_alpaca_orders = cancelable_alpaca_order_records(alpaca_orders)
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
    cancelable_order_count=len(cancelable_alpaca_orders),
    open_position_count=len(alpaca_positions),
)
setup_scorecard_rows = setup_scorecard_records(
    live,
    risk_approved=risk_check.approved,
    blocked_reasons=preflight_check.blocked_reasons or risk_check.rejected_reasons,
    enabled_inputs=setup_inputs,
    strategy_type=strategy_type,
)

trade_desk_tab, agent_loop_tab, broker_risk_tab, evidence_tab, readiness_tab = st.tabs(
    ["Trade", "Research", "Alpaca and Risk", "Records", "Safety"]
)
with trade_desk_tab:
    desk_cols = st.columns(4)
    metric_card(desk_cols[0], "Now", operator_state["State"], operator_state["Detail"])
    metric_card(desk_cols[1], "Reference Price", f"${float(live['last_p']):,.2f}", ticker)
    metric_card(desk_cols[2], "Strategy", strategy_label, "Selected in the sidebar")
    metric_card(desk_cols[3], "Session P&L", f"${session_pnl:,.2f}", "Since reset", "pos" if session_pnl >= 0 else "neg")
    st.info(f"Next action: {operator_state['Next Action']}")
    st.dataframe(pd.DataFrame(setup_scorecard_rows), use_container_width=True, hide_index=True)
    if intent is None:
        st.info("No trade right now. The strategy is waiting.")
    else:
        proposal_cols = st.columns(4)
        metric_card(proposal_cols[0], "Symbol", intent.symbol_clean, intent.side.upper())
        metric_card(proposal_cols[1], "Quantity", f"{intent.quantity:,}", intent.order_type)
        metric_card(proposal_cols[2], "Buy near", f"${intent.entry_price:,.2f}", "Strategy price")
        metric_card(proposal_cols[3], "Stop loss", f"${intent.stop_loss:,.2f}", "Risk rule")
        st.caption(intent.rationale)
        with st.expander("Trade idea details", expanded=False):
            st.dataframe(pd.DataFrame(proposal_records(trade_proposal)), use_container_width=True, hide_index=True)
with agent_loop_tab:
    st.markdown(
        agent_decision_summary(
            intent_present=intent is not None,
            thesis=trade_proposal.thesis.thesis,
            blocked_reasons=preflight_check.blocked_reasons or risk_check.rejected_reasons,
            next_action=operator_state["Next Action"],
        )
    )
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
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("#### Strategy context")
    st.dataframe(
        pd.DataFrame(strategy_context_records(live, entry_w, exit_w, ma_w)),
        use_container_width=True,
        hide_index=True,
    )
with broker_risk_tab:
    risk_broker_rows = [
        {"Item": "Trade allowed", "Value": plain_yes_no(preflight_check.ready)},
        {"Item": "Alpaca connected", "Value": plain_yes_no(alpaca_status.connected)},
        {"Item": "Kill Switch off", "Value": plain_yes_no(not effective_kill_switch)},
        {"Item": "Paper orders enabled", "Value": plain_yes_no(enable_alpaca_paper_orders)},
    ]
    st.dataframe(pd.DataFrame(risk_broker_rows), use_container_width=True, hide_index=True)
    if show_portfolio_evidence:
        st.dataframe(pd.DataFrame(preflight_records(preflight_check)), use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame(broker_status_records(broker_statuses)), use_container_width=True, hide_index=True)
    if alpaca_state_health.reasons:
        st.warning(" ".join(alpaca_state_health.reasons))
with evidence_tab:
    evidence_summary = [
        {"Item": "Actions recorded", "Value": len(st.session_state["session_audit_events"])},
        {"Item": "Alpaca paper orders tracked", "Value": len(st.session_state["tracked_alpaca_orders"])},
        {"Item": "Detailed records", "Value": "Hidden" if not show_portfolio_evidence else "Visible"},
    ]
    st.dataframe(pd.DataFrame(evidence_summary), use_container_width=True, hide_index=True)
    st.caption("Use Portfolio Evidence only when you want the full records.")
with readiness_tab:
    simple_ready_rows = [
        {"Check": "Alpaca connected", "Status": plain_yes_no(alpaca_status.connected)},
        {"Check": "Risk check passed", "Status": plain_yes_no(risk_check.approved)},
        {"Check": "Paper trading enabled", "Status": plain_yes_no(execution_mode == "paper")},
        {"Check": "Kill Switch off", "Status": plain_yes_no(not effective_kill_switch)},
        {"Check": "Live trading blocked", "Status": "Yes"},
    ]
    st.dataframe(pd.DataFrame(simple_ready_rows), use_container_width=True, hide_index=True)
    st.caption("Safety checks stay active. Detailed setup checks live in Portfolio Evidence.")

if show_portfolio_evidence:
    with st.expander("Safety summary", expanded=False):
        st.markdown("- Alpaca is the target broker, but live orders are still disabled.")
        st.markdown("- Paper orders require paper mode, passed risk checks, a connected paper account, and the paper account switch.")
        st.markdown("- The app cannot let the agent change risk rules, credentials, order code, or the Kill Switch.")
        st.markdown("- Live automation stays blocked while paper trading is being tested.")
        st.dataframe(pd.DataFrame(production_readiness_checks()), use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame(immutable_boundary_records()), use_container_width=True, hide_index=True)
        st.markdown("#### Broker failure examples")
        st.dataframe(pd.DataFrame(broker_state_simulation_records()), use_container_width=True, hide_index=True)

if show_portfolio_evidence:
    with st.expander("Session settings record", expanded=False):
        st.dataframe(pd.DataFrame(run_manifest_records([current_manifest_record])), use_container_width=True, hide_index=True)
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
            st.dataframe(pd.DataFrame(run_manifest_records(recent_manifests)), use_container_width=True, hide_index=True)
        st.caption("This saves the app settings used for a session. It does not contact Alpaca.")

    with st.expander("Live trading setup records", expanded=False):
        gitignore_text = Path(".gitignore").read_text(encoding="utf-8") if Path(".gitignore").exists() else ""
        live_lock_rows = live_mode_lockfile_records(live_lockfile_path)
        live_lockfile_present = any(row["Check"] == "Live trading locked" and row["Passed"] for row in live_lock_rows)
        st.markdown("#### Live trading lock")
        st.dataframe(pd.DataFrame(live_lock_rows), use_container_width=True, hide_index=True)
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
            use_container_width=True,
            hide_index=True,
        )
        st.caption("These checks are local only. The lock file does not enable live trading.")

page_section("2. Backtest chart and strategy results", "Start with the chart, then review the test results for the selected strategy settings.")
sub_section("2.1 Simulation chart and trade review", "Click a past simulated trade to highlight it on the chart.")
selected_idx = st.session_state.get("selected_trade_idx", None)
selected_trade = trade_log[selected_idx] if selected_idx is not None and 0 <= selected_idx < len(trade_log) else None
st.plotly_chart(
    build_chart(prices, smas, atrs, entry_w, exit_w, ma_w, labels, trade_log, selected_trade),
    use_container_width=True,
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
        "Stop $": t["stop"],
        "P&L $": t["pnl"],
        "% Account": t["pct_acct"],
    } for t in trade_log]).set_index("#")

    def color_pnl(val):
        return "color: #3B6D11" if val > 0 else "color: #A32D2D"

    event = st.dataframe(
        display_df.style.map(color_pnl, subset=["P&L $", "% Account"]),
        use_container_width=True,
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
        st.dataframe(pd.DataFrame(review_records(review)), use_container_width=True, hide_index=True)
        with st.expander("Lessons from this trade", expanded=False):
            for lesson in review.lessons:
                st.markdown(f"- {lesson}")
else:
    st.caption("No trades happened in this simulation run.")

sub_section("2.2 Strategy test results")
st.info(f"Current trading rule: {strategy_label}. Only the selected strategy can create the trade idea shown in this run.")
c1, c2, c3, c4 = st.columns(4)
pnl_color = "pos" if stats["total_pnl"] >= 0 else "neg"
metric_card(c1, "Final equity", f"${stats['final_equity']:,}", f"Started ${account:,}")
metric_card(c2, "Total P&L", f"${stats['total_pnl']:,}", f"{stats['return_pct']}% return", pnl_color)
metric_card(c3, "Win rate", f"{stats['win_rate']}%", f"{stats['wins']}W / {stats['losses']}L of {stats['total_trades']} trades")
metric_card(c4, "Avg win/loss", f"{stats['rr_ratio']}x", "Average winner vs loser")

c5, c6, c7 = st.columns(3)
metric_card(c5, "Worst drop", f"{stats['max_drawdown_pct']}%", "Largest equity pullback")
metric_card(c6, "Win/loss dollars", f"{stats['profit_factor']}x", "Total wins vs total losses")
metric_card(c7, "Time in trade", f"{stats['exposure_pct']}%", "Share of bars spent in a trade")

sub_section("2.3 Strategy comparison")
st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
st.caption("This compares the existing breakout strategy against the new trend pullback strategy on the same ticker and settings.")

sub_section("2.4 Test on newer price data")
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
            use_container_width=True,
            hide_index=True,
        )
    with st.expander("What this test used", expanded=False):
        st.markdown(
            f"Older bars used to tune/check rules: **{walk_forward_result.train_bars}**. "
            f"Newer bars used as the test: **{walk_forward_result.oos_bars}**. "
            f"Extra bars needed for indicators: **{walk_forward_result.warmup_bars}**."
        )
        for reason in walk_forward_result.reasons:
            st.markdown(f"- {reason}")

sub_section("2.5 Compare nearby strategy settings")
if not run_parameter_loop:
    st.caption("This is turned off. Enable it in the sidebar to compare nearby strategy settings.")
elif parameter_loop_error:
    st.warning(parameter_loop_error)
else:
    st.markdown(f"**Suggested setting:** {recommendation_summary(recommended_candidate)}")
    if recommended_candidate is not None:
        rec = recommended_candidate.config
        rec_cols = st.columns(4)
        metric_card(rec_cols[0], "Entry", rec.entry_window, "Bars")
        metric_card(rec_cols[1], "Exit", rec.exit_window, "Bars")
        metric_card(rec_cols[2], "ATR stop", rec.atr_stop_multiplier, "Multiplier")
        metric_card(rec_cols[3], "SMA filter", rec.moving_average_window, "Bars")
    if show_portfolio_evidence:
        st.dataframe(
            pd.DataFrame(candidate_records(parameter_candidates)),
            use_container_width=True,
            hide_index=True,
        )
    st.caption("This comparison can suggest strategy settings only. It cannot change risk limits, broker access, order mode, credentials, or the Kill Switch.")

page_section("3. Trade decision", "The strategy signal, agent trade idea, and risk decision for the current prices.")
sig = live["signal"]
if sig == "long":
    st.markdown('<span class="signal-long">BUY SIGNAL</span>', unsafe_allow_html=True)
elif sig == "exit":
    st.markdown('<span class="signal-exit">SELL SIGNAL</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="signal-flat">NO TRADE RIGHT NOW</span>', unsafe_allow_html=True)

sub_section("3.1 Setup quality")
st.dataframe(pd.DataFrame(setup_scorecard_rows), use_container_width=True, hide_index=True)

with st.expander("Required for BUY vs quality checks", expanded=False):
    st.markdown("#### Required for BUY")
    st.dataframe(pd.DataFrame(buy_requirement_records(live)), use_container_width=True, hide_index=True)
    st.caption("Every required BUY rule must pass before the app can create a BUY intent.")
    st.markdown("#### Quality checks only")
    st.dataframe(pd.DataFrame(optional_quality_input_records(setup_inputs)), use_container_width=True, hide_index=True)
    st.caption("Quality checks help you judge the setup. They do not create a BUY intent by themselves.")

with st.expander("Required for SELL vs quality checks", expanded=False):
    st.markdown("#### Required for automatic SELL")
    st.dataframe(
        pd.DataFrame(
            sell_requirement_records(
                live,
                exit_preview_count=len(exit_previews),
                exit_settings_saved=exit_settings_available if first_exit_symbol else None,
            )
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Every required SELL rule must pass before the app can send an automatic paper sell.")
    st.markdown("#### Quality checks only")
    st.dataframe(pd.DataFrame(optional_sell_quality_records(setup_inputs)), use_container_width=True, hide_index=True)
    st.caption("Quality checks help you review the position. They do not create an automatic SELL by themselves.")

lp = round(float(live["last_p"]), 2)
dh = round(float(live["don_high"]), 2)
dl = round(float(live["don_low"]), 2)
ls = round(float(live["last_sma"]), 2) if live["last_sma"] else "n/a"
la = round(float(live["last_atr"]), 2) if live["last_atr"] else "n/a"
r1_status = "breakout confirmed" if lp > dh else "below channel"
r2_status = "upward filter passed" if live["sma_up"] else "downward no entry"

with st.expander("Current strategy rules", expanded=False):
    if strategy_type == "pullback":
        pullback_depth = live.get("pullback_depth_pct")
        pullback_text = f"{float(pullback_depth):.1f}%" if pullback_depth else "n/a"
        st.markdown(f"""
        <div class="rule-box"><b>1. Trend:</b> Price should stay above a rising {ma_w}-bar average. Current trend filter: {r2_status}</div>
        <div class="rule-box"><b>2. Pullback:</b> Look for a controlled pullback near the {pullback_w}-bar average. Current pullback depth: <b>{pullback_text}</b></div>
        <div class="rule-box"><b>3. Momentum turn:</b> Buy only after price turns back up above the {momentum_w}-bar average. Current: <b>{'turn confirmed' if live.get('momentum_turn') else 'not confirmed'}</b></div>
        <div class="rule-box"><b>4. Risk:</b> Stop distance = <b>{atr_mult}x ATR = ${live['stop_from_entry']}</b>. Risk {risk_pct}% of ${live['balance']:,.0f}; position = <b>{live['pos_size']} shares</b></div>
        <div class="rule-box"><b>5. Exit:</b> Sell if price loses the pullback average or hits the ATR stop.</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="rule-box"><b>1. Entry:</b> Buy when price breaks above {entry_w}-bar high (<b>${dh}</b>). Current: <b>${lp}</b> - {r1_status}</div>
        <div class="rule-box"><b>2. Filter:</b> {ma_w}-bar SMA = <b>${ls}</b>. Trend filter: {r2_status}</div>
        <div class="rule-box"><b>3. Volatility:</b> ATR (14d) = <b>${la}</b>. Stop distance = <b>{atr_mult}x ATR = ${live['stop_from_entry']}</b></div>
        <div class="rule-box"><b>4. Position size:</b> Risk {risk_pct}% of ${live['balance']:,.0f} = <b>${live['balance'] * risk_dec:,.0f}</b>. Position = <b>{live['pos_size']} shares</b></div>
        <div class="rule-box"><b>5. Exit:</b> Sell when price touches {exit_w}-bar low (<b>${dl}</b>). Current: <b>${lp}</b> - {'exit triggered' if lp <= dl else 'holding'}</div>
        """, unsafe_allow_html=True)

with st.expander("Agent trade idea", expanded=False):
    proposal_summary = pd.DataFrame(proposal_records(trade_proposal))
    st.dataframe(proposal_summary, use_container_width=True, hide_index=True)
    st.markdown(f"**Trade idea:** {trade_proposal.thesis.thesis}")
    st.markdown(f"**What would make this wrong:** {trade_proposal.thesis.invalidation}")
    st.markdown("#### What the agent used")
    for item in trade_proposal.thesis.data_basis:
        st.markdown(f"- {item}")

sub_section("3.2 Final answer")
if intent is None:
    final_answer = "WAIT"
    final_detail = no_buy_reason(live)
elif preflight_check.blocked_reasons or risk_check.rejected_reasons:
    final_answer = "BLOCK"
    final_detail = (preflight_check.blocked_reasons or risk_check.rejected_reasons)[0]
elif intent is not None and preflight_check.ready:
    final_answer = "TRADE"
    final_detail = "Strategy setup and risk checks passed."
else:
    final_answer = "WAIT"
    final_detail = "No valid trade setup right now."
answer_color = {"TRADE": "#3B6D11", "WAIT": "#8A6D1D", "BLOCK": "#A32D2D"}[final_answer]
st.markdown(
    f"**Final answer:** "
    f"<span style='color:{answer_color};font-weight:700'>{final_answer}</span> - {final_detail}",
    unsafe_allow_html=True,
)
if intent is None:
    st.info(final_detail)
preflight_color = "#3B6D11" if preflight_check.ready else "#A32D2D"
st.markdown(
    f"**{operator_state['State']}:** "
    f"<span style='color:{preflight_color};font-weight:600'>"
    f"{operator_state['Next Action']}</span>",
    unsafe_allow_html=True,
)
if preflight_check.blocked_reasons:
    show_blockers("Why not", preflight_check.blocked_reasons)
elif execution_decision.requires_manual_approval:
    st.info("The trade passed risk checks. Use the paper trading section to send it.")
if show_portfolio_evidence:
    policy_tabs = st.tabs(["Risk limits", "Rule-by-rule checks"])
    with policy_tabs[0]:
        st.dataframe(
            pd.DataFrame(risk_policy_records(risk_limits)),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("The agent can suggest strategy settings only. It cannot change risk rules, credentials, order code, or the Kill Switch.")
    with policy_tabs[1]:
        st.dataframe(
            pd.DataFrame(preflight_records(preflight_check)),
            use_container_width=True,
            hide_index=True,
        )

checks_df = pd.DataFrame([{"Check": name.replace("_", " ").title(), "Passed": passed} for name, passed in risk_check.checks.items()])
if show_portfolio_evidence and not checks_df.empty:
    st.markdown("#### Risk check details")
    st.dataframe(checks_df, use_container_width=True, hide_index=True)

page_section("4. Paper trading", "Send, exit, or cancel Alpaca paper orders from one place. Detailed records stay in Portfolio Evidence.")
st.info(f"{operator_state['State']}: {operator_state['Next Action']}")
with st.expander("Alpaca connection and account", expanded=False):
    broker_summary_rows = [
        {"Item": "Alpaca connected", "Value": plain_yes_no(alpaca_status.connected)},
        {"Item": "Paper orders enabled", "Value": plain_yes_no(enable_alpaca_paper_orders)},
        {"Item": "Open Alpaca positions", "Value": len(alpaca_positions)},
        {"Item": "Alpaca orders waiting to fill", "Value": count_waiting_alpaca_orders(alpaca_orders)},
    ]
    st.dataframe(pd.DataFrame(broker_summary_rows), use_container_width=True, hide_index=True)
    if show_portfolio_evidence:
        st.markdown("#### Broker details")
        st.dataframe(
            pd.DataFrame(broker_status_records(broker_statuses)),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("#### Alpaca setup checks")
        st.dataframe(pd.DataFrame(alpaca_config_validation_records(alpaca_adapter.config)), use_container_width=True, hide_index=True)
    alpaca_tabs = st.tabs(["Alpaca account", "Alpaca positions", "Alpaca orders"])
    with alpaca_tabs[0]:
        account_records = alpaca_adapter.account_records()
        if account_records:
            st.dataframe(pd.DataFrame(account_records), use_container_width=True, hide_index=True)
        else:
            st.caption("No Alpaca account data available. Configure paper credentials and install alpaca-py.")
    with alpaca_tabs[1]:
        if alpaca_positions:
            st.dataframe(pd.DataFrame(alpaca_positions), use_container_width=True, hide_index=True)
        else:
            st.caption("No Alpaca positions available.")
    with alpaca_tabs[2]:
        if alpaca_orders:
            st.dataframe(pd.DataFrame(alpaca_orders), use_container_width=True, hide_index=True)
        else:
            st.caption("No Alpaca orders available.")
    reconcile_rows = reconcile_alpaca_positions(alpaca_positions, st.session_state["tracked_alpaca_orders"])
    if show_portfolio_evidence and reconcile_rows:
        st.markdown("#### Position tracking details")
        st.dataframe(pd.DataFrame(reconcile_rows), use_container_width=True, hide_index=True)
    if show_portfolio_evidence and exit_previews:
        st.markdown("#### Exit order details")
        st.dataframe(pd.DataFrame(exit_preview_records(exit_previews)), use_container_width=True, hide_index=True)
    if alpaca_state_health.reasons:
        st.warning(" ".join(alpaca_state_health.reasons))

with st.expander("Local simulator account", expanded=False):
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

sub_section("4.1 Open positions", "Select a paper position and manage its own exit automation settings.")
if alpaca_positions:
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

    try:
        selected_avg_entry = float(selected_position.get("Average Entry") or 0)
    except (TypeError, ValueError):
        selected_avg_entry = 0.0
    position_cols = st.columns(4)
    metric_card(position_cols[0], "Symbol", selected_position_symbol, "Alpaca paper")
    metric_card(position_cols[1], "Quantity", selected_position.get("Quantity", ""), "Current position")
    metric_card(position_cols[2], "Avg entry", f"${selected_avg_entry:,.2f}", "From Alpaca")
    metric_card(position_cols[3], "Auto exit", "On" if selected_exit_settings.get("auto_exit_enabled", True) else "Off", "Position setting")

    with st.expander("Entry and exit settings for selected position", expanded=False):
        st.markdown("#### Saved entry settings")
        if selected_entry_settings:
            st.dataframe(
                pd.DataFrame([{"Setting": key.replace("_", " ").title(), "Value": value} for key, value in selected_entry_settings.items()]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No saved entry settings found for this position.")

        with st.form(f"exit_settings_{selected_position_symbol}"):
            st.markdown("#### Edit exit settings")
            exit_auto_enabled = st.checkbox(
                "Auto exit this position",
                value=bool(selected_exit_settings.get("auto_exit_enabled", True)),
                key=f"auto_exit_enabled_{selected_position_symbol}",
            )
            exit_strategy_options = {
                "Breakout continuation": "breakout",
                "Trend pullback continuation": "pullback",
            }
            default_exit_strategy = str(selected_exit_settings.get("strategy_type", strategy_type))
            default_exit_strategy_label = next(
                (label for label, value in exit_strategy_options.items() if value == default_exit_strategy),
                strategy_label,
            )
            edited_exit_strategy_label = st.selectbox(
                "Exit strategy",
                list(exit_strategy_options.keys()),
                index=list(exit_strategy_options.keys()).index(default_exit_strategy_label),
                key=f"exit_strategy_{selected_position_symbol}",
            )
            edited_exit_strategy_type = exit_strategy_options[edited_exit_strategy_label]
            edited_exit_window = st.slider(
                "Sell exit length (bars)",
                5,
                30,
                int(selected_exit_settings.get("exit_window", exit_w)),
                step=5,
                key=f"exit_window_{selected_position_symbol}",
            )
            edited_atr_mult = st.slider(
                "Stop distance (ATR multiplier)",
                1.0,
                4.0,
                float(selected_exit_settings.get("atr_stop_multiplier", atr_mult)),
                step=0.5,
                key=f"exit_atr_mult_{selected_position_symbol}",
            )
            edited_trend_filter = st.slider(
                "Trend filter length (bars)",
                50,
                300,
                int(selected_exit_settings.get("moving_average_window", ma_w)),
                step=50,
                key=f"exit_trend_filter_{selected_position_symbol}",
            )
            edited_pullback_length = st.slider(
                "Pullback average length (bars)",
                10,
                50,
                int(selected_exit_settings.get("pullback_average_length", pullback_w)),
                step=5,
                key=f"exit_pullback_length_{selected_position_symbol}",
            )
            edited_momentum_length = st.slider(
                "Momentum turn length (bars)",
                3,
                20,
                int(selected_exit_settings.get("momentum_turn_length", momentum_w)),
                step=1,
                key=f"exit_momentum_length_{selected_position_symbol}",
            )
            save_exit_settings = st.form_submit_button("Save Exit Settings For This Position")

        if save_exit_settings:
            edited_exit_settings = {
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
            }
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
else:
    st.caption("No Alpaca paper positions are open.")

sub_section("4.2 Account safety")
monitor_color = {"OK": "#3B6D11", "WARN": "#8A6D1D", "BREACH": "#A32D2D"}.get(monitoring_result.status, "inherit")
monitor_label = {"OK": "OK", "WARN": "Needs attention", "BREACH": "Blocked"}.get(monitoring_result.status, monitoring_result.status)
current_market_advisory = market_session_advisory()
st.markdown(
    f"**Trading status:** "
    f"<span style='color:{monitor_color};font-weight:600'>{monitor_label}</span>",
    unsafe_allow_html=True,
)
st.caption(current_market_advisory.get("Message", ""))
if show_portfolio_evidence:
    st.dataframe(pd.DataFrame(monitoring_records(monitoring_result)), use_container_width=True, hide_index=True)
    st.dataframe(pd.DataFrame([current_market_advisory]), use_container_width=True, hide_index=True)
    st.markdown("#### Broker health details")
    st.dataframe(
        pd.DataFrame(
            broker_heartbeat_records(
                broker_connected=alpaca_status.connected,
                broker_state_stale=alpaca_state_health.stale,
                broker_reasons=alpaca_state_health.reasons,
                market_advisory=current_market_advisory,
            )
        ),
        use_container_width=True,
        hide_index=True,
    )
for alert in monitoring_result.alerts:
    if monitoring_result.status == "BREACH":
        st.error(alert)
    elif monitoring_result.status == "WARN":
        st.warning(alert)
    else:
        st.caption(alert)

sub_section("4.3 Manual paper actions", "Use these buttons when you want to send, sell, or cancel Alpaca paper orders yourself.")
can_submit = intent is not None and execution_mode == "paper"
submit_disabled = intent is None or not preflight_check.ready or execution_mode != "paper"
with st.expander("Local simulator order - does not contact Alpaca", expanded=False):
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
    st.caption("This updates only the in-app simulator. Use the Alpaca buttons below for Alpaca paper orders.")

alpaca_base_disabled = (
    intent is None
    or not preflight_check.ready
    or execution_mode != "paper"
    or not alpaca_status.connected
    or not enable_alpaca_paper_orders
    or not alpaca_preview.valid
    or alpaca_state_health.stale
    or bool(duplicate_alpaca_reasons)
    or bool(open_order_reasons)
    or duplicate_preview_submitted
)
if intent is not None:
    st.markdown("#### Manual paper buy")
    if show_portfolio_evidence:
        st.dataframe(pd.DataFrame(alpaca_preview_records(alpaca_preview)), use_container_width=True, hide_index=True)
    elif alpaca_preview.valid:
        st.info(f"Ready: buy {alpaca_preview.order.get('quantity', '')} {alpaca_preview.order.get('symbol', '')} in Alpaca paper.")
    if alpaca_preview.blocked_reasons:
        show_blockers("Order blocked", alpaca_preview.blocked_reasons)
    if duplicate_alpaca_reasons:
        show_blockers("Order blocked", duplicate_alpaca_reasons)
    if open_order_reasons:
        show_blockers("Order blocked", open_order_reasons)
    if duplicate_preview_submitted:
        st.warning("This paper order is already tracked in the app.")
    if alpaca_state_health.stale:
        st.warning("Refresh Alpaca positions and orders before sending this.")
    elif not duplicate_alpaca_reasons and not open_order_reasons and not duplicate_preview_submitted:
        st.success("Paper buy is ready to send.")

alpaca_submit_disabled = (
    alpaca_base_disabled
    or not alpaca_status.can_submit_orders
)
if st.button("Send Paper Buy to Alpaca", disabled=alpaca_submit_disabled):
    try:
        alpaca_order = alpaca_adapter.submit_order(
            intent,
            execution_decision,
            expected_preview_hash=alpaca_preview.preview_hash,
        )
        alpaca_event = AuditEvent(
            event_type="alpaca_paper_order_submitted",
            message="Alpaca paper order submitted through the gated adapter.",
            payload={
                "symbol": intent.symbol_clean,
                "side": intent.side,
                "quantity": intent.quantity,
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
                "status": str(getattr(alpaca_order, "status", "")),
                "submitted_at": str(getattr(alpaca_order, "submitted_at", "")),
                "strategy_settings": current_strategy_settings,
                "exit_settings": current_exit_settings,
            }
            st.session_state["tracked_alpaca_orders"].append(tracked_record)
            broker_state_store.upsert(tracked_record)
        st.session_state["session_audit_events"].append(alpaca_event)
        if persist_audit_log:
            audit_store.append(alpaca_event)
        st.success("Paper buy order sent to Alpaca.")
    except Exception as exc:
        alpaca_event = AuditEvent(
            event_type="alpaca_paper_order_blocked",
            message=str(exc),
            payload={"symbol": intent.symbol_clean if intent else None, "preview_hash": alpaca_preview.preview_hash},
        )
        st.session_state["session_audit_events"].append(alpaca_event)
        if persist_audit_log:
            audit_store.append(alpaca_event)
        st.error(f"Paper buy order blocked: {exc}")

if show_portfolio_evidence and tracked_alpaca_orders:
    st.markdown("#### Paper orders saved in the app")
    tracked_rows = [
        alpaca_adapter.tracked_order_record(
            item.get("broker_order_id", ""),
            item.get("preview_hash", ""),
        )
        for item in tracked_alpaca_orders
    ]
    st.dataframe(pd.DataFrame(tracked_rows), use_container_width=True, hide_index=True)

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
        st.markdown("#### Paper order history")
        st.dataframe(pd.DataFrame(lifecycle_summary), use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame(lifecycle_rows), use_container_width=True, hide_index=True)

    position_lifecycle_rows = alpaca_position_lifecycle_records(alpaca_positions, refreshed_order_state)
    if position_lifecycle_rows:
        if show_portfolio_evidence:
            st.markdown("#### Paper position history")
            st.dataframe(
                pd.DataFrame(alpaca_position_lifecycle_summary_records(position_lifecycle_rows)),
                use_container_width=True,
                hide_index=True,
            )
            st.dataframe(pd.DataFrame(position_lifecycle_rows), use_container_width=True, hide_index=True)
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
    with st.expander("Practice fill and exit records", expanded=False):
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
                use_container_width=True,
                hide_index=True,
            )
            st.dataframe(pd.DataFrame(simulated_lifecycle_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Practice exit check")
        exit_readiness_rows = simulated_exit_preview_readiness_records(
            selected_sim_order,
            alpaca_adapter.config,
        )
        st.dataframe(pd.DataFrame(exit_readiness_rows), use_container_width=True, hide_index=True)
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
)
auto_entry_status = auto_entry_decision(
    automation_level=automation_level,
    execution_mode=execution_mode,
    broker_connected=alpaca_status.connected,
    broker_can_submit=alpaca_status.can_submit_orders,
    paper_orders_enabled=enable_alpaca_paper_orders,
    kill_switch_enabled=effective_kill_switch,
    broker_state_stale=alpaca_state_health.stale,
    market_open=bool(current_market_advisory.get("Open", False)),
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
    automation_level=automation_level,
    execution_mode=execution_mode,
    broker_connected=alpaca_status.connected,
    broker_can_submit=alpaca_status.can_submit_orders,
    paper_orders_enabled=enable_alpaca_paper_orders,
    kill_switch_enabled=effective_kill_switch,
    broker_state_stale=alpaca_state_health.stale,
    market_open=bool(current_market_advisory.get("Open", False)),
    strategy_exit_ready=managed_exit_ready,
    strategy_exit_reason=managed_exit_reason,
    exit_preview_records=exit_preview_rows_for_auto,
    exit_blockers=exit_blockers_by_hash_for_auto,
    entry_settings_match=exit_settings_available,
    entry_settings_reason=exit_settings_reason,
    already_sent_hashes=set(st.session_state.get("auto_exit_sent_hashes", [])),
)
sub_section("4.4 Automation status")
automation_checked_at = pd.Timestamp.now(tz="America/Los_Angeles").isoformat()
if automation_level == "Manual review only":
    st.info("Automation is off. You click paper order buttons manually.")
elif auto_exit_status.ready:
    st.success(f"Auto exits can sell {auto_exit_status.quantity} {auto_exit_status.symbol} in Alpaca paper.")
elif auto_entry_status.ready:
    st.success(f"Auto entries and exits can buy {auto_entry_status.quantity} {auto_entry_status.symbol} in Alpaca paper.")
else:
    visible_status = auto_entry_status.status if automation_level == "Auto entries and exits" else auto_exit_status.status
    visible_reasons = auto_entry_status.reasons if automation_level == "Auto entries and exits" else auto_exit_status.reasons
    st.warning(f"{visible_status}: {'; '.join(visible_reasons)}")
if automation_level == "Auto entries and exits":
    st.caption(
        "Automatic paper buys use Section 4.4, not the manual button in Section 4.3. "
        "They send only when the market is open, Alpaca paper is connected, risk checks pass, and the app reruns."
    )
elif automation_level == "Auto exits only":
    st.caption("Auto exits can sell Alpaca paper positions. New paper buys still require the manual button in Section 4.3.")
runtime_state = AutomationRuntimeState(
    mode=automation_level_label,
    status=(
        "Exit ready"
        if auto_exit_status.ready
        else "Buy ready"
        if auto_entry_status.ready
        else "Manual"
        if automation_level == "Manual review only"
        else auto_entry_status.status
        if automation_level == "Auto entries and exits"
        else auto_exit_status.status
    ),
    last_checked_at=automation_checked_at,
    last_action=st.session_state.get("last_automation_action", "None"),
    blocked_reason=st.session_state.get("last_automation_blocked_reason", ""),
)
st.caption(
    f"Last checked: {runtime_state.last_checked_at}. "
    f"Last action: {runtime_state.last_action}."
)
if show_portfolio_evidence:
    st.markdown("#### Automation runtime")
    st.dataframe(pd.DataFrame(automation_runtime_records(runtime_state)), use_container_width=True, hide_index=True)
    st.markdown("#### Automatic buy check")
    st.dataframe(pd.DataFrame(auto_entry_decision_records(auto_entry_status)), use_container_width=True, hide_index=True)
    st.markdown("#### Automatic exit check")
    st.dataframe(pd.DataFrame(auto_exit_decision_records(auto_exit_status)), use_container_width=True, hide_index=True)
auto_entry_key = (
    automation_level,
    auto_entry_status.status,
    auto_entry_status.preview_hash,
    tuple(auto_entry_status.reasons),
)
if automation_level == "Auto entries and exits" and st.session_state.get("last_auto_entry_decision_key") != auto_entry_key:
    auto_entry_event = AuditEvent(
        event_type="auto_entry_decision_recorded",
        message=f"Automatic paper buy decision: {auto_entry_status.status}.",
        payload={
            "automation_level": automation_level,
            "status": auto_entry_status.status,
            "ready": auto_entry_status.ready,
            "symbol": auto_entry_status.symbol,
            "quantity": auto_entry_status.quantity,
            "preview_hash": auto_entry_status.preview_hash,
            "reasons": auto_entry_status.reasons,
            "checked_at": automation_checked_at,
            "broker_writes_submitted": 0,
        },
    )
    st.session_state["session_audit_events"].append(auto_entry_event)
    if persist_audit_log:
        audit_store.append(auto_entry_event)
    st.session_state["last_auto_entry_decision_key"] = auto_entry_key
auto_exit_key = (
    automation_level,
    auto_exit_status.status,
    auto_exit_status.preview_hash,
    tuple(auto_exit_status.reasons),
)
if automation_level != "Manual review only" and st.session_state.get("last_auto_exit_decision_key") != auto_exit_key:
    auto_exit_event = AuditEvent(
        event_type="auto_exit_decision_recorded",
        message=f"Automatic paper exit decision: {auto_exit_status.status}.",
        payload={
            "automation_level": automation_level,
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
                "quantity": auto_exit_intent.quantity if auto_exit_intent else "",
                "status": str(getattr(alpaca_auto_exit_order, "status", "")),
                "submitted_at": str(getattr(alpaca_auto_exit_order, "submitted_at", "")),
                "source": "auto_exit_only",
            }
            st.session_state["tracked_alpaca_orders"].append(tracked_record)
            broker_state_store.upsert(tracked_record)
        st.session_state["auto_exit_sent_hashes"].append(auto_exit_status.preview_hash)
        st.session_state["last_automation_action"] = f"Paper exit sent: {auto_exit_intent.quantity} {auto_exit_symbol}"
        st.session_state["last_automation_blocked_reason"] = ""
        auto_submit_event = AuditEvent(
            event_type="auto_paper_exit_submitted",
            message="Automatic paper exit sent to Alpaca.",
            payload={
                "symbol": auto_exit_symbol,
                "side": "sell",
                "quantity": auto_exit_intent.quantity if auto_exit_intent else "",
                "preview_hash": auto_exit_status.preview_hash,
                "broker_order_id": broker_order_id,
                "automation_level": automation_level,
                "broker_writes_submitted": 1,
            },
        )
        st.session_state["session_audit_events"].append(auto_submit_event)
        if persist_audit_log:
            audit_store.append(auto_submit_event)
        st.success("Automatic paper exit sent to Alpaca.")
        st.rerun()
    except Exception as exc:
        st.session_state["auto_exit_sent_hashes"].append(auto_exit_status.preview_hash)
        st.session_state["last_automation_action"] = "Paper exit blocked"
        st.session_state["last_automation_blocked_reason"] = str(exc)
        auto_block_event = AuditEvent(
            event_type="auto_paper_exit_blocked",
            message=str(exc),
            payload={
                "symbol": auto_exit_symbol,
                "preview_hash": auto_exit_status.preview_hash,
                "automation_level": automation_level,
                "broker_writes_submitted": 0,
            },
        )
        st.session_state["session_audit_events"].append(auto_block_event)
        if persist_audit_log:
            audit_store.append(auto_block_event)
        st.error(f"Automatic paper exit blocked: {exc}")

if (not auto_exit_status.ready) and auto_entry_status.ready:
    try:
        if intent is None:
            raise ValueError("Automatic buy could not find a current trade idea.")
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
                "status": str(getattr(alpaca_auto_entry_order, "status", "")),
                "submitted_at": str(getattr(alpaca_auto_entry_order, "submitted_at", "")),
                "source": "auto_entries_and_exits",
                "strategy_settings": current_strategy_settings,
                "exit_settings": current_exit_settings,
            }
            st.session_state["tracked_alpaca_orders"].append(tracked_record)
            broker_state_store.upsert(tracked_record)
        st.session_state["auto_entry_sent_hashes"].append(auto_entry_status.preview_hash)
        st.session_state["last_automation_action"] = f"Paper buy sent: {intent.quantity} {intent.symbol_clean}"
        st.session_state["last_automation_blocked_reason"] = ""
        auto_entry_submit_event = AuditEvent(
            event_type="auto_paper_entry_submitted",
            message="Automatic paper buy sent to Alpaca.",
            payload={
                "symbol": intent.symbol_clean,
                "side": intent.side,
                "quantity": intent.quantity,
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
        st.success("Automatic paper buy sent to Alpaca.")
        st.rerun()
    except Exception as exc:
        st.session_state["auto_entry_sent_hashes"].append(auto_entry_status.preview_hash)
        st.session_state["last_automation_action"] = "Paper buy blocked"
        st.session_state["last_automation_blocked_reason"] = str(exc)
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
        st.error(f"Automatic paper buy blocked: {exc}")

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
        st.dataframe(pd.DataFrame(alpaca_preview_records(alpaca_exit_preview)), use_container_width=True, hide_index=True)
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
            rationale="Exit order generated from existing Alpaca paper position.",
            source_signals=["alpaca_position_exit_preview"],
        )
        if selected_exit_position is not None
        else None
    )
    exit_decision = ExecutionDecision(
        mode="paper",
        approved_for_execution=True,
        requires_manual_approval=False,
        reason="Paper exit passed the exit checks.",
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
        st.success("Paper exit is ready to send.")

    exit_base_disabled = (
        selected_exit_intent is None
        or execution_mode != "paper"
        or not alpaca_status.connected
        or not enable_alpaca_paper_orders
        or alpaca_state_health.stale
        or not alpaca_exit_preview.valid
        or bool(exit_position_blockers)
        or bool(duplicate_exit_reasons)
    )
    alpaca_exit_submit_disabled = (
        exit_base_disabled
        or not alpaca_status.can_submit_orders
    )
    if st.button("Send Paper Exit to Alpaca", disabled=alpaca_exit_submit_disabled):
        try:
            alpaca_exit_order = alpaca_adapter.submit_order(
                selected_exit_intent,
                exit_decision,
                expected_preview_hash=alpaca_exit_preview.preview_hash,
            )
            exit_event = AuditEvent(
                event_type="alpaca_paper_exit_submitted",
                message="Alpaca paper exit submitted through the gated adapter.",
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
            st.success("Paper exit sent to Alpaca.")
        except Exception as exc:
            exit_event = AuditEvent(
                event_type="alpaca_paper_exit_blocked",
                message=str(exc),
                payload={"symbol": exit_symbol, "preview_hash": alpaca_exit_preview.preview_hash},
            )
            st.session_state["session_audit_events"].append(exit_event)
            if persist_audit_log:
                audit_store.append(exit_event)
            st.error(f"Paper exit blocked: {exc}")
    st.caption("This contacts Alpaca paper only.")

if cancelable_alpaca_orders:
    st.markdown("#### Cancel paper order")
    cancel_options = [
        f"{row['Symbol']} {row['Side']} {row['Quantity']} {row['Status']} ({row['Order ID']})"
        for row in cancelable_alpaca_orders
    ]
    selected_cancel_idx = st.selectbox(
        "Alpaca paper order to cancel",
        range(len(cancel_options)),
        format_func=lambda idx: cancel_options[idx],
    )
    selected_cancel_order = cancelable_alpaca_orders[selected_cancel_idx]
    alpaca_cancel_preview = build_alpaca_cancel_preview(selected_cancel_order, alpaca_adapter.config)
    if show_portfolio_evidence:
        st.dataframe(pd.DataFrame(alpaca_cancel_preview_records(alpaca_cancel_preview)), use_container_width=True, hide_index=True)
    if alpaca_cancel_preview.blocked_reasons:
        show_blockers("Cancel blocked", alpaca_cancel_preview.blocked_reasons)
    if alpaca_state_health.stale:
        st.warning("Refresh Alpaca positions and orders before canceling.")

    selected_cancel_order_id = selected_cancel_order.get("Alpaca Order ID") or selected_cancel_order.get("Broker Order ID", "")
    if alpaca_cancel_preview.valid and not alpaca_state_health.stale:
        st.success("Paper cancel is ready to send.")

    cancel_base_disabled = (
        execution_mode != "paper"
        or not alpaca_status.connected
        or not enable_alpaca_paper_orders
        or alpaca_state_health.stale
        or not alpaca_cancel_preview.valid
    )
    alpaca_cancel_submit_disabled = (
        cancel_base_disabled
        or not alpaca_status.can_submit_orders
    )
    if st.button("Send Paper Cancel to Alpaca", disabled=alpaca_cancel_submit_disabled):
        try:
            cancel_result = alpaca_adapter.cancel_order(
                selected_cancel_order_id,
                expected_cancel_hash=alpaca_cancel_preview.preview_hash,
            )
            cancel_event = AuditEvent(
                event_type="alpaca_paper_cancel_submitted",
                message="Alpaca paper order cancel submitted through the gated adapter.",
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
            st.success("Paper cancel sent to Alpaca.")
        except Exception as exc:
            cancel_event = AuditEvent(
                event_type="alpaca_paper_cancel_blocked",
                message=str(exc),
                payload={
                    "broker_order_id": selected_cancel_order_id,
                    "cancel_preview_hash": alpaca_cancel_preview.preview_hash,
                },
            )
            st.session_state["session_audit_events"].append(cancel_event)
            if persist_audit_log:
                audit_store.append(cancel_event)
            st.error(f"Paper cancel blocked: {exc}")
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
        st.dataframe(pd.DataFrame(automation_decision_records(automation_decision)), use_container_width=True, hide_index=True)
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
        st.markdown("#### Automation checks")
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
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("#### Paper actions ready")
        st.dataframe(pd.DataFrame(automation_candidates), use_container_width=True, hide_index=True)
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
        st.markdown("#### Market data")
        st.dataframe(pd.DataFrame(market_freshness_rows), use_container_width=True, hide_index=True)
        st.markdown("#### Strategy state")
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
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("#### Paper account")
        st.dataframe(pd.DataFrame(account_health_rows), use_container_width=True, hide_index=True)
        st.markdown("#### Restart check")
        st.dataframe(pd.DataFrame(restart_rows), use_container_width=True, hide_index=True)
        st.markdown("#### Timer check")
        st.dataframe(pd.DataFrame(scheduler_rows), use_container_width=True, hide_index=True)
        st.markdown("#### Safety blocks")
        st.dataframe(pd.DataFrame(paper_automation_gate_rows), use_container_width=True, hide_index=True)
        st.markdown("#### Automation decision")
        st.dataframe(
            pd.DataFrame(
                automation_supervisor_dry_run_records(
                    candidates=automation_candidates,
                    readiness_rows=readiness_rows,
                    halt_rows=automation_halt_rows,
                )
            ),
            use_container_width=True,
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
        st.markdown("#### Saved automation checks")
        st.dataframe(pd.DataFrame(automation_evidence_records(recent_automation_snapshots)), use_container_width=True, hide_index=True)
    st.caption("This check records what automation sees. It does not submit, exit, or cancel broker orders.")

position_records = paper_broker.position_records()
order_records = paper_broker.order_records()
if show_portfolio_evidence:
    st.markdown("#### Practice decision log")
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
        st.dataframe(pd.DataFrame(shadow_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No practice decisions saved this session.")

with st.expander("Local simulator positions and orders", expanded=False):
    exec_tabs = st.tabs(["Positions", "Orders"])
    with exec_tabs[0]:
        if position_records:
            st.dataframe(pd.DataFrame(position_records), use_container_width=True, hide_index=True)
        else:
            st.caption("No open paper positions.")
    with exec_tabs[1]:
        if order_records:
            st.dataframe(pd.DataFrame(order_records), use_container_width=True, hide_index=True)
        else:
            st.caption("No paper orders submitted.")

session_audit_records = [JsonlAuditStore.event_to_record(event) for event in st.session_state["session_audit_events"]]
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
current_risk_halt_rows = risk_halt_records(
    monitoring_result=monitoring_result,
    broker_connected=alpaca_status.connected,
    broker_state_stale=alpaca_state_health.stale,
    automation_ready_rows=readiness_rows,
)
if show_portfolio_evidence:
    page_section("5. Saved records", "Detailed session records, local simulator results, Alpaca paper history, and export tools.")
    sub_section("5.1 This session")
    st.dataframe(pd.DataFrame(session_summary_records(session_snapshot)), use_container_width=True, hide_index=True)
    timeline_rows = session_timeline_records(session_audit_records)
    if timeline_rows:
        st.dataframe(pd.DataFrame(timeline_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No session events recorded yet.")

    sub_section("5.2 Local simulator results")
    st.dataframe(pd.DataFrame(paper_performance_records(session_snapshot)), use_container_width=True, hide_index=True)
    if st.button("Save Paper Performance Review"):
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

    sub_section("5.3 Alpaca paper order history")
    st.dataframe(pd.DataFrame(alpaca_paper_activity_records(session_snapshot)), use_container_width=True, hide_index=True)
    st.caption("This is saved Alpaca paper order history. It does not affect simulator cash or simulator equity.")

    sub_section("5.4 Risk details")
    st.dataframe(
        pd.DataFrame(
            daily_risk_records(
                local_order_records=order_records,
                tracked_alpaca_orders=st.session_state["tracked_alpaca_orders"],
                account_equity=account,
                session_pnl=session_pnl,
                portfolio_exposure=paper_positions_notional + alpaca_positions_notional,
                limits=risk_limits,
            )
        ),
        use_container_width=True,
        hide_index=True,
    )
    sub_section("5.5 Current blocks")
    st.dataframe(pd.DataFrame(current_risk_halt_rows), use_container_width=True, hide_index=True)

current_evidence_records = session_audit_records
if persist_audit_log:
    current_evidence_records = audit_store.read_recent(limit=500) or session_audit_records
readiness_event_types = {record.get("event_type", "") for record in current_evidence_records}
recent_automation_records = automation_store.read_recent(limit=100)
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
    live_mode_blocked=True,
)
current_approval_ledger_rows = approval_ledger_records(current_evidence_records)

if show_portfolio_evidence:
    sub_section("5.6 Activity log")
    st.dataframe(
        pd.DataFrame(events_to_records(st.session_state["session_audit_events"])),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Saved activity log", expanded=False):
        st.caption(f"Path: {audit_store.path}")
        durable_records = audit_store.read_recent(limit=50) if persist_audit_log else []
        if durable_records:
            st.dataframe(pd.DataFrame(durable_records), use_container_width=True, hide_index=True)
        else:
            st.caption("No saved activity records found, or saving is turned off.")
    with st.expander("Detailed records", expanded=False):
        st.dataframe(
            pd.DataFrame(evidence_dashboard_records(current_evidence_records, st.session_state["tracked_alpaca_orders"])),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("#### Review history")
        st.dataframe(pd.DataFrame(approval_ledger_summary_records(current_approval_ledger_rows)), use_container_width=True, hide_index=True)
        if current_approval_ledger_rows:
            st.dataframe(pd.DataFrame(current_approval_ledger_rows), use_container_width=True, hide_index=True)
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
            st.dataframe(pd.DataFrame(evidence_package_records(evidence_package)), use_container_width=True, hide_index=True)
            st.success(f"Records exported to {output_path}")

    with st.expander("Live trading checklist", expanded=False):
        st.dataframe(pd.DataFrame(current_pre_live_readiness_rows), use_container_width=True, hide_index=True)
        st.caption("This checklist is based on saved records. Some items stay blocked until the matching action is recorded.")
