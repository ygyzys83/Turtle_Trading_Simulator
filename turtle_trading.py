import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agentloop_trader.audit import build_audit_events, events_to_records
from agentloop_trader.audit_store import JsonlAuditStore
from agentloop_trader.agents import build_trade_proposal, proposal_records
from agentloop_trader.automation import (
    AutomationDryRunStore,
    automation_decision_records,
    automation_evidence_records,
    automation_readiness_records,
    automation_supervisor_dry_run_records,
    build_automation_snapshot,
    paper_automation_candidate_records,
    evidence_dashboard_records,
    paper_automation_dry_run,
)
from agentloop_trader.backtest import simulate_turtle_strategy
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
    immutable_boundary_records,
    pre_live_readiness_report,
    production_readiness_checks,
)
from agentloop_trader.session_journal import (
    PaperSessionSnapshot,
    new_session_id,
    paper_performance_records,
    session_summary_records,
    session_timeline_records,
)
from agentloop_trader.shadow import record_shadow_decision, shadow_records

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

    data = data[required].dropna()
    if data.empty:
        raise ValueError(f"No usable rows for {clean}.")

    if interval == "4h":
        data = data.resample("4h").agg({
            "Close": "last",
            "High": "max",
            "Low": "min",
        }).dropna()

    data.attrs["symbol"] = clean
    return data


def metric_card(col, label, value, sub, color_class=""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


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


st.sidebar.title("Controls")
st.sidebar.markdown("#### Data")
data_source = st.sidebar.radio("Price data", ["Synthetic", "Stock via yfinance"], horizontal=True)
market_data = None
source_caption = "synthetic price data"
ticker = "SYNTH"

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

st.sidebar.markdown("#### Strategy")
account = st.sidebar.number_input("Account balance ($)", min_value=1000, max_value=10_000_000, value=50000, step=1000)
entry_w = st.sidebar.slider("Entry window (bars)", 10, 55, 20, step=5)
exit_w = st.sidebar.slider("Exit window (bars)", 5, 30, 10, step=5)
atr_mult = st.sidebar.slider("ATR stop multiplier", 1.0, 4.0, 2.0, step=0.5)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.5, 3.0, 1.0, step=0.5)
ma_w = st.sidebar.slider("MA trend filter (bars)", 50, 300, 200, step=50)

st.sidebar.markdown("#### Governance")
mode_options = {
    "Backtest only": "backtest_only",
    "Paper trading": "paper",
    "Shadow mode": "shadow",
    "Live with approval": "live_with_approval",
    "Automated live": "automated_live",
}
mode_label = st.sidebar.selectbox("Execution mode", list(mode_options.keys()), index=0)
execution_mode = mode_options[mode_label]
allowed_symbols_text = st.sidebar.text_input("Allowed symbols", value=ticker)
max_risk_limit = st.sidebar.slider("Max risk per trade (%)", 0.25, 5.0, 1.0, step=0.25)
max_notional_limit = st.sidebar.slider("Max position notional (%)", 5.0, 100.0, 25.0, step=5.0)
max_portfolio_exposure = st.sidebar.slider("Max portfolio exposure (%)", 10.0, 100.0, 75.0, step=5.0)
max_symbol_concentration = st.sidebar.slider("Max symbol concentration (%)", 5.0, 100.0, 35.0, step=5.0)
max_session_loss = st.sidebar.slider("Max session loss (%)", 0.5, 10.0, 2.0, step=0.5)
max_open_positions = st.sidebar.slider("Max open positions", 1, 20, 5, step=1)
kill_switch = st.sidebar.checkbox("Kill switch", value=False)
allowed_symbols = tuple(s.strip().upper() for s in allowed_symbols_text.split(",") if s.strip())
reset_paper_broker = st.sidebar.button("Reset paper broker")
emergency_disable_session = st.sidebar.button("Emergency disable session")
enable_alpaca_paper_orders = st.sidebar.checkbox("Enable Alpaca paper orders", value=False)
confirm_alpaca_paper_order = st.sidebar.checkbox("Confirm next Alpaca paper order", value=False)
confirm_alpaca_paper_cancel = st.sidebar.checkbox("Confirm next Alpaca paper cancel", value=False)
confirm_alpaca_paper_exit = st.sidebar.checkbox("Confirm next Alpaca paper exit", value=False)

st.sidebar.markdown("#### Evaluation")
run_walk_forward = st.sidebar.checkbox("Run out-of-sample evaluation", value=True)
train_fraction = st.sidebar.slider("Training split (%)", 55, 80, 65, step=5) / 100
run_parameter_loop = st.sidebar.checkbox("Run bounded parameter loop", value=False)
max_parameter_candidates = st.sidebar.slider("Parameter candidates", 4, 16, 8, step=4)

st.sidebar.markdown("#### Audit")
persist_audit_log = st.sidebar.checkbox("Persist audit log", value=True)
audit_log_path = st.sidebar.text_input("Audit log path", value="audit_logs/agentloop_audit.jsonl")
broker_state_path = st.sidebar.text_input("Broker state path", value="broker_state/alpaca_paper_orders.json")
automation_dry_run_path = st.sidebar.text_input("Automation dry-run path", value="automation_logs/paper_automation_dry_runs.jsonl")
run_manifest_path = st.sidebar.text_input("Run manifest path", value="audit_logs/run_manifests.jsonl")
evidence_export_path = st.sidebar.text_input("Evidence export path", value="audit_logs/latest_evidence_package.json")

if data_source == "Synthetic" and st.sidebar.button("Simulate new run", type="primary"):
    st.session_state["seed"] = np.random.randint(0, 100_000)
    st.session_state["selected_trade_idx"] = None

seed = st.session_state.get("seed", 42)
risk_dec = risk_pct / 100
if "paper_broker" not in st.session_state or st.session_state.get("paper_starting_cash") != account:
    st.session_state["paper_broker"] = PaperBroker(cash=float(account))
    st.session_state["paper_starting_cash"] = account
    st.session_state["paper_session_id"] = new_session_id()
    st.session_state["paper_session_started_at"] = pd.Timestamp.utcnow().isoformat()
    st.session_state["session_audit_events"] = []
    st.session_state["shadow_decisions"] = []
    st.session_state["last_audit_key"] = None
    st.session_state["armed_alpaca_preview_hash"] = None
    st.session_state["armed_alpaca_cancel_hash"] = None
    st.session_state["armed_alpaca_cancel_order_id"] = None
    st.session_state["armed_alpaca_exit_hash"] = None
    st.session_state["armed_alpaca_exit_symbol"] = None
    st.session_state["tracked_alpaca_orders"] = []
if reset_paper_broker:
    st.session_state["paper_broker"] = PaperBroker(cash=float(account))
    st.session_state["paper_starting_cash"] = account
    st.session_state["paper_session_id"] = new_session_id()
    st.session_state["paper_session_started_at"] = pd.Timestamp.utcnow().isoformat()
    st.session_state["session_audit_events"] = []
    st.session_state["shadow_decisions"] = []
    st.session_state["last_audit_key"] = None
    st.session_state["armed_alpaca_preview_hash"] = None
    st.session_state["armed_alpaca_cancel_hash"] = None
    st.session_state["armed_alpaca_cancel_order_id"] = None
    st.session_state["armed_alpaca_exit_hash"] = None
    st.session_state["armed_alpaca_exit_symbol"] = None
    st.session_state["tracked_alpaca_orders"] = []
    st.session_state["session_disabled"] = False
st.session_state.setdefault("shadow_decisions", [])
st.session_state.setdefault("paper_session_id", new_session_id())
st.session_state.setdefault("paper_session_started_at", pd.Timestamp.utcnow().isoformat())
st.session_state.setdefault("armed_alpaca_preview_hash", None)
st.session_state.setdefault("armed_alpaca_cancel_hash", None)
st.session_state.setdefault("armed_alpaca_cancel_order_id", None)
st.session_state.setdefault("armed_alpaca_exit_hash", None)
st.session_state.setdefault("armed_alpaca_exit_symbol", None)
st.session_state.setdefault("tracked_alpaca_orders", [])
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
        message="Emergency session disable was activated by the user.",
        payload={"source": "sidebar"},
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
    prices, smas, atrs, trade_log, live, stats, labels = simulate_turtle_strategy(
        account, entry_w, exit_w, atr_mult, risk_dec, ma_w, seed, market_data
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

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
        )
    except ValueError as exc:
        walk_forward_error = str(exc)

current_strategy_config = StrategyConfig(
    entry_window=entry_w,
    exit_window=exit_w,
    atr_stop_multiplier=atr_mult,
    risk_per_trade_pct=risk_pct,
    moving_average_window=ma_w,
)
parameter_candidates = []
recommended_candidate = None
parameter_loop_error = None
if run_parameter_loop:
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
st.caption(f"Governed turtle trend-following simulator - {source_caption}")
with st.expander("Production readiness posture", expanded=False):
    st.markdown("- Live capital path targets Alpaca, but live order submission remains disabled.")
    st.markdown("- Alpaca paper orders require paper mode, preflight pass, connected account, enable toggle, and confirmation toggle.")
    st.markdown("- Risk policy, kill switch, broker credentials, and execution code are not agent-modifiable.")
    st.markdown("- Unattended live trading requires paper, shadow, and manual-live evidence before activation.")
    st.dataframe(pd.DataFrame(production_readiness_checks()), use_container_width=True, hide_index=True)
    st.dataframe(pd.DataFrame(immutable_boundary_records()), use_container_width=True, hide_index=True)
    st.markdown("##### Broker state simulations")
    st.dataframe(pd.DataFrame(broker_state_simulation_records()), use_container_width=True, hide_index=True)

with st.expander("Run manifest", expanded=False):
    st.dataframe(pd.DataFrame(run_manifest_records([current_manifest_record])), use_container_width=True, hide_index=True)
    if st.button("Record Run Manifest"):
        manifest_store.append(current_run_manifest)
        manifest_event = AuditEvent(
            event_type="run_manifest_recorded",
            message="Run manifest recorded locally for this session.",
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
        st.markdown("##### Recent run manifests")
        st.dataframe(pd.DataFrame(run_manifest_records(recent_manifests)), use_container_width=True, hide_index=True)
    st.caption("Run manifests record local configuration and broker endpoint context. They do not contact Alpaca.")

c1, c2, c3, c4 = st.columns(4)
pnl_color = "pos" if stats["total_pnl"] >= 0 else "neg"
metric_card(c1, "Final equity", f"${stats['final_equity']:,}", f"Started ${account:,}")
metric_card(c2, "Total P&L", f"${stats['total_pnl']:,}", f"{stats['return_pct']}% return", pnl_color)
metric_card(c3, "Win rate", f"{stats['win_rate']}%", f"{stats['wins']}W / {stats['losses']}L of {stats['total_trades']} trades")
metric_card(c4, "Reward-to-risk", f"{stats['rr_ratio']}x", "Avg win / avg loss ratio")

c5, c6, c7 = st.columns(3)
metric_card(c5, "Max drawdown", f"{stats['max_drawdown_pct']}%", "Worst peak-to-trough equity decline")
metric_card(c6, "Profit factor", f"{stats['profit_factor']}x", "Gross wins / gross losses")
metric_card(c7, "Market exposure", f"{stats['exposure_pct']}%", "Share of bars spent in a trade")

st.markdown("##### Out-of-sample evaluation")
if walk_forward_result is None:
    if walk_forward_error:
        st.warning(walk_forward_error)
    else:
        st.caption("Out-of-sample evaluation is disabled.")
else:
    verdict_color = {
        "Pass": "#3B6D11",
        "Inconclusive": "#8A6D1D",
        "Needs review": "#A32D2D",
    }.get(walk_forward_result.verdict, "inherit")
    st.markdown(
        f"**Walk-forward verdict:** "
        f"<span style='color:{verdict_color};font-weight:600'>{walk_forward_result.verdict}</span>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        pd.DataFrame(walk_forward_records(walk_forward_result)),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Evaluation notes", expanded=False):
        st.markdown(
            f"Training bars: **{walk_forward_result.train_bars}**. "
            f"Out-of-sample bars: **{walk_forward_result.oos_bars}**. "
            f"Warmup bars supplied to the test segment: **{walk_forward_result.warmup_bars}**."
        )
        for reason in walk_forward_result.reasons:
            st.markdown(f"- {reason}")

st.markdown("##### Bounded agent loop")
if not run_parameter_loop:
    st.caption("Bounded parameter loop is disabled. Enable it in the sidebar to compare allowed strategy settings.")
elif parameter_loop_error:
    st.warning(parameter_loop_error)
else:
    st.markdown(f"**Recommendation:** {recommendation_summary(recommended_candidate)}")
    if recommended_candidate is not None:
        rec = recommended_candidate.config
        rec_cols = st.columns(4)
        metric_card(rec_cols[0], "Entry", rec.entry_window, "Bars")
        metric_card(rec_cols[1], "Exit", rec.exit_window, "Bars")
        metric_card(rec_cols[2], "ATR stop", rec.atr_stop_multiplier, "Multiplier")
        metric_card(rec_cols[3], "SMA filter", rec.moving_average_window, "Bars")
    st.dataframe(
        pd.DataFrame(candidate_records(parameter_candidates)),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("The loop may suggest only bounded strategy parameters. It cannot change risk limits, broker adapters, execution modes, credentials, or kill-switch behavior.")

sig = live["signal"]
if sig == "long":
    st.markdown('<span class="signal-long">ENTRY SIGNAL: BUY</span>', unsafe_allow_html=True)
elif sig == "exit":
    st.markdown('<span class="signal-exit">EXIT SIGNAL: SELL</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="signal-flat">NO SIGNAL - flat</span>', unsafe_allow_html=True)

st.markdown("##### The 5 rules - live values")
lp = round(float(live["last_p"]), 2)
dh = round(float(live["don_high"]), 2)
dl = round(float(live["don_low"]), 2)
ls = round(float(live["last_sma"]), 2) if live["last_sma"] else "n/a"
la = round(float(live["last_atr"]), 2) if live["last_atr"] else "n/a"
r1_status = "breakout confirmed" if lp > dh else "below channel"
r2_status = "upward filter passed" if live["sma_up"] else "downward no entry"

st.markdown(f"""
<div class="rule-box"><b>1. Entry:</b> Buy when price breaks above {entry_w}-bar high (<b>${dh}</b>). Current: <b>${lp}</b> - {r1_status}</div>
<div class="rule-box"><b>2. Filter:</b> {ma_w}-bar SMA = <b>${ls}</b>. Trend filter: {r2_status}</div>
<div class="rule-box"><b>3. Volatility:</b> ATR (14d) = <b>${la}</b>. Stop distance = <b>{atr_mult}x ATR = ${live['stop_from_entry']}</b></div>
<div class="rule-box"><b>4. Position size:</b> Risk {risk_pct}% of ${live['balance']:,.0f} = <b>${live['balance'] * risk_dec:,.0f}</b>. Position = <b>{live['pos_size']} shares</b></div>
<div class="rule-box"><b>5. Exit:</b> Sell when price touches {exit_w}-bar low (<b>${dl}</b>). Current: <b>${lp}</b> - {'exit triggered' if lp <= dl else 'holding'}</div>
""", unsafe_allow_html=True)

st.markdown("##### Research agent proposal")
proposal_summary = pd.DataFrame(proposal_records(trade_proposal))
st.dataframe(proposal_summary, use_container_width=True, hide_index=True)
st.markdown(f"**Thesis:** {trade_proposal.thesis.thesis}")
st.markdown(f"**Invalidation:** {trade_proposal.thesis.invalidation}")
with st.expander("Data basis", expanded=False):
    for item in trade_proposal.thesis.data_basis:
        st.markdown(f"- {item}")

st.markdown("##### Risk policy and preflight")
policy_tabs = st.tabs(["Policy", "Preflight"])
with policy_tabs[0]:
    st.dataframe(
        pd.DataFrame(risk_policy_records(risk_limits)),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("The agent loop may suggest only strategy parameters. It cannot modify risk policy, broker credentials, execution code, or kill-switch behavior.")
with policy_tabs[1]:
    preflight_color = "#3B6D11" if preflight_check.ready else "#A32D2D"
    st.markdown(
        f"**Preflight:** "
        f"<span style='color:{preflight_color};font-weight:600'>"
        f"{'READY' if preflight_check.ready else 'BLOCKED'}</span>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        pd.DataFrame(preflight_records(preflight_check)),
        use_container_width=True,
        hide_index=True,
    )
    if preflight_check.blocked_reasons:
        st.warning(" ".join(dict.fromkeys(preflight_check.blocked_reasons)))

st.markdown("##### Governed trade proposal")
if intent is None:
    st.info("No trade intent generated on the latest bar. The system remains flat.")
else:
    proposal_cols = st.columns(4)
    metric_card(proposal_cols[0], "Symbol", intent.symbol_clean, intent.side.upper())
    metric_card(proposal_cols[1], "Quantity", f"{intent.quantity:,}", intent.order_type)
    metric_card(proposal_cols[2], "Entry reference", f"${intent.entry_price:,.2f}", "Strategy reference price")
    metric_card(proposal_cols[3], "Stop loss", f"${intent.stop_loss:,.2f}", "Required by risk policy")
    st.caption(intent.rationale)

checks_df = pd.DataFrame([{"Check": name.replace("_", " ").title(), "Passed": passed} for name, passed in risk_check.checks.items()])
decision_color = "#3B6D11" if risk_check.approved else "#A32D2D"
st.markdown(
    f"**Risk decision:** <span style='color:{decision_color};font-weight:600'>"
    f"{'APPROVED' if risk_check.approved else 'REJECTED'}</span> "
    f"- **Execution mode:** {mode_label} - **Execution gate:** {execution_decision.reason}",
    unsafe_allow_html=True,
)
if risk_check.rejected_reasons:
    st.warning(" ".join(risk_check.rejected_reasons))
elif execution_decision.requires_manual_approval:
    st.info("The proposal passed risk checks, but this mode requires manual approval.")
if not checks_df.empty:
    st.dataframe(checks_df, use_container_width=True, hide_index=True)

st.markdown("##### Paper execution")
with st.expander("Broker adapters", expanded=False):
    st.dataframe(
        pd.DataFrame(broker_status_records(broker_statuses)),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Alpaca is the target external broker. Paper orders require preview, arm, confirmation, "
        "fresh broker state, and duplicate-exposure checks."
    )
    st.markdown("##### Alpaca config validation")
    st.dataframe(pd.DataFrame(alpaca_config_validation_records(alpaca_adapter.config)), use_container_width=True, hide_index=True)
    alpaca_tabs = st.tabs(["Alpaca Account", "Alpaca Positions", "Alpaca Orders"])
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
    if reconcile_rows:
        st.markdown("##### Alpaca position reconciliation")
        st.dataframe(pd.DataFrame(reconcile_rows), use_container_width=True, hide_index=True)
    exit_previews = build_exit_order_previews(alpaca_positions, alpaca_adapter.config)
    if exit_previews:
        st.markdown("##### Alpaca exit previews")
        st.dataframe(pd.DataFrame(exit_preview_records(exit_previews)), use_container_width=True, hide_index=True)
    if alpaca_state_health.reasons:
        st.warning(" ".join(alpaca_state_health.reasons))

broker_cols = st.columns(4)
metric_card(broker_cols[0], "Paper cash", f"${paper_broker.cash:,.2f}", "Session broker")
metric_card(broker_cols[1], "Open positions", f"{len(paper_broker.positions)}", "Paper portfolio")
metric_card(broker_cols[2], "Orders", f"{len(paper_broker.orders)}", "Submitted this session")
metric_card(
    broker_cols[3],
    "Execution status",
    "Ready" if preflight_check.ready else "Blocked",
    "Preflight passed" if preflight_check.ready else "Preflight blocked",
)
portfolio_cols = st.columns(3)
session_color = "pos" if session_pnl >= 0 else "neg"
metric_card(portfolio_cols[0], "Paper equity", f"${paper_equity:,.2f}", "Cash plus paper positions")
metric_card(portfolio_cols[1], "Session P&L", f"${session_pnl:,.2f}", "Since last reset", session_color)
metric_card(portfolio_cols[2], "Portfolio exposure", f"${paper_positions_notional:,.2f}", "Book value")

st.markdown("##### Session monitoring")
monitor_color = {"OK": "#3B6D11", "WARN": "#8A6D1D", "BREACH": "#A32D2D"}.get(monitoring_result.status, "inherit")
current_market_advisory = market_session_advisory()
st.markdown(
    f"**Monitoring status:** "
    f"<span style='color:{monitor_color};font-weight:600'>{monitoring_result.status}</span>",
    unsafe_allow_html=True,
)
st.dataframe(pd.DataFrame(monitoring_records(monitoring_result)), use_container_width=True, hide_index=True)
st.dataframe(pd.DataFrame([current_market_advisory]), use_container_width=True, hide_index=True)
st.markdown("##### Broker heartbeat and staleness policy")
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

can_submit = intent is not None and execution_mode == "paper"
submit_disabled = intent is None or not preflight_check.ready or execution_mode != "paper"
if st.button("Submit Paper Order", type="primary", disabled=submit_disabled):
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

alpaca_status = alpaca_adapter.status()
alpaca_preview = build_alpaca_order_preview(intent, execution_decision, alpaca_adapter.config)
tracked_alpaca_orders = st.session_state.get("tracked_alpaca_orders", [])
duplicate_alpaca_reasons = duplicate_exposure_reasons(intent, alpaca_positions)
open_order_reasons = open_order_exposure_reasons(intent, alpaca_orders)
duplicate_preview_submitted = preview_already_tracked(alpaca_preview.preview_hash, tracked_alpaca_orders)
armed_alpaca_hash = st.session_state.get("armed_alpaca_preview_hash")
alpaca_preview_armed = armed_alpaca_hash == alpaca_preview.preview_hash
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
    st.markdown("##### Alpaca paper preview")
    st.dataframe(pd.DataFrame(alpaca_preview_records(alpaca_preview)), use_container_width=True, hide_index=True)
    if alpaca_preview.blocked_reasons:
        st.warning(" ".join(alpaca_preview.blocked_reasons))
    if duplicate_alpaca_reasons:
        st.warning(" ".join(duplicate_alpaca_reasons))
    if open_order_reasons:
        st.warning(" ".join(open_order_reasons))
    if duplicate_preview_submitted:
        st.warning("This preview hash is already tracked from a prior Alpaca paper submission.")
    if alpaca_state_health.stale:
        st.warning("Alpaca broker state is stale; refresh positions/orders before submitting.")
    elif not duplicate_alpaca_reasons and not open_order_reasons and not duplicate_preview_submitted and alpaca_preview_armed:
        st.success("Alpaca paper order preview is armed and unchanged.")
    elif armed_alpaca_hash:
        st.warning("Alpaca paper order preview changed. Re-arm before submitting.")

arm_disabled = alpaca_base_disabled
if st.button("Arm Alpaca Paper Order", disabled=arm_disabled):
    st.session_state["armed_alpaca_preview_hash"] = alpaca_preview.preview_hash
    arm_event = AuditEvent(
        event_type="alpaca_paper_order_armed",
        message="Alpaca paper order preview armed for one matching submission.",
        payload={"preview_hash": alpaca_preview.preview_hash, **alpaca_preview.order},
    )
    st.session_state["session_audit_events"].append(arm_event)
    if persist_audit_log:
        audit_store.append(arm_event)
    st.rerun()

alpaca_submit_disabled = (
    alpaca_base_disabled
    or not alpaca_status.can_submit_orders
    or not alpaca_preview_armed
    or not confirm_alpaca_paper_order
)
if st.button("Submit Alpaca Paper Order", disabled=alpaca_submit_disabled):
    try:
        alpaca_order = alpaca_adapter.submit_order(
            intent,
            execution_decision,
            expected_preview_hash=armed_alpaca_hash,
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
            }
            st.session_state["tracked_alpaca_orders"].append(tracked_record)
            broker_state_store.upsert(tracked_record)
        st.session_state["armed_alpaca_preview_hash"] = None
        st.session_state["session_audit_events"].append(alpaca_event)
        if persist_audit_log:
            audit_store.append(alpaca_event)
        st.success("Alpaca paper order submitted.")
    except Exception as exc:
        st.session_state["armed_alpaca_preview_hash"] = None
        alpaca_event = AuditEvent(
            event_type="alpaca_paper_order_blocked",
            message=str(exc),
            payload={"symbol": intent.symbol_clean if intent else None, "preview_hash": alpaca_preview.preview_hash},
        )
        st.session_state["session_audit_events"].append(alpaca_event)
        if persist_audit_log:
            audit_store.append(alpaca_event)
        st.error(f"Alpaca paper order blocked: {exc}")

if tracked_alpaca_orders:
    st.markdown("##### Tracked Alpaca paper orders")
    tracked_rows = [
        alpaca_adapter.tracked_order_record(
            item.get("broker_order_id", ""),
            item.get("preview_hash", ""),
        )
        for item in tracked_alpaca_orders
    ]
    st.dataframe(pd.DataFrame(tracked_rows), use_container_width=True, hide_index=True)

if st.session_state.get("tracked_alpaca_orders"):
    tracked_refresh_rows = alpaca_adapter.refreshed_tracked_order_records(st.session_state["tracked_alpaca_orders"])
    lifecycle_alpaca_orders = {row.get("Broker Order ID", ""): row for row in alpaca_orders if row.get("Broker Order ID")}
    for row in tracked_refresh_rows:
        if row.get("Broker Order ID"):
            lifecycle_alpaca_orders[row["Broker Order ID"]] = row
    lifecycle_order_rows = list(lifecycle_alpaca_orders.values())
    lifecycle_rows = alpaca_order_lifecycle_records(st.session_state["tracked_alpaca_orders"], lifecycle_order_rows)
    refreshed_order_state = refresh_tracked_alpaca_orders(st.session_state["tracked_alpaca_orders"], lifecycle_order_rows)
    lifecycle_summary = alpaca_order_lifecycle_summary_records(
        refreshed_order_state
    )
    st.markdown("##### Alpaca paper order lifecycle")
    st.dataframe(pd.DataFrame(lifecycle_summary), use_container_width=True, hide_index=True)
    st.dataframe(pd.DataFrame(lifecycle_rows), use_container_width=True, hide_index=True)

    position_lifecycle_rows = alpaca_position_lifecycle_records(alpaca_positions, refreshed_order_state)
    if position_lifecycle_rows:
        st.markdown("##### Alpaca paper position lifecycle")
        st.dataframe(
            pd.DataFrame(alpaca_position_lifecycle_summary_records(position_lifecycle_rows)),
            use_container_width=True,
            hide_index=True,
        )
        st.dataframe(pd.DataFrame(position_lifecycle_rows), use_container_width=True, hide_index=True)
        st.caption("Alpaca paper position lifecycle is read-only. It connects filled tracked orders to current Alpaca Positions and exit-preview readiness.")

    refresh_state_disabled = not alpaca_status.connected or alpaca_state_health.stale
    if st.button("Refresh Alpaca Paper Order State", disabled=refresh_state_disabled):
        refreshed_orders = refreshed_order_state
        broker_state_store.replace_all(refreshed_orders)
        st.session_state["tracked_alpaca_orders"] = refreshed_orders
        refresh_event = AuditEvent(
            event_type="alpaca_paper_order_state_refreshed",
            message="Tracked Alpaca paper order state refreshed from Alpaca Orders.",
            payload={
                "tracked_orders": len(refreshed_orders),
                "lifecycle_summary": lifecycle_summary,
            },
        )
        st.session_state["session_audit_events"].append(refresh_event)
        if persist_audit_log:
            audit_store.append(refresh_event)
        st.rerun()
    st.caption("Refresh Alpaca Paper Order State reads Alpaca Orders and updates only the app's local tracking file.")

if exit_previews:
    st.markdown("##### Alpaca paper exit preview")
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
    st.dataframe(pd.DataFrame(alpaca_preview_records(alpaca_exit_preview)), use_container_width=True, hide_index=True)

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
        reason="Exit preview approved; paper submission remains manually gated.",
        risk_check=RiskCheckResult(approved=True, rejected_reasons=[], checks={"exit_position": True}),
    )
    exit_position_blockers = exit_position_reasons(alpaca_exit_preview, alpaca_positions)
    duplicate_exit_reasons = open_exit_order_reasons(alpaca_exit_preview, alpaca_orders)
    armed_exit_hash = st.session_state.get("armed_alpaca_exit_hash")
    armed_exit_symbol = st.session_state.get("armed_alpaca_exit_symbol")
    alpaca_exit_armed = armed_exit_hash == alpaca_exit_preview.preview_hash and armed_exit_symbol == exit_symbol

    if alpaca_exit_preview.blocked_reasons:
        st.warning(" ".join(alpaca_exit_preview.blocked_reasons))
    if exit_position_blockers:
        st.warning(" ".join(exit_position_blockers))
    if duplicate_exit_reasons:
        st.warning(" ".join(duplicate_exit_reasons))
    if alpaca_state_health.stale:
        st.warning("Alpaca broker state is stale; refresh positions/orders before exiting.")
    elif alpaca_exit_armed:
        st.success("Alpaca paper exit preview is armed and unchanged.")
    elif armed_exit_hash:
        st.warning("Alpaca paper exit preview changed. Re-arm before submitting.")

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
    if st.button("Arm Alpaca Paper Exit", disabled=exit_base_disabled):
        st.session_state["armed_alpaca_exit_hash"] = alpaca_exit_preview.preview_hash
        st.session_state["armed_alpaca_exit_symbol"] = exit_symbol
        exit_arm_event = AuditEvent(
            event_type="alpaca_paper_exit_armed",
            message="Alpaca paper exit preview armed for one matching submission.",
            payload={"preview_hash": alpaca_exit_preview.preview_hash, **alpaca_exit_preview.order},
        )
        st.session_state["session_audit_events"].append(exit_arm_event)
        if persist_audit_log:
            audit_store.append(exit_arm_event)
        st.rerun()

    alpaca_exit_submit_disabled = (
        exit_base_disabled
        or not alpaca_status.can_submit_orders
        or not alpaca_exit_armed
        or not confirm_alpaca_paper_exit
    )
    if st.button("Submit Alpaca Paper Exit", disabled=alpaca_exit_submit_disabled):
        try:
            alpaca_exit_order = alpaca_adapter.submit_order(
                selected_exit_intent,
                exit_decision,
                expected_preview_hash=armed_exit_hash,
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
            st.session_state["armed_alpaca_exit_hash"] = None
            st.session_state["armed_alpaca_exit_symbol"] = None
            st.session_state["session_audit_events"].append(exit_event)
            if persist_audit_log:
                audit_store.append(exit_event)
            st.success("Alpaca paper exit submitted.")
        except Exception as exc:
            st.session_state["armed_alpaca_exit_hash"] = None
            st.session_state["armed_alpaca_exit_symbol"] = None
            exit_event = AuditEvent(
                event_type="alpaca_paper_exit_blocked",
                message=str(exc),
                payload={"symbol": exit_symbol, "preview_hash": alpaca_exit_preview.preview_hash},
            )
            st.session_state["session_audit_events"].append(exit_event)
            if persist_audit_log:
                audit_store.append(exit_event)
            st.error(f"Alpaca paper exit blocked: {exc}")
    st.caption("Submit Alpaca Paper Exit contacts the Alpaca paper account. Stop for manual inspection before clicking it.")

cancelable_alpaca_orders = cancelable_alpaca_order_records(alpaca_orders)
if cancelable_alpaca_orders:
    st.markdown("##### Alpaca paper cancel preview")
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
    st.dataframe(pd.DataFrame(alpaca_cancel_preview_records(alpaca_cancel_preview)), use_container_width=True, hide_index=True)
    if alpaca_cancel_preview.blocked_reasons:
        st.warning(" ".join(alpaca_cancel_preview.blocked_reasons))
    if alpaca_state_health.stale:
        st.warning("Alpaca broker state is stale; refresh positions/orders before cancelling.")

    selected_cancel_order_id = selected_cancel_order.get("Broker Order ID", "")
    armed_cancel_hash = st.session_state.get("armed_alpaca_cancel_hash")
    armed_cancel_order_id = st.session_state.get("armed_alpaca_cancel_order_id")
    alpaca_cancel_armed = (
        armed_cancel_hash == alpaca_cancel_preview.preview_hash
        and armed_cancel_order_id == selected_cancel_order_id
    )
    if alpaca_cancel_armed:
        st.success("Alpaca paper cancel preview is armed and unchanged.")
    elif armed_cancel_hash:
        st.warning("Alpaca paper cancel preview changed. Re-arm before cancelling.")

    cancel_base_disabled = (
        execution_mode != "paper"
        or not alpaca_status.connected
        or not enable_alpaca_paper_orders
        or alpaca_state_health.stale
        or not alpaca_cancel_preview.valid
    )
    if st.button("Arm Alpaca Paper Cancel", disabled=cancel_base_disabled):
        st.session_state["armed_alpaca_cancel_hash"] = alpaca_cancel_preview.preview_hash
        st.session_state["armed_alpaca_cancel_order_id"] = selected_cancel_order_id
        cancel_arm_event = AuditEvent(
            event_type="alpaca_paper_cancel_armed",
            message="Alpaca paper cancel preview armed for one matching cancellation.",
            payload={"cancel_preview_hash": alpaca_cancel_preview.preview_hash, **alpaca_cancel_preview.cancel},
        )
        st.session_state["session_audit_events"].append(cancel_arm_event)
        if persist_audit_log:
            audit_store.append(cancel_arm_event)
        st.rerun()

    alpaca_cancel_submit_disabled = (
        cancel_base_disabled
        or not alpaca_status.can_submit_orders
        or not alpaca_cancel_armed
        or not confirm_alpaca_paper_cancel
    )
    if st.button("Cancel Alpaca Paper Order", disabled=alpaca_cancel_submit_disabled):
        try:
            cancel_result = alpaca_adapter.cancel_order(
                selected_cancel_order_id,
                expected_cancel_hash=armed_cancel_hash,
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
            st.session_state["armed_alpaca_cancel_hash"] = None
            st.session_state["armed_alpaca_cancel_order_id"] = None
            st.session_state["session_audit_events"].append(cancel_event)
            if persist_audit_log:
                audit_store.append(cancel_event)
            st.success("Alpaca paper order cancel submitted.")
        except Exception as exc:
            st.session_state["armed_alpaca_cancel_hash"] = None
            st.session_state["armed_alpaca_cancel_order_id"] = None
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
            st.error(f"Alpaca paper cancel blocked: {exc}")
    st.caption("Cancel Alpaca Paper Order contacts the Alpaca paper account. Use it only after the cancel preview is armed and confirmed.")

if intent is not None and execution_mode != "paper":
    st.caption("Switch execution mode to Paper trading to enable paper order submission.")
elif intent is not None and not preflight_check.ready:
    st.caption("Paper order submission is disabled until deterministic checks approve execution.")
elif can_submit:
    st.caption("Paper orders use the strategy reference price and never contact a live broker.")
if intent is not None:
    st.caption("Alpaca paper orders require paper mode, preflight pass, real ticker, connected Alpaca paper credentials, enable toggle, armed preview, and confirmation toggle.")

with st.expander("Paper automation dry-run", expanded=False):
    automation_decision = paper_automation_dry_run(
        intent=intent,
        risk_check=risk_check,
        preflight=preflight_check,
        broker_health=alpaca_state_health,
        duplicate_reasons=duplicate_alpaca_reasons + open_order_reasons,
        idempotency_blocked=duplicate_preview_submitted,
    )
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
    st.markdown("##### Paper automation readiness")
    readiness_rows = automation_readiness_records(
        broker_connected=alpaca_status.connected,
        broker_state_stale=alpaca_state_health.stale,
        manual_order_gate_enabled=enable_alpaca_paper_orders,
        kill_switch_enabled=effective_kill_switch,
        candidates=automation_candidates,
    )
    st.dataframe(
        pd.DataFrame(readiness_rows),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("##### Paper automation candidate queue")
    st.dataframe(pd.DataFrame(automation_candidates), use_container_width=True, hide_index=True)
    automation_halt_rows = risk_halt_records(
        monitoring_result=monitoring_result,
        broker_connected=alpaca_status.connected,
        broker_state_stale=alpaca_state_health.stale,
        automation_ready_rows=readiness_rows,
    )
    st.markdown("##### Paper automation supervisor dry-run")
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
    if st.button("Record Paper Automation Dry Run"):
        snapshot = build_automation_snapshot(
            session_id=st.session_state["paper_session_id"],
            candidates=automation_candidates,
            readiness=readiness_rows,
        )
        snapshot_record = automation_store.append(snapshot)
        automation_event = AuditEvent(
            event_type="paper_automation_dry_run_recorded",
            message="Paper automation dry-run snapshot recorded locally.",
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
    recent_automation_snapshots = automation_store.read_recent(limit=100)
    st.markdown("##### Automation evidence dashboard")
    st.dataframe(pd.DataFrame(automation_evidence_records(recent_automation_snapshots)), use_container_width=True, hide_index=True)
    st.caption("Dry-run only. This queue never submits, exits, or cancels broker orders.")

st.markdown("##### Shadow mode")
shadow_disabled = intent is None or execution_mode != "shadow"
if st.button("Record Shadow Decision", disabled=shadow_disabled):
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
    st.caption("Switch execution mode to Shadow mode to record would-have-traded decisions.")
elif intent is None:
    st.caption("No current trade intent is available to record in shadow mode.")
shadow_rows = shadow_records(st.session_state["shadow_decisions"])
if shadow_rows:
    st.dataframe(pd.DataFrame(shadow_rows), use_container_width=True, hide_index=True)
else:
    st.caption("No shadow decisions recorded this session.")

position_records = paper_broker.position_records()
order_records = paper_broker.order_records()
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
st.markdown("##### Paper session journal")
st.dataframe(pd.DataFrame(session_summary_records(session_snapshot)), use_container_width=True, hide_index=True)
timeline_rows = session_timeline_records(session_audit_records)
if timeline_rows:
    st.dataframe(pd.DataFrame(timeline_rows), use_container_width=True, hide_index=True)
else:
    st.caption("No session events recorded yet.")

st.markdown("##### Paper performance dashboard")
st.dataframe(pd.DataFrame(paper_performance_records(session_snapshot)), use_container_width=True, hide_index=True)
if st.button("Record Paper Performance Review"):
    performance_event = AuditEvent(
        event_type="paper_performance_reviewed",
        message="Paper performance dashboard was reviewed by the user.",
        payload={"session_id": st.session_state["paper_session_id"]},
    )
    st.session_state["session_audit_events"].append(performance_event)
    if persist_audit_log:
        audit_store.append(performance_event)
    st.rerun()
st.caption("Paper performance uses local paper broker records and tracked Alpaca paper state. It does not submit or cancel broker orders.")

st.markdown("##### Daily risk dashboard")
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
current_risk_halt_rows = risk_halt_records(
    monitoring_result=monitoring_result,
    broker_connected=alpaca_status.connected,
    broker_state_stale=alpaca_state_health.stale,
    automation_ready_rows=readiness_rows,
)
st.markdown("##### Risk halt reasons")
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

st.markdown("##### Audit log")
st.dataframe(
    pd.DataFrame(events_to_records(st.session_state["session_audit_events"])),
    use_container_width=True,
    hide_index=True,
)
with st.expander("Persistent audit log", expanded=False):
    st.caption(f"Path: {audit_store.path}")
    durable_records = audit_store.read_recent(limit=50) if persist_audit_log else []
    if durable_records:
        st.dataframe(pd.DataFrame(durable_records), use_container_width=True, hide_index=True)
    else:
        st.caption("No durable audit records found or persistence is disabled.")
with st.expander("Evidence dashboard", expanded=False):
    st.dataframe(
        pd.DataFrame(evidence_dashboard_records(current_evidence_records, st.session_state["tracked_alpaca_orders"])),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("##### Approval ledger")
    st.dataframe(pd.DataFrame(approval_ledger_summary_records(current_approval_ledger_rows)), use_container_width=True, hide_index=True)
    if current_approval_ledger_rows:
        st.dataframe(pd.DataFrame(current_approval_ledger_rows), use_container_width=True, hide_index=True)
    if st.button("Export Evidence Package"):
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
        st.success(f"Evidence package exported to {output_path}")

with st.expander("Pre-live readiness report", expanded=False):
    st.dataframe(pd.DataFrame(current_pre_live_readiness_rows), use_container_width=True, hide_index=True)
    st.caption("This report is evidence based. A check stays blocked until the matching event or reconciliation is recorded.")

selected_idx = st.session_state.get("selected_trade_idx", None)
selected_trade = trade_log[selected_idx] if selected_idx is not None and 0 <= selected_idx < len(trade_log) else None
st.plotly_chart(
    build_chart(prices, smas, atrs, entry_w, exit_w, ma_w, labels, trade_log, selected_trade),
    use_container_width=True,
    config={"scrollZoom": True},
)

st.markdown(f"##### All trades ({len(trade_log)} total) - click any row to highlight it on the chart above")
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
        st.markdown("##### Post-trade review")
        review = review_closed_trade(selected_trade, trade_proposal.thesis)
        st.dataframe(pd.DataFrame(review_records(review)), use_container_width=True, hide_index=True)
        with st.expander("Review lessons", expanded=False):
            for lesson in review.lessons:
                st.markdown(f"- {lesson}")
else:
    st.caption("No trades triggered in this simulation run.")
