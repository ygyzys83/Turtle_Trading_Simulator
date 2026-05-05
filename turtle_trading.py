import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import yfinance as yf
except ImportError:
    yf = None

st.set_page_config(page_title="Turtle Trading Simulator", layout="wide")

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


# ── Price generation ──────────────────────────────────────────────────────────

def gen_prices(n=400, seed=None):
    rng = np.random.default_rng(seed)
    prices = [100.0]
    vol = 0.015
    for i in range(1, n):
        drift = 0.0003 + (0.0005 if i > n * 0.4 else 0)
        shock = (rng.random() - 0.48) * vol * 2
        prices.append(max(10.0, prices[-1] * (1 + drift + shock)))
    return np.array(prices)


# ── Indicators ────────────────────────────────────────────────────────────────

def calc_atr(prices, n=14, highs=None, lows=None):
    prices = np.asarray(prices, dtype=float)
    atrs = [None] * len(prices)
    if highs is not None and lows is not None:
        highs = np.asarray(highs, dtype=float)
        lows  = np.asarray(lows,  dtype=float)
        tr = np.zeros(len(prices))
        tr[0] = highs[0] - lows[0]
        for i in range(1, len(prices)):
            tr[i] = max(highs[i] - lows[i],
                        abs(highs[i] - prices[i - 1]),
                        abs(lows[i]  - prices[i - 1]))
        for i in range(n, len(prices)):
            atrs[i] = float(np.mean(tr[i - n + 1:i + 1]))
        return atrs
    for i in range(n, len(prices)):
        hi = prices[i - n + 1:i + 1] * 1.005
        lo = prices[i - n + 1:i + 1] * 0.995
        atrs[i] = float(np.mean(hi - lo))
    return atrs


def calc_sma(prices, n):
    smas = [None] * len(prices)
    for i in range(n - 1, len(prices)):
        smas[i] = float(np.mean(prices[i - n + 1:i + 1]))
    return smas


# ── Data fetch ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker, period, interval):
    if yf is None:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")
    clean = ticker.strip().upper()
    if not clean:
        raise ValueError("Enter a ticker symbol.")

    # yfinance has no native 4h interval — download 1h and resample
    fetch_interval = "1h" if interval == "4h" else interval

    kwargs = dict(tickers=clean, period=period, interval=fetch_interval,
                  auto_adjust=True, progress=False, threads=False, prepost=False)
    try:
        data = yf.download(**kwargs, multi_level_index=False)
    except TypeError:
        data = yf.download(**kwargs)
    if data is None or data.empty:
        raise ValueError(f"No price data returned for {clean}.")
    if isinstance(data.columns, pd.MultiIndex):
        levels = [list(map(str, data.columns.get_level_values(i)))
                  for i in range(data.columns.nlevels)]
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

    # Resample 1h → 4h
    if interval == "4h":
        data = data.resample("4h").agg({
            "Close": "last",
            "High":  "max",
            "Low":   "min",
        }).dropna()

    return data


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(account, entry_w, exit_w, atr_mult, risk_pct_dec, ma_w,
             seed=None, market_data=None):

    if market_data is None:
        N      = 400
        prices = gen_prices(N, seed)
        highs  = lows = None
        labels = [f"Day {i+1}" for i in range(N)]
    else:
        prices = market_data["Close"].to_numpy(dtype=float)
        highs  = market_data["High"].to_numpy(dtype=float)
        lows   = market_data["Low"].to_numpy(dtype=float)
        N      = len(prices)
        labels = market_data.index.strftime("%Y-%m-%d").tolist()

    min_bars = max(entry_w, exit_w, ma_w, 14) + 2
    if N < min_bars:
        raise ValueError(
            f"Need at least {min_bars} bars for these settings; got {N}.")

    atrs = calc_atr(prices, 14, highs, lows)
    smas = calc_sma(prices, ma_w)

    trade_log   = []
    in_trade    = False
    entry_price = stop_price = shares = entry_bar = 0
    balance     = account
    start       = max(entry_w, exit_w, ma_w)

    for i in range(start, N):
        p   = prices[i]
        sma = smas[i]
        atr = atrs[i]
        if sma is None or atr is None:
            continue

        don_high = float(np.max(prices[i - entry_w:i]))
        don_low  = float(np.min(prices[i - exit_w:i]))
        ma_up    = sma > (smas[i - 1] if smas[i - 1] is not None else sma)

        if not in_trade:
            if p > don_high and ma_up:
                stop = p - atr_mult * atr
                risk = p - stop
                sz   = int((balance * risk_pct_dec) / risk) if risk > 0 else 0
                if sz > 0:
                    in_trade    = True
                    entry_price = p
                    stop_price  = stop
                    shares      = sz
                    entry_bar   = i
        else:
            if p <= stop_price or p <= don_low:
                pnl = (p - entry_price) * shares
                balance += pnl
                trade_log.append({
                    "trade":      len(trade_log) + 1,
                    "entry_date": labels[entry_bar],
                    "exit_date":  labels[i],
                    "entry_bar":  entry_bar,
                    "exit_bar":   i,
                    "entry":      round(entry_price, 2),
                    "exit":       round(p, 2),
                    "shares":     shares,
                    "stop":       round(stop_price, 2),
                    "pnl":        round(pnl, 2),
                    "pct_acct":   round(pnl / account * 100, 2),
                })
                in_trade = False
            else:
                stop_price = max(stop_price, p - atr_mult * atr)

    last_p   = float(prices[-1])
    last_atr = atrs[-1]
    last_sma = smas[-1]
    prev_sma = next((smas[i] for i in range(len(smas)-2, -1, -1)
                     if smas[i] is not None), last_sma)
    sma_up   = bool(last_sma and prev_sma and last_sma > prev_sma)
    dh_last  = float(np.max(prices[-1 - entry_w:-1]))
    dl_last  = float(np.min(prices[-1 - exit_w:-1]))
    pos_size = int((balance * risk_pct_dec) / (atr_mult * last_atr)) \
               if last_atr else 0

    signal = "flat"
    if not in_trade and last_p > dh_last and sma_up:
        signal = "long"
    elif in_trade and (last_p <= stop_price or last_p <= dl_last):
        signal = "exit"

    live = dict(last_p=last_p, last_atr=last_atr, last_sma=last_sma,
                sma_up=sma_up, don_high=dh_last, don_low=dl_last,
                pos_size=pos_size,
                stop_from_entry=round(atr_mult * last_atr, 2) if last_atr else 0,
                balance=balance, signal=signal)

    wins      = [t for t in trade_log if t["pnl"] > 0]
    losses    = [t for t in trade_log if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trade_log)
    win_rate  = round(len(wins) / len(trade_log) * 100) if trade_log else 0
    avg_win   = round(sum(t["pnl"] for t in wins)   / len(wins),   2) if wins   else 0
    avg_loss  = round(sum(t["pnl"] for t in losses) / len(losses), 2) if losses else 0
    rr        = round(abs(avg_win / avg_loss), 2) if avg_loss else 0

    stats = dict(
        final_equity=round(balance), total_pnl=round(total_pnl),
        return_pct=round(total_pnl / account * 100, 2),
        win_rate=win_rate, wins=len(wins), losses=len(losses),
        total_trades=len(trade_log), rr_ratio=rr,
    )

    return prices, smas, atrs, trade_log, live, stats, labels


# ── Chart ─────────────────────────────────────────────────────────────────────

def build_chart(prices, smas, atrs, entry_w, exit_w, ma_w, labels, trade_log, selected_trade=None):
    N = len(prices)
    x = labels

    dh = [float(np.max(prices[i - entry_w:i])) if i >= entry_w else None
          for i in range(N)]
    dl = [float(np.min(prices[i - exit_w:i]))  if i >= exit_w  else None
          for i in range(N)]

    fig = go.Figure()

    # Base price lines — mode="lines" prevents any stray dots
    fig.add_trace(go.Scatter(
        x=x, y=prices.tolist(), name="Price",
        mode="lines",
        line=dict(color="#4C9BE8", width=1.5),
        hovertemplate="%{x}<br>Price: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=smas, name=f"{ma_w}d SMA",
        mode="lines",
        line=dict(color="#F0A830", width=1, dash="dot"),
        hovertemplate="%{x}<br>SMA: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=dh, name=f"{entry_w}d High",
        mode="lines",
        line=dict(color="#5DBF8A", width=1, dash="dash"),
        hovertemplate="%{x}<br>" + str(entry_w) + "d High: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=dl, name=f"{exit_w}d Low",
        mode="lines",
        line=dict(color="#E8645A", width=1, dash="dash"),
        hovertemplate="%{x}<br>" + str(exit_w) + "d Low: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=atrs, name="ATR (14d)",
        mode="lines",
        line=dict(color="#B07FE8", width=1, dash="dot"),
        yaxis="y2",
        hovertemplate="%{x}<br>ATR: $%{y:.2f}<extra></extra>",
    ))

    # All entry/exit markers — mode="markers" only, no connecting line
    if trade_log:
        fig.add_trace(go.Scatter(
            x=[t["entry_date"] for t in trade_log],
            y=[t["entry"]      for t in trade_log],
            name="Entry", mode="markers",
            marker=dict(symbol="triangle-up", size=10, color="#5DBF8A",
                        opacity=0.55, line=dict(color="#2a7a50", width=1)),
            hovertemplate="Entry #%{customdata}<br>%{x}: $%{y:.2f}<extra></extra>",
            customdata=[t["trade"] for t in trade_log],
        ))
        fig.add_trace(go.Scatter(
            x=[t["exit_date"] for t in trade_log],
            y=[t["exit"]      for t in trade_log],
            name="Exit", mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="#E8645A",
                        opacity=0.55, line=dict(color="#a03020", width=1)),
            hovertemplate="Exit #%{customdata}<br>%{x}: $%{y:.2f}<extra></extra>",
            customdata=[t["trade"] for t in trade_log],
        ))

    # Selected trade highlight
    if selected_trade is not None:
        t  = selected_trade
        ei = t["entry_bar"]
        xi = t["exit_bar"]

        # Shaded hold period
        band_x = x[ei:xi + 1]
        band_y = prices[ei:xi + 1].tolist()
        fill   = ("rgba(93,191,138,0.10)" if t["pnl"] >= 0
                  else "rgba(232,100,90,0.10)")
        fig.add_trace(go.Scatter(
            x=band_x + band_x[::-1],
            y=band_y + [t["entry"]] * len(band_y),
            fill="toself", fillcolor=fill,
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))

        # Large entry marker
        fig.add_trace(go.Scatter(
            x=[t["entry_date"]], y=[t["entry"]],
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=18, color="#5DBF8A",
                        line=dict(color="white", width=2)),
            text=[f"  BUY #{t['trade']}"],
            textposition="middle right",
            textfont=dict(color="#5DBF8A", size=12),
            hovertemplate=(f"Trade #{t['trade']} Entry<br>"
                           f"{t['entry_date']}: ${t['entry']:.2f}<extra></extra>"),
            showlegend=False,
        ))

        # Large exit marker
        fig.add_trace(go.Scatter(
            x=[t["exit_date"]], y=[t["exit"]],
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=18, color="#E8645A",
                        line=dict(color="white", width=2)),
            text=[f"  SELL #{t['trade']}"],
            textposition="middle right",
            textfont=dict(color="#E8645A", size=12),
            hovertemplate=(f"Trade #{t['trade']} Exit<br>"
                           f"{t['exit_date']}: ${t['exit']:.2f}<extra></extra>"),
            showlegend=False,
        ))

    fig.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(
            title=dict(text="Time",
                       font=dict(size=12, color="rgba(160,160,160,0.9)"),
                       standoff=10),
            showgrid=False, tickfont=dict(size=11),
            linecolor="rgba(128,128,128,0.2)",
            tickcolor="rgba(128,128,128,0.4)",
            type="category", nticks=10,
        ),
        yaxis=dict(
            tickprefix="$", tickfont=dict(size=11),
            gridcolor="rgba(128,128,128,0.15)",
            linecolor="rgba(128,128,128,0.2)",
            tickcolor="rgba(128,128,128,0.4)",
            zerolinecolor="rgba(128,128,128,0.2)",
        ),
        yaxis2=dict(
            title=dict(text="ATR", font=dict(size=11,
                       color="rgba(176,127,232,0.8)")),
            tickprefix="$", tickfont=dict(size=10,
                            color="rgba(176,127,232,0.8)"),
            linecolor="rgba(128,128,128,0.2)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(180,180,180,0.9)"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(30,30,30,0.85)",
                        font=dict(color="white")),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("Controls")
st.sidebar.markdown("#### Data")
data_source = st.sidebar.radio("Price data", ["Synthetic", "Stock via yfinance"],
                               horizontal=True)

market_data    = None
source_caption = "synthetic price data"

if data_source == "Stock via yfinance":
    ticker   = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()
    interval = st.sidebar.selectbox(
        "Interval", ["1d", "4h", "1h", "30m", "15m", "5m", "1m"], index=0)
    if interval == "1d":
        period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        period_index   = 5
    elif interval == "4h":
        period_options = ["1mo", "3mo", "6mo", "1y"]
        period_index   = 3
    elif interval == "1h":
        period_options = ["1mo", "3mo", "6mo", "1y"]
        period_index   = 3
    elif interval in ("30m", "15m"):
        period_options = ["1mo", "3mo", "6mo"]
        period_index   = 2
    else:
        period_options = ["1d", "5d", "1mo"]
        period_index   = 2
    period = st.sidebar.selectbox("History period", period_options,
                                  index=period_index)
    if st.sidebar.button("Refresh stock data", type="primary"):
        fetch_stock_data.clear()
    try:
        with st.spinner(f"Fetching {ticker}…"):
            market_data = fetch_stock_data(ticker, period, interval)
        source_caption = (f"{ticker} via yfinance ({period}, {interval}); "
                          f"latest bar {market_data.index[-1]}")
        st.sidebar.caption(
            f"Loaded {len(market_data):,} bars. "
            "Yahoo intraday data may be delayed.")
    except Exception as exc:
        st.error(f"Could not load yfinance data: {exc}")
        st.stop()

st.sidebar.markdown("#### Strategy")
account  = st.sidebar.number_input("Account balance ($)", min_value=1000,
                                   max_value=10_000_000, value=50000, step=1000)
entry_w  = st.sidebar.slider("Entry window (bars)", 10, 55, 20, step=5,
                              help="Buy when price breaks above N-bar high")
exit_w   = st.sidebar.slider("Exit window (bars)",   5, 30, 10, step=5,
                              help="Sell when price touches N-bar low")
atr_mult = st.sidebar.slider("ATR stop multiplier", 1.0, 4.0, 2.0, step=0.5,
                              help="Stop = entry − N × ATR")
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.5, 3.0, 1.0, step=0.5,
                              help="% of account risked per trade")
ma_w     = st.sidebar.slider("MA trend filter (bars)", 50, 300, 200, step=50,
                              help="Only enter if SMA is sloping up")

if data_source == "Synthetic" and st.sidebar.button("Simulate new run",
                                                     type="primary"):
    st.session_state["seed"] = np.random.randint(0, 100_000)
    st.session_state["selected_trade_idx"] = None   # clear stale selection

seed     = st.session_state.get("seed", 42)
risk_dec = risk_pct / 100

try:
    prices, smas, atrs, trade_log, live, stats, labels = simulate(
        account, entry_w, exit_w, atr_mult, risk_dec, ma_w, seed, market_data)
except ValueError as exc:
    st.error(str(exc))
    st.stop()


# ── Header ────────────────────────────────────────────────────────────────────

st.title("Turtle Trading Simulator")
st.caption(f"Classic 5-rule trend-following strategy — {source_caption}")

# ── Metrics ───────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)

def metric_card(col, label, value, sub, color_class=""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

pnl_color = "pos" if stats["total_pnl"] >= 0 else "neg"
metric_card(c1, "Final equity",   f"${stats['final_equity']:,}",
            f"Started ${account:,}")
metric_card(c2, "Total P&L",      f"${stats['total_pnl']:,}",
            f"{stats['return_pct']}% return", pnl_color)
metric_card(c3, "Win rate",       f"{stats['win_rate']}%",
            f"{stats['wins']}W / {stats['losses']}L of {stats['total_trades']} trades")
metric_card(c4, "Reward-to-risk", f"{stats['rr_ratio']}×",
            "Avg win / avg loss ratio")

# ── Signal ────────────────────────────────────────────────────────────────────

sig = live["signal"]
if sig == "long":
    sig_html = '<span class="signal-long">ENTRY SIGNAL: BUY</span>'
elif sig == "exit":
    sig_html = '<span class="signal-exit">EXIT SIGNAL: SELL</span>'
else:
    sig_html = '<span class="signal-flat">NO SIGNAL — flat</span>'
st.markdown(sig_html, unsafe_allow_html=True)
st.markdown("")

# ── Rules ─────────────────────────────────────────────────────────────────────

st.markdown("##### The 5 rules — live values")
lp = round(float(live["last_p"]),   2)
dh = round(float(live["don_high"]), 2)
dl = round(float(live["don_low"]),  2)
ls = round(float(live["last_sma"]), 2) if live["last_sma"] else "—"
la = round(float(live["last_atr"]), 2) if live["last_atr"] else "—"

r1_status = "✅ breakout confirmed" if lp > dh else "below channel"
r2_status = "✅ upward — filter passed" if live["sma_up"] else "⚠️ downward — no entry"

st.markdown(f"""
<div class="rule-box"><b>1. Entry:</b> Buy when price breaks above {entry_w}-bar high (<b>${dh}</b>). Current: <b>${lp}</b> — {r1_status}</div>
<div class="rule-box"><b>2. Filter:</b> {ma_w}-bar SMA = <b>${ls}</b>. Trend filter: {r2_status}</div>
<div class="rule-box"><b>3. Volatility:</b> ATR (14d) = <b>${la}</b>. Stop = <b>{atr_mult}× ATR = ${live['stop_from_entry']}</b> below entry</div>
<div class="rule-box"><b>4. Position size:</b> Risk {risk_pct}% of ${live['balance']:,.0f} = <b>${live['balance'] * risk_dec:,.0f}</b>. Position = <b>{live['pos_size']} shares</b></div>
<div class="rule-box"><b>5. Exit:</b> Sell when price touches {exit_w}-bar low (<b>${dl}</b>). Current: <b>${lp}</b> — {'⚠️ exit triggered' if lp <= dl else 'holding'}</div>
""", unsafe_allow_html=True)

# ── Chart ─────────────────────────────────────────────────────────────────────

# Read selected trade index from session state — this was written on the
# PREVIOUS rerun so the chart always reflects the correct selection.
selected_idx   = st.session_state.get("selected_trade_idx", None)
selected_trade = (trade_log[selected_idx]
                  if selected_idx is not None
                  and 0 <= selected_idx < len(trade_log)
                  else None)

st.plotly_chart(
    build_chart(prices, smas, atrs, entry_w, exit_w, ma_w, labels, trade_log, selected_trade),
    use_container_width=True,
    config={"scrollZoom": True},
)

# ── Trade log ─────────────────────────────────────────────────────────────────

st.markdown(
    f"##### All trades ({len(trade_log)} total) — "
    "click any row to highlight it on the chart above"
)

if trade_log:
    display_df = pd.DataFrame([{
        "#":          t["trade"],
        "Entry Date": t["entry_date"],
        "Exit Date":  t["exit_date"],
        "Entry $":    t["entry"],
        "Exit $":     t["exit"],
        "Shares":     t["shares"],
        "Stop $":     t["stop"],
        "P&L $":      t["pnl"],
        "% Account":  t["pct_acct"],
    } for t in trade_log]).set_index("#")

    def color_pnl(val):
        return "color: #3B6D11" if val > 0 else "color: #A32D2D"

    styled = display_df.style.map(color_pnl, subset=["P&L $", "% Account"])

    event = st.dataframe(
        styled,
        use_container_width=True,
        height=min(400, 38 + 35 * len(display_df)),
        on_select="rerun",
        selection_mode="single-row",
    )

    # Derive the new selection index from the widget return value
    rows    = (event.selection.rows
               if event and hasattr(event, "selection")
               and event.selection else [])
    new_idx = rows[0] if rows else None

    # Only trigger a rerun if the selection actually changed
    if new_idx != st.session_state.get("selected_trade_idx"):
        st.session_state["selected_trade_idx"] = new_idx
        st.rerun()

    # Summary line beneath the table
    if selected_trade is not None:
        color   = "#3B6D11" if selected_trade["pnl"] >= 0 else "#A32D2D"
        pnl_str = f"${selected_trade['pnl']:,.2f}"
        st.markdown(
            f"**Selected — Trade #{selected_trade['trade']}:** &nbsp;"
            f"Entered {selected_trade['entry_date']} @ ${selected_trade['entry']} → "
            f"Exited {selected_trade['exit_date']} @ ${selected_trade['exit']} "
            f"&nbsp;·&nbsp; "
            f"P&L: <span style='color:{color};font-weight:600'>{pnl_str}</span> "
            f"({selected_trade['pct_acct']}% of account)",
            unsafe_allow_html=True,
        )
    else:
        st.caption("No trade selected. Click a row to highlight it on the chart.")

else:
    st.caption("No trades triggered in this simulation run.")