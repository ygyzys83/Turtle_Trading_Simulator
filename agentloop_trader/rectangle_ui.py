"""Small Streamlit surface for rectangle discovery and a research watchlist."""
from __future__ import annotations

import hashlib

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from agentloop_trader.rectangle_data import DailyHistoryProvider, UNIVERSE_LABEL, UNIVERSE_NOTE, history_bucket
from agentloop_trader.rectangle_jobs import start_range_scan
from agentloop_trader.rectangle_scanner import (
    DETECTOR_VERSION, Rectangle, filter_rectangles, rectangle_from_record, rectangle_records, refresh_saved_rectangle,
)
from agentloop_trader.rectangle_store import RectangleStore
from agentloop_trader.rectangle_timeframes import analysis_bars, latest_completed_date, TIMEFRAME_LABELS
from agentloop_trader.ui_theme import CHART_COLORS


def build_rectangle_chart(frame: pd.DataFrame, rectangle: Rectangle, *, show_volume=False):
    """Draw the detector's actual saved geometry without re-detecting anything."""
    context_start = pd.Timestamp(rectangle.prior_start) if rectangle.prior_start else pd.Timestamp(rectangle.start)-pd.DateOffset(months=6)-pd.Timedelta(days=7)
    frame = frame.loc[frame.index >= context_start]
    if frame.empty:
        raise ValueError("No cached candles are available for this rectangle.")
    show_volume = bool(show_volume and "Volume" in frame)
    fig = make_subplots(rows=2 if show_volume else 1, cols=1, shared_xaxes=True,
                        vertical_spacing=.04,
                        row_heights=[.78, .22] if show_volume else [1])
    x = frame.index.strftime("%Y-%m-%d").tolist()
    fig.add_trace(go.Candlestick(
        x=x, open=frame.Open, high=frame.High, low=frame.Low, close=frame.Close,
        name=f"Completed {TIMEFRAME_LABELS[rectangle.interval].lower()} candles", increasing_line_color=CHART_COLORS["entry"],
        decreasing_line_color=CHART_COLORS["sell"],
    ), row=1, col=1)
    fig.add_shape(type="rect", x0=rectangle.start, x1=rectangle.end,
                  y0=rectangle.support, y1=rectangle.resistance, fillcolor="rgba(91,143,249,.08)",
                  line=dict(color=CHART_COLORS["trend"], width=1), row=1, col=1)
    for level, label, color in ((rectangle.support, "Support", CHART_COLORS["entry"]),
                                (rectangle.resistance, "Resistance", CHART_COLORS["exit"])):
        fig.add_shape(type="rect", x0=rectangle.start, x1=x[-1],
                      y0=level-rectangle.tolerance, y1=level+rectangle.tolerance,
                      fillcolor="rgba(145,161,180,.10)", line_width=0, row=1, col=1)
        fig.add_trace(go.Scatter(x=[rectangle.start, x[-1]], y=[level, level],
                                 mode="lines", name=f"{label} ${level:,.2f}",
                                 line=dict(color=color, width=1.5)), row=1, col=1)
    for touches, level, label, color in (
        (rectangle.support_touches, rectangle.support, "Support tests", CHART_COLORS["entry"]),
        (rectangle.resistance_touches, rectangle.resistance, "Resistance tests", CHART_COLORS["exit"]),
    ):
        fig.add_trace(go.Scatter(x=list(touches), y=[level]*len(touches), mode="markers",
                                 name=label, marker=dict(size=7, color=color, symbol="circle-open")), row=1, col=1)
    if show_volume:
        colors = [CHART_COLORS["entry"] if c >= o else CHART_COLORS["sell"] for c, o in zip(frame.Close, frame.Open)]
        fig.add_trace(go.Bar(x=x, y=frame.Volume, name="Volume", marker_color=colors, opacity=.5), row=2, col=1)
    fig.update_layout(
        title=dict(text=""), height=590 if show_volume else 480,
        margin=dict(l=12, r=12, t=48, b=24),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, font=dict(size=10)),
        paper_bgcolor=CHART_COLORS["surface"], plot_bgcolor=CHART_COLORS["surface"],
        font=dict(color=CHART_COLORS["text"]), hovermode="x unified",
    )
    fig.update_xaxes(type="category", rangeslider_visible=False, nticks=9, showgrid=False,
                     categoryorder="array", categoryarray=x)
    fig.update_yaxes(gridcolor=CHART_COLORS["grid"], zeroline=False)
    fig.update_yaxes(tickprefix="$", row=1, col=1)
    return fig


@st.fragment(run_every=1.0)
def _scan_progress():
    job = st.session_state.get("rectangle_scan_job")
    if job is None:
        return
    snapshot = job.snapshot()
    if snapshot["done"]:
        st.session_state.pop("rectangle_scan_job", None)
        st.session_state["rectangle_scan_notice"] = snapshot
        st.rerun()
    st.progress(snapshot["progress"], text=snapshot["message"])
    if st.button("Cancel scan", key="rectangle_cancel"):
        job.cancel.set()
        st.caption("Canceling after the current download returns...")


def _table_selection(records, key):
    event = st.dataframe(
        pd.DataFrame(records), width="stretch", hide_index=True,
        on_select="rerun", selection_mode="single-row", key=key,
        column_config={
            "Support": st.column_config.NumberColumn(format="$%.2f"),
            "Resistance": st.column_config.NumberColumn(format="$%.2f"),
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Position in range (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Prior trend": st.column_config.TextColumn(help="Trend over the six months before this rectangle began. Up adds up to 10 ranking points; other labels do not exclude a range."),
        },
    )
    selection = event.selection.rows
    return selection[0] if selection and selection[0] < len(records) else 0


def _render_chart(store, rectangle, source, key, *, watch=None):
    frame, bucket = store.read_history(rectangle.symbol, source)
    if frame is None:
        st.warning("Cached chart history is missing. Run Scan for Trading Ranges to reload it.")
        return
    adjustment = ""
    try:
        if watch is not None:
            rectangle, adjustment = refresh_saved_rectangle(rectangle, watch["anchor_close"], frame)
        else:
            # Later/canceled scans can update history. Keep this result's as-of date.
            frame = frame.loc[frame.index <= pd.Timestamp(rectangle.end)]
            if frame.empty:
                raise ValueError("The saved scan's chart history is unavailable. Run another scan.")
            rectangle, adjustment = refresh_saved_rectangle(rectangle, rectangle.last_close, frame)
        frame = analysis_bars(frame, rectangle.interval)
        if frame.empty:
            raise ValueError("No completed candles are available for this rectangle.")
    except (ValueError, RuntimeError) as exc:
        st.warning(str(exc))
        return
    label = TIMEFRAME_LABELS[rectangle.interval]
    st.markdown(f"**{rectangle.symbol} · {label} · {rectangle.months:g}-month range · {rectangle.location}**")
    # Calculate from original-range candles so older saved research also reports
    # actual-box containment instead of its historical padded-zone value.
    range_closes = frame.loc[rectangle.start:rectangle.end, "Close"]
    box_containment = float(((range_closes >= rectangle.support) & (range_closes <= rectangle.resistance)).mean())
    st.caption(
        f"{rectangle.start} to {rectangle.end} · {len(rectangle.support_touches)} support tests · "
        f"{len(rectangle.resistance_touches)} resistance tests · "
        f"{box_containment:.1%} of closes inside the drawn box. "
        f"{label} candles through {frame.index[-1]:%Y-%m-%d} · {source}."
    )
    if frame.index[-1] < latest_completed_date(rectangle.interval):
        st.caption("Saved history may be out of date. Run a scan to refresh it.")
    if adjustment:
        st.caption(adjustment)
    if rectangle.prior_return_percent is not None:
        st.caption(f"Prior trend: {rectangle.prior_trend} · {rectangle.prior_return_percent:+.1f}% "
                   f"from {rectangle.prior_start} to {rectangle.prior_end}, before this range · "
                   f"+{rectangle.prior_trend_bonus:g} ranking points.")
    elif rectangle.prior_trend == "Insufficient history":
        st.caption("Prior trend: insufficient complete history for the six months before this range; no ranking bonus applied.")
    else:
        st.caption("Prior trend was not assessed for this saved range. Run a new scan for updated candidate rankings.")
    show_volume = st.toggle("Show volume", value=False, key=f"rectangle_volume_{key}")
    st.plotly_chart(build_rectangle_chart(frame, rectangle, show_volume=show_volume),
                    width="stretch", theme=None, config={"scrollZoom": True}, key=f"rectangle_chart_{key}")


def render_rectangle_scanner(api_key=None, api_secret=None, *, store_root="automation_logs/rectangle_research"):
    store = RectangleStore(store_root)
    st.markdown("#### Rectangle trading ranges")
    active = st.session_state.get("rectangle_scan_job") is not None
    label = st.radio("Timeframe", ["Daily", "Weekly"], horizontal=True,
                     key="rectangle_timeframe", disabled=active)
    interval = "1d" if label == "Daily" else "1w"
    st.caption(f"{UNIVERSE_LABEL} · {label} candles · 3–24 months · Research only")
    source = "Alpaca IEX" if api_key and api_secret else "Yahoo"
    st.caption(f"New scans use {source}. Sidebar ticker, strategy and interval do not affect this scan.")
    if st.button("Scan for Trading Ranges", type="primary", disabled=active):
        provider = DailyHistoryProvider(api_key, api_secret)
        st.session_state["rectangle_scan_job"] = start_range_scan(store, provider, interval=interval)
        st.session_state.pop("rectangle_scan_notice", None)
        st.rerun()
    if active:
        _scan_progress()
    notice = st.session_state.get("rectangle_scan_notice")
    if notice:
        if notice["error"]:
            st.error(notice["error"])
        elif notice["canceled"]:
            st.info(notice["message"])
    try:
        scan, all_watches = store.read_scan(interval), store.read("watchlist", {})
        results = [rectangle_from_record(row) for row in scan.get("rectangles", [])]
        originals = {key: rectangle_from_record(w["rectangle"]) for key, w in all_watches.items()}
        originals = {key: r for key, r in originals.items() if r.interval == interval}
        watches = {key: all_watches[key] for key in originals}
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Saved rectangle research could not be loaded: {exc}")
        return
    results_tab, watch_tab = st.tabs(["Scan results", f"Research watchlist ({len(watches)})"])
    with results_tab:
        if not scan:
            st.info(f"Click Scan for Trading Ranges to find {label.lower()} rectangles across Nasdaq stocks.")
        else:
            timestamp = pd.Timestamp(scan["scanned_at"]).tz_convert("America/Los_Angeles")
            if scan.get("detector_version", 1) < DETECTOR_VERSION:
                st.info("Saved results use earlier rectangle rules. Run Scan for Trading Ranges to apply "
                        "the current ranking and six-month prior-trend assessment.")
            st.caption(
                f"Last scan {timestamp:%b %d, %Y %I:%M %p %Z} · {scan['source']} · "
                f"{scan['successful']:,}/{scan['total']:,} histories checked · "
                f"{scan['skipped']:,} excluded by price/activity/history rules · "
                f"{len(scan['errors']):,} download/calculation problems."
            )
            st.caption(f"Completed {label.lower()} data dates: {scan['data_start']} to {scan['data_end']}.")
            controls = st.columns([1.2, 1.8, 1])
            location = controls[0].selectbox("Price location", ["All", "Near support", "Middle", "Near resistance"], key="rectangle_location")
            duration_key = f"rectangle_duration_{interval}"
            months = controls[1].slider("Range duration (months)", 3, 24, (3, 24), key=duration_key)
            only_new = controls[2].toggle("New candidates only", value=False, key="rectangle_only_new",
                                           help="Newly qualifying tickers compared with the previous scan using the same data source.")
            if only_new and scan.get("first_scan"):
                st.caption("This is the first scan for this data source; all matches are first discoveries.")
            filtered = filter_rectangles(results, months, location, set(scan["new_symbols"]) if only_new else None)
            st.caption(f"{len(filtered):,} matching tickers · ranked by rectangle quality + prior strength · click a row to view its chart.")
            if filtered:
                fingerprint = hashlib.sha256(str((interval, scan["scanned_at"], location, months, only_new)).encode()).hexdigest()[:12]
                index = _table_selection(rectangle_records(filtered), f"rectangle_results_{fingerprint}")
                selected = filtered[index]
                _render_chart(store, selected, scan["source"], "result")
                if store.watch_key(selected.symbol, interval) in watches:
                    st.caption("Saved to the research watchlist. Its original rectangle is preserved.")
                elif st.button("Save to Watchlist", key="rectangle_save", disabled=active):
                    try:
                        store.save_watch(selected, scan["source"])
                        st.rerun()
                    except (OSError, ValueError, RuntimeError) as exc:
                        st.error(f"Could not save the research candidate: {exc}")
            else:
                st.info("No rectangles match these filters." if results else "No qualifying rectangles found in this scan.")
            if scan["errors"]:
                with st.expander("Stocks that could not be checked"):
                    st.dataframe(pd.DataFrame(scan["errors"]), hide_index=True, width="stretch")
    with watch_tab:
        st.caption(f"Showing saved {label.lower()} ranges. Switch Timeframe to view the other watchlist. "
                   "Saving research does not enable a queued buy or change an exit plan. Prices refresh when you run a scan.")
        if not watches:
            st.info("Save a candidate from Scan results to track its original rectangle here.")
        else:
            records = []
            watch_keys = []
            for key, original in originals.items():
                watch_keys.append(key)
                watch = watches[key]
                symbol = original.symbol
                frame, bucket = store.read_history(symbol, watch["source"])
                status, price, as_of = "History unavailable", None, ""
                try:
                    if frame is not None:
                        current, _ = refresh_saved_rectangle(original, watch["anchor_close"], frame)
                        bars = analysis_bars(frame, original.interval)
                        price, as_of = current.last_close, bars.index[-1].strftime("%Y-%m-%d")
                        status = "Broke above" if current.location == "Above range" else "Broke below" if current.location == "Below range" else current.location
                        if bars.index[-1] < latest_completed_date(original.interval):
                            status += " · needs refresh"
                except (ValueError, RuntimeError):
                    status = "Price basis unavailable"
                records.append({"Ticker": symbol, "Timeframe": label, "Range (months)": original.months,
                                "Price": price, "Status": status, "Data date": as_of,
                                "Saved": pd.Timestamp(watch["saved_at"]).strftime("%Y-%m-%d")})
            key = hashlib.sha256(str((interval, [(r["Ticker"], r["Saved"]) for r in records])).encode()).hexdigest()[:10]
            index = _table_selection(records, f"rectangle_watches_{key}")
            symbol = records[index]["Ticker"]
            watch_key = watch_keys[index]
            _render_chart(store, originals[watch_key], watches[watch_key]["source"], "watch", watch=watches[watch_key])
            if st.button("Remove from Watchlist", key="rectangle_remove", disabled=active):
                try:
                    store.remove_watch(symbol, interval)
                    st.rerun()
                except (OSError, ValueError, RuntimeError) as exc:
                    st.error(f"Could not remove the research candidate: {exc}")
    with st.expander("Scan details"):
        st.caption(UNIVERSE_NOTE)
        if scan:
            st.caption(f"Nasdaq directory date: {scan['universe'].get('directory_date', 'Not supplied')}.")
        st.caption(
            "Fixed research rules: about 63–504 daily sessions or 13–104 completed trading weeks (3–24 months); "
            "at least two separated, confirmed tests at each boundary; "
            "at least 90% of closes inside the drawn support/resistance box; "
            "ranges with at least 95% containment receive a ranking bonus. "
            "Prior trend uses the six months strictly before each rectangle: Up requires at least a 10% advance, "
            "positive log-price slope and at least 0.30 trend-fit consistency. A qualifying uptrend earns up to "
            "10 extra ranking points based on the advance (capped at 30%) and consistency (full credit at 0.60). "
            "Down is the corresponding decline; other complete histories are Flat / mixed. "
            "Insufficient history earns no bonus. Prior trend never changes rectangle eligibility. "
            "Limited boundary drift and wick overshoots are allowed. "
            "Range width: 3–45% and at least three ATR on the selected timeframe. No prior uptrend is required. "
            "Weekly candles aggregate daily OHLCV using the Nasdaq trading calendar; unfinished weeks and weeks with missing sessions are excluded. "
            "Near support is the bottom quarter; near resistance is the top quarter. "
            "Basic liquidity floor: price $1 and 20-session average dollar volume $100,000 on the displayed feed. "
            "IEX activity is not consolidated-market volume. Ranking describes range structure, not the probability of a profitable trade."
        )
