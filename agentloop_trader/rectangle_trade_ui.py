"""Explicit research-to-queue review; only Save to Buy queue persists a plan."""
from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from agentloop_trader.buy_watchlist import BuyWatchlistStore, buy_watch_plan_id
from agentloop_trader.models import RiskLimits
from agentloop_trader.rectangle_data import DailyHistoryProvider, full_history_start
from agentloop_trader.rectangle_trade import (
    RECTANGLE_STRATEGY, build_rectangle_buy_plan, prepare_rectangle_trade, preview_rectangle_size,
)
from agentloop_trader.strategy_runtime import STRATEGY_TYPES, selected_strategy_result
from agentloop_trader.strategy_levels import build_buy_level_snapshot


def render_rectangle_trade_action(store, rectangle, watch, *, api_key=None, api_secret=None,
                                  defaults=None, risk_limits=None, buy_queue_path=None):
    identity = hashlib.sha256(str((rectangle.symbol,rectangle.interval,rectangle.start,rectangle.end,
                                   watch['source'],watch['anchor_close'])).encode()).hexdigest()[:12]
    draft_key = "rectangle_trade_draft"
    configured = bool(api_key and api_secret)
    if st.button("Create trade plan",key=f"rectangle_create_trade_{identity}",disabled=not configured,
                 help="Prepare a Daily entry plan for this saved range. Nothing is queued or ordered until Save to Buy queue."):
        st.session_state.pop(draft_key,None)
        try:
            research,_ = store.read_history(rectangle.symbol,watch['source'])
            if research is None:
                raise ValueError("Research history is missing. Run the scanner to refresh it.")
            with st.spinner("Preparing the trade plan with fresh Alpaca Daily candles..."):
                provider = DailyHistoryProvider(api_key,api_secret)
                frames = provider.fetch([rectangle.symbol],full_history_start())
                data = frames.get(rectangle.symbol)
                if data is None or data.empty:
                    raise ValueError("Alpaca did not return Daily history for this ticker.")
                data.attrs['symbol'] = rectangle.symbol
                draft = prepare_rectangle_trade(rectangle,watch['anchor_close'],watch['source'],research,data,
                                                defaults=defaults,risk_limits=risk_limits)
            st.session_state[draft_key] = {'identity':identity,**draft}
        except Exception as exc:
            # Provider errors must not echo credential-bearing text.
            message = str(exc) if isinstance(exc,(ValueError,PermissionError,RuntimeError)) else f"History could not be prepared ({type(exc).__name__})."
            for secret in (api_key,api_secret):
                if secret:
                    message = message.replace(secret,'[redacted]')
            st.error(message)
    if not configured:
        st.caption("Alpaca credentials are required to prepare a trade plan for the paper Buy queue.")
    draft = st.session_state.get(draft_key)
    if not draft or draft.get('identity') != identity:
        return
    st.markdown(f"**Trade plan · {rectangle.symbol}**")
    discovery = 'Weekly' if rectangle.interval == '1w' else 'Daily'
    st.caption(f"{discovery} setup / Daily entry · Alpaca IEX entry data · Preview only")
    options = {RECTANGLE_STRATEGY:'rectangle',**STRATEGY_TYPES}
    label = st.selectbox('Entry strategy',list(options),key=f'rectangle_trade_strategy_{identity}')
    settings = dict(draft['settings'])
    controls = st.columns(3)
    risk = controls[0].number_input('Risk per trade (%)',min_value=.01,max_value=100.,
        value=float(settings['risk_per_trade_pct']),step=.1,key=f'rectangle_trade_risk_{identity}')
    stop = controls[1].number_input('Initial stop (ATR)',min_value=.1,max_value=20.,
        value=float(settings['atr_stop_multiplier']),step=.1,key=f'rectangle_trade_stop_{identity}')
    exit_window = controls[2].number_input('Daily exit lookback',min_value=1,max_value=500,
        value=int(settings['exit_window']),key=f'rectangle_trade_exit_{identity}')
    settings.update(risk_per_trade_pct=risk,atr_stop_multiplier=stop,exit_window=exit_window)
    limits = RiskLimits(**draft['risk_limits'])
    context = settings['rectangle_context']
    st.caption(f"Frozen box: ${context['support']:,.2f}–${context['resistance']:,.2f} · "
               f"Daily candles through {context['review_data_date']}.")
    problem = ''
    if label == RECTANGLE_STRATEGY:
        try:
            estimate = preview_rectangle_size(settings,limits)
            st.dataframe(pd.DataFrame([
                {'Plan item':'Entry confirmation','Value':f"A new completed Daily close crosses above ${context['trigger']:,.2f}, after saving"},
                {'Plan item':'Maximum entry price','Value':f"${context['maximum_entry']:,.2f} (skip an extended breakout)"},
                {'Plan item':'Estimated initial stop at trigger','Value':f"${estimate.stop_loss:,.2f}"},
                {'Plan item':'Estimated shares at trigger','Value':f"{estimate.quantity:g}"},
                {'Plan item':'Estimated position value','Value':f"${estimate.estimated_notional:,.2f}"},
                {'Plan item':'Entry order','Value':'Limit at current price after confirmation'},
                {'Plan item':'Expires if untriggered','Value':'30 calendar days after saving'},
            ]),hide_index=True,width='stretch')
            if not context['support'] <= float(draft['frame'].Close.iloc[-1]) <= context['trigger']:
                problem = 'Price has already left this saved range. Refresh research or choose another existing strategy.'
            elif estimate.quantity <= 0:
                problem = 'The saved risk limits allow no position at this trigger.'
        except (ValueError,TypeError) as exc:
            problem = str(exc)
    else:
        st.info("This existing strategy follows its own BUY rules; it does not use the saved rectangle as its trigger. "
                "If its rules already pass, it can trade on the next worker check after saving.")
        try:
            result = selected_strategy_result(draft['frame'],{**settings,'strategy_type':options[label]},settings['account_size'],limits)
            snapshot = build_buy_level_snapshot(result['live'],interval='1d')
            st.dataframe(pd.DataFrame(snapshot['records']),hide_index=True,width='stretch')
        except (ValueError,TypeError) as exc:
            problem = str(exc)
    if risk > limits.max_risk_per_trade_pct:
        problem = f"Risk per trade exceeds the saved {limits.max_risk_per_trade_pct:g}% risk limit."
    if options[label] == 'rsi_scalp' and settings.get('rsi_stop_mode') == 'no_price_stop':
        problem = 'RSI no-price-stop mode is backtest-only and cannot be queued. Choose a price-protected strategy.'
    if problem:
        st.warning(problem)
    st.caption(f"Sizing is an estimate using ${settings['account_size']:,.0f}; the worker resizes using the actual paper account "
               f"and price when entry triggers. ATR stop, break-even at {settings['breakeven_after_r']:g}R, "
               f"trailing protection after {settings['trail_after_r']:g}R when enabled, "
               f"and the selected strategy exit apply. Rectangle strategy exit: prior {exit_window} Daily lows. "
               "Repeat after exit starts Off.")
    with st.expander('Saved risk limits and strategy inputs'):
        st.json({'risk_limits':draft['risk_limits'],'strategy_inputs':{
            k:v for k,v in settings.items() if k not in {'rectangle_context','rectangle_armed_at'}}})
    queue = BuyWatchlistStore(buy_queue_path)
    try:
        exists = any(p.plan_id == buy_watch_plan_id(rectangle.symbol,'1d',label) for p in queue.read())
    except RuntimeError as exc:
        st.error(str(exc))
        return
    if exists:
        st.info('This ticker/strategy already has a Buy queue plan. Manage it in Positions & Queue; remove it there before creating a replacement.')
    st.caption('Save to Buy queue arms this plan. When the worker and queued buys are enabled, a qualifying entry needs no further confirmation.')
    actions = st.columns(2)
    if actions[0].button('Save to Buy queue',type='primary',key=f'rectangle_arm_{identity}',disabled=bool(problem) or exists):
        try:
            reviewed = {**draft,'settings':settings}
            plan = build_rectangle_buy_plan(reviewed,strategy_label=label,strategy_type=options[label])
            queue.upsert(plan,allow_replace=False)
            st.success(f'{rectangle.symbol} saved to the Buy queue. Manage it in Positions & Queue. It will trade only when automation and its rules allow.')
        except (ValueError,RuntimeError) as exc:
            st.error(str(exc))
    if actions[1].button('Discard trade draft',key=f'rectangle_discard_{identity}'):
        st.session_state.pop(draft_key,None)
        st.rerun()
