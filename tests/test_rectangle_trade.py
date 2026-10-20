from dataclasses import asdict, replace
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from agentloop_trader.models import RiskLimits
from agentloop_trader.buy_watchlist import BuyWatchlistStore, buy_watch_plan_detail_records
from agentloop_trader.rectangle_scanner import detect_rectangles
from agentloop_trader.rectangle_trade import (
    RECTANGLE_STRATEGY, build_rectangle_buy_plan, prepare_rectangle_trade,
    evaluate_rectangle_strategy, rectangle_order_price_reason,
)
from agentloop_trader.strategy_runtime import selected_strategy_result, evaluate_exit_settings
from test_rectangle_scanner import box_bars


REVIEW = pd.Timestamp('2026-10-20 15:00Z')
ARM = REVIEW+pd.Timedelta(minutes=5)
CLOSE = pd.Timestamp('2026-10-20 22:00Z')


def reviewed_draft(interval='1d'):
    from test_rectangle_weekly import weekly_box
    frame = weekly_box(end='2026-10-19') if interval == '1w' else box_bars(end='2026-10-19')
    frame.attrs['symbol'] = 'BOX'
    row = detect_rectangles('BOX',frame,interval=interval,now=REVIEW)[0]
    # Research is frozen; execution is always completed Daily candles.
    anchor = float(frame.loc[row.end,'Close'])
    return prepare_rectangle_trade(row,anchor,'Alpaca IEX',frame,frame,now=REVIEW)


def breakout_data(draft, *, close=None):
    frame = draft['frame'].copy()
    trigger = draft['settings']['rectangle_context']['trigger']
    price = trigger+.1 if close is None else close
    frame.loc[pd.Timestamp('2026-10-20')] = [price-.1,price+.3,price-.5,price,1_000_000]
    return frame


@pytest.mark.parametrize('interval',['1d','1w'])
def test_draft_is_daily_for_both_discovery_timeframes_and_only_save_builds_armed_plan(interval):
    draft = reviewed_draft(interval)
    assert draft['settings']['rectangle_armed_at'] == ''
    assert draft['settings']['interval'] == '1d'
    assert draft['settings']['rectangle_context']['discovery_interval'] == interval
    plan = build_rectangle_buy_plan(draft,now=ARM)
    assert plan.enabled and not plan.repeat_after_exit
    assert plan.strategy_label == RECTANGLE_STRATEGY and plan.interval == '1d'
    assert plan.strategy_settings['rectangle_armed_at'] == ARM.isoformat()
    assert plan.price_data_source == 'Ticker (Alpaca)'
    assert draft['settings']['rectangle_armed_at'] == ''  # Draft not mutated by save.
    json.dumps(asdict(plan),allow_nan=False)


def test_new_completed_cross_produces_intent_without_hindsight_backtest():
    draft = reviewed_draft()
    plan = build_rectangle_buy_plan(draft,now=ARM)
    result = evaluate_rectangle_strategy(breakout_data(draft),plan.strategy_settings,100000,RiskLimits(),now=CLOSE)
    live = result['live']
    assert live['trade_intent'] is not None and live['trade_intent'].quantity > 0
    assert all(live['buy_requirements'].values())
    assert live['trade_intent'].stop_loss < live['last_p']
    assert result['stats'] is None and result['trade_log'] == [] and not live['backtest_available']
    assert plan.strategy_settings['rectangle_context'] == draft['settings']['rectangle_context']


@pytest.mark.parametrize('case',['unarmed','old_cross','wick_only','already_above','too_extended',
                                'basis_changed','missing_anchor','missing_session','wrong_ticker','wrong_interval','expired'])
def test_entry_blocks_invalid_late_stale_or_non_crossing_conditions(case):
    draft = reviewed_draft()
    plan = build_rectangle_buy_plan(draft,now=ARM)
    settings = dict(plan.strategy_settings)
    frame = breakout_data(draft)
    context = settings['rectangle_context']
    now = CLOSE
    if case == 'unarmed': settings['rectangle_armed_at'] = ''
    if case == 'old_cross': settings['rectangle_armed_at'] = CLOSE.isoformat()
    if case == 'wick_only':
        frame.iloc[-1,frame.columns.get_indexer(['Open','Low','Close'])] = context['trigger']-.1
    if case == 'already_above':
        frame.iloc[-2,frame.columns.get_indexer(['Open','High','Low','Close'])] = [context['trigger']+.1,context['trigger']+.5,context['trigger'],context['trigger']+.2]
        # Move the price-basis anchor to an earlier unchanged bar for this test.
        settings['rectangle_context'] = {**context,'anchor_date':frame.index[-3].strftime('%Y-%m-%d'),
                                        'anchor_close':float(frame.Close.iloc[-3])}
    if case == 'too_extended': frame = breakout_data(draft,close=context['maximum_entry']+1)
    if case == 'basis_changed': frame.loc[context['anchor_date'],['Open','High','Low','Close']] *= .5
    if case == 'missing_anchor': frame = frame.drop(pd.Timestamp(context['anchor_date']))
    if case == 'missing_session': frame = frame.drop(frame.index[-5])
    if case == 'wrong_ticker': frame.attrs['symbol'] = 'OTHER'
    if case == 'wrong_interval': settings['interval'] = '4h'
    if case == 'expired': settings['rectangle_armed_at'] = (CLOSE-pd.Timedelta(days=31)).isoformat()
    live = evaluate_rectangle_strategy(frame,settings,100000,now=now)['live']
    assert live['trade_intent'] is None
    assert not all(live['buy_requirements'].values())
    assert live['exit_level'] is not None and live['last_atr'] > 0  # Entry blocks cannot disable position exits.


def test_forming_candle_and_future_data_cannot_trigger_entry():
    draft = reviewed_draft()
    plan = build_rectangle_buy_plan(draft,now=ARM)
    frame = breakout_data(draft)
    live = evaluate_rectangle_strategy(frame,plan.strategy_settings,100000,now='2026-10-20 19:00Z')['live']
    assert live['trade_intent'] is None and live['last_completed_bar_at'].startswith('2026-10-19')
    changed = frame.copy()
    changed.loc[pd.Timestamp('2026-10-21')] = [900.,901.,899.,900.,1_000_000]
    before = evaluate_rectangle_strategy(frame,plan.strategy_settings,100000,now=CLOSE)['live']
    after = evaluate_rectangle_strategy(changed,plan.strategy_settings,100000,now=CLOSE)['live']
    assert before == after


def test_completed_support_loss_after_arming_invalidates_later_breakout():
    draft = reviewed_draft()
    context = draft['settings']['rectangle_context']
    settings = dict(build_rectangle_buy_plan(draft,now=ARM).strategy_settings)
    settings['rectangle_armed_at'] = '2026-10-15 15:00Z'
    frame = breakout_data(draft)
    frame.loc['2026-10-16',['Open','High','Low','Close']] = [context['support']-1,context['support']-.5,context['support']-2,context['support']-1]
    live = evaluate_rectangle_strategy(frame,settings,100000,now=CLOSE)['live']
    assert not live['buy_requirements']['Range support has held since arming']
    assert live['trade_intent'] is None


@pytest.mark.parametrize('case',['old_preview','new_session','outside_range','risk_excess','allowlist','unsupported','rsi_no_stop'])
def test_save_validation_cannot_arm_invalid_review(case):
    draft = reviewed_draft()
    now = ARM
    kwargs = {}
    if case == 'old_preview': now = REVIEW+pd.Timedelta(minutes=16)
    if case == 'new_session': now = CLOSE
    if case == 'outside_range':
        context = draft['settings']['rectangle_context']
        draft['frame'].iloc[-1,draft['frame'].columns.get_loc('Close')] = context['trigger']+1
    if case == 'risk_excess': draft['settings']['risk_per_trade_pct'] = 10
    if case == 'allowlist': draft['risk_limits']['allowed_symbols'] = ('OTHER',)
    if case == 'unsupported': kwargs = dict(strategy_label='Imaginary strategy',strategy_type='anything')
    if case == 'rsi_no_stop':
        kwargs = dict(strategy_label='RSI mean-reversion scalp',strategy_type='rsi_scalp')
        draft['settings']['rsi_stop_mode'] = 'no_price_stop'
    with pytest.raises(ValueError): build_rectangle_buy_plan(draft,now=now,**kwargs)


@pytest.mark.parametrize('label,strategy_type',[('Breakout continuation','breakout'),('Trendline breakout','trendline'),
    ('Trendline retest continuation','trendline_retest'),('Trend pullback continuation','pullback'),('RSI mean-reversion scalp','rsi_scalp')])
def test_existing_strategies_remain_selectable_and_use_own_rules(label,strategy_type):
    plan = build_rectangle_buy_plan(reviewed_draft(),strategy_label=label,strategy_type=strategy_type,now=ARM)
    assert plan.strategy_settings['strategy_type'] == strategy_type
    assert rectangle_order_price_reason(plan.strategy_settings,10000) == ''


def test_queue_save_can_atomically_refuse_replacement_without_rearming_existing_plan(tmp_path):
    store = BuyWatchlistStore(tmp_path/'queue.json')
    plan = build_rectangle_buy_plan(reviewed_draft(),now=ARM)
    saved = store.upsert(plan,allow_replace=False)
    store.update(plan.plan_id,enabled=False,order_sent_at=CLOSE.isoformat())
    before = store.path.read_bytes()
    with pytest.raises(ValueError,match='already exists'): store.upsert(plan,allow_replace=False)
    assert store.path.read_bytes() == before
    rows = buy_watch_plan_detail_records(saved)
    assert any(r['Input'] == 'Daily breakout trigger' for r in rows)


@pytest.mark.parametrize('offset,blocked',[(-.1,True),(0,True),(.1,False),(100,True)])
def test_actual_order_price_must_stay_above_trigger_and_below_entry_cap(offset,blocked):
    settings = build_rectangle_buy_plan(reviewed_draft(),now=ARM).strategy_settings
    assert bool(rectangle_order_price_reason(settings,settings['rectangle_context']['trigger']+offset)) is blocked


def test_prepare_reconciles_research_basis_and_rejects_stale_or_missing_execution_history():
    frame = box_bars(end='2026-10-19')
    row = detect_rectangles('BOX',frame,now=REVIEW)[0]
    stale = frame.iloc[:-1]
    with pytest.raises(ValueError,match='stale'): prepare_rectangle_trade(row,row.last_close,'Yahoo',frame,stale,now=REVIEW)
    split = frame.copy()
    split[['Open','High','Low','Close']] *= .5
    with pytest.raises(ValueError,match='price bases differ'): prepare_rectangle_trade(row,row.last_close,'Yahoo',frame,split,now=REVIEW)


def test_sizing_uses_current_account_and_risk_limits_not_preview_account():
    draft = reviewed_draft()
    plan = build_rectangle_buy_plan(draft,now=ARM)
    frame = breakout_data(draft)
    a = evaluate_rectangle_strategy(frame,plan.strategy_settings,100000,now=CLOSE)['live']['trade_intent']
    b = evaluate_rectangle_strategy(frame,plan.strategy_settings,10000,now=CLOSE)['live']['trade_intent']
    assert a.quantity > b.quantity*5
    limited = evaluate_rectangle_strategy(frame,plan.strategy_settings,100000,RiskLimits(max_quantity=2),now=CLOSE)['live']['trade_intent']
    assert limited.quantity == 2


def test_runtime_routing_and_expired_entry_do_not_disable_position_exit(monkeypatch):
    import agentloop_trader.rectangle_trade as module
    original = module.utc_now
    monkeypatch.setattr(module,'utc_now',lambda now=None: original(CLOSE if now is None else now))
    draft = reviewed_draft()
    plan = build_rectangle_buy_plan(draft,now=ARM)
    frame = breakout_data(draft)
    result = selected_strategy_result(frame,plan.strategy_settings,100000)
    assert result['live']['setup_type'] == 'rectangle'
    settings = {**plan.strategy_settings,'rectangle_armed_at':(CLOSE-pd.Timedelta(days=31)).isoformat(),
                'entry_reference_price':115.,'entry_stop_distance':5.,'auto_exit_enabled':True,
                'profit_protection_enabled':False}
    # A position with a current price under its protective stop must still exit.
    details = evaluate_exit_settings(settings,{'Symbol':'BOX','Quantity':1,'Average Entry':115.,'Market Value':105.},lambda *_:frame)
    assert details['ready'] is True


def test_queue_chart_and_buy_requirements_show_frozen_levels():
    from agentloop_trader.management_chart import build_management_chart
    from agentloop_trader.strategy_levels import build_buy_level_snapshot
    draft = reviewed_draft()
    plan = build_rectangle_buy_plan(draft,now=ARM)
    frame = breakout_data(draft)
    result = evaluate_rectangle_strategy(frame,plan.strategy_settings,100000,now=CLOSE)
    snapshot = build_buy_level_snapshot(result['live'],interval='1d')
    assert snapshot['next_buy_level'] == plan.strategy_settings['rectangle_context']['trigger']
    figure = build_management_chart(frame,plan.strategy_settings,result,title='Test queue chart',static_levels=[])
    names = {t.name for t in figure.data}
    assert {'Frozen rectangle resistance','Frozen Daily breakout trigger','Maximum rectangle entry'}.issubset(names)
