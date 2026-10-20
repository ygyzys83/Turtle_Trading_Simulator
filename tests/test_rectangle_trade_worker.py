from dataclasses import replace
from types import SimpleNamespace

import pytest

from agentloop_trader.automation_runtime import AutomationControl
from agentloop_trader.brokers import AlpacaConfig
from agentloop_trader.buy_watchlist import BuyWatchlistStore
from agentloop_trader.rectangle_trade import build_rectangle_buy_plan
from agentloop_trader.worker import _send_watchlist_entries
from test_rectangle_trade import reviewed_draft,breakout_data,ARM,CLOSE


def worker_fixture(tmp_path,monkeypatch):
    import agentloop_trader.rectangle_trade as module
    original = module.utc_now
    monkeypatch.setattr(module,'utc_now',lambda now=None:original(CLOSE if now is None else now))
    monkeypatch.setattr('agentloop_trader.worker._market_is_open',lambda adapter:True)
    draft = reviewed_draft()
    plan = build_rectangle_buy_plan(draft,now=ARM)
    store = BuyWatchlistStore(tmp_path/'queue.json')
    store.upsert(plan,allow_replace=False)
    frame = breakout_data(draft)
    calls=[]
    events=[]
    def submit(intent,decision,**kwargs):
        calls.append((intent,decision,kwargs))
        return SimpleNamespace(id='fake-order',symbol='BOX',side='buy',qty=intent.quantity,
                               type='limit',limit_price=intent.limit_price,status='accepted',
                               submitted_at=CLOSE.isoformat(),filled_qty=0,filled_avg_price=None)
    adapter = SimpleNamespace(config=AlpacaConfig(api_key='fake',api_secret='fake',paper=True),
        account_records=lambda:[{'Field':'Portfolio Value','Value':'10000'},
                                {'Field':'Cash','Value':'10000'}, {'Field':'Last Equity','Value':'10000'}],
        submit_order=submit,position_records=lambda **kwargs:[],order_records=lambda **kwargs:[])
    control = AutomationControl(mode='Auto entries and exits',paper_orders_enabled=True,full_automation_enabled=True)
    common=(adapter,[],[],[],lambda *_:frame,SimpleNamespace(append=events.append),store)
    return plan,control,common,calls,events,store


def test_armed_plan_sends_one_fake_order_and_attaches_existing_exit_management(tmp_path,monkeypatch):
    plan,control,common,calls,events,store = worker_fixture(tmp_path,monkeypatch)
    price = plan.strategy_settings['rectangle_context']['trigger']+.2
    sent,tracked,message = _send_watchlist_entries(control,*common,latest_prices={'BOX':price})
    assert sent == 1 and len(calls) == 1
    intent = calls[0][0]
    assert intent.order_type == 'limit' and intent.limit_price == price
    assert intent.quantity < 20  # Resized to the fake $10k account, not the $100k review account.
    assert tracked[-1]['exit_settings']['auto_exit_enabled'] is True
    assert tracked[-1]['exit_settings']['rectangle_context'] == plan.strategy_settings['rectangle_context']
    assert tracked[-1]['exit_settings']['entry_stop_distance'] > 0
    assert store.read()[0].enabled is False and store.read()[0].order_sent_at
    assert _send_watchlist_entries(control,*common,latest_prices={'BOX':price})[0] == 0
    assert len(calls) == 1


@pytest.mark.parametrize('case',['too_extended','back_inside','queued_buys_off','kill_switch','existing_position','low_cash'])
def test_worker_rechecks_actual_price_and_existing_execution_gates(tmp_path,monkeypatch,case):
    plan,control,common,calls,events,store = worker_fixture(tmp_path,monkeypatch)
    context = plan.strategy_settings['rectangle_context']
    quote = context['trigger']+.2
    common = list(common)
    if case == 'too_extended': quote = context['maximum_entry']+.1
    if case == 'back_inside': quote = context['trigger']-.1
    if case == 'queued_buys_off': control = replace(control,full_automation_enabled=False)
    if case == 'kill_switch': control = replace(control,kill_switch_enabled=True)
    if case == 'existing_position': common[1] = [{'Symbol':'BOX','Quantity':1,'Market Value':100}]
    if case == 'low_cash':
        common[0].account_records = lambda:[{'Field':'Portfolio Value','Value':'10000'},
                                           {'Field':'Cash','Value':'0'}, {'Field':'Last Equity','Value':'10000'}]
    sent,_,message = _send_watchlist_entries(control,*common,latest_prices={'BOX':quote})
    assert sent == 0 and not calls
    assert store.read()[0].enabled is True
