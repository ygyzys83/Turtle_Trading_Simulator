from dataclasses import asdict

import pandas as pd
import pytest

from agentloop_trader.buy_watchlist import BuyWatchlistStore
from agentloop_trader.models import RiskLimits
from agentloop_trader.rectangle_scanner import detect_rectangles
from agentloop_trader.rectangle_store import RectangleStore
from agentloop_trader.rectangle_timeframes import latest_completed_date
from test_rectangle_scanner import box_bars


@pytest.mark.parametrize('interval,strategy',[('1d','Rectangle breakout'),('1w','Rectangle breakout'),('1d','Breakout continuation')])
def test_create_and_review_do_not_queue_only_explicit_save_arms(tmp_path,monkeypatch,interval,strategy):
    from streamlit.testing.v1 import AppTest
    from agentloop_trader.rectangle_data import DailyHistoryProvider
    from test_rectangle_weekly import weekly_box
    frame = weekly_box(end=latest_completed_date('1d').strftime('%Y-%m-%d')) if interval == '1w' else box_bars(end=latest_completed_date('1d').strftime('%Y-%m-%d'))
    row = detect_rectangles('BOX',frame,interval=interval)[0]
    store = RectangleStore(tmp_path/'research')
    store.write_history('BOX','Alpaca IEX',frame,'test')
    store.save_watch(row,'Alpaca IEX')
    research_before = store.read('watchlist',{})
    calls=[]
    def preview(self,symbols,start):
        calls.append(symbols)
        return {'BOX':frame}
    monkeypatch.setattr(DailyHistoryProvider,'fetch',preview)
    # Do not instantiate the real app, worker, or broker. Only this review surface
    # can write the isolated test queue on an explicit UI action.
    queue_path = tmp_path/'queue.json'
    script = f'''from agentloop_trader.rectangle_ui import render_rectangle_scanner
render_rectangle_scanner('test-key','test-secret',store_root={str(tmp_path/'research')!r},buy_queue_path={str(queue_path)!r})'''
    app = AppTest.from_string(script)
    app.session_state['rectangle_timeframe'] = 'Weekly' if interval == '1w' else 'Daily'
    app.run()
    assert not app.exception and not calls and not queue_path.exists()
    next(b for b in app.button if b.label=='Create trade plan').click().run()
    assert not app.exception
    assert calls == [['BOX']] and not queue_path.exists()
    assert any(('Weekly setup / Daily entry' if interval=='1w' else 'Daily setup / Daily entry') in c.value for c in app.caption)
    if strategy != 'Rectangle breakout':
        next(s for s in app.selectbox if s.label=='Entry strategy').set_value(strategy).run()
        assert not app.exception and not queue_path.exists()
    save = next(b for b in app.button if b.label=='Save to Buy queue')
    assert not save.disabled
    save.click().run()
    assert not app.exception
    plans = BuyWatchlistStore(queue_path).read()
    assert len(plans)==1 and plans[0].enabled and not plans[0].repeat_after_exit
    assert plans[0].strategy_settings['strategy_type']==('rectangle' if strategy=='Rectangle breakout' else 'breakout') and plans[0].interval=='1d'
    assert calls == [['BOX']]  # Saving/tweaking a reviewed plan never starts a research scan.
    assert store.read('watchlist',{}) == research_before
    # A later rerender sees the durable existing plan and cannot rearm/replace it.
    app.run()
    assert not app.exception
    assert next(b for b in app.button if b.label=='Save to Buy queue').disabled


def test_discarding_draft_never_queues_and_missing_credentials_do_not_fetch(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    from agentloop_trader.rectangle_data import DailyHistoryProvider
    frame = box_bars(end=latest_completed_date('1d').strftime('%Y-%m-%d'))
    row = detect_rectangles('BOX',frame)[0]
    store = RectangleStore(tmp_path/'research')
    store.write_history('BOX','Alpaca IEX',frame,'test')
    store.save_watch(row,'Alpaca IEX')
    calls=[]
    monkeypatch.setattr(DailyHistoryProvider,'fetch',lambda self,*args:calls.append(args) or {'BOX':frame})
    queue_path=tmp_path/'queue.json'
    script=f'''from agentloop_trader.rectangle_ui import render_rectangle_scanner
render_rectangle_scanner(store_root={str(tmp_path/'research')!r},buy_queue_path={str(queue_path)!r})'''
    app=AppTest.from_string(script).run()
    assert not app.exception
    assert next(b for b in app.button if b.label=='Create trade plan').disabled
    assert not calls and not queue_path.exists()
    script=script.replace('render_rectangle_scanner(store_root=',"render_rectangle_scanner('test-key','test-secret',store_root=")
    app=AppTest.from_string(script).run()
    next(b for b in app.button if b.label=='Create trade plan').click().run()
    assert not app.exception and not queue_path.exists()
    next(b for b in app.button if b.label=='Discard trade draft').click().run()
    assert not app.exception and not queue_path.exists()
