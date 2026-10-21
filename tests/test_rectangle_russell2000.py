import csv
import io
from dataclasses import replace

import pandas as pd
import pytest

from agentloop_trader.rectangle_data import (
    DailyHistoryProvider, fetch_russell2000_universe, parse_russell2000_holdings,
)
from agentloop_trader.rectangle_jobs import ScanJob, run_range_scan
from agentloop_trader.rectangle_scanner import detect_rectangles
from agentloop_trader.rectangle_store import RectangleStore
from agentloop_trader.rectangle_timeframes import latest_completed_date
from agentloop_trader.rectangle_universes import universe_cache_name
from test_rectangle_research import FakeProvider, universe
from test_rectangle_universes import write_snapshot
from test_rectangle_weekly import weekly_box


def holdings(items=None, date=None, fund='iShares Russell 2000 ETF'):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([fund])
    writer.writerow(['Fund Holdings as of', date or pd.Timestamp.now(tz='UTC').strftime('%b %d, %Y')])
    writer.writerow([])
    writer.writerow(['Ticker', 'Name', 'Asset Class', 'Exchange', 'Price'])
    for ticker, asset, exchange, price in items or []:
        writer.writerow([ticker, ticker+' company', asset, exchange, price])
    return buffer.getvalue()


def test_parser_includes_nyse_american_share_classes_and_excludes_residuals():
    text = holdings([
        ('NAS','Equity','NASDAQ','10'), ('AMER','Equity','Nyse Mkt Llc','5'),
        ('AMEX','Equity','NYSE American','6'), ('MOG A','Equity','NYSE','100'),
        ('MOG.A','Equity','NYSE','100'), ('GEF-B','Equity','NYSE','50'),
        ('USD','Cash','-','100'), ('FUND','Money Market','-','1'),
        ('RTYZ6','Futures','Index And Options Market','2000'),
        ('CVR','Equity','NO MARKET (E.G. UNLISTED)','1.23'),
        ('OTC','Equity','Non-Nms Quotation Service (Nnqs)','0.21'),
        ('-','Equity','NASDAQ','0.00'), ('-','Equity','NO MARKET (E.G. UNLISTED)','0.00'),
    ])
    symbols, date = parse_russell2000_holdings('\ufeff'+text)
    assert symbols == ['AMER','AMEX','GEF.B','MOG.A','NAS']
    assert date == pd.Timestamp.now(tz='UTC').date().isoformat()


@pytest.mark.parametrize('bad', ['html','wrong_fund','missing_date','truncated','bad_exchange','bad_symbol','no_ticker','empty'])
def test_parser_rejects_bad_responses_without_hiding_malformed_equities(bad):
    text = holdings([('GOOD','Equity','NYSE','5')])
    if bad == 'html': text = '<html>Unavailable</html>'
    if bad == 'wrong_fund': text = holdings([('GOOD','Equity','NYSE','5')], fund='iShares Core S&P 500 ETF')
    if bad == 'missing_date': text = text.replace('Fund Holdings as of','Missing date')
    if bad == 'truncated': text += 'BAD,Bad company,Equity,NYSE\n'
    if bad == 'bad_exchange': text = holdings([('BAD','Equity','Unknown venue','5')])
    if bad == 'bad_symbol': text = holdings([('BAD/SYMBOL','Equity','NYSE','5')])
    if bad == 'no_ticker': text = holdings([('-','Equity','NASDAQ','5')])
    if bad == 'empty': text = holdings()
    with pytest.raises(ValueError): parse_russell2000_holdings(text)


def test_fetch_dispatch_count_freshness_and_response_size(monkeypatch):
    import agentloop_trader.rectangle_data as data
    items = [(f'T{i:04d}','Equity','NYSE','5') for i in range(1970)]
    content = holdings(items).encode()
    calls = []
    def download(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(content)
    monkeypatch.setattr(data, 'urlopen', download)
    result = data.fetch_universe('russell2000')
    assert result['universe_id'] == 'russell2000' and len(result['symbols']) == 1970
    assert result['source'] == data.RUSSELL2000_HOLDINGS_URL
    assert calls == [data.RUSSELL2000_HOLDINGS_URL]
    for count in (5, 1799, 2201):
        content = holdings([(f'T{i:04d}','Equity','NYSE','5') for i in range(count)]).encode()
        with pytest.raises(ValueError, match='count'): fetch_russell2000_universe()
    for days in (-20, 3):
        date = (pd.Timestamp.now(tz='UTC')+pd.Timedelta(days=days)).strftime('%b %d, %Y')
        content = holdings(items, date=date).encode()
        with pytest.raises(ValueError, match='stale'): fetch_russell2000_universe()
    content = b'x'*4_000_001
    with pytest.raises(ValueError, match='size'): fetch_russell2000_universe()


@pytest.mark.parametrize('interval', ['1d', '1w'])
def test_russell_scan_has_separate_snapshot_discovery_cache_and_shared_watchlist(tmp_path, interval):
    frame = weekly_box(end=latest_completed_date('1d').strftime('%Y-%m-%d'))
    store = RectangleStore(tmp_path)
    provider = FakeProvider({s:frame for s in ('BOTH','NAS','LARGE','SMALL')})
    for uid, symbols in [('nasdaq',['BOTH','NAS']), ('sp500',['BOTH','LARGE'])]:
        run_range_scan(store,provider,ScanJob(cancel=provider.cancel),universe(symbols),interval=interval,universe_id=uid)
    previous = {uid:store.read_scan(interval,uid) for uid in ('nasdaq','sp500')}
    job = ScanJob(cancel=provider.cancel)
    run_range_scan(store,provider,job,universe(['BOTH','SMALL']),interval=interval,universe_id='russell2000')
    assert not job.error
    result = store.read_scan(interval,'russell2000')
    assert set(result['new_symbols']) == {'BOTH','SMALL'}
    assert {r['symbol'] for r in result['rectangles']} == {'BOTH','SMALL'}
    assert provider.calls[-1][0] == ['SMALL']  # Overlap uses the same history cache.
    assert previous == {uid:store.read_scan(interval,uid) for uid in previous}
    assert universe_cache_name('russell2000') == 'universe_russell2000'
    assert store.read('universe_russell2000',{})['symbols'] == ['BOTH','SMALL']
    name = 'scan_russell2000' + ('_weekly' if interval == '1w' else '')
    assert (tmp_path/f'{name}.json').exists()
    row = detect_rectangles('BOTH',frame,interval=interval)[0]
    store.save_watch(row,'Alpaca IEX','nasdaq')
    store.save_watch(replace(row,support=row.support+.1),'Alpaca IEX','russell2000')
    key = store.watch_key('BOTH',interval)
    assert store.read('watchlist',{})[key]['universe_id'] == 'nasdaq'
    assert store.read('watchlist',{})[key]['rectangle']['support'] == row.support
    store.save_watch(replace(row,symbol='SMALL'),'Alpaca IEX','russell2000')
    assert store.read('watchlist',{})[store.watch_key('SMALL',interval)]['universe_id'] == 'russell2000'
    # A bad membership download must leave every saved scan unchanged.
    before = {uid:store.read_scan(interval,uid) for uid in ('nasdaq','sp500','russell2000')}
    store.write('universe_russell2000',{})
    def failure(): raise ValueError('Membership unavailable')
    job = ScanJob(cancel=provider.cancel)
    run_range_scan(store,provider,job,failure,interval=interval,universe_id='russell2000')
    assert job.error and before == {uid:store.read_scan(interval,uid) for uid in before}


def test_ui_russell_selection_is_explicit_and_shared_watchlist_provenance_is_correct(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import agentloop_trader.rectangle_ui as ui
    store = RectangleStore(tmp_path)
    frame = weekly_box(end=latest_completed_date('1d').strftime('%Y-%m-%d'))
    row = detect_rectangles('SMALL',frame)[0]
    write_snapshot(store,row,'russell2000')
    store.write_history('SMALL','Alpaca IEX',frame,'test')
    store.save_watch(row,'Alpaca IEX','russell2000')
    before = store.read('watchlist',{})
    calls = []
    def start(store,provider,**kwargs):
        calls.append(kwargs)
        return ScanJob(done=True,message='Test complete',progress=1.)
    def forbidden(*args,**kwargs): raise AssertionError('No downloads from switching universes.')
    monkeypatch.setattr(ui,'start_range_scan',start)
    monkeypatch.setattr(DailyHistoryProvider,'fetch',forbidden)
    script = f'from agentloop_trader.rectangle_ui import render_rectangle_scanner\nrender_rectangle_scanner(store_root={str(tmp_path)!r})'
    app = AppTest.from_string(script).run()
    app.selectbox(key='rectangle_universe').set_value('russell2000').run()
    assert not app.exception and not calls
    assert app.dataframe[0].value.Ticker.tolist() == ['SMALL']
    assert app.dataframe[1].value['Found in'].tolist() == ['Russell 2000']
    assert any('IWM' in c.value and 'proxy' in c.value for c in app.caption)
    for label in ('Higher lows','Tight near resistance','Supportive volume'):
        assert any(c.label == label for c in app.checkbox)
    assert app.selectbox(key='rectangle_location')
    next(b for b in app.button if b.label=='Scan for Trading Ranges').click().run()
    assert not app.exception and calls == [{'interval':'1d','universe_id':'russell2000'}]
    app.selectbox(key='rectangle_universe').set_value('sp500').run()
    assert not app.exception
    assert app.dataframe[0].value['Found in'].tolist() == ['Russell 2000']
    assert store.read('watchlist',{}) == before
