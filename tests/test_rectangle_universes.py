import csv
import io
from dataclasses import replace

import pandas as pd
import pytest

from agentloop_trader.rectangle_data import parse_sp500_holdings, fetch_sp500_universe
from agentloop_trader.rectangle_jobs import ScanJob, run_range_scan
from agentloop_trader.rectangle_scanner import DETECTOR_VERSION, detect_rectangles, rectangle_from_record
from agentloop_trader.rectangle_store import RectangleStore
from agentloop_trader.rectangle_timeframes import latest_completed_date
from agentloop_trader.rectangle_universes import validate_universe_id, universe_cache_name
from test_rectangle_research import FakeProvider, universe
from test_rectangle_weekly import weekly_box


def holdings(items, date=None):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(['iShares Core S&P 500 ETF'])
    writer.writerow(['Fund Holdings as of', date or pd.Timestamp.now(tz='UTC').strftime('%b %d, %Y')])
    writer.writerow([])
    writer.writerow(['Ticker', 'Name', 'Asset Class', 'Exchange'])
    for ticker, asset, exchange in items:
        writer.writerow([ticker, ticker+' company', asset, exchange])
    return buffer.getvalue()


def test_sp500_parser_filters_nonstocks_and_preserves_share_classes():
    text = holdings([('AAPL','Equity','NASDAQ'), ('JPM','Equity','NYSE'),
                     ('BRK B','Equity','NYSE'), ('BF-B','Equity','NYSE'),
                     ('CBOE','Equity','Cboe BZX'), ('BRK.B','Equity','NYSE'),
                     ('GOOG','Equity','NASDAQ'), ('GOOGL','Equity','NASDAQ'),
                     ('USD','Cash','-'), ('ESZ6','Futures','Index And Options Market'),
                     ('FUND','Money Market','-'), ('DEAD','Equity','NO MARKET (E.G. UNLISTED)')])
    symbols, date = parse_sp500_holdings('\ufeff'+text)
    assert symbols == ['AAPL','BF.B','BRK.B','CBOE','GOOG','GOOGL','JPM']
    assert date == pd.Timestamp.now(tz='UTC').date().isoformat()


@pytest.mark.parametrize('bad', ['html','wrong_fund','missing_date','bad_symbol','bad_exchange','truncated'])
def test_sp500_parser_rejects_unrecognized_or_partial_responses(bad):
    text = holdings([('AAPL','Equity','NASDAQ')])
    if bad == 'html': text = '<html>Unavailable</html>'
    if bad == 'wrong_fund': text = text.replace('iShares Core S&P 500 ETF','A different fund')
    if bad == 'missing_date': text = text.replace('Fund Holdings as of','Missing date')
    if bad == 'bad_symbol': text = holdings([('BAD/SYMBOL','Equity','NYSE')])
    if bad == 'bad_exchange': text = holdings([('AAPL','Equity','Unknown venue')])
    if bad == 'truncated': text += 'MSFT,Microsoft,Equity\n'
    with pytest.raises(ValueError): parse_sp500_holdings(text)


def test_sp500_fetch_validates_full_count_and_freshness_without_silent_fallback(monkeypatch):
    import agentloop_trader.rectangle_data as data
    content = holdings([(f'T{i:03d}','Equity','NYSE') for i in range(500)])
    requests = []
    def download(request, timeout):
        requests.append(request.full_url)
        return io.BytesIO(content.encode())
    monkeypatch.setattr(data, 'urlopen', download)
    result = fetch_sp500_universe()
    assert len(result['symbols']) == 500 and result['universe_id'] == 'sp500'
    assert requests == [data.SP500_HOLDINGS_URL]
    content = holdings([('AAPL','Equity','NASDAQ')])
    with pytest.raises(ValueError, match='count'): fetch_sp500_universe()
    content = holdings([(f'T{i:03d}','Equity','NYSE') for i in range(500)],
                       (pd.Timestamp.now(tz='UTC')-pd.Timedelta(days=20)).strftime('%b %d, %Y'))
    with pytest.raises(ValueError, match='stale'): fetch_sp500_universe()


@pytest.mark.parametrize('interval', ['1d','1w'])
def test_separate_universe_snapshots_keep_legacy_nasdaq_files_intact(tmp_path, interval):
    store = RectangleStore(tmp_path)
    nasdaq = {'interval':interval,'rectangles':[],'source':'legacy'}
    store.write_scan(interval, nasdaq)
    path = tmp_path / ('scan.json' if interval == '1d' else 'scan_weekly.json')
    before = path.read_bytes()
    sp500 = {'interval':interval,'universe_id':'sp500','rectangles':[],'source':'new'}
    store.write_scan(interval, sp500, 'sp500')
    assert store.read_scan(interval) == nasdaq
    assert store.read_scan(interval, 'sp500') == sp500
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='universe'): store.write_scan(interval, nasdaq, 'sp500')
    store.write('scan_sp500' if interval == '1d' else 'scan_sp500_weekly', nasdaq)
    with pytest.raises(ValueError, match='universe'): store.read_scan(interval, 'sp500')


@pytest.mark.parametrize('bad', ['all','../nasdaq','sp-500',''])
def test_unknown_universe_ids_are_rejected_before_storage_or_scan(tmp_path, bad):
    with pytest.raises(ValueError): validate_universe_id(bad)
    with pytest.raises(ValueError): RectangleStore(tmp_path).read_scan('1d',bad)


def test_universes_and_timeframes_have_independent_discovery_and_shared_history(tmp_path):
    frame = weekly_box(end=latest_completed_date('1d').strftime('%Y-%m-%d'))
    provider = FakeProvider({s:frame for s in ['BOTH','NAS','NYSE']})
    store = RectangleStore(tmp_path)
    for interval in ['1d','1w']:
        for uid, symbols in [('nasdaq',['BOTH','NAS']),('sp500',['BOTH','NYSE'])]:
            job = ScanJob(cancel=provider.cancel)
            run_range_scan(store,provider,job,universe(symbols),interval=interval,universe_id=uid)
            assert not job.error
            scan = store.read_scan(interval,uid)
            assert set(scan['new_symbols']) == set(symbols)
            assert {r['symbol'] for r in scan['rectangles']} == set(symbols)
            assert scan['universe_id'] == uid
    assert provider.calls[0][0] == ['BOTH','NAS']
    assert provider.calls[1][0] == ['NYSE']  # The overlap reuses daily cache across universes.
    assert len(provider.calls) == 2
    preserved = {i:store.read_scan(i,'sp500') for i in ['1d','1w']}
    run_range_scan(store,provider,ScanJob(cancel=provider.cancel),universe(['BOTH','NAS']),universe_id='nasdaq')
    assert store.read_scan('1d')['new_symbols'] == []
    assert preserved == {i:store.read_scan(i,'sp500') for i in ['1d','1w']}
    assert store.read('universe',{})['symbols'] == ['BOTH','NAS']
    assert store.read('universe_sp500',{})['symbols'] == ['BOTH','NYSE']


def test_watchlist_is_shared_and_original_range_is_not_duplicated_by_universe(tmp_path):
    store = RectangleStore(tmp_path)
    frame = weekly_box(end=latest_completed_date('1d').strftime('%Y-%m-%d'))
    row = detect_rectangles('BOTH',frame)[0]
    store.save_watch(row,'Alpaca IEX')
    original = store.read('watchlist',{})['BOTH']
    store.save_watch(replace(row,support=row.support+.1),'Alpaca IEX','sp500')
    assert store.read('watchlist',{}) == {'BOTH':original}
    store.save_watch(replace(row,symbol='NYSE'),'Alpaca IEX','sp500')
    watch = store.read('watchlist',{})
    assert set(watch) == {'BOTH','NYSE'} and watch['NYSE']['universe_id'] == 'sp500'
    assert rectangle_from_record(watch['BOTH']['rectangle']) == row


def test_runner_dispatches_selected_universe_and_failed_load_keeps_both_results(tmp_path, monkeypatch):
    import agentloop_trader.rectangle_jobs as jobs
    frame = weekly_box(end=latest_completed_date('1d').strftime('%Y-%m-%d'))
    store, provider = RectangleStore(tmp_path), FakeProvider({'NYSE':frame})
    requested = []
    def selected(uid):
        requested.append(uid)
        return {**universe(['NYSE'])(),'universe_id':uid}
    monkeypatch.setattr(jobs,'fetch_universe',selected)
    run_range_scan(store,provider,ScanJob(cancel=provider.cancel),universe_id='sp500')
    assert requested == ['sp500'] and not store.read_scan('1d')
    sp_before = store.read_scan('1d','sp500')
    store.write('universe_sp500',{})  # Force a refresh attempt.
    store.write_scan('1d',{'sentinel':'nasdaq'})
    def failure(uid): raise ValueError('Universe temporarily unavailable')
    monkeypatch.setattr(jobs,'fetch_universe',failure)
    job = ScanJob(cancel=provider.cancel)
    run_range_scan(store,provider,job,universe_id='sp500')
    assert job.error and store.read_scan('1d','sp500') == sp_before
    assert store.read_scan('1d') == {'sentinel':'nasdaq'}


def test_cross_universe_cache_mismatch_is_not_silently_used(tmp_path):
    store = RectangleStore(tmp_path)
    store.write('universe_sp500',{**universe(['WRONG'])(),'universe_id':'nasdaq'})
    provider = FakeProvider({})
    job = ScanJob(cancel=provider.cancel)
    run_range_scan(store,provider,job,universe_id='sp500')
    assert 'different identity' in job.error and not provider.calls


def write_snapshot(store, row, uid):
    store.write_scan(row.interval, {'universe_id':uid,'interval':row.interval,'detector_version':DETECTOR_VERSION,
        'scanned_at':pd.Timestamp.now(tz='UTC').isoformat(),'source':'Alpaca IEX',
        'rectangles':[row.record()],'successful':1,'total':1,'skipped':0,'errors':[],
        'new_symbols':[row.symbol],'data_start':row.end,'data_end':row.end,
        'universe':{'directory_date':'test','source':'test fixture'},'first_scan':True},uid)


def test_ui_switches_universes_without_scanning_and_keeps_shared_watchlist(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import agentloop_trader.rectangle_ui as ui
    from agentloop_trader.rectangle_data import DailyHistoryProvider,HISTORY_DAYS,history_bucket
    store = RectangleStore(tmp_path)
    frame = weekly_box(end=latest_completed_date('1d').strftime('%Y-%m-%d'))
    base = detect_rectangles('NAS',frame)[0]
    for uid,symbol in [('nasdaq','NAS'),('sp500','NYSE')]:
        row = replace(base,symbol=symbol)
        write_snapshot(store,row,uid)
        store.write_history(symbol,'Alpaca IEX',frame,history_bucket(),history_days=HISTORY_DAYS)
    store.save_watch(replace(base,symbol='NYSE'),'Alpaca IEX','sp500')
    before = store.read('watchlist',{})
    def forbidden(*args,**kwargs): raise AssertionError('Universe switching must not scan or download')
    monkeypatch.setattr(ui,'start_range_scan',forbidden)
    monkeypatch.setattr(DailyHistoryProvider,'fetch',forbidden)
    script = f'from agentloop_trader.rectangle_ui import render_rectangle_scanner\nrender_rectangle_scanner(store_root={str(tmp_path)!r})'
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert app.selectbox(key='rectangle_universe').value == 'nasdaq'
    assert app.dataframe[0].value.Ticker.tolist() == ['NAS']
    assert app.dataframe[1].value.Ticker.tolist() == ['NYSE']  # Saved S&P candidate is visible in Nasdaq mode.
    app.selectbox(key='rectangle_universe').set_value('sp500').run()
    assert not app.exception
    assert app.dataframe[0].value.Ticker.tolist() == ['NYSE']
    assert app.dataframe[1].value['Found in'].tolist() == ['S&P 500']
    app.selectbox(key='rectangle_universe').set_value('nasdaq').run()
    assert not app.exception and app.dataframe[0].value.Ticker.tolist() == ['NAS']
    assert store.read('watchlist',{}) == before


def test_new_universe_empty_state_and_scan_button_use_selected_identity(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import agentloop_trader.rectangle_ui as ui
    from agentloop_trader.rectangle_data import DailyHistoryProvider
    calls=[]
    def start(store,provider,**kwargs):
        calls.append(kwargs)
        return ScanJob(done=True,message='Test complete',progress=1.)
    monkeypatch.setattr(ui,'start_range_scan',start)
    def forbidden(*args,**kwargs): raise AssertionError('No live downloads in UI test')
    monkeypatch.setattr(DailyHistoryProvider,'fetch',forbidden)
    script = f'from agentloop_trader.rectangle_ui import render_rectangle_scanner\nrender_rectangle_scanner(store_root={str(tmp_path)!r})'
    app=AppTest.from_string(script)
    app.session_state['rectangle_universe']='sp500'
    app.run()
    assert not app.exception and any('S&P 500' in i.value for i in app.info)
    assert not calls
    next(b for b in app.button if b.label=='Scan for Trading Ranges').click().run()
    assert not app.exception
    assert calls == [{'interval':'1d','universe_id':'sp500'}]
