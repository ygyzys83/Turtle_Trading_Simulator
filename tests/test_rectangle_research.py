import json
import threading
from dataclasses import replace

import pandas as pd
import pytest

from agentloop_trader.rectangle_data import DailyHistoryProvider, ScanCancelled, merge_history, parse_nasdaq_directory
from agentloop_trader.rectangle_jobs import ScanJob, run_range_scan
from agentloop_trader.rectangle_scanner import detect_rectangles
from agentloop_trader.rectangle_store import RectangleStore
from agentloop_trader.rectangle_ui import build_rectangle_chart
from test_rectangle_scanner import box_bars


def current_bars():
    return box_bars(end=(pd.Timestamp.now().normalize()-pd.offsets.BDay()).strftime('%Y-%m-%d'))


class FakeProvider:
    source = 'Alpaca IEX'
    api_key = None
    api_secret = None

    def __init__(self, frames):
        self.frames = frames
        self.cancel = threading.Event()
        self.calls = []

    def check_cancel(self):
        if self.cancel.is_set():
            raise ScanCancelled()

    def fetch(self, symbols, start):
        self.check_cancel()
        self.calls.append((list(symbols), start))
        return {s: self.frames[s] for s in symbols if s in self.frames}


def universe(symbols):
    return lambda: {'symbols': symbols, 'fetched_at': pd.Timestamp.now(tz='UTC').isoformat(),
                    'directory_date': 'test', 'source': 'test fixture'}


def test_directory_filters_security_types_and_detects_bad_responses():
    header = 'Symbol|Security Name|Test Issue|ETF|Financial Status|NextShares\n'
    rows = [
        'GOOD|Good Inc - Common Stock|N|N|N|N',
        'ADR|Example - American Depositary Shares|N|N|N|N',
        'ORD|Example - Ordinary Shares|N|N|N|N',
        'FUND|Example ETF - Common Stock|N|Y|N|N',
        'PREF|Example - Preferred Stock|N|N|N|N',
        'WAR|Example - Warrants|N|N|N|N',
        'UNIT|Example - Units|N|N|N|N',
        'TEST|Example - Common Stock|Y|N|N|N',
        'BAD|Example - Common Stock|N|N|D|N',
        'File Creation Time: 1019202612:00|||||',
    ]
    symbols, date = parse_nasdaq_directory(header+'\n'.join(rows))
    assert symbols == ['ADR', 'GOOD', 'ORD']
    assert date == '1019202612:00'
    with pytest.raises(ValueError):
        parse_nasdaq_directory('<html>provider unavailable</html>')


def test_scan_persists_all_lengths_new_status_cache_and_partial_errors(tmp_path):
    store = RectangleStore(tmp_path)
    provider = FakeProvider({'BOX': current_bars()})
    job = ScanJob(cancel=provider.cancel)
    run_range_scan(store, provider, job, universe(['BOX', 'FAIL']))
    assert job.done and not job.error
    result = store.read('scan', {})
    assert len(result['rectangles']) > 1
    assert result['new_symbols'] == ['BOX']
    assert result['successful'] == 1 and result['errors'][0]['Ticker'] == 'FAIL'
    calls = len(provider.calls)
    run_range_scan(store, provider, ScanJob(cancel=provider.cancel), universe(['BOX', 'FAIL']))
    assert store.read('scan', {})['new_symbols'] == []
    assert provider.calls[-1][0] == ['FAIL']  # BOX was served from disk, not re-downloaded.
    assert len(provider.calls) == calls+1


def test_cancel_and_fatal_download_failure_preserve_previous_snapshot(tmp_path):
    store = RectangleStore(tmp_path)
    store.write('scan', {'sentinel': 'original'})
    provider = FakeProvider({})
    provider.cancel.set()
    job = ScanJob(cancel=provider.cancel)
    run_range_scan(store, provider, job, universe(['BOX']))
    assert job.canceled and store.read('scan', {}) == {'sentinel': 'original'}
    provider.cancel.clear()
    job = ScanJob(cancel=provider.cancel)
    run_range_scan(store, provider, job, universe(['BOX']))
    assert job.error and store.read('scan', {}) == {'sentinel': 'original'}


def test_failed_symbol_keeps_observed_presence_and_is_not_falsely_new_later(tmp_path):
    store = RectangleStore(tmp_path)
    store.write('scan', {'source': 'Alpaca IEX', 'observed': {'BOX': True}})
    provider = FakeProvider({'OTHER': current_bars()})
    run_range_scan(store, provider, ScanJob(cancel=provider.cancel), universe(['BOX', 'OTHER']))
    assert store.read('scan', {})['observed']['BOX'] is True
    provider.frames['BOX'] = current_bars()
    run_range_scan(store, provider, ScanJob(cancel=provider.cancel), universe(['BOX', 'OTHER']))
    assert 'BOX' not in store.read('scan', {})['new_symbols']


def test_incremental_cache_reloads_full_history_when_adjusted_prices_change(tmp_path):
    from agentloop_trader.rectangle_data import history_bucket
    store = RectangleStore(tmp_path)
    old = current_bars()
    store.write_history('BOX', 'Alpaca IEX', old, 'old')
    adjusted = old.copy()
    adjusted[['Open', 'High', 'Low', 'Close']] /= 2
    provider = FakeProvider({'BOX': adjusted})
    run_range_scan(store, provider, ScanJob(cancel=provider.cancel), universe(['BOX']))
    assert len(provider.calls) == 2
    frame, bucket = store.read_history('BOX', 'Alpaca IEX')
    assert bucket == history_bucket()
    assert frame.Close.iloc[-1] == pytest.approx(adjusted.Close.iloc[-1])
    assert merge_history(old, adjusted) is None


def test_watchlist_preserves_original_and_writes_cannot_corrupt_json(tmp_path):
    store = RectangleStore(tmp_path)
    original = detect_rectangles('BOX', current_bars())[0]
    store.save_watch(original, 'Alpaca IEX')
    store.save_watch(replace(original, support=original.support+1), 'Alpaca IEX')
    assert store.read('watchlist', {})['BOX']['rectangle']['support'] == original.support
    threads = [threading.Thread(target=store.save_watch, args=(replace(original, symbol=f'TEST{i}'), 'Alpaca IEX')) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(store.read('watchlist', {})) == 9
    assert not list(tmp_path.glob('*.tmp'))
    store.remove_watch('BOX')
    assert 'BOX' not in store.read('watchlist', {})
    assert not (tmp_path / 'buy_watchlist.json').exists()
    (tmp_path / 'watchlist.json').write_text('{broken', encoding='utf-8')
    with pytest.raises(ValueError, match='preserved'):
        store.save_watch(original, 'Alpaca IEX')
    assert (tmp_path / 'watchlist.json').read_text() == '{broken'


def test_alpaca_batched_pagination_exhausts_tokens_before_returning_any_symbol(monkeypatch):
    provider = DailyHistoryProvider('test-key', 'test-secret')
    responses = [
        {'bars': {'AAPL': [{'t': '2026-10-16T04:00:00Z', 'o': 100, 'h': 102, 'l': 99, 'c': 101, 'v': 100}]}, 'next_page_token': 'page2'},
        {'bars': {'MSFT': [{'t': '2026-10-16T04:00:00Z', 'o': 200, 'h': 202, 'l': 199, 'c': 201, 'v': 100}]}, 'next_page_token': None},
    ]
    calls = []
    def read(params):
        calls.append(dict(params))
        return responses.pop(0)
    monkeypatch.setattr(provider, '_json', read)
    result = provider.fetch(['AAPL', 'MSFT'], '2026-10-01')
    assert set(result) == {'AAPL', 'MSFT'}
    assert calls[1]['page_token'] == 'page2' and calls[0]['symbols'] == 'AAPL,MSFT'
    monkeypatch.setattr(provider, '_json', lambda _: {'bars': {}, 'next_page_token': 'repeat'})
    with pytest.raises(RuntimeError, match='partial data was discarded'):
        provider.fetch(['AAPL'], '2026-10-01')


def test_chart_contains_exact_geometry_and_volume_only_when_requested():
    frame = current_bars()
    rectangle = detect_rectangles('BOX', frame)[0]
    fig = build_rectangle_chart(frame, rectangle)
    assert not any(trace.type == 'bar' for trace in fig.data)
    support_trace = next(t for t in fig.data if t.name.startswith('Support $'))
    assert list(support_trace.y) == [rectangle.support, rectangle.support]
    assert any(s.x0 == rectangle.start and s.x1 == rectangle.end for s in fig.layout.shapes)
    volume = build_rectangle_chart(frame, rectangle, show_volume=True)
    assert any(trace.type == 'bar' for trace in volume.data)
    assert volume.layout.xaxis.rangeslider.visible is False


def test_no_scan_or_broker_dependencies_in_detector_store_or_ui():
    import inspect
    import agentloop_trader.rectangle_scanner as detector
    import agentloop_trader.rectangle_store as persistence
    import agentloop_trader.rectangle_ui as ui
    for module in (detector, persistence, ui):
        source = inspect.getsource(module)
        assert 'BuyWatchlistStore' not in source
        assert 'submit_order' not in source
        assert 'run_strategy_suite' not in source


def test_ui_renders_saved_results_and_filters_without_running_another_scan(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import agentloop_trader.rectangle_ui as ui
    store = RectangleStore(tmp_path)
    frame = current_bars()
    row = detect_rectangles('BOX', frame)[0]
    store.write_history('BOX', 'Alpaca IEX', frame, 'test')
    store.write('scan', {'scanned_at': pd.Timestamp.now(tz='UTC').isoformat(), 'source': 'Alpaca IEX',
                        'rectangles': [{**row.record(), 'containment': .9}], 'successful': 1, 'total': 1, 'skipped': 0,
                        'errors': [], 'new_symbols': ['BOX'], 'data_start': row.end,
                        'data_end': row.end, 'universe': {'directory_date': 'test'}, 'first_scan': True})
    def forbidden(*args, **kwargs):
        raise AssertionError('A filter/chart action must not start a scan or download.')
    monkeypatch.setattr(ui, 'start_range_scan', forbidden)
    monkeypatch.setattr(DailyHistoryProvider, 'fetch', forbidden)
    script = f'from agentloop_trader.rectangle_ui import render_rectangle_scanner\nrender_rectangle_scanner(store_root={str(tmp_path)!r})'
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert any('earlier rectangle rules' in item.value for item in app.info)
    assert any('100.0% of closes inside the drawn box' in item.value for item in app.caption)
    assert app.selectbox(key='rectangle_location').value == 'All'
    assert app.slider(key='rectangle_duration').value == (1, 18)
    app.toggle(key='rectangle_only_new').set_value(True).run()
    assert not app.exception
    app.toggle(key='rectangle_volume_result').set_value(True).run()
    assert not app.exception
    app.slider(key='rectangle_duration').set_value((17, 18)).run()
    assert not app.exception
    app.selectbox(key='rectangle_location').set_value('Near resistance').run()
    assert not app.exception
    app.selectbox(key='rectangle_location').set_value('All').run()
    assert not app.exception
    next(b for b in app.button if b.label == 'Save to Watchlist').click().run()
    assert not app.exception
    assert 'BOX' in store.read('watchlist', {})
    next(b for b in app.button if b.label == 'Remove from Watchlist').click().run()
    assert not app.exception
    assert store.read('watchlist', {}) == {}
    from agentloop_trader.rectangle_scanner import DETECTOR_VERSION
    payload = store.read('scan', {})
    payload['detector_version'] = DETECTOR_VERSION
    store.write('scan', payload)
    # Use a fresh UI session to inspect the persisted updated snapshot. AppTest
    # retains removed watchlist widget nodes after an explicit st.rerun().
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert not any('earlier rectangle rules' in item.value for item in app.info)


def test_one_month_rectangle_can_qualify_when_it_has_enough_independent_tests():
    import numpy as np
    frame = current_bars().iloc[-21:].copy()
    close = 100+10*np.sin(np.arange(21)*2*np.pi/8)
    for column, offset in [('Close', 0), ('Open', -.15), ('High', .55), ('Low', -.55)]:
        frame[column] = close+offset
    assert any(row.bars == 21 for row in detect_rectangles('SHORTBOX', frame))
