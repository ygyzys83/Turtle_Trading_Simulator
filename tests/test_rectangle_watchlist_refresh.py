"""Watchlist-only updates preserve original research and never run discovery."""
from dataclasses import replace
import json

import pandas as pd
import pytest

from agentloop_trader.rectangle_data import DailyHistoryProvider
from agentloop_trader.rectangle_jobs import ScanJob, run_watchlist_refresh
from agentloop_trader.rectangle_scanner import detect_rectangles, refresh_saved_rectangle
from agentloop_trader.rectangle_signals import analyze_breakout_signals
from agentloop_trader.rectangle_store import RectangleStore, exclusive_file
from agentloop_trader.rectangle_timeframes import analysis_bars, latest_completed_date
from agentloop_trader.indicators import calc_atr
from test_rectangle_research import FakeProvider
from test_rectangle_weekly import weekly_box


def fixture(store, *, source='Alpaca IEX', symbol='BOX', interval='1d'):
    frame = weekly_box(end=latest_completed_date('1d').strftime('%Y-%m-%d'))
    # Save before the current completed candle/week to exercise advancing dates.
    old = frame.iloc[:-10].copy()
    original = detect_rectangles(symbol, old, interval=interval)[0]
    store.save_watch(original, source, universe_id='sp500' if interval == '1w' else 'nasdaq')
    store.write_history(symbol, source, old, 'old')
    return original, old, frame


def test_refresh_downloads_only_watched_and_deduplicates_daily_weekly(tmp_path, monkeypatch):
    store = RectangleStore(tmp_path)
    original, old, frame = fixture(store)
    weekly, _, _ = fixture(store, interval='1w')
    before = (tmp_path/'watchlist.json').read_bytes()
    store.write('scan', {'sentinel': 'keep'})
    store.write('scan_sp500_weekly', {'sentinel': 'keep weekly'})
    def forbidden(*args, **kwargs):
        raise AssertionError('A watchlist refresh must not run discovery.')
    monkeypatch.setattr('agentloop_trader.rectangle_jobs.fetch_universe', forbidden)
    monkeypatch.setattr('agentloop_trader.rectangle_jobs.detect_rectangles', forbidden)
    provider = FakeProvider({'BOX': frame, 'UNWATCHED': frame})
    job = ScanJob(cancel=provider.cancel, kind='watchlist_refresh')
    run_watchlist_refresh(store, provider, job)
    assert job.done and not job.error and not job.problems and job.refreshed == 1
    assert len(provider.calls) == 1 and provider.calls[0][0] == ['BOX']
    fresh, bucket = store.read_history('BOX', provider.source)
    assert fresh.index[-1] == latest_completed_date('1d')
    assert bucket == fresh.index[-1].strftime('%Y-%m-%d')
    assert (tmp_path/'watchlist.json').read_bytes() == before
    assert store.read('scan', {}) == {'sentinel': 'keep'}
    assert store.read('scan_sp500_weekly', {}) == {'sentinel': 'keep weekly'}
    assert not store.history_path('UNWATCHED', provider.source).exists()
    for row in (original, weekly):
        current, _ = refresh_saved_rectangle(row, row.last_close, fresh, assess_signals=True)
        assert current.signals_as_of == latest_completed_date(row.interval).strftime('%Y-%m-%d')
        assert current.end == row.end and current.ranking_score == row.ranking_score
        assert current.support == row.support and current.resistance == row.resistance


@pytest.mark.parametrize('interval', ['1d', '1w'])
def test_current_checks_use_latest_candles_and_current_atr_not_saved_scan(tmp_path, interval):
    store = RectangleStore(tmp_path)
    row, _, frame = fixture(store, interval=interval)
    row = replace(row, higher_lows=True, tight_near_resistance=True, supportive_volume=True,
                  signals_as_of='2000-01-01', atr=999)
    current, _ = refresh_saved_rectangle(row, row.last_close, frame, assess_signals=True)
    bars = analysis_bars(frame, interval)
    atr = calc_atr(bars.Close.to_numpy(), 14, bars.High.to_numpy(), bars.Low.to_numpy())[-1]
    expected = analyze_breakout_signals(bars, start=row.start, end=bars.index[-1],
                                       support=row.support, resistance=row.resistance, atr=atr, interval=interval)
    for key, value in expected.items():
        assert getattr(current, key) == value
    historical, _ = refresh_saved_rectangle(row, row.last_close, frame)
    assert historical.signals_as_of == '2000-01-01'  # Other callers keep snapshot semantics.
    adjusted = frame.copy()
    adjusted[['Open', 'High', 'Low', 'Close']] /= 2
    rebased, message = refresh_saved_rectangle(row, row.last_close, adjusted, assess_signals=True)
    assert message and rebased.support == pytest.approx(row.support/2)
    assert rebased.higher_lows == current.higher_lows
    assert rebased.tight_near_resistance == current.tight_near_resistance
    assert rebased.higher_low_prices == tuple(p/2 for p in current.higher_low_prices)


@pytest.mark.parametrize('failure', ['missing', 'stale', 'basis', 'incomplete_week', 'exception'])
def test_bad_downloads_keep_old_history_and_report_partial_success(tmp_path, failure):
    store = RectangleStore(tmp_path)
    row, old, frame = fixture(store, interval='1w')
    fixture(store, symbol='GOOD')
    path = store.history_path('BOX', 'Alpaca IEX')
    before = path.read_bytes()
    frames = {'GOOD': frame}
    if failure == 'stale':
        frames['BOX'] = old
    elif failure == 'basis':
        frames['BOX'] = frame.drop(pd.Timestamp(row.end))
    elif failure == 'incomplete_week':
        frames['BOX'] = frame.drop(latest_completed_date('1w'))
    provider = FakeProvider(frames)
    if failure == 'exception':
        provider.fetch = lambda *args: (_ for _ in ()).throw(RuntimeError('secret provider URL'))
    job = ScanJob(cancel=provider.cancel)
    run_watchlist_refresh(store, provider, job)
    assert job.done and not job.error and job.problems
    assert path.read_bytes() == before
    assert 'secret provider URL' not in str(job.snapshot())
    assert job.refreshed == (0 if failure == 'exception' else 1)


def test_original_feeds_are_preserved_and_missing_credentials_reported(tmp_path, monkeypatch):
    store = RectangleStore(tmp_path)
    _, _, frame = fixture(store)
    fixture(store, source='Yahoo', symbol='YAHOO')
    calls = []
    monkeypatch.setattr(DailyHistoryProvider, 'fetch',
                        lambda self, symbols, start: calls.append((self.source, symbols)) or {'YAHOO': frame})
    alpaca = FakeProvider({'BOX': frame})
    job = ScanJob(cancel=alpaca.cancel)
    run_watchlist_refresh(store, alpaca, job)
    assert job.refreshed == 2 and not job.problems
    assert calls == [('Yahoo', ['YAHOO'])] and alpaca.calls[0][0] == ['BOX']
    # With no credentials the Alpaca record is not silently downloaded from Yahoo.
    yahoo = DailyHistoryProvider()
    before = store.history_path('BOX', 'Alpaca IEX').read_bytes()
    job = ScanJob(cancel=yahoo.cancel)
    run_watchlist_refresh(store, yahoo, job)
    assert job.refreshed == 1 and len(job.problems) == 1
    assert job.problems[0]['Source'] == 'Alpaca IEX'
    assert 'credentials' in job.problems[0]['Problem']
    assert store.history_path('BOX', 'Alpaca IEX').read_bytes() == before


def test_cancel_lock_and_empty_watchlist(tmp_path):
    store = RectangleStore(tmp_path)
    provider = FakeProvider({})
    job = ScanJob(cancel=provider.cancel)
    run_watchlist_refresh(store, provider, job)
    assert job.done and not job.error and not provider.calls
    fixture(store)
    before = store.history_path('BOX', provider.source).read_bytes()
    provider.cancel.set()
    job = ScanJob(cancel=provider.cancel)
    run_watchlist_refresh(store, provider, job)
    assert job.canceled and not provider.calls
    provider.cancel.clear()
    with exclusive_file(tmp_path/'scan.lock'):
        job = ScanJob(cancel=provider.cancel)
        run_watchlist_refresh(store, provider, job)
        assert job.error and not provider.calls
    assert store.history_path('BOX', provider.source).read_bytes() == before


def test_partial_daily_download_cannot_remove_cached_sessions(tmp_path):
    store = RectangleStore(tmp_path)
    row, old, frame = fixture(store)
    before = store.history_path('BOX', 'Alpaca IEX').read_bytes()
    frame = frame.drop(old.index[-3])  # Latest date and anchor still exist.
    provider = FakeProvider({'BOX': frame})
    job = ScanJob(cancel=provider.cancel)
    run_watchlist_refresh(store, provider, job)
    assert job.refreshed == 0 and len(job.problems) == 1
    assert 'cached sessions' in job.problems[0]['Problem']
    assert store.history_path('BOX', provider.source).read_bytes() == before


def test_ui_explicit_refresh_updates_chart_without_scan_or_arming(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import agentloop_trader.rectangle_ui as ui
    store = RectangleStore(tmp_path/'research')
    row, _, frame = fixture(store)
    before = store.read('watchlist', {})
    calls = []
    def forbidden(*args, **kwargs):
        raise AssertionError('Refresh must not run a universe scan.')
    monkeypatch.setattr(ui, 'start_range_scan', forbidden)
    monkeypatch.setattr(DailyHistoryProvider, 'fetch',
                        lambda self, symbols, start: calls.append(symbols) or {'BOX': frame})
    # Run the real refresh job synchronously for a deterministic UI test.
    def start(store, provider):
        job = ScanJob(cancel=provider.cancel, kind='watchlist_refresh')
        run_watchlist_refresh(store, provider, job)
        return job
    monkeypatch.setattr(ui, 'start_watchlist_refresh', start)
    queue = tmp_path/'buy_queue.json'
    script = f'''from agentloop_trader.rectangle_ui import render_rectangle_scanner
render_rectangle_scanner('fake-key','fake-secret',store_root={str(store.root)!r},buy_queue_path={str(queue)!r})'''
    app = AppTest.from_string(script).run()
    assert not app.exception and not calls
    assert any(f'Setup assessed through {row.end}' in c.value for c in app.caption)
    next(b for b in app.button if b.label == 'Refresh Watchlist').click().run()
    assert not app.exception and calls == [['BOX']]
    assert any('Watchlist refreshed:' in s.value for s in app.success)
    assert any(f'Setup assessed through {latest_completed_date("1d"):%Y-%m-%d}' in c.value for c in app.caption)
    chart = json.loads(app.get('plotly_chart')[-1].proto.spec)
    assert chart['data'][0]['x'][-1] == latest_completed_date('1d').strftime('%Y-%m-%d')
    assert store.read('watchlist', {}) == before and not queue.exists()
    assert not store.read('scan', {})
    app.run()
    assert not app.exception and calls == [['BOX']] and not queue.exists()
