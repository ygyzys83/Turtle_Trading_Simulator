from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from agentloop_trader.rectangle_scanner import (
    detect_rectangles, filter_rectangles, rectangle_from_record, refresh_saved_rectangle,
)
from agentloop_trader.rectangle_timeframes import (
    analysis_bars, completed_daily_bars, latest_completed_date, session_schedule, weekly_bars,
)
from agentloop_trader.rectangle_data import HISTORY_DAYS, full_history_start, history_bucket
from agentloop_trader.rectangle_store import RectangleStore
from agentloop_trader.rectangle_jobs import ScanJob, run_range_scan
from test_rectangle_research import FakeProvider, universe
from test_rectangle_scanner import box_bars, NOW


def sessions(start, end):
    return session_schedule(pd.Timestamp(start).year, pd.Timestamp(end).year).loc[start:end].index


def candle_fixture(start, end):
    index = sessions(start, end)
    x = np.arange(len(index), dtype=float)
    return pd.DataFrame({'Open': 100+x, 'High': 103+x, 'Low': 98+x,
                         'Close': 101+x, 'Volume': 100+x}, index=index)


def weekly_box(end='2026-10-19', length=756):
    frame = box_bars(length, end=end)
    # Slower swings make weekly high/low ranges proportionate to a long base.
    week = (frame.index.to_period('W-FRI').asi8-frame.index[0].to_period('W-FRI').ordinal)
    close = 100+10*np.sin(week*2*np.pi/10)
    frame[['Open', 'Close']] = np.column_stack([close-.05, close])
    frame['High'], frame['Low'] = close+.15, close-.15
    return frame


def test_weekly_ohlcv_aggregation_and_midweek_exclusion():
    frame = candle_fixture('2026-10-05', '2026-10-16')
    result = weekly_bars(frame, '2026-10-14 22:00Z')
    assert list(result.index) == [pd.Timestamp('2026-10-09')]
    week = frame.loc[:'2026-10-09']
    assert result.iloc[0].to_dict() == {
        'Open': week.Open.iloc[0], 'High': week.High.max(), 'Low': week.Low.min(),
        'Close': week.Close.iloc[-1], 'Volume': week.Volume.sum(),
    }
    before = weekly_bars(frame, '2026-10-16 19:59Z')
    after = weekly_bars(frame, '2026-10-16 20:00Z')
    assert len(before) == 1 and len(after) == 2


def test_holiday_short_week_completes_thursday_after_close():
    frame = candle_fixture('2026-03-23', '2026-04-10')
    assert weekly_bars(frame, '2026-04-02 19:59Z').index[-1] == pd.Timestamp('2026-03-27')
    result = weekly_bars(frame, '2026-04-02 20:00Z')
    assert result.index[-1] == pd.Timestamp('2026-04-02')
    assert result.Volume.iloc[-1] == frame.loc['2026-03-30':'2026-04-02', 'Volume'].sum()
    assert latest_completed_date('1w', '2026-04-03 12:00Z') == pd.Timestamp('2026-04-02')


def test_early_close_and_daylight_saving_time_are_respected():
    frame = candle_fixture('2026-11-23', '2026-11-27')
    assert pd.Timestamp('2026-11-27') not in completed_daily_bars(frame, '2026-11-27 17:59Z').index
    assert weekly_bars(frame, '2026-11-27 17:59Z').empty
    assert weekly_bars(frame, '2026-11-27 18:00Z').index[-1] == pd.Timestamp('2026-11-27')
    assert latest_completed_date('1d', '2026-11-27 18:00Z') == pd.Timestamp('2026-11-27')
    assert latest_completed_date('1w', '2026-03-06 20:59Z') == pd.Timestamp('2026-02-27')
    assert latest_completed_date('1w', '2026-03-06 21:00Z') == pd.Timestamp('2026-03-06')
    assert latest_completed_date('1w', '2026-03-13 19:59Z') == pd.Timestamp('2026-03-06')
    assert latest_completed_date('1w', '2026-03-13 20:00Z') == pd.Timestamp('2026-03-13')


def test_missing_session_and_initial_partial_week_are_not_used():
    frame = candle_fixture('2026-10-05', '2026-10-16').drop(pd.Timestamp('2026-10-07'))
    result = weekly_bars(frame, NOW)
    assert list(result.index) == [pd.Timestamp('2026-10-16')]
    partial = candle_fixture('2026-10-07', '2026-10-16')
    assert list(weekly_bars(partial, NOW).index) == [pd.Timestamp('2026-10-16')]


def test_cross_year_week_retains_all_expected_sessions():
    frame = candle_fixture('2025-12-29', '2026-01-09')
    result = weekly_bars(frame, '2026-01-10 00:00Z')
    assert len(result) == 2
    assert result.index[0] == pd.Timestamp('2026-01-02')
    assert result.Volume.iloc[0] == frame.loc[:'2026-01-02', 'Volume'].sum()


def test_weekly_ranges_cover_three_to_twenty_four_months_with_weekly_atr():
    frame = weekly_box()
    rows = detect_rectangles('BOX', frame, interval='1w', now=NOW)
    assert rows
    assert all(row.interval == '1w' and 3 <= row.months <= 24 for row in rows)
    assert any(row.bars == 104 and row.months == 24 for row in rows)
    from agentloop_trader.indicators import calc_atr
    bars = weekly_bars(frame, NOW)
    expected = calc_atr(bars.Close, 14, bars.High, bars.Low)[-1]
    assert all(row.atr == pytest.approx(expected) for row in rows)
    assert all(row.containment >= .9 for row in rows)


def test_ongoing_week_never_changes_a_confirmed_weekly_rectangle():
    frame = weekly_box(end='2026-10-23')
    now = pd.Timestamp('2026-10-21 22:00Z')
    before = detect_rectangles('BOX', frame, interval='1w', now=now)
    mask = frame.index >= pd.Timestamp('2026-10-19')
    frame.loc[mask, ['Open', 'High', 'Low', 'Close']] = [500, 510, 490, 505]
    assert detect_rectangles('BOX', frame, interval='1w', now=now) == before


def test_detector_does_not_compress_a_missing_week_into_a_valid_long_range():
    frame = weekly_box()
    frame = frame.drop(frame.loc['2026-08-17':'2026-08-21'].index)
    assert not detect_rectangles('BOX', frame, interval='1w', now=NOW)


def test_legacy_daily_records_survive_and_same_symbol_can_have_both_timeframes(tmp_path):
    frame = weekly_box()
    daily = detect_rectangles('BOX', frame, now=NOW)[0]
    weekly = detect_rectangles('BOX', frame, interval='1w', now=NOW)[0]
    store = RectangleStore(tmp_path)
    legacy = daily.record()
    legacy.pop('interval')
    store.write('watchlist', {'BOX': {'rectangle': legacy, 'anchor_close': daily.last_close,
                                    'source': 'Alpaca IEX', 'saved_at': NOW.isoformat()}})
    before = store.read('watchlist', {})['BOX']
    store.save_watch(weekly, 'Alpaca IEX')
    assert store.read('watchlist', {})['BOX'] == before
    assert rectangle_from_record(before['rectangle']).interval == '1d'
    assert set(store.read('watchlist', {})) == {'BOX', 'BOX|1w'}
    assert len(filter_rectangles([daily, weekly])) == 2
    store.remove_watch('BOX', '1w')
    assert store.read('watchlist', {}) == {'BOX': before}


def test_weekly_saved_geometry_rebases_after_split_using_weekly_close():
    frame = weekly_box()
    original = detect_rectangles('BOX', frame, interval='1w', now=NOW)[0]
    split = frame.copy()
    split[['Open', 'High', 'Low', 'Close']] /= 2
    current, note = refresh_saved_rectangle(original, original.last_close, split, now=NOW)
    assert current.support == pytest.approx(original.support/2)
    assert current.interval == '1w' and note
    assert current.last_close == pytest.approx(weekly_bars(split, NOW).Close.iloc[-1])


def test_scan_snapshots_and_new_discoveries_are_separate_for_each_timeframe(tmp_path):
    store = RectangleStore(tmp_path)
    end = latest_completed_date('1d').strftime('%Y-%m-%d')
    provider = FakeProvider({'BOX': weekly_box(end=end)})
    for interval in ['1d', '1w']:
        job = ScanJob(cancel=provider.cancel)
        run_range_scan(store, provider, job, universe(['BOX']), interval=interval)
        assert not job.error
        scan = store.read_scan(interval)
        assert scan['interval'] == interval and scan['new_symbols'] == ['BOX']
        assert all(row['interval'] == interval for row in scan['rectangles'])
    daily = store.read_scan('1d')
    calls = len(provider.calls)
    run_range_scan(store, provider, ScanJob(cancel=provider.cancel), universe(['BOX']), interval='1w')
    assert len(provider.calls) == calls  # Daily history is shared without another download.
    assert store.read_scan('1w')['new_symbols'] == []
    assert store.read_scan('1d') == daily


def test_old_short_cache_is_backfilled_once_even_when_daily_bucket_is_current(tmp_path):
    store = RectangleStore(tmp_path)
    end = latest_completed_date('1d').strftime('%Y-%m-%d')
    frame = weekly_box(end=end)
    store.write_history('BOX', 'Alpaca IEX', frame.iloc[-420:], history_bucket())
    provider = FakeProvider({'BOX': frame})
    run_range_scan(store, provider, ScanJob(cancel=provider.cancel), universe(['BOX']))
    assert provider.calls == [(['BOX'], full_history_start())]
    _, bucket = store.read_history('BOX', 'Alpaca IEX')
    assert bucket == history_bucket()
    assert store.read_history('BOX', 'Alpaca IEX')[0].attrs['history_days'] == HISTORY_DAYS
    run_range_scan(store, provider, ScanJob(cancel=provider.cancel), universe(['BOX']))
    assert len(provider.calls) == 1


def test_snapshot_and_timeframe_validation_reject_mixed_data(tmp_path):
    store = RectangleStore(tmp_path)
    with pytest.raises(ValueError):
        store.write_scan('1w', {'interval': '1d'})
    with pytest.raises(ValueError):
        store.write_scan('1w', {'interval': '1w', 'rectangles': [{'interval': '1d'}]})
    with pytest.raises(ValueError):
        analysis_bars(weekly_box(), '4h')
    with pytest.raises(ValueError):
        store.read_scan('4h')


def test_ui_switches_timeframe_and_uses_weekly_candles_without_downloads(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import agentloop_trader.rectangle_ui as ui
    from agentloop_trader.rectangle_scanner import DETECTOR_VERSION
    frame = weekly_box(end=latest_completed_date('1d').strftime('%Y-%m-%d'))
    store = RectangleStore(tmp_path)
    store.write_history('BOX', 'Alpaca IEX', frame, history_bucket(), history_days=HISTORY_DAYS)
    daily = detect_rectangles('BOX', frame)[0]
    weekly = detect_rectangles('BOX', frame, interval='1w')[0]
    store.save_watch(daily, 'Alpaca IEX')
    for row in [daily, weekly]:
        store.write_scan(row.interval, {
            'interval': row.interval, 'detector_version': DETECTOR_VERSION,
            'scanned_at': pd.Timestamp.now(tz='UTC').isoformat(), 'source': 'Alpaca IEX',
            'rectangles': [row.record()], 'successful': 1, 'total': 1, 'skipped': 0,
            'errors': [], 'new_symbols': ['BOX'], 'data_start': row.end, 'data_end': row.end,
            'universe': {'directory_date': 'test'}, 'first_scan': True,
        })
    def forbidden(*args, **kwargs):
        raise AssertionError('Timeframe/filter/chart/save actions must not scan or download.')
    monkeypatch.setattr(ui, 'start_range_scan', forbidden)
    monkeypatch.setattr(FakeProvider, 'fetch', forbidden)
    from agentloop_trader.rectangle_data import DailyHistoryProvider
    monkeypatch.setattr(DailyHistoryProvider, 'fetch', forbidden)
    script = f'from agentloop_trader.rectangle_ui import render_rectangle_scanner\nrender_rectangle_scanner(store_root={str(tmp_path)!r})'
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert app.radio(key='rectangle_timeframe').value == 'Daily'
    assert app.slider(key='rectangle_duration_1d').value == (3, 24)
    # Use a fresh session for each direction to avoid AppTest's removed-widget tree.
    app = AppTest.from_string(script)
    app.session_state['rectangle_timeframe'] = 'Weekly'
    app.run()
    assert not app.exception
    assert app.slider(key='rectangle_duration_1w').value == (3, 24)
    assert any('Weekly candles through' in c.value for c in app.caption)
    assert all('Daily candles through' not in c.value for c in app.caption)
    table = app.dataframe[0].value
    assert table.Timeframe.tolist() == ['Weekly']
    next(b for b in app.button if b.label == 'Save to Watchlist').click().run()
    assert not app.exception
    assert set(store.read('watchlist', {})) == {'BOX', 'BOX|1w'}
    next(b for b in app.button if b.label == 'Remove from Watchlist').click().run()
    assert not app.exception
    assert set(store.read('watchlist', {})) == {'BOX'}


def test_weekly_chart_uses_aggregated_volume_and_retains_exact_boundaries():
    from agentloop_trader.rectangle_ui import build_rectangle_chart
    frame = weekly_box()
    rectangle = detect_rectangles('BOX', frame, interval='1w', now=NOW)[0]
    bars = weekly_bars(frame, NOW)
    figure = build_rectangle_chart(bars, rectangle, show_volume=True)
    assert figure.data[0].name == 'Completed weekly candles'
    shown = bars.loc[bars.index >= pd.Timestamp(rectangle.start)-pd.Timedelta(days=90)]
    assert list(figure.data[0].close) == list(shown.Close)
    volume = next(trace for trace in figure.data if trace.type == 'bar')
    assert list(volume.y) == list(shown.Volume)
    support = next(trace for trace in figure.data if trace.name.startswith('Support $'))
    assert list(support.y) == [rectangle.support, rectangle.support]
