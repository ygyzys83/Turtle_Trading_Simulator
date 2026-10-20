from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from agentloop_trader.rectangle_signals import analyze_breakout_signals
from agentloop_trader.rectangle_scanner import (
    DETECTOR_VERSION, detect_rectangles, filter_rectangles, rectangle_from_record,
    rectangle_records, refresh_saved_rectangle,
)
from agentloop_trader.rectangle_timeframes import session_schedule, latest_completed_date
from agentloop_trader.rectangle_store import RectangleStore
from test_rectangle_scanner import box_bars, NOW


def candles(interval='1d', end='2026-10-19', count=None):
    schedule = session_schedule(2023, 2027).loc[:end]
    dates = schedule.index if interval == '1d' else schedule.groupby(schedule.index.to_period('W-FRI')).tail(1).index
    dates = dates[-(count or (80 if interval == '1d' else 26)):]
    return pd.DataFrame(dict(Open=115., High=116., Low=114., Close=115., Volume=1_000_000.), index=dates)


def assess(frame, interval='1d', **kwargs):
    options = dict(start=frame.index[0], end=frame.index[-1], support=90., resistance=120., atr=2., interval=interval)
    options.update(kwargs)
    return analyze_breakout_signals(frame, **options)


@pytest.mark.parametrize('interval', ['1d', '1w'])
@pytest.mark.parametrize('second, expected', [(104., True), (100.6, True), (100.3, False), (100., False), (99., False)])
def test_higher_lows_require_meaningful_rise_of_latest_confirmed_pivots(interval, second, expected):
    frame = candles(interval)
    frame.iloc[-10, frame.columns.get_loc('Low')] = 100.
    frame.iloc[-5, frame.columns.get_loc('Low')] = second
    result = assess(frame, interval)
    assert result['higher_lows'] is expected
    assert result['higher_low_dates'] == tuple(frame.index[[-10, -5]].strftime('%Y-%m-%d'))
    assert result['higher_low_prices'] == (100., second)


@pytest.mark.parametrize('interval', ['1d', '1w'])
def test_unconfirmed_low_is_not_counted_and_later_undercut_cancels_pass(interval):
    frame = candles(interval)
    frame.iloc[-10, frame.columns.get_loc('Low')] = 100.
    frame.iloc[-2, frame.columns.get_loc('Low')] = 104.
    assert assess(frame, interval)['higher_lows'] is None  # Two later bars do not yet exist.
    frame.iloc[-2, frame.columns.get_loc('Low')] = 114.
    frame.iloc[-5, frame.columns.get_loc('Low')] = 104.
    assert assess(frame, interval)['higher_lows'] is True
    frame.iloc[-1, frame.columns.get_loc('Low')] = 103.
    assert assess(frame, interval)['higher_lows'] is False


def tight_candles(interval):
    frame = candles(interval)
    count = 10 if interval == '1d' else 4
    frame.iloc[-2*count:-count, frame.columns.get_indexer(['High', 'Low'])] = [120., 110.]
    return frame


@pytest.mark.parametrize('interval', ['1d', '1w'])
def test_tightness_requires_upper_third_midpoint_holding_and_actual_contraction(interval):
    frame = tight_candles(interval)
    assert assess(frame, interval)['tight_near_resistance'] is True
    lower = frame.copy()
    lower.iloc[-1, lower.columns.get_indexer(['Open', 'High', 'Low', 'Close'])] = [108., 109., 107., 108.]
    assert assess(lower, interval)['tight_near_resistance'] is False
    below_midpoint = frame.copy()
    count = 10 if interval == '1d' else 4
    n = int(count*.2)+1
    below_midpoint.iloc[-count:-count+n, below_midpoint.columns.get_indexer(['Open', 'High', 'Low', 'Close'])] = [100., 101., 99., 100.]
    assert assess(below_midpoint, interval)['tight_near_resistance'] is False
    assert assess(candles(interval), interval)['tight_near_resistance'] is False  # No contraction.
    wicks = frame.copy()
    wicks.iloc[-count:, wicks.columns.get_indexer(['High', 'Low'])] = [125., 105.]
    assert assess(wicks, interval)['tight_near_resistance'] is False  # Tiny bodies, large wicks.
    broken = frame.copy()
    broken.iloc[-1, broken.columns.get_indexer(['Open', 'High', 'Low', 'Close'])] = [121., 122., 120., 121.]
    assert assess(broken, interval)['tight_near_resistance'] is False


@pytest.mark.parametrize('interval', ['1d', '1w'])
def test_candle_gaps_are_part_of_tightness_not_just_bodies_or_wicks(interval):
    frame = tight_candles(interval)
    count = 10 if interval == '1d' else 4
    # Small candles jump between the upper half's extremes: true range stays large.
    for offset in range(count):
        close = 106. if offset % 2 == 0 else 119.
        frame.iloc[-count+offset, frame.columns.get_indexer(['Open', 'High', 'Low', 'Close'])] = [close, close+.1, close-.1, close]
    assert assess(frame, interval)['tight_near_resistance'] is False


def volume_candles(interval, supportive=True):
    frame = candles(interval)
    close = 115.+np.arange(len(frame)) % 2
    frame['Open'] = frame['Close'] = close
    frame['High'], frame['Low'] = close+.5, close-.5
    frame['Volume'] = np.where(frame.Close.diff() > 0, 2_000_000., 1_000_000.)
    if not supportive:
        frame['Volume'] = 3_000_000.-frame.Volume
    return frame


@pytest.mark.parametrize('interval', ['1d', '1w'])
def test_supportive_volume_compares_typical_advance_and_pullback_activity(interval):
    frame = volume_candles(interval)
    assert assess(frame, interval)['supportive_volume'] is True
    assert assess(volume_candles(interval, False), interval)['supportive_volume'] is False
    # One event spike does not dictate the median for all pullbacks.
    last_down = frame.index[frame.Close.diff() < 0][-1]
    frame.loc[last_down, 'Volume'] = 100_000_000.
    assert assess(frame, interval)['supportive_volume'] is True


@pytest.mark.parametrize('bad', ['missing', 'zero', 'nan', 'infinite', 'few_declines'])
def test_unusable_volume_or_inadequate_directional_samples_are_unknown(bad):
    frame = volume_candles('1d')
    if bad == 'missing':
        frame = frame.drop(columns='Volume')
    elif bad == 'few_declines':
        frame['Close'] = np.arange(len(frame))+100.
    else:
        frame.iloc[-1, frame.columns.get_loc('Volume')] = {'zero': 0., 'nan': np.nan, 'infinite': np.inf}[bad]
    assert assess(frame)['supportive_volume'] is None


def test_holiday_weeks_do_not_create_false_lighter_pullback_volume():
    frame = candles('1w', end='2026-01-09')
    schedule = session_schedule(2025, 2026)
    counts = schedule.groupby(schedule.index.to_period('W-FRI')).size().reindex(frame.index.to_period('W-FRI')).to_numpy()
    frame['Close'] = 115.+np.cumsum(np.where(counts < 5, -.2, .2))
    frame['Volume'] = counts*400_000.
    result = assess(frame, '1w')
    assert result['supportive_volume'] is False  # Identical activity per session.
    assert '100%' in result['signal_details'][2]


@pytest.mark.parametrize('volume', [0., np.nan, np.inf])
def test_missing_daily_volume_is_not_hidden_by_weekly_aggregation(volume):
    from agentloop_trader.rectangle_timeframes import weekly_bars
    frame = box_bars()
    frame['Volume'] = frame.Volume.astype(float)
    frame.iloc[-6, frame.columns.get_loc('Volume')] = volume
    bars = weekly_bars(frame, NOW)
    assert assess(bars, '1w')['supportive_volume'] is None


@pytest.mark.parametrize('interval', ['1d', '1w'])
def test_missing_candles_and_short_context_produce_unknown_instead_of_false_passes(interval):
    frame = volume_candles(interval)
    missing = frame.drop(frame.index[-3])
    result = assess(missing, interval)
    assert all(result[key] is None for key in ['higher_lows', 'tight_near_resistance', 'supportive_volume'])
    result = assess(frame.tail(3), interval)
    assert all(result[key] is None for key in ['higher_lows', 'tight_near_resistance', 'supportive_volume'])


@pytest.mark.parametrize('interval', ['1d', '1w'])
def test_assessments_stay_inside_rectangle_and_ignore_later_data(interval):
    frame = volume_candles(interval)
    end = frame.index[-5]
    expected = assess(frame.loc[:end], interval)
    changed = frame.copy()
    changed.loc[changed.index > end, ['Open', 'High', 'Low', 'Close', 'Volume']] = [1., 2., .5, 1., 0.]
    assert assess(changed, interval, end=end) == expected
    short_range = assess(frame, interval, start=frame.index[-3])
    assert short_range['higher_lows'] is None
    assert short_range['supportive_volume'] is None


def test_signal_flags_are_price_scale_invariant():
    frame = tight_candles('1d')
    frame.iloc[-40, frame.columns.get_loc('Low')] = 100.
    frame.iloc[-25, frame.columns.get_loc('Low')] = 104.
    expected = assess(frame)
    scaled = frame.copy()
    scaled[['Open', 'High', 'Low', 'Close']] *= .01
    actual = assess(scaled, support=.9, resistance=1.2, atr=.02)
    for key in ['higher_lows', 'tight_near_resistance', 'supportive_volume']:
        assert actual[key] is expected[key]
    assert actual['higher_low_prices'] == pytest.approx(np.array(expected['higher_low_prices'])*.01)


def test_optional_filters_require_all_selected_passes_before_deduplicating():
    row = detect_rectangles('BOX', box_bars(), now=NOW)[0]
    first = replace(row, quality=95, higher_lows=True, tight_near_resistance=False, supportive_volume=None)
    second = replace(row, quality=90, higher_lows=True, tight_near_resistance=True, supportive_volume=True)
    assert filter_rectangles([second, first]) == [first]
    assert filter_rectangles([first, second], higher_lows=True, tight_near_resistance=True) == [second]
    assert filter_rectangles([first], supportive_volume=True) == []
    assert filter_rectangles([second], higher_lows=True, tight_near_resistance=True, supportive_volume=True) == [second]
    other = replace(second, symbol='OTHER', quality=99)
    assert filter_rectangles([second, other], higher_lows=True) == [other, second]  # Existing rank is retained.


def test_scan_metadata_roundtrip_and_legacy_unknowns_without_rewriting_watchlist(tmp_path):
    row = detect_rectangles('BOX', box_bars(), now=NOW)[0]
    assert row.signals_as_of == row.end and len(row.signal_details) == 3
    assert rectangle_from_record(json.loads(json.dumps(row.record()))) == row
    legacy = row.record()
    for key in ['higher_lows', 'tight_near_resistance', 'supportive_volume', 'higher_low_dates',
                'higher_low_prices', 'signals_as_of', 'signal_details']:
        legacy.pop(key)
    old = rectangle_from_record(legacy)
    records = rectangle_records([old])[0]
    assert all(records[name] == 'Unknown' for name in ['Higher lows', 'Tight near resistance', 'Supportive volume'])
    store = RectangleStore(tmp_path)
    store.save_watch(row, 'Yahoo')
    assert rectangle_from_record(store.read('watchlist', {})['BOX']['rectangle']) == row


def test_chart_markers_follow_saved_geometry_and_split_adjustments():
    from agentloop_trader.rectangle_ui import build_rectangle_chart
    frame = box_bars()
    row = detect_rectangles('BOX', frame, now=NOW)[0]
    assert row.higher_low_dates
    split = frame.copy()
    split[['Open', 'High', 'Low', 'Close']] *= .5
    current, _ = refresh_saved_rectangle(row, row.last_close, split, now=NOW)
    assert current.higher_low_prices == pytest.approx(np.array(row.higher_low_prices)*.5)
    fig = build_rectangle_chart(split, current)
    markers = next(t for t in fig.data if t.name == 'Confirmed pullback lows')
    assert list(markers.x) == list(row.higher_low_dates)
    assert list(markers.y) == list(current.higher_low_prices)
    midpoint = next(t for t in fig.data if t.name == 'Range midpoint')
    assert list(midpoint.y) == [(current.support+current.resistance)/2]*2


@pytest.mark.parametrize('interval', ['1d', '1w'])
def test_three_post_scan_controls_filter_locally_without_scan_or_network(tmp_path, monkeypatch, interval):
    from streamlit.testing.v1 import AppTest
    import agentloop_trader.rectangle_ui as ui
    from agentloop_trader.rectangle_data import DailyHistoryProvider, HISTORY_DAYS, history_bucket
    from test_rectangle_weekly import weekly_box
    end = latest_completed_date('1d').strftime('%Y-%m-%d')
    frame = weekly_box(end=end)
    row = detect_rectangles('BOX', frame, interval=interval)[0]
    rows = [replace(row, symbol='ALL', quality=90, higher_lows=True, tight_near_resistance=True, supportive_volume=True),
            replace(row, symbol='PRICE', quality=95, higher_lows=True, tight_near_resistance=True, supportive_volume=False),
            replace(row, symbol='VOL', quality=80, higher_lows=False, tight_near_resistance=False, supportive_volume=True),
            replace(row, symbol='OLD', quality=75, higher_lows=None, tight_near_resistance=None, supportive_volume=None)]
    store = RectangleStore(tmp_path)
    for r in rows:
        store.write_history(r.symbol, 'Alpaca IEX', frame, history_bucket(), history_days=HISTORY_DAYS)
    store.write_scan(interval, {'interval': interval, 'detector_version': DETECTOR_VERSION,
        'scanned_at': pd.Timestamp.now(tz='UTC').isoformat(), 'source': 'Alpaca IEX',
        'rectangles': [r.record() for r in rows], 'successful': 4, 'total': 4, 'skipped': 0,
        'errors': [], 'new_symbols': ['ALL', 'PRICE'], 'data_start': row.end, 'data_end': row.end,
        'universe': {'directory_date': 'test'}, 'first_scan': True})
    before = store.read_scan(interval)
    def forbidden(*args, **kwargs):
        raise AssertionError('Post-scan filters must not start scans or download data.')
    monkeypatch.setattr(ui, 'start_range_scan', forbidden)
    monkeypatch.setattr(DailyHistoryProvider, 'fetch', forbidden)
    script = f'from agentloop_trader.rectangle_ui import render_rectangle_scanner\nrender_rectangle_scanner(store_root={str(tmp_path)!r})'
    app = AppTest.from_string(script)
    app.session_state['rectangle_timeframe'] = 'Daily' if interval == '1d' else 'Weekly'
    app.run()
    assert not app.exception
    assert app.dataframe[0].value.Ticker.tolist() == ['PRICE', 'ALL', 'VOL', 'OLD']
    assert len(app.checkbox) == 3 and not any(c.value for c in app.checkbox)
    app.checkbox(key=f'rectangle_higher_lows_{interval}').check().run()
    assert not app.exception
    assert app.dataframe[0].value.Ticker.tolist() == ['PRICE', 'ALL']
    app.checkbox(key=f'rectangle_tight_{interval}').check().run()
    assert not app.exception
    app.checkbox(key=f'rectangle_supportive_volume_{interval}').check().run()
    assert not app.exception
    assert app.dataframe[0].value.Ticker.tolist() == ['ALL']
    assert any('saved scan assessment' in c.value for c in app.caption)
    assert store.read_scan(interval) == before
