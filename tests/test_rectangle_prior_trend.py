from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from agentloop_trader.rectangle_scanner import (
    DETECTOR_VERSION, analyze_prior_trend, detect_rectangles, filter_rectangles,
    rectangle_from_record, rectangle_records,
)
from agentloop_trader.rectangle_timeframes import analysis_bars, latest_completed_date
from agentloop_trader.rectangle_store import RectangleStore
from test_rectangle_scanner import box_bars, NOW


def trend_history(rate, *, end='2026-10-19'):
    frame = box_bars(end=end)
    elapsed = (frame.index-frame.index[0]).days.to_numpy()/365.25
    close = 100*np.exp(rate*elapsed)
    frame['Close'], frame['Open'] = close, close*.999
    frame['High'], frame['Low'] = close*1.005, close*.995
    return frame


@pytest.mark.parametrize('interval', ['1d', '1w'])
@pytest.mark.parametrize('rate, direction', [(.4, 'Up'), (-.4, 'Down'), (0., 'Flat / mixed')])
def test_pre_range_context_classifies_trend_consistently_on_both_timeframes(interval, rate, direction):
    bars = analysis_bars(trend_history(rate), interval, NOW)
    base_start = bars.index[-(126 if interval == '1d' else 26)]
    result = analyze_prior_trend(bars, base_start, interval)
    assert result.direction == direction
    assert result.bonus > 0 if direction == 'Up' else result.bonus == 0
    assert 0 <= result.bonus <= 10
    assert pd.Timestamp(result.end) < base_start
    first = float(bars.loc[result.start, 'Close'])
    last = float(bars.loc[result.end, 'Close'])
    assert result.return_percent == pytest.approx((last/first-1)*100)


@pytest.mark.parametrize('interval', ['1d', '1w'])
def test_prices_inside_and_after_rectangle_cannot_influence_prior_trend(interval):
    bars = analysis_bars(trend_history(.4), interval, NOW)
    base_start = bars.index[-(126 if interval == '1d' else 26)]
    expected = analyze_prior_trend(bars, base_start, interval)
    changed = bars.copy()
    changed.loc[changed.index >= base_start, ['Open', 'High', 'Low', 'Close']] *= .01
    assert analyze_prior_trend(changed, base_start, interval) == expected
    assert pd.Timestamp(expected.end) < (base_start if interval == '1d' else base_start.to_period('W-FRI').start_time)


@pytest.mark.parametrize('interval', ['1d', '1w'])
def test_missing_or_short_prior_context_is_flagged_without_bonus(interval):
    bars = analysis_bars(trend_history(.4), interval, NOW)
    start = bars.index[-(126 if interval == '1d' else 26)]
    full = analyze_prior_trend(bars, start, interval)
    short = bars.loc[bars.index >= start-pd.DateOffset(months=4)]
    assert analyze_prior_trend(short, start, interval).direction == 'Insufficient history'
    # Remove one expected session/week within the six-month context.
    missing = bars.drop(bars.loc[full.start:full.end].index[10])
    result = analyze_prior_trend(missing, start, interval)
    assert result.direction == 'Insufficient history' and result.bonus == 0


def test_noise_and_small_moves_do_not_receive_uptrend_bonus():
    bars = analysis_bars(trend_history(.1), '1d', NOW)  # About 5% over six months.
    result = analyze_prior_trend(bars, bars.index[-126])
    assert result.direction == 'Flat / mixed' and result.bonus == 0
    base_start = bars.index[-126]
    boundary = bars.index < base_start
    bars.loc[boundary, 'Close'] = 100+30*np.sin(np.arange(boundary.sum())*2*np.pi/7)
    bars.loc[bars.index[boundary][-1], 'Close'] = 140  # An endpoint spike alone is not a consistent trend.
    result = analyze_prior_trend(bars, base_start)
    assert result.direction == 'Flat / mixed' and result.bonus == 0


@pytest.mark.parametrize('move, direction', [(10., 'Up'), (-10., 'Down')])
def test_exact_ten_percent_boundary_is_symmetric(move, direction):
    bars = analysis_bars(trend_history(0.), '1d', NOW)
    start = bars.index[-126]
    context = analyze_prior_trend(bars, start)
    mask = (bars.index >= pd.Timestamp(context.start)) & (bars.index <= pd.Timestamp(context.end))
    bars.loc[mask, 'Close'] = np.geomspace(100, 100+move, mask.sum())
    assert analyze_prior_trend(bars, start).direction == direction


def based_history(up, end='2026-10-19'):
    frame = box_bars(end=end)
    start = frame.index[-252]
    mask = frame.index < start
    days = (frame.index[mask]-frame.index[mask][-1]).days.to_numpy()
    prior_close = 100*np.exp(.5*days/365.25) if up else np.full(mask.sum(), 100.)
    frame.loc[mask, 'Close'] = prior_close
    frame.loc[mask, 'Open'] = prior_close*.999
    frame.loc[mask, 'High'] = prior_close*1.005
    frame.loc[mask, 'Low'] = prior_close*.995
    return frame


def test_prior_strength_changes_ranking_but_not_rectangle_geometry_or_eligibility(monkeypatch):
    import agentloop_trader.rectangle_scanner as scanner
    monkeypatch.setattr(scanner, 'WINDOWS', (252,))
    up = detect_rectangles('UP', based_history(True), now=NOW)[0]
    flat = detect_rectangles('FLAT', based_history(False), now=NOW)[0]
    assert (up.support, up.resistance, up.quality, up.containment) == (flat.support, flat.resistance, flat.quality, flat.containment)
    assert up.prior_trend == 'Up' and flat.prior_trend == 'Flat / mixed'
    assert up.ranking_score > flat.ranking_score
    assert filter_rectangles([flat, up])[0] == up
    assert rectangle_records([up])[0]['Prior trend'] == 'Up'
    # A modest prior-trend bonus cannot overwhelm a much cleaner rectangle.
    strong_shape = replace(flat, quality=100)
    weak_shape = replace(up, quality=80, prior_trend_bonus=10)
    assert filter_rectangles([weak_shape, strong_shape])[0] == strong_shape


def test_best_range_per_ticker_uses_combined_rank_and_ties_favor_shape(monkeypatch):
    import agentloop_trader.rectangle_scanner as scanner
    monkeypatch.setattr(scanner, 'WINDOWS', (252,))
    row = detect_rectangles('BOX', based_history(False), now=NOW)[0]
    a = replace(row, bars=126, quality=90, prior_trend='Up', prior_trend_bonus=8)
    b = replace(row, bars=252, quality=95, prior_trend_bonus=0)
    assert filter_rectangles([b, a]) == [a]
    # Same combined score: prefer the stronger rectangle structure.
    b = replace(b, quality=98)
    assert filter_rectangles([a, b]) == [b]


def test_legacy_records_remain_readable_and_prior_context_survives_save(tmp_path, monkeypatch):
    import agentloop_trader.rectangle_scanner as scanner
    monkeypatch.setattr(scanner, 'WINDOWS', (252,))
    row = detect_rectangles('BOX', based_history(True), now=NOW)[0]
    saved = row.record()
    assert rectangle_from_record(saved) == row
    legacy = {k: v for k, v in saved.items() if not k.startswith('prior_')}
    old = rectangle_from_record(legacy)
    assert old.prior_trend == 'Not assessed' and old.ranking_score == old.quality
    store = RectangleStore(tmp_path)
    store.save_watch(row, 'Alpaca IEX')
    assert rectangle_from_record(store.read('watchlist', {})['BOX']['rectangle']) == row


def test_table_and_chart_explain_prior_trend_without_rescanning(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import agentloop_trader.rectangle_scanner as scanner
    import agentloop_trader.rectangle_ui as ui
    from agentloop_trader.rectangle_data import DailyHistoryProvider, HISTORY_DAYS, history_bucket
    monkeypatch.setattr(scanner, 'WINDOWS', (252,))
    frame = based_history(True, end=latest_completed_date('1d').strftime('%Y-%m-%d'))
    row = detect_rectangles('BOX', frame)[0]
    store = RectangleStore(tmp_path)
    store.write_history('BOX', 'Alpaca IEX', frame, history_bucket(), history_days=HISTORY_DAYS)
    store.write_scan('1d', {'interval': '1d', 'detector_version': DETECTOR_VERSION,
        'scanned_at': pd.Timestamp.now(tz='UTC').isoformat(), 'source': 'Alpaca IEX',
        'rectangles': [row.record()], 'successful': 1, 'total': 1, 'skipped': 0,
        'errors': [], 'new_symbols': ['BOX'], 'data_start': row.end, 'data_end': row.end,
        'universe': {'directory_date': 'test'}, 'first_scan': True})
    def forbidden(*args, **kwargs):
        raise AssertionError('Viewing trend context must not start a scan or download.')
    monkeypatch.setattr(ui, 'start_range_scan', forbidden)
    monkeypatch.setattr(DailyHistoryProvider, 'fetch', forbidden)
    script = f'from agentloop_trader.rectangle_ui import render_rectangle_scanner\nrender_rectangle_scanner(store_root={str(tmp_path)!r})'
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert app.dataframe[0].value['Prior trend'].tolist() == ['Up']
    assert any('Prior trend: Up' in c.value and 'before this range' in c.value for c in app.caption)
    figure = ui.build_rectangle_chart(frame, row)
    assert figure.data[0].x[0] == row.prior_start
    assert figure.data[0].x[0] < row.start
