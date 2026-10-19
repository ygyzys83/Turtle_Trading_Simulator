from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from agentloop_trader.rectangle_scanner import (
    daily_bars, detect_rectangles, filter_rectangles, rectangle_from_record, refresh_saved_rectangle,
)


NOW = pd.Timestamp('2026-10-20 22:00:00Z')


def box_bars(length=756, end='2026-10-19', slope=0, scale=1):
    from agentloop_trader.rectangle_timeframes import session_schedule
    date = pd.Timestamp(end)
    index = session_schedule(date.year-int(length/200)-2, date.year).loc[:date].index[-length:]
    x = np.arange(length)
    close = (100 + 10*np.sin(x*2*np.pi/20) + slope*x)*scale
    return pd.DataFrame({'Open': close-.15*scale, 'High': close+.55*scale,
                         'Low': close-.55*scale, 'Close': close,
                         'Volume': 1_000_000}, index=index)


def test_detects_flat_ranges_at_multiple_lengths_and_counts_independent_tests():
    rows = detect_rectangles('EXAMPLE', box_bars(), now=NOW)
    assert rows
    assert min(r.months for r in rows) == 3
    assert max(r.months for r in rows) == 24
    assert all(88 < r.support < 92 and 108 < r.resistance < 112 for r in rows)
    assert all(len(r.support_touches) >= 2 and len(r.resistance_touches) >= 2 for r in rows)
    assert all((pd.to_datetime(r.support_touches).to_series().diff().dropna().dt.days >= 5).all() for r in rows)


def test_daily_scan_does_not_use_future_or_forming_bar_or_future_pivot_confirmation():
    data = box_bars(end='2026-10-20')
    morning = pd.Timestamp('2026-10-20 15:00Z')
    before = detect_rectangles('BOX', data.iloc[:-1], now=morning)
    data.loc[data.index[-1], ['Open', 'Close', 'High', 'Low']] = [900, 900, 901, 899]
    assert detect_rectangles('BOX', data, now=morning) == before
    extended = pd.concat([data, box_bars(10, end='2026-11-03')])
    assert detect_rectangles('BOX', extended, now=morning) == before
    assert all(pd.Timestamp(t) <= data.index[-4] for r in before for t in r.support_touches+r.resistance_touches)


@pytest.mark.parametrize('slope', [.15, -.15])
def test_sloping_channels_do_not_qualify_as_rectangles(slope):
    assert not detect_rectangles('CHANNEL', box_bars(length=420, slope=slope), now=NOW)


def test_recent_breakdown_does_not_get_recentered_into_a_new_box():
    data = box_bars()
    data.iloc[-1, data.columns.get_indexer(['Open', 'Close', 'High', 'Low'])] = [78, 77, 79, 76]
    assert not detect_rectangles('BROKEN', data, now=NOW)


def test_price_scale_does_not_change_detected_windows_or_score():
    a = detect_rectangles('BOX', box_bars(), now=NOW)
    b = detect_rectangles('BOX', box_bars(scale=10), now=NOW)
    assert [r.bars for r in a] == [r.bars for r in b]
    assert [r.quality for r in a] == [r.quality for r in b]
    assert b[0].support == pytest.approx(a[0].support*10)


def test_filters_choose_matching_duration_before_deduplicating_and_do_not_mutate():
    rows = detect_rectangles('BOX', box_bars(), now=NOW)
    a, b = replace(rows[0], bars=63, quality=99), replace(rows[0], bars=252, quality=80)
    assert filter_rectangles([a, b], (6, 18)) == [b]
    assert filter_rectangles([a, b], new_symbols=set()) == []
    assert filter_rectangles([a, b], new_symbols={'BOX'}) == [a]
    support = replace(a, last_close=a.support+.1*(a.resistance-a.support))
    assert filter_rectangles([support], location='Near support') == [support]
    assert not filter_rectangles([support], location='Near resistance')
    assert a.bars == 63


def test_watchlist_keeps_original_geometry_and_adjusts_basis_after_split():
    data = box_bars()
    original = detect_rectangles('BOX', data, now=NOW)[0]
    split = data.copy()
    split[['Open', 'High', 'Low', 'Close']] /= 2
    split.loc[split.index[-1]+pd.offsets.BDay(), :] = [62, 63, 61, 62, 2_000_000]
    current, note = refresh_saved_rectangle(original, original.last_close, split, now=NOW)
    assert current.support == pytest.approx(original.support/2)
    assert current.resistance == pytest.approx(original.resistance/2)
    assert current.location == 'Above range'
    assert 'corporate actions' in note
    assert original.support > current.support
    assert rectangle_from_record(original.record()) == original
    with pytest.raises(ValueError, match='missing'):
        refresh_saved_rectangle(original, original.last_close, split.iloc[-1:], now=NOW)


def test_short_invalid_or_sparse_history_is_not_a_rectangle():
    assert not detect_rectangles('SHORT', box_bars(20), now=NOW)
    bad = box_bars()
    bad['High'] = bad['Low'] - 1
    with pytest.raises(ValueError):
        detect_rectangles('BAD', bad, now=NOW)
    sparse = box_bars(50)
    sparse.index = pd.date_range('2022-01-01', periods=50, freq='MS')
    assert not detect_rectangles('SPARSE', sparse, now=NOW)


def test_daily_timestamp_normalization_preserves_exchange_session_date():
    data = box_bars()
    data.index = data.index.tz_localize('America/New_York').tz_convert('UTC')
    assert daily_bars(data, NOW).index[-1] == pd.Timestamp('2026-10-19')


def _containment_fixture(outside_count):
    # Stable independent boundaries at 90/110. A short excursion is inside the
    # old padded zone, but its closes are outside the rectangle actually drawn.
    from agentloop_trader.rectangle_timeframes import session_schedule
    index = session_schedule(2025, 2026).loc[:'2026-10-19'].index[-100:]
    frame = pd.DataFrame({'Open': 100., 'High': 101., 'Low': 99.,
                          'Close': 100., 'Volume': 1_000_000}, index=index)
    for i in (5, 25, 45, 65, 85):
        frame.iloc[i-1:i+2, frame.columns.get_loc('High')] = [109.9, 110., 109.9]
        frame.iloc[i-1:i+2, frame.columns.get_loc('Close')] = 109.
    for i in (15, 35, 55, 75, 95):
        frame.iloc[i-1:i+2, frame.columns.get_loc('Low')] = [90.1, 90., 90.1]
        frame.iloc[i-1:i+2, frame.columns.get_loc('Close')] = 91.
    frame.iloc[7:7+outside_count, frame.columns.get_indexer(['Open', 'High', 'Low', 'Close'])] = [109.8, 110.2, 109.5, 110.1]
    return frame


@pytest.mark.parametrize('outside_count, qualifies', [(0, True), (5, True), (6, True), (10, True), (11, False)])
def test_90_percent_gate_counts_closes_inside_drawn_box_not_padded_zones(monkeypatch, outside_count, qualifies):
    import agentloop_trader.rectangle_scanner as scanner
    monkeypatch.setattr(scanner, 'WINDOWS', (100,))
    frame = _containment_fixture(outside_count)
    rows = scanner.detect_rectangles('BOX', frame, now=NOW)
    assert bool(rows) is qualifies
    if qualifies:
        row = rows[0]
        expected = float(((frame.Close >= row.support) & (frame.Close <= row.resistance)).mean())
        assert row.containment == pytest.approx(expected)
        assert row.containment == pytest.approx(1-outside_count/100)


def test_95_percent_bonus_improves_ranking_without_becoming_a_requirement():
    from agentloop_trader.rectangle_scanner import _rectangle_quality
    typical = dict(edge_drift=.1, touch_count=6, span=.5)
    below = _rectangle_quality(.94, **typical)
    clean = _rectangle_quality(.95, **typical)
    assert clean-below > 5  # Bonus in addition to the continuous containment contribution.
    assert _rectangle_quality(1., 0., 10, .7) == 100
    assert _rectangle_quality(.90, 0., 10, .7) > _rectangle_quality(.95, .25, 4, .35)


def test_displayed_best_rectangle_uses_cleanliness_bonus(monkeypatch):
    import agentloop_trader.rectangle_scanner as scanner
    monkeypatch.setattr(scanner, 'WINDOWS', (100,))
    row = scanner.detect_rectangles('BOX', _containment_fixture(0), now=NOW)[0]
    typical = dict(edge_drift=.1, touch_count=6, span=.5)
    acceptable = replace(row, bars=126, containment=.94, quality=scanner._rectangle_quality(.94, **typical))
    cleaner = replace(row, bars=84, containment=.95, quality=scanner._rectangle_quality(.95, **typical))
    assert scanner.filter_rectangles([acceptable, cleaner]) == [cleaner]
