import pandas as pd

from agentloop_trader.scanner import ScannerCandidateStore, scan_universe, scanner_records


def _bars(symbol):
    index = pd.date_range("2026-01-01", periods=260, freq="h", tz="UTC")
    base = pd.Series(range(260), index=index).astype(float)
    data = pd.DataFrame(
        {
            "Close": 100 + base * 0.2,
            "High": 101 + base * 0.2,
            "Low": 99 + base * 0.2,
            "Volume": 1_000_000,
        },
        index=index,
    )
    data.attrs["symbol"] = symbol
    return data


def test_scan_universe_returns_sorted_records():
    settings = {
        "strategy_label": "Breakout continuation",
        "strategy_type": "breakout",
        "entry_window": 20,
        "exit_window": 10,
        "atr_stop_multiplier": 2.0,
        "risk_per_trade_pct": 1.0,
        "moving_average_window": 50,
        "pullback_average_length": 20,
        "momentum_turn_length": 5,
    }

    candidates, errors = scan_universe(["AAPL", "AAPL", "MSFT"], _bars, settings, 100000, max_symbols=5)
    records = scanner_records(candidates)

    assert errors == []
    assert [candidate.symbol for candidate in candidates] == ["AAPL", "MSFT"]
    assert records[0]["Ticker"] == "AAPL"
    assert "Current Read" in records[0]


def test_scanner_store_roundtrips_candidates(tmp_path):
    settings = {
        "strategy_label": "Breakout continuation",
        "strategy_type": "breakout",
        "entry_window": 20,
        "exit_window": 10,
        "atr_stop_multiplier": 2.0,
        "risk_per_trade_pct": 1.0,
        "moving_average_window": 50,
    }
    candidates, errors = scan_universe(["AAPL"], _bars, settings, 100000)
    store = ScannerCandidateStore(tmp_path / "scan.json")

    store.save(candidates, errors)
    loaded, loaded_errors = store.read()

    assert loaded_errors == []
    assert loaded[0].symbol == "AAPL"
