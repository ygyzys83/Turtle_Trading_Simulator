from datetime import UTC, datetime
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from agentloop_trader.assets import floor_quantity, normalize_symbol
from agentloop_trader.automation_runtime import AutomationControl
from agentloop_trader.backtest import _backtest_limited_quantity
from agentloop_trader.broker_governance import build_exit_intent_from_position
from agentloop_trader.brokers import AlpacaConfig, build_alpaca_order_preview
from agentloop_trader.buy_watchlist import BuyWatchPlan, BuyWatchlistStore, buy_watch_plan_id
from agentloop_trader.fees import (
    alpaca_crypto_fee_rate_percent,
    estimate_alpaca_crypto_order_fees,
)
from agentloop_trader.market_data import (
    completed_price_bars,
    fetch_alpaca_crypto_bars,
    fetch_alpaca_latest_crypto_trades,
)
from agentloop_trader.models import ExecutionDecision, RiskCheckResult, RiskLimits, TradeIntent
from agentloop_trader.risk import constrain_trade_intent_to_limits
from agentloop_trader.worker import _send_entry


def approved_decision() -> ExecutionDecision:
    risk = RiskCheckResult(True, [], {"approved": True})
    return ExecutionDecision("paper", True, False, "Approved", risk)


def test_crypto_symbols_and_quantity_precision_are_normalized():
    assert normalize_symbol("btcusd", "crypto") == "BTC/USD"
    assert normalize_symbol("ETH-USD", "crypto") == "ETH/USD"
    assert floor_quantity(0.123456789, "crypto") == pytest.approx(0.12345678)
    assert floor_quantity(9.9, "equity") == 9


def test_alpaca_crypto_fee_tiers_use_maker_and_taker_rates():
    assert alpaca_crypto_fee_rate_percent(liquidity="maker", trailing_30d_volume=0) == 0.15
    assert alpaca_crypto_fee_rate_percent(liquidity="taker", trailing_30d_volume=0) == 0.25
    assert alpaca_crypto_fee_rate_percent(liquidity="taker", trailing_30d_volume=600_000) == 0.20
    fees = estimate_alpaca_crypto_order_fees(side="buy", quantity=0.1, price=50_000)
    assert fees.trade_value == 5_000
    assert fees.total == 12.50


def test_crypto_bars_complete_on_weekends_without_stock_session_rules():
    data = pd.DataFrame(
        {"Open": [100, 101], "High": [102, 103], "Low": [99, 100], "Close": [101, 102]},
        index=pd.to_datetime(["2026-07-12T10:00:00Z", "2026-07-12T11:00:00Z"]),
    )
    data.attrs.update({"symbol": "BTC/USD", "asset_class": "crypto"})
    completed = completed_price_bars(data, "1h", now=datetime(2026, 7, 12, 11, 30, tzinfo=UTC))
    assert len(completed) == 1
    assert completed.attrs["latest_price"] == 102


def test_alpaca_crypto_data_uses_crypto_endpoint_and_pair(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def read(self):
            payload = {"bars": {"BTC/USD": [{"t": "2026-07-10T10:00:00Z", "o": 50000, "h": 51000, "l": 49000, "c": 50500, "v": 2}]}}
            return json.dumps(payload).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return Response()

    monkeypatch.setattr("agentloop_trader.market_data.urlopen", fake_urlopen)
    bars = fetch_alpaca_crypto_bars("BTCUSD", "1mo", "1h", "key", "secret")
    assert "/v1beta3/crypto/us/bars" in captured["url"]
    assert "symbols=BTC%2FUSD" in captured["url"]
    assert bars.attrs["asset_class"] == "crypto"


def test_four_hour_crypto_data_aggregates_supported_hourly_bars(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def read(self):
            bars = [
                {
                    "t": f"2026-07-10T{hour:02d}:00:00Z",
                    "o": 100 + hour,
                    "h": 102 + hour,
                    "l": 99 + hour,
                    "c": 101 + hour,
                    "v": hour + 1,
                }
                for hour in range(8)
            ]
            return json.dumps({"bars": {"BTC/USD": bars}}).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return Response()

    monkeypatch.setattr("agentloop_trader.market_data.urlopen", fake_urlopen)

    bars = fetch_alpaca_crypto_bars("BTC/USD", "2y", "4h", "key", "secret")

    assert "timeframe=1Hour" in captured["url"]
    assert "timeframe=4Hour" not in captured["url"]
    assert len(bars) == 2
    assert bars.iloc[0].to_dict() == {
        "Open": 100.0,
        "Close": 104.0,
        "High": 105.0,
        "Low": 99.0,
        "Volume": 10.0,
    }
    assert bars.iloc[1]["Volume"] == 26


def test_latest_crypto_trade_uses_crypto_endpoint(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def read(self): return json.dumps({"trades": {"BTC/USD": {"p": 60001.25}}}).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return Response()

    monkeypatch.setattr("agentloop_trader.market_data.urlopen", fake_urlopen)
    result = fetch_alpaca_latest_crypto_trades(["BTCUSD"], "key", "secret")
    assert result == {"BTC/USD": 60001.25}
    assert "/v1beta3/crypto/us/latest/trades" in captured["url"]


def test_crypto_backtest_and_live_risk_sizing_keep_fractional_quantity():
    quantity = _backtest_limited_quantity(
        raw_quantity=0.25,
        account_equity=100_000,
        entry_price=60_000,
        stop_price=58_000,
        session_pnl=0,
        limits=RiskLimits(max_risk_per_trade_pct=0.5, max_position_notional_pct=5, max_symbol_concentration_pct=5),
        asset_class="crypto",
    )
    assert 0 < quantity < 0.1

    intent = TradeIntent(
        symbol="BTC/USD", side="buy", quantity=0.25, asset_class="crypto",
        time_in_force="gtc", entry_price=60_000, stop_loss=58_000,
    )
    constrained = constrain_trade_intent_to_limits(
        intent, 100_000, RiskLimits(max_risk_per_trade_pct=0.5, max_position_notional_pct=5, max_symbol_concentration_pct=5),
    )
    assert constrained is not None
    assert 0 < constrained.quantity < 0.1


def test_crypto_preview_uses_gtc_and_crypto_fees():
    intent = TradeIntent(
        symbol="BTC/USD", side="buy", quantity=0.05, asset_class="crypto",
        order_type="limit", time_in_force="gtc", limit_price=60_000, entry_price=60_000, stop_loss=58_000,
    )
    preview = build_alpaca_order_preview(intent, approved_decision(), AlpacaConfig("key", "secret", paper=True))
    assert preview.valid
    assert preview.order["asset_class"] == "crypto"
    assert preview.order["estimated_alpaca_fees"] == "$7.50"
    assert preview.order["possible_maker_fee"] == "$4.50"

    invalid = build_alpaca_order_preview(
        TradeIntent(symbol="BTC/USD", side="buy", quantity=0.05, asset_class="crypto", entry_price=60_000),
        approved_decision(),
        AlpacaConfig("key", "secret", paper=True),
    )
    assert not invalid.valid
    assert any("GTC or IOC" in reason for reason in invalid.blocked_reasons)


def test_crypto_exit_preserves_fractional_quantity_and_uses_gtc():
    intent = build_exit_intent_from_position({"Symbol": "BTCUSD", "Asset Type": "crypto", "Quantity": "0.1234"})
    assert intent is not None
    assert intent.symbol == "BTC/USD"
    assert intent.quantity == pytest.approx(0.1234)
    assert intent.time_in_force == "gtc"


def test_crypto_worker_entry_does_not_wait_for_stock_market_hours(monkeypatch):
    called = {}
    control = AutomationControl(
        mode="Auto entries and exits", full_automation_enabled=True, paper_orders_enabled=True,
        symbol="BTC/USD", asset_class="crypto", price_data_source="Crypto (Alpaca)",
    )
    adapter = SimpleNamespace(
        config=AlpacaConfig("key", "secret", paper=True),
        account_records=lambda: [
            {"Field": "Portfolio Value", "Value": "100000"},
            {"Field": "Cash", "Value": "100000"},
            {"Field": "Last Equity", "Value": "100000"},
        ],
        market_is_open=lambda: False,
    )

    def selected(*args, **kwargs):
        called["evaluated"] = True
        return {"live": {"trade_intent": None}}

    monkeypatch.setattr("agentloop_trader.worker.selected_strategy_result", selected)
    sent, _, message = _send_entry(control, adapter, [], [], [], lambda *_: object(), SimpleNamespace(append=lambda event: None))
    assert called["evaluated"] is True
    assert sent == 0
    assert message == "No BUY setup right now."


def test_crypto_watchlist_persists_asset_type(tmp_path):
    store = BuyWatchlistStore(tmp_path / "watchlist.json")
    plan = BuyWatchPlan(
        plan_id=buy_watch_plan_id("BTC/USD", "1h", "Breakout continuation", "crypto"),
        symbol="BTC/USD", interval="1h", history="1y", price_data_source="Crypto (Alpaca)",
        strategy_label="Breakout continuation", asset_class="crypto",
    )
    store.upsert(plan)
    assert store.read()[0].asset_class == "crypto"
