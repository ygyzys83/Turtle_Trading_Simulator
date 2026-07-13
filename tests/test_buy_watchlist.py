import pytest

from agentloop_trader.buy_watchlist import (
    MAX_BUY_WATCHLIST_ITEMS,
    BuyWatchPlan,
    BuyWatchlistStore,
    buy_watch_plan_id,
    buy_watch_plan_detail_records,
    buy_watchlist_records,
)


def _plan(symbol="AAPL", interval="1h", strategy="Breakout continuation"):
    return BuyWatchPlan(
        plan_id=buy_watch_plan_id(symbol, interval, strategy),
        symbol=symbol,
        interval=interval,
        history="2y",
        price_data_source="Ticker (Alpaca)",
        strategy_label=strategy,
        strategy_settings={"strategy_label": strategy},
    )


def test_watchlist_upserts_one_ticker_interval_strategy_setup(tmp_path):
    store = BuyWatchlistStore(tmp_path / "watchlist.json")

    store.upsert(_plan())
    store.upsert(_plan())

    plans = store.read()
    assert len(plans) == 1
    assert plans[0].symbol == "AAPL"
    assert buy_watchlist_records(plans)[0]["Status"] == "Waiting for BUY"


def test_watchlist_allows_two_strategies_for_one_ticker(tmp_path):
    store = BuyWatchlistStore(tmp_path / "watchlist.json")

    store.upsert(_plan(strategy="Breakout continuation"))
    store.upsert(_plan(strategy="Trend pullback continuation"))

    assert len(store.read()) == 2


def test_watchlist_enforces_maximum_queued_setups(tmp_path):
    store = BuyWatchlistStore(tmp_path / "watchlist.json")
    for index in range(MAX_BUY_WATCHLIST_ITEMS):
        store.upsert(_plan(symbol=f"T{index}"))

    with pytest.raises(ValueError, match=str(MAX_BUY_WATCHLIST_ITEMS)):
        store.upsert(_plan(symbol="TOO-MANY"))


def test_saved_setup_details_show_strategy_risk_order_and_exit_inputs():
    plan = BuyWatchPlan(
        plan_id=buy_watch_plan_id("TSLA", "4h", "Trend pullback continuation"),
        symbol="TSLA",
        interval="4h",
        history="5y",
        price_data_source="Ticker (Alpaca)",
        strategy_label="Trend pullback continuation",
        strategy_settings={
            "entry_window": 20,
            "exit_window": 10,
            "atr_stop_multiplier": 2.5,
            "risk_per_trade_pct": 1.0,
            "moving_average_window": 50,
            "pullback_average_length": 20,
            "momentum_turn_length": 5,
            "rsi_entry_filter_enabled": True,
            "breakeven_after_r": 1.0,
            "trail_after_r": 2.0,
            "trailing_atr_multiplier": 3.0,
        },
        risk_limits={
            "max_risk_per_trade_pct": 1.0,
            "max_position_notional_pct": 5.0,
            "max_portfolio_exposure_pct": 80.0,
            "max_symbol_concentration_pct": 5.0,
            "max_session_loss_pct": 2.0,
            "max_open_positions": 20,
        },
        order_style="Limit below current price",
        limit_adjustment_pct=0.25,
        repeat_after_exit=True,
    )

    rows = buy_watch_plan_detail_records(plan)
    by_input = {row["Input"]: row["Saved Value"] for row in rows}

    assert by_input["Sell exit length"] == "10 bars"
    assert by_input["Stop distance"] == "2.5 ATR"
    assert by_input["RSI 50-70 BUY rule"] == "On"
    assert by_input["Max new order size"] == "5.0%"
    assert by_input["Buy limit discount"] == "0.25%"
    assert by_input["Start ATR trail after"] == "2.0R"
    assert by_input["Repeat after exit"] == "On"
    assert by_input["Order reference price"] == "Latest available Alpaca IEX trade at execution"


def test_watchlist_records_show_repeat_after_exit_state():
    repeating = _plan(symbol="TSLA")
    repeating = BuyWatchPlan(**{**repeating.__dict__, "repeat_after_exit": True})

    record = buy_watchlist_records([repeating])[0]

    assert record["Repeat After Exit"] == "On"
