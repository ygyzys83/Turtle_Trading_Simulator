from concurrent.futures import ThreadPoolExecutor

import pytest

from agentloop_trader.buy_watchlist import (
    MAX_BUY_WATCHLIST_ITEMS,
    BuyWatchPlan,
    BuyWatchlistStore,
    buy_watch_plan_id,
    buy_watch_plan_detail_records,
    buy_watch_plan_sidebar_inputs,
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


def test_removing_one_ticker_setup_keeps_the_other_setup(tmp_path):
    store = BuyWatchlistStore(tmp_path / "watchlist.json")
    older = store.upsert(_plan(symbol="TSLA", strategy="Breakout continuation"))
    newer = store.upsert(_plan(symbol="TSLA", strategy="Trend pullback continuation"))

    assert store.remove(older.plan_id) is True

    remaining = store.read()
    assert [plan.plan_id for plan in remaining] == [newer.plan_id]
    assert remaining[0].symbol == "TSLA"


def test_watchlist_enforces_maximum_queued_setups(tmp_path):
    store = BuyWatchlistStore(tmp_path / "watchlist.json")
    for index in range(MAX_BUY_WATCHLIST_ITEMS):
        store.upsert(_plan(symbol=f"T{index}"))

    with pytest.raises(ValueError, match=str(MAX_BUY_WATCHLIST_ITEMS)):
        store.upsert(_plan(symbol="TOO-MANY"))


def test_invalid_watchlist_is_not_silently_treated_as_empty(tmp_path):
    path = tmp_path / "watchlist.json"
    path.write_text('[{"plan_id": "saved-plan"}]]', encoding="utf-8")
    store = BuyWatchlistStore(path)

    with pytest.raises(RuntimeError, match="was not treated as empty"):
        store.read()


def test_concurrent_watchlist_updates_preserve_every_setup(tmp_path):
    path = tmp_path / "watchlist.json"
    plans = [_plan(symbol=f"T{index}") for index in range(8)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda plan: BuyWatchlistStore(path).upsert(plan), plans))

    saved = BuyWatchlistStore(path).read()
    assert {plan.symbol for plan in saved} == {plan.symbol for plan in plans}
    assert not list(tmp_path.glob("*.tmp"))


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


def test_saved_setup_can_restore_matching_sidebar_inputs():
    plan = BuyWatchPlan(
        plan_id=buy_watch_plan_id("TSLA", "4h", "Trend pullback continuation"),
        symbol="TSLA",
        interval="4h",
        history="5y",
        price_data_source="Ticker (Alpaca)",
        strategy_label="Trend pullback continuation",
        strategy_settings={
            "account_size": 125_000,
            "strategy_label": "Trend pullback continuation",
            "entry_window": 30,
            "exit_window": 20,
            "atr_stop_multiplier": 1.75,
            "risk_per_trade_pct": 0.5,
            "moving_average_window": 100,
            "pullback_average_length": 50,
            "momentum_turn_length": 12,
            "automation_refresh_seconds": 30,
            "auto_cancel_stale_limit_orders": True,
            "stale_limit_order_minutes": 120,
            "allow_limit_buys_outside_market_hours": True,
            "reentry_cooldown_minutes": 120,
        },
        risk_limits={
            "allowed_symbols": ["TSLA", "NVDA"],
            "max_risk_per_trade_pct": 0.75,
            "max_position_notional_pct": 10.0,
            "max_portfolio_exposure_pct": 70.0,
            "max_symbol_concentration_pct": 10.0,
            "max_session_loss_pct": 2.5,
            "max_open_positions": 12,
            "allow_add_to_existing_position": True,
        },
        order_style="Limit below current price",
        limit_adjustment_pct=0.35,
    )

    values = buy_watch_plan_sidebar_inputs(plan)

    assert values["ticker_or_pair_input"] == "TSLA"
    assert values["price_interval_input"] == "4h"
    assert values["history_period_input"] == "5y"
    assert values["strategy_label_input"] == "Trend pullback continuation"
    assert values["pullback_average_length_input"] == 50
    assert values["atr_stop_multiplier_input"] == 1.75
    assert values["max_risk_limit_input"] == 0.75
    assert values["allowed_symbols_input"] == "TSLA, NVDA"
    assert values["paper_buy_order_style_input"] == "Limit below current price"
    assert values["paper_buy_limit_adjustment_pct_input"] == 0.35
    assert values["stale_limit_order_label_input"] == "2 hours"


def test_watchlist_records_show_repeat_after_exit_state():
    repeating = _plan(symbol="TSLA")
    repeating = BuyWatchPlan(**{**repeating.__dict__, "repeat_after_exit": True})

    record = buy_watchlist_records([repeating])[0]

    assert record["Repeat After Exit"] == "On"


def test_watchlist_records_show_current_buy_level_and_distance():
    plan = _plan(symbol="TSLA")
    plan = BuyWatchPlan(**{
        **plan.__dict__,
        "latest_price": 200.0,
        "next_buy_level": 210.0,
        "distance_to_buy_pct": 5.0,
    })

    record = buy_watchlist_records([plan])[0]

    assert record["Current Price"] == "$200.00"
    assert record["Next BUY Level"] == "$210.00"
    assert record["Distance To BUY"] == "+5.00%"


def test_saved_rsi_setup_shows_profit_only_exit_setting():
    plan = BuyWatchPlan(
        plan_id=buy_watch_plan_id("BTC/USD", "5m", "RSI mean-reversion scalp"),
        symbol="BTC/USD",
        interval="5m",
        history="1mo",
        price_data_source="Crypto (Alpaca)",
        strategy_label="RSI mean-reversion scalp",
        strategy_settings={"strategy_type": "rsi_scalp", "rsi_profit_only_exit": True},
    )

    by_input = {row["Input"]: row["Saved Value"] for row in buy_watch_plan_detail_records(plan)}

    assert by_input["Require profit for RSI exit"] == "On"
