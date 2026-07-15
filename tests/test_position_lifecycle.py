import pandas as pd
import pytest

from agentloop_trader.position_lifecycle import (
    current_position_cycle,
    initialize_exit_settings_for_position,
    replace_exit_rules,
    resolve_position_plan,
    synchronize_position_plans,
    upsert_position_plan,
)
from agentloop_trader.strategy_runtime import evaluate_exit_settings


def _order(
    order_id: str,
    side: str,
    quantity: float,
    filled_at: str,
    *,
    symbol: str = "NVDA",
    status: str = "filled",
    average_fill: float | None = None,
) -> dict:
    return {
        "Alpaca Order ID": order_id,
        "Symbol": symbol,
        "Side": f"OrderSide.{side.upper()}",
        "Status": f"OrderStatus.{status.upper()}",
        "Filled Qty": str(quantity if status in {"filled", "partially_filled"} else 0),
        "Avg Fill": str(average_fill) if average_fill is not None else "",
        "Submitted": filled_at,
        "Filled": filled_at if status in {"filled", "partially_filled"} else "",
    }


def _position(quantity: float, average_entry: float, symbol: str = "NVDA") -> dict:
    return {
        "Symbol": symbol,
        "Asset Type": "equity",
        "Quantity": str(quantity),
        "Average Entry": str(average_entry),
    }


def _tracked(order_id: str, *, high: float | None = None, status: str = "filled") -> dict:
    settings = {
        "symbol": "NVDA",
        "interval": "1h",
        "entry_stop_distance": 6.78,
        "entry_reference_price": 211.58,
        "entry_stop_loss": 204.8,
        "auto_exit_enabled": True,
    }
    if high is not None:
        settings["highest_high_since_entry"] = high
        settings["last_exit_trigger_price"] = high - 1
        settings["last_exit_trigger_source"] = "break-even stop"
    return {
        "broker_order_id": order_id,
        "symbol": "NVDA",
        "side": "buy",
        "status": status,
        "strategy_settings": dict(settings),
        "exit_settings": dict(settings),
    }


def test_full_exit_then_reentry_creates_a_new_position_cycle():
    orders = [
        _order("old-buy", "buy", 24, "2026-07-09T16:33:04+00:00", average_fill=202.48),
        _order("old-sell", "sell", 24, "2026-07-10T19:00:59+00:00", average_fill=210.15),
        _order("new-buy", "buy", 23, "2026-07-15T19:20:40+00:00", average_fill=210.758696),
    ]

    cycle = current_position_cycle(_position(23, 210.758696), orders)

    assert cycle.reliable
    assert cycle.cycle_id == "new-buy"
    assert cycle.basis_order_id == "new-buy"
    assert cycle.buy_order_ids == ("new-buy",)


def test_replacing_exit_rules_preserves_cycle_entry_and_dynamic_state():
    existing = {
        "position_cycle_id": "new-buy",
        "position_basis_order_id": "new-buy",
        "position_basis_filled_quantity": 23,
        "entry_reference_price": 210.758696,
        "entry_atr": 4.52,
        "highest_high_since_entry": 214.0,
        "risk_per_trade_pct": 0.25,
        "risk_limits_at_entry": {"max_risk_per_trade_pct": 0.5},
        "exit_window": 30,
    }

    replaced = replace_exit_rules(existing, {"exit_window": 10, "entry_reference_price": 999})

    assert replaced["exit_window"] == 10
    assert replaced["entry_reference_price"] == pytest.approx(210.758696)
    assert replaced["highest_high_since_entry"] == 214.0
    assert replaced["position_cycle_id"] == "new-buy"
    assert replaced["risk_limits_at_entry"] == {"max_risk_per_trade_pct": 0.5}


def test_durable_filled_order_can_reconstruct_cycle_outside_fresh_order_window():
    tracked = _tracked("older-current-buy")
    tracked.update({
        "status": "filled",
        "filled_quantity": "23",
        "average_fill_price": "210.758696",
        "submitted_at": "2026-07-15T19:20:37+00:00",
        "filled_at": "2026-07-15T19:20:40+00:00",
    })

    resolution = resolve_position_plan(_position(23, 210.758696), [], [tracked])

    assert resolution.cycle.reliable
    assert resolution.cycle.cycle_id == "older-current-buy"
    assert resolution.exit_settings["entry_reference_price"] == pytest.approx(210.758696)


def test_live_fill_uses_new_order_settings_even_when_local_status_is_stale():
    orders = [
        _order("old-buy", "buy", 24, "2026-07-09T16:33:04+00:00", average_fill=202.48),
        _order("old-sell", "sell", 24, "2026-07-10T19:00:59+00:00", average_fill=210.15),
        _order("new-buy", "buy", 23, "2026-07-15T19:20:40+00:00", average_fill=210.758696),
    ]
    tracked = [
        _tracked("old-buy", high=229.42),
        _tracked("new-buy", status="new"),
    ]

    resolution = resolve_position_plan(_position(23, 210.758696), orders, tracked)

    assert resolution.managed
    assert resolution.settings_source_order_id == "new-buy"
    assert resolution.entry_settings["entry_broker_order_id"] == "new-buy"
    assert resolution.entry_settings["entry_reference_price"] == 210.758696
    assert resolution.exit_settings["highest_high_since_entry"] == 210.758696
    assert "last_exit_trigger_price" not in resolution.exit_settings


def test_old_cycle_settings_never_attach_when_new_buy_is_manual():
    orders = [
        _order("old-buy", "buy", 24, "2026-07-09T16:33:04+00:00"),
        _order("old-sell", "sell", 24, "2026-07-10T19:00:59+00:00"),
        _order("manual-buy", "buy", 23, "2026-07-15T19:20:40+00:00"),
    ]

    resolution = resolve_position_plan(
        _position(23, 210.75),
        orders,
        [_tracked("old-buy", high=229.42)],
    )

    assert not resolution.managed
    assert resolution.entry_settings is None
    assert resolution.exit_settings is None
    assert "no saved" in resolution.reason.lower()


def test_canceled_buy_never_starts_a_position_cycle():
    orders = [
        _order("canceled", "buy", 10, "2026-07-14T10:00:00+00:00", status="canceled"),
        _order("filled", "buy", 5, "2026-07-15T10:00:00+00:00"),
    ]

    cycle = current_position_cycle(_position(5, 100), orders)

    assert cycle.cycle_id == "filled"
    assert cycle.buy_order_ids == ("filled",)


def test_add_on_rebases_the_position_plan_to_combined_average_and_latest_fill():
    orders = [
        _order("first-buy", "buy", 40, "2026-07-15T10:00:00+00:00", average_fill=100),
        _order("add-on", "buy", 60, "2026-07-15T12:00:00+00:00", average_fill=110),
    ]
    tracked = [_tracked("first-buy", high=130)]

    resolution = resolve_position_plan(_position(100, 106), orders, tracked)

    assert resolution.managed
    assert resolution.cycle.cycle_id == "first-buy"
    assert resolution.cycle.basis_order_id == "add-on"
    assert resolution.exit_settings["entry_broker_order_id"] == "add-on"
    assert resolution.exit_settings["entry_reference_price"] == 106
    assert resolution.exit_settings["entry_stop_loss"] == 99.22
    assert resolution.exit_settings["highest_high_since_entry"] == 106
    assert "last_exit_trigger_price" not in resolution.exit_settings


def test_partial_sell_keeps_the_same_position_cycle():
    orders = [
        _order("buy", "buy", 100, "2026-07-15T10:00:00+00:00"),
        _order("partial-sell", "sell", 40, "2026-07-15T11:00:00+00:00"),
    ]

    cycle = current_position_cycle(_position(60, 100), orders)

    assert cycle.reliable
    assert cycle.cycle_id == "buy"
    assert cycle.reconstructed_quantity == 60


def test_additional_partial_fill_rebases_same_order_without_using_pre_fill_high():
    partial_order = _order(
        "one-order", "buy", 10, "2026-07-15T10:00:00+00:00",
        status="partially_filled", average_fill=100,
    )
    partial_position = _position(10, 100)
    first = upsert_position_plan(
        partial_position,
        [partial_order],
        [_tracked("one-order", high=105, status="new")],
        {
            "entry_stop_distance": 5.0,
            "highest_high_since_entry": 105.0,
            "last_exit_trigger_price": 100.0,
            "auto_exit_enabled": True,
        },
    )
    filled_order = _order("one-order", "buy", 20, "2026-07-15T10:01:00+00:00", average_fill=102)

    resolution = resolve_position_plan(_position(20, 102), [filled_order], first)

    assert resolution.exit_settings["position_basis_order_id"] == "one-order"
    assert resolution.exit_settings["position_basis_filled_quantity"] == 20
    assert resolution.exit_settings["entry_reference_price"] == 102
    assert resolution.exit_settings["entry_stop_loss"] == 97
    assert resolution.exit_settings["highest_high_since_entry"] == 102
    assert "last_exit_trigger_price" not in resolution.exit_settings


def test_partial_fill_without_final_fill_time_uses_submission_as_entry_boundary():
    order = _order(
        "partial", "buy", 10, "2026-07-15T10:00:00+00:00",
        status="partially_filled", average_fill=100,
    )
    order["Filled"] = ""

    cycle = current_position_cycle(_position(10, 100), [order])

    assert cycle.reliable
    assert cycle.started_at == "2026-07-15T10:00:00+00:00"
    assert cycle.basis_filled_at == "2026-07-15T10:00:00+00:00"


def test_partial_sell_does_not_reset_high_water_mark():
    orders = [
        _order("buy", "buy", 100, "2026-07-15T10:00:00+00:00", average_fill=100),
        _order("partial-sell", "sell", 40, "2026-07-15T11:00:00+00:00", average_fill=110),
    ]
    tracked = [_tracked("buy")]
    first = resolve_position_plan(_position(100, 100), [orders[0]], tracked)
    plan = upsert_position_plan(_position(100, 100), [orders[0]], tracked, first.exit_settings, first.entry_settings)
    position_plan = next(row for row in plan if row.get("source") == "position_plan")
    position_plan["exit_settings"].update({
        "highest_high_since_entry": 120,
        "last_exit_trigger_price": 119,
    })
    after_sell = resolve_position_plan(_position(60, 100), orders, plan)

    assert after_sell.exit_settings["highest_high_since_entry"] == 120
    assert after_sell.exit_settings["last_exit_trigger_price"] == 119


def test_managed_add_on_tightens_combined_stop_to_saved_risk_budget():
    first_order = _order("first-buy", "buy", 50, "2026-07-15T10:00:00+00:00", average_fill=100)
    first_position = _position(50, 100)
    initial = upsert_position_plan(
        first_position,
        [first_order],
        [_tracked("first-buy")],
        {
            "entry_stop_distance": 8.0,
            "auto_exit_enabled": True,
            "risk_per_trade_pct": 0.5,
            "sizing_account_equity": 100_000,
            "risk_limits_at_entry": {"max_risk_per_trade_pct": 0.5},
        },
    )
    position_plan = next(row for row in initial if row.get("source") == "position_plan")
    position_plan["exit_settings"].update({
        "highest_high_since_entry": 110.0,
        "last_exit_trigger_price": 105.0,
        "last_exit_trigger_source": "ATR trail",
    })
    add_on = _order("add-on", "buy", 50, "2026-07-15T12:00:00+00:00", average_fill=110)

    resolution = resolve_position_plan(_position(100, 105), [first_order, add_on], initial)

    assert resolution.exit_settings["entry_reference_price"] == 105
    assert resolution.exit_settings["entry_stop_distance"] == 5.0
    assert resolution.exit_settings["entry_stop_loss"] == 100
    assert resolution.exit_settings["position_risk_budget"] == 500
    assert resolution.exit_settings["position_risk_at_initial_stop"] == 500
    assert resolution.exit_settings["highest_high_since_entry"] == 110
    assert resolution.exit_settings["last_exit_trigger_price"] == 105


def test_quantity_mismatch_fails_closed():
    orders = [_order("buy", "buy", 20, "2026-07-15T10:00:00+00:00")]

    resolution = resolve_position_plan(_position(23, 100), orders, [_tracked("buy")])

    assert not resolution.cycle.reliable
    assert not resolution.managed
    assert resolution.exit_settings is None
    assert "reconstructs 20" in resolution.reason


def test_position_plan_record_is_keyed_to_cycle_and_preserves_entry_snapshot():
    orders = [_order("new-buy", "buy", 23, "2026-07-15T19:20:40+00:00", average_fill=210.758696)]
    tracked = [_tracked("new-buy", status="new")]
    position = _position(23, 210.758696)

    updated = upsert_position_plan(
        position,
        orders,
        tracked,
        {"entry_stop_distance": 6.78, "auto_exit_enabled": True, "interval": "1h"},
    )
    resolution = resolve_position_plan(position, orders, updated)

    assert resolution.plan_record_id == "position-plan-new-buy"
    assert resolution.managed
    assert resolution.entry_settings["entry_broker_order_id"] == "new-buy"
    assert resolution.exit_settings["entry_stop_loss"] == pytest.approx(203.978696)


def test_synchronize_migrates_current_order_settings_to_one_cycle_plan():
    orders = [_order("new-buy", "buy", 23, "2026-07-15T19:20:40+00:00")]
    tracked = [_tracked("new-buy", status="new")]

    updated, resolutions, changed = synchronize_position_plans(
        [_position(23, 210.758696)],
        orders,
        tracked,
    )

    assert changed
    assert sum(row.get("source") == "position_plan" for row in updated) == 1
    assert resolutions["NVDA"].plan_record_id == "position-plan-new-buy"


def test_synchronize_is_idempotent():
    orders = [_order("new-buy", "buy", 23, "2026-07-15T19:20:40+00:00")]
    first, _, _ = synchronize_position_plans(
        [_position(23, 210.758696)],
        orders,
        [_tracked("new-buy", status="new")],
    )

    second, _, changed = synchronize_position_plans(
        [_position(23, 210.758696)],
        orders,
        first,
    )

    assert not changed
    assert second == first


def test_synchronize_marks_prior_cycle_plan_closed_after_full_exit_and_reentry():
    old_orders = [_order("old-buy", "buy", 24, "2026-07-09T16:33:04+00:00", average_fill=202.48)]
    old_position = _position(24, 202.48)
    old_plan = upsert_position_plan(
        old_position,
        old_orders,
        [_tracked("old-buy")],
        {"entry_stop_distance": 6.0, "auto_exit_enabled": True},
    )
    all_orders = [
        *old_orders,
        _order("old-sell", "sell", 24, "2026-07-10T19:00:59+00:00", average_fill=210.15),
        _order("new-buy", "buy", 23, "2026-07-15T19:20:40+00:00", average_fill=210.758696),
    ]

    updated, resolutions, changed = synchronize_position_plans(
        [_position(23, 210.758696)],
        all_orders,
        old_plan,
    )

    prior = next(row for row in updated if row.get("broker_order_id") == "position-plan-old-buy")
    assert changed
    assert prior["status"] == "closed_position_cycle"
    assert prior["position_closing_order_id"] == "old-sell"
    assert prior["position_cycle_closed_at"] == "2026-07-10T19:00:59+00:00"
    assert not resolutions["NVDA"].managed


def test_nvda_reentry_cannot_inherit_prior_cycle_high_or_break_even(monkeypatch):
    orders = [
        _order("old-buy", "buy", 24, "2026-07-09T16:33:04+00:00", average_fill=202.48),
        _order("old-sell", "sell", 24, "2026-07-10T19:00:59+00:00", average_fill=210.15),
        _order("new-buy", "buy", 23, "2026-07-15T19:20:40+00:00", average_fill=210.758696),
    ]
    old = _tracked("old-buy", high=213.775)
    new = _tracked("new-buy", status="new")
    new["exit_settings"].update({
        "exit_mode": "atr_only",
        "profit_protection_enabled": True,
        "breakeven_after_r": 1.0,
        "trail_after_r": 2.0,
        "trailing_atr_multiplier": 3.0,
    })
    new["strategy_settings"] = dict(new["exit_settings"])
    position = _position(23, 210.758696)
    resolution = resolve_position_plan(position, orders, [old, new])

    data = pd.DataFrame(
        {
            "Close": [212.5, 210.9, 211.39],
            "High": [213.775, 211.1, 211.4],
            "Low": [211.8, 210.5, 210.8],
        },
        index=pd.to_datetime([
            "2026-07-15T18:00:00+00:00",
            "2026-07-15T19:00:00+00:00",
            "2026-07-15T20:00:00+00:00",
        ]),
    )
    data.attrs["latest_price"] = 211.39
    data.attrs["latest_high"] = 211.4
    monkeypatch.setattr(
        "agentloop_trader.strategy_runtime.selected_strategy_result",
        lambda *_: {"live": {"last_p": 211.39, "last_atr": 3.39, "exit_level": 190.0}},
    )

    details = evaluate_exit_settings(resolution.exit_settings, position, lambda *_: data)

    assert resolution.settings_source_order_id == "new-buy"
    assert resolution.exit_settings["entry_reference_price"] == pytest.approx(210.758696)
    assert resolution.exit_settings["highest_high_since_entry"] == pytest.approx(210.758696)
    assert details["highest_high_since_entry"] == pytest.approx(211.4)
    assert details["highest_profit_r"] == pytest.approx((211.4 - 210.758696) / 6.78)
    assert details["profit_r"] == pytest.approx((211.39 - 210.758696) / 6.78)
    assert details["breakeven_stop_price"] is None
    assert details["original_stop_price"] == pytest.approx(203.978696)
    assert details["trigger_source"] == "fill-adjusted initial stop"


def test_first_fill_replaces_planned_high_water_with_actual_average_fill():
    orders = [_order("new-buy", "buy", 23, "2026-07-15T19:20:40+00:00", average_fill=210.75)]
    tracked = [_tracked("new-buy", status="new")]
    tracked[0]["strategy_settings"].update({
        "entry_broker_order_id": "new-buy",
        "planned_entry_price": 212.00,
        "highest_high_since_entry": 212.00,
        "last_exit_trigger_price": 205.22,
    })
    tracked[0]["exit_settings"] = dict(tracked[0]["strategy_settings"])

    resolution = resolve_position_plan(_position(23, 210.75), orders, tracked)

    assert resolution.exit_settings["entry_broker_order_id"] == "new-buy"
    assert resolution.exit_settings["entry_reference_price"] == 210.75
    assert resolution.exit_settings["highest_high_since_entry"] == 210.75
    assert "last_exit_trigger_price" not in resolution.exit_settings


def test_manual_position_plan_uses_its_own_average_fill_and_atr():
    orders = [_order("manual-buy", "buy", 36, "2026-07-15T19:20:40+00:00", symbol="SPCX", average_fill=138.50)]
    cycle = current_position_cycle(_position(36, 138.50, symbol="SPCX"), orders)

    settings = initialize_exit_settings_for_position(
        cycle,
        {
            "symbol": "UNRELATED",
            "entry_atr": 20.0,
            "entry_stop_distance": 30.0,
            "highest_high_since_entry": 999.0,
            "interval": "4h",
            "auto_exit_enabled": True,
        },
        current_atr=5.63,
        atr_multiplier=1.5,
    )

    assert settings["symbol"] == "SPCX"
    assert settings["entry_reference_price"] == 138.50
    assert settings["entry_atr"] == 5.63
    assert settings["entry_stop_distance"] == pytest.approx(8.445)
    assert settings["entry_stop_loss"] == pytest.approx(130.055)
    assert settings["highest_high_since_entry"] == 138.50
    assert settings["entry_broker_order_id"] == "manual-buy"
