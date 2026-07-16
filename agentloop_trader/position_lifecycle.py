from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# Alpaca deducts a crypto BUY fee from the crypto received. Order history
# therefore reports the gross fill while the position reports the net quantity.
_MAX_ALPACA_CRYPTO_BUY_FEE_RATE = 0.0025
_CRYPTO_QUANTITY_TOLERANCE = 1e-8


_DYNAMIC_EXIT_FIELDS = {
    "highest_high_since_entry",
    "highest_rsi_since_entry",
    "last_exit_checked_at",
    "last_exit_snapshot",
    "last_exit_trigger_price",
    "last_exit_trigger_source",
}

_POSITION_OWNED_SETTING_FIELDS = {
    *_DYNAMIC_EXIT_FIELDS,
    "position_cycle_id",
    "position_basis_order_id",
    "position_buy_order_ids",
    "position_cycle_started_at",
    "position_basis_filled_at",
    "position_basis_filled_quantity",
    "position_quantity",
    "position_average_entry",
    "position_risk_budget",
    "position_risk_at_initial_stop",
    "entry_broker_order_id",
    "entry_filled_at",
    "entry_submitted_at",
    "actual_average_entry",
    "actual_basis_fill_price",
    "entry_reference_price",
    "planned_entry_price",
    "planned_order_type",
    "planned_limit_price",
    "planned_quantity",
    "entry_atr",
    "entry_atr_pct",
    "entry_source",
    "exit_plan_started_at",
    "account_size",
    "sizing_account_source",
    "sizing_account_equity",
    "sizing_available_cash",
    "risk_limits_at_entry",
    "risk_per_trade_pct",
    "entry_rsi",
    "entry_rsi_setup_low",
    "entry_rsi_sell_level",
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default


def _enum_value(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.rsplit(".", 1)[-1]


def _order_id(order: dict[str, Any]) -> str:
    return str(
        order.get("Alpaca Order ID")
        or order.get("Broker Order ID")
        or order.get("broker_order_id")
        or ""
    ).strip()


def _order_symbol(order: dict[str, Any]) -> str:
    return str(order.get("Symbol") or order.get("symbol") or "").strip().upper()


def _order_side(order: dict[str, Any]) -> str:
    return _enum_value(order.get("Side") if "Side" in order else order.get("side"))


def _order_status(order: dict[str, Any]) -> str:
    return _enum_value(order.get("Status") if "Status" in order else order.get("status"))


def _filled_quantity(order: dict[str, Any]) -> float:
    value = order.get("Filled Qty") if "Filled Qty" in order else order.get("filled_quantity")
    return max(0.0, _number(value))


def _order_time(order: dict[str, Any], *, filled: bool) -> datetime | None:
    keys = ("Filled", "filled_at") if filled else ("Submitted", "submitted_at", "created_at")
    for key in keys:
        value = order.get(key)
        if value in (None, "", "None"):
            continue
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def _iso_time(order: dict[str, Any], *, filled: bool) -> str:
    parsed = _order_time(order, filled=filled)
    return parsed.isoformat() if parsed is not None else ""


def _effective_fill_time(order: dict[str, Any]) -> str:
    parsed = _order_time(order, filled=True) or _order_time(order, filled=False)
    return parsed.isoformat() if parsed is not None else ""


@dataclass(frozen=True)
class PositionCycle:
    symbol: str
    asset_class: str
    cycle_id: str
    basis_order_id: str
    buy_order_ids: tuple[str, ...]
    started_at: str
    basis_filled_at: str
    basis_submitted_at: str
    basis_fill_price: float | None
    basis_filled_quantity: float
    broker_quantity: float
    reconstructed_quantity: float
    average_entry: float
    reliable: bool
    reason: str


@dataclass(frozen=True)
class PositionPlanResolution:
    cycle: PositionCycle | None
    entry_settings: dict[str, Any] | None
    exit_settings: dict[str, Any] | None
    plan_record_id: str
    settings_source_order_id: str
    managed: bool
    reason: str


def _filled_orders_for_symbol(symbol: str, alpaca_orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        dict(order)
        for order in alpaca_orders
        if _order_symbol(order) == symbol
        and _order_id(order)
        and _filled_quantity(order) > 0
        and _order_side(order) in {"buy", "sell"}
    ]
    return sorted(
        rows,
        key=lambda order: (
            _order_time(order, filled=True)
            or _order_time(order, filled=False)
            or datetime.min.replace(tzinfo=UTC),
            _order_id(order),
        ),
    )


def lifecycle_order_history(
    alpaca_orders: list[dict[str, Any]],
    tracked_orders: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine durable and fresh orders by exact broker ID, preferring Alpaca facts."""
    by_id: dict[str, dict[str, Any]] = {}
    for order in tracked_orders:
        source = str(order.get("source") or "").strip().lower()
        if source in {"position_plan", "position_observation", "adopted_alpaca_position"}:
            continue
        order_id = _order_id(order)
        if order_id:
            by_id[order_id] = dict(order)
    for order in alpaca_orders:
        order_id = _order_id(order)
        if not order_id:
            continue
        by_id[order_id] = {**by_id.get(order_id, {}), **dict(order)}
    return list(by_id.values())


def current_position_cycle(
    position: dict[str, Any],
    alpaca_orders: list[dict[str, Any]],
) -> PositionCycle | None:
    symbol = str(position.get("Symbol") or "").strip().upper()
    broker_quantity = abs(_number(position.get("Quantity")))
    if not symbol or broker_quantity <= 0:
        return None

    asset_class = str(position.get("Asset Type") or "equity").strip().lower()
    is_crypto = asset_class == "crypto" or "/" in symbol
    net_quantity = 0.0
    cycle_gross_buy_quantity = 0.0
    cycle_buys: list[dict[str, Any]] = []
    for order in _filled_orders_for_symbol(symbol, alpaca_orders):
        quantity = _filled_quantity(order)
        if _order_side(order) == "buy":
            if net_quantity <= 1e-8:
                cycle_buys = []
                net_quantity = 0.0
                cycle_gross_buy_quantity = 0.0
            net_quantity += quantity
            cycle_gross_buy_quantity += quantity
            cycle_buys.append(order)
        elif net_quantity > 0:
            net_quantity = max(0.0, net_quantity - quantity)
            crypto_fee_residual = (
                cycle_gross_buy_quantity * _MAX_ALPACA_CRYPTO_BUY_FEE_RATE
                if is_crypto
                else 0.0
            )
            if net_quantity <= max(1e-8, crypto_fee_residual + _CRYPTO_QUANTITY_TOLERANCE):
                net_quantity = 0.0
                cycle_gross_buy_quantity = 0.0
                cycle_buys = []

    if not cycle_buys:
        return PositionCycle(
            symbol=symbol,
            asset_class=asset_class,
            cycle_id="",
            basis_order_id="",
            buy_order_ids=(),
            started_at="",
            basis_filled_at="",
            basis_submitted_at="",
            basis_fill_price=None,
            basis_filled_quantity=0.0,
            broker_quantity=broker_quantity,
            reconstructed_quantity=net_quantity,
            average_entry=_number(position.get("Average Entry")),
            reliable=False,
            reason="The open position could not be matched to a current filled BUY cycle in Alpaca order history.",
        )

    first_buy = cycle_buys[0]
    basis_buy = cycle_buys[-1]
    buy_ids = tuple(_order_id(order) for order in cycle_buys)
    tolerance = (
        _CRYPTO_QUANTITY_TOLERANCE
        if is_crypto
        else max(1e-6, broker_quantity * 1e-6)
    )
    quantity_difference = net_quantity - broker_quantity
    maximum_crypto_buy_fees = (
        cycle_gross_buy_quantity * _MAX_ALPACA_CRYPTO_BUY_FEE_RATE
        if is_crypto
        else 0.0
    )
    exact_quantity_match = abs(quantity_difference) <= tolerance
    crypto_fee_adjusted_match = (
        is_crypto
        and -tolerance <= quantity_difference <= maximum_crypto_buy_fees + tolerance
    )
    quantity_matches = exact_quantity_match or crypto_fee_adjusted_match
    return PositionCycle(
        symbol=symbol,
        asset_class=asset_class,
        cycle_id=_order_id(first_buy),
        basis_order_id=_order_id(basis_buy),
        buy_order_ids=buy_ids,
        started_at=_effective_fill_time(first_buy),
        basis_filled_at=_effective_fill_time(basis_buy),
        basis_submitted_at=_iso_time(basis_buy, filled=False),
        basis_fill_price=(
            _number(basis_buy.get("Avg Fill") if "Avg Fill" in basis_buy else basis_buy.get("average_fill_price"))
            or None
        ),
        basis_filled_quantity=_filled_quantity(basis_buy),
        broker_quantity=broker_quantity,
        reconstructed_quantity=net_quantity,
        average_entry=_number(position.get("Average Entry")),
        reliable=quantity_matches,
        reason=(
            "Current Alpaca crypto position matched after accounting for BUY fees deducted from the crypto received."
            if crypto_fee_adjusted_match and not exact_quantity_match
            else "Current Alpaca position matched to its filled BUY and SELL history."
            if quantity_matches
            else (
                f"Alpaca position quantity is {broker_quantity:g}, but the available order history reconstructs "
                f"{net_quantity:g}. Automatic exit is paused until the lifecycle can be reconciled."
            )
        ),
    )


def _plan_record_id(cycle_id: str) -> str:
    return f"position-plan-{cycle_id}"


def _closed_cycles(alpaca_orders: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    closures: dict[str, dict[str, str]] = {}
    symbols = {_order_symbol(order) for order in alpaca_orders if _order_symbol(order)}
    for symbol in symbols:
        net_quantity = 0.0
        cycle_id = ""
        for order in _filled_orders_for_symbol(symbol, alpaca_orders):
            quantity = _filled_quantity(order)
            if _order_side(order) == "buy":
                if net_quantity <= 1e-8:
                    cycle_id = _order_id(order)
                    net_quantity = 0.0
                net_quantity += quantity
                continue
            if net_quantity <= 0:
                continue
            net_quantity = max(0.0, net_quantity - quantity)
            if net_quantity <= 1e-8 and cycle_id:
                closures[cycle_id] = {
                    "position_cycle_closed_at": _effective_fill_time(order),
                    "position_closing_order_id": _order_id(order),
                }
                cycle_id = ""
    return closures


def _observation_record(cycle: PositionCycle) -> dict[str, Any]:
    return {
        "broker_order_id": f"position-observation-{cycle.cycle_id}",
        "symbol": cycle.symbol,
        "asset_class": cycle.asset_class,
        "side": "buy",
        "status": "observed_position",
        "source": "position_observation",
        "position_cycle_id": cycle.cycle_id,
        "position_basis_order_id": cycle.basis_order_id,
        "position_buy_order_ids": list(cycle.buy_order_ids),
        "position_quantity": cycle.broker_quantity,
        "position_average_entry": cycle.average_entry,
        "position_basis_filled_quantity": cycle.basis_filled_quantity,
    }


def _bind_settings_to_cycle(settings: dict[str, Any], cycle: PositionCycle) -> dict[str, Any]:
    updated = dict(settings)
    prior_cycle = str(updated.get("position_cycle_id") or "").strip()
    prior_basis = str(updated.get("position_basis_order_id") or "").strip()
    prior_basis_quantity = _number(updated.get("position_basis_filled_quantity"), -1.0)
    same_cycle = prior_cycle == cycle.cycle_id and bool(prior_cycle)
    basis_order_changed = prior_basis != cycle.basis_order_id
    basis_fill_increased = (
        prior_basis == cycle.basis_order_id
        and prior_basis_quantity >= 0
        and cycle.basis_filled_quantity > prior_basis_quantity + 1e-8
    )
    basis_changed = basis_order_changed or basis_fill_increased
    starts_new_cycle = basis_changed and not same_cycle
    completes_same_order_fill = basis_fill_increased and same_cycle
    if starts_new_cycle or completes_same_order_fill:
        for key in _DYNAMIC_EXIT_FIELDS:
            updated.pop(key, None)

    updated.update({
        "symbol": cycle.symbol,
        "asset_class": cycle.asset_class,
        "position_cycle_id": cycle.cycle_id,
        "position_basis_order_id": cycle.basis_order_id,
        "position_buy_order_ids": list(cycle.buy_order_ids),
        "position_cycle_started_at": cycle.started_at,
        "position_basis_filled_at": cycle.basis_filled_at,
        "position_quantity": cycle.broker_quantity,
        "position_average_entry": cycle.average_entry,
        "position_basis_filled_quantity": cycle.basis_filled_quantity,
        "entry_broker_order_id": cycle.basis_order_id,
        "entry_filled_at": cycle.basis_filled_at,
        "entry_submitted_at": cycle.basis_submitted_at,
        "actual_average_entry": cycle.average_entry,
        "entry_reference_price": cycle.average_entry,
        "actual_basis_fill_price": cycle.basis_fill_price,
    })
    distance = _number(updated.get("entry_stop_distance"), -1.0)
    if distance <= 0:
        planned_entry = _number(updated.get("planned_entry_price"), 0.0)
        planned_stop = _number(updated.get("entry_stop_loss"), 0.0)
        distance = planned_entry - planned_stop if planned_entry > planned_stop > 0 else 0.0
    if distance > 0 and basis_changed and same_cycle:
        account_equity = _number(
            updated.get("sizing_account_equity"),
            _number(updated.get("account_size"), 0.0),
        )
        strategy_risk_pct = _number(updated.get("risk_per_trade_pct"), 0.0)
        saved_limits = updated.get("risk_limits_at_entry") or {}
        max_risk_pct = _number(saved_limits.get("max_risk_per_trade_pct"), 0.0)
        risk_percentages = [value for value in (strategy_risk_pct, max_risk_pct) if value > 0]
        if account_equity > 0 and risk_percentages and cycle.broker_quantity > 0:
            risk_budget = account_equity * min(risk_percentages) / 100
            distance = min(distance, risk_budget / cycle.broker_quantity)
            updated["position_risk_budget"] = risk_budget
    if distance > 0 and cycle.average_entry > 0:
        updated["entry_stop_distance"] = distance
        updated["entry_stop_loss"] = cycle.average_entry - distance
        updated["position_risk_at_initial_stop"] = distance * cycle.broker_quantity
    if (starts_new_cycle or completes_same_order_fill) and cycle.average_entry > 0:
        updated["highest_high_since_entry"] = cycle.average_entry
    return updated


def replace_exit_rules(
    existing: dict[str, Any],
    replacement: dict[str, Any],
) -> dict[str, Any]:
    """Replace editable exit rules while retaining fill-cycle and entry-owned facts."""
    merged = dict(replacement)
    for key in _POSITION_OWNED_SETTING_FIELDS:
        if key in existing:
            merged[key] = existing[key]
    return merged


def initialize_exit_settings_for_position(
    cycle: PositionCycle,
    template: dict[str, Any],
    *,
    current_atr: float,
    atr_multiplier: float,
) -> dict[str, Any]:
    """Create the first plan for an existing position from that position's own market data."""
    if not cycle.reliable or not cycle.cycle_id:
        raise ValueError(f"Cannot initialize an exit plan: {cycle.reason}")
    if current_atr <= 0 or atr_multiplier <= 0:
        raise ValueError("A positive ATR and ATR multiplier are required to initialize an exit plan.")
    settings = dict(template)
    for key in _DYNAMIC_EXIT_FIELDS:
        settings.pop(key, None)
    distance = current_atr * atr_multiplier
    settings.update({
        "symbol": cycle.symbol,
        "asset_class": cycle.asset_class,
        "entry_source": "existing Alpaca position",
        "exit_plan_started_at": datetime.now(UTC).isoformat(),
        "entry_atr": current_atr,
        "entry_atr_pct": current_atr / cycle.average_entry * 100 if cycle.average_entry > 0 else None,
        "entry_stop_atr_multiplier": atr_multiplier,
        "atr_stop_multiplier": atr_multiplier,
        "entry_stop_distance": distance,
        "entry_reference_price": cycle.average_entry,
        "planned_entry_price": cycle.average_entry,
        "entry_stop_loss": cycle.average_entry - distance,
    })
    settings = _bind_settings_to_cycle(settings, cycle)
    settings.update({
        "highest_high_since_entry": cycle.average_entry,
        "last_exit_trigger_price": cycle.average_entry - distance,
        "last_exit_trigger_source": "fill-adjusted initial stop",
    })
    return settings


def resolve_position_plan(
    position: dict[str, Any],
    alpaca_orders: list[dict[str, Any]],
    tracked_orders: list[dict[str, Any]],
) -> PositionPlanResolution:
    complete_orders = lifecycle_order_history(alpaca_orders, tracked_orders)
    cycle = current_position_cycle(position, complete_orders)
    if cycle is None:
        return PositionPlanResolution(None, None, None, "", "", False, "No open position was found.")
    if not cycle.reliable:
        return PositionPlanResolution(cycle, None, None, "", "", False, cycle.reason)

    expected_plan_id = _plan_record_id(cycle.cycle_id)
    plan_records = [
        record for record in tracked_orders
        if str(record.get("source") or "").strip().lower() == "position_plan"
        and str(record.get("position_cycle_id") or "").strip() == cycle.cycle_id
    ]
    source_record: dict[str, Any] | None = plan_records[-1] if plan_records else None
    managed = source_record is not None
    if source_record is None:
        tracked_by_id = {
            str(record.get("broker_order_id") or "").strip(): record
            for record in tracked_orders
            if record.get("broker_order_id")
        }
        for order_id in reversed(cycle.buy_order_ids):
            candidate = tracked_by_id.get(order_id)
            if candidate and (candidate.get("exit_settings") or candidate.get("strategy_settings")):
                source_record = candidate
                break

    if source_record is None:
        return PositionPlanResolution(
            cycle, None, None, expected_plan_id, "", False,
            "This Alpaca position has no saved entry or exit plan. Automatic exit is off.",
        )

    raw_entry = source_record.get("strategy_settings") or source_record.get("exit_settings")
    raw_exit = source_record.get("exit_settings") or source_record.get("strategy_settings")
    entry_settings = _bind_settings_to_cycle(dict(raw_entry), cycle) if raw_entry else None
    exit_settings = _bind_settings_to_cycle(dict(raw_exit), cycle) if raw_exit else None
    source_order_id = str(source_record.get("broker_order_id") or "").strip()
    return PositionPlanResolution(
        cycle=cycle,
        entry_settings=entry_settings,
        exit_settings=exit_settings,
        plan_record_id=expected_plan_id,
        settings_source_order_id=source_order_id,
        managed=bool(exit_settings),
        reason="Saved position plan matched to the current Alpaca fill cycle." if exit_settings else "No saved exit plan.",
    )


def upsert_position_plan(
    position: dict[str, Any],
    alpaca_orders: list[dict[str, Any]],
    tracked_orders: list[dict[str, Any]],
    exit_settings: dict[str, Any],
    entry_settings: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    complete_orders = lifecycle_order_history(alpaca_orders, tracked_orders)
    cycle = current_position_cycle(position, complete_orders)
    if cycle is None or not cycle.reliable or not cycle.cycle_id:
        reason = cycle.reason if cycle is not None else "No open position was found."
        raise ValueError(f"Cannot save this position plan: {reason}")

    existing = resolve_position_plan(position, alpaca_orders, tracked_orders)
    base_entry = entry_settings or existing.entry_settings or {
        "symbol": cycle.symbol,
        "asset_class": cycle.asset_class,
        "entry_source": "manual Alpaca position",
    }
    bound_entry = _bind_settings_to_cycle(base_entry, cycle)
    bound_exit = _bind_settings_to_cycle(exit_settings, cycle)
    record = {
        "broker_order_id": _plan_record_id(cycle.cycle_id),
        "symbol": cycle.symbol,
        "asset_class": cycle.asset_class,
        "side": "buy",
        "status": "managed_exit_settings",
        "source": "position_plan",
        "position_cycle_id": cycle.cycle_id,
        "position_basis_order_id": cycle.basis_order_id,
        "position_buy_order_ids": list(cycle.buy_order_ids),
        "position_basis_filled_quantity": cycle.basis_filled_quantity,
        "strategy_settings": bound_entry,
        "exit_settings": bound_exit,
    }
    updated = [
        dict(row) for row in tracked_orders
        if str(row.get("broker_order_id") or "").strip() != record["broker_order_id"]
    ]
    updated.append(record)
    return updated


def synchronize_position_plans(
    positions: list[dict[str, Any]],
    alpaca_orders: list[dict[str, Any]],
    tracked_orders: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, PositionPlanResolution], bool]:
    updated = [dict(row) for row in tracked_orders]
    resolutions: dict[str, PositionPlanResolution] = {}
    changed = False
    for position in positions:
        symbol = str(position.get("Symbol") or "").strip().upper()
        resolution = resolve_position_plan(position, alpaca_orders, updated)
        if resolution.cycle and resolution.cycle.reliable:
            observation = _observation_record(resolution.cycle)
            existing_observation = next(
                (
                    row for row in updated
                    if str(row.get("broker_order_id") or "").strip() == observation["broker_order_id"]
                ),
                None,
            )
            if existing_observation != observation:
                updated = [
                    row for row in updated
                    if str(row.get("broker_order_id") or "").strip() != observation["broker_order_id"]
                ]
                updated.append(observation)
                changed = True
        if resolution.exit_settings and resolution.cycle:
            expected_id = resolution.plan_record_id
            existing_plan = next(
                (row for row in updated if str(row.get("broker_order_id") or "").strip() == expected_id),
                None,
            )
            expected = upsert_position_plan(
                position,
                alpaca_orders,
                updated,
                resolution.exit_settings,
                resolution.entry_settings,
            )
            expected_plan = next(
                row for row in expected
                if str(row.get("broker_order_id") or "").strip() == expected_id
            )
            if existing_plan != expected_plan:
                updated = expected
                changed = True
            resolution = resolve_position_plan(position, alpaca_orders, updated)
        resolutions[symbol] = resolution
    closures = _closed_cycles(lifecycle_order_history(alpaca_orders, updated))
    for index, row in enumerate(updated):
        source = str(row.get("source") or "").strip().lower()
        cycle_id = str(row.get("position_cycle_id") or "").strip()
        closure = closures.get(cycle_id)
        if source not in {"position_plan", "position_observation"} or not closure:
            continue
        closed_status = "closed_position_cycle" if source == "position_plan" else "closed_position_observation"
        expected = {**row, **closure, "status": closed_status}
        if expected != row:
            updated[index] = expected
            changed = True
    return updated, resolutions, changed
